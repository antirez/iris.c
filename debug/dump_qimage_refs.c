/*
 * Compare the Qwen-Image transformer with reference images against the
 * tensors dumped by debug/qimage_reference.py --image: RoPE tables, the
 * output of every block, per-step velocities and a free run. With no
 * segments.txt in REF_DIR it runs plain text-to-image.
 *
 * This file includes iris_transformer_qimage.c to reach its layout code
 * and its block hook, so link everything except that object.
 *
 * Build, in bash, after make blas:
 *   cc -O3 -march=native -ffast-math -DUSE_BLAS -DACCELERATE_NEW_LAPACK -I. \
 *      -o /tmp/dump_qimage_refs debug/dump_qimage_refs.c \
 *      $(ls *.o | grep -v -e '^main.o' -e '^iris_transformer_qimage.o') \
 *      -lm -framework Accelerate
 * After make mps:
 *   cc -O3 -march=native -ffast-math -DUSE_BLAS -DUSE_METAL -DIRIS_DEV \
 *      -DACCELERATE_NEW_LAPACK -I. -o /tmp/dump_qimage_refs_mps debug/dump_qimage_refs.c \
 *      $(ls *.mps.o | grep -v -e '^main.mps.o' -e '^iris_transformer_qimage.mps.o') \
 *      iris_metal.o iris_dev_metal.o -lm -framework Accelerate -framework Metal \
 *      -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
 *      -framework Foundation
 * After make cuda:
 *   gcc -O3 -march=native -ffast-math -DIRIS_DEV -I. -o /tmp/dump_qimage_refs debug/dump_qimage_refs.c \
 *      $(ls *.o | grep -v -e '^main.o' -e '^iris_transformer_qimage.o') \
 *      -lm -L$CUDA_HOME/lib64 -lcublas -lcudart -lstdc++
 *
 * Usage: dump_qimage_refs MODEL_DIR REF_DIR LATENT_H LATENT_W
 *
 * Reads from REF_DIR: emb.bin, noise.bin, sigmas.txt, step<i>_x.bin,
 * step<i>_v.bin, latent.bin, rope_cos.bin, rope_sin.bin, block<l>_s<0|1>.bin
 * (skipped if absent) and, with references, segments.txt, grid_thw.txt (reference sizes),
 * image_pad_mask.txt and ref<i>_latent.bin. Prints relative L2 errors.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "iris.h"
#include "iris_qimage.h"
#ifdef USE_METAL
#include "iris_metal.h"
#endif
#ifdef IRIS_DEV
#include "iris_dev.h"
#endif

#define CH IRIS_QIMAGE_LATENT_CHANNELS
#define MAX_REFS 16

static void block_hook(iris_transformer_qimage_t *tf, const float *x, int rows);
#define QI_BLOCK_HOOK(tf, x, rows) block_hook(tf, x, rows)
#include "../iris_transformer_qimage.c"

static const char *ref_dir;
static int hook_step = -2;      /* -2 off, -1 prefill, else the step */
static int hook_block;
static double block_err[2];     /* sums of squares, error and reference */

static FILE *open_ref(const char *name, const char *mode) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", ref_dir, name);
    return fopen(path, mode);
}

static float *read_range(const char *name, size_t first, size_t count) {
    FILE *f = open_ref(name, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", name);
        exit(1);
    }
    if (!count) {
        fseek(f, 0, SEEK_END);
        count = (size_t)ftell(f) / sizeof(float);
    }
    fseek(f, (long)(first * sizeof(float)), SEEK_SET);
    float *data = malloc(count * sizeof(float));
    if (!data || fread(data, sizeof(float), count, f) != count) {
        fprintf(stderr, "cannot read %s\n", name);
        exit(1);
    }
    fclose(f);
    return data;
}

static void accumulate(double *sums, const float *a, const float *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        sums[0] += d * d;
        sums[1] += (double)b[i] * b[i];
    }
}

static double rel_l2(const float *a, const float *b, size_t n) {
    double sums[2] = {0, 0};
    accumulate(sums, a, b, n);
    return sqrt(sums[0] / sums[1]);
}

/* Prefill: the first rows of block<l>_s0. Step 0: the rows after them.
 * Step 1: the whole of block<l>_s1, which has only the target rows. */
static void block_hook(iris_transformer_qimage_t *tf, const float *x, int rows) {
    if (hook_step < -1 || hook_step > 1) return;
    char name[64];
    snprintf(name, sizeof(name), "block%d_s%d.bin", hook_block, hook_step == 1 ? 1 : 0);
    FILE *f = open_ref(name, "rb");
    if (!f) {   /* run without --blocks: velocities and the free run only */
        hook_block++;
        return;
    }
    fclose(f);
    size_t n = (size_t)rows * tf->dim;
    float *host = malloc(n * sizeof(float));
#ifdef IRIS_DEV
    if (tf->dev) iris_dev_download(host, x, n * sizeof(float));
    else
#endif
    memcpy(host, x, n * sizeof(float));
    size_t first = hook_step == 0 ? (size_t)tf->prefix_seq * tf->dim : 0;
    float *ref = read_range(name, first, n);
    double e = rel_l2(host, ref, n);
    printf("block %2d %s: %.3e\n", hook_block,
           hook_step < 0 ? "prefix" : hook_step ? "s1 target" : "s0 target", e);
    accumulate(block_err, host, ref, n);
    free(ref);
    free(host);
    hook_block++;
}

/* [N, 64] tokens <-> [64, N] planes */
static void transpose(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}

/* Slot rows of emb.bin dropped, as the text encoder will hand them over. */
static float *strip_slots(const float *emb, int rows, int td, int *txt_seq) {
    float *txt = malloc((size_t)rows * td * sizeof(float));
    FILE *f = open_ref("image_pad_mask.txt", "r");
    int t = 0, slot;
    if (!f) {
        fprintf(stderr, "cannot open image_pad_mask.txt\n");
        exit(1);
    }
    for (int r = 0; r < rows; r++) {
        if (fscanf(f, "%d", &slot) != 1) {
            fprintf(stderr, "short image_pad_mask.txt\n");
            exit(1);
        }
        if (!slot) memcpy(txt + (size_t)t++ * td, emb + (size_t)r * td, td * sizeof(float));
    }
    fclose(f);
    *txt_seq = t;
    return txt;
}

/* The references from segments.txt (start end is_text) and grid_thw.txt. */
static int read_refs(iris_qimage_ref_t *refs) {
    FILE *seg = open_ref("segments.txt", "r");
    FILE *grid = open_ref("grid_thw.txt", "r");
    int start, end, is_text, n = 0, text = 0;
    if (!seg) return 0;
    while (fscanf(seg, "%d %d %d", &start, &end, &is_text) == 3) {
        if (is_text) {
            text += end - start;
            continue;
        }
        int t, h, w;
        if (n == MAX_REFS || !grid || fscanf(grid, "%d %d %d", &t, &h, &w) != 3 ||
            h * w != end - start) {
            fprintf(stderr, "bad grid_thw.txt for segment %d..%d\n", start, end);
            exit(1);
        }
        char name[64];
        snprintf(name, sizeof(name), "ref%d_latent.bin", n);
        refs[n] = (iris_qimage_ref_t){text, h, w, read_range(name, 0, (size_t)CH * h * w)};
        n++;
    }
    fclose(seg);
    if (grid) fclose(grid);
    return n;
}

static void compare_rope(const iris_transformer_qimage_t *tf, const float *txt, int txt_seq,
                         const iris_qimage_ref_t *refs, int n, int h, int w) {
    int half = tf->head_dim / 2, L = txt_seq, tgt = h * w;
    for (int i = 0; i < n; i++) L += refs[i].h * refs[i].w;
    size_t cells = (size_t)(L + tgt) * half;
    float *c = malloc(cells * sizeof(float)), *s = malloc(cells * sizeof(float));
    int (*pos)[QI_ROPE_AXES] = malloc((size_t)L * sizeof(*pos));
    qi_run_t *runs = malloc((size_t)(2 * n + 1) * sizeof(qi_run_t));
    int nruns;
    int frame = prefix_layout(tf, runs, &nruns, pos, txt, txt_seq, refs, n);
    rope_rows(tf, c, s, (const int (*)[QI_ROPE_AXES])pos, L);
    image_rope(tf, frame, h, w, c + (size_t)L * half, s + (size_t)L * half);
    float *rc = read_range("rope_cos.bin", 0, cells), *rs = read_range("rope_sin.bin", 0, cells);
    double maxd = 0;
    for (size_t i = 0; i < cells; i++) {
        double d = fabs((double)c[i] - rc[i]) + fabs((double)s[i] - rs[i]);
        if (d > maxd) maxd = d;
    }
    printf("prefix rows %d, runs %d, target frame %d\n", L, nruns, frame);
    printf("rope: cos %.3e, sin %.3e, max abs %.3e\n", rel_l2(c, rc, cells),
           rel_l2(s, rs, cells), maxd);
    free(c);
    free(s);
    free(pos);
    free(runs);
    free(rc);
    free(rs);
}

int main(int argc, char **argv) {
    if (argc != 5) {
        fprintf(stderr, "usage: %s MODEL_DIR REF_DIR LATENT_H LATENT_W\n", argv[0]);
        return 1;
    }
    ref_dir = argv[2];
    int h = atoi(argv[3]), w = atoi(argv[4]), n = h * w;

#ifdef USE_METAL
    iris_metal_init();
#endif
#ifdef IRIS_DEV
    iris_dev_init();
    printf("device: %s\n", iris_dev_available() ? iris_dev_name() : "none");
#endif

    FILE *f = open_ref("sigmas.txt", "r");
    float sigmas[IRIS_MAX_STEPS + 1];
    int num_sigmas = 0;
    while (f && num_sigmas <= IRIS_MAX_STEPS && fscanf(f, "%f", &sigmas[num_sigmas]) == 1)
        num_sigmas++;
    if (!f || num_sigmas < 2) {
        fprintf(stderr, "cannot read sigmas.txt\n");
        return 1;
    }
    fclose(f);
    int steps = num_sigmas - 1;

    iris_transformer_qimage_t *tf = iris_transformer_load_qimage(argv[1]);
    if (!tf) {
        fprintf(stderr, "transformer load failed\n");
        return 1;
    }

    FILE *emb_f = open_ref("emb.bin", "rb");
    if (!emb_f) {
        fprintf(stderr, "cannot open emb.bin\n");
        return 1;
    }
    fseek(emb_f, 0, SEEK_END);
    int rows = (int)((size_t)ftell(emb_f) / sizeof(float) / tf->text_dim);
    fclose(emb_f);
    float *emb = read_range("emb.bin", 0, (size_t)rows * tf->text_dim);

    iris_qimage_ref_t refs[MAX_REFS];
    int num_refs = read_refs(refs), txt_seq = rows;
    float *txt = num_refs ? strip_slots(emb, rows, tf->text_dim, &txt_seq) : emb;
    printf("text rows %d of %d, references %d\n", txt_seq, rows, num_refs);
    compare_rope(tf, txt, txt_seq, refs, num_refs, h, w);

    hook_step = -1;
    if (!iris_transformer_prefill_qimage(tf, txt, txt_seq, refs, num_refs)) {
        fprintf(stderr, "prefill failed\n");
        return 1;
    }

    /* The reference's step<i>_x.bin is the latent after step i. */
    float *noise = read_range("noise.bin", 0, (size_t)CH * n);
    float *x = malloc((size_t)CH * n * sizeof(float));
    float *tokens = malloc((size_t)CH * n * sizeof(float));
    for (int i = 0; i < steps; i++) {
        char name[64];
        if (i == 0) {
            memcpy(x, noise, (size_t)CH * n * sizeof(float));
        } else {
            snprintf(name, sizeof(name), "step%d_x.bin", i - 1);
            float *ref_x = read_range(name, 0, (size_t)CH * n);
            transpose(x, ref_x, n, CH);
            free(ref_x);
        }
        hook_step = i;
        hook_block = 0;
        float *v = iris_transformer_forward_qimage(tf, x, h, w, sigmas[i]);
        if (!v) {
            fprintf(stderr, "forward failed at step %d\n", i);
            return 1;
        }
        transpose(tokens, v, CH, n);
        snprintf(name, sizeof(name), "step%d_v.bin", i);
        float *ref_v = read_range(name, 0, (size_t)CH * n);
        printf("step %d v: %.3e\n", i, rel_l2(tokens, ref_v, (size_t)CH * n));
        free(ref_v);
        free(v);
    }
    printf("all blocks: %.3e\n", sqrt(block_err[0] / block_err[1]));

    /* The same inputs must not run the blocks again, a changed latent must. */
    hook_step = -1;
    hook_block = 0;
    iris_transformer_prefill_qimage(tf, txt, txt_seq, refs, num_refs);
    printf("repeated prefill ran %d blocks\n", hook_block);
    if (num_refs) {
        float *first = (float *)refs[0].latent;
        hook_step = -2;
        first[0] += 1.0f;
        iris_transformer_prefill_qimage(tf, txt, txt_seq, refs, num_refs);
        first[0] -= 1.0f;
        hook_step = -1;
        hook_block = 0;
        iris_transformer_prefill_qimage(tf, txt, txt_seq, refs, num_refs);
        printf("prefill after a changed latent ran %d blocks\n", hook_block);
    }
    hook_step = -2;

    float *z = iris_sample_euler_qimage(tf, noise, CH, h, w, sigmas, steps, NULL);
    if (!z) {
        fprintf(stderr, "sampling failed\n");
        return 1;
    }
    float *ref_z = read_range("latent.bin", 0, (size_t)CH * n);
    printf("free-run latent: %.3e\n", rel_l2(z, ref_z, (size_t)CH * n));

    free(ref_z);
    free(z);
    free(x);
    free(tokens);
    free(noise);
    if (txt != emb) free(txt);
    free(emb);
    for (int i = 0; i < num_refs; i++) free((void *)refs[i].latent);
    iris_transformer_free_qimage(tf);
    return 0;
}
