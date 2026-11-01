/*
 * Compare the Qwen3-VL vision tower with the tensors dumped by
 * debug/qimage_reference.py --image, stage by stage, on the path of the
 * build (device if available, otherwise CPU). Includes iris_qwen3_vision.c to
 * reach its stages, so link everything except its object.
 *
 * Build (after make blas, in bash, from the repository root):
 *   cc -O3 -ffast-math -DUSE_BLAS -DACCELERATE_NEW_LAPACK -I. -o /tmp/dump_vision \
 *      debug/dump_qimage_vision.c $(ls *.o | grep -v 'iris_qwen3_vision\|^main\|iris_cli\|linenoise\|embcache') \
 *      -lm -framework Accelerate
 * MPS (after make mps, in bash):
 *   cc -O3 -ffast-math -DUSE_BLAS -DUSE_METAL -DIRIS_DEV -DACCELERATE_NEW_LAPACK -I. \
 *      -o /tmp/dump_vision_mps debug/dump_qimage_vision.c \
 *      $(ls *.mps.o | grep -v 'iris_qwen3_vision\|^main\|iris_cli\|linenoise\|embcache') \
 *      iris_metal.o iris_dev_metal.o -lm -framework Accelerate -framework Metal \
 *      -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
 *      -framework Foundation
 * CUDA (after make cuda):
 *   gcc -O3 -march=native -ffast-math -DIRIS_DEV -I. -o /tmp/dump_vision debug/dump_qimage_vision.c \
 *      $(ls *.o | grep -v 'iris_qwen3_vision\|^main\|iris_cli\|linenoise\|embcache') \
 *      -lm -L$CUDA_HOME/lib64 -lcublas -lcudart -lstdc++
 *
 * Usage: dump_qimage_vision TEXT_ENCODER_DIR REF_DIR
 *
 * Each image runs alone; the dumps hold all images, split by grid_thw.txt.
 * Prints the relative L2 and the max absolute difference of every stage, and
 * exits non-zero if a stage is over its tolerance in stage_tols.
 */

#include "../iris_qwen3_vision.c"
#ifdef USE_METAL
#include "iris_metal.h"
#endif

static float *read_floats(const char *dir, const char *name, size_t expect) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    size_t n = (size_t)ftell(f) / sizeof(float);
    rewind(f);
    float *data = malloc(n * sizeof(float));
    if (!data || fread(data, sizeof(float), n, f) != n || (expect && n != expect)) {
        fprintf(stderr, "cannot read %s (%zu floats, expected %zu)\n", path, n, expect);
        exit(1);
    }
    fclose(f);
    return data;
}

static int g_failed = 0;
static int g_on_device = 0;

/* Relative L2 per stage, about 1.25x the worst of a correct build over four
 * cases (r1, r2, 512x320, 1024x1024). The error grows along the tower and with
 * the image size, so a larger image can exceed the last stages. These are
 * rounding floors, not a target: on the CPU upstream's own f32 tower against
 * f64 is 2.5e-5 on vis_block26 and 2.4e-5 on image_embeds (r1), 7.5e-5 on
 * image_embeds (1024x1024); on the device upstream's bf16 tower is 5.1e-2 and
 * 1.6e-1 on image_embeds. Tighten them if the numbers improve, and treat a
 * stage going over as a bug. An error below the bf16 floor passes on the
 * device (tanh GELU in the mergers: 1.6e-3 on deep0; exact GELU in the blocks,
 * 3.3% on image_embeds against 2.4%). The CPU run catches the first,
 * check_gelu() the second. */
static const struct {
    const char *name;
    double cpu, device;
} stage_tols[] = {
    {"vis_patch.bin",    1e-6,   2.5e-3},
    {"vis_in.bin",       1e-6,   2.5e-3},
    {"vis_block0.bin",   4e-6,   4e-3},
    {"vis_block8.bin",   1.2e-5, 8.5e-3},
    {"deep0.bin",        1.1e-5, 9.5e-3},
    {"vis_block16.bin",  2e-5,   1.5e-2},
    {"deep1.bin",        2.3e-5, 1.8e-2},
    {"vis_block24.bin",  2.5e-5, 1.8e-2},
    {"deep2.bin",        3.7e-5, 2.7e-2},
    {"vis_block26.bin",  7.9e-5, 5.7e-2},
    {"image_embeds.bin", 1e-4,   7.6e-2},
};

static double stage_tol(const char *name) {
    for (size_t i = 0; i < sizeof(stage_tols) / sizeof(stage_tols[0]); i++)
        if (!strcmp(stage_tols[i].name, name))
            return g_on_device ? stage_tols[i].device : stage_tols[i].cpu;
    fprintf(stderr, "no tolerance for %s\n", name);
    exit(1);
}

/* c[rows, cols] against rows of the reference at row offset, stride ld. */
static void compare(const char *stage, const float *c, const float *ref, size_t rows,
                    size_t cols, size_t ld, double tol) {
    double diff = 0, norm = 0, max_abs = 0;
    size_t bad = 0;
    for (size_t r = 0; r < rows; r++)
        for (size_t i = 0; i < cols; i++) {
            uint32_t bits;     /* isnan() folds away under -ffast-math */
            memcpy(&bits, &c[r * cols + i], sizeof(bits));
            if ((bits & 0x7F800000u) == 0x7F800000u) {
                bad++;
                continue;
            }
            double a = c[r * cols + i], b = ref[r * ld + i], d = a - b;
            diff += d * d;
            norm += b * b;
            if (fabs(d) > max_abs) max_abs = fabs(d);
        }
    double rel = sqrt(diff / (norm > 0 ? norm : 1));
    printf("  %-16s rel L2 %.3e  max abs %.3e%s\n", stage, rel, max_abs,
           bad ? "  NOT FINITE" : rel > tol ? "  OVER TOLERANCE" : "");
    if (bad || rel > tol) g_failed = 1;
}

/* The reference file name, read once per stage and freed by the caller. */
static float *ref_rows(const char *dir, const char *name, size_t total_rows, size_t cols,
                       size_t row0) {
    float *all = read_floats(dir, name, total_rows * cols);
    return memmove(all, all + row0 * cols, (total_rows - row0) * cols * sizeof(float));
}

static void check(const char *dir, const char *name, const float *c, size_t total_rows,
                  size_t rows, size_t cols, size_t row0) {
    float *ref = ref_rows(dir, name, total_rows, cols, row0);
    compare(name, c, ref, rows, cols, cols, stage_tol(name));
    free(ref);
}

/* C's px [n, c * p * p] against the processor's [n, c * t * p * p], first frame. */
static void check_pixels(const qwen3_vision_t *v, const char *dir, const vis_input_t *in,
                         size_t total, size_t row0) {
    size_t area = (size_t)v->patch * v->patch, row = VIS_CHANNELS * v->temporal * area;
    float *ref = ref_rows(dir, "pixel_values.bin", total, row, row0);
    double max_abs = 0;
    for (size_t p = 0; p < (size_t)in->n; p++)
        for (size_t c = 0; c < VIS_CHANNELS; c++)
            for (size_t i = 0; i < area; i++) {
                float a = in->px[(p * VIS_CHANNELS + c) * area + i];
                float b = ref[(p * VIS_CHANNELS * v->temporal + c * v->temporal) * area + i];
                if (fabs(a - b) > max_abs) max_abs = fabs(a - b);
            }
    free(ref);
    printf("  %-12s max abs %.3e\n", "pixels", max_abs);
    if (max_abs > 1e-6) g_failed = 1;
}

static void check_rope(const qwen3_vision_t *v, const char *dir, const vis_input_t *in,
                       size_t total, size_t row0, double tol) {
    size_t half = v->head_dim / 2;
    float *c = ref_rows(dir, "vis_rope_cos.bin", total, 2 * half, row0);
    float *s = ref_rows(dir, "vis_rope_sin.bin", total, 2 * half, row0);
    compare("rope cos", in->cos_t, c, in->n, half, 2 * half, tol);
    compare("rope sin", in->sin_t, s, in->n, half, 2 * half, tol);
    free(c);
    free(s);
}

static int dumped_block(int l) {
    return l == 0 || l == 8 || l == 16 || l == 24 || l == 26;
}

static void check_block(const qwen3_vision_t *v, const char *dir, const float *x, int l,
                        size_t total, size_t n, size_t row0) {
    char name[32];
    snprintf(name, sizeof(name), "vis_block%d.bin", l);
    check(dir, name, x, total, n, v->dim, row0);
}

static void check_output(const qwen3_vision_t *v, const char *dir, const float *out, int k,
                         size_t total_s, size_t s, size_t s0) {
    char name[32];
    if (k < 0) snprintf(name, sizeof(name), "image_embeds.bin");
    else snprintf(name, sizeof(name), "deep%d.bin", k);
    check(dir, name, out, total_s, s, v->out_dim, s0);
}

/* Stage by stage on the CPU, in f32. */
static void run_cpu(const qwen3_vision_t *v, const char *dir, const vis_input_t *in,
                    size_t total, size_t row0) {
    size_t n = in->n, dim = v->dim, m2 = (size_t)v->merge * v->merge;
    size_t s = n / m2, total_s = total / m2, s0 = row0 / m2;
    float *out = malloc(s * v->out_dim * sizeof(float));
    vis_work_t w;
    if (!work_alloc(v, &w, in->n)) exit(1);
    patch_embed(v, w.x, in);
    check(dir, "vis_patch.bin", w.x, total, n, dim, row0);
    add(w.x, in->pos, n * dim);
    check(dir, "vis_in.bin", w.x, total, n, dim, row0);
    for (int l = 0; l < v->depth; l++) {
        int k;
        const vis_merger_t *m;
        if (!block_forward(v, &v->blocks[l], &w, in)) exit(1);
        if (dumped_block(l)) check_block(v, dir, w.x, l, total, n, row0);
        if ((m = deep_merger(v, l, &k))) {
            merger_forward(v, m, &w, out, in->n);
            check_output(v, dir, out, k, total_s, s, s0);
        }
    }
    merger_forward(v, &v->merger, &w, out, in->n);
    check_output(v, dir, out, -1, total_s, s, s0);
    work_free(&w);
    free(out);
}

#ifdef IRIS_DEV
/* The blocks' activation on the device against the CPU's, which the f32 stages
 * pin. No rounding floor here: correct is 4.8e-7, exact GELU 4.7e-4. */
static void check_gelu(void) {
    enum { N = 1024 };
    float host[N], dev[N];
    double max_abs = 0;
    for (int i = 0; i < N; i++) host[i] = -8.0f + 16.0f * i / (N - 1);
    float *d = iris_dev_alloc(sizeof(host));
    iris_dev_upload(d, host, sizeof(host));
    iris_dev_gelu_tanh(d, N);
    iris_dev_download(dev, d, sizeof(dev));
    iris_dev_free(d);
    iris_gelu_tanh(host, N);
    for (int i = 0; i < N; i++)
        if (fabs((double)dev[i] - host[i]) > max_abs) max_abs = fabs((double)dev[i] - host[i]);
    printf("gelu_tanh max abs %.3e%s\n", max_abs, max_abs > 1e-5 ? "  OVER TOLERANCE" : "");
    if (max_abs > 1e-5) g_failed = 1;
}

/* The same on the device. */
static void run_dev(qwen3_vision_t *v, const char *dir, const vis_input_t *in,
                    size_t total, size_t row0) {
    size_t n = in->n, dim = v->dim, m2 = (size_t)v->merge * v->merge;
    size_t s = n / m2, total_s = total / m2, s0 = row0 / m2;
    float *out = malloc(s * v->out_dim * sizeof(float));
    float *x = malloc(n * dim * sizeof(float));
    vis_dev_t d;
    if (!dev_load(v) || !dev_work_alloc(v, &d, in->n)) exit(1);
    dev_patch_embed(v, &d, in);
    iris_dev_download(x, d.x, n * dim * sizeof(float));
    check(dir, "vis_patch.bin", x, total, n, dim, row0);
    dev_add_position(v, &d, in);
    iris_dev_download(x, d.x, n * dim * sizeof(float));
    check(dir, "vis_in.bin", x, total, n, dim, row0);
    for (int l = 0; l < v->depth; l++) {
        int k;
        const vis_merger_t *m;
        dev_block(v, &v->blocks[l], &d, in->n);
        if (dumped_block(l)) {
            iris_dev_download(x, d.x, n * dim * sizeof(float));
            check_block(v, dir, x, l, total, n, row0);
        }
        if ((m = deep_merger(v, l, &k))) {
            if (!dev_merger(v, m, &d, out, in->n)) exit(1);
            check_output(v, dir, out, k, total_s, s, s0);
        }
    }
    if (!dev_merger(v, &v->merger, &d, out, in->n)) exit(1);
    check_output(v, dir, out, -1, total_s, s, s0);
    iris_dev_free(d.x);
    free(x);
    free(out);
}
#endif

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: %s TEXT_ENCODER_DIR REF_DIR\n", argv[0]);
        return 1;
    }
    const char *dir = argv[2];
#ifdef USE_METAL
    iris_metal_init();
#endif
#ifdef IRIS_DEV
    iris_dev_init();
    printf("device: %s\n", iris_dev_available() ? iris_dev_name() : "none");
    if (iris_dev_available()) check_gelu();
#endif
    qwen3_vision_t *v = qwen3_vision_load(argv[1]);
    if (!v) return 1;

    char path[1024];
    snprintf(path, sizeof(path), "%s/grid_thw.txt", dir);
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        return 1;
    }
    int t, gh[16], gw[16], images = 0;
    size_t total = 0;
    while (images < 16 && fscanf(f, "%d %d %d", &t, &gh[images], &gw[images]) == 3) {
        total += (size_t)gh[images] * gw[images];
        images++;
    }
    fclose(f);

    size_t row0 = 0;
    for (int i = 0; i < images; i++) {
        int h = gh[i] * v->patch, w = gw[i] * v->patch;
        size_t n = (size_t)gh[i] * gw[i], m2 = (size_t)v->merge * v->merge;
        printf("image %d: %dx%d, %zu patches\n", i, w, h, n);
        char name[32];
        snprintf(name, sizeof(name), "ref%d_vision_rgb.bin", i);
        float *rgb = read_floats(dir, name, (size_t)3 * h * w);

        vis_input_t in;
        if (!input_prepare(v, &in, rgb, h, w)) return 1;
        check_pixels(v, dir, &in, total, row0);
        check_rope(v, dir, &in, total, row0, 1e-6);
#ifdef IRIS_DEV
        g_on_device = iris_dev_available();
        if (g_on_device) run_dev(v, dir, &in, total, row0);
#endif
        if (!g_on_device) run_cpu(v, dir, &in, total, row0);
        input_free(&in);

        /* The public entry point, whichever path it takes. */
        float *embeds, *deep[QWEN3_VISION_DEEPSTACK];
        int tokens = qwen3_vision_encode(v, rgb, h, w, &embeds, deep);
        if (tokens != (int)(n / m2)) {
            fprintf(stderr, "qwen3_vision_encode returned %d tokens\n", tokens);
            return 1;
        }
        printf("  public API:\n");
        check_output(v, dir, embeds, -1, total / m2, n / m2, row0 / m2);
        for (int k = 0; k < QWEN3_VISION_DEEPSTACK; k++) {
            check_output(v, dir, deep[k], k, total / m2, n / m2, row0 / m2);
            free(deep[k]);
        }
        free(embeds);
        free(rgb);
        row0 += n;
    }
    qwen3_vision_free(v);
    printf(g_failed ? "FAILED\n" : "OK\n");
    return g_failed;
}
