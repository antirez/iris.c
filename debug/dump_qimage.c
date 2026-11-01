/*
 * Run the Qwen-Image transformer and VAE decoder on the tensors dumped by
 * debug/qimage_reference.py, to compare them with it one at a time.
 *
 * Build (after make blas, from the repository root):
 *   make lib
 *   cc -O2 -I. -o /tmp/dump_qimage debug/dump_qimage.c libiris.a -lm -framework Accelerate
 * MPS (after make mps, in bash):
 *   cc -O2 -I. -DUSE_METAL -DIRIS_DEV -o /tmp/dump_qimage_mps debug/dump_qimage.c \
 *      $(ls *.mps.o | grep -v main.mps.o) iris_metal.o iris_dev_metal.o -lm \
 *      -framework Accelerate -framework Metal -framework MetalPerformanceShaders \
 *      -framework MetalPerformanceShadersGraph -framework Foundation
 *
 * Usage: dump_qimage MODEL_DIR REF_DIR OUT_DIR LATENT_H LATENT_W
 *
 * Reads emb.bin, noise.bin, sigmas.txt, step<i>_x.bin and latent.bin from
 * REF_DIR. Writes to OUT_DIR, in the reference's layouts:
 *   step<i>_v.bin  velocity [N, 64] on the reference latent of that step,
 *                  so errors do not accumulate across steps
 *   latent.bin     latent [64, H, W] after sampling on C velocities only
 *   vae.png        reference latent decoded by the C VAE
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "iris.h"
#include "iris_qimage.h"
#ifdef USE_METAL
#include "iris_metal.h"
#endif
#ifdef IRIS_DEV
#include "iris_dev.h"
#endif

#define CH IRIS_QIMAGE_LATENT_CHANNELS

static float *read_floats(const char *dir, const char *name, size_t *count) {
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
    if (!data || fread(data, sizeof(float), n, f) != n) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(1);
    }
    fclose(f);
    if (count) *count = n;
    return data;
}

static void write_floats(const char *dir, const char *name, const float *data, size_t n) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(data, sizeof(float), n, f) != n) {
        fprintf(stderr, "cannot write %s\n", path);
        exit(1);
    }
    fclose(f);
}

/* [N, 64] tokens <-> [64, N] planes */
static void transpose(float *dst, const float *src, int rows, int cols) {
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            dst[c * rows + r] = src[r * cols + c];
}

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s MODEL_DIR REF_DIR OUT_DIR LATENT_H LATENT_W\n", argv[0]);
        return 1;
    }
    const char *ref = argv[2], *out = argv[3];
    int h = atoi(argv[4]), w = atoi(argv[5]), n = h * w;

#ifdef USE_METAL
    iris_metal_init();
#endif
#ifdef IRIS_DEV
    iris_dev_init();
    printf("device: %s\n", iris_dev_available() ? iris_dev_name() : "none");
#endif

    char path[1024];
    snprintf(path, sizeof(path), "%s/sigmas.txt", ref);
    FILE *f = fopen(path, "r");
    float sigmas[IRIS_MAX_STEPS + 1];
    int num_sigmas = 0;
    while (f && num_sigmas <= IRIS_MAX_STEPS && fscanf(f, "%f", &sigmas[num_sigmas]) == 1)
        num_sigmas++;
    if (!f || num_sigmas < 2) {
        fprintf(stderr, "cannot read %s\n", path);
        return 1;
    }
    fclose(f);
    int steps = num_sigmas - 1;

    iris_ctx *ctx = iris_load_dir(argv[1]);
    if (!ctx) {
        fprintf(stderr, "failed to load %s: %s\n", argv[1], iris_get_error());
        return 1;
    }

    /* The VAE first: on GPU builds the transformer can take most memory. */
    float *latent = read_floats(ref, "latent.bin", NULL);
    iris_image *img = iris_decode_latent(ctx, latent, h, w);
    snprintf(path, sizeof(path), "%s/vae.png", out);
    if (!img || iris_image_save(img, path) != 0) {
        fprintf(stderr, "VAE decode failed\n");
        return 1;
    }
    iris_image_free(img);
    free(latent);

    size_t emb_size;
    float *emb = read_floats(ref, "emb.bin", &emb_size);
    int txt_seq = (int)(emb_size / iris_text_dim(ctx));
    iris_transformer_qimage_t *tf = iris_transformer_load_qimage(argv[1]);
    if (!tf || !iris_transformer_prefill_qimage(tf, emb, txt_seq, NULL, 0)) {
        fprintf(stderr, "transformer load or prefill failed\n");
        return 1;
    }
    free(emb);

    /* The reference's step<i>_x.bin is the latent after step i. */
    float *noise = read_floats(ref, "noise.bin", NULL);
    float *x = malloc((size_t)CH * n * sizeof(float));
    float *tokens = malloc((size_t)CH * n * sizeof(float));
    for (int i = 0; i < steps; i++) {
        if (i == 0) {
            memcpy(x, noise, (size_t)CH * n * sizeof(float));
        } else {
            char name[64];
            snprintf(name, sizeof(name), "step%d_x.bin", i - 1);
            float *ref_x = read_floats(ref, name, NULL);
            transpose(x, ref_x, n, CH);
            free(ref_x);
        }
        float *v = iris_transformer_forward_qimage(tf, x, h, w, sigmas[i]);
        if (!v) {
            fprintf(stderr, "forward failed at step %d\n", i);
            return 1;
        }
        transpose(tokens, v, CH, n);
        char name[64];
        snprintf(name, sizeof(name), "step%d_v.bin", i);
        write_floats(out, name, tokens, (size_t)CH * n);
        free(v);
        printf("step %d/%d\n", i + 1, steps);
    }

    /* Same weights, same prefix, C velocities only. */
    float *z = iris_sample_euler_qimage(tf, noise, CH, h, w, sigmas, steps, NULL);
    if (!z) {
        fprintf(stderr, "sampling failed\n");
        return 1;
    }
    write_floats(out, "latent.bin", z, (size_t)CH * n);

    free(z);
    free(x);
    free(tokens);
    free(noise);
    iris_transformer_free_qimage(tf);
    iris_free(ctx);
    return 0;
}
