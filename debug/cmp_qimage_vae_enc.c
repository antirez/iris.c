/*
 * Compare the Qwen-Image VAE encoder, stage by stage, with the dumps of
 * debug/qimage_reference.py --image, feeding it ref<i>_rgba.bin so that the
 * resize differences of reference preparation stay out.
 *
 * This file includes iris_vae_qimage.c to get at its stage hooks, so link
 * every other object and compile with the matching flags (from the
 * repository root, in bash).
 *
 * After make blas:
 *   cc -O3 -march=native -ffast-math -DUSE_BLAS -DACCELERATE_NEW_LAPACK -I. \
 *      -o /tmp/cmp_vae_enc debug/cmp_qimage_vae_enc.c \
 *      $(ls *.o | grep -vE '^(main|iris_vae_qimage|iris_cli|linenoise|embcache)\.o') \
 *      -lm -framework Accelerate
 * After make mps:
 *   cc -O3 -march=native -ffast-math -DUSE_BLAS -DUSE_METAL -DIRIS_DEV -DACCELERATE_NEW_LAPACK \
 *      -I. -o /tmp/cmp_vae_enc_mps debug/cmp_qimage_vae_enc.c \
 *      $(ls *.mps.o | grep -vE '^(main|iris_vae_qimage)\.mps\.o') iris_metal.o iris_dev_metal.o \
 *      -lm -framework Accelerate -framework Metal -framework MetalPerformanceShaders \
 *      -framework MetalPerformanceShadersGraph -framework Foundation
 * After make cuda:
 *   gcc -O3 -march=native -ffast-math -DIRIS_DEV -I. -o /tmp/cmp_vae_enc debug/cmp_qimage_vae_enc.c \
 *      $(ls *.o | grep -vE '^(main|iris_vae_qimage|iris_cli|linenoise|embcache)\.o') \
 *      -lm -L$CUDA_HOME/lib64 -lcublas -lcudart -lstdc++
 *
 * Usage: cmp_vae_enc MODEL_DIR REF_DIR NUM_REFS H W [TOLERANCE]
 *
 * H, W: reference size in pixels. enc_<stage>.bin exist for the first
 * reference only. Prints the relative L2 error of each stage and of
 * ref<i>_latent.bin, and fails if one exceeds TOLERANCE (default 1e-5).
 *
 * The f32 noise floor is above that default on some stages: upstream's own
 * f32 run differs from its f64 run by 1.9e-5 at down3 and 2.0e-5 at norm_out,
 * and C and upstream each carry their own noise. Pass 3e-5 for the CPU path.
 * In bf16 upstream itself is 3.3e-2 from f32 at down3; pass 0.04 on the GPU.
 */

#include <stdio.h>
#include <stdlib.h>

static void stage_host(const char *name, int index, const float *x, int c, int h, int w);
static void stage_dev(const char *name, int index, const float *x, int c, int h, int w);
#define QV_STAGE(name, index, x, c, h, w) stage_host(name, index, x, c, h, w)
#define QD_STAGE(name, index, x, c, h, w) stage_dev(name, index, x, c, h, w)
#include "iris_vae_qimage.c"

#ifdef USE_METAL
#include "iris_metal.h"
#endif

static const char *g_ref;
static int g_stages = 1, g_fail;
static double g_tol = 1e-5;

static float *read_floats(const char *path, size_t n) {
    FILE *f = fopen(path, "rb");
    float *data = malloc(n * sizeof(float));
    if (!f || fread(data, sizeof(float), n, f) != n) {
        fprintf(stderr, "cannot read %zu floats from %s\n", n, path);
        exit(1);
    }
    fclose(f);
    return data;
}

static void report(const char *label, const float *x, const float *ref, size_t n) {
    double e = 0, r = 0;
    for (size_t i = 0; i < n; i++) {
        e += (x[i] - ref[i]) * (double)(x[i] - ref[i]);
        r += ref[i] * (double)ref[i];
    }
    double rel = sqrt(e / r);
    printf("%-14s rel L2 %.3e\n", label, rel);
    if (!(rel <= g_tol)) g_fail = 1;
}

static void stage_host(const char *name, int index, const float *x, int c, int h, int w) {
    if (!g_stages) return;
    char label[64], path[1024];
    if (index >= 0) snprintf(label, sizeof(label), "%s%d", name, index);
    else snprintf(label, sizeof(label), "%s", name);
    snprintf(path, sizeof(path), "%s/enc_%s.bin", g_ref, label);
    size_t n = (size_t)c * h * w;
    float *ref = read_floats(path, n);
    /* The dump is before the SiLU, which the C norm applies. */
    if (!strcmp(name, "norm_out"))
        for (size_t i = 0; i < n; i++) ref[i] = ref[i] / (1.0f + exp(-ref[i]));
    report(label, x, ref, n);
    free(ref);
}

static void stage_dev(const char *name, int index, const float *x, int c, int h, int w) {
#ifdef IRIS_DEV
    size_t hw = (size_t)h * w;
    float *hwc = malloc((size_t)c * hw * sizeof(float));
    float *nchw = malloc((size_t)c * hw * sizeof(float));
    iris_dev_download(hwc, x, (size_t)c * hw * sizeof(float));
    for (int ch = 0; ch < c; ch++)
        for (size_t p = 0; p < hw; p++) nchw[ch * hw + p] = hwc[p * c + ch];
    stage_host(name, index, nchw, c, h, w);
    free(hwc);
    free(nchw);
#else
    (void)name, (void)index, (void)x, (void)c, (void)h, (void)w;
#endif
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s MODEL_DIR REF_DIR NUM_REFS H W [TOLERANCE]\n", argv[0]);
        return 1;
    }
    g_ref = argv[2];
    int refs = atoi(argv[3]), h = atoi(argv[4]), w = atoi(argv[5]);
    if (argc > 6) g_tol = atof(argv[6]);

#ifdef USE_METAL
    iris_metal_init();
#endif
#ifdef IRIS_DEV
    iris_dev_init();
    printf("device: %s\n", iris_dev_available() ? iris_dev_name() : "none");
#endif
    iris_vae_qimage_t *vae = iris_vae_load_qimage(argv[1]);
    if (!vae) return 1;

    for (int i = 0; i < refs; i++) {
        char path[1024], label[32];
        snprintf(path, sizeof(path), "%s/ref%d_rgba.bin", g_ref, i);
        float *rgba = read_floats(path, (size_t)4 * h * w);
        g_stages = i == 0;
        float *z = iris_vae_encode_qimage(vae, rgba, h, w);
        if (!z) return 1;
        size_t n = (size_t)vae->z_dim * (h / 16) * (w / 16);
        snprintf(path, sizeof(path), "%s/ref%d_latent.bin", g_ref, i);
        float *ref = read_floats(path, n);
        snprintf(label, sizeof(label), "ref%d_latent", i);
        report(label, z, ref, n);
        free(z);
        free(ref);
        free(rgba);
    }

    /* Release frees the encoder; encoding again must reload it. */
    iris_vae_release_qimage(vae);
    if (vae->enc) g_fail = 1;
    g_stages = 0;
    char path[1024];
    snprintf(path, sizeof(path), "%s/ref0_rgba.bin", g_ref);
    float *rgba = read_floats(path, (size_t)4 * h * w);
    float *z = iris_vae_encode_qimage(vae, rgba, h, w);
    if (!z) return 1;
    size_t n = (size_t)vae->z_dim * (h / 16) * (w / 16);
    snprintf(path, sizeof(path), "%s/ref0_latent.bin", g_ref);
    report("reloaded", z, read_floats(path, n), n);
    iris_vae_free_qimage(vae);
    printf("%s\n", g_fail ? "FAIL" : "ok");
    return g_fail;
}
