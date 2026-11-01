/*
 * Run the Qwen-Image reference pipeline end to end from real images with
 * given noise, at any reference resolution, and compare the final latent
 * with a debug/qimage_reference.py --image run. Without REF_DIR/noise.bin
 * it uses the seed 42 noise. Writes OUT/latent.bin and OUT/image.png.
 *
 * Build, in bash, after make blas:
 *   cc -O3 -march=native -ffast-math -DUSE_BLAS -DACCELERATE_NEW_LAPACK -I. \
 *      -o /tmp/dump_qimage_e2e debug/dump_qimage_e2e.c \
 *      $(ls *.o | grep -v -e '^main.o') -lm -framework Accelerate
 * After make mps:
 *   cc -O3 -march=native -ffast-math -DUSE_BLAS -DUSE_METAL -DIRIS_DEV \
 *      -DACCELERATE_NEW_LAPACK -I. -o /tmp/dump_qimage_e2e_mps debug/dump_qimage_e2e.c \
 *      $(ls *.mps.o | grep -v -e '^main.mps.o') \
 *      iris_metal.o iris_dev_metal.o -lm -framework Accelerate -framework Metal \
 *      -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
 *      -framework Foundation
 * After make cuda:
 *   gcc -O3 -march=native -ffast-math -DIRIS_DEV -I. -o /tmp/dump_qimage_e2e debug/dump_qimage_e2e.c \
 *      $(ls *.o | grep -v -e '^main.o') -lm -L$CUDA_HOME/lib64 -lcublas -lcudart -lstdc++
 *
 * Usage: dump_qimage_e2e MODEL_DIR REF_DIR OUT W H RESOLUTION PROMPT IMAGE...
 * W H and RESOLUTION as qimage_reference.py -W -H --ref-resolution, 0 for
 * the default (the target follows the last reference).
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "iris.h"
#include "iris_qimage.h"
#ifdef USE_METAL
#include "iris_metal.h"
#endif
#ifdef IRIS_DEV
#include "iris_dev.h"
#endif

#define MAX_REFS 16

static float *read_floats(const char *path, size_t *count) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *count = (size_t)ftell(f) / sizeof(float);
    fseek(f, 0, SEEK_SET);
    float *data = malloc(*count * sizeof(float));
    if (!data || fread(data, sizeof(float), *count, f) != *count) exit(1);
    fclose(f);
    return data;
}

int main(int argc, char **argv) {
    if (argc < 9 || argc - 8 > MAX_REFS) {
        fprintf(stderr, "usage: %s MODEL_DIR REF_DIR OUT W H RESOLUTION PROMPT IMAGE...\n", argv[0]);
        return 1;
    }
    const char *ref_dir = argv[2], *out = argv[3], *prompt = argv[7];
    int res = atoi(argv[6]), n = argc - 8;
    int area = res > 0 ? res * res : IRIS_QIMAGE_REF_AREA;

#ifdef USE_METAL
    iris_metal_init();
#endif
#ifdef IRIS_DEV
    iris_dev_init();
#endif

    iris_image *refs[MAX_REFS];
    for (int i = 0; i < n; i++) {
        refs[i] = iris_image_load(argv[8 + i]);
        if (!refs[i]) {
            fprintf(stderr, "cannot load %s\n", argv[8 + i]);
            return 1;
        }
    }

    iris_ctx *ctx = iris_load_dir(argv[1]);
    if (!ctx) {
        fprintf(stderr, "failed to load %s: %s\n", argv[1], iris_get_error());
        return 1;
    }
    iris_set_mmap(ctx, 1);

    char path[1024];
    size_t noise_size = 0, ref_size = 0;
    snprintf(path, sizeof(path), "%s/noise.bin", ref_dir);
    float *noise = read_floats(path, &noise_size);
    iris_params p = IRIS_PARAMS_DEFAULT;
    p.width = atoi(argv[4]);
    p.height = atoi(argv[5]);
    p.seed = 42;

    float *latent = NULL;
    iris_image *img = iris_generate_refs_qimage(ctx, prompt, (const iris_image **)refs, n, area,
                                                noise, (int)noise_size, &p, &latent);
    if (!img) {
        fprintf(stderr, "generation failed: %s\n", iris_get_error());
        return 1;
    }
    size_t cells = (size_t)IRIS_QIMAGE_LATENT_CHANNELS * (img->height / 16) * (img->width / 16);

    snprintf(path, sizeof(path), "%s/image.png", out);
    iris_image_save(img, path);
    snprintf(path, sizeof(path), "%s/latent.bin", out);
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(latent, sizeof(float), cells, f) != cells) return 1;
    fclose(f);

    snprintf(path, sizeof(path), "%s/latent.bin", ref_dir);
    float *ref = read_floats(path, &ref_size);
    if (ref && ref_size == cells) {
        double e = 0, r = 0;
        for (size_t i = 0; i < cells; i++) {
            double d = (double)latent[i] - ref[i];
            e += d * d;
            r += (double)ref[i] * ref[i];
        }
        printf("final latent: %.3e\n", sqrt(e / r));
    }
    free(ref);
    free(latent);
    free(noise);
    iris_image_free(img);
    for (int i = 0; i < n; i++) iris_image_free(refs[i]);
    iris_free(ctx);
    return 0;
}
