/*
 * Write the initial noise iris generates for a seed, so reference runs
 * (debug/qimage_reference.py --noise, debug/flux3_reference.py --noise)
 * start from the same latent as the C code.
 *
 * Build (after make blas / make generic, from the repository root):
 *   make lib
 *   cc -O2 -I. -o /tmp/dump_noise debug/dump_noise.c libiris.a -lm -lopenblas
 * (Accelerate builds link with -framework Accelerate instead of -lopenblas.)
 *
 * Usage: /tmp/dump_noise CHANNELS LATENT_H LATENT_W SEED OUT.bin
 * Qwen-Image: CHANNELS 64, LATENT_H = height / 16, LATENT_W = width / 16.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

float *iris_init_noise(int batch, int channels, int h, int w, int64_t seed);

int main(int argc, char **argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s CHANNELS LATENT_H LATENT_W SEED OUT.bin\n", argv[0]);
        return 1;
    }
    int c = atoi(argv[1]), h = atoi(argv[2]), w = atoi(argv[3]);
    float *noise = iris_init_noise(1, c, h, w, atoll(argv[4]));
    FILE *f = fopen(argv[5], "wb");
    if (!noise || !f || fwrite(noise, sizeof(float), (size_t)c * h * w, f) != (size_t)c * h * w) {
        fprintf(stderr, "failed to write %s\n", argv[5]);
        return 1;
    }
    fclose(f);
    free(noise);
    return 0;
}
