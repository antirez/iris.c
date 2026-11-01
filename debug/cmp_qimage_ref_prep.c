/*
 * Compare Qwen-Image reference preparation with the dumps of
 * debug/qimage_reference.py --image.
 *
 * Build (after make blas, from the repository root):
 *   make lib
 *   cc -O2 -I. -o /tmp/cmp_ref_prep debug/cmp_qimage_ref_prep.c libiris.a -lm -framework Accelerate
 *
 * Usage: cmp_ref_prep REF_DIR REF_RESOLUTION IMAGE...
 *
 * For image i: prepares IMAGE at REF_RESOLUTION^2 and prints the RMS error
 * (in 0..255) against ref<i>_rgba.bin and ref<i>_vision_rgb.bin. Then feeds
 * the dumped pixels back as an input of the same size, where no resize
 * happens, and checks the bytes come back exactly.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "iris.h"
#include "iris_qimage.h"

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

static double rms(const float *a, const float *b, size_t n, double scale) {
    double s = 0;
    for (size_t i = 0; i < n; i++) s += (a[i] - b[i]) * (double)(a[i] - b[i]);
    return sqrt(s / n) * scale;
}

static float *vision_floats(const iris_image *rgba) {
    size_t npix = (size_t)rgba->width * rgba->height;
    iris_image *vis = iris_ref_vision_qimage(rgba);
    float *f = malloc(npix * 3 * sizeof(float));
    for (int c = 0; c < 3; c++)
        for (size_t p = 0; p < npix; p++) f[c * npix + p] = vis->data[p * 3 + c];
    iris_image_free(vis);
    return f;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s REF_DIR REF_RESOLUTION IMAGE...\n", argv[0]);
        return 1;
    }
    int res = atoi(argv[2]), fail = 0;

    /* calculate_dimensions on a few sizes; 48 and 80 are half-even ties. */
    static const int sizes[][5] = {
        {1024, 1024, 1024, 1024, 1024}, {1920, 1080, 1024, 1376, 768},
        {333, 1000, 1024, 576, 1760},   {3000, 1000, 320, 544, 192},
        {1, 1, 80, 64, 64},             {1, 1, 48, 64, 64},
    };
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++) {
        const int *s = sizes[i];
        int w, h;
        iris_ref_size_qimage(s[0], s[1], s[2] * s[2], &w, &h);
        if (w != s[3] || h != s[4]) {
            printf("size %dx%d at %d: got %dx%d\n", s[0], s[1], s[2], w, h);
            fail = 1;
        }
    }

    for (int i = 0; i < argc - 3; i++) {
        iris_image *src = iris_image_load(argv[i + 3]);
        iris_image *rgba = src ? iris_ref_prepare_qimage(src, res * res) : NULL;
        if (!rgba) return 1;
        int w = rgba->width, h = rgba->height;
        size_t npix = (size_t)w * h;

        char path[1024];
        snprintf(path, sizeof(path), "%s/ref%d_rgba.bin", argv[1], i);
        float *ref = read_floats(path, npix * 4);
        snprintf(path, sizeof(path), "%s/ref%d_vision_rgb.bin", argv[1], i);
        float *ref_vis = read_floats(path, npix * 3);

        double e_rgba = rms(iris_ref_pixels_qimage(rgba), ref, npix * 4, 127.5);
        double e_vis = rms(vision_floats(rgba), ref_vis, npix * 3, 1.0);

        /* Same size: the dumped bytes must come back exactly. */
        iris_image *same = iris_image_create(w, h, 4);
        for (int c = 0; c < 4; c++)
            for (size_t p = 0; p < npix; p++)
                same->data[p * 4 + c] = (uint8_t)((ref[c * npix + p] + 1.0f) * 127.5f + 0.5f);
        iris_image *out = iris_ref_prepare_qimage(same, w * h);
        int exact = out->width == w && out->height == h &&
                    !memcmp(out->data, same->data, npix * 4) &&
                    !memcmp(vision_floats(out), ref_vis, npix * 3 * sizeof(float));
        /* -ffast-math may reassociate 2 * (v / 255) - 1 in the last bit. */
        float *again = iris_ref_pixels_qimage(out);
        for (size_t p = 0; p < npix * 4; p++) exact = exact && fabsf(again[p] - ref[p]) < 2e-7f;

        printf("ref%d %dx%d -> %dx%d: rgba rms %.3f/255, vision rms %.3f/255, same-size %s\n",
               i, src->width, src->height, w, h, e_rgba, e_vis, exact ? "exact" : "DIFFERS");
        if (e_rgba > 2 || e_vis > 2 || !exact) fail = 1;
    }
    return fail;
}
