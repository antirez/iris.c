/*
 * Write the text embeddings iris computes for a prompt, to compare with
 * the emb.bin of debug/qimage_reference.py or debug/flux3_reference.py.
 *
 * Build (after make blas / make mps, from the repository root):
 *   make lib
 *   cc -O2 -I. -o /tmp/dump_emb debug/dump_emb.c libiris.a -lm -lopenblas
 * (Accelerate builds link with -framework Accelerate. For MPS, build with
 * -DUSE_METAL -DIRIS_DEV and link as in debug/dump_qimage.c.)
 *
 * Usage: /tmp/dump_emb MODEL_DIR PROMPT OUT.bin
 * Writes [seq, text_dim] raw float32 and prints seq and text_dim.
 */

#include <stdio.h>
#include <stdlib.h>
#include "iris.h"
#ifdef USE_METAL
#include "iris_metal.h"
#endif
#ifdef IRIS_DEV
#include "iris_dev.h"
#endif

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s MODEL_DIR PROMPT OUT.bin\n", argv[0]);
        return 1;
    }
#ifdef USE_METAL
    iris_metal_init();
#endif
#ifdef IRIS_DEV
    iris_dev_init();
#endif
    iris_ctx *ctx = iris_load_dir(argv[1]);
    if (!ctx) {
        fprintf(stderr, "failed to load %s: %s\n", argv[1], iris_get_error());
        return 1;
    }
    iris_set_mmap(ctx, 1);   /* as the CLI */
    int seq = 0, dim = iris_text_dim(ctx);
    float *emb = iris_encode_text(ctx, argv[2], &seq);
    FILE *f = fopen(argv[3], "wb");
    if (!emb || !f || fwrite(emb, sizeof(float), (size_t)seq * dim, f) != (size_t)seq * dim) {
        fprintf(stderr, "failed to encode or write %s\n", argv[3]);
        return 1;
    }
    fclose(f);
    printf("seq %d text_dim %d\n", seq, dim);
    free(emb);
    iris_free(ctx);
    return 0;
}
