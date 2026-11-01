/*
 * Compare the Qwen-Image text encoder with reference images against the
 * tensors dumped by debug/qimage_reference.py --image, starting from the
 * dumped vision embeddings: tokens and positions exactly, then the outputs
 * of layers 0, 1, 2 and 35 and the final embeddings.
 *
 * This file includes iris_qwen3.c to reach its model and to change the layers
 * it returns, so link everything except that object.
 *
 * Build, in bash, after make blas:
 *   cc -O3 -march=native -ffast-math -DUSE_BLAS -DACCELERATE_NEW_LAPACK -I. \
 *      -o /tmp/dump_qimage_text debug/dump_qimage_text.c \
 *      $(ls *.o | grep -v -e '^main.o' -e '^iris_qwen3.o') -lm -framework Accelerate
 * After make mps:
 *   cc -O3 -march=native -ffast-math -DUSE_BLAS -DUSE_METAL -DIRIS_DEV \
 *      -DACCELERATE_NEW_LAPACK -I. -o /tmp/dump_qimage_text_mps debug/dump_qimage_text.c \
 *      $(ls *.mps.o | grep -v -e '^main.mps.o' -e '^iris_qwen3.mps.o') \
 *      iris_metal.o iris_dev_metal.o -lm -framework Accelerate -framework Metal \
 *      -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph \
 *      -framework Foundation
 * After make cuda:
 *   gcc -O3 -march=native -ffast-math -DIRIS_DEV -I. -o /tmp/dump_qimage_text debug/dump_qimage_text.c \
 *      $(ls *.o | grep -v -e '^main.o' -e '^iris_qwen3.o') \
 *      -lm -L$CUDA_HOME/lib64 -lcublas -lcudart -lstdc++
 *
 * Usage: dump_qimage_text MODEL_DIR REF_DIR PROMPT [MMAP (default 1)]
 *
 * Reads from REF_DIR: tokens.txt, position_ids.bin, grid_thw.txt,
 * image_embeds.bin, deep<k>.bin, txt_layer<l>.bin, txt_layer3_in.bin,
 * image_pad_mask.txt, emb.bin. Prints relative L2 errors and exits non-zero
 * if tokens or positions differ.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../iris_qwen3.c"
#ifdef USE_METAL
#include "iris_metal.h"
#endif
#ifdef IRIS_DEV
#include "iris_dev.h"
#endif

#define MAX_IMAGES 16

static const char *ref_dir;

static void *read_file(const char *name, size_t *bytes) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", ref_dir, name);
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path);
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    size_t n = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    void *data = malloc(n);
    if (!data || fread(data, 1, n, f) != n) {
        fprintf(stderr, "cannot read %s\n", path);
        exit(1);
    }
    fclose(f);
    *bytes = n;
    return data;
}

static double rel_l2(const float *a, const float *b, size_t n) {
    double e = 0, r = 0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - b[i];
        e += d * d;
        r += (double)b[i] * b[i];
    }
    return sqrt(e / r);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL_DIR REF_DIR PROMPT [MMAP]\n", argv[0]);
        return 1;
    }
    ref_dir = argv[2];
    const char *prompt = argv[3];
    int use_mmap = argc > 4 ? atoi(argv[4]) : 1;

#ifdef USE_METAL
    iris_metal_init();
#endif
#ifdef IRIS_DEV
    iris_dev_init();
#endif

    /* Vision tower output, split per image */
    char name[64];
    size_t bytes;
    float *embeds = read_file("image_embeds.bin", &bytes), *deep[QWEN3_VISION_DEEPSTACK];
    for (int k = 0; k < QWEN3_VISION_DEEPSTACK; k++) {
        snprintf(name, sizeof(name), "deep%d.bin", k);
        deep[k] = read_file(name, &bytes);
    }
    qwen3_image_t images[MAX_IMAGES];
    int n = 0, slots[MAX_IMAGES], rows[MAX_IMAGES];
    size_t offset = 0;
    char path[1024];
    snprintf(path, sizeof(path), "%s/grid_thw.txt", ref_dir);
    FILE *f = fopen(path, "r");
    int t, gh, gw;
    while (f && n < MAX_IMAGES && fscanf(f, "%d %d %d", &t, &gh, &gw) == 3) {
        qwen3_image_t *img = &images[n];
        img->h = gh / 2;
        img->w = gw / 2;
        img->embeds = embeds + offset;
        for (int k = 0; k < QWEN3_VISION_DEEPSTACK; k++) img->deep[k] = deep[k] + offset;
        slots[n++] = img->h * img->w;
        offset += (size_t)(img->h * img->w) * 4096;
    }
    if (!f || !n) {
        fprintf(stderr, "cannot read grid_thw.txt\n");
        return 1;
    }
    fclose(f);

    qwen3_encoder_t *enc = qwen3_encoder_load(argv[1], use_mmap);
    if (!enc) return 1;
    qwen3_set_extraction_mode(enc, QWEN3_EXTRACT_QIMAGE);
    qwen3_model_t *model = enc->model;
    size_t hidden = model->hidden_size;

    /* Tokens, exactly */
    int num_tokens, num_system, bad = 0;
    int *tokens = qwen3_tokenize_chat_images(enc->tokenizer, QWEN3_QIMAGE_SYSTEM, prompt,
                                             slots, n, &num_tokens, &num_system, rows);
    snprintf(path, sizeof(path), "%s/tokens.txt", ref_dir);
    f = fopen(path, "r");
    int count = 0, id;
    while (f && fscanf(f, "%d", &id) == 1) {
        if (count >= num_tokens || tokens[count] != id) bad = 1;
        count++;
    }
    if (f) fclose(f);
    printf("tokens: %d of %d, %s, system %d\n", num_tokens, count,
           !bad && count == num_tokens ? "identical" : "DIFFERENT", num_system);
    if (bad || count != num_tokens) return 1;
    free(tokens);

    /* Layers 0, 1, 2, 35. The stream holds deepstack k after layer k. */
    int pos[MAX_IMAGES], seq;
    size_t cells = (size_t)num_tokens * hidden;
    for (int l = 0; l < 3; l++) model->out_layers[l] = l;
    model->out_layers[3] = model->num_layers - 1;
    model->num_out_layers = 4;
    float *emb = qwen3_encode_images(enc, prompt, images, n, &seq, pos);
    if (!emb) return 1;
    free(emb);
    model->images = images;
    model->image_rows = rows;
    model->num_images = n;
    for (int l = 0; l < 3; l++) {
        snprintf(name, sizeof(name), "txt_layer%d.bin", l);
        float *ref = read_file(name, &bytes);
        qwen3_add_deepstack(model, ref, l);
        printf("layer %d (with deepstack): %.3e\n", l, rel_l2(model->layer_outputs[l], ref, cells));
        free(ref);
    }
    float *ref = read_file("txt_layer3_in.bin", &bytes);
    printf("layer 3 input: %.3e\n", rel_l2(model->layer_outputs[2], ref, cells));
    free(ref);
    ref = read_file("txt_layer35.bin", &bytes);
    printf("layer 35: %.3e\n", rel_l2(model->layer_outputs[3], ref, cells));
    free(ref);

    /* Positions, exactly: a table built from the dump must be identical.
     * position_ids is [axes, 1, seq], with the three spatial axes last. */
    int half = model->head_dim / 2;
    size_t table = (size_t)num_tokens * half * sizeof(float);
    float *c = malloc(table), *s = malloc(table);
    qwen3_set_rope_positions(model, NULL, num_tokens);
    memcpy(c, model->rope_cos, table);
    memcpy(s, model->rope_sin, table);
    int *ids = read_file("position_ids.bin", &bytes);
    int axes = (int)(bytes / sizeof(int) / num_tokens), *last = ids + (size_t)(axes - 3) * num_tokens;
    for (int r = 0; r < num_tokens; r++)
        qwen3_rope_row(model, r, last[r], last[num_tokens + r], last[2 * num_tokens + r]);
    bad = memcmp(c, model->rope_cos, table) || memcmp(s, model->rope_sin, table);
    printf("positions (%d axes): %s\n", axes, bad ? "DIFFERENT" : "identical");
    model->images = NULL;
    model->image_rows = NULL;
    model->num_images = 0;
    if (bad) return 1;
    free(ids);
    free(c);
    free(s);

    /* The real output: text rows of emb.bin, block positions from the mask */
    qwen3_set_extraction_mode(enc, QWEN3_EXTRACT_QIMAGE);
    emb = qwen3_encode_images(enc, prompt, images, n, &seq, pos);
    if (!emb) return 1;
    float *ref_emb = read_file("emb.bin", &bytes);
    int ref_rows = (int)(bytes / sizeof(float) / hidden), kept = 0, block = 0, was_slot = 0;
    snprintf(path, sizeof(path), "%s/image_pad_mask.txt", ref_dir);
    f = fopen(path, "r");
    float *ref_txt = malloc((size_t)ref_rows * hidden * sizeof(float));
    for (int r = 0; r < ref_rows; r++) {
        int slot;
        if (!f || fscanf(f, "%d", &slot) != 1) {
            fprintf(stderr, "short image_pad_mask.txt\n");
            return 1;
        }
        if (slot && !was_slot && (block >= n || pos[block++] != kept)) bad = 1;
        if (!slot) memcpy(ref_txt + kept++ * hidden, ref_emb + r * hidden, hidden * sizeof(float));
        was_slot = slot;
    }
    fclose(f);
    if (seq != kept || block != n) bad = 1;
    printf("text rows %d of %d, block positions:", seq, kept);
    for (int i = 0; i < n; i++) printf(" %d", pos[i]);
    printf(" %s\n", bad ? "DIFFERENT" : "identical");
    if (bad) return 1;
    printf("embeddings (text rows): %.3e\n", rel_l2(emb, ref_txt, (size_t)seq * hidden));

    free(ref_txt);
    free(ref_emb);
    free(emb);
    for (int k = 0; k < QWEN3_VISION_DEEPSTACK; k++) free(deep[k]);
    free(embeds);
    qwen3_encoder_free(enc);
    return 0;
}
