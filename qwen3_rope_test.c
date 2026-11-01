/* Text encoder positions with images (MRoPE). Rounding hides wrong positions in
 * the output images, so "make ropetest" checks the RoPE tables against a layout
 * written by hand. It needs no weights. This file includes iris_qwen3.c to reach
 * them, so it builds without that object. */
#include "iris_qwen3.c"

/* (frame, height, width) of each row: 3 text tokens, a 2x3 image, 2 text tokens,
 * a 3x1 image, 1 text token. An image takes max(h, w) positions. */
static const int expected[][3] = {
    {0, 0, 0}, {1, 1, 1}, {2, 2, 2},
    {3, 3, 3}, {3, 3, 4}, {3, 3, 5},
    {3, 4, 3}, {3, 4, 4}, {3, 4, 5},
    {6, 6, 6}, {7, 7, 7},
    {8, 8, 8}, {8, 9, 8}, {8, 10, 8},
    {11, 11, 11},
};

int main(void) {
    enum { SEQ = sizeof(expected) / sizeof(expected[0]), HEAD_DIM = 128, HALF = HEAD_DIM / 2 };
    static float cos_table[SEQ * HALF], sin_table[SEQ * HALF];
    const qwen3_image_t images[2] = {{.h = 2, .w = 3}, {.h = 3, .w = 1}};
    const int image_rows[2] = {3, 11};

    /* mrope_section [24, 20, 20], as in the Qwen3-VL config */
    qwen3_model_t model = {
        .head_dim = HEAD_DIM, .rope_theta = 5e6f, .mrope_h = 20, .mrope_w = 20,
        .rope_cos = cos_table, .rope_sin = sin_table,
        .images = images, .image_rows = image_rows, .num_images = 2,
    };
    qwen3_set_rope_positions(&model, NULL, SEQ);

    for (int s = 0; s < SEQ; s++) {
        for (int i = 0; i < HALF; i++) {
            /* Below 60 the frequencies take turns: frame, height, width. */
            int axis = i < 60 ? i % 3 : 0;
            float angle = expected[s][axis] / powf(5e6f, (float)(2 * i) / HEAD_DIM);
            if (fabsf(cos_table[s * HALF + i] - cosf(angle)) > 1e-6f ||
                fabsf(sin_table[s * HALF + i] - sinf(angle)) > 1e-6f) {
                fprintf(stderr, "row %d frequency %d: not at axis %d position %d\n",
                        s, i, axis, expected[s][axis]);
                return 1;
            }
        }
    }
    return 0;
}
