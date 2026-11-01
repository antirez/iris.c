/*
 * Qwen3-VL vision tower for Iris (Qwen-Image 2.1 reference images).
 *
 * One image at a time: patches -> 27 pre-norm blocks over all patches of the
 * image -> a main merger after the last block and deepstack mergers after
 * blocks 8, 16 and 24, each giving one [dim] row per 2x2 block of patches.
 * Patches run in block-major order (block row, block column, row in block,
 * column in block), so the merged rows are in raster order and the four
 * patches of a block are contiguous.
 *
 * Weights stay as bf16 in the mmapped safetensors files, under model.visual.
 * The CPU path converts each matrix to f32 right before its matmul. With a
 * device backend (CUDA / ROCm / Metal, see iris_dev.h) they are uploaded on
 * the first image and the blocks run on the GPU.
 */

#include "iris_qwen3.h"
#include "iris_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef IRIS_DEV
#include "iris_dev.h"
#endif

#define VIS_MAX_SHARDS 16
#define VIS_ROPE_THETA 10000.0f
#define VIS_EPS        1e-6f      /* LayerNorm */
#define VIS_CHANNELS   3

/* Linear with bias: y = x @ w^T + b. dw, db: device copies. */
typedef struct {
    const uint16_t *w;    /* mmapped [out, in] */
    float *b;             /* [out] */
    int in, out;
    uint16_t *dw;
    float *db;
} vis_linear_t;

/* LayerNorm with affine weight w and bias b. dscale, dshift: device copies
 * of w - 1 and b, for iris_dev_layer_norm_mod(). */
typedef struct {
    float *w, *b;
    int dim;
    float *dscale, *dshift;
} vis_norm_t;

typedef struct {
    vis_norm_t norm1, norm2;
    vis_linear_t qkv, proj, fc1, fc2;
} vis_block_t;

typedef struct {
    vis_norm_t norm;      /* per patch (main) or per merged row (deepstack) */
    vis_linear_t fc1, fc2;
} vis_merger_t;

struct qwen3_vision {
    safetensors_file_t *sf[VIS_MAX_SHARDS];
    int num_sf;

    int depth, dim, heads, head_dim, mlp, out_dim;
    int patch, merge, temporal, side;   /* side: position table [side^2, dim] */
    int deep_index[QWEN3_VISION_DEEPSTACK];

    float *patch_w, *patch_b;           /* [dim, VIS_CHANNELS * patch^2], temporal slices summed */
    float *pos;                         /* [side^2, dim] */
    vis_block_t *blocks;
    vis_merger_t merger, deep[QWEN3_VISION_DEEPSTACK];

    int dev_loaded;
    uint16_t *dev_patch_w;
    float *dev_patch_b;
};

/* What the host prepares for one image. */
typedef struct {
    int n;                /* patches */
    float *px;            /* [n, VIS_CHANNELS * patch^2] */
    float *pos;           /* [n, dim] */
    float *cos_t, *sin_t; /* [n, head_dim / 2] */
} vis_input_t;

/* ========================================================================
 * Image Input
 * ======================================================================== */

/* Row and column of patch p in the block-major order. */
static void patch_position(const qwen3_vision_t *v, int p, int gw, int *row, int *col) {
    int m = v->merge, blocks_w = gw / m, block = p / (m * m);
    *row = (block / blocks_w) * m + (p / m) % m;
    *col = (block % blocks_w) * m + p % m;
}

/* Taps t0, t1 = t0 + 1 (clamped) and the weight of t1, for position i of an
 * axis of length size, on the table with align_corners. */
static void axis_taps(int i, int size, int side, int *t0, int *t1, float *frac) {
    float src = (float)i * (side - 1) / (float)(size > 1 ? size - 1 : 1);
    float base = floorf(src);
    *t0 = (int)base < side - 1 ? (int)base : side - 1;
    *t1 = *t0 + 1 < side - 1 ? *t0 + 1 : side - 1;
    *frac = src - base;
}

static void position_embed(const qwen3_vision_t *v, float *out, int n, int gh, int gw) {
    int dim = v->dim;
    for (int p = 0; p < n; p++) {
        int row, col, h[2], w[2];
        float fh, fw;
        patch_position(v, p, gw, &row, &col);
        axis_taps(row, gh, v->side, &h[0], &h[1], &fh);
        axis_taps(col, gw, v->side, &w[0], &w[1], &fw);
        float *o = out + (size_t)p * dim;
        memset(o, 0, dim * sizeof(float));
        for (int a = 0; a < 2; a++) {
            for (int b = 0; b < 2; b++) {
                float weight = (a ? fh : 1.0f - fh) * (b ? fw : 1.0f - fw);
                const float *t = v->pos + (size_t)(h[a] * v->side + w[b]) * dim;
                for (int d = 0; d < dim; d++) o[d] += weight * t[d];
            }
        }
    }
}

/* [h * freq | w * freq], with the same frequencies on both axes. */
static void rope_tables(const qwen3_vision_t *v, float *cos_t, float *sin_t, int n, int gw) {
    int half = v->head_dim / 2, freqs = half / 2;
    float *inv = malloc(freqs * sizeof(float));
    for (int k = 0; k < freqs; k++)
        inv[k] = 1.0f / powf(VIS_ROPE_THETA, (float)(2 * k) / half);
    for (int p = 0; p < n; p++) {
        int row, col;
        patch_position(v, p, gw, &row, &col);
        float *c = cos_t + (size_t)p * half, *s = sin_t + (size_t)p * half;
        for (int k = 0; k < freqs; k++) {
            c[k] = cosf(row * inv[k]);
            s[k] = sinf(row * inv[k]);
            c[freqs + k] = cosf(col * inv[k]);
            s[freqs + k] = sinf(col * inv[k]);
        }
    }
    free(inv);
}

/* px[p, (c * patch + y) * patch + x] = rgb(c, y, x) mapped from 0..255 to -1..1 */
static void patch_pixels(const qwen3_vision_t *v, float *px, const float *rgb,
                         int n, int h, int w) {
    int ps = v->patch, gw = w / ps;
    for (int p = 0; p < n; p++) {
        int row, col;
        patch_position(v, p, gw, &row, &col);
        float *o = px + (size_t)p * VIS_CHANNELS * ps * ps;
        for (int c = 0; c < VIS_CHANNELS; c++)
            for (int y = 0; y < ps; y++)
                for (int x = 0; x < ps; x++) {
                    size_t at = ((size_t)c * h + row * ps + y) * w + col * ps + x;
                    *o++ = 2.0f * rgb[at] / 255.0f - 1.0f;
                }
    }
}

static void input_free(vis_input_t *in) {
    free(in->px);
    free(in->pos);
    free(in->cos_t);
    free(in->sin_t);
}

static int input_prepare(const qwen3_vision_t *v, vis_input_t *in, const float *rgb,
                         int h, int w) {
    int gh = h / v->patch, gw = w / v->patch, half = v->head_dim / 2;
    size_t n = (size_t)gh * gw;
    in->n = gh * gw;
    in->px = malloc(n * VIS_CHANNELS * v->patch * v->patch * sizeof(float));
    in->pos = malloc(n * v->dim * sizeof(float));
    in->cos_t = malloc(n * half * sizeof(float));
    in->sin_t = malloc(n * half * sizeof(float));
    if (!in->px || !in->pos || !in->cos_t || !in->sin_t) {
        input_free(in);
        return 0;
    }
    patch_pixels(v, in->px, rgb, in->n, h, w);
    position_embed(v, in->pos, in->n, gh, gw);
    rope_tables(v, in->cos_t, in->sin_t, in->n, gw);
    return 1;
}

/* ========================================================================
 * CPU Path
 * ======================================================================== */

/* Activations for n patches. a and wide also hold merger rows. */
typedef struct {
    float *x;      /* [n, dim] */
    float *norm;   /* [n, dim] */
    float *qkv;    /* [n, 3 * dim] */
    float *a;      /* [n, dim] */
    float *wide;   /* [n, max(mlp, dim)] */
} vis_work_t;

static void work_free(vis_work_t *w) {
    free(w->x);
    free(w->norm);
    free(w->qkv);
    free(w->a);
    free(w->wide);
}

static int work_alloc(const qwen3_vision_t *v, vis_work_t *w, int n) {
    size_t row = v->dim, wide = v->mlp > v->dim ? v->mlp : v->dim;
    w->x = malloc(n * row * sizeof(float));
    w->norm = malloc(n * row * sizeof(float));
    w->qkv = malloc(n * 3 * row * sizeof(float));
    w->a = malloc(n * row * sizeof(float));
    w->wide = malloc(n * wide * sizeof(float));
    if (!w->x || !w->norm || !w->qkv || !w->a || !w->wide) {
        work_free(w);
        return 0;
    }
    return 1;
}

static void linear(float *y, const float *x, const vis_linear_t *l, int rows) {
    iris_linear_nobias_bf16(y, x, l->w, rows, l->in, l->out);
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < l->out; c++) y[(size_t)r * l->out + c] += l->b[c];
}

static void layer_norm(float *out, const float *x, const vis_norm_t *ln, int rows) {
    int dim = ln->dim;
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * dim;
        float *o = out + (size_t)r * dim;
        double mean = 0, var = 0;
        for (int i = 0; i < dim; i++) mean += xr[i];
        mean /= dim;
        for (int i = 0; i < dim; i++) {
            double d = xr[i] - mean;
            var += d * d;
        }
        float inv = (float)(1.0 / sqrt(var / dim + VIS_EPS));
        float m = (float)mean;
        for (int i = 0; i < dim; i++) o[i] = (xr[i] - m) * inv * ln->w[i] + ln->b[i];
    }
}

/* Rotate the pairs (i, i + head_dim / 2) of every head, stride ld. */
static void apply_rope(float *x, int ld, const float *cos_t, const float *sin_t,
                       int seq, int heads, int head_dim) {
    int half = head_dim / 2;
    for (int s = 0; s < seq; s++) {
        const float *c = cos_t + (size_t)s * half, *sn = sin_t + (size_t)s * half;
        for (int h = 0; h < heads; h++) {
            float *r = x + (size_t)s * ld + h * head_dim;
            for (int i = 0; i < half; i++) {
                float x0 = r[i], x1 = r[half + i];
                r[i] = x0 * c[i] - x1 * sn[i];
                r[half + i] = x1 * c[i] + x0 * sn[i];
            }
        }
    }
}

/* Attention over [n, 3 * dim] q | k | v rows, one contiguous head at a time. */
static int attention(float *out, const float *qkv, int n, int heads, int head_dim) {
    int dim = heads * head_dim;
    size_t hd = head_dim, rows = n;
    float scale = 1.0f / sqrtf((float)head_dim);
    float *q = malloc(3 * rows * hd * sizeof(float));
    float *o = malloc(rows * hd * sizeof(float));
    float *scores = malloc(rows * rows * sizeof(float));
    if (!q || !o || !scores) {
        free(q);
        free(o);
        free(scores);
        return 0;
    }
    float *k = q + rows * hd, *v = k + rows * hd;
    for (int h = 0; h < heads; h++) {
        for (size_t r = 0; r < rows; r++) {
            const float *in = qkv + r * 3 * dim + h * hd;
            memcpy(q + r * hd, in, hd * sizeof(float));
            memcpy(k + r * hd, in + dim, hd * sizeof(float));
            memcpy(v + r * hd, in + 2 * dim, hd * sizeof(float));
        }
        iris_matmul_t(scores, q, k, n, head_dim, n);
        for (size_t i = 0; i < rows * rows; i++) scores[i] *= scale;
        iris_softmax_cpu(scores, n, n);
        iris_matmul(o, scores, v, n, n, head_dim);
        for (size_t r = 0; r < rows; r++)
            memcpy(out + r * dim + h * hd, o + r * hd, hd * sizeof(float));
    }
    free(q);
    free(o);
    free(scores);
    return 1;
}

static void gelu_exact(float *x, size_t n) {
    for (size_t i = 0; i < n; i++)
        x[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.70710678118654752f));
}

static void add(float *x, const float *a, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] += a[i];
}

/* x = patch embedding of the input (before the position embedding). */
static void patch_embed(const qwen3_vision_t *v, float *x, const vis_input_t *in) {
    int dim = v->dim, row_len = VIS_CHANNELS * v->patch * v->patch;
    iris_matmul_t(x, in->px, v->patch_w, in->n, row_len, dim);
    for (int r = 0; r < in->n; r++)
        for (int c = 0; c < dim; c++) x[(size_t)r * dim + c] += v->patch_b[c];
}

static int block_forward(const qwen3_vision_t *v, const vis_block_t *b, vis_work_t *w,
                         const vis_input_t *in) {
    int n = in->n, dim = v->dim, ld = 3 * dim;
    layer_norm(w->norm, w->x, &b->norm1, n);
    linear(w->qkv, w->norm, &b->qkv, n);
    apply_rope(w->qkv, ld, in->cos_t, in->sin_t, n, v->heads, v->head_dim);
    apply_rope(w->qkv + dim, ld, in->cos_t, in->sin_t, n, v->heads, v->head_dim);
    if (!attention(w->a, w->qkv, n, v->heads, v->head_dim)) return 0;
    linear(w->wide, w->a, &b->proj, n);
    add(w->x, w->wide, (size_t)n * dim);

    layer_norm(w->norm, w->x, &b->norm2, n);
    linear(w->wide, w->norm, &b->fc1, n);
    iris_gelu_tanh(w->wide, (size_t)n * v->mlp);
    linear(w->a, w->wide, &b->fc2, n);
    add(w->x, w->a, (size_t)n * dim);
    return 1;
}

/* out[n / merge^2, out_dim]: norm (per patch or per flattened block), then
 * the two linears over the flattened blocks. */
static void merger_forward(const qwen3_vision_t *v, const vis_merger_t *m, vis_work_t *w,
                           float *out, int n) {
    int merged = n / (v->merge * v->merge);
    layer_norm(w->norm, w->x, &m->norm, (int)((size_t)n * v->dim / m->norm.dim));
    linear(w->wide, w->norm, &m->fc1, merged);
    gelu_exact(w->wide, (size_t)merged * m->fc1.out);
    linear(out, w->wide, &m->fc2, merged);
}

/* The deepstack merger reading the output of block l, or NULL. */
static const vis_merger_t *deep_merger(const qwen3_vision_t *v, int l, int *k) {
    for (*k = 0; *k < QWEN3_VISION_DEEPSTACK; (*k)++)
        if (v->deep_index[*k] == l) return &v->deep[*k];
    return NULL;
}

static int cpu_forward(const qwen3_vision_t *v, const vis_input_t *in, float *embeds,
                       float **deep) {
    vis_work_t w;
    if (!work_alloc(v, &w, in->n)) return 0;
    patch_embed(v, w.x, in);
    add(w.x, in->pos, (size_t)in->n * v->dim);
    int ok = 1;
    for (int l = 0; ok && l < v->depth; l++) {
        int k;
        const vis_merger_t *m;
        ok = block_forward(v, &v->blocks[l], &w, in);
        if (ok && (m = deep_merger(v, l, &k))) merger_forward(v, m, &w, deep[k], in->n);
    }
    if (ok) merger_forward(v, &v->merger, &w, embeds, in->n);
    work_free(&w);
    return ok;
}

#ifdef IRIS_DEV
/* ========================================================================
 * GPU Path (CUDA / ROCm / Metal)
 * ======================================================================== */

/* Activations for n patches, in one allocation. out and norm also hold
 * merger rows. */
typedef struct {
    float *x;             /* [n, dim] residual stream */
    float *out;           /* [n, max(3 * dim, mlp)] f32 linear output */
    float *cos_t, *sin_t; /* [n, head_dim / 2] */
    float *px;            /* [n, row_len] */
    uint16_t *px_b;       /* [n, row_len] */
    uint16_t *norm;       /* [n, dim] */
    uint16_t *qkv;        /* [n, 3 * dim] */
    uint16_t *attn;       /* [n, dim] */
    uint16_t *hid;        /* [n, mlp] */
} vis_dev_t;

static void dev_free_linear(vis_linear_t *l) {
    iris_dev_free(l->dw);
    iris_dev_free(l->db);
    l->dw = NULL;
    l->db = NULL;
}

static void dev_free_norm(vis_norm_t *n) {
    iris_dev_free(n->dscale);
    iris_dev_free(n->dshift);
    n->dscale = n->dshift = NULL;
}

static void dev_free_merger(vis_merger_t *m) {
    dev_free_norm(&m->norm);
    dev_free_linear(&m->fc1);
    dev_free_linear(&m->fc2);
}

static void dev_unload(qwen3_vision_t *v) {
    for (int l = 0; v->blocks && l < v->depth; l++) {
        vis_block_t *b = &v->blocks[l];
        dev_free_norm(&b->norm1);
        dev_free_norm(&b->norm2);
        dev_free_linear(&b->qkv);
        dev_free_linear(&b->proj);
        dev_free_linear(&b->fc1);
        dev_free_linear(&b->fc2);
    }
    dev_free_merger(&v->merger);
    for (int k = 0; k < QWEN3_VISION_DEEPSTACK; k++) dev_free_merger(&v->deep[k]);
    iris_dev_free(v->dev_patch_w);
    iris_dev_free(v->dev_patch_b);
    v->dev_patch_w = NULL;
    v->dev_patch_b = NULL;
    v->dev_loaded = 0;
}

static float *dev_f32(const float *src, size_t n) {
    float *d = iris_dev_alloc(n * sizeof(float));
    if (d) iris_dev_upload(d, src, n * sizeof(float));
    return d;
}

static int dev_load_linear(vis_linear_t *l) {
    size_t bytes = (size_t)l->in * l->out * sizeof(uint16_t);
    l->dw = iris_dev_alloc(bytes);
    if (l->dw) iris_dev_upload(l->dw, l->w, bytes);
    l->db = dev_f32(l->b, l->out);
    return l->dw && l->db;
}

static int dev_load_norm(vis_norm_t *n) {
    float *scale = malloc(n->dim * sizeof(float));
    if (!scale) return 0;
    for (int i = 0; i < n->dim; i++) scale[i] = n->w[i] - 1.0f;
    n->dscale = dev_f32(scale, n->dim);
    n->dshift = dev_f32(n->b, n->dim);
    free(scale);
    return n->dscale && n->dshift;
}

static int dev_load_merger(vis_merger_t *m) {
    return dev_load_norm(&m->norm) && dev_load_linear(&m->fc1) && dev_load_linear(&m->fc2);
}

/* Upload the tower. Returns 0 if the device runs out of memory, so the CPU
 * path is used for this image. */
static int dev_load(qwen3_vision_t *v) {
    if (v->dev_loaded) return 1;
    size_t patch_n = (size_t)v->dim * VIS_CHANNELS * v->patch * v->patch;
    float *w = dev_f32(v->patch_w, patch_n);
    v->dev_patch_w = iris_dev_alloc(patch_n * sizeof(uint16_t));
    v->dev_patch_b = dev_f32(v->patch_b, v->dim);
    int ok = w && v->dev_patch_w && v->dev_patch_b;
    if (ok) iris_dev_f32_to_bf16(v->dev_patch_w, w, patch_n);
    iris_dev_sync();
    iris_dev_free(w);

    for (int l = 0; ok && l < v->depth; l++) {
        vis_block_t *b = &v->blocks[l];
        ok = dev_load_norm(&b->norm1) && dev_load_norm(&b->norm2) && dev_load_linear(&b->qkv) &&
             dev_load_linear(&b->proj) && dev_load_linear(&b->fc1) && dev_load_linear(&b->fc2);
    }
    ok = ok && dev_load_merger(&v->merger);
    for (int k = 0; ok && k < QWEN3_VISION_DEEPSTACK; k++) ok = dev_load_merger(&v->deep[k]);
    if (!ok) {
        fprintf(stderr, "Qwen3 vision: not enough GPU memory, using the CPU\n");
        dev_unload(v);
        return 0;
    }
    iris_dev_sync();
    v->dev_loaded = 1;
    return 1;
}

static int dev_work_alloc(const qwen3_vision_t *v, vis_dev_t *d, int n) {
    size_t rows = n, dim = v->dim, half = v->head_dim / 2;
    size_t row_len = (size_t)VIS_CHANNELS * v->patch * v->patch;
    size_t wide = 3 * dim > (size_t)v->mlp ? 3 * dim : (size_t)v->mlp;
    size_t floats = rows * (dim + wide + 2 * half + row_len);
    size_t halves = rows * (row_len + dim + 3 * dim + dim + v->mlp);
    float *base = iris_dev_alloc(floats * sizeof(float) + halves * sizeof(uint16_t));
    if (!base) return 0;
    d->x = base;
    d->out = d->x + rows * dim;
    d->cos_t = d->out + rows * wide;
    d->sin_t = d->cos_t + rows * half;
    d->px = d->sin_t + rows * half;
    d->px_b = (uint16_t *)(d->px + rows * row_len);
    d->norm = d->px_b + rows * row_len;
    d->qkv = d->norm + rows * dim;
    d->attn = d->qkv + rows * 3 * dim;
    d->hid = d->attn + rows * dim;
    return 1;
}

/* d->x = patch embedding, and the RoPE tables go to the device. */
static void dev_patch_embed(const qwen3_vision_t *v, vis_dev_t *d, const vis_input_t *in) {
    int n = in->n, dim = v->dim, row_len = VIS_CHANNELS * v->patch * v->patch;
    size_t half = v->head_dim / 2, px = (size_t)n * row_len;
    iris_dev_upload(d->px, in->px, px * sizeof(float));
    iris_dev_upload(d->cos_t, in->cos_t, n * half * sizeof(float));
    iris_dev_upload(d->sin_t, in->sin_t, n * half * sizeof(float));
    iris_dev_f32_to_bf16(d->px_b, d->px, px);
    iris_dev_linear_f32(d->x, d->px_b, v->dev_patch_w, n, row_len, dim);
    iris_dev_add_bias(d->x, v->dev_patch_b, n, dim);
}

static void dev_add_position(const qwen3_vision_t *v, vis_dev_t *d, const vis_input_t *in) {
    size_t n = (size_t)in->n * v->dim;
    iris_dev_upload(d->out, in->pos, n * sizeof(float));
    iris_dev_add(d->x, d->out, n);
}

/* d->out = x @ W^T + b for bf16 x. */
static void dev_linear(vis_dev_t *d, const uint16_t *x, const vis_linear_t *l, int rows) {
    iris_dev_linear_f32(d->out, x, l->dw, rows, l->in, l->out);
    iris_dev_add_bias(d->out, l->db, rows, l->out);
}

static void dev_block(const qwen3_vision_t *v, const vis_block_t *b, vis_dev_t *d, int n) {
    int dim = v->dim, ld = 3 * dim, heads = v->heads, hd = v->head_dim;
    iris_dev_layer_norm_mod(d->norm, d->x, b->norm1.dshift, b->norm1.dscale, n, dim, VIS_EPS);
    dev_linear(d, d->norm, &b->qkv, n);
    iris_dev_f32_to_bf16(d->qkv, d->out, (size_t)n * ld);
    iris_dev_rope(d->qkv, ld, d->cos_t, d->sin_t, n, heads, hd);
    iris_dev_rope(d->qkv + dim, ld, d->cos_t, d->sin_t, n, heads, hd);
    iris_dev_attention(d->attn, dim, d->qkv, d->qkv + dim, d->qkv + 2 * dim, ld, n, n,
                       heads, heads, hd, 1.0f / sqrtf((float)hd), 0, NULL);
    dev_linear(d, d->attn, &b->proj, n);
    iris_dev_residual(d->x, d->out, NULL, n, dim);

    iris_dev_layer_norm_mod(d->norm, d->x, b->norm2.dshift, b->norm2.dscale, n, dim, VIS_EPS);
    dev_linear(d, d->norm, &b->fc1, n);
    iris_dev_gelu_tanh(d->out, (size_t)n * v->mlp);
    iris_dev_f32_to_bf16(d->hid, d->out, (size_t)n * v->mlp);
    dev_linear(d, d->hid, &b->fc2, n);
    iris_dev_residual(d->x, d->out, NULL, n, dim);
}

/* Same as merger_forward(). Exact GELU stays on the host (four mergers). */
static int dev_merger(const qwen3_vision_t *v, const vis_merger_t *m, vis_dev_t *d,
                      float *out, int n) {
    int merged = n / (v->merge * v->merge);
    size_t hid = (size_t)merged * m->fc1.out;
    float *host = malloc(hid * sizeof(float));
    if (!host) return 0;
    iris_dev_layer_norm_mod(d->norm, d->x, m->norm.dshift, m->norm.dscale,
                            (int)((size_t)n * v->dim / m->norm.dim), m->norm.dim, VIS_EPS);
    dev_linear(d, d->norm, &m->fc1, merged);
    iris_dev_download(host, d->out, hid * sizeof(float));
    gelu_exact(host, hid);
    iris_dev_upload(d->out, host, hid * sizeof(float));
    free(host);
    iris_dev_f32_to_bf16(d->norm, d->out, hid);
    dev_linear(d, d->norm, &m->fc2, merged);
    iris_dev_download(out, d->out, (size_t)merged * m->fc2.out * sizeof(float));
    return 1;
}

static int dev_forward(const qwen3_vision_t *v, const vis_input_t *in, float *embeds,
                       float **deep) {
    vis_dev_t d;
    if (!dev_work_alloc(v, &d, in->n)) return 0;
    dev_patch_embed(v, &d, in);
    dev_add_position(v, &d, in);
    int ok = 1;
    for (int l = 0; ok && l < v->depth; l++) {
        int k;
        const vis_merger_t *m;
        dev_block(v, &v->blocks[l], &d, in->n);
        if ((m = deep_merger(v, l, &k))) ok = dev_merger(v, m, &d, deep[k], in->n);
    }
    ok = ok && dev_merger(v, &v->merger, &d, embeds, in->n);
    iris_dev_free(d.x);
    return ok;
}
#endif /* IRIS_DEV */

/* ========================================================================
 * Loading
 * ======================================================================== */

static const safetensor_t *find_tensor(const qwen3_vision_t *v, const char *name,
                                       safetensors_file_t **file) {
    for (int f = 0; f < v->num_sf; f++) {
        const safetensor_t *t = safetensors_find(v->sf[f], name);
        if (t) {
            *file = v->sf[f];
            return t;
        }
    }
    fprintf(stderr, "Qwen3 vision: missing tensor %s\n", name);
    return NULL;
}

/* The tensor as new f32 data, if it has n elements. */
static float *get_f32(const qwen3_vision_t *v, const char *prefix, const char *suffix, int64_t n) {
    char name[256];
    snprintf(name, sizeof(name), "%s.%s", prefix, suffix);
    safetensors_file_t *sf;
    const safetensor_t *t = find_tensor(v, name, &sf);
    if (!t) return NULL;
    if (safetensor_numel(t) != n) {
        fprintf(stderr, "Qwen3 vision: %s must have %lld elements\n", name, (long long)n);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

/* The mmapped bf16 tensor, if it has n elements. */
static const uint16_t *get_bf16(const qwen3_vision_t *v, const char *name, int64_t n) {
    safetensors_file_t *sf;
    const safetensor_t *t = find_tensor(v, name, &sf);
    if (!t) return NULL;
    if (safetensor_numel(t) != n || !safetensor_is_bf16(t)) {
        fprintf(stderr, "Qwen3 vision: %s must be bf16 with %lld elements\n", name, (long long)n);
        return NULL;
    }
    return safetensors_get_bf16_direct(sf, t);
}

static int load_linear(const qwen3_vision_t *v, vis_linear_t *l, const char *prefix,
                       int in, int out) {
    char name[256];
    snprintf(name, sizeof(name), "%s.weight", prefix);
    l->w = get_bf16(v, name, (int64_t)in * out);
    l->b = get_f32(v, prefix, "bias", out);
    l->in = in;
    l->out = out;
    return l->w && l->b;
}

static int load_norm(const qwen3_vision_t *v, vis_norm_t *n, const char *prefix, int dim) {
    n->w = get_f32(v, prefix, "weight", dim);
    n->b = get_f32(v, prefix, "bias", dim);
    n->dim = dim;
    return n->w && n->b;
}

static int load_block(const qwen3_vision_t *v, vis_block_t *b, int l) {
    char p[128];
    int dim = v->dim, ok;
    snprintf(p, sizeof(p), "model.visual.blocks.%d.norm1", l);
    ok = load_norm(v, &b->norm1, p, dim);
    snprintf(p, sizeof(p), "model.visual.blocks.%d.norm2", l);
    ok = load_norm(v, &b->norm2, p, dim) && ok;
    snprintf(p, sizeof(p), "model.visual.blocks.%d.attn.qkv", l);
    ok = load_linear(v, &b->qkv, p, dim, 3 * dim) && ok;
    snprintf(p, sizeof(p), "model.visual.blocks.%d.attn.proj", l);
    ok = load_linear(v, &b->proj, p, dim, dim) && ok;
    snprintf(p, sizeof(p), "model.visual.blocks.%d.mlp.linear_fc1", l);
    ok = load_linear(v, &b->fc1, p, dim, v->mlp) && ok;
    snprintf(p, sizeof(p), "model.visual.blocks.%d.mlp.linear_fc2", l);
    ok = load_linear(v, &b->fc2, p, v->mlp, dim) && ok;
    return ok;
}

static int load_merger(const qwen3_vision_t *v, vis_merger_t *m, const char *prefix,
                       int norm_dim) {
    char p[128];
    int merged = v->dim * v->merge * v->merge, ok;
    snprintf(p, sizeof(p), "%s.norm", prefix);
    ok = load_norm(v, &m->norm, p, norm_dim);
    snprintf(p, sizeof(p), "%s.linear_fc1", prefix);
    ok = load_linear(v, &m->fc1, p, merged, merged) && ok;
    snprintf(p, sizeof(p), "%s.linear_fc2", prefix);
    ok = load_linear(v, &m->fc2, p, merged, v->out_dim) && ok;
    return ok;
}

/* Sum the temporal slices of the Conv3d weight [dim, c, t, p, p]: the frames
 * of an image are identical. */
static int load_patch_embed(qwen3_vision_t *v) {
    const char *prefix = "model.visual.patch_embed.proj";
    int dim = v->dim, area = v->patch * v->patch, t = v->temporal;
    size_t row_len = (size_t)VIS_CHANNELS * area;
    float *w = get_f32(v, prefix, "weight", (int64_t)dim * VIS_CHANNELS * t * area);
    v->patch_b = get_f32(v, prefix, "bias", dim);
    v->patch_w = calloc(dim * row_len, sizeof(float));
    int ok = w && v->patch_b && v->patch_w;
    for (int o = 0; ok && o < dim; o++)
        for (int c = 0; c < VIS_CHANNELS; c++)
            for (int f = 0; f < t; f++) {
                const float *src = w + ((size_t)o * VIS_CHANNELS * t + c * t + f) * area;
                float *dst = v->patch_w + (o * (size_t)VIS_CHANNELS + c) * area;
                for (int i = 0; i < area; i++) dst[i] += src[i];
            }
    free(w);
    return ok;
}

/* A positive integer of the vision_config object. */
static int config_int(const char *vision, const char *key) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(vision, pat);
    p = p ? strchr(p, ':') : NULL;
    int value = p ? atoi(p + 1) : 0;
    if (value <= 0) fprintf(stderr, "Qwen3 vision: bad %s in config.json\n", key);
    return value;
}

static char *read_file(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    char *s = n >= 0 ? malloc((size_t)n + 1) : NULL;
    if (s && fread(s, 1, (size_t)n, f) == (size_t)n) {
        s[n] = '\0';
    } else {
        free(s);
        s = NULL;
    }
    fclose(f);
    return s;
}

static int load_config(qwen3_vision_t *v, const char *dir) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", dir);
    char *json = read_file(path);
    const char *c = json ? strstr(json, "\"vision_config\"") : NULL;
    if (!c) {
        fprintf(stderr, "Qwen3 vision: no vision_config in %s\n", path);
        free(json);
        return 0;
    }
    v->depth = config_int(c, "depth");
    v->dim = config_int(c, "hidden_size");
    v->heads = config_int(c, "num_heads");
    v->mlp = config_int(c, "intermediate_size");
    v->out_dim = config_int(c, "out_hidden_size");
    v->patch = config_int(c, "patch_size");
    v->merge = config_int(c, "spatial_merge_size");
    v->temporal = config_int(c, "temporal_patch_size");
    v->side = (int)lround(sqrt(config_int(c, "num_position_embeddings")));

    const char *p = strstr(c, "\"deepstack_visual_indexes\"");
    p = p ? strchr(p, '[') : NULL;
    int ok = p != NULL;
    for (int k = 0; ok && k < QWEN3_VISION_DEEPSTACK; k++) {
        char *end;
        v->deep_index[k] = (int)strtol(p + 1, &end, 10);
        ok = end != p + 1;
        p = strchr(end, k == QWEN3_VISION_DEEPSTACK - 1 ? ']' : ',');
        ok = ok && p;
    }
    free(json);
    if (!ok) fprintf(stderr, "Qwen3 vision: bad deepstack_visual_indexes\n");
    ok = ok && v->depth && v->dim && v->heads && v->mlp && v->out_dim && v->patch &&
         v->merge && v->temporal && v->side;
    if (ok) v->head_dim = v->dim / v->heads;
    return ok;
}

static void free_merger(vis_merger_t *m) {
    free(m->norm.w);
    free(m->norm.b);
    free(m->fc1.b);
    free(m->fc2.b);
}

void qwen3_vision_free(qwen3_vision_t *v) {
    if (!v) return;
#ifdef IRIS_DEV
    if (v->dev_loaded) dev_unload(v);
#endif
    for (int l = 0; v->blocks && l < v->depth; l++) {
        vis_block_t *b = &v->blocks[l];
        free(b->norm1.w);
        free(b->norm1.b);
        free(b->norm2.w);
        free(b->norm2.b);
        free(b->qkv.b);
        free(b->proj.b);
        free(b->fc1.b);
        free(b->fc2.b);
    }
    free(v->blocks);
    free_merger(&v->merger);
    for (int k = 0; k < QWEN3_VISION_DEEPSTACK; k++) free_merger(&v->deep[k]);
    free(v->patch_w);
    free(v->patch_b);
    free(v->pos);
    for (int f = 0; f < v->num_sf; f++) safetensors_close(v->sf[f]);
    free(v);
}

qwen3_vision_t *qwen3_vision_load(const char *text_encoder_dir) {
    qwen3_vision_t *v = calloc(1, sizeof(*v));
    if (!v) return NULL;
    v->num_sf = open_safetensors_shards(text_encoder_dir, v->sf, VIS_MAX_SHARDS);
    int ok = v->num_sf > 0 && load_config(v, text_encoder_dir);
    if (ok) {
        v->blocks = calloc(v->depth, sizeof(vis_block_t));
        ok = v->blocks != NULL;
    }
    ok = ok && load_patch_embed(v);
    if (ok) {
        int table = v->side * v->side, merged = v->dim * v->merge * v->merge;
        v->pos = get_f32(v, "model.visual.pos_embed", "weight", (int64_t)table * v->dim);
        ok = v->pos != NULL;
        for (int l = 0; ok && l < v->depth; l++) ok = load_block(v, &v->blocks[l], l);
        ok = ok && load_merger(v, &v->merger, "model.visual.merger", v->dim);
        for (int k = 0; ok && k < QWEN3_VISION_DEEPSTACK; k++) {
            char p[64];
            snprintf(p, sizeof(p), "model.visual.deepstack_merger_list.%d", k);
            ok = load_merger(v, &v->deep[k], p, merged);
        }
    }
    if (!ok) {
        qwen3_vision_free(v);
        return NULL;
    }
    return v;
}

int qwen3_vision_encode(qwen3_vision_t *v, const float *rgb, int h, int w,
                        float **embeds, float *deep[QWEN3_VISION_DEEPSTACK]) {
    int block = v->patch * v->merge;
    if (h <= 0 || w <= 0 || h % block || w % block) {
        fprintf(stderr, "Qwen3 vision: image size %dx%d is not a multiple of %d\n", w, h, block);
        return 0;
    }
    int tokens = (h / block) * (w / block);
    size_t bytes = (size_t)tokens * v->out_dim * sizeof(float);
    vis_input_t in;
    float *out[QWEN3_VISION_DEEPSTACK + 1] = {0};
    int ok = input_prepare(v, &in, rgb, h, w);
    if (!ok) return 0;
    for (int i = 0; i <= QWEN3_VISION_DEEPSTACK; i++) {
        out[i] = malloc(bytes);
        ok = ok && out[i];
    }
    if (ok) {
        int done = 0;
#ifdef IRIS_DEV
        if (iris_dev_available() && dev_load(v)) done = dev_forward(v, &in, out[0], out + 1);
#endif
        ok = done || cpu_forward(v, &in, out[0], out + 1);
    }
    input_free(&in);
    if (!ok) {
        for (int i = 0; i <= QWEN3_VISION_DEEPSTACK; i++) free(out[i]);
        return 0;
    }
    *embeds = out[0];
    memcpy(deep, out + 1, sizeof(out) - sizeof(out[0]));
    return tokens;
}
