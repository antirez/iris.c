/*
 * Qwen-Image 2.1 Transformer Implementation
 *
 * Implements QwenImage21Transformer2DModel (about 7B parameters):
 * - Single-stream blocks over the joint sequence: text rows with the
 *   reference images interleaved at their prompt slots, then the target
 *   image. Image tokens are the unpatched latent pixels (64 channels) in
 *   raster order.
 * - Block: x += tanh(gate1) * attn(LN(x) * (1 + scale1)), then
 *          x += tanh(gate2) * mlp(LN(x) * (1 + scale2)).
 *   LayerNorm has no affine part and the modulation no shift. The MLP is
 *   out(silu(gate_layer(n)) * proj(n)).
 * - One modulation projection shared by all blocks: [scale1, gate1, scale2,
 *   gate2]. The target is modulated from t, everything else from t = 0.
 * - Block-causal attention: the sequence is causal, except that an image
 *   block sees all of itself. Separate to_q/k/v, per-head RMSNorm on Q and K.
 * - 3-axis RoPE (frame, height, width; 16 + 56 + 56 dims, theta 10000),
 *   consecutive pairs. A position p starts at 0. Text sits at (p, p, p),
 *   p++; image token (r, c) at (p, r - (H - H/2), c - (W - W/2)), and
 *   p += max(H, W) after the block.
 * - Text input: zero-centered RMSNorm (weight + 1), linear, GELU-tanh,
 *   linear. Output: LN(x) * (1 + scale) (no shift), projection to 64.
 *
 * The prefix (text and references) never attends to the target and its
 * modulation does not depend on the step, so its keys and values are
 * computed once (the prefill) and every denoising step runs only the
 * target tokens against them. That is the reference pipeline's KV cache,
 * filled before the first step instead of during it.
 *
 * Weights stay as bf16 in the mmapped safetensors files. The CPU path
 * converts each matrix to f32 right before its matmul. With a device
 * backend (CUDA / ROCm / Metal, see iris_dev.h) the weights are uploaded
 * once and the blocks run on the GPU.
 */

#include "iris.h"
#include "iris_kernels.h"
#include "iris_safetensors.h"
#include "iris_qimage.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef IRIS_DEV
#include "iris_dev.h"
#endif

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#define QI_MAX_SHARDS   16
#define QI_ROPE_AXES    3
#define QI_T_CHANNELS   256   /* Sinusoidal timestep embedding size */
#define QI_ROPE_THETA   10000.0f
#define QI_ATTN_QCHUNK  256   /* Queries per attention chunk (CPU path) */

/* Called after every block with its output; debug/dump_qimage_refs.c
 * compares it with the reference. */
#ifndef QI_BLOCK_HOOK
#define QI_BLOCK_HOOK(tf, x, rows)
#endif

/* ========================================================================
 * Data Structures
 * ======================================================================== */

typedef struct {
    const uint16_t *to_q, *to_k, *to_v;  /* [dim, dim] */
    const uint16_t *to_out;              /* [dim, dim] */
    float *norm_q, *norm_k;              /* [head_dim] */
    const uint16_t *mlp_gate, *mlp_proj; /* [mlp, dim] */
    const uint16_t *mlp_out;             /* [dim, mlp] */
} qi_block_t;

struct iris_transformer_qimage {
    /* Architecture (from transformer/config.json) */
    int dim;                /* 4096 */
    int heads;              /* 32 */
    int head_dim;           /* 128 */
    int mlp;                /* 12288 (dim * mlp_ratio) */
    int num_layers;         /* 32 */
    int in_channels;        /* 64 */
    int out_channels;       /* 64 */
    int text_dim;           /* 4096 (context_in_dim) */
    int axes_dim[QI_ROPE_AXES]; /* 16, 56, 56 */
    float eps;              /* 1e-6 */

    /* Embedders */
    const uint16_t *t_lin1, *t_lin2;     /* timestep MLP: [dim, 256], [dim, dim] */
    float *txt_norm;                     /* [text_dim], effective weight (stored + 1) */
    const uint16_t *txt_in1, *txt_in2;   /* [dim, text_dim], [dim, dim] */
    const uint16_t *img_in;              /* [dim, in_channels] */
    const uint16_t *mod;                 /* [4*dim, dim]: scale1, gate1, scale2, gate2 */

    qi_block_t *blocks;                  /* [num_layers] */

    const uint16_t *norm_out;            /* [dim, dim]: output scale */
    const uint16_t *proj_out;            /* [out_channels, dim] */

    safetensors_file_t *sf[QI_MAX_SHARDS];
    int num_sf;

    /* Prefix from the last prefill */
    int prefix_seq;         /* Rows: text and references, 0 = none */
    int target_frame;       /* Frame position of the target */
    char *prefix_key;       /* The prefill's inputs, to recognize a repeat */
    size_t prefix_key_bytes;
    float *prefix_kv;       /* CPU: [num_layers][2][prefix_seq][dim], post-RoPE K, then V */

    /* Device copy of the model (GPU path), NULL when running on the CPU */
    struct qi_dev *dev;

    /* CPU work memory, grown on demand */
    float *wbuf;            /* f32 copy of the current weight matrix */
    size_t wbuf_elems;
    float *work;            /* activations, see ensure_work() */
    size_t work_elems;
};

/* ========================================================================
 * Basic Operations
 * ======================================================================== */

/* y[rows, out] = x[rows, in] @ W[out, in]^T with bf16 W. */
static void linear_bf16(iris_transformer_qimage_t *tf, float *y, const float *x,
                        const uint16_t *W, int rows, int in, int out) {
#ifdef USE_METAL
    /* W points into the mmapped weights, so it is a valid key for the
     * Metal weight cache (the f32 scratch below would not be). */
    (void)tf;
    iris_linear_nobias_bf16(y, x, W, rows, in, out);
#else
    size_t n = (size_t)in * out;
    if (n > tf->wbuf_elems) {
        free(tf->wbuf);
        tf->wbuf = malloc(n * sizeof(float));
        tf->wbuf_elems = tf->wbuf ? n : 0;
        if (!tf->wbuf) return;
    }
    for (size_t i = 0; i < n; i++) {
        uint32_t bits = (uint32_t)W[i] << 16;
        memcpy(tf->wbuf + i, &bits, sizeof(float));
    }
    iris_linear_nobias(y, x, tf->wbuf, rows, in, out);
#endif
}

/* out = LayerNorm(x) * (1 + scale), no affine parameters, no shift. */
static void layer_norm_scale(float *out, const float *x, const float *scale,
                             int rows, int dim, float eps) {
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
        float inv = (float)(1.0 / sqrt(var / dim + eps));
        float m = (float)mean;
        for (int i = 0; i < dim; i++) o[i] = (xr[i] - m) * inv * (1.0f + scale[i]);
    }
}

/* Per-head RMSNorm with weight, in place. x: [seq, heads * head_dim] */
static void head_rms_norm(float *x, const float *w, int seq, int heads,
                          int head_dim, float eps) {
    for (int s = 0; s < seq * heads; s++) {
        float *v = x + (size_t)s * head_dim;
        float ss = 0;
        for (int i = 0; i < head_dim; i++) ss += v[i] * v[i];
        float inv = 1.0f / sqrtf(ss / head_dim + eps);
        for (int i = 0; i < head_dim; i++) v[i] = v[i] * inv * w[i];
    }
}

/* Rotate consecutive pairs (x0, x1) of every head by the token's angles.
 * cos/sin: [seq, head_dim/2] */
static void apply_rope(float *x, const float *cos_t, const float *sin_t,
                       int seq, int heads, int head_dim) {
    int half = head_dim / 2;
    for (int s = 0; s < seq; s++) {
        const float *c = cos_t + (size_t)s * half;
        const float *sn = sin_t + (size_t)s * half;
        for (int h = 0; h < heads; h++) {
            float *v = x + ((size_t)s * heads + h) * head_dim;
            for (int i = 0; i < half; i++) {
                float x0 = v[2 * i], x1 = v[2 * i + 1];
                v[2 * i]     = x0 * c[i] - x1 * sn[i];
                v[2 * i + 1] = x1 * c[i] + x0 * sn[i];
            }
        }
    }
}

/* Attention of seq_q queries over seq_k keys, all [rows, heads * head_dim].
 * causal: 0 = all keys, n > 0 = query i sees keys j < i + n. Processed per
 * head in chunks of queries to bound the score buffer [QCHUNK, seq_k]. */
static void attention(float *out, const float *q, const float *k, const float *v,
                      float *scores, int seq_q, int seq_k, int heads, int head_dim,
                      int causal) {
    int dim = heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    for (int h = 0; h < heads; h++) {
        const float *qh = q + h * head_dim;
        const float *kh = k + h * head_dim;
        const float *vh = v + h * head_dim;
        float *oh = out + h * head_dim;
        for (int q0 = 0; q0 < seq_q; q0 += QI_ATTN_QCHUNK) {
            int nq = seq_q - q0 < QI_ATTN_QCHUNK ? seq_q - q0 : QI_ATTN_QCHUNK;
#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nq, seq_k, head_dim,
                        scale, qh + (size_t)q0 * dim, dim, kh, dim, 0.0f, scores, seq_k);
#else
            for (int i = 0; i < nq; i++) {
                const float *qi = qh + (size_t)(q0 + i) * dim;
                for (int j = 0; j < seq_k; j++) {
                    const float *kj = kh + (size_t)j * dim;
                    float dot = 0;
                    for (int d = 0; d < head_dim; d++) dot += qi[d] * kj[d];
                    scores[(size_t)i * seq_k + j] = dot * scale;
                }
            }
#endif
            if (causal)
                for (int i = 0; i < nq; i++)
                    for (int j = q0 + i + causal; j < seq_k; j++)
                        scores[(size_t)i * seq_k + j] = -1e30f;  /* not -INFINITY: -ffast-math */
            iris_softmax(scores, nq, seq_k);
#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, nq, head_dim, seq_k,
                        1.0f, scores, seq_k, vh, dim, 0.0f, oh + (size_t)q0 * dim, dim);
#else
            for (int i = 0; i < nq; i++) {
                float *oi = oh + (size_t)(q0 + i) * dim;
                memset(oi, 0, head_dim * sizeof(float));
                for (int j = 0; j < seq_k; j++) {
                    float p = scores[(size_t)i * seq_k + j];
                    const float *vj = vh + (size_t)j * dim;
                    for (int d = 0; d < head_dim; d++) oi[d] += p * vj[d];
                }
            }
#endif
        }
    }
}

/* x[r, c] += tanh(gate[c]) * a[r, c] */
static void gated_residual(float *x, const float *a, const float *gate, int rows, int dim) {
    float *tg = malloc(dim * sizeof(float));
    for (int i = 0; i < dim; i++) tg[i] = tanhf(gate[i]);
    for (int r = 0; r < rows; r++) {
        float *xr = x + (size_t)r * dim;
        const float *ar = a + (size_t)r * dim;
        for (int i = 0; i < dim; i++) xr[i] += tg[i] * ar[i];
    }
    free(tg);
}

/* ========================================================================
 * Timestep Embedding and Modulation
 * ======================================================================== */

static void sinusoidal(float *out, float sigma);

/* temb[dim] = lin2(silu(lin1(sinusoidal(1000 * sigma)))), with the
 * [cos | sin] halves of QwenImage21TemporalTimesteps. */
static void timestep_embed(iris_transformer_qimage_t *tf, float *temb, float sigma) {
    float sin_emb[QI_T_CHANNELS];
    sinusoidal(sin_emb, sigma);
    float *hidden = malloc(tf->dim * sizeof(float));
    linear_bf16(tf, hidden, sin_emb, tf->t_lin1, 1, QI_T_CHANNELS, tf->dim);
    iris_silu(hidden, tf->dim);
    linear_bf16(tf, temb, hidden, tf->t_lin2, 1, tf->dim, tf->dim);
    free(hidden);
}

/* Modulation for one timestep: mod[4*dim] = [scale1, gate1, scale2, gate2]
 * shared by every block, and (if fin is not NULL) the output layer scale
 * fin[dim]. Both read silu(temb). */
static void modulation(iris_transformer_qimage_t *tf, float *mod, float *fin, float sigma) {
    float *temb = malloc(tf->dim * sizeof(float));
    timestep_embed(tf, temb, sigma);
    iris_silu(temb, tf->dim);
    linear_bf16(tf, mod, temb, tf->mod, 1, tf->dim, 4 * tf->dim);
    if (fin) linear_bf16(tf, fin, temb, tf->norm_out, 1, tf->dim, tf->dim);
    free(temb);
}

/* Sinusoidal timestep embedding [QI_T_CHANNELS] of 1000 * sigma. */
static void sinusoidal(float *out, float sigma) {
    int half = QI_T_CHANNELS / 2;
    float t = 1000.0f * sigma;
    for (int i = 0; i < half; i++) {
        float freq = expf(-logf(10000.0f) * (float)i / (float)half);
        out[i] = cosf(t * freq);
        out[half + i] = sinf(t * freq);
    }
}

/* ========================================================================
 * RoPE Tables
 * ======================================================================== */

/* Fill cos/sin rows [seq, head_dim/2] for tokens at 3-axis positions
 * pos[seq][3]. Like the reference, the frequencies and angles are float32
 * (torch.pow / torch.outer), only the cos/sin are taken in double. */
static void rope_rows(const iris_transformer_qimage_t *tf, float *cos_t, float *sin_t,
                      const int (*pos)[QI_ROPE_AXES], int seq) {
    int half = tf->head_dim / 2;
    for (int s = 0; s < seq; s++) {
        int p = 0;
        for (int ax = 0; ax < QI_ROPE_AXES; ax++) {
            int d = tf->axes_dim[ax];
            for (int j = 0; j < d / 2; j++, p++) {
                float freq = 1.0f / powf(QI_ROPE_THETA, (float)(2 * j) / (float)d);
                float angle = (float)pos[s][ax] * freq;
                cos_t[(size_t)s * half + p] = (float)cos((double)angle);
                sin_t[(size_t)s * half + p] = (float)sin((double)angle);
            }
        }
    }
}

/* Positions of an h x w image: the frame axis stays at frame, rows and
 * columns are centred on zero. */
static void image_positions(int (*pos)[QI_ROPE_AXES], int frame, int h, int w) {
    for (int r = 0; r < h; r++) {
        for (int c = 0; c < w; c++) {
            int *p = pos[r * w + c];
            p[0] = frame;
            p[1] = r - (h - h / 2);
            p[2] = c - (w - w / 2);
        }
    }
}

/* RoPE tables of the target, an h x w image at frame. */
static void image_rope(const iris_transformer_qimage_t *tf, int frame, int h, int w,
                       float *cos_t, float *sin_t) {
    int (*pos)[QI_ROPE_AXES] = malloc((size_t)h * w * sizeof(*pos));
    image_positions(pos, frame, h, w);
    rope_rows(tf, cos_t, sin_t, (const int (*)[QI_ROPE_AXES])pos, h * w);
    free(pos);
}

/* ========================================================================
 * Prefix Layout
 * ======================================================================== */

/* Queries [start, start + rows) over the first keys keys, with causal as in
 * attention(). in: what the rows are made from, text rows [rows, text_dim]
 * (causal) or a latent [channels, rows] (not causal). */
typedef struct {
    int start, rows, keys, causal;
    const float *in;
} qi_run_t;

/* Lay the prefix out: at most 2n + 1 runs, where text sees everything
 * before it and is causal over itself, and an image block sees everything
 * before it and all of itself, and pos[rows][3], the positions of all its
 * rows. Returns the frame position of the target. */
static int prefix_layout(const iris_transformer_qimage_t *tf, qi_run_t *runs, int *nruns,
                         int (*pos)[QI_ROPE_AXES], const float *txt, int txt_seq,
                         const iris_qimage_ref_t *refs, int n) {
    int count = 0, row = 0, t = 0, p = 0;
    for (int i = 0; i <= n; i++) {
        int end = i < n ? refs[i].pos : txt_seq;
        if (end > t) {
            int rows = end - t;
            runs[count++] = (qi_run_t){row, rows, row + rows, row + 1,
                                       txt + (size_t)t * tf->text_dim};
            for (; t < end; t++, row++, p++) pos[row][0] = pos[row][1] = pos[row][2] = p;
        }
        if (i < n) {
            int h = refs[i].h, w = refs[i].w;
            runs[count++] = (qi_run_t){row, h * w, row + h * w, 0, refs[i].latent};
            image_positions(pos + row, p, h, w);
            row += h * w;
            p += h > w ? h : w;
        }
    }
    *nruns = count;
    return p;
}

/* Latent [channels, n] -> tokens [n, channels]. */
static void latent_tokens(float *tokens, const float *latent, int n, int channels) {
    for (int c = 0; c < channels; c++)
        for (int i = 0; i < n; i++)
            tokens[(size_t)i * channels + c] = latent[(size_t)c * n + i];
}

/* ========================================================================
 * Block Forward (CPU)
 * ======================================================================== */

/* Work memory for rows query tokens attending to kv_rows keys. */
typedef struct {
    float *norm, *q, *k, *v, *attn, *proj, *gate, *up, *scores, *rope_cos, *rope_sin;
} qi_work_t;

static int ensure_work(iris_transformer_qimage_t *tf, int rows, int kv_rows, qi_work_t *w) {
    size_t rd = (size_t)rows * tf->dim, kd = (size_t)kv_rows * tf->dim;
    size_t rm = (size_t)rows * tf->mlp, rh = (size_t)rows * (tf->head_dim / 2);
    size_t total = 4 * rd + 2 * kd + 2 * rm + (size_t)QI_ATTN_QCHUNK * kv_rows + 2 * rh;
    if (total > tf->work_elems) {
        free(tf->work);
        tf->work = malloc(total * sizeof(float));
        tf->work_elems = tf->work ? total : 0;
        if (!tf->work) return 0;
    }
    float *p = tf->work;
    w->norm = p;      p += rd;
    w->q = p;         p += rd;
    w->attn = p;      p += rd;
    w->proj = p;      p += rd;
    w->k = p;         p += kd;
    w->v = p;         p += kd;
    w->gate = p;      p += rm;
    w->up = p;        p += rm;
    w->rope_cos = p;  p += rh;
    w->rope_sin = p;  p += rh;
    w->scores = p;
    return 1;
}

/* The attention half of block b for x[rows, dim] (in place), up to the
 * keys: n = LN(x) * (1 + scale1), q/k/v projections, head norms, RoPE.
 * K and V of these rows go to w->k / w->v at row kv_off (the rows before
 * it hold the cached prefix). */
static void block_qkv(iris_transformer_qimage_t *tf, const qi_block_t *b, const float *x,
                      int rows, const float *mod, int kv_off, qi_work_t *w) {
    int dim = tf->dim;
    float *k = w->k + (size_t)kv_off * dim, *v = w->v + (size_t)kv_off * dim;
    layer_norm_scale(w->norm, x, mod, rows, dim, tf->eps);
    linear_bf16(tf, w->q, w->norm, b->to_q, rows, dim, dim);
    linear_bf16(tf, k, w->norm, b->to_k, rows, dim, dim);
    linear_bf16(tf, v, w->norm, b->to_v, rows, dim, dim);
    head_rms_norm(w->q, b->norm_q, rows, tf->heads, tf->head_dim, tf->eps);
    head_rms_norm(k, b->norm_k, rows, tf->heads, tf->head_dim, tf->eps);
    apply_rope(w->q, w->rope_cos, w->rope_sin, rows, tf->heads, tf->head_dim);
    apply_rope(k, w->rope_cos, w->rope_sin, rows, tf->heads, tf->head_dim);
}

/* The rest of block b, after block_qkv(): the attention runs over the keys
 * in w->k / w->v, gated residual, then the gated MLP. */
static void block_finish(iris_transformer_qimage_t *tf, const qi_block_t *b, float *x,
                         int rows, const qi_run_t *runs, int nruns, const float *mod,
                         qi_work_t *w) {
    int dim = tf->dim, mlp = tf->mlp;
    for (const qi_run_t *u = runs; u < runs + nruns; u++)
        attention(w->attn + (size_t)u->start * dim, w->q + (size_t)u->start * dim, w->k, w->v,
                  w->scores, u->rows, u->keys, tf->heads, tf->head_dim, u->causal);
    linear_bf16(tf, w->proj, w->attn, b->to_out, rows, dim, dim);
    gated_residual(x, w->proj, mod + dim, rows, dim);

    layer_norm_scale(w->norm, x, mod + 2 * dim, rows, dim, tf->eps);
    linear_bf16(tf, w->gate, w->norm, b->mlp_gate, rows, dim, mlp);
    linear_bf16(tf, w->up, w->norm, b->mlp_proj, rows, dim, mlp);
    for (size_t i = 0; i < (size_t)rows * mlp; i++)
        w->gate[i] = w->gate[i] / (1.0f + expf(-w->gate[i])) * w->up[i];
    linear_bf16(tf, w->proj, w->gate, b->mlp_out, rows, mlp, dim);
    gated_residual(x, w->proj, mod + 3 * dim, rows, dim);
    QI_BLOCK_HOOK(tf, x, rows);
}

/* n = RMSNorm(txt) * (stored weight + 1), the text input norm. */
static void text_norm(const iris_transformer_qimage_t *tf, float *n, const float *txt,
                      int txt_seq) {
    int td = tf->text_dim;
    for (int s = 0; s < txt_seq; s++) {
        const float *t = txt + (size_t)s * td;
        float *o = n + (size_t)s * td;
        double ss = 0;
        for (int i = 0; i < td; i++) ss += (double)t[i] * t[i];
        float inv = (float)(1.0 / sqrt(ss / td + tf->eps));
        for (int i = 0; i < td; i++) o[i] = t[i] * inv * tf->txt_norm[i];
    }
}

/* Text input projection: zero-centered RMSNorm, linear, GELU-tanh, linear. */
static void text_in(iris_transformer_qimage_t *tf, float *x, const float *txt, int txt_seq) {
    int dim = tf->dim, td = tf->text_dim;
    float *n = malloc((size_t)txt_seq * td * sizeof(float));
    float *hid = malloc((size_t)txt_seq * dim * sizeof(float));
    text_norm(tf, n, txt, txt_seq);
    linear_bf16(tf, hid, n, tf->txt_in1, txt_seq, td, dim);
    iris_gelu_tanh(hid, (size_t)txt_seq * dim);
    linear_bf16(tf, x, hid, tf->txt_in2, txt_seq, dim, dim);
    free(hid);
    free(n);
}

#ifdef IRIS_DEV
/* ========================================================================
 * GPU Path (CUDA / ROCm / Metal)
 * ======================================================================== */

/* Device copy of a block, with the projections fused into one GEMM each:
 * w_qkv = [to_q; to_k; to_v] and w_mlp_in = [gate_layer; proj]. */
typedef struct {
    uint16_t *w_qkv;         /* [3*dim, dim] */
    uint16_t *w_out;         /* [dim, dim] */
    uint16_t *w_mlp_in;      /* [2*mlp, dim] */
    uint16_t *w_mlp_out;     /* [dim, mlp] */
    float *norm_q, *norm_k;  /* [head_dim] */
} qi_dev_block_t;

struct qi_dev {
    qi_dev_block_t *blocks;
    uint16_t *t_lin1, *t_lin2, *mod, *norm_out, *proj_out, *img_in, *txt_in1, *txt_in2;

    /* Prefix: per block the post-RoPE K | V rows */
    uint16_t **prefix;       /* [num_layers], each [prefix_seq, 2*dim] */

    /* Activations for rows query tokens and kv_rows key tokens */
    int rows, kv_rows;
    float *x;                /* [rows, dim] residual stream */
    uint16_t *norm;          /* [rows, dim] modulated LayerNorm output */
    uint16_t *qkv;           /* [kv_rows, 3*dim]: cached prefix rows, then these rows */
    uint16_t *attn;          /* [rows, dim] attention output */
    float *out;              /* [rows, dim] projection output */
    uint16_t *mlp_h;         /* [rows, 2*mlp] gate, up */
    uint16_t *mlp_act;       /* [rows, mlp] */
    float *rope_cos, *rope_sin;  /* [rows, head_dim/2] */

    /* Model inputs (f32 upload + bf16 copy), grown on demand */
    size_t in_elems;
    float *in_f32;
    uint16_t *in_bf16;

    /* Timestep embedding and modulation, see vec_layout() */
    float *vec;
    uint16_t *vec_bf16;
};

/* Offsets (in floats) inside dev->vec. mod holds scale1, tanh(gate1),
 * scale2, tanh(gate2); fin the output layer scale. */
typedef struct {
    size_t sin, hidden, temb, mod, fin, total;
} qi_vec_layout_t;

static qi_vec_layout_t vec_layout(const iris_transformer_qimage_t *tf) {
    qi_vec_layout_t v;
    size_t dim = tf->dim;
    v.sin = 0;
    v.hidden = v.sin + QI_T_CHANNELS;
    v.temb = v.hidden + dim;
    v.mod = v.temb + dim;
    v.fin = v.mod + 4 * dim;
    v.total = v.fin + dim;
    return v;
}

static uint16_t *dev_bf16(const uint16_t *src, size_t n) {
    uint16_t *d = iris_dev_alloc(n * sizeof(uint16_t));
    if (d) iris_dev_upload(d, src, n * sizeof(uint16_t));
    return d;
}

static float *dev_f32(const float *src, size_t n) {
    float *d = iris_dev_alloc(n * sizeof(float));
    if (d) iris_dev_upload(d, src, n * sizeof(float));
    return d;
}

/* Upload the parts of a matrix (contiguous rows each) one after another. */
static uint16_t *dev_stack(const uint16_t *const *parts, const size_t *elems, int nparts) {
    size_t total = 0;
    for (int i = 0; i < nparts; i++) total += elems[i];
    uint16_t *d = iris_dev_alloc(total * sizeof(uint16_t));
    if (!d) return NULL;
    size_t off = 0;
    for (int i = 0; i < nparts; i++) {
        iris_dev_upload(d + off, parts[i], elems[i] * sizeof(uint16_t));
        off += elems[i];
    }
    return d;
}

static void dev_free_activations(struct qi_dev *d) {
    iris_dev_free(d->x);
    iris_dev_free(d->norm);
    iris_dev_free(d->qkv);
    iris_dev_free(d->attn);
    iris_dev_free(d->out);
    iris_dev_free(d->mlp_h);
    iris_dev_free(d->mlp_act);
    iris_dev_free(d->rope_cos);
    iris_dev_free(d->rope_sin);
    d->x = d->out = d->rope_cos = d->rope_sin = NULL;
    d->norm = d->qkv = d->attn = d->mlp_h = d->mlp_act = NULL;
    d->rows = d->kv_rows = 0;
}

static void dev_free_prefix(iris_transformer_qimage_t *tf) {
    for (int l = 0; l < tf->num_layers; l++) {
        iris_dev_free(tf->dev->prefix[l]);
        tf->dev->prefix[l] = NULL;
    }
}

static void dev_free(iris_transformer_qimage_t *tf) {
    struct qi_dev *d = tf->dev;
    if (!d) return;
    if (d->blocks) {
        for (int l = 0; l < tf->num_layers; l++) {
            qi_dev_block_t *b = &d->blocks[l];
            iris_dev_free(b->w_qkv);
            iris_dev_free(b->w_out);
            iris_dev_free(b->w_mlp_in);
            iris_dev_free(b->w_mlp_out);
            iris_dev_free(b->norm_q);
            iris_dev_free(b->norm_k);
        }
        free(d->blocks);
    }
    iris_dev_free(d->t_lin1);
    iris_dev_free(d->t_lin2);
    iris_dev_free(d->mod);
    iris_dev_free(d->norm_out);
    iris_dev_free(d->proj_out);
    iris_dev_free(d->img_in);
    iris_dev_free(d->txt_in1);
    iris_dev_free(d->txt_in2);
    if (d->prefix) {
        dev_free_prefix(tf);
        free(d->prefix);
    }
    dev_free_activations(d);
    iris_dev_free(d->in_f32);
    iris_dev_free(d->in_bf16);
    iris_dev_free(d->vec);
    iris_dev_free(d->vec_bf16);
    free(d);
    tf->dev = NULL;
}

/* Upload the model to the device. Returns 0 (leaving tf->dev NULL) if the
 * device runs out of memory, so the CPU path is used instead. */
static int dev_load(iris_transformer_qimage_t *tf) {
    struct qi_dev *d = calloc(1, sizeof(*d));
    if (!d) return 0;
    tf->dev = d;
    size_t dim = tf->dim, mlp = tf->mlp, dd = dim * dim;
    int ok = 1;

    d->t_lin1 = dev_bf16(tf->t_lin1, dim * QI_T_CHANNELS);
    d->t_lin2 = dev_bf16(tf->t_lin2, dd);
    d->mod = dev_bf16(tf->mod, 4 * dd);
    d->norm_out = dev_bf16(tf->norm_out, dd);
    d->proj_out = dev_bf16(tf->proj_out, (size_t)tf->out_channels * dim);
    d->img_in = dev_bf16(tf->img_in, dim * tf->in_channels);
    d->txt_in1 = dev_bf16(tf->txt_in1, dim * tf->text_dim);
    d->txt_in2 = dev_bf16(tf->txt_in2, dd);
    ok = d->t_lin1 && d->t_lin2 && d->mod && d->norm_out && d->proj_out && d->img_in &&
         d->txt_in1 && d->txt_in2;

    d->blocks = ok ? calloc(tf->num_layers, sizeof(qi_dev_block_t)) : NULL;
    d->prefix = ok ? calloc(tf->num_layers, sizeof(uint16_t *)) : NULL;
    ok = ok && d->blocks && d->prefix;
    for (int l = 0; ok && l < tf->num_layers; l++) {
        const qi_block_t *b = &tf->blocks[l];
        qi_dev_block_t *db = &d->blocks[l];
        const uint16_t *qkv[3] = {b->to_q, b->to_k, b->to_v};
        const size_t qkv_n[3] = {dd, dd, dd};
        const uint16_t *mlp_in[2] = {b->mlp_gate, b->mlp_proj};
        const size_t mlp_n[2] = {mlp * dim, mlp * dim};
        db->w_qkv = dev_stack(qkv, qkv_n, 3);
        db->w_out = dev_bf16(b->to_out, dd);
        db->w_mlp_in = dev_stack(mlp_in, mlp_n, 2);
        db->w_mlp_out = dev_bf16(b->mlp_out, dim * mlp);
        db->norm_q = dev_f32(b->norm_q, tf->head_dim);
        db->norm_k = dev_f32(b->norm_k, tf->head_dim);
        ok = db->w_qkv && db->w_out && db->w_mlp_in && db->w_mlp_out && db->norm_q && db->norm_k;
    }

    qi_vec_layout_t vl = vec_layout(tf);
    d->vec = ok ? iris_dev_alloc(vl.total * sizeof(float)) : NULL;
    d->vec_bf16 = ok ? iris_dev_alloc(vl.total * sizeof(uint16_t)) : NULL;
    ok = ok && d->vec && d->vec_bf16;

    if (!ok) {
        fprintf(stderr, "Qwen-Image: not enough GPU memory, using the CPU\n");
        dev_free(tf);
        return 0;
    }
    iris_dev_sync();
    return 1;
}

static int dev_ensure(iris_transformer_qimage_t *tf, int rows, int kv_rows, size_t in_elems) {
    struct qi_dev *d = tf->dev;
    size_t dim = tf->dim, mlp = tf->mlp, half = tf->head_dim / 2;
    if (rows > d->rows || kv_rows > d->kv_rows) {
        if (kv_rows < d->kv_rows) kv_rows = d->kv_rows;
        if (rows < d->rows) rows = d->rows;
        dev_free_activations(d);
        d->x = iris_dev_alloc((size_t)rows * dim * sizeof(float));
        d->norm = iris_dev_alloc((size_t)rows * dim * sizeof(uint16_t));
        d->qkv = iris_dev_alloc((size_t)kv_rows * 3 * dim * sizeof(uint16_t));
        d->attn = iris_dev_alloc((size_t)rows * dim * sizeof(uint16_t));
        d->out = iris_dev_alloc((size_t)rows * dim * sizeof(float));
        d->mlp_h = iris_dev_alloc((size_t)rows * 2 * mlp * sizeof(uint16_t));
        d->mlp_act = iris_dev_alloc((size_t)rows * mlp * sizeof(uint16_t));
        d->rope_cos = iris_dev_alloc((size_t)rows * half * sizeof(float));
        d->rope_sin = iris_dev_alloc((size_t)rows * half * sizeof(float));
        if (!d->x || !d->norm || !d->qkv || !d->attn || !d->out || !d->mlp_h ||
            !d->mlp_act || !d->rope_cos || !d->rope_sin) {
            dev_free_activations(d);
            return 0;
        }
        d->rows = rows;
        d->kv_rows = kv_rows;
    }
    if (in_elems > d->in_elems) {
        iris_dev_free(d->in_f32);
        iris_dev_free(d->in_bf16);
        d->in_f32 = iris_dev_alloc(in_elems * sizeof(float));
        d->in_bf16 = iris_dev_alloc(in_elems * sizeof(uint16_t));
        d->in_elems = (d->in_f32 && d->in_bf16) ? in_elems : 0;
        if (!d->in_elems) return 0;
    }
    return 1;
}

/* y[rows, out] = in[rows, in_dim] (f32 on the host) @ W^T on the device. */
static void dev_input_linear(iris_transformer_qimage_t *tf, float *y, const float *in,
                             int rows, int in_dim, const uint16_t *W, int out) {
    struct qi_dev *d = tf->dev;
    size_t n = (size_t)rows * in_dim;
    iris_dev_upload(d->in_f32, in, n * sizeof(float));
    iris_dev_f32_to_bf16(d->in_bf16, d->in_f32, n);
    iris_dev_linear_f32(y, d->in_bf16, W, rows, in_dim, out);
}

/* Modulation for sigma into dev->vec (gates through tanh) and, if fin is
 * set, the output layer scale. */
static void dev_modulation(iris_transformer_qimage_t *tf, float sigma, int fin) {
    struct qi_dev *d = tf->dev;
    qi_vec_layout_t vl = vec_layout(tf);
    int dim = tf->dim;
    float sin_host[QI_T_CHANNELS];
    sinusoidal(sin_host, sigma);
    iris_dev_upload(d->vec + vl.sin, sin_host, sizeof(sin_host));

    float *hidden = d->vec + vl.hidden, *temb = d->vec + vl.temb, *mod = d->vec + vl.mod;
    uint16_t *sin_b = d->vec_bf16 + vl.sin, *hidden_b = d->vec_bf16 + vl.hidden;
    uint16_t *temb_b = d->vec_bf16 + vl.temb;
    iris_dev_f32_to_bf16(sin_b, d->vec + vl.sin, QI_T_CHANNELS);
    iris_dev_linear_f32(hidden, sin_b, d->t_lin1, 1, QI_T_CHANNELS, dim);
    iris_dev_silu(hidden, dim);
    iris_dev_f32_to_bf16(hidden_b, hidden, dim);
    iris_dev_linear_f32(temb, hidden_b, d->t_lin2, 1, dim, dim);
    iris_dev_silu(temb, dim);
    iris_dev_f32_to_bf16(temb_b, temb, dim);
    iris_dev_linear_f32(mod, temb_b, d->mod, 1, dim, 4 * dim);
    if (fin) iris_dev_linear_f32(d->vec + vl.fin, temb_b, d->norm_out, 1, dim, dim);

    /* The residual kernel takes the gate as is: apply tanh on the host. */
    float *m = malloc(4 * dim * sizeof(float));
    iris_dev_download(m, mod, 4 * dim * sizeof(float));
    for (int i = 0; i < dim; i++) {
        m[dim + i] = tanhf(m[dim + i]);
        m[3 * dim + i] = tanhf(m[3 * dim + i]);
    }
    iris_dev_upload(mod, m, 4 * dim * sizeof(float));
    free(m);
}

/* Attention half of block l up to the keys, for the rows of dev->x: their
 * q, k, v go to rows [kv_off, kv_off + rows) of dev->qkv. */
static void dev_block_qkv(iris_transformer_qimage_t *tf, const qi_dev_block_t *b,
                          int rows, int kv_off) {
    struct qi_dev *d = tf->dev;
    int dim = tf->dim, ld = 3 * dim;
    const float *mod = d->vec + vec_layout(tf).mod;
    uint16_t *h = d->qkv + (size_t)kv_off * ld;
    iris_dev_layer_norm_mod(d->norm, d->x, NULL, mod, rows, dim, tf->eps);
    iris_dev_linear_bf16(h, d->norm, b->w_qkv, rows, dim, ld);
    iris_dev_head_norm_rope(h, ld, b->norm_q, d->rope_cos, d->rope_sin, rows, tf->heads,
                            tf->head_dim, tf->eps, 0);
    iris_dev_head_norm_rope(h + dim, ld, b->norm_k, d->rope_cos, d->rope_sin, rows, tf->heads,
                            tf->head_dim, tf->eps, 0);
}

/* Rest of the block: the attention runs (with the queries from row kv_off
 * of dev->qkv), then the gated residual and the gated MLP. */
static void dev_block_finish(iris_transformer_qimage_t *tf, const qi_dev_block_t *b,
                             int rows, int kv_off, const qi_run_t *runs, int nruns) {
    struct qi_dev *d = tf->dev;
    int dim = tf->dim, mlp = tf->mlp, ld = 3 * dim, hd = tf->head_dim;
    const float *mod = d->vec + vec_layout(tf).mod;
    for (const qi_run_t *u = runs; u < runs + nruns; u++)
        iris_dev_attention(d->attn + (size_t)u->start * dim, dim,
                           d->qkv + (size_t)(kv_off + u->start) * ld, d->qkv + dim,
                           d->qkv + 2 * dim, ld, u->rows, u->keys, tf->heads, tf->heads, hd,
                           1.0f / sqrtf((float)hd), u->causal, NULL);
    iris_dev_linear_f32(d->out, d->attn, b->w_out, rows, dim, dim);
    iris_dev_residual(d->x, d->out, mod + dim, rows, dim);

    iris_dev_layer_norm_mod(d->norm, d->x, NULL, mod + 2 * dim, rows, dim, tf->eps);
    iris_dev_linear_bf16(d->mlp_h, d->norm, b->w_mlp_in, rows, dim, 2 * mlp);
    iris_dev_swiglu(d->mlp_act, mlp, d->mlp_h, d->mlp_h + mlp, 2 * mlp, rows, mlp);
    iris_dev_linear_f32(d->out, d->mlp_act, b->w_mlp_out, rows, mlp, dim);
    iris_dev_residual(d->x, d->out, mod + 3 * dim, rows, dim);
    QI_BLOCK_HOOK(tf, d->x, rows);
}

/* Text input into y: norm and GELU on the host (once per prompt), the two
 * projections on the device. */
static void dev_text_in(iris_transformer_qimage_t *tf, float *y, const float *txt, int rows) {
    struct qi_dev *d = tf->dev;
    int dim = tf->dim;
    float *host = malloc((size_t)rows * (tf->text_dim > dim ? tf->text_dim : dim) * sizeof(float));
    text_norm(tf, host, txt, rows);
    dev_input_linear(tf, d->out, host, rows, tf->text_dim, d->txt_in1, dim);
    iris_dev_download(host, d->out, (size_t)rows * dim * sizeof(float));
    iris_gelu_tanh(host, (size_t)rows * dim);
    dev_input_linear(tf, y, host, rows, dim, d->txt_in2, dim);
    free(host);
}

static int dev_prefill(iris_transformer_qimage_t *tf, const qi_run_t *runs, int nruns, int L,
                       const float *rope_cos, const float *rope_sin) {
    struct qi_dev *d = tf->dev;
    int dim = tf->dim, half = tf->head_dim / 2, ch = tf->in_channels;
    size_t in_elems = (size_t)L * (tf->text_dim > dim ? tf->text_dim : dim);
    dev_free_prefix(tf);
    int ok = dev_ensure(tf, L, L, in_elems);
    for (int l = 0; ok && l < tf->num_layers; l++) {
        d->prefix[l] = iris_dev_alloc((size_t)L * 2 * dim * sizeof(uint16_t));
        ok = d->prefix[l] != NULL;
    }
    if (!ok) {
        fprintf(stderr, "Qwen-Image: GPU allocation failed for %d prefix tokens\n", L);
        return 0;
    }

    dev_modulation(tf, 0.0f, 0);

    float *tokens = malloc((size_t)L * ch * sizeof(float));
    for (const qi_run_t *u = runs; u < runs + nruns; u++) {
        float *y = d->x + (size_t)u->start * dim;
        if (u->causal) {
            dev_text_in(tf, y, u->in, u->rows);
        } else {
            latent_tokens(tokens, u->in, u->rows, ch);
            dev_input_linear(tf, y, tokens, u->rows, ch, d->img_in, dim);
        }
    }
    free(tokens);

    iris_dev_upload(d->rope_cos, rope_cos, (size_t)L * half * sizeof(float));
    iris_dev_upload(d->rope_sin, rope_sin, (size_t)L * half * sizeof(float));

    for (int l = 0; l < tf->num_layers; l++) {
        const qi_dev_block_t *b = &d->blocks[l];
        dev_block_qkv(tf, b, L, 0);
        iris_dev_copy_2d(d->prefix[l], 2 * dim, d->qkv + dim, 3 * dim, L, 2 * dim);
        if (l + 1 < tf->num_layers) dev_block_finish(tf, b, L, 0, runs, nruns);
    }
    iris_dev_sync();
    dev_free_activations(d);
    return 1;
}

static float *dev_forward(iris_transformer_qimage_t *tf, const float *x_in, int h, int w,
                          float sigma) {
    struct qi_dev *d = tf->dev;
    int dim = tf->dim, half = tf->head_dim / 2, L = tf->prefix_seq, n = h * w;
    int ch = tf->in_channels, och = tf->out_channels;
    qi_run_t run = {0, n, L + n, 0, NULL};
    if (!dev_ensure(tf, n, L + n, (size_t)n * ch)) {
        fprintf(stderr, "Qwen-Image: GPU allocation failed for %d tokens\n", n);
        return NULL;
    }

    dev_modulation(tf, sigma, 1);

    float *tokens = malloc((size_t)n * (ch > och ? ch : och) * sizeof(float));
    latent_tokens(tokens, x_in, n, ch);
    dev_input_linear(tf, d->x, tokens, n, ch, d->img_in, dim);

    float *rc = malloc((size_t)n * half * sizeof(float));
    float *rs = malloc((size_t)n * half * sizeof(float));
    image_rope(tf, tf->target_frame, h, w, rc, rs);
    iris_dev_upload(d->rope_cos, rc, (size_t)n * half * sizeof(float));
    iris_dev_upload(d->rope_sin, rs, (size_t)n * half * sizeof(float));
    free(rc);
    free(rs);

    for (int l = 0; l < tf->num_layers; l++) {
        const qi_dev_block_t *b = &d->blocks[l];
        iris_dev_copy_2d(d->qkv + dim, 3 * dim, d->prefix[l], 2 * dim, L, 2 * dim);
        dev_block_qkv(tf, b, n, L);
        dev_block_finish(tf, b, n, L, &run, 1);
    }

    iris_dev_layer_norm_mod(d->norm, d->x, NULL, d->vec + vec_layout(tf).fin, n, dim, tf->eps);
    iris_dev_linear_f32(d->out, d->norm, d->proj_out, n, dim, och);
    iris_dev_download(tokens, d->out, (size_t)n * och * sizeof(float));

    float *out = malloc((size_t)och * n * sizeof(float));
    for (int i = 0; i < n; i++)
        for (int c = 0; c < och; c++)
            out[(size_t)c * n + i] = tokens[(size_t)i * och + c];
    free(tokens);
    return out;
}
#endif /* IRIS_DEV */

/* ========================================================================
 * Prefill and Forward
 * ======================================================================== */

static int cpu_prefill(iris_transformer_qimage_t *tf, const qi_run_t *runs, int nruns, int L,
                       float *rope_cos, float *rope_sin) {
    int dim = tf->dim, ch = tf->in_channels;
    size_t ld = (size_t)L * dim;
    free(tf->prefix_kv);
    tf->prefix_kv = malloc((size_t)tf->num_layers * 2 * ld * sizeof(float));
    float *x = malloc(ld * sizeof(float));
    float *mod = malloc(4 * dim * sizeof(float));
    float *tokens = malloc((size_t)L * ch * sizeof(float));
    qi_work_t w;
    if (!tf->prefix_kv || !x || !mod || !tokens || !ensure_work(tf, L, L, &w)) {
        free(x);
        free(mod);
        free(tokens);
        return 0;
    }
    w.rope_cos = rope_cos;
    w.rope_sin = rope_sin;

    modulation(tf, mod, NULL, 0.0f);
    for (const qi_run_t *u = runs; u < runs + nruns; u++) {
        float *y = x + (size_t)u->start * dim;
        if (u->causal) {
            text_in(tf, y, u->in, u->rows);
        } else {
            latent_tokens(tokens, u->in, u->rows, ch);
            linear_bf16(tf, y, tokens, tf->img_in, u->rows, ch, dim);
        }
    }

    for (int l = 0; l < tf->num_layers; l++) {
        const qi_block_t *b = &tf->blocks[l];
        block_qkv(tf, b, x, L, mod, 0, &w);
        float *kv = tf->prefix_kv + (size_t)l * 2 * ld;
        memcpy(kv, w.k, ld * sizeof(float));
        memcpy(kv + ld, w.v, ld * sizeof(float));
        /* The last block's prefix output is never read. */
        if (l + 1 < tf->num_layers) block_finish(tf, b, x, L, runs, nruns, mod, &w);
    }

    free(x);
    free(mod);
    free(tokens);
    return 1;
}

int iris_transformer_prefill_qimage(iris_transformer_qimage_t *tf,
                                    const float *txt, int txt_seq,
                                    const iris_qimage_ref_t *refs, int n) {
    if (txt_seq <= 0) return 0;

    /* All the inputs, to recognize a repeat. */
    size_t txt_bytes = (size_t)txt_seq * tf->text_dim * sizeof(float);
    size_t key_bytes = txt_bytes;
    int L = txt_seq, half = tf->head_dim / 2;
    for (int i = 0; i < n; i++) {
        size_t tokens = (size_t)refs[i].h * refs[i].w;
        L += (int)tokens;
        key_bytes += 3 * sizeof(int) + tokens * tf->in_channels * sizeof(float);
    }
    char *key = malloc(key_bytes);
    if (!key) return 0;
    memcpy(key, txt, txt_bytes);
    char *kp = key + txt_bytes;
    for (int i = 0; i < n; i++) {
        size_t bytes = (size_t)refs[i].h * refs[i].w * tf->in_channels * sizeof(float);
        int shape[3] = {refs[i].pos, refs[i].h, refs[i].w};
        memcpy(kp, shape, sizeof(shape));
        memcpy(kp + sizeof(shape), refs[i].latent, bytes);
        kp += sizeof(shape) + bytes;
    }
    if (tf->prefix_seq && tf->prefix_key_bytes == key_bytes &&
        !memcmp(tf->prefix_key, key, key_bytes)) {
        free(key);
        return 1;
    }
    free(tf->prefix_key);
    tf->prefix_key = key;
    tf->prefix_key_bytes = key_bytes;
    tf->prefix_seq = 0;

    qi_run_t *runs = malloc((size_t)(2 * n + 1) * sizeof(qi_run_t));
    int (*pos)[QI_ROPE_AXES] = malloc((size_t)L * sizeof(*pos));
    float *rc = malloc((size_t)L * half * sizeof(float));
    float *rs = malloc((size_t)L * half * sizeof(float));
    int ok = runs && pos && rc && rs;
    if (ok) {
        int nruns;
        int frame = prefix_layout(tf, runs, &nruns, pos, txt, txt_seq, refs, n);
        rope_rows(tf, rc, rs, (const int (*)[QI_ROPE_AXES])pos, L);
#ifdef IRIS_DEV
        if (tf->dev) ok = dev_prefill(tf, runs, nruns, L, rc, rs);
        else
#endif
        ok = cpu_prefill(tf, runs, nruns, L, rc, rs);
        if (ok) {
            tf->prefix_seq = L;
            tf->target_frame = frame;
        }
    }
    free(runs);
    free(pos);
    free(rc);
    free(rs);
    return ok;
}

float *iris_transformer_forward_qimage(iris_transformer_qimage_t *tf,
                                       const float *x_in, int h, int w_lat, float sigma) {
    if (tf->prefix_seq <= 0) {
        fprintf(stderr, "Qwen-Image: forward without a prefix\n");
        return NULL;
    }
#ifdef IRIS_DEV
    if (tf->dev) return dev_forward(tf, x_in, h, w_lat, sigma);
#endif
    int dim = tf->dim, L = tf->prefix_seq, n = h * w_lat;
    size_t ld = (size_t)L * dim;
    qi_run_t run = {0, n, L + n, 0, NULL};

    qi_work_t w;
    if (!ensure_work(tf, n, L + n, &w)) return NULL;

    float *mod = malloc(4 * dim * sizeof(float));
    float *fin = malloc(dim * sizeof(float));
    float *x = malloc((size_t)n * dim * sizeof(float));
    int ch = tf->in_channels > tf->out_channels ? tf->in_channels : tf->out_channels;
    float *tokens = malloc((size_t)n * ch * sizeof(float));
    modulation(tf, mod, fin, sigma);

    latent_tokens(tokens, x_in, n, tf->in_channels);
    linear_bf16(tf, x, tokens, tf->img_in, n, tf->in_channels, dim);
    image_rope(tf, tf->target_frame, h, w_lat, w.rope_cos, w.rope_sin);

    for (int l = 0; l < tf->num_layers; l++) {
        const qi_block_t *b = &tf->blocks[l];
        const float *kv = tf->prefix_kv + (size_t)l * 2 * ld;
        memcpy(w.k, kv, ld * sizeof(float));
        memcpy(w.v, kv + ld, ld * sizeof(float));
        block_qkv(tf, b, x, n, mod, L, &w);
        block_finish(tf, b, x, n, &run, 1, mod, &w);
    }

    /* Output layer: LN(x) * (1 + scale), projection, back to [ch, n]. */
    layer_norm_scale(w.norm, x, fin, n, dim, tf->eps);
    linear_bf16(tf, tokens, w.norm, tf->proj_out, n, dim, tf->out_channels);
    float *out = malloc((size_t)tf->out_channels * n * sizeof(float));
    for (int i = 0; i < n; i++)
        for (int c = 0; c < tf->out_channels; c++)
            out[(size_t)c * n + i] = tokens[(size_t)i * tf->out_channels + c];

    free(tokens);
    free(x);
    free(fin);
    free(mod);
    return out;
}

int iris_transformer_channels_qimage(const iris_transformer_qimage_t *tf) {
    return tf->out_channels;
}

/* ========================================================================
 * Loading
 * ======================================================================== */

static const safetensor_t *find_tensor(iris_transformer_qimage_t *tf, const char *name,
                                       safetensors_file_t **file) {
    for (int f = 0; f < tf->num_sf; f++) {
        const safetensor_t *t = safetensors_find(tf->sf[f], name);
        if (t) {
            *file = tf->sf[f];
            return t;
        }
    }
    return NULL;
}

/* Zero-copy pointer to a bf16 weight inside the mmapped shard. */
static const uint16_t *get_bf16(iris_transformer_qimage_t *tf, const char *name) {
    safetensors_file_t *sf;
    const safetensor_t *t = find_tensor(tf, name, &sf);
    if (!t || !safetensor_is_bf16(t)) {
        fprintf(stderr, "Qwen-Image: missing bf16 tensor %s\n", name);
        return NULL;
    }
    return safetensors_get_bf16_direct(sf, t);
}

/* Small tensors are converted to f32 (caller frees). */
static float *get_f32(iris_transformer_qimage_t *tf, const char *name) {
    safetensors_file_t *sf;
    const safetensor_t *t = find_tensor(tf, name, &sf);
    if (!t) {
        fprintf(stderr, "Qwen-Image: missing tensor %s\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

static int load_block(iris_transformer_qimage_t *tf, qi_block_t *b, int l) {
    char name[256];
#define QI_BF16(field, suffix) \
    snprintf(name, sizeof(name), "transformer_blocks.%d.%s", l, suffix); \
    b->field = get_bf16(tf, name);
    QI_BF16(to_q, "attn.to_q.weight");
    QI_BF16(to_k, "attn.to_k.weight");
    QI_BF16(to_v, "attn.to_v.weight");
    QI_BF16(to_out, "attn.to_out.0.weight");
    QI_BF16(mlp_gate, "img_mlp.gate_layer.weight");
    QI_BF16(mlp_proj, "img_mlp.proj.weight");
    QI_BF16(mlp_out, "img_mlp.out.weight");
#undef QI_BF16
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_q.weight", l);
    b->norm_q = get_f32(tf, name);
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_k.weight", l);
    b->norm_k = get_f32(tf, name);
    return b->to_q && b->to_k && b->to_v && b->to_out && b->mlp_gate && b->mlp_proj &&
           b->mlp_out && b->norm_q && b->norm_k;
}

/* Read a number from config.json, keeping def if absent. */
static double config_num(const char *json, const char *key, double def) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p || !(p = strchr(p, ':'))) return def;
    p++;
    while (*p == ' ') p++;
    if (!strncmp(p, "true", 4)) return 1;
    if (!strncmp(p, "false", 5)) return 0;
    if (!strncmp(p, "null", 4)) return def;
    return atof(p);
}

/* Open transformer/diffusion_pytorch_model*.safetensors (single file or
 * the shards named in the index). */
static int open_shards(iris_transformer_qimage_t *tf, const char *model_dir) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors.index.json",
             model_dir);
    FILE *f = fopen(path, "r");
    if (!f) {
        snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors",
                 model_dir);
        tf->sf[0] = safetensors_open(path);
        tf->num_sf = tf->sf[0] ? 1 : 0;
        return tf->num_sf;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *json = malloc(size + 1);
    if (!json) { fclose(f); return 0; }
    size_t n = fread(json, 1, size, f);
    json[n] = '\0';
    fclose(f);

    /* Unique shard file names among the weight_map values. */
    char names[QI_MAX_SHARDS][128];
    int count = 0;
    for (char *p = json; (p = strstr(p, ".safetensors\"")) != NULL; p++) {
        char *start = p;
        while (start > json && start[-1] != '"') start--;
        int len = (int)(p - start) + (int)strlen(".safetensors");
        if (len >= 128) continue;
        char fname[128];
        memcpy(fname, start, len);
        fname[len] = '\0';
        int seen = 0;
        for (int i = 0; i < count; i++) if (!strcmp(names[i], fname)) seen = 1;
        if (!seen && count < QI_MAX_SHARDS) strcpy(names[count++], fname);
    }
    free(json);

    for (int i = 0; i < count; i++) {
        snprintf(path, sizeof(path), "%s/transformer/%s", model_dir, names[i]);
        tf->sf[i] = safetensors_open(path);
        if (!tf->sf[i]) {
            fprintf(stderr, "Qwen-Image: failed to open %s\n", path);
            return 0;
        }
        tf->num_sf = i + 1;
    }
    return tf->num_sf;
}

static int read_config(const char *model_dir, char *cfg, size_t size) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/transformer/config.json", model_dir);
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Qwen-Image: missing %s\n", path);
        return 0;
    }
    size_t n = fread(cfg, 1, size - 1, f);
    cfg[n] = '\0';
    fclose(f);
    return 1;
}

/* Load the transformer from <model_dir>/transformer. Weights are not
 * copied: they are used straight from the mmapped safetensors shards. */
iris_transformer_qimage_t *iris_transformer_load_qimage(const char *model_dir) {
    iris_transformer_qimage_t *tf = calloc(1, sizeof(*tf));
    if (!tf) return NULL;

    char cfg[8192];
    if (!read_config(model_dir, cfg, sizeof(cfg))) {
        free(tf);
        return NULL;
    }

    tf->heads = (int)config_num(cfg, "num_attention_heads", 32);
    tf->head_dim = (int)config_num(cfg, "attention_head_dim", 128);
    tf->dim = tf->heads * tf->head_dim;
    tf->mlp = tf->dim * (int)config_num(cfg, "mlp_ratio", 3);
    tf->num_layers = (int)config_num(cfg, "num_layers", 32);
    tf->in_channels = (int)config_num(cfg, "in_channels", 64);
    tf->out_channels = (int)config_num(cfg, "out_channels", tf->in_channels);
    tf->text_dim = (int)config_num(cfg, "context_in_dim", 4096);
    tf->eps = (float)config_num(cfg, "eps", 1e-6);
    tf->axes_dim[0] = 16;
    tf->axes_dim[1] = tf->axes_dim[2] = 56;
    const char *axes = strstr(cfg, "\"axes_dims_rope\"");
    if (axes && (axes = strchr(axes, '['))) {
        for (int i = 0; i < QI_ROPE_AXES && axes; i++) {
            tf->axes_dim[i] = atoi(axes + 1);
            axes = strchr(axes + 1, ',');
        }
    }
    int patch = (int)config_num(cfg, "patch_size", 1);
    if (patch != 1 || tf->axes_dim[0] + tf->axes_dim[1] + tf->axes_dim[2] != tf->head_dim) {
        fprintf(stderr, "Qwen-Image: unsupported config (patch_size %d, rope axes %d+%d+%d)\n",
                patch, tf->axes_dim[0], tf->axes_dim[1], tf->axes_dim[2]);
        goto error;
    }
    if (!config_num(cfg, "causal_condition", 1)) {
        fprintf(stderr, "Qwen-Image: causal_condition=false is not supported\n");
        goto error;
    }

    if (!open_shards(tf, model_dir)) goto error;

    tf->t_lin1 = get_bf16(tf, "time_text_embed.timestep_embedder.linear_1.weight");
    tf->t_lin2 = get_bf16(tf, "time_text_embed.timestep_embedder.linear_2.weight");
    tf->txt_in1 = get_bf16(tf, "txt_in.in_layer.weight");
    tf->txt_in2 = get_bf16(tf, "txt_in.out_layer.weight");
    tf->img_in = get_bf16(tf, "img_in.weight");
    tf->mod = get_bf16(tf, "modulation.1.weight");
    tf->norm_out = get_bf16(tf, "norm_out.linear.weight");
    tf->proj_out = get_bf16(tf, "proj_out.weight");
    tf->txt_norm = get_f32(tf, "txt_in.text_norm.weight");
    if (!tf->t_lin1 || !tf->t_lin2 || !tf->txt_in1 || !tf->txt_in2 || !tf->img_in ||
        !tf->mod || !tf->norm_out || !tf->proj_out || !tf->txt_norm) goto error;
    /* Stored zero-centered: the effective scale is weight + 1. */
    for (int i = 0; i < tf->text_dim; i++) tf->txt_norm[i] += 1.0f;

    tf->blocks = calloc(tf->num_layers, sizeof(qi_block_t));
    if (!tf->blocks) goto error;
    for (int l = 0; l < tf->num_layers; l++)
        if (!load_block(tf, &tf->blocks[l], l)) goto error;

#ifdef IRIS_DEV
    if (iris_dev_available()) dev_load(tf);
#endif
    if (iris_verbose)
        fprintf(stderr, "  Qwen-Image transformer: dim=%d heads=%d blocks=%d mlp=%d%s\n",
                tf->dim, tf->heads, tf->num_layers, tf->mlp, tf->dev ? " on GPU" : "");
    return tf;

error:
    iris_transformer_free_qimage(tf);
    return NULL;
}

#ifdef USE_METAL
#define QI_GRAPH_BYTES ((size_t)6 << 30)
#else
#define QI_GRAPH_BYTES 0
#endif

size_t iris_transformer_memory_qimage(const char *model_dir, int rows, int text_rows) {
    char cfg[8192];
    if (!read_config(model_dir, cfg, sizeof(cfg))) return 0;
    size_t heads = (size_t)config_num(cfg, "num_attention_heads", 32);
    size_t dim = heads * (size_t)config_num(cfg, "attention_head_dim", 128);
    size_t mlp = dim * (size_t)config_num(cfg, "mlp_ratio", 3);
    size_t layers = (size_t)config_num(cfg, "num_layers", 32);
    /* Each row: its keys and values in every block, one block's activations
     * and its keys and values in the attention scratch. The rest is on a GPU
     * the weights and, on Metal, what the graphs of one attention hold (10
     * references, measured). On the CPU the weights are clean mapped pages,
     * and the 2 GB are the stages' leftovers next to the prefix (r9). */
    size_t kv_bytes = 4, fixed = (size_t)2 << 30;
    (void)text_rows;   /* CUDA and ROCm only */
#ifdef IRIS_DEV
    if (iris_dev_available()) {
        kv_bytes = 2;
        fixed = QI_GRAPH_BYTES + layers * (4 * dim * dim + 3 * mlp * dim) * 2;
#ifndef USE_METAL
        /* Causal text takes the GEMM attention: scores in f32 and bf16 for
         * every head, up to ATTN_SCRATCH, and its input projections. The keys
         * of text after a reference include the rows before it. */
        size_t scores = (size_t)text_rows * rows * heads * 6;
        fixed += (scores < ATTN_SCRATCH ? scores : ATTN_SCRATCH) + (size_t)text_rows * dim * 6;
#endif
    }
#endif
    size_t row = layers * 2 * dim * kv_bytes + 18 * dim + 6 * mlp + 4 * dim;
    return row * rows + fixed;
}

void iris_transformer_release_prefix_qimage(iris_transformer_qimage_t *tf) {
#ifdef IRIS_DEV
    if (tf->dev) {
        dev_free_prefix(tf);
        dev_free_activations(tf->dev);
    }
#endif
    tf->prefix_seq = 0;
    free(tf->prefix_key);
    tf->prefix_key = NULL;
    tf->prefix_key_bytes = 0;
    free(tf->prefix_kv);
    tf->prefix_kv = NULL;
    free(tf->work);
    tf->work = NULL;
    tf->work_elems = 0;
}

void iris_transformer_free_qimage(iris_transformer_qimage_t *tf) {
    if (!tf) return;
#ifdef IRIS_DEV
    dev_free(tf);
#endif
    if (tf->blocks) {
        for (int l = 0; l < tf->num_layers; l++) {
            free(tf->blocks[l].norm_q);
            free(tf->blocks[l].norm_k);
        }
        free(tf->blocks);
    }
    free(tf->txt_norm);
    for (int i = 0; i < tf->num_sf; i++) safetensors_close(tf->sf[i]);
    free(tf->prefix_key);
    free(tf->prefix_kv);
    free(tf->wbuf);
    free(tf->work);
    free(tf);
}
