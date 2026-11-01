/*
 * Qwen-Image 2.1 VAE
 *
 * AutoencoderKLQwenImage21, a Wan-derived VAE specialized to a single frame:
 * its "causal 3D" convolutions are plain 2D convolutions (symmetric padding),
 * and with one frame the temporal convolutions never run, so everything is
 * 2D. Decoder:
 *
 *   x = post_quant_conv(latent * std + mean)        1x1, z_dim -> z_dim
 *   x = conv_in(x)                                  3x3, z_dim -> top width
 *   x = mid: res, attention, res
 *   x = up blocks (residual): res blocks, nearest 2x + 3x3 conv, plus a
 *       DupUp shortcut of the block input
 *   x = conv_out(silu(norm_out(x)))                 -> RGBA
 *   clamp to [-1, 1]
 *
 * Encoder, the mirror image, loaded at the first encode:
 *
 *   x = conv_in(rgba)                               3x3, 4 -> base width
 *   x = down blocks (residual): res blocks, 3x3 conv with stride 2 (zero
 *       padding right and bottom), plus an AvgDown shortcut of the block input
 *   x = mid: res, attention, res
 *   x = quant_conv(conv_out(silu(norm_out(x))))     -> mean | logvar
 *   latent = (mean - latents_mean) / latents_std
 *
 * Norms are per-pixel RMS norms over the channels: x / ||x|| * sqrt(C) *
 * gamma (F.normalize, no epsilon inside the root). Res block: shortcut
 * (1x1 when the channel count changes) + conv2(silu(norm2(conv1(silu(
 * norm1(x)))))). The mid attention is single-head over all latent pixels.
 *
 * Activations are f32 NCHW [C, H, W], channels-last on the GPU. Weights are
 * converted to f32 at load (the checkpoint may be f32 or bf16).
 */

#include "iris.h"
#include "iris_kernels.h"
#include "iris_safetensors.h"
#include "iris_qimage.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

#ifdef IRIS_DEV
#include "iris_dev.h"
#endif

#define QV_MAX_CH      512   /* latent channels / config array bound */
#define QV_MAX_LEVELS  8     /* entries of dim_mult */
#define QV_MAX_RES     4     /* res blocks per up block */
#define QV_ATTN_QCHUNK 256   /* queries per attention chunk */

/* Stage hooks for debug/cmp_qimage_vae_enc.c, which includes this file. */
#ifndef QV_STAGE
#define QV_STAGE(name, index, x, c, h, w)
#define QD_STAGE(name, index, x, c, h, w)
#endif

/* ========================================================================
 * Data Structures
 * ======================================================================== */

typedef struct {
    float *w, *b;            /* [cout, cin, k, k], [cout] */
    int cin, cout, k;
} qv_conv_t;

typedef struct {
    int cin, cout;
    float *gamma1, *gamma2;  /* [cin], [cout] */
    qv_conv_t conv1, conv2;
    qv_conv_t shortcut;      /* 1x1, only when cin != cout (w == NULL otherwise) */
} qv_res_t;

typedef struct {
    int cin, cout;
    int num_res;
    qv_res_t res[QV_MAX_RES];
    int resample;            /* 2x up or downsampling with a shortcut */
    int factor_t;            /* temporal factor of the shortcut (1 or 2) */
    qv_conv_t conv;          /* 3x3 conv of the resampling */
} qv_block_t;

typedef struct {
    float *gamma;            /* norm [C] */
    qv_conv_t qkv;           /* 1x1, C -> 3C */
    qv_conv_t proj;          /* 1x1, C -> C */
} qv_attn_t;

/* One direction of the VAE. */
typedef struct {
    qv_conv_t quant;         /* post_quant_conv or quant_conv */
    qv_conv_t conv_in, conv_out;
    qv_res_t mid_res[2];
    qv_attn_t attn;
    int num_blocks;
    qv_block_t block[QV_MAX_LEVELS];
    float *norm_out;         /* [C] */
#ifdef IRIS_DEV
    struct qv_dev *dev;      /* device copy, uploaded at the first use */
#endif
} qv_net_t;

struct iris_vae_qimage {
    int z_dim;
    int out_channels;        /* 4 (RGBA) */
    float mean[QV_MAX_CH], std[QV_MAX_CH];
    qv_net_t dec;
    qv_net_t *enc;           /* NULL until the first encode */

    /* What the encoder load needs */
    char *dir;               /* model directory */
    int in_channels, enc_res, levels;
    int enc_dims[QV_MAX_LEVELS + 1];
    int enc_ft[QV_MAX_LEVELS];
};

/* ========================================================================
 * Layers
 * ======================================================================== */

/* Convolution with zero padding k/2 and stride 1 (caller frees). */
static float *conv(const qv_conv_t *c, const float *x, int h, int w) {
    float *out = malloc((size_t)c->cout * h * w * sizeof(float));
    if (out) iris_conv2d(out, x, c->w, c->b, 1, c->cin, c->cout, h, w, c->k, c->k, 1, c->k / 2);
    return out;
}

/* In place: x = x / max(||x||, 1e-12) * sqrt(C) * gamma per pixel, then
 * SiLU if silu is set. */
static void pixel_rms_norm(float *x, const float *gamma, int c, int hw, int silu) {
    float *inv = calloc(hw, sizeof(float));
    for (int ch = 0; ch < c; ch++) {
        const float *xc = x + (size_t)ch * hw;
        for (int p = 0; p < hw; p++) inv[p] += xc[p] * xc[p];
    }
    float root_c = sqrtf((float)c);
    for (int p = 0; p < hw; p++) {
        float norm = sqrtf(inv[p]);
        inv[p] = root_c / (norm > 1e-12f ? norm : 1e-12f);
    }
    for (int ch = 0; ch < c; ch++) {
        float *xc = x + (size_t)ch * hw;
        float g = gamma[ch];
        for (int p = 0; p < hw; p++) {
            float v = xc[p] * inv[p] * g;
            xc[p] = silu ? v / (1.0f + expf(-v)) : v;
        }
    }
    free(inv);
}

/* Residual block on x [cin, h, w]. Returns the output [cout, h, w] and
 * frees x when consume is set (x is then used as scratch). */
static float *res_forward(const qv_res_t *r, float *x, int h, int w, int consume) {
    size_t hw = (size_t)h * w;
    float *skip = r->shortcut.w ? conv(&r->shortcut, x, h, w) : NULL;
    float *t = x;
    if (!consume) {
        t = malloc((size_t)r->cin * hw * sizeof(float));
        memcpy(t, x, (size_t)r->cin * hw * sizeof(float));
    } else if (!skip) {
        skip = malloc((size_t)r->cout * hw * sizeof(float));
        memcpy(skip, x, (size_t)r->cout * hw * sizeof(float));
    }
    pixel_rms_norm(t, r->gamma1, r->cin, (int)hw, 1);
    float *h1 = conv(&r->conv1, t, h, w);
    free(t);
    pixel_rms_norm(h1, r->gamma2, r->cout, (int)hw, 1);
    float *out = conv(&r->conv2, h1, h, w);
    free(h1);
    const float *s = skip ? skip : x;   /* identity shortcut reads x */
    for (size_t i = 0; i < (size_t)r->cout * hw; i++) out[i] += s[i];
    free(skip);
    return out;
}

/* Single-head self-attention over the pixels of x [c, h, w], in place:
 * x += proj(softmax(q^T k / sqrt(c)) v), with q, k, v from a 1x1 conv of
 * the normalized x. */
static void mid_attention(const qv_attn_t *a, float *x, int c, int h, int w) {
    int n = h * w;
    size_t cn = (size_t)c * n;
    float *t = malloc(cn * sizeof(float));
    memcpy(t, x, cn * sizeof(float));
    pixel_rms_norm(t, a->gamma, c, n, 0);
    float *qkv = conv(&a->qkv, t, h, w);   /* [3c, n]: q, k, v */
    const float *q = qkv, *k = qkv + cn, *v = qkv + 2 * cn;
    float *scores = malloc((size_t)QV_ATTN_QCHUNK * n * sizeof(float));
    float scale = 1.0f / sqrtf((float)c);

    for (int q0 = 0; q0 < n; q0 += QV_ATTN_QCHUNK) {
        int nq = n - q0 < QV_ATTN_QCHUNK ? n - q0 : QV_ATTN_QCHUNK;
        /* scores[i, j] = scale * sum_c q[c, q0 + i] * k[c, j] */
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, nq, n, c,
                    scale, q + q0, n, k, n, 0.0f, scores, n);
#else
        memset(scores, 0, (size_t)nq * n * sizeof(float));
        for (int ch = 0; ch < c; ch++) {
            const float *qc = q + (size_t)ch * n + q0, *kc = k + (size_t)ch * n;
            for (int i = 0; i < nq; i++) {
                float qi = qc[i] * scale;
                float *si = scores + (size_t)i * n;
                for (int j = 0; j < n; j++) si[j] += qi * kc[j];
            }
        }
#endif
        iris_softmax(scores, nq, n);
        /* t[c, q0 + i] = sum_j v[c, j] * scores[i, j] */
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, c, nq, n,
                    1.0f, v, n, scores, n, 0.0f, t + q0, n);
#else
        for (int ch = 0; ch < c; ch++) {
            const float *vc = v + (size_t)ch * n;
            for (int i = 0; i < nq; i++) {
                const float *si = scores + (size_t)i * n;
                float acc = 0;
                for (int j = 0; j < n; j++) acc += vc[j] * si[j];
                t[(size_t)ch * n + q0 + i] = acc;
            }
        }
#endif
    }
    free(scores);
    free(qkv);
    float *o = conv(&a->proj, t, h, w);
    for (size_t i = 0; i < cn; i++) x[i] += o[i];
    free(o);
    free(t);
}

/* out [cout, 2h, 2w] += DupUp(in [cin, h, w]) with first_chunk set
 * (QwenImage21DupUp3D): channels repeated, folded into
 * (cout, factor_t, 2, 2), pixel-shuffled, temporal slot factor_t - 1 kept:
 *   out[o, 2y+i, 2x+j] += in[(((o*ft + ft-1)*2 + i)*2 + j) / repeats, y, x] */
static void dup_up_add(float *out, const float *in, int cin, int cout, int ft, int h, int w) {
    int repeats = cout * ft * 4 / cin;
    int ow = 2 * w;
    for (int o = 0; o < cout; o++) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                int src = (((o * ft + ft - 1) * 2 + i) * 2 + j) / repeats;
                const float *s = in + (size_t)src * h * w;
                float *d = out + (size_t)o * 4 * h * w;
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        d[(size_t)(2 * y + i) * ow + 2 * x + j] += s[(size_t)y * w + x];
            }
        }
    }
}

/* Up block on x [cin, h, w] (consumed). Returns [cout, h', w'] and updates
 * h, w when the block upsamples. */
static float *up_forward(const qv_block_t *u, float *x, int *h, int *w) {
    float *cur = x;
    for (int r = 0; r < u->num_res; r++)
        cur = res_forward(&u->res[r], cur, *h, *w, r > 0 || !u->resample);
    if (!u->resample) return cur;

    float *big = malloc((size_t)u->cout * 4 * *h * *w * sizeof(float));
    iris_upsample_nearest(big, cur, 1, u->cout, *h, *w, 2, 2);
    free(cur);
    float *out = conv(&u->conv, big, 2 * *h, 2 * *w);
    free(big);
    dup_up_add(out, x, u->cin, u->cout, u->factor_t, *h, *w);
    free(x);
    *h *= 2;
    *w *= 2;
    return out;
}

/* out [cout, h, w] += AvgDown(in [cin, 2h, 2w]) (QwenImage21AvgDown3D): a zero
 * frame before the real one, space-to-channel, mean over g channels:
 *   out[((c*ft + ft-1)*2 + i)*2 + j) / g, y, x] += in[c, 2y+i, 2x+j] / g
 * with g = cin * ft * 4 / cout. */
static void avg_down_add(float *out, const float *in, int cin, int cout, int ft, int h, int w) {
    int g = cin * ft * 4 / cout;
    float scale = 1.0f / g;
    for (int c = 0; c < cin; c++) {
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                int o = (((c * ft + ft - 1) * 2 + i) * 2 + j) / g;
                const float *s = in + (size_t)c * 4 * h * w;
                float *d = out + (size_t)o * h * w;
                for (int y = 0; y < h; y++)
                    for (int x = 0; x < w; x++)
                        d[(size_t)y * w + x] += scale * s[(size_t)(2 * y + i) * 2 * w + 2 * x + j];
            }
        }
    }
}

/* 3x3 convolution with stride 2 of x [cin, h, w], zero-padded by 1 on the
 * right and bottom (caller frees). */
static float *down_conv(const qv_conv_t *c, const float *x, int h, int w) {
    float *pad = calloc((size_t)c->cin * (h + 1) * (w + 1), sizeof(float));
    float *out = malloc((size_t)c->cout * (h / 2) * (w / 2) * sizeof(float));
    for (int ch = 0; ch < c->cin; ch++)
        for (int y = 0; y < h; y++)
            memcpy(pad + ((size_t)ch * (h + 1) + y) * (w + 1), x + ((size_t)ch * h + y) * w,
                   (size_t)w * sizeof(float));
    iris_conv2d(out, pad, c->w, c->b, 1, c->cin, c->cout, h + 1, w + 1, c->k, c->k, 2, 0);
    free(pad);
    return out;
}

/* Down block on x [cin, h, w] (consumed). Returns [cout, h', w'] and updates
 * h, w when the block downsamples. */
static float *down_forward(const qv_block_t *b, float *x, int *h, int *w) {
    size_t n = (size_t)b->cin * *h * *w;
    float *cur = x;
    for (int r = 0; r < b->num_res; r++)
        cur = res_forward(&b->res[r], cur, *h, *w, r > 0);
    if (b->resample) {
        float *out = down_conv(&b->conv, cur, *h, *w);
        free(cur);
        cur = out;
        *h /= 2;
        *w /= 2;
        avg_down_add(cur, x, b->cin, b->cout, b->factor_t, *h, *w);
    } else {
        for (size_t i = 0; i < n; i++) cur[i] += x[i];
    }
    free(x);
    return cur;
}

/* The encoder on rgba [in_channels, h, w]: mean | logvar [2z, h', w'] (caller
 * frees). */
static float *encode_host(const qv_net_t *e, const float *rgba, int h, int w) {
    float *x = conv(&e->conv_in, rgba, h, w);
    QV_STAGE("conv_in", -1, x, e->conv_in.cout, h, w);
    for (int i = 0; i < e->num_blocks; i++) {
        x = down_forward(&e->block[i], x, &h, &w);
        QV_STAGE("down", i, x, e->block[i].cout, h, w);
    }
    int c = e->mid_res[0].cout;
    x = res_forward(&e->mid_res[0], x, h, w, 1);
    mid_attention(&e->attn, x, c, h, w);
    x = res_forward(&e->mid_res[1], x, h, w, 1);
    QV_STAGE("mid", -1, x, c, h, w);
    pixel_rms_norm(x, e->norm_out, c, h * w, 1);
    QV_STAGE("norm_out", -1, x, c, h, w);
    float *y = conv(&e->conv_out, x, h, w);
    free(x);
    QV_STAGE("conv_out", -1, y, e->conv_out.cout, h, w);
    x = conv(&e->quant, y, h, w);
    free(y);
    QV_STAGE("quant", -1, x, e->quant.cout, h, w);
    return x;
}

/* [channels, h, w] floats in [-1, 1] to an image: clamp, then round(v * 255)
 * like diffusers, dropping the alpha channel of a fully opaque image. */
static iris_image *rgba_to_image(const float *rgba, int channels, int h, int w) {
    size_t npix = (size_t)h * w;
    iris_image *img = iris_image_create(w, h, channels);
    if (!img) return NULL;
    int opaque = channels == 4;
    for (size_t p = 0; p < npix; p++) {
        for (int ch = 0; ch < channels; ch++) {
            float v = rgba[ch * npix + p];
            v = v < -1.0f ? -1.0f : (v > 1.0f ? 1.0f : v);
            img->data[p * channels + ch] = (uint8_t)((v + 1.0f) * 127.5f + 0.5f);
        }
        if (opaque && img->data[p * channels + 3] != 255) opaque = 0;
    }
    if (opaque) {
        for (size_t p = 0; p < npix; p++)
            memmove(img->data + p * 3, img->data + p * 4, 3);
        img->channels = 3;
    }
    return img;
}

#ifdef IRIS_DEV
/* ========================================================================
 * GPU Path (CUDA / ROCm / Metal)
 * ======================================================================== */

typedef struct {
    uint16_t *w;             /* bf16 [cout, k * k * cin] */
    float *b;                /* [cout] */
} qd_conv_t;

typedef struct {
    float *gamma1, *gamma2;
    qd_conv_t conv1, conv2, shortcut;
} qd_res_t;

typedef struct {
    float *gamma;
    qd_conv_t qkv, proj;
} qd_attn_t;

struct qv_dev {
    qd_conv_t quant, conv_in, conv_out;
    qd_res_t mid_res[2];
    qd_attn_t attn;
    qd_res_t res[QV_MAX_LEVELS][QV_MAX_RES];
    qd_conv_t conv[QV_MAX_LEVELS];
    float *norm_out;

    /* Activations, sized by dv_buffers() */
    float *buf[4];           /* x, y, t1, t2 */
    float *skip;             /* input of a resampling block */
    uint16_t *attn_buf;      /* q, k, v, out */
    size_t buf_elems, skip_elems, attn_elems;
};

static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

static float *dv_f32(const float *src, size_t n) {
    float *d = iris_dev_alloc(n * sizeof(float));
    if (d) iris_dev_upload(d, src, n * sizeof(float));
    return d;
}

/* Upload a convolution: weight [cout, cin, k, k] to the channels-last
 * layout of iris_dev_conv2d. A missing convolution stays NULL. */
static int dv_conv(qd_conv_t *d, const qv_conv_t *c) {
    if (!c->w) return 1;
    int kk = c->k * c->k;
    size_t row = (size_t)kk * c->cin, n = row * c->cout;
    uint16_t *tmp = malloc(n * sizeof(uint16_t));
    if (!tmp) return 0;
    for (int o = 0; o < c->cout; o++)
        for (int ci = 0; ci < c->cin; ci++)
            for (int t = 0; t < kk; t++)
                tmp[o * row + (size_t)t * c->cin + ci] =
                    f32_to_bf16(c->w[((size_t)o * c->cin + ci) * kk + t]);
    d->w = iris_dev_alloc(n * sizeof(uint16_t));
    if (d->w) iris_dev_upload(d->w, tmp, n * sizeof(uint16_t));
    free(tmp);
    d->b = dv_f32(c->b, c->cout);
    return d->w && d->b;
}

static int dv_res(qd_res_t *d, const qv_res_t *r) {
    d->gamma1 = dv_f32(r->gamma1, r->cin);
    d->gamma2 = dv_f32(r->gamma2, r->cout);
    return d->gamma1 && d->gamma2 && dv_conv(&d->conv1, &r->conv1) &&
           dv_conv(&d->conv2, &r->conv2) && dv_conv(&d->shortcut, &r->shortcut);
}

static void dv_free_conv(qd_conv_t *d) {
    iris_dev_free(d->w);
    iris_dev_free(d->b);
}

static void dv_free_res(qd_res_t *d) {
    iris_dev_free(d->gamma1);
    iris_dev_free(d->gamma2);
    dv_free_conv(&d->conv1);
    dv_free_conv(&d->conv2);
    dv_free_conv(&d->shortcut);
}

static void dv_free_buffers(struct qv_dev *d) {
    for (int i = 0; i < 4; i++) {
        iris_dev_free(d->buf[i]);
        d->buf[i] = NULL;
    }
    iris_dev_free(d->skip);
    iris_dev_free(d->attn_buf);
    d->skip = NULL;
    d->attn_buf = NULL;
    d->buf_elems = d->skip_elems = d->attn_elems = 0;
}

static void dv_free(qv_net_t *n) {
    struct qv_dev *d = n->dev;
    if (!d) return;
    dv_free_conv(&d->quant);
    dv_free_conv(&d->conv_in);
    dv_free_conv(&d->conv_out);
    dv_free_conv(&d->attn.qkv);
    dv_free_conv(&d->attn.proj);
    dv_free_res(&d->mid_res[0]);
    dv_free_res(&d->mid_res[1]);
    for (int i = 0; i < n->num_blocks; i++) {
        for (int r = 0; r < n->block[i].num_res; r++) dv_free_res(&d->res[i][r]);
        dv_free_conv(&d->conv[i]);
    }
    iris_dev_free(d->attn.gamma);
    iris_dev_free(d->norm_out);
    dv_free_buffers(d);
    free(d);
    n->dev = NULL;
}

static int dv_load(qv_net_t *n) {
    struct qv_dev *d = calloc(1, sizeof(*d));
    if (!d) return 0;
    n->dev = d;
    int top = n->mid_res[0].cout, last = n->conv_out.cin;
    int ok = dv_conv(&d->quant, &n->quant) && dv_conv(&d->conv_in, &n->conv_in) &&
             dv_conv(&d->conv_out, &n->conv_out) && dv_conv(&d->attn.qkv, &n->attn.qkv) &&
             dv_conv(&d->attn.proj, &n->attn.proj) && dv_res(&d->mid_res[0], &n->mid_res[0]) &&
             dv_res(&d->mid_res[1], &n->mid_res[1]);
    for (int i = 0; ok && i < n->num_blocks; i++) {
        for (int r = 0; ok && r < n->block[i].num_res; r++)
            ok = dv_res(&d->res[i][r], &n->block[i].res[r]);
        ok = ok && dv_conv(&d->conv[i], &n->block[i].conv);
    }
    d->attn.gamma = ok ? dv_f32(n->attn.gamma, top) : NULL;
    d->norm_out = ok ? dv_f32(n->norm_out, last) : NULL;
    if (!d->attn.gamma || !d->norm_out) {
        dv_free(n);
        return 0;
    }
    iris_dev_sync();
    return 1;
}

/* Grow the activations to these element counts. */
static size_t dv_bytes(size_t need, size_t skip, size_t attn) {
    return (4 * need + skip) * sizeof(float) + attn * sizeof(uint16_t);
}

static int dv_buffers(struct qv_dev *d, size_t need, size_t skip, size_t attn) {
    if (need <= d->buf_elems && skip <= d->skip_elems && attn <= d->attn_elems) return 1;
    dv_free_buffers(d);
    /* Metal allocates past its budget: ask first. */
    if (dv_bytes(need, skip, attn) > iris_dev_memory(0)) return 0;
    int ok = 1;
    for (int i = 0; i < 4; i++) {
        d->buf[i] = iris_dev_alloc(need * sizeof(float));
        ok = ok && d->buf[i];
    }
    d->skip = iris_dev_alloc(skip * sizeof(float));
    d->attn_buf = iris_dev_alloc(attn * sizeof(uint16_t));
    if (!ok || !d->skip || !d->attn_buf) {
        dv_free_buffers(d);
        return 0;
    }
    d->buf_elems = need;
    d->skip_elems = skip;
    d->attn_elems = attn;
    return 1;
}

/* Decoder activations for a latent of h x w pixels, in elements of each buffer. */
static void dv_sizes(const qv_net_t *n, int h, int w, size_t *need_out, size_t *skip_out,
                     size_t *attn_out) {
    size_t hw = (size_t)h * w, top = n->mid_res[0].cout;
    size_t need = hw * 3 * top, skip = 0;
    *attn_out = hw * 4 * top;
    for (int i = 0; i < n->num_blocks; i++) {
        const qv_block_t *u = &n->block[i];
        size_t c = u->cin > u->cout ? u->cin : u->cout;
        if (hw * c > need) need = hw * c;
        if (u->resample) {
            if (hw * u->cin > skip) skip = hw * u->cin;
            hw *= 4;
            if (hw * u->cout > need) need = hw * u->cout;
        }
    }
    *need_out = need;
    *skip_out = skip;
}

static int dv_ensure(qv_net_t *n, int h, int w) {
    size_t need, skip, attn;
    dv_sizes(n, h, w, &need, &skip, &attn);
    return dv_buffers(n->dev, need, skip, attn);
}

/* Size the encoder activations for an image of h x w pixels. */
static int dv_ensure_enc(qv_net_t *n, int h, int w) {
    size_t hw = (size_t)h * w, top = n->mid_res[0].cout, need = 0;
    for (int i = 0; i < n->num_blocks; i++) {
        const qv_block_t *b = &n->block[i];
        size_t c = b->cin > b->cout ? b->cin : b->cout;
        if (hw * c > need) need = hw * c;
        if (b->resample) hw /= 4;
    }
    if (hw * 3 * top > need) need = hw * 3 * top;
    return dv_buffers(n->dev, need, need, hw * 4 * top);
}

static void dv_conv2d(float *out, const float *in, const qd_conv_t *d, const qv_conv_t *c,
                      int h, int w) {
    iris_dev_conv2d(out, in, d->w, d->b, h, w, c->cin, c->cout, c->k, 1, c->k / 2, c->k / 2);
}

/* out = shortcut(x) + conv2(silu(norm2(conv1(silu(norm1(x)))))) */
static void dv_res_forward(const qd_res_t *d, const qv_res_t *r, float *out, const float *x,
                           float *t1, float *t2, int h, int w) {
    int hw = h * w;
    iris_dev_pixel_rms_norm(t1, x, d->gamma1, hw, r->cin, 1);
    dv_conv2d(t2, t1, &d->conv1, &r->conv1, h, w);
    iris_dev_pixel_rms_norm(t2, t2, d->gamma2, hw, r->cout, 1);
    dv_conv2d(out, t2, &d->conv2, &r->conv2, h, w);
    if (d->shortcut.w) {
        dv_conv2d(t1, x, &d->shortcut, &r->shortcut, h, w);
        iris_dev_add(out, t1, (size_t)hw * r->cout);
    } else {
        iris_dev_add(out, x, (size_t)hw * r->cout);
    }
}

/* Single-head self-attention over the n pixels, plus the residual. */
static void dv_attention(const qv_net_t *net, float *out, const float *x, float *t1,
                         float *t2, int n, int c) {
    const qd_attn_t *a = &net->dev->attn;
    uint16_t *qkv = net->dev->attn_buf, *o = qkv + (size_t)n * 3 * c;
    iris_dev_pixel_rms_norm(t1, x, a->gamma, n, c, 0);
    dv_conv2d(t2, t1, &a->qkv, &net->attn.qkv, n, 1);
    iris_dev_f32_to_bf16(qkv, t2, (size_t)n * 3 * c);
    iris_dev_attention(o, c, qkv, qkv + c, qkv + 2 * c, 3 * c, n, n, 1, 1, c,
                       1.0f / sqrtf((float)c), 0, NULL);
    iris_dev_linear_f32(out, o, a->proj.w, n, c, c);
    iris_dev_add_bias(out, a->proj.b, n, c);
    iris_dev_add(out, x, (size_t)n * c);
}

int iris_vae_reserve_qimage(iris_vae_qimage_t *vae, int h, int w) {
    if (!iris_dev_available()) return 1;
    if (!vae->dec.dev && !dv_load(&vae->dec)) return 0;
    return dv_ensure(&vae->dec, h, w);
}

size_t iris_vae_memory_qimage(const iris_vae_qimage_t *vae, int h, int w) {
    if (!iris_dev_available()) return 0;
    size_t need, skip, attn;
    dv_sizes(&vae->dec, h, w, &need, &skip, &attn);
    return dv_bytes(need, skip, attn);
}

static iris_image *decode_dev(iris_vae_qimage_t *vae, const float *latent, int h, int w) {
    if (!iris_vae_reserve_qimage(vae, h, w)) return NULL;
    const qv_net_t *n = &vae->dec;
    struct qv_dev *d = n->dev;
    float *x = d->buf[0], *y = d->buf[1], *t1 = d->buf[2], *t2 = d->buf[3], *tmp;
#define DV_SWAP() (tmp = x, x = y, y = tmp)

    int z = vae->z_dim, top = n->mid_res[0].cout;
    size_t hw = (size_t)h * w;
    float *host = malloc((size_t)z * hw * sizeof(float));
    if (!host) return NULL;
    for (int c = 0; c < z; c++)
        for (size_t p = 0; p < hw; p++)
            host[p * z + c] = latent[c * hw + p] * vae->std[c] + vae->mean[c];
    iris_dev_upload(x, host, (size_t)z * hw * sizeof(float));
    free(host);

    dv_conv2d(y, x, &d->quant, &n->quant, h, w);
    dv_conv2d(x, y, &d->conv_in, &n->conv_in, h, w);
    dv_res_forward(&d->mid_res[0], &n->mid_res[0], y, x, t1, t2, h, w);
    dv_attention(n, x, y, t1, t2, (int)hw, top);
    dv_res_forward(&d->mid_res[1], &n->mid_res[1], y, x, t1, t2, h, w);
    DV_SWAP();

    for (int i = 0; i < n->num_blocks; i++) {
        const qv_block_t *u = &n->block[i];
        /* A byte copy of the block input for the DupUp shortcut. */
        if (u->resample)
            iris_dev_copy_2d((uint16_t *)d->skip, 2 * u->cin, (const uint16_t *)x, 2 * u->cin,
                             h * w, 2 * u->cin);
        for (int r = 0; r < u->num_res; r++) {
            dv_res_forward(&d->res[i][r], &u->res[r], y, x, t1, t2, h, w);
            DV_SWAP();
        }
        if (u->resample) {
            iris_dev_upsample2x(y, x, h, w, u->cout);
            dv_conv2d(x, y, &d->conv[i], &u->conv, 2 * h, 2 * w);
            iris_dev_dup_up_add(x, d->skip, h, w, u->cin, u->cout, u->factor_t);
            h *= 2;
            w *= 2;
        }
        if (iris_vae_progress_callback) iris_vae_progress_callback(i + 1, n->num_blocks);
    }
#undef DV_SWAP

    int c = n->conv_out.cin, channels = vae->out_channels;
    iris_dev_pixel_rms_norm(y, x, d->norm_out, h * w, c, 1);
    dv_conv2d(t1, y, &d->conv_out, &n->conv_out, h, w);

    hw = (size_t)h * w;
    float *hwc = malloc((size_t)channels * hw * sizeof(float));
    float *rgba = malloc((size_t)channels * hw * sizeof(float));
    iris_image *img = NULL;
    if (hwc && rgba) {
        iris_dev_download(hwc, t1, (size_t)channels * hw * sizeof(float));
        for (int ch = 0; ch < channels; ch++)
            for (size_t p = 0; p < hw; p++) rgba[ch * hw + p] = hwc[p * channels + ch];
        img = rgba_to_image(rgba, channels, h, w);
    }
    free(hwc);
    free(rgba);
    return img;
}

/* The encoder on the device: the mean [z, h', w'] (caller frees), or NULL. */
static float *encode_dev(const iris_vae_qimage_t *vae, const float *rgba, int h, int w) {
    qv_net_t *n = vae->enc;
    if (!n->dev && !dv_load(n)) return NULL;
    if (!dv_ensure_enc(n, h, w)) return NULL;
    struct qv_dev *d = n->dev;
    float *x = d->buf[0], *y = d->buf[1], *z = d->skip, *t1 = d->buf[2], *t2 = d->buf[3];

    int cin = n->conv_in.cin;
    size_t hw = (size_t)h * w;
    float *hwc = malloc((size_t)cin * hw * sizeof(float));
    if (!hwc) return NULL;
    for (int c = 0; c < cin; c++)
        for (size_t p = 0; p < hw; p++) hwc[p * cin + c] = rgba[c * hw + p];
    iris_dev_upload(x, hwc, (size_t)cin * hw * sizeof(float));
    free(hwc);

    dv_conv2d(y, x, &d->conv_in, &n->conv_in, h, w);
    float *tmp = x;
    x = y;
    y = tmp;
    QD_STAGE("conv_in", -1, x, n->conv_in.cout, h, w);

    /* x stays the block input; the res blocks alternate between y and z. */
    for (int i = 0; i < n->num_blocks; i++) {
        const qv_block_t *b = &n->block[i];
        float *src = x;
        for (int r = 0; r < b->num_res; r++) {
            dv_res_forward(&d->res[i][r], &b->res[r], y, src, t1, t2, h, w);
            tmp = src == x ? z : src;
            src = y;
            y = tmp;
        }
        if (b->resample) {
            iris_dev_conv2d(y, src, d->conv[i].w, d->conv[i].b, h, w, b->cout, b->cout,
                            b->conv.k, 2, 0, 1);
            iris_dev_avg_down_add(y, x, h, w, b->cin, b->cout, b->factor_t);
            h /= 2;
            w /= 2;
            tmp = x;
            x = y;
            y = src;
            z = tmp;
        } else {
            iris_dev_add(src, x, (size_t)h * w * b->cout);
            tmp = x;
            x = src;
            z = tmp;
        }
        QD_STAGE("down", i, x, b->cout, h, w);
    }

    int top = n->mid_res[0].cout;
    hw = (size_t)h * w;
    dv_res_forward(&d->mid_res[0], &n->mid_res[0], y, x, t1, t2, h, w);
    dv_attention(n, x, y, t1, t2, (int)hw, top);
    dv_res_forward(&d->mid_res[1], &n->mid_res[1], y, x, t1, t2, h, w);
    QD_STAGE("mid", -1, y, top, h, w);
    iris_dev_pixel_rms_norm(x, y, d->norm_out, (int)hw, top, 1);
    QD_STAGE("norm_out", -1, x, top, h, w);
    dv_conv2d(y, x, &d->conv_out, &n->conv_out, h, w);
    QD_STAGE("conv_out", -1, y, n->conv_out.cout, h, w);
    dv_conv2d(x, y, &d->quant, &n->quant, h, w);
    QD_STAGE("quant", -1, x, n->quant.cout, h, w);

    int moments = n->quant.cout, zc = vae->z_dim;
    float *out = malloc((size_t)zc * hw * sizeof(float));
    hwc = malloc((size_t)moments * hw * sizeof(float));
    if (!out || !hwc) {
        free(out);
        free(hwc);
        return NULL;
    }
    iris_dev_download(hwc, x, (size_t)moments * hw * sizeof(float));
    for (int c = 0; c < zc; c++)
        for (size_t p = 0; p < hw; p++) out[c * hw + p] = hwc[p * moments + c];
    free(hwc);
    return out;
}
#else
int iris_vae_reserve_qimage(iris_vae_qimage_t *vae, int h, int w) {
    (void)vae, (void)h, (void)w;
    return 1;
}

size_t iris_vae_memory_qimage(const iris_vae_qimage_t *vae, int h, int w) {
    (void)vae, (void)h, (void)w;
    return 0;
}
#endif /* IRIS_DEV */

/* ========================================================================
 * Decode
 * ======================================================================== */

iris_image *iris_vae_decode_qimage(iris_vae_qimage_t *vae, const float *latent,
                                   int h, int w) {
#ifdef IRIS_DEV
    if (iris_dev_available()) {
        iris_image *img = decode_dev(vae, latent, h, w);
        if (img) return img;
        fprintf(stderr, "Qwen-Image VAE: GPU decode failed, using the CPU\n");
    }
#endif
    size_t hw = (size_t)h * w;
    float *z = malloc((size_t)vae->z_dim * hw * sizeof(float));
    for (int c = 0; c < vae->z_dim; c++)
        for (size_t p = 0; p < hw; p++)
            z[c * hw + p] = latent[c * hw + p] * vae->std[c] + vae->mean[c];

    const qv_net_t *n = &vae->dec;
    float *x = conv(&n->quant, z, h, w);
    free(z);
    z = conv(&n->conv_in, x, h, w);
    free(x);
    x = res_forward(&n->mid_res[0], z, h, w, 1);
    mid_attention(&n->attn, x, n->mid_res[0].cout, h, w);
    x = res_forward(&n->mid_res[1], x, h, w, 1);

    int oh = h, ow = w;
    for (int i = 0; i < n->num_blocks; i++) {
        x = up_forward(&n->block[i], x, &oh, &ow);
        if (iris_vae_progress_callback) iris_vae_progress_callback(i + 1, n->num_blocks);
    }

    int c = n->conv_out.cin;
    pixel_rms_norm(x, n->norm_out, c, oh * ow, 1);
    float *rgba = conv(&n->conv_out, x, oh, ow);
    free(x);

    iris_image *img = rgba_to_image(rgba, vae->out_channels, oh, ow);
    free(rgba);
    return img;
}

/* ========================================================================
 * Loading
 * ======================================================================== */

/* Read a JSON number array "key": [..] into out (at most max values).
 * Returns the count, 0 when absent. true/false read as 1/0. */
static int json_array(const char *json, const char *key, float *out, int max) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p || !(p = strchr(p + strlen(pat), ':'))) return 0;
    p += 1 + strspn(p + 1, " \t\r\n");
    if (*p != '[') return 0;
    p++;
    int n = 0;
    while (n < max) {
        p += strspn(p, " \t\r\n");
        if (*p == ']' || !*p) break;
        if (!strncmp(p, "true", 4)) { out[n++] = 1; p += 4; }
        else if (!strncmp(p, "false", 5)) { out[n++] = 0; p += 5; }
        else {
            char *end;
            out[n] = strtof(p, &end);
            if (end == p) break;
            n++;
            p = end;
        }
        p += strspn(p, " \t\r\n");
        if (*p == ',') p++;
    }
    return n;
}

static double json_num(const char *json, const char *key, double def) {
    char pat[64];
    snprintf(pat, sizeof(pat), "\"%s\"", key);
    const char *p = strstr(json, pat);
    if (!p || !(p = strchr(p + strlen(pat), ':'))) return def;
    p += 1 + strspn(p + 1, " \t\r\n");
    if (!strncmp(p, "null", 4)) return def;
    if (!strncmp(p, "true", 4)) return 1;
    if (!strncmp(p, "false", 5)) return 0;
    return atof(p);
}

/* Tensor as f32 (caller frees), checking its element count. */
static float *load_f32(safetensors_file_t *sf, const char *name, int64_t numel) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "Qwen-Image VAE: missing tensor %s\n", name);
        return NULL;
    }
    if (safetensor_numel(t) != numel) {
        fprintf(stderr, "Qwen-Image VAE: %s has %lld elements, expected %lld\n", name,
                (long long)safetensor_numel(t), (long long)numel);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

static int load_conv(safetensors_file_t *sf, qv_conv_t *c, const char *prefix,
                     int cin, int cout, int k) {
    char name[256];
    c->cin = cin;
    c->cout = cout;
    c->k = k;
    snprintf(name, sizeof(name), "%s.weight", prefix);
    c->w = load_f32(sf, name, (int64_t)cout * cin * k * k);
    snprintf(name, sizeof(name), "%s.bias", prefix);
    c->b = load_f32(sf, name, cout);
    return c->w && c->b;
}

static int load_res(safetensors_file_t *sf, qv_res_t *r, const char *prefix, int cin, int cout) {
    char name[256];
    r->cin = cin;
    r->cout = cout;
    snprintf(name, sizeof(name), "%s.norm1.gamma", prefix);
    r->gamma1 = load_f32(sf, name, cin);
    snprintf(name, sizeof(name), "%s.norm2.gamma", prefix);
    r->gamma2 = load_f32(sf, name, cout);
    snprintf(name, sizeof(name), "%s.conv1", prefix);
    int ok = r->gamma1 && r->gamma2 && load_conv(sf, &r->conv1, name, cin, cout, 3);
    snprintf(name, sizeof(name), "%s.conv2", prefix);
    ok = ok && load_conv(sf, &r->conv2, name, cout, cout, 3);
    if (ok && cin != cout) {
        snprintf(name, sizeof(name), "%s.conv_shortcut", prefix);
        ok = load_conv(sf, &r->shortcut, name, cin, cout, 1);
    }
    return ok;
}

/* The res blocks and the resampling convolution of a block. */
static int load_block(safetensors_file_t *sf, qv_block_t *b, const char *prefix,
                      const char *sampler) {
    char name[256];
    int ok = 1;
    for (int r = 0; ok && r < b->num_res; r++) {
        snprintf(name, sizeof(name), "%s.resnets.%d", prefix, r);
        ok = load_res(sf, &b->res[r], name, r ? b->cout : b->cin, b->cout);
    }
    if (ok && b->resample) {
        snprintf(name, sizeof(name), "%s.%s.resample.1", prefix, sampler);
        ok = load_conv(sf, &b->conv, name, b->cout, b->cout, 3);
    }
    return ok;
}

/* Mid block of a direction ("decoder" or "encoder"). */
static int load_mid(safetensors_file_t *sf, qv_net_t *n, const char *dir, int c) {
    char name[256];
    snprintf(name, sizeof(name), "%s.mid_block.resnets.0", dir);
    int ok = load_res(sf, &n->mid_res[0], name, c, c);
    snprintf(name, sizeof(name), "%s.mid_block.resnets.1", dir);
    ok = ok && load_res(sf, &n->mid_res[1], name, c, c);
    snprintf(name, sizeof(name), "%s.mid_block.attentions.0.to_qkv", dir);
    ok = ok && load_conv(sf, &n->attn.qkv, name, c, 3 * c, 1);
    snprintf(name, sizeof(name), "%s.mid_block.attentions.0.proj", dir);
    ok = ok && load_conv(sf, &n->attn.proj, name, c, c, 1);
    snprintf(name, sizeof(name), "%s.mid_block.attentions.0.norm.gamma", dir);
    return ok && (n->attn.gamma = load_f32(sf, name, c)) != NULL;
}

static void free_conv(qv_conv_t *c) {
    free(c->w);
    free(c->b);
}

static void free_res(qv_res_t *r) {
    free(r->gamma1);
    free(r->gamma2);
    free_conv(&r->conv1);
    free_conv(&r->conv2);
    free_conv(&r->shortcut);
}

static void free_net(qv_net_t *n) {
#ifdef IRIS_DEV
    dv_free(n);
#endif
    free_conv(&n->quant);
    free_conv(&n->conv_in);
    free_conv(&n->conv_out);
    free_res(&n->mid_res[0]);
    free_res(&n->mid_res[1]);
    free(n->attn.gamma);
    free_conv(&n->attn.qkv);
    free_conv(&n->attn.proj);
    for (int i = 0; i < n->num_blocks; i++) {
        for (int r = 0; r < n->block[i].num_res; r++) free_res(&n->block[i].res[r]);
        free_conv(&n->block[i].conv);
    }
    free(n->norm_out);
}

/* Load the decoder from <model_dir>/vae (config.json and the weights). */
iris_vae_qimage_t *iris_vae_load_qimage(const char *model_dir) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/vae/config.json", model_dir);
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "Qwen-Image VAE: missing %s\n", path);
        return NULL;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *cfg = malloc(size + 1);
    size_t got = fread(cfg, 1, size, f);
    cfg[got] = '\0';
    fclose(f);

    iris_vae_qimage_t *vae = calloc(1, sizeof(*vae));
    float dim_mult[QV_MAX_LEVELS], tdown[QV_MAX_LEVELS];
    int levels = json_array(cfg, "dim_mult", dim_mult, QV_MAX_LEVELS);
    int ntdown = json_array(cfg, "temperal_downsample", tdown, QV_MAX_LEVELS);
    int base = (int)json_num(cfg, "decoder_base_dim", json_num(cfg, "base_dim", 96));
    int num_res = (int)json_num(cfg, "num_res_blocks", 2) + 1;
    int residual = (int)json_num(cfg, "is_residual", 0);
    int patch = (int)json_num(cfg, "patch_size", 1);
    vae->z_dim = (int)json_num(cfg, "z_dim", 64);
    vae->out_channels = (int)json_num(cfg, "out_channels", 4);
    vae->in_channels = (int)json_num(cfg, "in_channels", 4);
    vae->enc_res = num_res - 1;
    vae->levels = levels;
    int enc_base = (int)json_num(cfg, "base_dim", 96);
    int nmean = json_array(cfg, "latents_mean", vae->mean, QV_MAX_CH);
    int nstd = json_array(cfg, "latents_std", vae->std, QV_MAX_CH);
    free(cfg);

    if (levels < 2 || ntdown != levels - 1 || !residual || patch > 1 ||
        num_res > QV_MAX_RES || vae->z_dim > QV_MAX_CH || nmean != vae->z_dim ||
        nstd != vae->z_dim || (vae->out_channels != 3 && vae->out_channels != 4) ||
        (vae->in_channels != 3 && vae->in_channels != 4)) {
        fprintf(stderr, "Qwen-Image VAE: unsupported config in %s\n", path);
        free(vae);
        return NULL;
    }

    snprintf(path, sizeof(path), "%s/vae/diffusion_pytorch_model.safetensors", model_dir);
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) {
        fprintf(stderr, "Qwen-Image VAE: cannot open %s\n", path);
        free(vae);
        return NULL;
    }

    /* Decoder widths: the top width, then dim_mult reversed. */
    int dims[QV_MAX_LEVELS + 1];
    dims[0] = base * (int)dim_mult[levels - 1];
    for (int i = 0; i < levels; i++) dims[i + 1] = base * (int)dim_mult[levels - 1 - i];
    int top = dims[0], z = vae->z_dim;

    /* Encoder widths, for its lazy load: one width per block input, then the
     * last block's output; the last block does not downsample. */
    vae->enc_dims[0] = enc_base;
    for (int i = 0; i < levels; i++) {
        vae->enc_dims[i + 1] = enc_base * (int)dim_mult[i];
        vae->enc_ft[i] = i < levels - 1 && tdown[i] != 0 ? 2 : 1;
    }
    vae->dir = strdup(model_dir);

    qv_net_t *n = &vae->dec;
    int ok = load_conv(sf, &n->quant, "post_quant_conv", z, z, 1) &&
             load_conv(sf, &n->conv_in, "decoder.conv_in", z, top, 3) &&
             load_mid(sf, n, "decoder", top);

    /* Up blocks; temperal_upsample is temperal_downsample reversed. */
    char name[256];
    for (int i = 0; ok && i < levels; i++) {
        qv_block_t *u = &n->block[n->num_blocks++];
        u->cin = dims[i];
        u->cout = dims[i + 1];
        u->num_res = num_res;
        u->resample = i != levels - 1;
        u->factor_t = (u->resample && tdown[levels - 2 - i] != 0) ? 2 : 1;
        if (u->resample && (u->cout * u->factor_t * 4) % u->cin) {
            fprintf(stderr, "Qwen-Image VAE: up block %d widths %d -> %d\n", i, u->cin, u->cout);
            ok = 0;
            break;
        }
        snprintf(name, sizeof(name), "decoder.up_blocks.%d", i);
        ok = load_block(sf, u, name, "upsampler");
    }
    if (ok) {
        int last = dims[levels];
        n->norm_out = load_f32(sf, "decoder.norm_out.gamma", last);
        ok = n->norm_out &&
             load_conv(sf, &n->conv_out, "decoder.conv_out", last, vae->out_channels, 3);
    }
    safetensors_close(sf);

    if (!ok) {
        iris_vae_free_qimage(vae);
        return NULL;
    }
    if (iris_verbose)
        fprintf(stderr, "  Qwen-Image VAE: z=%d base=%d levels=%d out=%d\n",
                vae->z_dim, base, levels, vae->out_channels);
    return vae;
}

/* Load the encoder from the file the decoder came from. */
static qv_net_t *load_encoder(const iris_vae_qimage_t *vae) {
    char path[4096];
    snprintf(path, sizeof(path), "%s/vae/diffusion_pytorch_model.safetensors", vae->dir);
    safetensors_file_t *sf = safetensors_open(path);
    if (!sf) {
        fprintf(stderr, "Qwen-Image VAE: cannot open %s\n", path);
        return NULL;
    }
    qv_net_t *n = calloc(1, sizeof(*n));
    const int *dims = vae->enc_dims;
    int top = dims[vae->levels], moments = 2 * vae->z_dim;
    int ok = load_conv(sf, &n->quant, "quant_conv", moments, moments, 1) &&
             load_conv(sf, &n->conv_in, "encoder.conv_in", vae->in_channels, dims[0], 3) &&
             load_mid(sf, n, "encoder", top);
    char name[256];
    for (int i = 0; ok && i < vae->levels; i++) {
        qv_block_t *b = &n->block[n->num_blocks++];
        b->cin = dims[i];
        b->cout = dims[i + 1];
        b->num_res = vae->enc_res;
        b->resample = i != vae->levels - 1;
        b->factor_t = vae->enc_ft[i];
        if (b->resample && (b->cin * b->factor_t * 4) % b->cout) {
            fprintf(stderr, "Qwen-Image VAE: down block %d widths %d -> %d\n", i, b->cin, b->cout);
            ok = 0;
            break;
        }
        snprintf(name, sizeof(name), "encoder.down_blocks.%d", i);
        ok = load_block(sf, b, name, "downsampler");
    }
    if (ok) {
        n->norm_out = load_f32(sf, "encoder.norm_out.gamma", top);
        ok = n->norm_out && load_conv(sf, &n->conv_out, "encoder.conv_out", top, moments, 3);
    }
    safetensors_close(sf);
    if (!ok) {
        free_net(n);
        free(n);
        return NULL;
    }
    return n;
}

static void free_encoder(iris_vae_qimage_t *vae) {
    if (!vae->enc) return;
    free_net(vae->enc);
    free(vae->enc);
    vae->enc = NULL;
}

/* ========================================================================
 * Encode
 * ======================================================================== */

float *iris_vae_encode_qimage(iris_vae_qimage_t *vae, const float *rgba, int h, int w) {
    int scale = 1 << (vae->levels - 1);
    if (h % scale || w % scale) {
        fprintf(stderr, "Qwen-Image VAE: %dx%d is not a multiple of %d\n", w, h, scale);
        return NULL;
    }
    if (!vae->enc && !(vae->enc = load_encoder(vae))) return NULL;

    float *z = NULL;
#ifdef IRIS_DEV
    if (iris_dev_available()) {
        z = encode_dev(vae, rgba, h, w);
        if (!z) fprintf(stderr, "Qwen-Image VAE: GPU encode failed, using the CPU\n");
    }
#endif
    if (!z) z = encode_host(vae->enc, rgba, h, w);

    /* The posterior's mode is its mean, the first z_dim channels. */
    size_t hw = (size_t)(h / scale) * (w / scale);
    for (int c = 0; c < vae->z_dim; c++)
        for (size_t p = 0; p < hw; p++)
            z[c * hw + p] = (z[c * hw + p] - vae->mean[c]) / vae->std[c];
    return z;
}

void iris_vae_release_qimage(iris_vae_qimage_t *vae) {
#ifdef IRIS_DEV
    dv_free(&vae->dec);
#endif
    free_encoder(vae);
}

void iris_vae_free_qimage(iris_vae_qimage_t *vae) {
    if (!vae) return;
    free_encoder(vae);
    free_net(&vae->dec);
    free(vae->dir);
    free(vae);
}
