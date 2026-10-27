/*
 * Iris VAE Implementation
 *
 * AutoencoderKLFlux2 - Variational Autoencoder for FLUX.2
 * Encodes images to latent space and decodes latents to images.
 *
 * Architecture:
 * - 32 latent channels (128 after patchification)
 * - 16x spatial compression
 * - Channel multipliers: [1, 2, 4, 4] -> [128, 256, 512, 512]
 * - GroupNorm (32 groups) + Swish activation
 */

#include "iris.h"
#include "iris_kernels.h"
#include "iris_safetensors.h"
#include <limits.h>
#ifdef USE_METAL
#include "iris_metal.h"
#endif
/* CUDA / ROCm device backend (Metal builds keep the native Metal VAE) */
#if defined(IRIS_DEV) && !defined(USE_METAL)
#define VAE_DEV
#include "iris_dev.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ========================================================================
 * VAE Data Structures
 * ======================================================================== */

/* Residual block weights */
typedef struct {
    float *norm1_weight, *norm1_bias;   /* [channels] */
    float *conv1_weight, *conv1_bias;   /* [out_ch, in_ch, 3, 3] */
    float *norm2_weight, *norm2_bias;   /* [channels] */
    float *conv2_weight, *conv2_bias;   /* [out_ch, out_ch, 3, 3] */
    float *skip_weight, *skip_bias;     /* [out_ch, in_ch, 1, 1] if in_ch != out_ch */
    int in_channels;
    int out_channels;
} vae_resblock_t;

/* Self-attention block weights */
typedef struct {
    float *norm_weight, *norm_bias;     /* [channels] */
    float *q_weight, *q_bias;           /* [channels, channels, 1, 1] */
    float *k_weight, *k_bias;           /* [channels, channels, 1, 1] */
    float *v_weight, *v_bias;           /* [channels, channels, 1, 1] */
    float *out_weight, *out_bias;       /* [channels, channels, 1, 1] */
    int channels;
} vae_attnblock_t;

/* Downsample block (stride-2 conv) */
typedef struct {
    float *conv_weight, *conv_bias;     /* [channels, channels, 3, 3] */
    int channels;
} vae_downsample_t;

/* Upsample block (nearest + conv) */
typedef struct {
    float *conv_weight, *conv_bias;     /* [channels, channels, 3, 3] */
    int channels;
} vae_upsample_t;

/* VAE context */
typedef struct iris_vae {
    /* Configuration */
    int z_channels;         /* 32 (Flux) or 16 (Z-Image) */
    int latent_channels;    /* z_channels * 4: 128 (Flux) or 64 (Z-Image) */
    int base_channels;      /* 128 */
    int ch_mult[4];         /* {1, 2, 4, 4} */
    int num_res_blocks;     /* 2 */
    int num_groups;         /* 32 */
    float eps;              /* 1e-4 (Flux batch_norm_eps) or 1e-6 (Z-Image) */
    float scaling_factor;   /* 0 = use batch norm (Flux), else Z-Image scaling */
    float shift_factor;     /* 0 = use batch norm (Flux), else Z-Image shift */

    /* Encoder weights */
    float *enc_conv_in_weight, *enc_conv_in_bias;   /* [128, 3, 3, 3] */

    /* Encoder down blocks: 4 levels, each with num_res_blocks + optional downsample */
    vae_resblock_t *enc_down_blocks;    /* 4 * 2 = 8 resblocks */
    vae_downsample_t *enc_downsample;   /* 3 downsamples (not at last level) */

    /* Encoder mid block */
    vae_resblock_t enc_mid_block1;
    vae_attnblock_t enc_mid_attn;
    vae_resblock_t enc_mid_block2;

    /* Encoder output */
    float *enc_norm_out_weight, *enc_norm_out_bias; /* [512] */
    float *enc_conv_out_weight, *enc_conv_out_bias; /* [64, 512, 3, 3] */

    /* Decoder weights */
    float *dec_conv_in_weight, *dec_conv_in_bias;   /* [512, z_channels, 3, 3] */

    /* Decoder mid block */
    vae_resblock_t dec_mid_block1;
    vae_attnblock_t dec_mid_attn;
    vae_resblock_t dec_mid_block2;

    /* Decoder up blocks: 4 levels, each with num_res_blocks+1 + optional upsample */
    vae_resblock_t *dec_up_blocks;      /* 4 * 3 = 12 resblocks */
    vae_upsample_t *dec_upsample;       /* 3 upsamples */

    /* Decoder output */
    float *dec_norm_out_weight, *dec_norm_out_bias; /* [128] */
    float *dec_conv_out_weight, *dec_conv_out_bias; /* [3, 128, 3, 3] */

    /* Normalization stats for latent space (Flux only, NULL for Z-Image) */
    float *bn_mean;         /* [latent_channels] */
    float *bn_var;          /* [latent_channels] */

    /* Post-quantization conv (1x1) - Flux only, NULL for Z-Image */
    float *quant_conv_weight;       /* [z_ch*2, z_ch*2, 1, 1] - encoder */
    float *quant_conv_bias;         /* [z_ch*2] */
    float *post_quant_conv_weight;  /* [z_ch, z_ch, 1, 1] - decoder */
    float *post_quant_conv_bias;    /* [z_ch] */

    /* Working memory (allocated for max image size) */
    int max_h, max_w;
    float *work1, *work2, *work3;
    size_t work_size;

    /* Device copy of the decoder (CUDA / ROCm), NULL until first used */
    struct vae_dev *dev;
} iris_vae_t;

/* Forward declarations */
void iris_vae_free(iris_vae_t *vae);

void (*iris_vae_memory_callback)(int width, int height, size_t bytes) = NULL;

/* Scratch contents do not survive an encode/decode call. Grow capacity between
 * calls, releasing old buffers first to avoid reserving both sizes at once. */
int iris_vae_reserve(iris_vae_t *vae, int batch, int H, int W) {
    if (!vae || batch <= 0 || H <= 0 || W <= 0 || H > 4096 || W > 4096)
        return 0;
    /* CPU kernels use int element counts. A decoder upsample can have twice
     * base_channels at full resolution, including on GPU fallback. */
    size_t max_pixels = (size_t)INT_MAX / (2 * vae->base_channels);
    if ((size_t)batch > max_pixels / H / W) return 0;
    size_t pixels = (size_t)batch * H * W;
    size_t bytes = 4 * (size_t)vae->base_channels * pixels * sizeof(float);
    if (bytes <= vae->work_size) return 1;

    if (iris_vae_memory_callback) iris_vae_memory_callback(W, H, 3 * bytes);
    free(vae->work1);
    free(vae->work2);
    free(vae->work3);
    vae->work1 = vae->work2 = vae->work3 = NULL;
    vae->work_size = 0;
    float *a = malloc(bytes), *b = malloc(bytes), *c = malloc(bytes);
    if (!a || !b || !c) {
        free(a);
        free(b);
        free(c);
        return 0;
    }
    vae->work1 = a;
    vae->work2 = b;
    vae->work3 = c;
    vae->work_size = bytes;
    return 1;
}
#ifdef VAE_DEV
static float *vae_encode_dev(iris_vae_t *vae, const float *img, int H, int W,
                             int *out_h, int *out_w);
#endif

/* ========================================================================
 * Helper Functions
 * ======================================================================== */

static void vae_conv2d(float *out, const float *in,
                       const float *weight, const float *bias,
                       int batch, int in_ch, int out_ch, int H, int W,
                       int kH, int kW, int stride, int padding) {
#ifdef USE_METAL
    if (!iris_metal_available()) {
        iris_metal_init();
    }
    if (iris_metal_available() &&
        iris_metal_conv2d(out, in, weight, bias,
                          batch, in_ch, out_ch, H, W,
                          kH, kW, stride, padding)) {
        return;
    }
#endif
    iris_conv2d(out, in, weight, bias, batch, in_ch, out_ch,
                H, W, kH, kW, stride, padding);
}

/* FLUX.2 VAE uses asymmetric padding for stride-2 downsampling convolutions:
 * pad right and bottom by 1, then do a VALID 3x3/stride-2 conv.
 *
 * This matches the reference implementation (e.g. diffusers' Downsample2D)
 * and avoids a ~7px top/left shift that shows up as a border in img2img. */
static void vae_pad_right_bottom(float *out, const float *in,
                                 int batch, int channels, int H, int W) {
    int Hp = H + 1;
    int Wp = W + 1;
    size_t in_plane = (size_t)H * (size_t)W;
    size_t out_plane = (size_t)Hp * (size_t)Wp;

    memset(out, 0, (size_t)batch * (size_t)channels * out_plane * sizeof(float));

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            const float *src = in + ((size_t)b * (size_t)channels + (size_t)c) * in_plane;
            float *dst = out + ((size_t)b * (size_t)channels + (size_t)c) * out_plane;
            for (int y = 0; y < H; y++) {
                memcpy(dst + (size_t)y * (size_t)Wp,
                       src + (size_t)y * (size_t)W,
                       (size_t)W * sizeof(float));
            }
        }
    }
}

/* Swish activation in-place */
static void swish_inplace(float *x, int n) {
    iris_silu(x, n);
}

/* ResNet residual block: norm -> swish -> conv -> norm -> swish -> conv,
 * with a skip connection that uses 1x1 conv when channels change.
 * This is the core building block of both the VAE encoder and decoder,
 * stacked at each resolution level to learn hierarchical features. */
static void resblock_forward(float *out, const float *x,
                             const vae_resblock_t *block,
                             float *work, int batch, int H, int W,
                             int num_groups, float eps) {
    int in_ch = block->in_channels;
    int out_ch = block->out_channels;
    int spatial = H * W;

    /* Shortcut/skip connection */
    if (in_ch != out_ch) {
        /* 1x1 conv for channel adjustment */
        vae_conv2d(out, x, block->skip_weight, block->skip_bias,
                    batch, in_ch, out_ch, H, W, 1, 1, 1, 0);
    } else {
        iris_copy(out, x, batch * in_ch * spatial);
    }

    /* Main path: norm1 -> swish -> conv1 -> norm2 -> swish -> conv2 */

    /* GroupNorm + Swish */
    iris_group_norm(work, x, block->norm1_weight, block->norm1_bias,
                    batch, in_ch, H, W, num_groups, eps);
    swish_inplace(work, batch * in_ch * spatial);

    /* Conv1: in_ch -> out_ch */
    float *conv1_out = work + batch * in_ch * spatial;
    vae_conv2d(conv1_out, work, block->conv1_weight, block->conv1_bias,
                batch, in_ch, out_ch, H, W, 3, 3, 1, 1);

    /* GroupNorm + Swish */
    iris_group_norm(work, conv1_out, block->norm2_weight, block->norm2_bias,
                    batch, out_ch, H, W, num_groups, eps);
    swish_inplace(work, batch * out_ch * spatial);

    /* Conv2: out_ch -> out_ch */
    vae_conv2d(conv1_out, work, block->conv2_weight, block->conv2_bias,
                batch, out_ch, out_ch, H, W, 3, 3, 1, 1);

    /* Add residual */
    iris_add_inplace(out, conv1_out, batch * out_ch * spatial);
}

/* Single-head self-attention over spatial dimensions. Reshapes [C,H,W]
 * to [C, H*W], computes full attention, then reshapes back. Used only in
 * the bottleneck (mid_block) of both encoder and decoder, where the spatial
 * resolution is small enough for O(n^2) attention to be tractable, giving
 * the model global spatial reasoning at the coarsest level.
 * Returns 0 on success, -1 on OOM. */
static int attnblock_forward(float *out, const float *x,
                             const vae_attnblock_t *block,
                             float *work, int batch, int H, int W,
                             int num_groups, float eps) {
    int ch = block->channels;
    int spatial = H * W;

    /* GroupNorm */
    iris_group_norm(work, x, block->norm_weight, block->norm_bias,
                    batch, ch, H, W, num_groups, eps);

    /* Project to Q, K, V using 1x1 convs */
    float *q = work + batch * ch * spatial;
    float *k = q + batch * ch * spatial;
    float *v = k + batch * ch * spatial;

    vae_conv2d(q, work, block->q_weight, block->q_bias,
                batch, ch, ch, H, W, 1, 1, 1, 0);
    vae_conv2d(k, work, block->k_weight, block->k_bias,
                batch, ch, ch, H, W, 1, 1, 1, 0);
    vae_conv2d(v, work, block->v_weight, block->v_bias,
                batch, ch, ch, H, W, 1, 1, 1, 0);

    /* Reshape: [B, C, H, W] -> [B, 1, HW, C] for attention */
    /* (We compute attention with heads=1 for simplicity) */
    float scale = 1.0f / sqrtf((float)ch);

    float *attn_out = v + batch * ch * spatial;

    /* Allocate attention work buffers once outside the batch loop */
    float *q_t = (float *)malloc(spatial * ch * sizeof(float));
    float *k_t = (float *)malloc(spatial * ch * sizeof(float));
    float *v_t = (float *)malloc(spatial * ch * sizeof(float));
    float *o_t = (float *)malloc(spatial * ch * sizeof(float));
    float *scores = (float *)malloc((size_t)spatial * spatial * sizeof(float));

    /* Check for allocation failures */
    if (!q_t || !k_t || !v_t || !o_t || !scores) {
        free(q_t);
        free(k_t);
        free(v_t);
        free(o_t);
        free(scores);
        return -1;  /* OOM */
    }

    for (int b = 0; b < batch; b++) {
        float *qb = q + b * ch * spatial;
        float *kb = k + b * ch * spatial;
        float *vb = v + b * ch * spatial;
        float *ob = attn_out + b * ch * spatial;

        /* Transpose [C, HW] -> [HW, C] */
        for (int c = 0; c < ch; c++) {
            for (int i = 0; i < spatial; i++) {
                q_t[i * ch + c] = qb[c * spatial + i] * scale;
                k_t[i * ch + c] = kb[c * spatial + i];
                v_t[i * ch + c] = vb[c * spatial + i];
            }
        }

        /* Q @ K^T using BLAS: [HW, C] @ [C, HW] -> [HW, HW] */
        iris_matmul_t(scores, q_t, k_t, spatial, ch, spatial);

        /* Softmax */
        iris_softmax(scores, spatial, spatial);

        /* scores @ V using BLAS: [HW, HW] @ [HW, C] -> [HW, C] */
        iris_matmul(o_t, scores, v_t, spatial, spatial, ch);

        /* Transpose output back [HW, C] -> [C, HW] */
        for (int c = 0; c < ch; c++) {
            for (int i = 0; i < spatial; i++) {
                ob[c * spatial + i] = o_t[i * ch + c];
            }
        }
    }

    free(q_t);
    free(k_t);
    free(v_t);
    free(o_t);
    free(scores);

    /* Project output */
    vae_conv2d(work, attn_out, block->out_weight, block->out_bias,
                batch, ch, ch, H, W, 1, 1, 1, 0);

    /* Add residual */
    iris_add(out, x, work, batch * ch * spatial);
    return 0;
}

/* ========================================================================
 * Encoder Forward Pass
 * ======================================================================== */

/* Turn the encoder output moments [B, 2 * z_ch, h, w] into the normalized,
 * patchified latent [B, latent_ch, h/2, w/2]: take the mean (first z_ch
 * channels), patchify 2x2, then batch norm (Flux) or (x - shift) * scale
 * (Z-Image). Returns a new buffer. */
static float *vae_finish_latent(iris_vae_t *vae, const float *moments, int batch,
                                int latent_h, int latent_w, int *out_h, int *out_w) {
    int z_ch = vae->z_channels * 2;
    int z_spatial = latent_h * latent_w;
    float *mean = (float *)malloc(batch * vae->z_channels * z_spatial * sizeof(float));
    for (int b = 0; b < batch; b++) {
        memcpy(mean + b * vae->z_channels * z_spatial,
               moments + b * z_ch * z_spatial,
               vae->z_channels * z_spatial * sizeof(float));
    }

    /* Patchify: [B, z_ch, H/8, W/8] -> [B, latent_ch, H/16, W/16] */
    int patch_h = latent_h / 2;
    int patch_w = latent_w / 2;
    int lat_ch = vae->latent_channels;
    int n = batch * lat_ch * patch_h * patch_w;
    float *latent = (float *)malloc(n * sizeof(float));
    iris_patchify(latent, mean, batch, vae->z_channels, latent_h, latent_w, 2);
    free(mean);

    /* Normalize latent space */
    if (vae->scaling_factor != 0.0f) {
        /* Z-Image: latent = (latent - shift) * scaling */
        for (int i = 0; i < n; i++)
            latent[i] = (latent[i] - vae->shift_factor) * vae->scaling_factor;
    } else {
        /* Flux: batch normalize */
        float *tmp = (float *)malloc(n * sizeof(float));
        iris_batch_norm(tmp, latent, vae->bn_mean, vae->bn_var, NULL, NULL,
                        batch, lat_ch, patch_h, patch_w, vae->eps);
        free(latent);
        latent = tmp;
    }

    *out_h = patch_h;
    *out_w = patch_w;
    return latent;
}

/* Encode an RGB image to latent space for img2img conditioning.
 * Runs the encoder CNN (down_blocks halve resolution 3 times for 8x compression),
 * takes the mean of the latent distribution, then patchifies and normalizes.
 * Flux uses batch normalization; Z-Image uses (latent - shift) * scale.
 * Output: [batch, latent_ch, H/16, W/16] where latent_ch is 128 (Flux) or
 * 64 (Z-Image) after patchification. */
float *iris_vae_encode(iris_vae_t *vae, const float *img,
                       int batch, int H, int W,
                       int *out_h, int *out_w) {
    if (!iris_vae_reserve(vae, batch, H, W)) return NULL;
    /*
     * Encoder path:
     * [B, 3, H, W] -> conv_in -> down_blocks -> mid_block -> norm -> conv_out
     * -> [B, 64, H/8, W/8] (32 mean + 32 logvar, use mean only)
     * -> patchify 2x2 -> [B, 128, H/16, W/16]
     * -> batch_norm
     */

#ifdef VAE_DEV
    if (iris_dev_available() && batch == 1) {
        float *latent = vae_encode_dev(vae, img, H, W, out_h, out_w);
        if (latent) return latent;
        fprintf(stderr, "VAE: GPU encode failed, using the CPU\n");
    }
#endif

    int ch_mult[4] = {1, 2, 4, 4};
    float *x = vae->work1;
    float *work = vae->work2;

    int cur_h = H, cur_w = W;

    /* Conv in: 3 -> 128 */
    vae_conv2d(x, img, vae->enc_conv_in_weight, vae->enc_conv_in_bias,
                batch, 3, vae->base_channels, H, W, 3, 3, 1, 1);

    int block_idx = 0;
    int down_idx = 0;
    int progress = 0;
    int total_blocks = 4 * vae->num_res_blocks + 3;  /* down resblocks + mid */

    /* Down blocks */
    for (int level = 0; level < 4; level++) {
        int ch_out = vae->base_channels * ch_mult[level];

        for (int r = 0; r < vae->num_res_blocks; r++) {
            vae_resblock_t *block = &vae->enc_down_blocks[block_idx++];
            resblock_forward(work, x, block, vae->work3,
                             batch, cur_h, cur_w, vae->num_groups, vae->eps);
            iris_copy(x, work, batch * ch_out * cur_h * cur_w);
            if (iris_vae_progress_callback)
                iris_vae_progress_callback(progress++, total_blocks);
        }

        /* Downsample (except last level) */
        if (level < 3) {
            vae_downsample_t *ds = &vae->enc_downsample[down_idx++];
            /* Asymmetric padding: pad right and bottom by 1.
             * Implemented explicitly to match training/reference impl. */
            float *padded = vae->work3;
            int padded_h = cur_h + 1;
            int padded_w = cur_w + 1;
            int new_h = (padded_h - 3) / 2 + 1;
            int new_w = (padded_w - 3) / 2 + 1;
            vae_pad_right_bottom(padded, x, batch, ch_out, cur_h, cur_w);
            vae_conv2d(work, padded, ds->conv_weight, ds->conv_bias,
                       batch, ch_out, ch_out, padded_h, padded_w, 3, 3, 2, 0);
            cur_h = new_h;
            cur_w = new_w;
            iris_copy(x, work, batch * ch_out * cur_h * cur_w);
        }
    }

    int mid_ch = vae->base_channels * ch_mult[3];  /* 512 */

    /* Mid block: resblock -> attn -> resblock */
    resblock_forward(work, x, &vae->enc_mid_block1, vae->work3,
                     batch, cur_h, cur_w, vae->num_groups, vae->eps);
    if (iris_vae_progress_callback)
        iris_vae_progress_callback(progress++, total_blocks);
    if (attnblock_forward(x, work, &vae->enc_mid_attn, vae->work3,
                          batch, cur_h, cur_w, vae->num_groups, vae->eps) < 0) {
        return NULL;  /* OOM in attention */
    }
    if (iris_vae_progress_callback)
        iris_vae_progress_callback(progress++, total_blocks);
    resblock_forward(work, x, &vae->enc_mid_block2, vae->work3,
                     batch, cur_h, cur_w, vae->num_groups, vae->eps);
    iris_copy(x, work, batch * mid_ch * cur_h * cur_w);
    if (iris_vae_progress_callback)
        iris_vae_progress_callback(progress++, total_blocks);

    /* Output: norm -> swish -> conv */
    iris_group_norm(work, x, vae->enc_norm_out_weight, vae->enc_norm_out_bias,
                    batch, mid_ch, cur_h, cur_w, vae->num_groups, vae->eps);
    swish_inplace(work, batch * mid_ch * cur_h * cur_w);

    /* Conv out: 512 -> 64 (32 mean + 32 logvar) */
    int z_ch = vae->z_channels * 2;  /* 64 */
    vae_conv2d(x, work, vae->enc_conv_out_weight, vae->enc_conv_out_bias,
                batch, mid_ch, z_ch, cur_h, cur_w, 3, 3, 1, 1);

    /* Quant conv: z_ch*2 -> z_ch*2 (1x1 conv) - Flux only */
    if (vae->quant_conv_weight) {
        vae_conv2d(work, x, vae->quant_conv_weight, vae->quant_conv_bias,
                   batch, z_ch, z_ch, cur_h, cur_w, 1, 1, 1, 0);
        iris_copy(x, work, batch * z_ch * cur_h * cur_w);
    }

    return vae_finish_latent(vae, x, batch, cur_h, cur_w, out_h, out_w);
}

/* ========================================================================
 * GPU-Resident Decoder
 * ======================================================================== */

#ifdef USE_METAL

/* GPU resblock: all operations on GPU, returns new tensor */
static iris_gpu_tensor_t resblock_forward_gpu(iris_gpu_tensor_t x,
                                               const vae_resblock_t *block,
                                               int batch, int H, int W,
                                               int num_groups, float eps) {
    int in_ch = block->in_channels;
    int out_ch = block->out_channels;
    int spatial = H * W;
    int n = batch * out_ch * spatial;

    /* Skip connection */
    iris_gpu_tensor_t skip;
    if (in_ch != out_ch) {
        skip = iris_gpu_conv2d_f32(x, block->skip_weight, block->skip_bias,
                                    batch, in_ch, out_ch, H, W, 1, 1, 1, 0);
    } else {
        skip = iris_gpu_tensor_alloc((size_t)n);
        iris_gpu_copy_f32(skip, x, (size_t)n);
    }
    if (!skip) return NULL;

    /* Main path: norm1 -> swish -> conv1 -> norm2 -> swish -> conv2 */
    iris_gpu_tensor_t work = iris_gpu_tensor_alloc((size_t)batch * in_ch * spatial);
    if (!work) { iris_gpu_tensor_free(skip); return NULL; }

    iris_gpu_group_norm_f32(work, x, block->norm1_weight, block->norm1_bias,
                             batch, in_ch, spatial, num_groups, eps);
    iris_gpu_swish_f32(work, work, batch * in_ch * spatial);

    iris_gpu_tensor_t conv1_out = iris_gpu_conv2d_f32(work, block->conv1_weight, block->conv1_bias,
                                                       batch, in_ch, out_ch, H, W, 3, 3, 1, 1);
    iris_gpu_tensor_free(work);
    if (!conv1_out) { iris_gpu_tensor_free(skip); return NULL; }

    work = iris_gpu_tensor_alloc((size_t)batch * out_ch * spatial);
    if (!work) { iris_gpu_tensor_free(skip); iris_gpu_tensor_free(conv1_out); return NULL; }

    iris_gpu_group_norm_f32(work, conv1_out, block->norm2_weight, block->norm2_bias,
                             batch, out_ch, spatial, num_groups, eps);
    iris_gpu_swish_f32(work, work, batch * out_ch * spatial);
    iris_gpu_tensor_free(conv1_out);

    conv1_out = iris_gpu_conv2d_f32(work, block->conv2_weight, block->conv2_bias,
                                     batch, out_ch, out_ch, H, W, 3, 3, 1, 1);
    iris_gpu_tensor_free(work);
    if (!conv1_out) { iris_gpu_tensor_free(skip); return NULL; }

    /* Residual: skip += conv_out */
    iris_gpu_add_f32(skip, skip, conv1_out, n);
    iris_gpu_tensor_free(conv1_out);

    return skip;
}

/* GPU-resident VAE decode.
 * Keeps all data on GPU, only syncs for mid-block attention (CPU) and final output.
 * Returns NULL on failure (caller falls back to CPU path). */
static iris_image *vae_decode_gpu(iris_vae_t *vae, const float *latent,
                                   int batch, int latent_h, int latent_w) {
    if (!iris_metal_available()) return NULL;

    int ch_mult[4] = {1, 2, 4, 4};

    /* Denormalize + unpatchify on CPU (small data, fast) */
    float *cpu_x = vae->work1;
    float *cpu_work = vae->work2;
    int lat_ch = vae->latent_channels;

    int z_spatial = latent_h * latent_w;
    iris_copy(cpu_x, latent, batch * lat_ch * z_spatial);

    if (vae->scaling_factor != 0.0f) {
        /* Z-Image: latent = latent / scaling + shift */
        int n = batch * lat_ch * z_spatial;
        for (int i = 0; i < n; i++)
            cpu_x[i] = cpu_x[i] / vae->scaling_factor + vae->shift_factor;
    } else {
        /* Flux: batch denormalize: x = x * sqrt(var + eps) + mean */
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < lat_ch; c++) {
                float mean = vae->bn_mean[c];
                float std = sqrtf(vae->bn_var[c] + vae->eps);
                for (int i = 0; i < z_spatial; i++) {
                    int idx = b * lat_ch * z_spatial + c * z_spatial + i;
                    cpu_x[idx] = cpu_x[idx] * std + mean;
                }
            }
        }
    }

    int unpatch_h = latent_h * 2;
    int unpatch_w = latent_w * 2;
    iris_unpatchify(cpu_work, cpu_x, batch, vae->z_channels, latent_h, latent_w, 2);
    iris_copy(cpu_x, cpu_work, batch * vae->z_channels * unpatch_h * unpatch_w);
    int cur_h = unpatch_h, cur_w = unpatch_w;

    /* Upload to GPU and start batch */
    size_t x_size = (size_t)batch * vae->z_channels * cur_h * cur_w;
    iris_gpu_tensor_t x = iris_gpu_tensor_create(cpu_x, x_size);
    if (!x) return NULL;

    iris_gpu_batch_begin();
    iris_gpu_tensor_t t;

    /* Post-quantization conv (1x1) - Flux only */
    if (vae->post_quant_conv_weight) {
        t = iris_gpu_conv2d_f32(x, vae->post_quant_conv_weight, vae->post_quant_conv_bias,
                                batch, vae->z_channels, vae->z_channels, cur_h, cur_w, 1, 1, 1, 0);
        iris_gpu_tensor_free(x);
        if (!t) { iris_gpu_batch_end(); return NULL; }
        x = t;
    }

    /* Conv in: z_channels -> 512 */
    int mid_ch = vae->base_channels * ch_mult[3];
    t = iris_gpu_conv2d_f32(x, vae->dec_conv_in_weight, vae->dec_conv_in_bias,
                             batch, vae->z_channels, mid_ch, cur_h, cur_w, 3, 3, 1, 1);
    iris_gpu_tensor_free(x);
    if (!t) { iris_gpu_batch_end(); return NULL; }
    x = t;

    /* Mid block: resblock1 */
    int progress = 0;
    int total_blocks = 3 + 4 * (vae->num_res_blocks + 1);

    t = resblock_forward_gpu(x, &vae->dec_mid_block1, batch, cur_h, cur_w, vae->num_groups, vae->eps);
    iris_gpu_tensor_free(x);
    if (!t) { iris_gpu_batch_end(); return NULL; }
    x = t;
    if (iris_vae_progress_callback) iris_vae_progress_callback(progress++, total_blocks);

    /* Mid block attention: sync to CPU, run attention, upload back */
    {
        size_t attn_size = (size_t)batch * mid_ch * cur_h * cur_w;
        float *cpu_attn_in = cpu_work;

        iris_gpu_batch_end();  /* Sync: execute everything queued so far */

        /* Download GPU tensor to CPU */
        iris_gpu_tensor_read(x, cpu_attn_in);

        /* Run attention on CPU (uses existing attnblock_forward) */
        float *cpu_attn_out = cpu_x;
        if (attnblock_forward(cpu_attn_out, cpu_attn_in, &vae->dec_mid_attn,
                               vae->work3, batch, cur_h, cur_w,
                               vae->num_groups, vae->eps) < 0) {
            iris_gpu_tensor_free(x);
            return NULL;
        }
        if (iris_vae_progress_callback) iris_vae_progress_callback(progress++, total_blocks);

        /* Upload result back to GPU */
        iris_gpu_tensor_free(x);
        x = iris_gpu_tensor_create(cpu_attn_out, attn_size);
        if (!x) return NULL;

        iris_gpu_batch_begin();  /* Start new batch for remaining work */
    }

    /* Mid block: resblock2 */
    t = resblock_forward_gpu(x, &vae->dec_mid_block2, batch, cur_h, cur_w, vae->num_groups, vae->eps);
    iris_gpu_tensor_free(x);
    if (!t) { iris_gpu_batch_end(); return NULL; }
    x = t;
    if (iris_vae_progress_callback) iris_vae_progress_callback(progress++, total_blocks);

    int block_idx = 0;
    int up_idx = 0;

    /* Up blocks (reverse order of channels) */
    for (int level = 3; level >= 0; level--) {
        int ch_out = vae->base_channels * ch_mult[level];

        for (int r = 0; r < vae->num_res_blocks + 1; r++) {
            vae_resblock_t *block = &vae->dec_up_blocks[block_idx++];
            t = resblock_forward_gpu(x, block, batch, cur_h, cur_w, vae->num_groups, vae->eps);
            iris_gpu_tensor_free(x);
            if (!t) { iris_gpu_batch_end(); return NULL; }
            x = t;
            if (iris_vae_progress_callback) iris_vae_progress_callback(progress++, total_blocks);
        }

        /* Upsample (except level 0) */
        if (level > 0) {
            vae_upsample_t *us = &vae->dec_upsample[up_idx++];
            int new_h = cur_h * 2;
            int new_w = cur_w * 2;

            t = iris_gpu_upsample_nearest_2x_f32(x, ch_out, cur_h, cur_w);
            iris_gpu_tensor_free(x);
            if (!t) { iris_gpu_batch_end(); return NULL; }

            x = iris_gpu_conv2d_f32(t, us->conv_weight, us->conv_bias,
                                     batch, ch_out, ch_out, new_h, new_w, 3, 3, 1, 1);
            iris_gpu_tensor_free(t);
            if (!x) { iris_gpu_batch_end(); return NULL; }

            cur_h = new_h;
            cur_w = new_w;
        }
    }

    /* Output: norm -> swish -> conv_out */
    int out_ch = vae->base_channels;  /* 128 */
    size_t final_size = (size_t)batch * out_ch * cur_h * cur_w;
    t = iris_gpu_tensor_alloc(final_size);
    if (!t) { iris_gpu_tensor_free(x); iris_gpu_batch_end(); return NULL; }
    iris_gpu_group_norm_f32(t, x, vae->dec_norm_out_weight, vae->dec_norm_out_bias,
                             batch, out_ch, cur_h * cur_w, vae->num_groups, vae->eps);
    iris_gpu_swish_f32(t, t, (int)final_size);
    iris_gpu_tensor_free(x);

    x = iris_gpu_conv2d_f32(t, vae->dec_conv_out_weight, vae->dec_conv_out_bias,
                             batch, out_ch, 3, cur_h, cur_w, 3, 3, 1, 1);
    iris_gpu_tensor_free(t);
    if (!x) { iris_gpu_batch_end(); return NULL; }

    /* Execute everything and read result */
    iris_gpu_batch_end();

    int H = cur_h;
    int W = cur_w;
    size_t rgb_size = (size_t)batch * 3 * H * W;
    float *rgb = (float *)malloc(rgb_size * sizeof(float));
    if (!rgb) { iris_gpu_tensor_free(x); return NULL; }
    iris_gpu_tensor_read(x, rgb);
    iris_gpu_tensor_free(x);

    /* Convert to image */
    iris_image *img = iris_image_create(W, H, 3);
    if (!img) { free(rgb); return NULL; }

    for (int y = 0; y < H; y++) {
        for (int c = 0; c < W; c++) {
            for (int ch = 0; ch < 3; ch++) {
                float val = rgb[ch * H * W + y * W + c];
                val = (val + 1.0f) * 0.5f;
                val = val * 255.0f;
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                img->data[(y * W + c) * 3 + ch] = (uint8_t)(val + 0.5f);
            }
        }
    }

    free(rgb);
    return img;
}

#endif /* USE_METAL */

/* ========================================================================
 * Decoder Forward Pass
 * ======================================================================== */

/* Decode latents back to an RGB image. Reverses the encode normalization
 * (Flux: batch denorm, Z-Image: x/scale + shift), unpatchifies, then runs
 * the decoder CNN (up_blocks double resolution 3 times). Converts the
 * float output to uint8 RGB. Tries GPU-resident decode first for speed,
 * falling back to CPU on failure. */
/* Undo the latent normalization (batch norm for Flux, scaling/shift for
 * Z-Image) and unpatchify: [B, latent_ch, h, w] -> out [B, z_ch, 2h, 2w].
 * tmp must hold B * latent_ch * h * w floats. */
static void vae_prepare_latent(iris_vae_t *vae, float *out, float *tmp, const float *latent,
                               int batch, int latent_h, int latent_w) {
    int lat_ch = vae->latent_channels;
    int z_spatial = latent_h * latent_w;
    iris_copy(tmp, latent, batch * lat_ch * z_spatial);

    if (vae->scaling_factor != 0.0f) {
        /* Z-Image: latent = latent / scaling + shift */
        int n = batch * lat_ch * z_spatial;
        for (int i = 0; i < n; i++)
            tmp[i] = tmp[i] / vae->scaling_factor + vae->shift_factor;
    } else {
        /* Flux: batch denormalize: x = x * sqrt(var + eps) + mean */
        for (int b = 0; b < batch; b++) {
            for (int c = 0; c < lat_ch; c++) {
                float mean = vae->bn_mean[c];
                float std = sqrtf(vae->bn_var[c] + vae->eps);
                for (int i = 0; i < z_spatial; i++) {
                    int idx = b * lat_ch * z_spatial + c * z_spatial + i;
                    tmp[idx] = tmp[idx] * std + mean;
                }
            }
        }
    }
    iris_unpatchify(out, tmp, batch, vae->z_channels, latent_h, latent_w, 2);
}

#ifdef VAE_DEV
/* ========================================================================
 * GPU Decoder (CUDA / ROCm)
 *
 * Activations are channels-last f32 images [h * w, c]: 3x3 convolutions
 * become im2col + one GEMM with the pixels as rows, and the final output
 * is already interleaved RGB.
 * ======================================================================== */

typedef struct {
    float *norm1_w, *norm1_b, *norm2_w, *norm2_b;   /* [channels] */
    uint16_t *conv1_w, *conv2_w, *skip_w;           /* bf16 [cout, k*k*cin] */
    float *conv1_b, *conv2_b, *skip_b;
    int cin, cout;
} vae_dev_resblock_t;

typedef struct {
    float *norm_w, *norm_b;             /* [c] */
    uint16_t *qkv_w, *out_w;            /* [3c, c] (q, k, v fused), [c, c] */
    float *qkv_b, *out_b;
} vae_dev_attn_t;

struct vae_dev {
    /* Decoder */
    uint16_t *post_quant_w, *conv_in_w, *conv_out_w;
    float *post_quant_b, *conv_in_b, *conv_out_b;
    vae_dev_resblock_t mid1, mid2;
    vae_dev_resblock_t *up;             /* [4 * (num_res_blocks + 1)] */
    uint16_t *upsample_w[3];
    float *upsample_b[3];
    vae_dev_attn_t attn;
    float *norm_out_w, *norm_out_b;

    /* Encoder, uploaded on first use */
    int enc_loaded;
    uint16_t *enc_conv_in_w, *enc_conv_out_w, *quant_w;
    float *enc_conv_in_b, *enc_conv_out_b, *quant_b;
    vae_dev_resblock_t enc_mid1, enc_mid2;
    vae_dev_resblock_t *down;           /* [4 * num_res_blocks] */
    uint16_t *downsample_w[3];
    float *downsample_b[3];
    vae_dev_attn_t enc_attn;
    float *enc_norm_out_w, *enc_norm_out_b;

    /* Activation buffers, sized for the last decoded image */
    size_t buf_elems;
    float *buf[4];
    size_t attn_elems;
    uint16_t *attn_bf16;                /* [hw, 3c] q, k, v + [hw, c] output */
};

static uint16_t f32_to_bf16_host(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

static float *vdev_f32(const float *src, size_t n) {
    if (!src) return NULL;
    float *d = iris_dev_alloc(n * sizeof(float));
    if (d) iris_dev_upload(d, src, n * sizeof(float));
    return d;
}

/* Upload rows of bf16 converted from f32, concatenating parts that each
 * provide the given number of elements. */
static uint16_t *vdev_bf16_rows(const float *const *parts, const size_t *counts, int nparts) {
    size_t total = 0;
    for (int i = 0; i < nparts; i++) total += counts[i];
    uint16_t *tmp = malloc(total * sizeof(uint16_t));
    if (!tmp) return NULL;
    size_t off = 0;
    for (int i = 0; i < nparts; i++)
        for (size_t j = 0; j < counts[i]; j++) tmp[off++] = f32_to_bf16_host(parts[i][j]);
    uint16_t *d = iris_dev_alloc(total * sizeof(uint16_t));
    if (d) iris_dev_upload(d, tmp, total * sizeof(uint16_t));
    free(tmp);
    return d;
}

/* Convolution weight [cout, cin, k, k] (PyTorch layout) to the im2col
 * layout [cout, (ky, kx, cin)] in bf16. */
static uint16_t *vdev_conv_w(const float *w, int cout, int cin, int k) {
    if (!w) return NULL;
    size_t n = (size_t)cout * cin * k * k;
    float *re = malloc(n * sizeof(float));
    if (!re) return NULL;
    for (int o = 0; o < cout; o++)
        for (int c = 0; c < cin; c++)
            for (int t = 0; t < k * k; t++)
                re[((size_t)o * k * k + t) * cin + c] = w[((size_t)o * cin + c) * k * k + t];
    const float *parts[1] = {re};
    uint16_t *d = vdev_bf16_rows(parts, &n, 1);
    free(re);
    return d;
}

static int vdev_load_resblock(vae_dev_resblock_t *db, const vae_resblock_t *b) {
    db->cin = b->in_channels;
    db->cout = b->out_channels;
    db->norm1_w = vdev_f32(b->norm1_weight, db->cin);
    db->norm1_b = vdev_f32(b->norm1_bias, db->cin);
    db->norm2_w = vdev_f32(b->norm2_weight, db->cout);
    db->norm2_b = vdev_f32(b->norm2_bias, db->cout);
    db->conv1_w = vdev_conv_w(b->conv1_weight, db->cout, db->cin, 3);
    db->conv1_b = vdev_f32(b->conv1_bias, db->cout);
    db->conv2_w = vdev_conv_w(b->conv2_weight, db->cout, db->cout, 3);
    db->conv2_b = vdev_f32(b->conv2_bias, db->cout);
    int ok = db->norm1_w && db->norm1_b && db->norm2_w && db->norm2_b &&
             db->conv1_w && db->conv1_b && db->conv2_w && db->conv2_b;
    if (db->cin != db->cout) {
        db->skip_w = vdev_conv_w(b->skip_weight, db->cout, db->cin, 1);
        db->skip_b = vdev_f32(b->skip_bias, db->cout);
        ok &= db->skip_w && db->skip_b;
    }
    return ok;
}

static void vdev_free_resblock(vae_dev_resblock_t *db) {
    iris_dev_free(db->norm1_w);
    iris_dev_free(db->norm1_b);
    iris_dev_free(db->norm2_w);
    iris_dev_free(db->norm2_b);
    iris_dev_free(db->conv1_w);
    iris_dev_free(db->conv1_b);
    iris_dev_free(db->conv2_w);
    iris_dev_free(db->conv2_b);
    iris_dev_free(db->skip_w);
    iris_dev_free(db->skip_b);
}

static int vdev_load_attn(vae_dev_attn_t *da, const vae_attnblock_t *a) {
    size_t cc = (size_t)a->channels * a->channels;
    const float *qkv_w[3] = {a->q_weight, a->k_weight, a->v_weight};
    size_t qkv_n[3] = {cc, cc, cc};
    const float *out_w[1] = {a->out_weight};
    da->norm_w = vdev_f32(a->norm_weight, a->channels);
    da->norm_b = vdev_f32(a->norm_bias, a->channels);
    da->qkv_w = vdev_bf16_rows(qkv_w, qkv_n, 3);
    da->out_w = vdev_bf16_rows(out_w, &cc, 1);
    float *qkv_b = malloc(3 * a->channels * sizeof(float));
    if (qkv_b) {
        memcpy(qkv_b, a->q_bias, a->channels * sizeof(float));
        memcpy(qkv_b + a->channels, a->k_bias, a->channels * sizeof(float));
        memcpy(qkv_b + 2 * a->channels, a->v_bias, a->channels * sizeof(float));
        da->qkv_b = vdev_f32(qkv_b, 3 * a->channels);
        free(qkv_b);
    }
    da->out_b = vdev_f32(a->out_bias, a->channels);
    return da->norm_w && da->norm_b && da->qkv_w && da->out_w && da->qkv_b && da->out_b;
}

static void vdev_free_attn(vae_dev_attn_t *da) {
    iris_dev_free(da->norm_w);
    iris_dev_free(da->norm_b);
    iris_dev_free(da->qkv_w);
    iris_dev_free(da->qkv_b);
    iris_dev_free(da->out_w);
    iris_dev_free(da->out_b);
}

static void vdev_free_buffers(struct vae_dev *d) {
    for (int i = 0; i < 4; i++) {
        iris_dev_free(d->buf[i]);
        d->buf[i] = NULL;
    }
    iris_dev_free(d->attn_bf16);
    d->attn_bf16 = NULL;
    d->buf_elems = d->attn_elems = 0;
}

static void vdev_free(iris_vae_t *vae) {
    struct vae_dev *d = vae->dev;
    if (!d) return;
    iris_dev_free(d->post_quant_w);
    iris_dev_free(d->post_quant_b);
    iris_dev_free(d->conv_in_w);
    iris_dev_free(d->conv_in_b);
    iris_dev_free(d->conv_out_w);
    iris_dev_free(d->conv_out_b);
    vdev_free_resblock(&d->mid1);
    vdev_free_resblock(&d->mid2);
    if (d->up) {
        for (int i = 0; i < 4 * (vae->num_res_blocks + 1); i++) vdev_free_resblock(&d->up[i]);
        free(d->up);
    }
    for (int i = 0; i < 3; i++) {
        iris_dev_free(d->upsample_w[i]);
        iris_dev_free(d->upsample_b[i]);
    }
    vdev_free_attn(&d->attn);
    iris_dev_free(d->norm_out_w);
    iris_dev_free(d->norm_out_b);

    iris_dev_free(d->enc_conv_in_w);
    iris_dev_free(d->enc_conv_in_b);
    iris_dev_free(d->enc_conv_out_w);
    iris_dev_free(d->enc_conv_out_b);
    iris_dev_free(d->quant_w);
    iris_dev_free(d->quant_b);
    vdev_free_resblock(&d->enc_mid1);
    vdev_free_resblock(&d->enc_mid2);
    if (d->down) {
        for (int i = 0; i < 4 * vae->num_res_blocks; i++) vdev_free_resblock(&d->down[i]);
        free(d->down);
    }
    for (int i = 0; i < 3; i++) {
        iris_dev_free(d->downsample_w[i]);
        iris_dev_free(d->downsample_b[i]);
    }
    vdev_free_attn(&d->enc_attn);
    iris_dev_free(d->enc_norm_out_w);
    iris_dev_free(d->enc_norm_out_b);
    vdev_free_buffers(d);
    free(d);
    vae->dev = NULL;
}

static int vdev_load(iris_vae_t *vae) {
    struct vae_dev *d = calloc(1, sizeof(*d));
    if (!d) return 0;
    vae->dev = d;
    int z = vae->z_channels, mid = vae->base_channels * vae->ch_mult[3];
    int base = vae->base_channels;
    int ok = 1;

    if (vae->post_quant_conv_weight) {
        d->post_quant_w = vdev_conv_w(vae->post_quant_conv_weight, z, z, 1);
        d->post_quant_b = vdev_f32(vae->post_quant_conv_bias, z);
        ok &= d->post_quant_w && d->post_quant_b;
    }
    d->conv_in_w = vdev_conv_w(vae->dec_conv_in_weight, mid, z, 3);
    d->conv_in_b = vdev_f32(vae->dec_conv_in_bias, mid);
    ok &= d->conv_in_w && d->conv_in_b;
    ok &= vdev_load_resblock(&d->mid1, &vae->dec_mid_block1);
    ok &= vdev_load_resblock(&d->mid2, &vae->dec_mid_block2);

    ok &= vdev_load_attn(&d->attn, &vae->dec_mid_attn);

    int nup = 4 * (vae->num_res_blocks + 1);
    d->up = calloc(nup, sizeof(vae_dev_resblock_t));
    ok &= d->up != NULL;
    for (int i = 0; ok && i < nup; i++) ok &= vdev_load_resblock(&d->up[i], &vae->dec_up_blocks[i]);
    for (int i = 0; ok && i < 3; i++) {
        int c = vae->dec_upsample[i].channels;
        d->upsample_w[i] = vdev_conv_w(vae->dec_upsample[i].conv_weight, c, c, 3);
        d->upsample_b[i] = vdev_f32(vae->dec_upsample[i].conv_bias, c);
        ok &= d->upsample_w[i] && d->upsample_b[i];
    }
    d->norm_out_w = vdev_f32(vae->dec_norm_out_weight, base);
    d->norm_out_b = vdev_f32(vae->dec_norm_out_bias, base);
    d->conv_out_w = vdev_conv_w(vae->dec_conv_out_weight, 3, base, 3);
    d->conv_out_b = vdev_f32(vae->dec_conv_out_bias, 3);
    ok &= d->norm_out_w && d->norm_out_b && d->conv_out_w && d->conv_out_b;

    if (!ok) vdev_free(vae);
    return ok;
}

/* Largest activation (in floats) of the encoder / decoder for an image of
 * h8 x w8 pixels at the latent resolution (1/8 of the image). */
static size_t vdev_largest_tensor(iris_vae_t *vae, int h8, int w8) {
    int mid = vae->base_channels * vae->ch_mult[3];
    size_t need = (size_t)h8 * w8 * 3 * mid;   /* attention q, k, v */
    size_t hw = (size_t)h8 * w8;
    for (int level = 3; level >= 0; level--) {
        size_t c = (size_t)vae->base_channels * vae->ch_mult[level];
        if (hw * c > need) need = hw * c;
        if (level > 0) {
            hw *= 4;
            if (hw * c > need) need = hw * c;
        }
    }
    return need;
}

/* Size the activation buffers for an image of h8 x w8 latent pixels. */
static int vdev_ensure_buffers(iris_vae_t *vae, int h8, int w8) {
    struct vae_dev *d = vae->dev;
    int mid = vae->base_channels * vae->ch_mult[3];
    size_t attn = (size_t)h8 * w8 * 4 * mid;
    size_t need = vdev_largest_tensor(vae, h8, w8);
    if (need <= d->buf_elems && attn <= d->attn_elems) return 1;
    vdev_free_buffers(d);
    for (int i = 0; i < 4; i++) {
        d->buf[i] = iris_dev_alloc(need * sizeof(float));
        if (!d->buf[i]) {
            vdev_free_buffers(d);
            return 0;
        }
    }
    d->attn_bf16 = iris_dev_alloc(attn * sizeof(uint16_t));
    if (!d->attn_bf16) {
        vdev_free_buffers(d);
        return 0;
    }
    d->buf_elems = need;
    d->attn_elems = attn;
    return 1;
}

/* out = x + conv2(silu(gn2(conv1(silu(gn1(x)))))), with a 1x1 conv on the
 * skip path when the channel count changes. */
static void vdev_resblock(const vae_dev_resblock_t *b, float *out, const float *x,
                          float *t1, float *t2, int h, int w, int groups, float eps) {
    size_t hw = (size_t)h * w;
    iris_dev_group_norm(t1, x, b->norm1_w, b->norm1_b, hw, b->cin, groups, eps, 1);
    iris_dev_conv2d(t2, t1, b->conv1_w, b->conv1_b, h, w, b->cin, b->cout, 3, 1, 1, 1);
    iris_dev_group_norm(t1, t2, b->norm2_w, b->norm2_b, hw, b->cout, groups, eps, 1);
    iris_dev_conv2d(out, t1, b->conv2_w, b->conv2_b, h, w, b->cout, b->cout, 3, 1, 1, 1);
    if (b->skip_w) {
        iris_dev_conv2d(t2, x, b->skip_w, b->skip_b, h, w, b->cin, b->cout, 1, 1, 0, 0);
        iris_dev_add(out, t2, hw * b->cout);
    } else {
        iris_dev_add(out, x, hw * b->cout);
    }
}

/* Single-head self-attention over the pixels, plus the residual. */
static void vdev_attention(struct vae_dev *d, const vae_dev_attn_t *a, float *out,
                           const float *x, float *t1, float *t2, int hw, int c,
                           int groups, float eps) {
    uint16_t *qkv = d->attn_bf16, *o = qkv + (size_t)hw * 3 * c;
    iris_dev_group_norm(t1, x, a->norm_w, a->norm_b, hw, c, groups, eps, 0);
    iris_dev_conv2d(t2, t1, a->qkv_w, a->qkv_b, hw, 1, c, 3 * c, 1, 1, 0, 0);
    iris_dev_f32_to_bf16(qkv, t2, (size_t)hw * 3 * c);
    iris_dev_attention(o, c, qkv, qkv + c, qkv + 2 * c, 3 * c, hw, hw, 1, 1, c,
                       1.0f / sqrtf((float)c), 0, NULL);
    iris_dev_linear_f32(out, o, a->out_w, hw, c, c);
    iris_dev_add_bias(out, a->out_b, hw, c);
    iris_dev_add(out, x, (size_t)hw * c);
}

static iris_image *vae_decode_dev(iris_vae_t *vae, const float *latent,
                                  int latent_h, int latent_w) {
    if (!vae->dev && !vdev_load(vae)) return NULL;
    int h = latent_h * 2, w = latent_w * 2;   /* unpatchified latent size */
    if (!vdev_ensure_buffers(vae, h, w)) return NULL;
    struct vae_dev *d = vae->dev;
    int z = vae->z_channels, groups = vae->num_groups;
    int mid = vae->base_channels * vae->ch_mult[3];
    float eps = vae->eps;
    float *x = d->buf[0], *y = d->buf[1], *t1 = d->buf[2], *t2 = d->buf[3], *tmp;
#define VDEV_SWAP() (tmp = x, x = y, y = tmp)

    /* Latent preparation on the host, uploaded channels-last. */
    size_t n = (size_t)z * h * w;
    float *chw = malloc(n * sizeof(float));
    float *scratch = malloc((size_t)vae->latent_channels * latent_h * latent_w * sizeof(float));
    float *hwc = malloc(n * sizeof(float));
    if (!chw || !scratch || !hwc) {
        free(chw);
        free(scratch);
        free(hwc);
        return NULL;
    }
    vae_prepare_latent(vae, chw, scratch, latent, 1, latent_h, latent_w);
    for (int c = 0; c < z; c++)
        for (int p = 0; p < h * w; p++) hwc[(size_t)p * z + c] = chw[(size_t)c * h * w + p];
    iris_dev_upload(x, hwc, n * sizeof(float));
    free(chw);
    free(scratch);
    free(hwc);

    if (d->post_quant_w) {
        iris_dev_conv2d(y, x, d->post_quant_w, d->post_quant_b, h, w, z, z, 1, 1, 0, 0);
        VDEV_SWAP();
    }
    iris_dev_conv2d(y, x, d->conv_in_w, d->conv_in_b, h, w, z, mid, 3, 1, 1, 1);
    VDEV_SWAP();

    /* Mid block */
    int progress = 0, total_blocks = 3 + 4 * (vae->num_res_blocks + 1);
    vdev_resblock(&d->mid1, y, x, t1, t2, h, w, groups, eps);
    VDEV_SWAP();
    vdev_attention(d, &d->attn, y, x, t1, t2, h * w, mid, groups, eps);
    VDEV_SWAP();
    vdev_resblock(&d->mid2, y, x, t1, t2, h, w, groups, eps);
    VDEV_SWAP();
    progress += 3;
    if (iris_vae_progress_callback) iris_vae_progress_callback(progress - 1, total_blocks);

    /* Up blocks */
    int block = 0;
    for (int level = 3; level >= 0; level--) {
        int c = vae->base_channels * vae->ch_mult[level];
        for (int r = 0; r < vae->num_res_blocks + 1; r++) {
            vdev_resblock(&d->up[block++], y, x, t1, t2, h, w, groups, eps);
            VDEV_SWAP();
            if (iris_vae_progress_callback) iris_vae_progress_callback(progress++, total_blocks);
        }
        if (level > 0) {
            iris_dev_upsample2x(y, x, h, w, c);
            h *= 2;
            w *= 2;
            iris_dev_conv2d(x, y, d->upsample_w[3 - level], d->upsample_b[3 - level],
                            h, w, c, c, 3, 1, 1, 1);
        }
    }
#undef VDEV_SWAP

    /* Output: norm -> swish -> conv to RGB */
    iris_dev_group_norm(y, x, d->norm_out_w, d->norm_out_b, (size_t)h * w,
                        vae->base_channels, groups, eps, 1);
    iris_dev_conv2d(t1, y, d->conv_out_w, d->conv_out_b, h, w, vae->base_channels, 3, 3, 1, 1, 1);
    float *rgb = malloc((size_t)h * w * 3 * sizeof(float));
    iris_image *img = rgb ? iris_image_create(w, h, 3) : NULL;
    if (!img) {
        free(rgb);
        return NULL;
    }
    iris_dev_download(rgb, t1, (size_t)h * w * 3 * sizeof(float));
    for (size_t i = 0; i < (size_t)h * w * 3; i++) {
        float val = (rgb[i] + 1.0f) * 0.5f * 255.0f;
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        img->data[i] = (uint8_t)(val + 0.5f);
    }
    free(rgb);
    return img;
}
static int vdev_load_encoder(iris_vae_t *vae) {
    struct vae_dev *d = vae->dev;
    int base = vae->base_channels, mid = base * vae->ch_mult[3], z2 = vae->z_channels * 2;
    int ok = 1;

    d->enc_conv_in_w = vdev_conv_w(vae->enc_conv_in_weight, base, 3, 3);
    d->enc_conv_in_b = vdev_f32(vae->enc_conv_in_bias, base);
    ok &= d->enc_conv_in_w && d->enc_conv_in_b;
    int ndown = 4 * vae->num_res_blocks;
    d->down = calloc(ndown, sizeof(vae_dev_resblock_t));
    ok &= d->down != NULL;
    for (int i = 0; ok && i < ndown; i++) ok &= vdev_load_resblock(&d->down[i], &vae->enc_down_blocks[i]);
    for (int i = 0; ok && i < 3; i++) {
        int c = vae->enc_downsample[i].channels;
        d->downsample_w[i] = vdev_conv_w(vae->enc_downsample[i].conv_weight, c, c, 3);
        d->downsample_b[i] = vdev_f32(vae->enc_downsample[i].conv_bias, c);
        ok &= d->downsample_w[i] && d->downsample_b[i];
    }
    ok &= vdev_load_resblock(&d->enc_mid1, &vae->enc_mid_block1);
    ok &= vdev_load_resblock(&d->enc_mid2, &vae->enc_mid_block2);
    ok &= vdev_load_attn(&d->enc_attn, &vae->enc_mid_attn);
    d->enc_norm_out_w = vdev_f32(vae->enc_norm_out_weight, mid);
    d->enc_norm_out_b = vdev_f32(vae->enc_norm_out_bias, mid);
    d->enc_conv_out_w = vdev_conv_w(vae->enc_conv_out_weight, z2, mid, 3);
    d->enc_conv_out_b = vdev_f32(vae->enc_conv_out_bias, z2);
    ok &= d->enc_norm_out_w && d->enc_norm_out_b && d->enc_conv_out_w && d->enc_conv_out_b;
    if (vae->quant_conv_weight) {
        d->quant_w = vdev_conv_w(vae->quant_conv_weight, z2, z2, 1);
        d->quant_b = vdev_f32(vae->quant_conv_bias, z2);
        ok &= d->quant_w && d->quant_b;
    }
    d->enc_loaded = ok;
    return ok;
}

/* GPU encoder, same structure as iris_vae_encode(). img: [3, H, W]. */
static float *vae_encode_dev(iris_vae_t *vae, const float *img, int H, int W,
                             int *out_h, int *out_w) {
    if (!vae->dev && !vdev_load(vae)) return NULL;
    if (!vae->dev->enc_loaded && !vdev_load_encoder(vae)) return NULL;
    if (!vdev_ensure_buffers(vae, H / 8, W / 8)) return NULL;
    struct vae_dev *d = vae->dev;
    int groups = vae->num_groups, base = vae->base_channels;
    int mid = base * vae->ch_mult[3], z2 = vae->z_channels * 2;
    float eps = vae->eps;
    float *x = d->buf[0], *y = d->buf[1], *t1 = d->buf[2], *t2 = d->buf[3], *tmp;
#define VDEV_SWAP() (tmp = x, x = y, y = tmp)

    /* Upload channels-last. */
    size_t hw = (size_t)H * W;
    float *hwc = malloc(hw * 3 * sizeof(float));
    if (!hwc) return NULL;
    for (int c = 0; c < 3; c++)
        for (size_t p = 0; p < hw; p++) hwc[p * 3 + c] = img[c * hw + p];
    iris_dev_upload(x, hwc, hw * 3 * sizeof(float));
    free(hwc);

    int h = H, w = W;
    iris_dev_conv2d(y, x, d->enc_conv_in_w, d->enc_conv_in_b, h, w, 3, base, 3, 1, 1, 1);
    VDEV_SWAP();

    int block = 0, progress = 0, total_blocks = 4 * vae->num_res_blocks + 3;
    for (int level = 0; level < 4; level++) {
        int c = base * vae->ch_mult[level];
        for (int r = 0; r < vae->num_res_blocks; r++) {
            vdev_resblock(&d->down[block++], y, x, t1, t2, h, w, groups, eps);
            VDEV_SWAP();
            if (iris_vae_progress_callback) iris_vae_progress_callback(progress++, total_blocks);
        }
        if (level < 3) {
            /* Pad right and bottom by 1, then a valid 3x3 / stride 2 conv. */
            iris_dev_conv2d(y, x, d->downsample_w[level], d->downsample_b[level],
                            h, w, c, c, 3, 2, 0, 1);
            h = (h + 1 - 3) / 2 + 1;
            w = (w + 1 - 3) / 2 + 1;
            VDEV_SWAP();
        }
    }

    vdev_resblock(&d->enc_mid1, y, x, t1, t2, h, w, groups, eps);
    VDEV_SWAP();
    vdev_attention(d, &d->enc_attn, y, x, t1, t2, h * w, mid, groups, eps);
    VDEV_SWAP();
    vdev_resblock(&d->enc_mid2, y, x, t1, t2, h, w, groups, eps);
    VDEV_SWAP();
    if (iris_vae_progress_callback) iris_vae_progress_callback(total_blocks - 1, total_blocks);

    iris_dev_group_norm(y, x, d->enc_norm_out_w, d->enc_norm_out_b, (size_t)h * w, mid,
                        groups, eps, 1);
    iris_dev_conv2d(x, y, d->enc_conv_out_w, d->enc_conv_out_b, h, w, mid, z2, 3, 1, 1, 1);
    if (d->quant_w) {
        iris_dev_conv2d(y, x, d->quant_w, d->quant_b, h, w, z2, z2, 1, 1, 0, 0);
        VDEV_SWAP();
    }
#undef VDEV_SWAP

    /* Download the moments and finish on the host (channel-first). */
    size_t n = (size_t)h * w * z2;
    float *moments_hwc = malloc(n * sizeof(float)), *moments = malloc(n * sizeof(float));
    if (!moments_hwc || !moments) {
        free(moments_hwc);
        free(moments);
        return NULL;
    }
    iris_dev_download(moments_hwc, x, n * sizeof(float));
    for (int c = 0; c < z2; c++)
        for (size_t p = 0; p < (size_t)h * w; p++) moments[c * (size_t)h * w + p] = moments_hwc[p * z2 + c];
    free(moments_hwc);
    float *latent = vae_finish_latent(vae, moments, 1, h, w, out_h, out_w);
    free(moments);
    return latent;
}
#endif /* VAE_DEV */

iris_image *iris_vae_decode(iris_vae_t *vae, const float *latent,
                            int batch, int latent_h, int latent_w) {
    if (latent_h <= 0 || latent_w <= 0 || latent_h > 256 || latent_w > 256 ||
        !iris_vae_reserve(vae, batch, latent_h * 16, latent_w * 16)) return NULL;
#ifdef USE_METAL
    /* Try GPU-resident path first (eliminates CPU<->GPU round-trips per conv) */
    if (iris_metal_available()) {
        iris_image *gpu_result = vae_decode_gpu(vae, latent, batch, latent_h, latent_w);
        if (gpu_result) return gpu_result;
        /* Fall through to CPU path on failure */
    }
#endif
#ifdef VAE_DEV
    if (iris_dev_available() && batch == 1) {
        iris_image *gpu_result = vae_decode_dev(vae, latent, latent_h, latent_w);
        if (gpu_result) return gpu_result;
        fprintf(stderr, "VAE: GPU decode failed, using the CPU\n");
    }
#endif

    /*
     * Decoder path:
     * [B, latent_ch, H/16, W/16]
     * -> denormalize (batch denorm for Flux, scaling/shift for Z-Image)
     * -> unpatchify -> [B, z_ch, H/8, W/8]
     * -> [post_quant_conv] -> conv_in -> mid_block -> up_blocks -> norm -> conv_out
     * -> [B, 3, H, W]
     */

    int ch_mult[4] = {1, 2, 4, 4};
    float *x = vae->work1;
    float *work = vae->work2;

    /* Denormalize and unpatchify: [B, latent_ch, H/16, W/16] -> [B, z_ch, H/8, W/8] */
    int unpatch_h = latent_h * 2;
    int unpatch_w = latent_w * 2;
    vae_prepare_latent(vae, x, work, latent, batch, latent_h, latent_w);

    int cur_h = unpatch_h, cur_w = unpatch_w;

    /* Post-quantization conv (1x1) - Flux only */
    if (vae->post_quant_conv_weight) {
        vae_conv2d(work, x, vae->post_quant_conv_weight, vae->post_quant_conv_bias,
                    batch, vae->z_channels, vae->z_channels, cur_h, cur_w, 1, 1, 1, 0);
        iris_copy(x, work, batch * vae->z_channels * cur_h * cur_w);
    }

    /* Conv in: 32 -> 512 */
    int mid_ch = vae->base_channels * ch_mult[3];  /* 512 */
    vae_conv2d(work, x, vae->dec_conv_in_weight, vae->dec_conv_in_bias,
                batch, vae->z_channels, mid_ch, cur_h, cur_w, 3, 3, 1, 1);
    iris_copy(x, work, batch * mid_ch * cur_h * cur_w);

    /* Mid block: resblock -> attn -> resblock */
    int progress = 0;
    int total_blocks = 3 + 4 * (vae->num_res_blocks + 1);  /* mid + up resblocks */

    resblock_forward(work, x, &vae->dec_mid_block1, vae->work3,
                     batch, cur_h, cur_w, vae->num_groups, vae->eps);
    if (iris_vae_progress_callback)
        iris_vae_progress_callback(progress++, total_blocks);
    if (attnblock_forward(x, work, &vae->dec_mid_attn, vae->work3,
                          batch, cur_h, cur_w, vae->num_groups, vae->eps) < 0) {
        return NULL;  /* OOM in attention */
    }
    if (iris_vae_progress_callback)
        iris_vae_progress_callback(progress++, total_blocks);
    resblock_forward(work, x, &vae->dec_mid_block2, vae->work3,
                     batch, cur_h, cur_w, vae->num_groups, vae->eps);
    iris_copy(x, work, batch * mid_ch * cur_h * cur_w);
    if (iris_vae_progress_callback)
        iris_vae_progress_callback(progress++, total_blocks);

    int block_idx = 0;
    int up_idx = 0;

    /* Up blocks (reverse order of channels) */
    for (int level = 3; level >= 0; level--) {
        int ch_out = vae->base_channels * ch_mult[level];

        /* num_res_blocks + 1 resblocks per level */
        for (int r = 0; r < vae->num_res_blocks + 1; r++) {
            vae_resblock_t *block = &vae->dec_up_blocks[block_idx++];
            resblock_forward(work, x, block, vae->work3,
                             batch, cur_h, cur_w, vae->num_groups, vae->eps);
            iris_copy(x, work, batch * ch_out * cur_h * cur_w);
            if (iris_vae_progress_callback)
                iris_vae_progress_callback(progress++, total_blocks);
        }

        /* Upsample (except level 0) */
        if (level > 0) {
            vae_upsample_t *us = &vae->dec_upsample[up_idx++];
            int new_h = cur_h * 2;
            int new_w = cur_w * 2;

            /* Nearest neighbor upsample */
            iris_upsample_nearest(work, x, batch, ch_out, cur_h, cur_w, 2, 2);

            /* Conv for refinement */
            vae_conv2d(x, work, us->conv_weight, us->conv_bias,
                        batch, ch_out, ch_out, new_h, new_w, 3, 3, 1, 1);

            cur_h = new_h;
            cur_w = new_w;
        }
    }

    int out_ch = vae->base_channels;  /* 128 */

    /* Output: norm -> swish -> conv */
    iris_group_norm(work, x, vae->dec_norm_out_weight, vae->dec_norm_out_bias,
                    batch, out_ch, cur_h, cur_w, vae->num_groups, vae->eps);
    swish_inplace(work, batch * out_ch * cur_h * cur_w);

    /* Conv out: 128 -> 3 */
    vae_conv2d(x, work, vae->dec_conv_out_weight, vae->dec_conv_out_bias,
                batch, out_ch, 3, cur_h, cur_w, 3, 3, 1, 1);

    /* Convert to image */
    int H = cur_h;
    int W = cur_w;

    iris_image *img = iris_image_create(W, H, 3);
    if (!img) return NULL;

    /* Denormalize from [-1, 1] to [0, 255] and convert to uint8 */
    for (int y = 0; y < H; y++) {
        for (int c = 0; c < W; c++) {
            for (int ch = 0; ch < 3; ch++) {
                float val = x[ch * H * W + y * W + c];
                val = (val + 1.0f) * 0.5f;  /* [-1,1] -> [0,1] */
                val = val * 255.0f;
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                img->data[(y * W + c) * 3 + ch] = (uint8_t)(val + 0.5f);
            }
        }
    }

    return img;
}

/* ========================================================================
 * VAE Loading and Memory Management
 * ======================================================================== */

static int read_uint32(FILE *f, uint32_t *val) {
    return fread(val, sizeof(uint32_t), 1, f) == 1;
}

static float *read_floats(FILE *f, int count) {
    float *data = (float *)malloc(count * sizeof(float));
    if (!data) return NULL;
    if (fread(data, sizeof(float), count, f) != (size_t)count) {
        free(data);
        return NULL;
    }
    return data;
}

static int load_resblock(FILE *f, vae_resblock_t *block) {
    if (!read_uint32(f, (uint32_t *)&block->in_channels)) return 0;
    if (!read_uint32(f, (uint32_t *)&block->out_channels)) return 0;

    int in_ch = block->in_channels;
    int out_ch = block->out_channels;

    block->norm1_weight = read_floats(f, in_ch);
    block->norm1_bias = read_floats(f, in_ch);
    block->conv1_weight = read_floats(f, out_ch * in_ch * 3 * 3);
    block->conv1_bias = read_floats(f, out_ch);
    block->norm2_weight = read_floats(f, out_ch);
    block->norm2_bias = read_floats(f, out_ch);
    block->conv2_weight = read_floats(f, out_ch * out_ch * 3 * 3);
    block->conv2_bias = read_floats(f, out_ch);

    if (in_ch != out_ch) {
        block->skip_weight = read_floats(f, out_ch * in_ch);
        block->skip_bias = read_floats(f, out_ch);
    } else {
        block->skip_weight = NULL;
        block->skip_bias = NULL;
    }

    return block->norm1_weight && block->conv1_weight && block->conv2_weight;
}

static int load_attnblock(FILE *f, vae_attnblock_t *block) {
    if (!read_uint32(f, (uint32_t *)&block->channels)) return 0;

    int ch = block->channels;

    block->norm_weight = read_floats(f, ch);
    block->norm_bias = read_floats(f, ch);
    block->q_weight = read_floats(f, ch * ch);
    block->q_bias = read_floats(f, ch);
    block->k_weight = read_floats(f, ch * ch);
    block->k_bias = read_floats(f, ch);
    block->v_weight = read_floats(f, ch * ch);
    block->v_bias = read_floats(f, ch);
    block->out_weight = read_floats(f, ch * ch);
    block->out_bias = read_floats(f, ch);

    return block->norm_weight && block->q_weight && block->out_weight;
}

static void free_resblock(vae_resblock_t *block) {
    free(block->norm1_weight);
    free(block->norm1_bias);
    free(block->conv1_weight);
    free(block->conv1_bias);
    free(block->norm2_weight);
    free(block->norm2_bias);
    free(block->conv2_weight);
    free(block->conv2_bias);
    free(block->skip_weight);
    free(block->skip_bias);
}

static void free_attnblock(vae_attnblock_t *block) {
    free(block->norm_weight);
    free(block->norm_bias);
    free(block->q_weight);
    free(block->q_bias);
    free(block->k_weight);
    free(block->k_bias);
    free(block->v_weight);
    free(block->v_bias);
    free(block->out_weight);
    free(block->out_bias);
}

iris_vae_t *iris_vae_load(FILE *f) {
    iris_vae_t *vae = calloc(1, sizeof(iris_vae_t));
    if (!vae) return NULL;

    /* Read config */
    uint32_t config[6];
    if (fread(config, sizeof(uint32_t), 6, f) != 6) goto error;

    vae->z_channels = config[0];
    vae->latent_channels = vae->z_channels * 4;  /* 2x2 patchify */
    vae->base_channels = config[1];
    vae->num_res_blocks = config[2];
    vae->num_groups = config[3];
    vae->max_h = config[4];
    vae->max_w = config[5];
    vae->scaling_factor = 0.0f;
    vae->shift_factor = 0.0f;

    vae->ch_mult[0] = IRIS_VAE_CH_MULT_0;
    vae->ch_mult[1] = IRIS_VAE_CH_MULT_1;
    vae->ch_mult[2] = IRIS_VAE_CH_MULT_2;
    vae->ch_mult[3] = IRIS_VAE_CH_MULT_3;
    vae->eps = 1e-4f;  /* batch_norm_eps from config */

    /* Read encoder conv_in */
    vae->enc_conv_in_weight = read_floats(f, vae->base_channels * 3 * 3 * 3);
    vae->enc_conv_in_bias = read_floats(f, vae->base_channels);

    /* Read encoder down blocks */
    int num_down_blocks = 4 * vae->num_res_blocks;
    vae->enc_down_blocks = calloc(num_down_blocks, sizeof(vae_resblock_t));
    for (int i = 0; i < num_down_blocks; i++) {
        if (!load_resblock(f, &vae->enc_down_blocks[i])) goto error;
    }

    /* Read encoder downsamples */
    vae->enc_downsample = calloc(3, sizeof(vae_downsample_t));
    for (int i = 0; i < 3; i++) {
        int ch = vae->base_channels * vae->ch_mult[i];
        vae->enc_downsample[i].channels = ch;
        vae->enc_downsample[i].conv_weight = read_floats(f, ch * ch * 3 * 3);
        vae->enc_downsample[i].conv_bias = read_floats(f, ch);
    }

    /* Read encoder mid block */
    if (!load_resblock(f, &vae->enc_mid_block1)) goto error;
    if (!load_attnblock(f, &vae->enc_mid_attn)) goto error;
    if (!load_resblock(f, &vae->enc_mid_block2)) goto error;

    /* Read encoder output */
    int mid_ch = vae->base_channels * vae->ch_mult[3];
    vae->enc_norm_out_weight = read_floats(f, mid_ch);
    vae->enc_norm_out_bias = read_floats(f, mid_ch);
    vae->enc_conv_out_weight = read_floats(f, vae->z_channels * 2 * mid_ch * 3 * 3);
    vae->enc_conv_out_bias = read_floats(f, vae->z_channels * 2);

    /* Read decoder conv_in */
    vae->dec_conv_in_weight = read_floats(f, mid_ch * vae->z_channels * 3 * 3);
    vae->dec_conv_in_bias = read_floats(f, mid_ch);

    /* Read decoder mid block */
    if (!load_resblock(f, &vae->dec_mid_block1)) goto error;
    if (!load_attnblock(f, &vae->dec_mid_attn)) goto error;
    if (!load_resblock(f, &vae->dec_mid_block2)) goto error;

    /* Read decoder up blocks */
    int num_up_blocks = 4 * (vae->num_res_blocks + 1);
    vae->dec_up_blocks = calloc(num_up_blocks, sizeof(vae_resblock_t));
    for (int i = 0; i < num_up_blocks; i++) {
        if (!load_resblock(f, &vae->dec_up_blocks[i])) goto error;
    }

    /* Read decoder upsamples */
    vae->dec_upsample = calloc(3, sizeof(vae_upsample_t));
    for (int i = 0; i < 3; i++) {
        int ch = vae->base_channels * vae->ch_mult[3 - i];
        vae->dec_upsample[i].channels = ch;
        vae->dec_upsample[i].conv_weight = read_floats(f, ch * ch * 3 * 3);
        vae->dec_upsample[i].conv_bias = read_floats(f, ch);
    }

    /* Read decoder output */
    vae->dec_norm_out_weight = read_floats(f, vae->base_channels);
    vae->dec_norm_out_bias = read_floats(f, vae->base_channels);
    vae->dec_conv_out_weight = read_floats(f, 3 * vae->base_channels * 3 * 3);
    vae->dec_conv_out_bias = read_floats(f, 3);

    /* Read batch norm stats */
    vae->bn_mean = read_floats(f, vae->latent_channels);
    vae->bn_var = read_floats(f, vae->latent_channels);

    /* Allocate working memory */
    size_t max_spatial = (size_t)vae->max_h * vae->max_w;
    size_t max_channels = mid_ch;  /* 512 */
    vae->work_size = 4 * max_channels * max_spatial * sizeof(float);
    vae->work1 = (float *)malloc(vae->work_size);
    vae->work2 = (float *)malloc(vae->work_size);
    vae->work3 = (float *)malloc(vae->work_size);

    if (!vae->work1 || !vae->work2 || !vae->work3) goto error;

    return vae;

error:
    iris_vae_free(vae);
    return NULL;
}

void iris_vae_free(iris_vae_t *vae) {
    if (!vae) return;

    free(vae->enc_conv_in_weight);
    free(vae->enc_conv_in_bias);

    if (vae->enc_down_blocks) {
        for (int i = 0; i < 4 * vae->num_res_blocks; i++) {
            free_resblock(&vae->enc_down_blocks[i]);
        }
        free(vae->enc_down_blocks);
    }

    if (vae->enc_downsample) {
        for (int i = 0; i < 3; i++) {
            free(vae->enc_downsample[i].conv_weight);
            free(vae->enc_downsample[i].conv_bias);
        }
        free(vae->enc_downsample);
    }

    free_resblock(&vae->enc_mid_block1);
    free_attnblock(&vae->enc_mid_attn);
    free_resblock(&vae->enc_mid_block2);

    free(vae->enc_norm_out_weight);
    free(vae->enc_norm_out_bias);
    free(vae->enc_conv_out_weight);
    free(vae->enc_conv_out_bias);
    free(vae->quant_conv_weight);
    free(vae->quant_conv_bias);

    free(vae->dec_conv_in_weight);
    free(vae->dec_conv_in_bias);

    free_resblock(&vae->dec_mid_block1);
    free_attnblock(&vae->dec_mid_attn);
    free_resblock(&vae->dec_mid_block2);

    if (vae->dec_up_blocks) {
        for (int i = 0; i < 4 * (vae->num_res_blocks + 1); i++) {
            free_resblock(&vae->dec_up_blocks[i]);
        }
        free(vae->dec_up_blocks);
    }

    if (vae->dec_upsample) {
        for (int i = 0; i < 3; i++) {
            free(vae->dec_upsample[i].conv_weight);
            free(vae->dec_upsample[i].conv_bias);
        }
        free(vae->dec_upsample);
    }

    free(vae->dec_norm_out_weight);
    free(vae->dec_norm_out_bias);
    free(vae->dec_conv_out_weight);
    free(vae->dec_conv_out_bias);

    free(vae->bn_mean);
    free(vae->bn_var);
    free(vae->post_quant_conv_weight);
    free(vae->post_quant_conv_bias);

    free(vae->work1);
    free(vae->work2);
    free(vae->work3);

#ifdef VAE_DEV
    vdev_free(vae);
#endif
    free(vae);
}

/* ========================================================================
 * Image Preprocessing
 * ======================================================================== */

/* Convert image to tensor [B, 3, H, W] normalized to [-1, 1] */
float *iris_image_to_tensor(const iris_image *img) {
    int H = img->height;
    int W = img->width;
    int C = img->channels;

    float *tensor = (float *)malloc(3 * H * W * sizeof(float));
    if (!tensor) return NULL;

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < 3; c++) {
                float val;
                if (c < C) {
                    val = (float)img->data[(y * W + x) * C + c];
                } else {
                    val = 0.0f;  /* Pad with zeros if grayscale */
                }
                val = val / 255.0f;         /* [0, 255] -> [0, 1] */
                val = val * 2.0f - 1.0f;    /* [0, 1] -> [-1, 1] */
                tensor[c * H * W + y * W + x] = val;
            }
        }
    }

    return tensor;
}

/* ========================================================================
 * Safetensors Loading
 * ======================================================================== */

static float *get_sf_tensor(safetensors_file_t *sf, const char *name) {
    const safetensor_t *t = safetensors_find(sf, name);
    if (!t) {
        fprintf(stderr, "Error: required tensor %s not found\n", name);
        return NULL;
    }
    return safetensors_get_f32(sf, t);
}

static int load_resblock_sf(safetensors_file_t *sf, vae_resblock_t *block,
                             const char *prefix, int in_ch, int out_ch) {
    char name[256];

    block->in_channels = in_ch;
    block->out_channels = out_ch;

    snprintf(name, sizeof(name), "%s.norm1.weight", prefix);
    block->norm1_weight = get_sf_tensor(sf, name);
    snprintf(name, sizeof(name), "%s.norm1.bias", prefix);
    block->norm1_bias = get_sf_tensor(sf, name);

    snprintf(name, sizeof(name), "%s.conv1.weight", prefix);
    block->conv1_weight = get_sf_tensor(sf, name);
    snprintf(name, sizeof(name), "%s.conv1.bias", prefix);
    block->conv1_bias = get_sf_tensor(sf, name);

    snprintf(name, sizeof(name), "%s.norm2.weight", prefix);
    block->norm2_weight = get_sf_tensor(sf, name);
    snprintf(name, sizeof(name), "%s.norm2.bias", prefix);
    block->norm2_bias = get_sf_tensor(sf, name);

    snprintf(name, sizeof(name), "%s.conv2.weight", prefix);
    block->conv2_weight = get_sf_tensor(sf, name);
    snprintf(name, sizeof(name), "%s.conv2.bias", prefix);
    block->conv2_bias = get_sf_tensor(sf, name);

    if (in_ch != out_ch) {
        snprintf(name, sizeof(name), "%s.conv_shortcut.weight", prefix);
        block->skip_weight = get_sf_tensor(sf, name);
        snprintf(name, sizeof(name), "%s.conv_shortcut.bias", prefix);
        block->skip_bias = get_sf_tensor(sf, name);
    } else {
        block->skip_weight = NULL;
        block->skip_bias = NULL;
    }

    /* Check required tensors */
    if (!block->norm1_weight || !block->conv1_weight || !block->conv2_weight) {
        return -1;
    }
    return 0;
}

static int load_attnblock_sf(safetensors_file_t *sf, vae_attnblock_t *block,
                              const char *prefix, int channels) {
    char name[256];

    block->channels = channels;

    snprintf(name, sizeof(name), "%s.group_norm.weight", prefix);
    block->norm_weight = get_sf_tensor(sf, name);
    snprintf(name, sizeof(name), "%s.group_norm.bias", prefix);
    block->norm_bias = get_sf_tensor(sf, name);

    snprintf(name, sizeof(name), "%s.to_q.weight", prefix);
    block->q_weight = get_sf_tensor(sf, name);
    snprintf(name, sizeof(name), "%s.to_q.bias", prefix);
    block->q_bias = get_sf_tensor(sf, name);

    snprintf(name, sizeof(name), "%s.to_k.weight", prefix);
    block->k_weight = get_sf_tensor(sf, name);
    snprintf(name, sizeof(name), "%s.to_k.bias", prefix);
    block->k_bias = get_sf_tensor(sf, name);

    snprintf(name, sizeof(name), "%s.to_v.weight", prefix);
    block->v_weight = get_sf_tensor(sf, name);
    snprintf(name, sizeof(name), "%s.to_v.bias", prefix);
    block->v_bias = get_sf_tensor(sf, name);

    snprintf(name, sizeof(name), "%s.to_out.0.weight", prefix);
    block->out_weight = get_sf_tensor(sf, name);
    snprintf(name, sizeof(name), "%s.to_out.0.bias", prefix);
    block->out_bias = get_sf_tensor(sf, name);

    /* Check required tensors */
    if (!block->norm_weight || !block->q_weight || !block->out_weight) {
        return -1;
    }
    return 0;
}

/* Load VAE weights from safetensors format. Extended version takes z_channels
 * (16 for Z-Image, 32 for Flux) and scaling/shift factors to support both
 * VAE variants. Builds the full encoder and decoder weight hierarchy:
 * conv_in -> down_blocks (4 levels, channel multipliers [1,2,4,4]) ->
 * mid_block -> up_blocks -> conv_out. Also loads batch norm stats if present. */
iris_vae_t *iris_vae_load_safetensors_ex(safetensors_file_t *sf,
                                          int z_channels,
                                          float scaling_factor,
                                          float shift_factor,
                                          int allow_large_images) {
    iris_vae_t *vae = calloc(1, sizeof(iris_vae_t));
    if (!vae) return NULL;

    char name[256];
    int ch_mult[4] = {1, 2, 4, 4};

    /* Set config */
    vae->z_channels = z_channels;
    vae->latent_channels = z_channels * 4;  /* 2x2 patchify */
    vae->base_channels = 128;
    vae->num_res_blocks = 2;
    vae->num_groups = 32;
    vae->max_h = IRIS_VAE_MAX_DIM;
    vae->max_w = IRIS_VAE_MAX_DIM;
    vae->scaling_factor = scaling_factor;
    vae->shift_factor = shift_factor;
    vae->eps = (scaling_factor != 0.0f) ? 1e-6f : 1e-4f;

    vae->ch_mult[0] = ch_mult[0];
    vae->ch_mult[1] = ch_mult[1];
    vae->ch_mult[2] = ch_mult[2];
    vae->ch_mult[3] = ch_mult[3];

    /* Encoder conv_in */
    vae->enc_conv_in_weight = get_sf_tensor(sf, "encoder.conv_in.weight");
    vae->enc_conv_in_bias = get_sf_tensor(sf, "encoder.conv_in.bias");

    /* Encoder down blocks */
    int num_down_blocks = 4 * vae->num_res_blocks;
    vae->enc_down_blocks = calloc(num_down_blocks, sizeof(vae_resblock_t));

    int block_idx = 0;
    for (int level = 0; level < 4; level++) {
        int ch = vae->base_channels * ch_mult[level];
        int prev_ch = (level == 0) ? vae->base_channels : vae->base_channels * ch_mult[level - 1];

        for (int r = 0; r < vae->num_res_blocks; r++) {
            int in_ch = (r == 0 && level > 0) ? prev_ch : ch;
            snprintf(name, sizeof(name), "encoder.down_blocks.%d.resnets.%d", level, r);
            load_resblock_sf(sf, &vae->enc_down_blocks[block_idx++], name, in_ch, ch);
        }
    }

    /* Encoder downsamples */
    vae->enc_downsample = calloc(3, sizeof(vae_downsample_t));
    for (int i = 0; i < 3; i++) {
        int ch = vae->base_channels * ch_mult[i];
        vae->enc_downsample[i].channels = ch;
        snprintf(name, sizeof(name), "encoder.down_blocks.%d.downsamplers.0.conv.weight", i);
        vae->enc_downsample[i].conv_weight = get_sf_tensor(sf, name);
        snprintf(name, sizeof(name), "encoder.down_blocks.%d.downsamplers.0.conv.bias", i);
        vae->enc_downsample[i].conv_bias = get_sf_tensor(sf, name);
    }

    /* Encoder mid block */
    int mid_ch = vae->base_channels * ch_mult[3];  /* 512 */
    load_resblock_sf(sf, &vae->enc_mid_block1, "encoder.mid_block.resnets.0", mid_ch, mid_ch);
    load_attnblock_sf(sf, &vae->enc_mid_attn, "encoder.mid_block.attentions.0", mid_ch);
    load_resblock_sf(sf, &vae->enc_mid_block2, "encoder.mid_block.resnets.1", mid_ch, mid_ch);

    /* Encoder output */
    vae->enc_norm_out_weight = get_sf_tensor(sf, "encoder.conv_norm_out.weight");
    vae->enc_norm_out_bias = get_sf_tensor(sf, "encoder.conv_norm_out.bias");
    vae->enc_conv_out_weight = get_sf_tensor(sf, "encoder.conv_out.weight");
    vae->enc_conv_out_bias = get_sf_tensor(sf, "encoder.conv_out.bias");

    /* Quant conv (Flux only - Z-Image doesn't have these) */
    if (safetensors_find(sf, "quant_conv.weight")) {
        vae->quant_conv_weight = get_sf_tensor(sf, "quant_conv.weight");
        vae->quant_conv_bias = get_sf_tensor(sf, "quant_conv.bias");
    }

    /* Decoder conv_in */
    vae->dec_conv_in_weight = get_sf_tensor(sf, "decoder.conv_in.weight");
    vae->dec_conv_in_bias = get_sf_tensor(sf, "decoder.conv_in.bias");

    /* Decoder mid block */
    load_resblock_sf(sf, &vae->dec_mid_block1, "decoder.mid_block.resnets.0", mid_ch, mid_ch);
    load_attnblock_sf(sf, &vae->dec_mid_attn, "decoder.mid_block.attentions.0", mid_ch);
    load_resblock_sf(sf, &vae->dec_mid_block2, "decoder.mid_block.resnets.1", mid_ch, mid_ch);

    /* Decoder up blocks (reverse order) */
    int num_up_blocks = 4 * (vae->num_res_blocks + 1);
    vae->dec_up_blocks = calloc(num_up_blocks, sizeof(vae_resblock_t));

    block_idx = 0;
    for (int level = 3; level >= 0; level--) {
        int ch = vae->base_channels * ch_mult[level];
        int prev_ch = (level == 3) ? mid_ch : vae->base_channels * ch_mult[level + 1];

        for (int r = 0; r < vae->num_res_blocks + 1; r++) {
            int in_ch = (r == 0) ? prev_ch : ch;
            int up_idx = 3 - level;
            snprintf(name, sizeof(name), "decoder.up_blocks.%d.resnets.%d", up_idx, r);
            load_resblock_sf(sf, &vae->dec_up_blocks[block_idx++], name, in_ch, ch);
        }
    }

    /* Decoder upsamples - up_blocks[0,1,2] have upsamplers, up_blocks[3] does not
     * dec_upsample[0] -> up_blocks.0 (level 3, 512 ch)
     * dec_upsample[1] -> up_blocks.1 (level 2, 512 ch)
     * dec_upsample[2] -> up_blocks.2 (level 1, 256 ch) */
    vae->dec_upsample = calloc(3, sizeof(vae_upsample_t));
    for (int i = 0; i < 3; i++) {
        int ch = vae->base_channels * ch_mult[3 - i];
        vae->dec_upsample[i].channels = ch;
        snprintf(name, sizeof(name), "decoder.up_blocks.%d.upsamplers.0.conv.weight", i);
        vae->dec_upsample[i].conv_weight = get_sf_tensor(sf, name);
        snprintf(name, sizeof(name), "decoder.up_blocks.%d.upsamplers.0.conv.bias", i);
        vae->dec_upsample[i].conv_bias = get_sf_tensor(sf, name);
    }

    /* Decoder output */
    vae->dec_norm_out_weight = get_sf_tensor(sf, "decoder.conv_norm_out.weight");
    vae->dec_norm_out_bias = get_sf_tensor(sf, "decoder.conv_norm_out.bias");
    vae->dec_conv_out_weight = get_sf_tensor(sf, "decoder.conv_out.weight");
    vae->dec_conv_out_bias = get_sf_tensor(sf, "decoder.conv_out.bias");

    /* Batch norm stats (Flux only) */
    const safetensor_t *bn_mean_t = safetensors_find(sf, "bn.running_mean");
    if (bn_mean_t) {
        vae->bn_mean = safetensors_get_f32(sf, bn_mean_t);
        const safetensor_t *bn_var_t = safetensors_find(sf, "bn.running_var");
        vae->bn_var = bn_var_t ? safetensors_get_f32(sf, bn_var_t) : NULL;
    }

    /* Post-quantization conv (Flux only - Z-Image doesn't have this) */
    if (safetensors_find(sf, "post_quant_conv.weight")) {
        vae->post_quant_conv_weight = get_sf_tensor(sf, "post_quant_conv.weight");
        vae->post_quant_conv_bias = get_sf_tensor(sf, "post_quant_conv.bias");
    }

    /* Fallback: if no batch norm found and no scaling factor, use identity */
    if (!vae->bn_mean && vae->scaling_factor == 0.0f) {
        int lc = vae->latent_channels;
        vae->bn_mean = calloc(lc, sizeof(float));
        vae->bn_var = malloc(lc * sizeof(float));
        for (int i = 0; i < lc; i++) vae->bn_var[i] = 1.0f;
    }

    /* Allocate working memory
     * The decoder upsamples from H/8 to full H resolution.
     * At full resolution (level 0), we have base_channels (128) channels.
     * work1/work2: hold main tensors, max 128 * H * W
     * work3: used for resblock/attention ops, needs ~4x main buffer
     *
     * Memory per buffer = 4 * 128 * H * W = 512 * H * W floats
     * For 1024x1024: ~2GB per buffer, ~6GB total working memory
     * For 1792x1792: ~6GB per buffer, ~18GB total working memory
     */
    if (allow_large_images) return vae;
    size_t max_spatial = (size_t)vae->max_h * vae->max_w;
    size_t max_channels = vae->base_channels;  /* 128 at full resolution */
    vae->work_size = 4 * max_channels * max_spatial * sizeof(float);
    vae->work1 = malloc(vae->work_size);
    vae->work2 = malloc(vae->work_size);
    vae->work3 = malloc(vae->work_size);

    if (!vae->work1 || !vae->work2 || !vae->work3) {
        iris_vae_free(vae);
        return NULL;
    }

    return vae;
}

/* Backward-compatible wrapper: loads with Flux defaults (z_channels=32, no scaling) */
iris_vae_t *iris_vae_load_safetensors(safetensors_file_t *sf) {
    return iris_vae_load_safetensors_ex(sf, 32, 0.0f, 0.0f, 0);
}
