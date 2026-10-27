/*
 * Iris device (GPU) compute API
 *
 * A small set of operations for GPU-resident inference, implemented by
 * iris_cuda.cu for NVIDIA GPUs (CUDA) and AMD GPUs (ROCm/HIP, same source),
 * and by iris_dev_metal.m for Apple GPUs.
 *
 * - Tensors are plain device pointers from iris_dev_alloc(); bf16 values
 *   are stored as uint16_t, matrices are row-major.
 * - Operations are enqueued in order on a single stream: results are
 *   visible to later operations, and to the host after iris_dev_download()
 *   or iris_dev_sync().
 * - Weights are uploaded explicitly into buffers owned by the model, so
 *   there is no host-pointer keyed cache that could go stale.
 */

#ifndef IRIS_DEV_H
#define IRIS_DEV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize the device. Returns 1 if a usable GPU was found. */
int iris_dev_init(void);
int iris_dev_available(void);
/* Non-causal MHA avoids a full quadratic attention-score allocation. */
int iris_dev_has_memory_efficient_attention(void);
const char *iris_dev_name(void);
void iris_dev_cleanup(void);   /* Release the stream and BLAS handle */

/* Memory */
void *iris_dev_alloc(size_t bytes);       /* NULL on failure */
void iris_dev_free(void *p);
/* Both wait for the copy: the host buffer can be reused on return. */
void iris_dev_upload(void *dst, const void *src, size_t bytes);
void iris_dev_download(void *dst, const void *src, size_t bytes);

/* Upload rows of row_bytes bytes with independent strides (in bytes). */
void iris_dev_upload_2d(void *dst, size_t dst_stride, const void *src, size_t src_stride,
                        size_t row_bytes, size_t rows);
void iris_dev_sync(void);

/* dst[i] = bf16(src[i]) */
void iris_dev_f32_to_bf16(uint16_t *dst, const float *src, size_t n);

/* y[rows, out] = x[rows, in] @ W[out, in]^T: bf16 inputs, f32 accumulation,
 * f32 or bf16 output. All matrices contiguous. */
void iris_dev_linear_f32(float *y, const uint16_t *x, const uint16_t *W,
                         int rows, int in, int out);
void iris_dev_linear_bf16(uint16_t *y, const uint16_t *x, const uint16_t *W,
                          int rows, int in, int out);

/* out = LayerNorm(x) * (1 + scale) + shift, no affine parameters.
 * shift/scale: [dim], either may be NULL (zero). */
void iris_dev_layer_norm_mod(uint16_t *out, const float *x,
                             const float *shift, const float *scale,
                             int rows, int dim, float eps);

/* out = RMSNorm(x) * w, w: [dim] */
void iris_dev_rms_norm(uint16_t *out, const float *x, const float *w,
                       int rows, int dim, float eps);

/* RMSNorm(x) * w * (1 + scale), with optional scale [dim]. */
void iris_dev_rms_norm_mod(uint16_t *out, const float *x, const float *w,
                           const float *scale, int rows, int dim, float eps);

/* x += tanh(gate) * RMSNorm(a) * w, entirely in f32.
 * gate [dim] may be NULL for an ungated residual. */
void iris_dev_rms_norm_residual(float *x, const float *a, const float *w,
                                const float *gate, int rows, int dim, float eps);

/* In place on the bf16 heads x[s, h * head_dim + d] (row stride ld):
 * per-head RMSNorm with weight w [head_dim], then rotation by the angles of
 * row s (rope_cos/sin: [seq, head_dim / 2]). Pairs are consecutive
 * (2i, 2i + 1), or (i, i + head_dim / 2) when split_half is set. */
void iris_dev_head_norm_rope(uint16_t *x, int ld, const float *w,
                             const float *rope_cos, const float *rope_sin,
                             int seq, int heads, int head_dim, float eps,
                             int split_half);

/* Scaled dot-product attention over bf16 [seq, heads * head_dim] tensors.
 * q, k, v share the row stride ldi, out has row stride ldo. With fewer kv
 * heads (GQA), query head h uses kv head h / (heads / kv_heads).
 * causal: query i ignores keys j > i. key_mask: device int [seq_k],
 * 0 = ignored key, or NULL. */
void iris_dev_attention(uint16_t *out, int ldo,
                        const uint16_t *q, const uint16_t *k, const uint16_t *v,
                        int ldi, int seq_q, int seq_k, int heads, int kv_heads,
                        int head_dim, float scale, int causal,
                        const int *key_mask);

/* out[r, c] = silu(gate[r, c]) * up[r, c]; gate/up row stride ldi,
 * out row stride ldo. */
void iris_dev_swiglu(uint16_t *out, int ldo, const uint16_t *gate,
                     const uint16_t *up, int ldi, int rows, int cols);

/* x[r, c] += gate[c] * a[r, c], or x += a when gate is NULL. */
void iris_dev_residual(float *x, const float *a, const float *gate,
                       int rows, int dim);

/* Elementwise helpers on f32 vectors. */
void iris_dev_silu(float *x, size_t n);
void iris_dev_add(float *x, const float *a, size_t n);   /* x += a */
void iris_dev_add_bias(float *y, const float *bias, int rows, int cols);  /* y[r, c] += bias[c] */
void iris_dev_fill_rows(float *out, const float *row, int rows, int dim); /* out[r, c] = row[c] */

/* Convolution layers on channels-last f32 images [h * w, channels]. */

/* k x k convolution of [h * width, cin] into [oh * ow, cout], with zero
 * padding of pad_before / pad_after pixels on each axis:
 *   oh = (h + pad_before + pad_after - k) / stride + 1 (same for ow)
 *   out[(oy, ox), o] = bias[o] + sum_{ky, kx, c} w[o, (ky * k + kx) * cin + c] *
 *                      in[(oy * stride + ky - pad_before, ox * stride + kx - pad_before), c]
 * w: bf16 [cout, k * k * cin]. bias may be NULL. */
void iris_dev_conv2d(float *out, const float *in, const uint16_t *w, const float *bias,
                     int h, int width, int cin, int cout, int k, int stride,
                     int pad_before, int pad_after);

/* GroupNorm over [hw, c] (groups of c / groups channels) with affine
 * gamma/beta [c], optionally followed by SiLU. */
void iris_dev_group_norm(float *out, const float *in, const float *gamma, const float *beta,
                         int hw, int c, int groups, float eps, int silu);

/* Nearest-neighbor 2x upsampling of [h * w, c] into [2h * 2w, c]. */
void iris_dev_upsample2x(float *out, const float *in, int h, int width, int c);

#ifdef __cplusplus
}
#endif

#endif /* IRIS_DEV_H */
