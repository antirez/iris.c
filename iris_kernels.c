/*
 * Iris Math Kernels - Implementation
 *
 * Math operations for Iris inference.
 * Uses Metal/MPS on Apple Silicon, BLAS otherwise.
 */

#include "iris_kernels.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

/* Use Metal for GPU acceleration on Apple Silicon */
#ifdef USE_METAL
#include "iris_metal.h"
#endif

/* Use BLAS for matrix operations when enabled via Makefile */
#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/* Minimum matrix size to use GPU (smaller matrices are faster on CPU) */
#define MIN_GPU_ELEMENTS (512 * 512)

/* fast_expf is defined in iris_kernels.h */

/* Progress callbacks - set by caller before inference */
iris_substep_callback_t iris_substep_callback = NULL;
iris_step_callback_t iris_step_callback = NULL;
iris_phase_callback_t iris_phase_callback = NULL;
iris_step_image_callback_t iris_step_image_callback = NULL;
void *iris_step_image_vae = NULL;
iris_text_progress_callback_t iris_text_progress_callback = NULL;
iris_vae_progress_callback_t iris_vae_progress_callback = NULL;
int iris_verbose = 0;

/* ========================================================================
 * Random Number Generator (xoshiro256**)
 * ======================================================================== */

static uint64_t rng_state[4] = {
    0x853c49e6748fea9bULL,
    0xda3e39cb94b95bdbULL,
    0x647c4677a2884327ULL,
    0xc6e7918d2e2969f5ULL
};

static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

static uint64_t xoshiro256ss(void) {
    const uint64_t result = rotl(rng_state[1] * 5, 7) * 9;
    const uint64_t t = rng_state[1] << 17;
    rng_state[2] ^= rng_state[0];
    rng_state[3] ^= rng_state[1];
    rng_state[1] ^= rng_state[2];
    rng_state[0] ^= rng_state[3];
    rng_state[2] ^= t;
    rng_state[3] = rotl(rng_state[3], 45);
    return result;
}

void iris_rng_seed(uint64_t seed) {
    /* SplitMix64 to initialize state from seed */
    for (int i = 0; i < 4; i++) {
        seed += 0x9e3779b97f4a7c15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng_state[i] = z ^ (z >> 31);
    }
}

float iris_random_uniform(void) {
    return (xoshiro256ss() >> 11) * (1.0 / 9007199254740992.0);
}

float iris_random_normal(void) {
    /* Box-Muller transform */
    float u1 = iris_random_uniform();
    float u2 = iris_random_uniform();
    /* Avoid log(0) */
    while (u1 == 0.0f) u1 = iris_random_uniform();
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265358979323846f * u2);
}

void iris_randn(float *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = iris_random_normal();
    }
}

void iris_rand(float *out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = iris_random_uniform();
    }
}

/* ========================================================================
 * Basic Element-wise Operations
 * ======================================================================== */

void iris_add(float *out, const float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + b[i];
    }
}

void iris_add_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];
    }
}

void iris_mul_inplace(float *a, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] *= b[i];
    }
}

void iris_axpy(float *a, float scale, const float *b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += scale * b[i];
    }
}

/* ========================================================================
 * Matrix Operations
 * ======================================================================== */

/* General matrix multiply C = A @ B. Routes to Metal GPU when the matrix
 * is large enough that GPU compute outweighs the CPU-GPU transfer cost,
 * otherwise falls back to BLAS sgemm or a naive triple loop. This is the
 * backbone operation: every linear projection in the transformer, text
 * encoder, and VAE bottleneck goes through here. */
void iris_matmul(float *C, const float *A, const float *B,
                 int M, int K, int N) {
    /* C[M,N] = A[M,K] @ B[K,N] */

#ifdef USE_METAL
    size_t matrix_elements = (size_t)M * N;
    if (iris_metal_available() && matrix_elements >= MIN_GPU_ELEMENTS) {
        iris_metal_sgemm(0, 0,  /* no transpose */
                         M, N, K,
                         1.0f,
                         A, K,
                         B, N,
                         0.0f,
                         C, N);
        return;
    }
#endif

#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, A, K, B, N,
                0.0f, C, N);
#else
    /* Fallback: naive implementation */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
#endif
}

void iris_matmul_t(float *C, const float *A, const float *B,
                   int M, int K, int N) {
    /* C[M,N] = A[M,K] @ B[N,K]^T */

#ifdef USE_METAL
    size_t matrix_elements = (size_t)M * N;
    if (iris_metal_available() && matrix_elements >= MIN_GPU_ELEMENTS) {
        iris_metal_sgemm(0, 1,  /* no transpose A, transpose B */
                         M, N, K,
                         1.0f,
                         A, K,
                         B, K,
                         0.0f,
                         C, N);
        return;
    }
#endif

#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f, A, K, B, K,
                0.0f, C, N);
#else
    /* Fallback: naive implementation */
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[m * K + k] * B[n * K + k];
            }
            C[m * N + n] = sum;
        }
    }
#endif
}

void iris_linear(float *y, const float *x, const float *W, const float *b,
                 int seq_len, int in_dim, int out_dim) {
    /* y[seq, out] = x[seq, in] @ W[out, in]^T + b[out] */

#ifdef USE_METAL
    /* Use Metal GPU for large matrices */
    size_t matrix_elements = (size_t)seq_len * out_dim;
    if (iris_metal_available() && matrix_elements >= MIN_GPU_ELEMENTS) {
        /* Metal sgemm: C = alpha * A @ B^T
         * A[M, K] = x[seq_len, in_dim]
         * B[N, K] = W[out_dim, in_dim] (transposed)
         * C[M, N] = y[seq_len, out_dim]
         */
        iris_metal_sgemm_cached(0, 1,  /* no transpose A, transpose B */
                                seq_len, out_dim, in_dim,
                                1.0f,
                                x, in_dim,
                                W, in_dim,
                                0.0f,
                                y, out_dim);

        /* Add bias if present */
        if (b != NULL) {
            for (int s = 0; s < seq_len; s++) {
                for (int o = 0; o < out_dim; o++) {
                    y[s * out_dim + o] += b[o];
                }
            }
        }
        return;
    }
#endif

#ifdef USE_BLAS
    /* Use BLAS sgemm: C = alpha * A @ B^T + beta * C
     * A[M, K] = x[seq_len, in_dim]
     * B[N, K] = W[out_dim, in_dim]
     * C[M, N] = y[seq_len, out_dim]
     */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, out_dim, in_dim,
                1.0f, x, in_dim, W, in_dim,
                0.0f, y, out_dim);

    /* Add bias if present */
    if (b != NULL) {
        for (int s = 0; s < seq_len; s++) {
            for (int o = 0; o < out_dim; o++) {
                y[s * out_dim + o] += b[o];
            }
        }
    }
#else
    /* Fallback: naive implementation */
    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * in_dim;
        float *y_row = y + s * out_dim;
        for (int o = 0; o < out_dim; o++) {
            const float *w_row = W + o * in_dim;
            float sum = (b != NULL) ? b[o] : 0.0f;
            for (int i = 0; i < in_dim; i++) {
                sum += x_row[i] * w_row[i];
            }
            y_row[o] = sum;
        }
    }
#endif
}

void iris_linear_nobias(float *y, const float *x, const float *W,
                        int seq_len, int in_dim, int out_dim) {
    iris_linear(y, x, W, NULL, seq_len, in_dim, out_dim);
}

void iris_linear_nobias_bf16(float *y, const float *x, const uint16_t *W_bf16,
                             int seq_len, int in_dim, int out_dim) {
    /* y[seq, out] = x[seq, in] @ W[out, in]^T */

#ifdef USE_METAL
    /* Use Metal GPU for bf16 matmul - provides 2x memory bandwidth */
    size_t matrix_elements = (size_t)seq_len * out_dim;
    if (iris_metal_available() && matrix_elements >= MIN_GPU_ELEMENTS) {
        /* Metal bf16 sgemm: C = alpha * A @ B^T
         * A[M, K] = x[seq_len, in_dim] (f32)
         * B[N, K] = W[out_dim, in_dim] (bf16, transposed)
         * C[M, N] = y[seq_len, out_dim] (f32)
         */
        iris_metal_sgemm_bf16(0, 1,  /* no transpose A, transpose B */
                              seq_len, out_dim, in_dim,
                              1.0f,
                              x, in_dim,
                              W_bf16, in_dim,
                              0.0f,
                              y, out_dim);
        return;
    }
#endif

    /* Fallback: convert bf16 to f32 and use regular linear */
    float *W_f32 = (float *)malloc((size_t)out_dim * in_dim * sizeof(float));
    if (!W_f32) return;

    /* Convert bf16 to f32 */
    for (int i = 0; i < out_dim * in_dim; i++) {
        uint32_t f32_bits = ((uint32_t)W_bf16[i]) << 16;
        memcpy(&W_f32[i], &f32_bits, sizeof(float));
    }

    iris_linear_nobias(y, x, W_f32, seq_len, in_dim, out_dim);
    free(W_f32);
}

/* ========================================================================
 * GPU Batch Operations
 * ======================================================================== */

void iris_gpu_begin_batch(void) {
#ifdef USE_METAL
    iris_metal_begin_batch();
#endif
}

void iris_gpu_end_batch(void) {
#ifdef USE_METAL
    iris_metal_end_batch();
#endif
}

/* ========================================================================
 * Convolution Operations
 * ======================================================================== */

/* 2D convolution via im2col + GEMM: reshapes input so each column is a
 * flattened receptive field, then multiplies by the kernel weight matrix.
 * Tiles spatially to bound memory usage for large feature maps. This is the
 * standard approach for BLAS/GPU-friendly convolution, used throughout the
 * VAE encoder and decoder. */
void iris_conv2d(float *out, const float *in, const float *weight, const float *bias,
                 int batch, int in_ch, int out_ch, int H, int W,
                 int kH, int kW, int stride, int padding) {
    int outH = (H + 2 * padding - kH) / stride + 1;
    int outW = (W + 2 * padding - kW) / stride + 1;

#ifdef USE_BLAS
    /* im2col + BLAS optimization with tiling for large convolutions */
    size_t col_size = (size_t)in_ch * kH * kW * outH * outW;
    size_t max_col_size = (size_t)256 * 1024 * 1024;  /* 1GB limit */

    /* For large convolutions, process in row tiles */
    int tile_rows = outH;
    if (col_size > max_col_size) {
        /* Calculate how many rows we can process at once */
        size_t row_size = (size_t)in_ch * kH * kW * outW;
        tile_rows = (int)(max_col_size / row_size);
        if (tile_rows < 1) tile_rows = 1;
    }

    size_t tile_col_size = (size_t)in_ch * kH * kW * tile_rows * outW;
    float *col = malloc(tile_col_size * sizeof(float));
    if (!col) {
        goto naive_fallback;
    }

    for (int b = 0; b < batch; b++) {
        const float *in_b = in + b * in_ch * H * W;
        float *out_b = out + b * out_ch * outH * outW;

        /* Process in tiles of rows */
        for (int tile_start = 0; tile_start < outH; tile_start += tile_rows) {
            int tile_end = tile_start + tile_rows;
            if (tile_end > outH) tile_end = outH;
            int tile_h = tile_end - tile_start;
            int tile_pixels = tile_h * outW;

            /* im2col for this tile: col[in_ch*kH*kW, tile_pixels] */
            int col_row = 0;
            for (int ic = 0; ic < in_ch; ic++) {
                for (int kh = 0; kh < kH; kh++) {
                    for (int kw = 0; kw < kW; kw++) {
                        for (int oh = tile_start; oh < tile_end; oh++) {
                            for (int ow = 0; ow < outW; ow++) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;
                                int col_idx = col_row * tile_pixels + (oh - tile_start) * outW + ow;
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    col[col_idx] = in_b[ic * H * W + ih * W + iw];
                                } else {
                                    col[col_idx] = 0.0f;
                                }
                            }
                        }
                        col_row++;
                    }
                }
            }

            /* BLAS sgemm: tmp[out_ch, tile_pixels] = weight[out_ch, K] @ col[K, tile_pixels]
             * where K = in_ch * kH * kW */
            int K = in_ch * kH * kW;

            /* Write sgemm output directly to out_b using strided ldc.
             * Row oc of sgemm output goes to out_b[oc * outH*outW + tile_start*outW],
             * which is exactly the right position in NCHW layout. */
            float *out_tile = out_b + tile_start * outW;
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        out_ch, tile_pixels, K,
                        1.0f, weight, K,
                        col, tile_pixels,
                        0.0f, out_tile, outH * outW);
        }

        /* Add bias */
        if (bias != NULL) {
            for (int oc = 0; oc < out_ch; oc++) {
                float b_val = bias[oc];
                float *out_ch_ptr = out_b + oc * outH * outW;
                for (int i = 0; i < outH * outW; i++) {
                    out_ch_ptr[i] += b_val;
                }
            }
        }
    }

    free(col);
    return;

naive_fallback:
#endif
    /* Naive implementation (fallback) */
    for (int b = 0; b < batch; b++) {
        for (int oc = 0; oc < out_ch; oc++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    float sum = (bias != NULL) ? bias[oc] : 0.0f;

                    for (int ic = 0; ic < in_ch; ic++) {
                        for (int kh = 0; kh < kH; kh++) {
                            for (int kw = 0; kw < kW; kw++) {
                                int ih = oh * stride - padding + kh;
                                int iw = ow * stride - padding + kw;

                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    int in_idx = b * in_ch * H * W + ic * H * W + ih * W + iw;
                                    int w_idx = oc * in_ch * kH * kW + ic * kH * kW + kh * kW + kw;
                                    sum += in[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }

                    int out_idx = b * out_ch * outH * outW + oc * outH * outW + oh * outW + ow;
                    out[out_idx] = sum;
                }
            }
        }
    }
}

/* ========================================================================
 * Normalization
 * ======================================================================== */

void iris_rms_norm(float *out, const float *x, const float *weight,
                   int seq_len, int hidden, float eps) {
#ifdef USE_METAL
    /* Use GPU for RMSNorm only for very large tensors
     * The CPU-GPU sync overhead usually outweighs benefits for smaller ops */
    size_t elements = (size_t)seq_len * hidden;
    if (iris_metal_shaders_available() && elements >= 1024 * 1024) {
        iris_metal_rms_norm(out, x, weight, seq_len, hidden, eps);
        return;
    }
#endif

    for (int s = 0; s < seq_len; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        /* Compute RMS */
        float sum_sq = 0.0f;
        for (int i = 0; i < hidden; i++) {
            sum_sq += x_row[i] * x_row[i];
        }
        float rms = sqrtf(sum_sq / hidden + eps);
        float rms_inv = 1.0f / rms;

        /* Normalize and scale */
        for (int i = 0; i < hidden; i++) {
            out_row[i] = x_row[i] * rms_inv * weight[i];
        }
    }
}

void iris_group_norm(float *out, const float *x, const float *gamma, const float *beta,
                     int batch, int channels, int H, int W, int num_groups, float eps) {
    int channels_per_group = channels / num_groups;
    int spatial = H * W;

    for (int b = 0; b < batch; b++) {
        for (int g = 0; g < num_groups; g++) {
            int c_start = g * channels_per_group;
            int c_end = c_start + channels_per_group;

            float mean = 0.0f;
            int count = 0;
            for (int c = c_start; c < c_end; c++) {
                for (int i = 0; i < spatial; i++) {
                    int idx = b * channels * spatial + c * spatial + i;
                    mean += x[idx];
                    count++;
                }
            }
            mean /= count;

            float var = 0.0f;
            for (int c = c_start; c < c_end; c++) {
                for (int i = 0; i < spatial; i++) {
                    int idx = b * channels * spatial + c * spatial + i;
                    float diff = x[idx] - mean;
                    var += diff * diff;
                }
            }
            var /= count;

            float std_inv = 1.0f / sqrtf(var + eps);

            for (int c = c_start; c < c_end; c++) {
                for (int i = 0; i < spatial; i++) {
                    int idx = b * channels * spatial + c * spatial + i;
                    float norm = (x[idx] - mean) * std_inv;
                    out[idx] = gamma[c] * norm + beta[c];
                }
            }
        }
    }
}

void iris_batch_norm(float *out, const float *x,
                     const float *running_mean, const float *running_var,
                     const float *gamma, const float *beta,
                     int batch, int channels, int H, int W, float eps) {
    int spatial = H * W;

    for (int c = 0; c < channels; c++) {
        float mean = running_mean[c];
        float var = running_var[c];
        float std_inv = 1.0f / sqrtf(var + eps);
        float g = (gamma != NULL) ? gamma[c] : 1.0f;
        float b_val = (beta != NULL) ? beta[c] : 0.0f;

        for (int n = 0; n < batch; n++) {
            for (int i = 0; i < spatial; i++) {
                int idx = n * channels * spatial + c * spatial + i;
                out[idx] = g * (x[idx] - mean) * std_inv + b_val;
            }
        }
    }
}

/* ========================================================================
 * Activation Functions
 * ======================================================================== */

void iris_silu(float *x, int n) {
#ifdef USE_METAL
    /* Use GPU for very large arrays (overhead not worth it for small ones) */
    if (iris_metal_shaders_available() && n >= 4 * 1024 * 1024) {
        iris_metal_silu(x, n);
        return;
    }
#endif

    for (int i = 0; i < n; i++) {
        float val = x[i];
        x[i] = val / (1.0f + fast_expf(-val));
    }
}

void iris_gelu_tanh(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) {
        float u = x[i];
        x[i] = 0.5f * u * (1.0f + tanhf(0.7978845608028654f * (u + 0.044715f * u * u * u)));
    }
}

/* Fused SiLU(gate) * up in a single pass - avoids double memory traversal */
void iris_silu_mul(float *gate, const float *up, int n) {
#ifdef USE_METAL
    if (iris_metal_shaders_available() && n >= 4 * 1024 * 1024) {
        iris_metal_silu_mul(gate, up, n);
        return;
    }
#endif

    for (int i = 0; i < n; i++) {
        float val = gate[i];
        gate[i] = (val / (1.0f + fast_expf(-val))) * up[i];
    }
}

/* CPU-only softmax. Safe to call from worker threads (no Metal dispatch). */
void iris_softmax_cpu(float *x, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        float *row = x + r * cols;

        /* Find max for numerical stability */
        float max_val = row[0];
        for (int c = 1; c < cols; c++) {
            if (row[c] > max_val) max_val = row[c];
        }

        /* Compute exp and sum */
        float sum = 0.0f;
        for (int c = 0; c < cols; c++) {
            row[c] = fast_expf(row[c] - max_val);
            sum += row[c];
        }

        /* Normalize */
        float inv_sum = 1.0f / sum;
        for (int c = 0; c < cols; c++) {
            row[c] *= inv_sum;
        }
    }
}

void iris_softmax(float *x, int rows, int cols) {
#ifdef USE_METAL
    /* Use GPU only for very large softmax operations
     * Sync overhead usually dominates for smaller ops */
    if (iris_metal_shaders_available() && (size_t)rows * cols >= 4 * 1024 * 1024) {
        iris_metal_softmax(x, rows, cols);
        return;
    }
#endif
    iris_softmax_cpu(x, rows, cols);
}

/* ========================================================================
 * Attention Operations
 * ======================================================================== */

/* Scaled dot-product attention: softmax(Q @ K^T / sqrt(d)) @ V.
 * This is the naive implementation that materializes the full seq_q x seq_k
 * attention matrix. Used only for small sequences; the transformer's main
 * attention path uses iris_flash_attention() or the GPU kernel instead. */
void iris_attention(float *out, const float *Q, const float *K, const float *V,
                    int batch, int heads, int seq_q, int seq_k, int head_dim,
                    float scale) {
    /* Allocate attention scores */
    float *scores = (float *)malloc(seq_q * seq_k * sizeof(float));

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < heads; h++) {
            const float *q = Q + (b * heads + h) * seq_q * head_dim;
            const float *k = K + (b * heads + h) * seq_k * head_dim;
            const float *v = V + (b * heads + h) * seq_k * head_dim;
            float *o = out + (b * heads + h) * seq_q * head_dim;

            /* scores = Q @ K^T * scale */
            for (int i = 0; i < seq_q; i++) {
                for (int j = 0; j < seq_k; j++) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot += q[i * head_dim + d] * k[j * head_dim + d];
                    }
                    scores[i * seq_k + j] = dot * scale;
                }
            }

            /* softmax */
            iris_softmax(scores, seq_q, seq_k);

            /* out = scores @ V */
            for (int i = 0; i < seq_q; i++) {
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_k; j++) {
                        sum += scores[i * seq_k + j] * v[j * head_dim + d];
                    }
                    o[i * head_dim + d] = sum;
                }
            }
        }
    }

    free(scores);
}

/* ========================================================================
 * Flash Attention - Memory-Efficient Tiled Attention
 *
 * Uses online softmax algorithm to compute attention without materializing
 * the full [seq_q, seq_k] attention matrix. Reduces memory from O(n²) to O(n).
 *
 * Algorithm (for each query position):
 * 1. Initialize: max_score = -inf, sum = 0, output = 0
 * 2. For each key/value block:
 *    - Compute local scores = Q @ K^T * scale
 *    - Update running max and sum with correction factors
 *    - Accumulate weighted values into output
 * 3. Normalize: output /= sum
 *
 * Reference: "FlashAttention: Fast and Memory-Efficient Exact Attention"
 * ======================================================================== */

/*
 * Flash attention for a single head.
 * Q: [seq_q, head_dim], K: [seq_k, head_dim], V: [seq_k, head_dim]
 * out: [seq_q, head_dim]
 * Uses O(head_dim) working memory per query instead of O(seq_k).
 */
static void flash_attention_head(float *out,
                                  const float *Q, const float *K, const float *V,
                                  int seq_q, int seq_k, int head_dim, float scale) {
    /* Process each query position independently */
    for (int i = 0; i < seq_q; i++) {
        const float *q_row = Q + i * head_dim;
        float *o_row = out + i * head_dim;

        /* Running statistics for online softmax */
        float max_score = -1e30f;  /* Large negative value (avoid -INFINITY with -ffast-math) */
        float sum_exp = 0.0f;

        /* Initialize output to zero */
        for (int d = 0; d < head_dim; d++) {
            o_row[d] = 0.0f;
        }

        /* Iterate over all key/value positions */
        for (int j = 0; j < seq_k; j++) {
            const float *k_row = K + j * head_dim;
            const float *v_row = V + j * head_dim;

            /* Compute attention score: Q[i] · K[j] * scale */
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_row[d] * k_row[d];
            }
            score *= scale;

            /* Online softmax update */
            if (score > max_score) {
                /* New maximum found - rescale previous accumulations */
                float correction = fast_expf(max_score - score);
                sum_exp = sum_exp * correction + 1.0f;
                for (int d = 0; d < head_dim; d++) {
                    o_row[d] = o_row[d] * correction + v_row[d];
                }
                max_score = score;
            } else {
                /* Score is less than current max */
                float weight = fast_expf(score - max_score);
                sum_exp += weight;
                for (int d = 0; d < head_dim; d++) {
                    o_row[d] += weight * v_row[d];
                }
            }
        }

        /* Normalize by sum */
        float inv_sum = 1.0f / sum_exp;
        for (int d = 0; d < head_dim; d++) {
            o_row[d] *= inv_sum;
        }
    }
}

/*
 * Flash attention with BLAS-optimized tiling.
 * Processes queries in tiles for better cache utilization.
 * Uses BLAS for tile-level matrix operations when available.
 *
 * Q: [seq_q, head_dim], K: [seq_k, head_dim], V: [seq_k, head_dim]
 * out: [seq_q, head_dim]
 * tile_scores: scratch buffer of size [q_tile_size, k_tile_size]
 */
static void flash_attention_head_tiled(float *out,
                                        const float *Q, const float *K, const float *V,
                                        int seq_q, int seq_k, int head_dim, float scale,
                                        float *tile_scores, int q_tile_size, int k_tile_size) {
    /* Per-query running statistics: max_score[seq_q], sum_exp[seq_q] */
    float *max_scores = (float *)malloc(seq_q * sizeof(float));
    float *sum_exps = (float *)malloc(seq_q * sizeof(float));

    /* Initialize */
    for (int i = 0; i < seq_q; i++) {
        max_scores[i] = -1e30f;  /* Large negative value (avoid -INFINITY with -ffast-math) */
        sum_exps[i] = 0.0f;
    }
    memset(out, 0, seq_q * head_dim * sizeof(float));

    /* Process in tiles over K/V dimension */
    for (int k_start = 0; k_start < seq_k; k_start += k_tile_size) {
        int k_end = (k_start + k_tile_size < seq_k) ? k_start + k_tile_size : seq_k;
        int k_len = k_end - k_start;

        /* Process in tiles over Q dimension */
        for (int q_start = 0; q_start < seq_q; q_start += q_tile_size) {
            int q_end = (q_start + q_tile_size < seq_q) ? q_start + q_tile_size : seq_q;
            int q_len = q_end - q_start;

            const float *Q_tile = Q + q_start * head_dim;
            const float *K_tile = K + k_start * head_dim;
            const float *V_tile = V + k_start * head_dim;
            float *out_tile = out + q_start * head_dim;

            /* Compute tile scores: Q_tile @ K_tile^T * scale */
#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        q_len, k_len, head_dim,
                        scale, Q_tile, head_dim, K_tile, head_dim,
                        0.0f, tile_scores, k_tile_size);
#else
            for (int qi = 0; qi < q_len; qi++) {
                for (int ki = 0; ki < k_len; ki++) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot += Q_tile[qi * head_dim + d] * K_tile[ki * head_dim + d];
                    }
                    tile_scores[qi * k_tile_size + ki] = dot * scale;
                }
            }
#endif

            /* Online softmax update for this tile */
            for (int qi = 0; qi < q_len; qi++) {
                int i = q_start + qi;
                float *score_row = tile_scores + qi * k_tile_size;
                float *o_row = out_tile + qi * head_dim;

                /* Find max in this tile */
                float tile_max = score_row[0];
                for (int ki = 1; ki < k_len; ki++) {
                    if (score_row[ki] > tile_max) tile_max = score_row[ki];
                }

                /* Compute correction factors */
                float old_max = max_scores[i];
                float new_max = (tile_max > old_max) ? tile_max : old_max;

                /* Rescale old accumulations if needed */
                if (old_max > -1e29f) {  /* Check if we have prior accumulations */
                    float correction = fast_expf(old_max - new_max);
                    sum_exps[i] *= correction;
                    for (int d = 0; d < head_dim; d++) {
                        o_row[d] *= correction;
                    }
                }

                /* Accumulate this tile's contribution */
                for (int ki = 0; ki < k_len; ki++) {
                    float weight = fast_expf(score_row[ki] - new_max);
                    sum_exps[i] += weight;
                    const float *v_row = V_tile + ki * head_dim;
                    for (int d = 0; d < head_dim; d++) {
                        o_row[d] += weight * v_row[d];
                    }
                }

                max_scores[i] = new_max;
            }
        }
    }

    /* Final normalization */
    for (int i = 0; i < seq_q; i++) {
        float inv_sum = 1.0f / sum_exps[i];
        float *o_row = out + i * head_dim;
        for (int d = 0; d < head_dim; d++) {
            o_row[d] *= inv_sum;
        }
    }

    free(max_scores);
    free(sum_exps);
}

/*
 * Flash attention for multi-head attention.
 * Works on [seq, heads*head_dim] layout (same as transformer tensors).
 *
 * Q: [seq_q, heads * head_dim]
 * K: [seq_k, heads * head_dim]
 * V: [seq_k, heads * head_dim]
 * out: [seq_q, heads * head_dim]
 *
 * Memory usage: O(seq_q + tile_size²) instead of O(seq_q * seq_k)
 */
void iris_flash_attention(float *out, const float *Q, const float *K, const float *V,
                          int seq_q, int seq_k, int heads, int head_dim, float scale) {
    /* Tile sizes for cache efficiency */
    int q_tile_size = 32;  /* Process 32 queries at a time */
    int k_tile_size = 64;  /* Process 64 keys at a time */

    /* Allocate tile scratch buffer */
    float *tile_scores = (float *)malloc(q_tile_size * k_tile_size * sizeof(float));

    /* Process each head */
    for (int h = 0; h < heads; h++) {
        const float *Q_head = Q + h * head_dim;
        const float *K_head = K + h * head_dim;
        const float *V_head = V + h * head_dim;
        float *out_head = out + h * head_dim;

        /* Stride between consecutive positions for this head */
        int hidden = heads * head_dim;

        /* For small sequences, use simple non-tiled version */
        if (seq_q <= 64 && seq_k <= 128) {
            /* Extract head data into contiguous buffers */
            float *Q_contig = (float *)malloc(seq_q * head_dim * sizeof(float));
            float *K_contig = (float *)malloc(seq_k * head_dim * sizeof(float));
            float *V_contig = (float *)malloc(seq_k * head_dim * sizeof(float));
            float *out_contig = (float *)malloc(seq_q * head_dim * sizeof(float));

            for (int i = 0; i < seq_q; i++) {
                for (int d = 0; d < head_dim; d++) {
                    Q_contig[i * head_dim + d] = Q_head[i * hidden + d];
                }
            }
            for (int j = 0; j < seq_k; j++) {
                for (int d = 0; d < head_dim; d++) {
                    K_contig[j * head_dim + d] = K_head[j * hidden + d];
                    V_contig[j * head_dim + d] = V_head[j * hidden + d];
                }
            }

            flash_attention_head(out_contig, Q_contig, K_contig, V_contig,
                                 seq_q, seq_k, head_dim, scale);

            /* Copy back with stride */
            for (int i = 0; i < seq_q; i++) {
                for (int d = 0; d < head_dim; d++) {
                    out_head[i * hidden + d] = out_contig[i * head_dim + d];
                }
            }

            free(Q_contig);
            free(K_contig);
            free(V_contig);
            free(out_contig);
        } else {
            /* For larger sequences, use tiled version with strided access */
            /* Extract head data into contiguous buffers for BLAS efficiency */
            float *Q_contig = (float *)malloc(seq_q * head_dim * sizeof(float));
            float *K_contig = (float *)malloc(seq_k * head_dim * sizeof(float));
            float *V_contig = (float *)malloc(seq_k * head_dim * sizeof(float));
            float *out_contig = (float *)malloc(seq_q * head_dim * sizeof(float));

            for (int i = 0; i < seq_q; i++) {
                for (int d = 0; d < head_dim; d++) {
                    Q_contig[i * head_dim + d] = Q_head[i * hidden + d];
                }
            }
            for (int j = 0; j < seq_k; j++) {
                for (int d = 0; d < head_dim; d++) {
                    K_contig[j * head_dim + d] = K_head[j * hidden + d];
                    V_contig[j * head_dim + d] = V_head[j * hidden + d];
                }
            }

            flash_attention_head_tiled(out_contig, Q_contig, K_contig, V_contig,
                                        seq_q, seq_k, head_dim, scale,
                                        tile_scores, q_tile_size, k_tile_size);

            /* Copy back with stride */
            for (int i = 0; i < seq_q; i++) {
                for (int d = 0; d < head_dim; d++) {
                    out_head[i * hidden + d] = out_contig[i * head_dim + d];
                }
            }

            free(Q_contig);
            free(K_contig);
            free(V_contig);
            free(out_contig);
        }
    }

    free(tile_scores);
}

/* Apply precomputed RoPE (Rotary Position Embedding) in-place using the
 * split-half convention: dim d pairs with dim d+half for rotation. This is
 * the Flux convention (4-axis, split-half); Z-Image uses consecutive pairs
 * via a separate kernel. RoPE lets the transformer learn relative position
 * from the dot-product structure of Q and K. */
void iris_apply_rope(float *x, const float *freqs,
                     int batch, int seq, int heads, int head_dim) {
    /* x: [batch, seq, heads, head_dim]
     * freqs: [seq, head_dim/2, 2] (cos, sin)
     * Apply rotary embedding to pairs of dimensions */

    int half_dim = head_dim / 2;

    for (int b = 0; b < batch; b++) {
        for (int s = 0; s < seq; s++) {
            for (int h = 0; h < heads; h++) {
                float *vec = x + ((b * seq + s) * heads + h) * head_dim;

                for (int d = 0; d < half_dim; d++) {
                    float cos_val = freqs[s * half_dim * 2 + d * 2];
                    float sin_val = freqs[s * half_dim * 2 + d * 2 + 1];

                    float x0 = vec[d];
                    float x1 = vec[d + half_dim];

                    vec[d] = x0 * cos_val - x1 * sin_val;
                    vec[d + half_dim] = x0 * sin_val + x1 * cos_val;
                }
            }
        }
    }
}

void iris_compute_rope_freqs(float *freqs, const int *pos, int seq, int dim, float theta) {
    int half_dim = dim / 2;

    for (int s = 0; s < seq; s++) {
        float p = (float)pos[s];
        for (int d = 0; d < half_dim; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / (float)dim);
            float angle = p * freq;
            freqs[s * half_dim * 2 + d * 2] = cosf(angle);
            freqs[s * half_dim * 2 + d * 2 + 1] = sinf(angle);
        }
    }
}

/* ========================================================================
 * Pooling and Reshape
 * ======================================================================== */

void iris_upsample_nearest(float *out, const float *in,
                           int batch, int channels, int H, int W,
                           int scale_h, int scale_w) {
    int outH = H * scale_h;
    int outW = W * scale_w;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int oh = 0; oh < outH; oh++) {
                for (int ow = 0; ow < outW; ow++) {
                    int ih = oh / scale_h;
                    int iw = ow / scale_w;
                    int in_idx = b * channels * H * W + c * H * W + ih * W + iw;
                    int out_idx = b * channels * outH * outW + c * outH * outW + oh * outW + ow;
                    out[out_idx] = in[in_idx];
                }
            }
        }
    }
}

/* Convert spatial latent to patch tokens for the diffusion transformer.
 * Groups each ps x ps spatial block into a single token vector:
 * [batch, channels, H, W] -> [batch, channels*ps*ps, H/ps, W/ps].
 * The transformer operates on these patch tokens, not individual spatial
 * positions, reducing sequence length by ps*ps (4x for ps=2). */
void iris_patchify(float *out, const float *in,
                   int batch, int channels, int H, int W, int patch_size) {
    /* [B, C, H, W] -> [B, C*p*p, H/p, W/p] */
    int p = patch_size;
    int outH = H / p;
    int outW = W / p;
    int out_ch = channels * p * p;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int ph = 0; ph < outH; ph++) {
                for (int pw = 0; pw < outW; pw++) {
                    for (int pi = 0; pi < p; pi++) {
                        for (int pj = 0; pj < p; pj++) {
                            int ih = ph * p + pi;
                            int iw = pw * p + pj;
                            int in_idx = b * channels * H * W + c * H * W + ih * W + iw;

                            int out_c = c * p * p + pi * p + pj;
                            int out_idx = b * out_ch * outH * outW + out_c * outH * outW + ph * outW + pw;
                            out[out_idx] = in[in_idx];
                        }
                    }
                }
            }
        }
    }
}

void iris_unpatchify(float *out, const float *in,
                     int batch, int channels, int H, int W, int patch_size) {
    /* [B, C*p*p, H, W] -> [B, C, H*p, W*p] */
    int p = patch_size;
    int in_ch = channels * p * p;
    int outH = H * p;
    int outW = W * p;

    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int ph = 0; ph < H; ph++) {
                for (int pw = 0; pw < W; pw++) {
                    for (int pi = 0; pi < p; pi++) {
                        for (int pj = 0; pj < p; pj++) {
                            int in_c = c * p * p + pi * p + pj;
                            int in_idx = b * in_ch * H * W + in_c * H * W + ph * W + pw;

                            int oh = ph * p + pi;
                            int ow = pw * p + pj;
                            int out_idx = b * channels * outH * outW + c * outH * outW + oh * outW + ow;
                            out[out_idx] = in[in_idx];
                        }
                    }
                }
            }
        }
    }
}

/* ========================================================================
 * Utility Functions
 * ======================================================================== */

void iris_copy(float *dst, const float *src, int n) {
    memcpy(dst, src, n * sizeof(float));
}
