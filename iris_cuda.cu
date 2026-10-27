/*
 * Iris device backend for NVIDIA (CUDA) and AMD (ROCm/HIP) GPUs.
 *
 * Implements iris_dev.h. The same source builds with nvcc, or with hipcc
 * when IRIS_HIP is defined: the few runtime and BLAS names that differ are
 * mapped below. Matrix multiplications use cuBLAS / hipBLAS with bf16
 * inputs and f32 accumulation (tensor cores / WMMA); everything else is a
 * small set of portable kernels operating on row-major tensors.
 */

#include "iris_dev.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef IRIS_HIP
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#define GPU(name) hip##name
typedef hipblasHandle_t blas_handle_t;
typedef hipblasStatus_t blas_status_t;
typedef hipDataType blas_dtype_t;
typedef hipDeviceProp_t dev_prop_t;
#define BLAS_OP_N HIPBLAS_OP_N
#define BLAS_OP_T HIPBLAS_OP_T
#define BLAS_R_16BF HIP_R_16BF
#define BLAS_R_32F HIP_R_32F
#define BLAS_COMPUTE_32F HIPBLAS_COMPUTE_32F
#define BLAS_GEMM_DEFAULT HIPBLAS_GEMM_DEFAULT
#define BLAS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define blasCreate hipblasCreate
#define blasDestroy hipblasDestroy
#define blasSetStream hipblasSetStream
#define blasGemmEx hipblasGemmEx
#define blasGemmStridedBatchedEx hipblasGemmStridedBatchedEx
#else
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define GPU(name) cuda##name
typedef cublasHandle_t blas_handle_t;
typedef cublasStatus_t blas_status_t;
typedef cudaDataType_t blas_dtype_t;
typedef cudaDeviceProp dev_prop_t;
#define BLAS_OP_N CUBLAS_OP_N
#define BLAS_OP_T CUBLAS_OP_T
#define BLAS_R_16BF CUDA_R_16BF
#define BLAS_R_32F CUDA_R_32F
#define BLAS_COMPUTE_32F CUBLAS_COMPUTE_32F
#define BLAS_GEMM_DEFAULT CUBLAS_GEMM_DEFAULT
#define BLAS_SUCCESS CUBLAS_STATUS_SUCCESS
#define blasCreate cublasCreate
#define blasDestroy cublasDestroy
#define blasSetStream cublasSetStream
#define blasGemmEx cublasGemmEx
#define blasGemmStridedBatchedEx cublasGemmStridedBatchedEx
#endif

#define ROW_THREADS 256     /* Threads for one-row-per-block kernels */
#define EW_THREADS 256      /* Threads for elementwise kernels */
#define ATTN_SCRATCH ((size_t)1 << 30)  /* Score buffer budget (bytes) */
#define CONV_SCRATCH ((size_t)256 << 20) /* im2col buffer budget (bytes) */
#define GN_BLOCKS 64                     /* Blocks per group for GroupNorm stats */
#define STAGE_BYTES ((size_t)64 << 20)   /* Pinned upload staging buffer (CUDA, each of 2) */

/* Flash attention tiling */
#define FA_BR 128                        /* queries per block (16 per wave / warp) */
#define FA_THREADS (FA_BR * 2)
#define FA_BC 64                         /* keys per tile */
#define FA_D 128                         /* head dim */
#define FA_PAD 8                         /* shared row padding (bf16): no bank conflicts */
#define FA_STRIDE (FA_D + FA_PAD)        /* shared row stride (bf16) */
#define FA_SMEM (4 * FA_BC * FA_STRIDE * 2)  /* CUDA: 2 stages of K and V tiles */

static int g_ready = 0;
static int g_flash = 0;      /* flash attention kernel available on this GPU */
static char g_name[256] = "none";
static GPU(Stream_t) g_stream;
static blas_handle_t g_blas;

#ifndef IRIS_HIP
/* CUDA uploads go through two pinned staging buffers: the CPU fills one
 * while the DMA engine copies the other. Pageable copies from mmapped
 * files can be very slow (~0.5-3 GB/s on a GB10). The HIP runtime's own
 * pageable copies are faster than staging on the GPUs tested. */
static void *g_stage[2] = {NULL, NULL};
static GPU(Event_t) g_stage_done[2];
#endif

/* Attention scratch (scores f32 + probabilities bf16), grown on demand. */
static void *g_attn_scratch = NULL;
static size_t g_attn_scratch_size = 0;

#ifdef IRIS_HIP
/* bf16 GEMM output later widened to f32, see iris_dev_linear_f32(). */
static uint16_t *g_gemm_scratch = NULL;
static size_t g_gemm_scratch_elems = 0;
#endif

/* Convolution scratch (im2col columns, bf16) and GroupNorm statistics. */
static uint16_t *g_conv_scratch = NULL;
static float *g_gn_scratch = NULL;    /* partial sums, then mean/rstd */
static size_t g_gn_scratch_size = 0;

__global__ void k_flash_attention(uint16_t *out, int ldo, const uint16_t *q, const uint16_t *k,
                                  const uint16_t *v, int ldi, int seq_q, int seq_k,
                                  float scale_log2, const int *key_mask);

static void check(GPU(Error_t) err, const char *what) {
    if (err != GPU(Success))
        fprintf(stderr, "iris_dev: %s failed: %s\n", what, GPU(GetErrorString)(err));
}

/* ========================================================================
 * Setup and Memory
 * ======================================================================== */

extern "C" int iris_dev_init(void) {
    if (g_ready) return 1;
    int count = 0;
    if (GPU(GetDeviceCount)(&count) != GPU(Success) || count == 0) return 0;
    dev_prop_t prop;
    if (GPU(GetDeviceProperties)(&prop, 0) != GPU(Success)) return 0;
    snprintf(g_name, sizeof(g_name), "%s", prop.name);
#ifdef IRIS_HIP
    /* The HIP flash attention kernel uses the RDNA3 (gfx11) WMMA layout. */
    g_flash = strncmp(prop.gcnArchName, "gfx11", 5) == 0;
#else
    /* mma.sync bf16 / cp.async / ldmatrix, and more than 48 KB of shared memory. */
    g_flash = prop.major >= 8 &&
              cudaFuncSetAttribute(k_flash_attention, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                   FA_SMEM) == cudaSuccess;
#endif
    if (GPU(StreamCreate)(&g_stream) != GPU(Success)) return 0;
#ifndef IRIS_HIP
    for (int i = 0; i < 2; i++) {
        if (cudaMallocHost(&g_stage[i], STAGE_BYTES) != cudaSuccess) return 0;
        if (cudaEventCreate(&g_stage_done[i]) != cudaSuccess) return 0;
    }
#endif
    if (blasCreate(&g_blas) != BLAS_SUCCESS) return 0;
    blasSetStream(g_blas, g_stream);
    g_ready = 1;
    return 1;
}

extern "C" int iris_dev_available(void) { return g_ready; }
/* Both flash attention and the GEMM fallback bound the score scratch. */
extern "C" int iris_dev_has_memory_efficient_attention(void) { return g_ready; }
extern "C" const char *iris_dev_name(void) { return g_name; }

extern "C" void iris_dev_cleanup(void) {
    if (!g_ready) return;
    iris_dev_free(g_attn_scratch);
    g_attn_scratch = NULL;
    g_attn_scratch_size = 0;
    iris_dev_free(g_conv_scratch);
    g_conv_scratch = NULL;
#ifdef IRIS_HIP
    iris_dev_free(g_gemm_scratch);
    g_gemm_scratch = NULL;
    g_gemm_scratch_elems = 0;
#endif
    iris_dev_free(g_gn_scratch);
    g_gn_scratch = NULL;
    g_gn_scratch_size = 0;
    blasDestroy(g_blas);
#ifndef IRIS_HIP
    for (int i = 0; i < 2; i++) {
        check(cudaFreeHost(g_stage[i]), "free host");
        check(cudaEventDestroy(g_stage_done[i]), "event destroy");
    }
#endif
    check(GPU(StreamDestroy)(g_stream), "stream destroy");
    g_ready = 0;
}

extern "C" void *iris_dev_alloc(size_t bytes) {
    void *p = NULL;
    if (GPU(Malloc)(&p, bytes ? bytes : 1) != GPU(Success)) return NULL;
    return p;
}

extern "C" void iris_dev_free(void *p) {
    if (!p) return;
    /* Pending work may still use the buffer. */
    check(GPU(StreamSynchronize)(g_stream), "sync");
    check(GPU(Free)(p), "free");
}

/* Upload rows of row_bytes bytes. CUDA packs whole rows into the staging
 * buffers (a row larger than a staging buffer is split in pieces). */
extern "C" void iris_dev_upload_2d(void *dst, size_t dst_stride, const void *src,
                                   size_t src_stride, size_t row_bytes, size_t rows) {
#ifdef IRIS_HIP
    check(hipMemcpy2DAsync(dst, dst_stride, src, src_stride, row_bytes, rows,
                           hipMemcpyHostToDevice, g_stream), "upload");
#else
    size_t piece = row_bytes < STAGE_BYTES ? row_bytes : STAGE_BYTES;
    size_t pieces_per_row = (row_bytes + piece - 1) / piece;
    size_t rows_per_chunk = row_bytes <= STAGE_BYTES ? STAGE_BYTES / row_bytes : 1;
    int buf = 0;
    for (size_t r0 = 0; r0 < rows; r0 += rows_per_chunk) {
        size_t nr = rows - r0 < rows_per_chunk ? rows - r0 : rows_per_chunk;
        for (size_t p = 0; p < pieces_per_row; p++) {
            size_t off = p * piece, len = row_bytes - off < piece ? row_bytes - off : piece;
            /* Wait until the DMA of the previous use of this buffer is done. */
            check(GPU(EventSynchronize)(g_stage_done[buf]), "event sync");
            char *stage = (char *)g_stage[buf];
            for (size_t r = 0; r < nr; r++)
                memcpy(stage + r * len, (const char *)src + (r0 + r) * src_stride + off, len);
            check(GPU(Memcpy2DAsync)((char *)dst + r0 * dst_stride + off, dst_stride, stage, len,
                                     len, nr, GPU(MemcpyHostToDevice), g_stream), "upload");
            check(GPU(EventRecord)(g_stage_done[buf], g_stream), "event record");
            buf ^= 1;
        }
    }
#endif
    check(GPU(StreamSynchronize)(g_stream), "sync");
}

extern "C" void iris_dev_upload(void *dst, const void *src, size_t bytes) {
    if (bytes) iris_dev_upload_2d(dst, bytes, src, bytes, bytes, 1);
}

extern "C" void iris_dev_download(void *dst, const void *src, size_t bytes) {
    check(GPU(MemcpyAsync)(dst, src, bytes, GPU(MemcpyDeviceToHost), g_stream), "download");
    check(GPU(StreamSynchronize)(g_stream), "sync");
}

extern "C" void iris_dev_sync(void) {
    check(GPU(StreamSynchronize)(g_stream), "sync");
}

/* ========================================================================
 * Kernel Helpers
 * ======================================================================== */

__device__ __forceinline__ float bf2f(uint16_t v) {
    return __uint_as_float((uint32_t)v << 16);
}

/* Round to nearest even. */
__device__ __forceinline__ uint16_t f2bf(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7FFF + ((u >> 16) & 1);
    return (uint16_t)(u >> 16);
}

/* Sum (or max) over the block; blockDim.x must be a power of two. */
__device__ float block_reduce(float v, float *sh, int is_max) {
    sh[threadIdx.x] = v;
    __syncthreads();
    for (unsigned s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            float o = sh[threadIdx.x + s];
            sh[threadIdx.x] = is_max ? fmaxf(sh[threadIdx.x], o) : sh[threadIdx.x] + o;
        }
        __syncthreads();
    }
    float r = sh[0];
    __syncthreads();
    return r;
}

static unsigned ew_blocks(size_t n) {
    return (unsigned)((n + EW_THREADS - 1) / EW_THREADS);
}

/* ========================================================================
 * Elementwise Kernels
 * ======================================================================== */

__global__ void k_f32_to_bf16(uint16_t *dst, const float *src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = f2bf(src[i]);
}

/* dst[i] = sum of the bf16 partial results src[p * n + i], p < parts. */
__global__ void k_sum_bf16(float *dst, const uint16_t *src, int parts, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = 0.0f;
    for (int p = 0; p < parts; p++) s += bf2f(src[p * n + i]);
    dst[i] = s;
}

__global__ void k_silu(float *x, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = x[i] / (1.0f + expf(-x[i]));
}

__global__ void k_add(float *x, const float *a, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += a[i];
}

__global__ void k_fill_rows(float *out, const float *row, int dim, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = row[i % dim];
}

__global__ void k_swiglu(uint16_t *out, int ldo, const uint16_t *gate,
                         const uint16_t *up, int ldi, int cols, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    size_t r = i / cols, c = i % cols;
    float g = bf2f(gate[r * ldi + c]);
    out[r * ldo + c] = f2bf(g / (1.0f + expf(-g)) * bf2f(up[r * ldi + c]));
}

__global__ void k_residual(float *x, const float *a, const float *gate,
                           int dim, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] += (gate ? gate[i % dim] : 1.0f) * a[i];
}

extern "C" void iris_dev_f32_to_bf16(uint16_t *dst, const float *src, size_t n) {
    if (n) k_f32_to_bf16<<<ew_blocks(n), EW_THREADS, 0, g_stream>>>(dst, src, n);
}

extern "C" void iris_dev_silu(float *x, size_t n) {
    if (n) k_silu<<<ew_blocks(n), EW_THREADS, 0, g_stream>>>(x, n);
}

extern "C" void iris_dev_add(float *x, const float *a, size_t n) {
    if (n) k_add<<<ew_blocks(n), EW_THREADS, 0, g_stream>>>(x, a, n);
}

extern "C" void iris_dev_fill_rows(float *out, const float *row, int rows, int dim) {
    size_t n = (size_t)rows * dim;
    if (n) k_fill_rows<<<ew_blocks(n), EW_THREADS, 0, g_stream>>>(out, row, dim, n);
}

extern "C" void iris_dev_swiglu(uint16_t *out, int ldo, const uint16_t *gate,
                                const uint16_t *up, int ldi, int rows, int cols) {
    size_t n = (size_t)rows * cols;
    if (n) k_swiglu<<<ew_blocks(n), EW_THREADS, 0, g_stream>>>(out, ldo, gate, up, ldi, cols, n);
}

extern "C" void iris_dev_residual(float *x, const float *a, const float *gate,
                                  int rows, int dim) {
    size_t n = (size_t)rows * dim;
    if (n) k_residual<<<ew_blocks(n), EW_THREADS, 0, g_stream>>>(x, a, gate, dim, n);
}

/* ========================================================================
 * Normalization Kernels
 * ======================================================================== */

__global__ void k_layer_norm_mod(uint16_t *out, const float *x, const float *shift,
                                 const float *scale, int dim, float eps) {
    __shared__ float sh[ROW_THREADS];
    const float *xr = x + (size_t)blockIdx.x * dim;
    uint16_t *o = out + (size_t)blockIdx.x * dim;

    float s = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i];
    float mean = block_reduce(s, sh, 0) / dim;
    float v = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float d = xr[i] - mean;
        v += d * d;
    }
    float inv = rsqrtf(block_reduce(v, sh, 0) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float n = (xr[i] - mean) * inv;
        if (scale) n *= 1.0f + scale[i];
        if (shift) n += shift[i];
        o[i] = f2bf(n);
    }
}

__global__ void k_rms_norm(uint16_t *out, const float *x, const float *w,
                           const float *scale, int dim, float eps) {
    __shared__ float sh[ROW_THREADS];
    const float *xr = x + (size_t)blockIdx.x * dim;
    uint16_t *o = out + (size_t)blockIdx.x * dim;

    float s = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i] * xr[i];
    float inv = rsqrtf(block_reduce(s, sh, 0) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = xr[i] * inv * w[i];
        if (scale) v *= 1.0f + scale[i];
        o[i] = f2bf(v);
    }
}

__global__ void k_rms_norm_residual(float *x, const float *a, const float *w,
                                    const float *gate, int dim, float eps) {
    __shared__ float sh[ROW_THREADS];
    const float *ar = a + (size_t)blockIdx.x * dim;
    float *xr = x + (size_t)blockIdx.x * dim;
    float s = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) s += ar[i] * ar[i];
    float inv = rsqrtf(block_reduce(s, sh, 0) / dim + eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float v = ar[i] * inv * w[i];
        xr[i] += (gate ? tanhf(gate[i]) : 1.0f) * v;
    }
}

/* One block per (row, head), one thread per rotated pair. */
__global__ void k_head_norm_rope(uint16_t *x, int ld, const float *w,
                                 const float *rope_cos, const float *rope_sin,
                                 int head_dim, float eps, int split_half) {
    extern __shared__ float sh[];
    int s = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    int half = head_dim / 2;
    uint16_t *v = x + (size_t)s * ld + (size_t)h * head_dim;
    int i0 = split_half ? i : 2 * i;
    int i1 = split_half ? i + half : 2 * i + 1;

    float a = bf2f(v[i0]), b = bf2f(v[i1]);
    float inv = rsqrtf(block_reduce(a * a + b * b, sh, 0) / head_dim + eps);
    a *= inv * w[i0];
    b *= inv * w[i1];
    float c = rope_cos[(size_t)s * half + i], sn = rope_sin[(size_t)s * half + i];
    v[i0] = f2bf(a * c - b * sn);
    v[i1] = f2bf(b * c + a * sn);
}

extern "C" void iris_dev_layer_norm_mod(uint16_t *out, const float *x,
                                        const float *shift, const float *scale,
                                        int rows, int dim, float eps) {
    if (rows > 0)
        k_layer_norm_mod<<<rows, ROW_THREADS, 0, g_stream>>>(out, x, shift, scale, dim, eps);
}

extern "C" void iris_dev_rms_norm(uint16_t *out, const float *x, const float *w,
                                  int rows, int dim, float eps) {
    if (rows > 0) k_rms_norm<<<rows, ROW_THREADS, 0, g_stream>>>(out, x, w, NULL, dim, eps);
}

extern "C" void iris_dev_rms_norm_mod(uint16_t *out, const float *x, const float *w,
                                      const float *scale, int rows, int dim, float eps) {
    if (rows > 0) k_rms_norm<<<rows, ROW_THREADS, 0, g_stream>>>(out, x, w, scale, dim, eps);
}

extern "C" void iris_dev_rms_norm_residual(float *x, const float *a, const float *w,
                                           const float *gate, int rows, int dim, float eps) {
    if (rows > 0)
        k_rms_norm_residual<<<rows, ROW_THREADS, 0, g_stream>>>(x, a, w, gate, dim, eps);
}

extern "C" void iris_dev_head_norm_rope(uint16_t *x, int ld, const float *w,
                                        const float *rope_cos, const float *rope_sin,
                                        int seq, int heads, int head_dim, float eps,
                                        int split_half) {
    if (seq <= 0) return;
    int threads = head_dim / 2;   /* power of two for block_reduce */
    k_head_norm_rope<<<dim3(seq, heads), threads, threads * sizeof(float), g_stream>>>(
        x, ld, w, rope_cos, rope_sin, head_dim, eps, split_half);
}

/* ========================================================================
 * Matrix Multiplication
 * ======================================================================== */

/* Row-major y[rows, out] = x[rows, in] @ W[out, in]^T, i.e. column-major
 * y^T = W^T(T) * x^T. With split > 1, slice p of the reduction (columns
 * [p * in / split, (p + 1) * in / split) of x and W) goes to the partial
 * result y + p * rows * out. */
static void linear(void *y, blas_dtype_t y_type, const uint16_t *x, const uint16_t *W,
                   int rows, int in, int out, int split) {
    const float one = 1.0f, zero = 0.0f;
    int k = in / split;
    blas_status_t st = split == 1 ?
        blasGemmEx(g_blas, BLAS_OP_T, BLAS_OP_N, out, rows, in, &one,
                   W, BLAS_R_16BF, in, x, BLAS_R_16BF, in, &zero,
                   y, y_type, out, BLAS_COMPUTE_32F, BLAS_GEMM_DEFAULT) :
        blasGemmStridedBatchedEx(g_blas, BLAS_OP_T, BLAS_OP_N, out, rows, k, &one,
                   W, BLAS_R_16BF, in, k, x, BLAS_R_16BF, in, k, &zero,
                   y, y_type, out, (long long)rows * out, split,
                   BLAS_COMPUTE_32F, BLAS_GEMM_DEFAULT);
    if (st != BLAS_SUCCESS)
        fprintf(stderr, "iris_dev: GEMM %dx%dx%d failed\n", rows, out, in);
}

extern "C" void iris_dev_linear_f32(float *y, const uint16_t *x, const uint16_t *W,
                                    int rows, int in, int out) {
#ifdef IRIS_HIP
    /* rocBLAS has no fast bf16 -> f32 GEMM kernels on every GPU (4x slower
     * on gfx1151), so produce bf16 and widen it. Long reductions are split
     * in slices of at least 3072, summed in f32: rocBLAS picks much faster
     * kernels for them (gfx1151, 4176 x 3072 x 12288: 18.2 -> 13.5 ms). */
    int split = 1;
    while (in % (16 * split) == 0 && in / (2 * split) >= 3072) split *= 2;
    size_t n = (size_t)rows * out;
    if (split * n > g_gemm_scratch_elems) {
        iris_dev_free(g_gemm_scratch);
        g_gemm_scratch = (uint16_t *)iris_dev_alloc(split * n * sizeof(uint16_t));
        g_gemm_scratch_elems = g_gemm_scratch ? split * n : 0;
    }
    if (g_gemm_scratch) {
        linear(g_gemm_scratch, BLAS_R_16BF, x, W, rows, in, out, split);
        k_sum_bf16<<<ew_blocks(n), EW_THREADS, 0, g_stream>>>(y, g_gemm_scratch, split, n);
        return;
    }
#endif
    linear(y, BLAS_R_32F, x, W, rows, in, out, 1);
}

extern "C" void iris_dev_linear_bf16(uint16_t *y, const uint16_t *x, const uint16_t *W,
                                     int rows, int in, int out) {
    linear(y, BLAS_R_16BF, x, W, rows, in, out, 1);
}

/* ========================================================================
 * Attention
 * ======================================================================== */

/* Softmax over each row of scores S [rows, seq_k] (f32) into P (bf16).
 * Row r is query q0 + r % nq of its head. */
__global__ void k_softmax(uint16_t *P, const float *S, int nq, int seq_k, int q0,
                          int causal, const int *key_mask) {
    __shared__ float sh[ROW_THREADS];
    size_t row = blockIdx.x;
    int qi = q0 + (int)(row % nq);
    const float *s = S + row * seq_k;
    uint16_t *p = P + row * seq_k;
    int limit = causal ? (qi + 1 < seq_k ? qi + 1 : seq_k) : seq_k;

    float m = -INFINITY;
    for (int j = threadIdx.x; j < limit; j += blockDim.x)
        if (!key_mask || key_mask[j]) m = fmaxf(m, s[j]);
    m = block_reduce(m, sh, 1);
    float sum = 0;
    for (int j = threadIdx.x; j < limit; j += blockDim.x)
        if (!key_mask || key_mask[j]) sum += expf(s[j] - m);
    sum = block_reduce(sum, sh, 0);
    float inv = sum > 0 ? 1.0f / sum : 0.0f;
    for (int j = threadIdx.x; j < seq_k; j += blockDim.x) {
        int valid = j < limit && (!key_mask || key_mask[j]);
        p[j] = f2bf(valid ? expf(s[j] - m) * inv : 0.0f);
    }
}


__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
    return (uint32_t)f2bf(lo) | ((uint32_t)f2bf(hi) << 16);
}

#ifdef IRIS_HIP
/* ========================================================================
 * Flash Attention (ROCm, RDNA3 WMMA)
 *
 * 128 queries per block, one wave of 32 per 16 queries, 64-key tiles in
 * LDS (the next one is fetched into registers while the current one is
 * processed), adapted to the gfx11 WMMA layout:
 * A and B operands hold a row / column per lane (replicated in both
 * half-waves), the accumulator holds column (lane % 16), rows
 * 2e + lane / 16. Computing the transposed products S^T = K Q^T and
 * O^T += V^T P^T puts one query in each lane: the online softmax state
 * is per lane, and P^T becomes the next B operand with one exchange
 * between the half-waves.
 * ======================================================================== */

#define FA_LOADS (FA_BC * FA_D / 8 / FA_THREADS)   /* 16-byte loads per thread per tile */

/* Each thread's share of a 64-key tile, staged in registers so that the
 * global loads of the next tile overlap the work on the current one. */
__device__ __forceinline__ void fa_fetch_tile(uint4 *kr, uint4 *vr, const uint16_t *k,
                                              const uint16_t *v, int ldi, size_t hoff,
                                              int k0, int seq_k) {
    for (int c = 0; c < FA_LOADS; c++) {
        int i = threadIdx.x + c * FA_THREADS, r = i / (FA_D / 8), c8 = (i % (FA_D / 8)) * 8;
        int key = k0 + r;
        kr[c] = vr[c] = make_uint4(0, 0, 0, 0);
        if (key < seq_k) {
            kr[c] = *(const uint4 *)(k + (size_t)key * ldi + hoff + c8);
            vr[c] = *(const uint4 *)(v + (size_t)key * ldi + hoff + c8);
        }
    }
}

/* Store a fetched tile: keys row-major into ks, values transposed into vt. */
__device__ __forceinline__ void fa_store_tile(uint16_t (*ks)[FA_D + FA_PAD],
                                              uint16_t (*vt)[FA_BC + FA_PAD],
                                              const uint4 *kr, const uint4 *vr) {
    for (int c = 0; c < FA_LOADS; c++) {
        int i = threadIdx.x + c * FA_THREADS, r = i / (FA_D / 8), c8 = (i % (FA_D / 8)) * 8;
        *(uint4 *)&ks[r][c8] = kr[c];
        const uint16_t *ve = (const uint16_t *)&vr[c];
        for (int j = 0; j < 8; j++) vt[c8 + j][r] = ve[j];
    }
}

/* The value of the lane in the other half-wave (lane ^ 16), without LDS. */
__device__ __forceinline__ float fa_swap_halves(float v) {
    return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(v), __float_as_int(v),
                                                       0x76543210, 0xfedcba98, false, false));
}

typedef short fa_bf16x16 __attribute__((ext_vector_type(16)));
typedef float fa_f32x8 __attribute__((ext_vector_type(8)));

__global__ void __launch_bounds__(FA_THREADS)
k_flash_attention(uint16_t *out, int ldo, const uint16_t *q, const uint16_t *k,
                  const uint16_t *v, int ldi, int seq_q, int seq_k, float scale_log2,
                  const int *key_mask) {
#if defined(__GFX11__)
    __shared__ __align__(16) uint16_t ks[FA_BC][FA_D + FA_PAD];
    __shared__ __align__(16) uint16_t vt[FA_D][FA_BC + FA_PAD];
    int wave = threadIdx.x / 32, lane = threadIdx.x % 32;
    int col = lane % 16, half = lane / 16;
    size_t hoff = (size_t)blockIdx.y * FA_D;
    int qrow = blockIdx.x * FA_BR + wave * 16 + col;   /* this lane's query */

    /* Q^T as B operands: column = query, k-step kk = dims [16kk, 16kk + 16). */
    fa_bf16x16 qb[8];
    for (int kk = 0; kk < 8; kk++) {
        if (qrow < seq_q) qb[kk] = *(const fa_bf16x16 *)(q + (size_t)qrow * ldi + hoff + kk * 16);
        else for (int e = 0; e < 16; e++) qb[kk][e] = 0;
    }

    fa_f32x8 o[8];   /* O^T: dim tile dt, rows dt*16 + 2e + half, column = query */
    for (int dt = 0; dt < 8; dt++) for (int e = 0; e < 8; e++) o[dt][e] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    uint4 kr[FA_LOADS], vr[FA_LOADS];
    fa_fetch_tile(kr, vr, k, v, ldi, hoff, 0, seq_k);
    for (int k0 = 0; k0 < seq_k; k0 += FA_BC) {
        __syncthreads();
        fa_store_tile(ks, vt, kr, vr);
        __syncthreads();
        if (k0 + FA_BC < seq_k) fa_fetch_tile(kr, vr, k, v, ldi, hoff, k0 + FA_BC, seq_k);

        /* Bit i: key k0 + i is in range and not masked. Wave-uniform, built
         * once per tile: reading the mask in the score loop spills registers. */
        int ka = k0 + lane, kb = ka + 32;
        unsigned long long keys =
            (__ballot(ka < seq_k && (!key_mask || key_mask[ka])) & 0xffffffffull) |
            ((unsigned long long)__ballot(kb < seq_k && (!key_mask || key_mask[kb])) << 32);

        /* Online softmax per 16-key subtile (fewer live registers than a
         * whole tile): s[e] = score of key k0 + st*16 + 2e + half. */
        for (int st = 0; st < FA_BC / 16; st++) {
            fa_f32x8 s;
            for (int e = 0; e < 8; e++) s[e] = 0.0f;
            for (int kk = 0; kk < 8; kk++)
                s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    *(const fa_bf16x16 *)&ks[st * 16 + col][kk * 16], qb[kk], s);

            float mx = m;
            for (int e = 0; e < 8; e++) {
                if (!((keys >> (st * 16 + 2 * e + half)) & 1)) s[e] = -INFINITY;
                mx = fmaxf(mx, s[e] * scale_log2);
            }
            mx = fmaxf(mx, fa_swap_halves(mx));
            float mr = mx == -INFINITY ? 0.0f : mx;   /* finite while all keys are masked */
            float alpha = exp2f(m - mr);
            m = mx;

            /* P^T as a B operand: column = query, k = key within the subtile.
             * The even keys live in half 0, the odd ones in half 1. */
            float rs = 0.0f;
            fa_bf16x16 pb;
            for (int e = 0; e < 8; e++) {
                float p = exp2f(s[e] * scale_log2 - mr);
                rs += p;
                short mine = (short)f2bf(p), other = (short)f2bf(fa_swap_halves(p));
                pb[2 * e] = half ? other : mine;
                pb[2 * e + 1] = half ? mine : other;
            }
            l = l * alpha + rs + fa_swap_halves(rs);

            for (int dt = 0; dt < 8; dt++) {
                for (int e = 0; e < 8; e++) o[dt][e] *= alpha;
                o[dt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
                    *(const fa_bf16x16 *)&vt[dt * 16 + col][st * 16], pb, o[dt]);
            }
        }
    }

    if (qrow < seq_q) {
        float il = l > 0.0f ? 1.0f / l : 0.0f;
        uint16_t *orow = out + (size_t)qrow * ldo + hoff;
        for (int dt = 0; dt < 8; dt++)
            for (int e = 0; e < 8; e++) orow[dt * 16 + 2 * e + half] = f2bf(o[dt][e] * il);
    }
#endif
}
#else
/* ========================================================================
 * Flash Attention (CUDA, tensor cores)
 *
 * FlashAttention-2 forward for head_dim 128 with mma.sync m16n8k16 bf16
 * instructions. A block handles 128 queries of one head (8 warps, 16 rows
 * each) and streams the keys in tiles of 64 through shared memory, double
 * buffered with cp.async; operand fragments come from ldmatrix (.trans for
 * V, kept row-major). Scores stay in registers with an online softmax and
 * become the A operand of the P @ V product directly (the m16n8
 * accumulator layout of two adjacent tiles is the m16k16 operand layout).
 * ======================================================================== */


/* c[16x8] += a[16x16] * b[16x8]; operands in the PTX fragment layouts. */
__device__ __forceinline__ void mma_16816(float *c, const uint32_t *a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/* Four 8x8 b16 matrices from shared memory; lanes 8m..8m+7 address the
 * rows of matrix m. */
__device__ __forceinline__ void ldmatrix_x4(uint32_t *r, const uint16_t *p) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t *r, const uint16_t *p) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}

/* Asynchronous 16-byte global -> shared copy, zero filled when !valid. */
__device__ __forceinline__ void cp_async16(uint16_t *dst, const uint16_t *src, int valid) {
    uint32_t d = (uint32_t)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(d), "l"(src), "r"(valid ? 16 : 0));
}

/* Start copying the 64-key tile at k0 of K and V into one stage. */
__device__ __forceinline__ void fa_issue_tile(uint16_t *ks, uint16_t *vs, const uint16_t *k,
                                              const uint16_t *v, int ldi, size_t hoff,
                                              int k0, int seq_k) {
    for (int i = threadIdx.x; i < FA_BC * FA_D / 8; i += blockDim.x) {
        int r = i / (FA_D / 8), c8 = (i % (FA_D / 8)) * 8;
        int valid = k0 + r < seq_k;
        size_t off = (size_t)(valid ? k0 + r : 0) * ldi + hoff + c8;
        cp_async16(ks + r * FA_STRIDE + c8, k + off, valid);
        cp_async16(vs + r * FA_STRIDE + c8, v + off, valid);
    }
    asm volatile("cp.async.commit_group;\n");
}

__global__ void __launch_bounds__(FA_THREADS)
k_flash_attention(uint16_t *out, int ldo, const uint16_t *q, const uint16_t *k,
                  const uint16_t *v, int ldi, int seq_q, int seq_k, float scale_log2,
                  const int *key_mask) {
    extern __shared__ __align__(16) uint16_t fa_smem[];
    uint16_t *kst[2] = {fa_smem, fa_smem + 2 * FA_BC * FA_STRIDE};
    uint16_t *vst[2] = {fa_smem + FA_BC * FA_STRIDE, fa_smem + 3 * FA_BC * FA_STRIDE};
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int g = lane >> 2, t = lane & 3;     /* fragment row group / column pair */
    size_t hoff = (size_t)blockIdx.y * FA_D;
    int r0 = blockIdx.x * FA_BR + warp * 16 + g, r1 = r0 + 8;

    /* Q as A fragments: k-step kk covers dims [16kk, 16kk + 16). */
    uint32_t qa[8][4];
    for (int kk = 0; kk < 8; kk++) {
        int c = kk * 16 + 2 * t;
        const uint16_t *q0 = q + (size_t)r0 * ldi + hoff + c;
        const uint16_t *q1 = q + (size_t)r1 * ldi + hoff + c;
        qa[kk][0] = r0 < seq_q ? *(const uint32_t *)q0 : 0;
        qa[kk][1] = r1 < seq_q ? *(const uint32_t *)q1 : 0;
        qa[kk][2] = r0 < seq_q ? *(const uint32_t *)(q0 + 8) : 0;
        qa[kk][3] = r1 < seq_q ? *(const uint32_t *)(q1 + 8) : 0;
    }

    float o[16][4];
    for (int n = 0; n < 16; n++) o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    int ntiles = (seq_k + FA_BC - 1) / FA_BC;

    fa_issue_tile(kst[0], vst[0], k, v, ldi, hoff, 0, seq_k);
    for (int ti = 0; ti < ntiles; ti++) {
        int k0 = ti * FA_BC, cur = ti & 1;
        if (ti + 1 < ntiles) {
            fa_issue_tile(kst[cur ^ 1], vst[cur ^ 1], k, v, ldi, hoff, k0 + FA_BC, seq_k);
            asm volatile("cp.async.wait_group 1;\n");
        } else {
            asm volatile("cp.async.wait_group 0;\n");
        }
        __syncthreads();
        const uint16_t *ks = kst[cur], *vs = vst[cur];

        /* S = Q K^T: 16 rows x 64 keys per warp, in 8 tiles of 8 keys. */
        float s[8][4];
        for (int j = 0; j < 8; j++) {
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
            const uint16_t *kr = ks + (j * 8 + (lane & 7)) * FA_STRIDE + (lane >> 3) * 8;
            for (int kk = 0; kk < 8; kk += 2) {
                uint32_t b[4];
                ldmatrix_x4(b, kr + kk * 16);
                mma_16816(s[j], qa[kk], b[0], b[1]);
                mma_16816(s[j], qa[kk + 1], b[2], b[3]);
            }
        }
        if (key_mask || k0 + FA_BC > seq_k) {
            for (int j = 0; j < 8; j++)
                for (int e = 0; e < 4; e++) {
                    int key = k0 + j * 8 + 2 * t + (e & 1);
                    if (key >= seq_k || (key_mask && !key_mask[key])) s[j][e] = -INFINITY;
                }
        }

        /* Online softmax (base 2) for rows r0 (elements 0, 1) and r1 (2, 3). */
        float mx0 = m0, mx1 = m1;
        for (int j = 0; j < 8; j++) {
            mx0 = fmaxf(mx0, fmaxf(s[j][0], s[j][1]) * scale_log2);
            mx1 = fmaxf(mx1, fmaxf(s[j][2], s[j][3]) * scale_log2);
        }
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, off));
        }
        /* Finite references while all keys of a row are masked. */
        float e0 = mx0 == -INFINITY ? 0.0f : mx0, e1 = mx1 == -INFINITY ? 0.0f : mx1;
        float a0 = exp2f(m0 - e0), a1 = exp2f(m1 - e1);
        m0 = mx0;
        m1 = mx1;
        float rs0 = 0.0f, rs1 = 0.0f;
        uint32_t pa[4][4];   /* P as A fragments, k-step kk = keys [16kk, 16kk + 16) */
        for (int j = 0; j < 8; j++) {
            float p00 = exp2f(s[j][0] * scale_log2 - e0), p01 = exp2f(s[j][1] * scale_log2 - e0);
            float p10 = exp2f(s[j][2] * scale_log2 - e1), p11 = exp2f(s[j][3] * scale_log2 - e1);
            rs0 += p00 + p01;
            rs1 += p10 + p11;
            pa[j / 2][(j % 2) * 2] = pack_bf16x2(p00, p01);
            pa[j / 2][(j % 2) * 2 + 1] = pack_bf16x2(p10, p11);
        }
        for (int off = 1; off < 4; off <<= 1) {
            rs0 += __shfl_xor_sync(0xffffffff, rs0, off);
            rs1 += __shfl_xor_sync(0xffffffff, rs1, off);
        }
        l0 = l0 * a0 + rs0;
        l1 = l1 * a1 + rs1;

        /* O = O * alpha + P V, two 8-dim tiles per ldmatrix. */
        int mat = lane >> 3;
        const uint16_t *vr = vs + ((mat & 1) * 8 + (lane & 7)) * FA_STRIDE + (mat >> 1) * 8;
        for (int dt = 0; dt < 16; dt += 2) {
            for (int e = 0; e < 2; e++) {
                o[dt + e][0] *= a0;
                o[dt + e][1] *= a0;
                o[dt + e][2] *= a1;
                o[dt + e][3] *= a1;
            }
            for (int kk = 0; kk < 4; kk++) {
                uint32_t b[4];
                ldmatrix_x4_trans(b, vr + kk * 16 * FA_STRIDE + dt * 8);
                mma_16816(o[dt], pa[kk], b[0], b[1]);
                mma_16816(o[dt + 1], pa[kk], b[2], b[3]);
            }
        }
        __syncthreads();   /* this stage is refilled by the next iteration */
    }

    float il0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, il1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    for (int n = 0; n < 16; n++) {
        int d = n * 8 + 2 * t;
        if (r0 < seq_q)
            *(uint32_t *)(out + (size_t)r0 * ldo + hoff + d) = pack_bf16x2(o[n][0] * il0, o[n][1] * il0);
        if (r1 < seq_q)
            *(uint32_t *)(out + (size_t)r1 * ldo + hoff + d) = pack_bf16x2(o[n][2] * il1, o[n][3] * il1);
    }
}
#endif /* IRIS_HIP */

static int ensure_attn_scratch(size_t bytes) {
    if (bytes <= g_attn_scratch_size) return 1;
    iris_dev_free(g_attn_scratch);
    g_attn_scratch = iris_dev_alloc(bytes);
    g_attn_scratch_size = g_attn_scratch ? bytes : 0;
    return g_attn_scratch != NULL;
}

/* Attention as two batched GEMMs around a softmax, in chunks of queries so
 * that the score buffer stays within ATTN_SCRATCH.
 *
 * In column-major terms, for one head with Q, K, V row-major and stride ldi:
 *   S^T [seq_k, nq] = K^T(T) * Q^T          (alpha = scale)
 *   O^T [hd, nq]    = V^T * P^T              (P row-major [nq, seq_k]) */
extern "C" void iris_dev_attention(uint16_t *out, int ldo,
                                   const uint16_t *q, const uint16_t *k, const uint16_t *v,
                                   int ldi, int seq_q, int seq_k, int heads, int kv_heads,
                                   int head_dim, float scale, int causal,
                                   const int *key_mask) {
    /* 16-byte aligned rows are needed by the vectorized tile loads. */
    if (g_flash && head_dim == FA_D && heads == kv_heads && !causal &&
        ldi % 8 == 0 && ldo % 2 == 0 &&
        ((uintptr_t)q | (uintptr_t)k | (uintptr_t)v | (uintptr_t)out) % 16 == 0) {
        dim3 grid((seq_q + FA_BR - 1) / FA_BR, heads);
#ifdef IRIS_HIP
        size_t smem = 0;
#else
        size_t smem = FA_SMEM;
#endif
        k_flash_attention<<<grid, FA_THREADS, smem, g_stream>>>(out, ldo, q, k, v, ldi, seq_q, seq_k,
                                                         scale * 1.4426950408889634f, key_mask);
        return;
    }
    size_t per_query = (size_t)heads * seq_k * (sizeof(float) + sizeof(uint16_t));
    int nq_max = (int)(ATTN_SCRATCH / per_query);
    if (nq_max < 1) nq_max = 1;
    if (nq_max > seq_q) nq_max = seq_q;
    if (!ensure_attn_scratch(per_query * nq_max)) {
        fprintf(stderr, "iris_dev: attention scratch allocation failed\n");
        return;
    }
    float *S = (float *)g_attn_scratch;
    uint16_t *P = (uint16_t *)(S + (size_t)heads * nq_max * seq_k);
    int hpk = heads / kv_heads;   /* query heads per kv head */
    const float one = 1.0f, zero = 0.0f;

    for (int q0 = 0; q0 < seq_q; q0 += nq_max) {
        int nq = seq_q - q0 < nq_max ? seq_q - q0 : nq_max;
        long long s_stride = (long long)nq * seq_k;
        /* With one query head per kv head all heads go in one batch,
         * otherwise one batch per kv head with the K/V matrix shared. */
        int groups = hpk == 1 ? 1 : kv_heads;
        int batch = hpk == 1 ? heads : hpk;
        long long kv_stride = hpk == 1 ? head_dim : 0;
        for (int g = 0; g < groups; g++) {
            int h0 = g * hpk;
            blasGemmStridedBatchedEx(g_blas, BLAS_OP_T, BLAS_OP_N, seq_k, nq, head_dim, &scale,
                k + (size_t)g * head_dim, BLAS_R_16BF, ldi, kv_stride,
                q + (size_t)q0 * ldi + (size_t)h0 * head_dim, BLAS_R_16BF, ldi, head_dim,
                &zero, S + (size_t)h0 * s_stride, BLAS_R_32F, seq_k, s_stride,
                batch, BLAS_COMPUTE_32F, BLAS_GEMM_DEFAULT);
        }
        k_softmax<<<heads * nq, ROW_THREADS, 0, g_stream>>>(P, S, nq, seq_k, q0,
                                                            causal, key_mask);
        for (int g = 0; g < groups; g++) {
            int h0 = g * hpk;
            blasGemmStridedBatchedEx(g_blas, BLAS_OP_N, BLAS_OP_N, head_dim, nq, seq_k, &one,
                v + (size_t)g * head_dim, BLAS_R_16BF, ldi, kv_stride,
                P + (size_t)h0 * s_stride, BLAS_R_16BF, seq_k, s_stride,
                &zero, out + (size_t)q0 * ldo + (size_t)h0 * head_dim, BLAS_R_16BF, ldo,
                head_dim, batch, BLAS_COMPUTE_32F, BLAS_GEMM_DEFAULT);
        }
    }
}

/* ========================================================================
 * Convolution Layers (channels-last images)
 * ======================================================================== */

__global__ void k_add_bias(float *y, const float *bias, int cols, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += bias[i % cols];
}

extern "C" void iris_dev_add_bias(float *y, const float *bias, int rows, int cols) {
    size_t n = (size_t)rows * cols;
    if (n) k_add_bias<<<ew_blocks(n), EW_THREADS, 0, g_stream>>>(y, bias, cols, n);
}

/* Columns for output pixels [p0, p0 + np): row p holds the k x k
 * neighborhood, (ky, kx, c) order, zero outside the image. */
/* Each thread converts V consecutive channels of one tap, V = 1 or 8
 * (cin % 8 == 0: 16-byte loads and stores). n counts the groups of V and
 * fits an int, the scratch being bounded. */
template <int V>
__global__ void k_im2col(uint16_t *cols, const float *in, int h, int w, int cin, int k,
                         int stride, int pad, int ow, int p0, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int row_groups = k * k * cin / V;
    int p = p0 + i / row_groups;
    int r = (i % row_groups) * V;
    int tap = r / cin, c = r % cin;
    int y = (p / ow) * stride + tap / k - pad;
    int x = (p % ow) * stride + tap % k - pad;
    const float *src = in + ((size_t)y * w + x) * cin + c;
    int inside = y >= 0 && y < h && x >= 0 && x < w;
    if (V == 8) {
        float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f), b = a;
        if (inside) {
            a = ((const float4 *)src)[0];
            b = ((const float4 *)src)[1];
        }
        *(uint4 *)(cols + (size_t)i * 8) = make_uint4(pack_bf16x2(a.x, a.y), pack_bf16x2(a.z, a.w),
                                                      pack_bf16x2(b.x, b.y), pack_bf16x2(b.z, b.w));
    } else {
        cols[i] = inside ? f2bf(src[0]) : 0;
    }
}

extern "C" void iris_dev_conv2d(float *out, const float *in, const uint16_t *w, const float *bias,
                                int h, int width, int cin, int cout, int k, int stride,
                                int pad_before, int pad_after) {
    int oh = (h + pad_before + pad_after - k) / stride + 1;
    int ow = (width + pad_before + pad_after - k) / stride + 1;
    int ohw = oh * ow;
    int row_len = k * k * cin;
    if (!g_conv_scratch) g_conv_scratch = (uint16_t *)iris_dev_alloc(CONV_SCRATCH);
    if (!g_conv_scratch) {
        fprintf(stderr, "iris_dev: convolution scratch allocation failed\n");
        return;
    }
    int chunk = (int)(CONV_SCRATCH / ((size_t)row_len * sizeof(uint16_t)));
    if (chunk > ohw) chunk = ohw;
    for (int p0 = 0; p0 < ohw; p0 += chunk) {
        int np = ohw - p0 < chunk ? ohw - p0 : chunk;
        if (cin % 8 == 0) {
            int n = np * row_len / 8;
            k_im2col<8><<<ew_blocks(n), EW_THREADS, 0, g_stream>>>(g_conv_scratch, in, h, width,
                                                                   cin, k, stride, pad_before,
                                                                   ow, p0, n);
        } else {
            int n = np * row_len;
            k_im2col<1><<<ew_blocks(n), EW_THREADS, 0, g_stream>>>(g_conv_scratch, in, h, width,
                                                                   cin, k, stride, pad_before,
                                                                   ow, p0, n);
        }
        iris_dev_linear_f32(out + (size_t)p0 * cout, g_conv_scratch, w, np, row_len, cout);
    }
    if (bias) iris_dev_add_bias(out, bias, ohw, cout);
}

/* GroupNorm statistics, two passes: partial sums of x (pass 0) or of
 * (x - mean)^2 (pass 1) per (group, block), then per-group reduction. */
__global__ void k_gn_partial(float *part, const float *in, const float *mean, int hw,
                             int c, int cg) {
    __shared__ float sh[ROW_THREADS];
    int g = blockIdx.x, b = blockIdx.y;
    size_t n = (size_t)hw * cg;
    size_t per_block = (n + gridDim.y - 1) / gridDim.y;
    size_t start = (size_t)b * per_block, end = start + per_block < n ? start + per_block : n;
    float m = mean ? mean[g] : 0.0f;
    float s = 0;
    for (size_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        float v = in[(i / cg) * c + (size_t)g * cg + i % cg];
        s += mean ? (v - m) * (v - m) : v;
    }
    s = block_reduce(s, sh, 0);
    if (threadIdx.x == 0) part[(size_t)g * gridDim.y + b] = s;
}

/* stats[g] = mean (pass 0) or stats[groups + g] = rstd (pass 1). */
__global__ void k_gn_finalize(float *stats, const float *part, int groups, int nblocks,
                              double count, float eps, int pass) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    double s = 0;
    for (int b = 0; b < nblocks; b++) s += part[(size_t)g * nblocks + b];
    if (pass == 0) stats[g] = (float)(s / count);
    else stats[groups + g] = (float)(1.0 / sqrt(s / count + eps));
}

__global__ void k_gn_apply(float *out, const float *in, const float *stats, const float *gamma,
                           const float *beta, int c, int cg, int groups, int silu, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ch = (int)(i % c), g = ch / cg;
    float v = (in[i] - stats[g]) * stats[groups + g] * gamma[ch] + beta[ch];
    out[i] = silu ? v / (1.0f + expf(-v)) : v;
}

extern "C" void iris_dev_group_norm(float *out, const float *in, const float *gamma,
                                    const float *beta, int hw, int c, int groups, float eps,
                                    int silu) {
    size_t need = (size_t)groups * GN_BLOCKS + 2 * groups;
    if (need > g_gn_scratch_size) {
        iris_dev_free(g_gn_scratch);
        g_gn_scratch = (float *)iris_dev_alloc(need * sizeof(float));
        g_gn_scratch_size = g_gn_scratch ? need : 0;
        if (!g_gn_scratch) return;
    }
    float *part = g_gn_scratch, *stats = g_gn_scratch + (size_t)groups * GN_BLOCKS;
    int cg = c / groups;
    double count = (double)hw * cg;
    int fin_blocks = (groups + EW_THREADS - 1) / EW_THREADS;
    for (int pass = 0; pass < 2; pass++) {
        k_gn_partial<<<dim3(groups, GN_BLOCKS), ROW_THREADS, 0, g_stream>>>(
            part, in, pass ? stats : NULL, hw, c, cg);
        k_gn_finalize<<<fin_blocks, EW_THREADS, 0, g_stream>>>(stats, part, groups,
                                                              GN_BLOCKS, count, eps, pass);
    }
    size_t n = (size_t)hw * c;
    k_gn_apply<<<ew_blocks(n), EW_THREADS, 0, g_stream>>>(out, in, stats, gamma, beta, c, cg,
                                                         groups, silu, n);
}

__global__ void k_upsample2x(float *out, const float *in, int h, int w, int c, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int ch = (int)(i % c);
    size_t p = i / c;
    int x = (int)(p % (2 * w)), y = (int)(p / (2 * w));
    out[i] = in[((size_t)(y / 2) * w + x / 2) * c + ch];
}

extern "C" void iris_dev_upsample2x(float *out, const float *in, int h, int width, int c) {
    size_t n = (size_t)4 * h * width * c;
    if (n) k_upsample2x<<<ew_blocks(n), EW_THREADS, 0, g_stream>>>(out, in, h, width, c, n);
}
