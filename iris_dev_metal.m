/*
 * Iris device backend for Apple GPUs (Metal).
 *
 * Implements iris_dev.h on Metal. Apple GPUs use unified memory, so device
 * "pointers" are the CPU addresses of shared-storage MTLBuffers: a sorted
 * allocation table maps any pointer inside a buffer back to (buffer, offset)
 * when an operation is encoded. Work is encoded in order into one command
 * buffer, which iris_dev_sync() / iris_dev_download() commit and wait for
 * (uploads wait too, since they write the buffers from the CPU). GEMMs and
 * attention use MPSGraph (bf16 inputs, f32 accumulation); the other
 * operations are compute kernels compiled at init.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include "iris_dev.h"
#include <stdio.h>
#include <string.h>

#define EW_THREADS 256
#define ROW_THREADS 256
#define CONV_SCRATCH ((size_t)256 << 20)   /* im2col buffer budget (bytes) */
#define GN_BLOCKS 64

static int g_ready = 0;
static char g_name[256] = "none";
static id<MTLDevice> g_device;
static id<MTLCommandQueue> g_queue;
static id<MTLCommandBuffer> g_cmd;              /* open command buffer, or nil */
static NSMutableArray<id<MTLBuffer>> *g_buffers; /* sorted by contents address */
static id<MTLBuffer> g_dummy;                   /* bound for NULL arguments */
static NSMutableDictionary *g_graphs;           /* MPSGraph cache by shape key */

/* Scratch for attention (contiguous q, k, v, out), convolution, GroupNorm. */
static uint16_t *g_attn_scratch = NULL;
static size_t g_attn_scratch_elems = 0;
static uint16_t *g_conv_scratch = NULL;
static float *g_gn_scratch = NULL;
static size_t g_gn_scratch_size = 0;

static id<MTLComputePipelineState> p_f32_to_bf16, p_silu, p_add, p_add_bias, p_swiglu,
    p_residual, p_fill_rows, p_layer_norm_mod, p_rms_norm, p_rms_norm_residual, p_key_mask_bias,
    p_head_norm_rope, p_copy_2d, p_im2col,
    p_gn_partial, p_gn_finalize, p_gn_apply, p_upsample2x;

/* ========================================================================
 * Kernels
 * ======================================================================== */

static NSString *kernel_source = @R"MSL(
#include <metal_stdlib>
using namespace metal;

inline float bf2f(ushort v) { return as_type<float>(uint(v) << 16); }
inline ushort f2bf(float f) {
    uint u = as_type<uint>(f);
    u += 0x7FFF + ((u >> 16) & 1);
    return ushort(u >> 16);
}

/* Sum (or max) over the threadgroup; the size must be a power of two. */
inline float group_reduce(float v, threadgroup float *sh, uint tid, uint n, bool is_max) {
    sh[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = n / 2; s > 0; s >>= 1) {
        if (tid < s) sh[tid] = is_max ? max(sh[tid], sh[tid + s]) : sh[tid] + sh[tid + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float r = sh[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

kernel void f32_to_bf16(device ushort *dst, device const float *src, constant ulong &n,
                        uint i [[thread_position_in_grid]]) {
    if (i < n) dst[i] = f2bf(src[i]);
}

kernel void silu(device float *x, constant ulong &n, uint i [[thread_position_in_grid]]) {
    if (i < n) x[i] = x[i] / (1.0f + exp(-x[i]));
}

kernel void add(device float *x, device const float *a, constant ulong &n,
                uint i [[thread_position_in_grid]]) {
    if (i < n) x[i] += a[i];
}

kernel void add_bias(device float *y, device const float *bias, constant int &cols,
                     constant ulong &n, uint i [[thread_position_in_grid]]) {
    if (i < n) y[i] += bias[i % cols];
}

kernel void swiglu(device ushort *out, device const ushort *gate, device const ushort *up,
                   constant int &ldo, constant int &ldi, constant int &cols, constant ulong &n,
                   uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    ulong r = i / cols, c = i % cols;
    float g = bf2f(gate[r * ldi + c]);
    out[r * ldo + c] = f2bf(g / (1.0f + exp(-g)) * bf2f(up[r * ldi + c]));
}

kernel void fill_rows(device float *out, device const float *row, constant int &dim,
                      constant ulong &n, uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = row[i % dim];
}

kernel void residual(device float *x, device const float *a, device const float *gate,
                     constant int &dim, constant int &has_gate, constant ulong &n,
                     uint i [[thread_position_in_grid]]) {
    if (i < n) x[i] += (has_gate ? gate[i % dim] : 1.0f) * a[i];
}

kernel void layer_norm_mod(device ushort *out, device const float *x, device const float *shift,
                           device const float *scale, constant int &dim, constant float &eps,
                           constant int &has_shift, constant int &has_scale,
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint nt [[threads_per_threadgroup]]) {
    threadgroup float sh[256];
    device const float *xr = x + (ulong)row * dim;
    device ushort *o = out + (ulong)row * dim;
    float s = 0;
    for (int i = tid; i < dim; i += nt) s += xr[i];
    float mean = group_reduce(s, sh, tid, nt, false) / dim;
    float v = 0;
    for (int i = tid; i < dim; i += nt) { float d = xr[i] - mean; v += d * d; }
    float inv = rsqrt(group_reduce(v, sh, tid, nt, false) / dim + eps);
    for (int i = tid; i < dim; i += nt) {
        float n = (xr[i] - mean) * inv;
        if (has_scale) n *= 1.0f + scale[i];
        if (has_shift) n += shift[i];
        o[i] = f2bf(n);
    }
}

kernel void rms_norm(device ushort *out, device const float *x, device const float *w,
                     device const float *scale, constant int &dim, constant float &eps,
                     constant int &has_scale,
                     uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint nt [[threads_per_threadgroup]]) {
    threadgroup float sh[256];
    device const float *xr = x + (ulong)row * dim;
    device ushort *o = out + (ulong)row * dim;
    float s = 0;
    for (int i = tid; i < dim; i += nt) s += xr[i] * xr[i];
    float inv = rsqrt(group_reduce(s, sh, tid, nt, false) / dim + eps);
    for (int i = tid; i < dim; i += nt) {
        float v = xr[i] * inv * w[i];
        if (has_scale) v *= 1.0f + scale[i];
        o[i] = f2bf(v);
    }
}

kernel void rms_norm_residual(device float *x, device const float *a, device const float *w,
                              device const float *gate, constant int &dim, constant float &eps,
                              constant int &has_gate, uint row [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint nt [[threads_per_threadgroup]]) {
    threadgroup float sh[256];
    device const float *ar = a + (ulong)row * dim;
    device float *xr = x + (ulong)row * dim;
    float s = 0;
    for (int i = tid; i < dim; i += nt) s += ar[i] * ar[i];
    float inv = rsqrt(group_reduce(s, sh, tid, nt, false) / dim + eps);
    for (int i = tid; i < dim; i += nt)
        xr[i] += (has_gate ? precise::tanh(gate[i]) : 1.0f) * ar[i] * inv * w[i];
}

/* Additive attention mask: 0 for kept keys, -inf for masked ones. */
kernel void key_mask_bias(device ushort *bias, device const int *mask, constant ulong &n,
                          uint i [[thread_position_in_grid]]) {
    if (i < n) bias[i] = mask[i] ? 0 : 0xFF80;
}

kernel void head_norm_rope(device ushort *x, device const float *w, device const float *rc,
                           device const float *rs, constant int &ld, constant int &head_dim,
                           constant float &eps, constant int &split_half,
                           uint2 tg [[threadgroup_position_in_grid]],
                           uint2 tid [[thread_position_in_threadgroup]],
                           uint2 ntg [[threads_per_threadgroup]]) {
    threadgroup float sh[128];
    uint i = tid.x, nt = ntg.x;
    int half_dim = head_dim / 2;
    device ushort *v = x + (ulong)tg.x * ld + (ulong)tg.y * head_dim;
    int i0 = split_half ? int(i) : 2 * int(i);
    int i1 = split_half ? int(i) + half_dim : 2 * int(i) + 1;
    float a = bf2f(v[i0]), b = bf2f(v[i1]);
    float inv = rsqrt(group_reduce(a * a + b * b, sh, i, nt, false) / head_dim + eps);
    a *= inv * w[i0];
    b *= inv * w[i1];
    float c = rc[(ulong)tg.x * half_dim + i], s = rs[(ulong)tg.x * half_dim + i];
    v[i0] = f2bf(a * c - b * s);
    v[i1] = f2bf(b * c + a * s);
}

/* dst[r * ldd + c] = src[r * lds + c] (bf16) */
kernel void copy_2d(device ushort *dst, device const ushort *src, constant int &ldd,
                    constant int &lds, constant int &cols, constant ulong &n,
                    uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    ulong r = i / cols, c = i % cols;
    dst[r * ldd + c] = src[r * lds + c];
}

kernel void im2col(device ushort *cols, device const float *in, constant int &h, constant int &w,
                   constant int &cin, constant int &k, constant int &stride, constant int &pad,
                   constant int &ow, constant int &p0, constant ulong &n,
                   uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    int row_len = k * k * cin;
    int p = p0 + int(i / row_len);
    int r = int(i % row_len);
    int tap = r / cin, c = r % cin;
    int y = (p / ow) * stride + tap / k - pad;
    int x = (p % ow) * stride + tap % k - pad;
    float v = (y >= 0 && y < h && x >= 0 && x < w) ? in[((ulong)y * w + x) * cin + c] : 0.0f;
    cols[i] = f2bf(v);
}

kernel void gn_partial(device float *part, device const float *in, device const float *mean,
                       constant int &hw, constant int &c, constant int &cg, constant int &pass,
                       uint2 tg [[threadgroup_position_in_grid]],
                       uint2 ng [[threadgroups_per_grid]],
                       uint2 tid2 [[thread_position_in_threadgroup]],
                       uint2 ntg [[threads_per_threadgroup]]) {
    threadgroup float sh[256];
    uint g = tg.x, b = tg.y, tid = tid2.x, nt = ntg.x;
    ulong n = (ulong)hw * cg;
    ulong per = (n + ng.y - 1) / ng.y;
    ulong start = (ulong)b * per, end = min(start + per, n);
    float m = pass ? mean[g] : 0.0f;
    float s = 0;
    for (ulong i = start + tid; i < end; i += nt) {
        float v = in[(i / cg) * c + (ulong)g * cg + i % cg];
        s += pass ? (v - m) * (v - m) : v;
    }
    s = group_reduce(s, sh, tid, nt, false);
    if (tid == 0) part[(ulong)g * ng.y + b] = s;
}

kernel void gn_finalize(device float *stats, device const float *part, constant int &groups,
                        constant int &nblocks, constant float &count, constant float &eps,
                        constant int &pass, uint g [[thread_position_in_grid]]) {
    if (int(g) >= groups) return;
    float s = 0;
    for (int b = 0; b < nblocks; b++) s += part[(ulong)g * nblocks + b];
    if (pass == 0) stats[g] = s / count;
    else stats[groups + g] = rsqrt(s / count + eps);
}

kernel void gn_apply(device float *out, device const float *in, device const float *stats,
                     device const float *gamma, device const float *beta, constant int &c,
                     constant int &cg, constant int &groups, constant int &silu_act,
                     constant ulong &n, uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    int ch = int(i % c), g = ch / cg;
    float v = (in[i] - stats[g]) * stats[groups + g] * gamma[ch] + beta[ch];
    out[i] = silu_act ? v / (1.0f + exp(-v)) : v;
}

kernel void upsample2x(device float *out, device const float *in, constant int &w,
                       constant int &c, constant ulong &n, uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    int ch = int(i % c);
    ulong p = i / c;
    int x = int(p % (2 * w)), y = int(p / (2 * w));
    out[i] = in[((ulong)(y / 2) * w + x / 2) * c + ch];
}
)MSL";

/* ========================================================================
 * Setup, Memory, Command Buffer
 * ======================================================================== */

static id<MTLComputePipelineState> pipeline(id<MTLLibrary> lib, NSString *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    id<MTLComputePipelineState> p = fn ? [g_device newComputePipelineStateWithFunction:fn error:&err] : nil;
    if (!p) fprintf(stderr, "iris_dev: Metal kernel %s failed\n", name.UTF8String);
    return p;
}

int iris_dev_init(void) {
    if (g_ready) return 1;
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device || !NSClassFromString(@"MPSGraph")) return 0;
        g_queue = [g_device newCommandQueue];
        NSError *err = nil;
        id<MTLLibrary> lib = [g_device newLibraryWithSource:kernel_source options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "iris_dev: Metal kernels failed to compile: %s\n",
                    err.localizedDescription.UTF8String);
            return 0;
        }
        p_f32_to_bf16 = pipeline(lib, @"f32_to_bf16");
        p_silu = pipeline(lib, @"silu");
        p_add = pipeline(lib, @"add");
        p_add_bias = pipeline(lib, @"add_bias");
        p_swiglu = pipeline(lib, @"swiglu");
        p_residual = pipeline(lib, @"residual");
        p_fill_rows = pipeline(lib, @"fill_rows");
        p_layer_norm_mod = pipeline(lib, @"layer_norm_mod");
        p_rms_norm = pipeline(lib, @"rms_norm");
        p_rms_norm_residual = pipeline(lib, @"rms_norm_residual");
        p_key_mask_bias = pipeline(lib, @"key_mask_bias");
        p_head_norm_rope = pipeline(lib, @"head_norm_rope");
        p_copy_2d = pipeline(lib, @"copy_2d");
        p_im2col = pipeline(lib, @"im2col");
        p_gn_partial = pipeline(lib, @"gn_partial");
        p_gn_finalize = pipeline(lib, @"gn_finalize");
        p_gn_apply = pipeline(lib, @"gn_apply");
        p_upsample2x = pipeline(lib, @"upsample2x");
        if (!p_f32_to_bf16 || !p_silu || !p_add || !p_add_bias || !p_swiglu || !p_residual ||
            !p_fill_rows || !p_layer_norm_mod || !p_rms_norm || !p_rms_norm_residual ||
            !p_key_mask_bias || !p_head_norm_rope || !p_copy_2d || !p_im2col ||
            !p_gn_partial || !p_gn_finalize || !p_gn_apply || !p_upsample2x)
            return 0;
        g_buffers = [NSMutableArray array];
        g_graphs = [NSMutableDictionary dictionary];
        g_dummy = [g_device newBufferWithLength:16 options:MTLResourceStorageModeShared];
        snprintf(g_name, sizeof(g_name), "%s", g_device.name.UTF8String);
    }
    g_ready = 1;
    return 1;
}

int iris_dev_available(void) { return g_ready; }

int iris_dev_has_memory_efficient_attention(void) {
    return g_ready && [MPSGraph instancesRespondToSelector:
        @selector(scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:scale:name:)];
}
const char *iris_dev_name(void) { return g_name; }

/* Commit the open command buffer and wait for it. */
static void flush(void) {
    if (!g_cmd) return;
    [g_cmd commit];
    [g_cmd waitUntilCompleted];
    if (g_cmd.status == MTLCommandBufferStatusError)
        fprintf(stderr, "iris_dev: Metal command buffer failed: %s\n",
                g_cmd.error.localizedDescription.UTF8String);
    g_cmd = nil;
}

static id<MTLCommandBuffer> cmd(void) {
    if (!g_cmd) g_cmd = [g_queue commandBuffer];
    return g_cmd;
}

void iris_dev_sync(void) {
    @autoreleasepool { flush(); }
}

void iris_dev_cleanup(void) {
    if (!g_ready) return;
    iris_dev_sync();
    iris_dev_free(g_attn_scratch);
    iris_dev_free(g_conv_scratch);
    iris_dev_free(g_gn_scratch);
    g_attn_scratch = NULL;
    g_conv_scratch = NULL;
    g_gn_scratch = NULL;
    g_attn_scratch_elems = g_gn_scratch_size = 0;
    g_graphs = nil;
    g_buffers = nil;
    g_ready = 0;
}

/* Index of the first buffer starting after address a. */
static NSUInteger upper_bound(uintptr_t a) {
    NSUInteger lo = 0, hi = g_buffers.count;
    while (lo < hi) {
        NSUInteger mid = (lo + hi) / 2;
        if ((uintptr_t)g_buffers[mid].contents <= a) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

/* The buffer containing p and p's offset in it. */
static id<MTLBuffer> find_buffer(const void *p, NSUInteger *offset) {
    uintptr_t a = (uintptr_t)p;
    NSUInteger i = upper_bound(a);
    if (i == 0) return nil;
    id<MTLBuffer> b = g_buffers[i - 1];
    uintptr_t start = (uintptr_t)b.contents;
    if (a >= start + b.length) return nil;
    *offset = a - start;
    return b;
}

void *iris_dev_alloc(size_t bytes) {
    @autoreleasepool {
        id<MTLBuffer> b = [g_device newBufferWithLength:(bytes ? bytes : 16)
                                               options:MTLResourceStorageModeShared];
        if (!b) return NULL;
        [g_buffers insertObject:b atIndex:upper_bound((uintptr_t)b.contents)];
        return b.contents;
    }
}

void iris_dev_free(void *p) {
    if (!p) return;
    @autoreleasepool {
        flush();   /* pending work may still use the buffer */
        NSUInteger off;
        id<MTLBuffer> b = find_buffer(p, &off);
        if (b) [g_buffers removeObjectIdenticalTo:b];
    }
}

void iris_dev_upload(void *dst, const void *src, size_t bytes) {
    @autoreleasepool { flush(); }
    memcpy(dst, src, bytes);
}

void iris_dev_upload_2d(void *dst, size_t dst_stride, const void *src, size_t src_stride,
                        size_t row_bytes, size_t rows) {
    @autoreleasepool { flush(); }
    for (size_t r = 0; r < rows; r++)
        memcpy((char *)dst + r * dst_stride, (const char *)src + r * src_stride, row_bytes);
}

void iris_dev_download(void *dst, const void *src, size_t bytes) {
    @autoreleasepool { flush(); }
    memcpy(dst, src, bytes);
}

/* ========================================================================
 * Kernel Encoding Helpers
 * ======================================================================== */

static id<MTLComputeCommandEncoder> begin(id<MTLComputePipelineState> p) {
    id<MTLComputeCommandEncoder> e = [cmd() computeCommandEncoder];
    [e setComputePipelineState:p];
    return e;
}

static void arg_ptr(id<MTLComputeCommandEncoder> e, const void *p, int index) {
    NSUInteger off = 0;
    id<MTLBuffer> b = p ? find_buffer(p, &off) : nil;
    if (p && !b) fprintf(stderr, "iris_dev: pointer %p is not a device buffer\n", p);
    [e setBuffer:(b ? b : g_dummy) offset:off atIndex:index];
}

static void arg_int(id<MTLComputeCommandEncoder> e, int v, int index) {
    [e setBytes:&v length:sizeof(v) atIndex:index];
}

static void arg_float(id<MTLComputeCommandEncoder> e, float v, int index) {
    [e setBytes:&v length:sizeof(v) atIndex:index];
}

static void arg_ulong(id<MTLComputeCommandEncoder> e, uint64_t v, int index) {
    [e setBytes:&v length:sizeof(v) atIndex:index];
}

/* One thread per element of n. */
static void end_elementwise(id<MTLComputeCommandEncoder> e, size_t n) {
    [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(EW_THREADS, 1, 1)];
    [e endEncoding];
}

/* One threadgroup per row. */
static void end_rows(id<MTLComputeCommandEncoder> e, int rows) {
    [e dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
      threadsPerThreadgroup:MTLSizeMake(ROW_THREADS, 1, 1)];
    [e endEncoding];
}

/* ========================================================================
 * Elementwise and Normalization
 * ======================================================================== */

void iris_dev_f32_to_bf16(uint16_t *dst, const float *src, size_t n) {
    if (!n) return;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> e = begin(p_f32_to_bf16);
        arg_ptr(e, dst, 0);
        arg_ptr(e, src, 1);
        arg_ulong(e, n, 2);
        end_elementwise(e, n);
    }
}

void iris_dev_silu(float *x, size_t n) {
    if (!n) return;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> e = begin(p_silu);
        arg_ptr(e, x, 0);
        arg_ulong(e, n, 1);
        end_elementwise(e, n);
    }
}

void iris_dev_add(float *x, const float *a, size_t n) {
    if (!n) return;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> e = begin(p_add);
        arg_ptr(e, x, 0);
        arg_ptr(e, a, 1);
        arg_ulong(e, n, 2);
        end_elementwise(e, n);
    }
}

void iris_dev_add_bias(float *y, const float *bias, int rows, int cols) {
    size_t n = (size_t)rows * cols;
    if (!n) return;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> e = begin(p_add_bias);
        arg_ptr(e, y, 0);
        arg_ptr(e, bias, 1);
        arg_int(e, cols, 2);
        arg_ulong(e, n, 3);
        end_elementwise(e, n);
    }
}

void iris_dev_swiglu(uint16_t *out, int ldo, const uint16_t *gate, const uint16_t *up,
                     int ldi, int rows, int cols) {
    size_t n = (size_t)rows * cols;
    if (!n) return;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> e = begin(p_swiglu);
        arg_ptr(e, out, 0);
        arg_ptr(e, gate, 1);
        arg_ptr(e, up, 2);
        arg_int(e, ldo, 3);
        arg_int(e, ldi, 4);
        arg_int(e, cols, 5);
        arg_ulong(e, n, 6);
        end_elementwise(e, n);
    }
}

void iris_dev_fill_rows(float *out, const float *row, int rows, int dim) {
    size_t n = (size_t)rows * dim;
    if (!n) return;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> e = begin(p_fill_rows);
        arg_ptr(e, out, 0);
        arg_ptr(e, row, 1);
        arg_int(e, dim, 2);
        arg_ulong(e, n, 3);
        end_elementwise(e, n);
    }
}

void iris_dev_residual(float *x, const float *a, const float *gate, int rows, int dim) {
    size_t n = (size_t)rows * dim;
    if (!n) return;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> e = begin(p_residual);
        arg_ptr(e, x, 0);
        arg_ptr(e, a, 1);
        arg_ptr(e, gate, 2);
        arg_int(e, dim, 3);
        arg_int(e, gate != NULL, 4);
        arg_ulong(e, n, 5);
        end_elementwise(e, n);
    }
}

void iris_dev_layer_norm_mod(uint16_t *out, const float *x, const float *shift,
                             const float *scale, int rows, int dim, float eps) {
    if (rows <= 0) return;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> e = begin(p_layer_norm_mod);
        arg_ptr(e, out, 0);
        arg_ptr(e, x, 1);
        arg_ptr(e, shift, 2);
        arg_ptr(e, scale, 3);
        arg_int(e, dim, 4);
        arg_float(e, eps, 5);
        arg_int(e, shift != NULL, 6);
        arg_int(e, scale != NULL, 7);
        end_rows(e, rows);
    }
}

void iris_dev_rms_norm_mod(uint16_t *out, const float *x, const float *w, const float *scale,
                           int rows, int dim, float eps) {
    if (rows <= 0) return;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> e = begin(p_rms_norm);
        arg_ptr(e, out, 0);
        arg_ptr(e, x, 1);
        arg_ptr(e, w, 2);
        arg_ptr(e, scale, 3);
        arg_int(e, dim, 4);
        arg_float(e, eps, 5);
        arg_int(e, scale != NULL, 6);
        end_rows(e, rows);
    }
}

void iris_dev_rms_norm(uint16_t *out, const float *x, const float *w, int rows, int dim,
                       float eps) {
    iris_dev_rms_norm_mod(out, x, w, NULL, rows, dim, eps);
}

void iris_dev_rms_norm_residual(float *x, const float *a, const float *w, const float *gate,
                                int rows, int dim, float eps) {
    if (rows <= 0) return;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> e = begin(p_rms_norm_residual);
        arg_ptr(e, x, 0);
        arg_ptr(e, a, 1);
        arg_ptr(e, w, 2);
        arg_ptr(e, gate, 3);
        arg_int(e, dim, 4);
        arg_float(e, eps, 5);
        arg_int(e, gate != NULL, 6);
        end_rows(e, rows);
    }
}

void iris_dev_head_norm_rope(uint16_t *x, int ld, const float *w, const float *rope_cos,
                             const float *rope_sin, int seq, int heads, int head_dim,
                             float eps, int split_half) {
    if (seq <= 0) return;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> e = begin(p_head_norm_rope);
        arg_ptr(e, x, 0);
        arg_ptr(e, w, 1);
        arg_ptr(e, rope_cos, 2);
        arg_ptr(e, rope_sin, 3);
        arg_int(e, ld, 4);
        arg_int(e, head_dim, 5);
        arg_float(e, eps, 6);
        arg_int(e, split_half, 7);
        [e dispatchThreadgroups:MTLSizeMake(seq, heads, 1)
          threadsPerThreadgroup:MTLSizeMake(head_dim / 2, 1, 1)];
        [e endEncoding];
    }
}

static void copy_2d(uint16_t *dst, int ldd, const uint16_t *src, int lds, int rows, int cols) {
    size_t n = (size_t)rows * cols;
    id<MTLComputeCommandEncoder> e = begin(p_copy_2d);
    arg_ptr(e, dst, 0);
    arg_ptr(e, src, 1);
    arg_int(e, ldd, 2);
    arg_int(e, lds, 3);
    arg_int(e, cols, 4);
    arg_ulong(e, n, 5);
    end_elementwise(e, n);
}

/* ========================================================================
 * MPSGraph: GEMM and Attention
 * ======================================================================== */

@interface IrisDevGraph : NSObject
@property(strong) MPSGraph *graph;
@property(strong) NSArray<MPSGraphTensor *> *inputs;
@property(strong) NSArray<NSArray<NSNumber *> *> *shapes;
@property(strong) MPSGraphTensor *output;
@property(strong) NSArray<NSNumber *> *outShape;
@property MPSDataType outType;
@end
@implementation IrisDevGraph
@end

/* Graph data for a device pointer, as a tensor of the given shape. */
static MPSGraphTensorData *tensor_data(const void *p, NSArray<NSNumber *> *shape, MPSDataType type) {
    NSUInteger off = 0;
    id<MTLBuffer> b = find_buffer(p, &off);
    if (!b) return nil;
    if (off == 0) return [[MPSGraphTensorData alloc] initWithMTLBuffer:b shape:shape dataType:type];
    MPSNDArrayDescriptor *d = [MPSNDArrayDescriptor descriptorWithDataType:type shape:shape];
    MPSNDArray *a = [[MPSNDArray alloc] initWithBuffer:b offset:off descriptor:d];
    return [[MPSGraphTensorData alloc] initWithMPSNDArray:a];
}

static void run_graph(IrisDevGraph *g, NSArray *ptrs, void *out) {
    NSMutableDictionary *feeds = [NSMutableDictionary dictionary];
    for (NSUInteger i = 0; i < g.inputs.count; i++)
        feeds[g.inputs[i]] = tensor_data([ptrs[i] pointerValue], g.shapes[i], MPSDataTypeBFloat16);
    NSDictionary *results = @{g.output : tensor_data(out, g.outShape, g.outType)};
    MPSCommandBuffer *mcb = [MPSCommandBuffer commandBufferWithCommandBuffer:cmd()];
    [g.graph encodeToCommandBuffer:mcb feeds:feeds targetOperations:nil
                 resultsDictionary:results executionDescriptor:nil];
    /* MPSGraph may commit and continue: keep encoding into the live buffer. */
    g_cmd = mcb.rootCommandBuffer;
}

static IrisDevGraph *linear_graph(int rows, int in, int out, MPSDataType out_type) {
    NSString *key = [NSString stringWithFormat:@"lin %d %d %d %d", rows, in, out, (int)out_type];
    IrisDevGraph *g = g_graphs[key];
    if (g) return g;
    g = [IrisDevGraph new];
    g.graph = [MPSGraph new];
    NSArray *xs = @[@(rows), @(in)], *ws = @[@(out), @(in)];
    MPSGraphTensor *x = [g.graph placeholderWithShape:xs dataType:MPSDataTypeBFloat16 name:nil];
    MPSGraphTensor *w = [g.graph placeholderWithShape:ws dataType:MPSDataTypeBFloat16 name:nil];
    MPSGraphTensor *wt = [g.graph transposeTensor:w dimension:0 withDimension:1 name:nil];
    MPSGraphTensor *y = [g.graph matrixMultiplicationWithPrimaryTensor:x secondaryTensor:wt name:nil];
    if (out_type != MPSDataTypeBFloat16) y = [g.graph castTensor:y toType:out_type name:nil];
    g.inputs = @[x, w];
    g.shapes = @[xs, ws];
    g.output = y;
    g.outShape = @[@(rows), @(out)];
    g.outType = out_type;
    g_graphs[key] = g;
    return g;
}

static void linear(void *y, MPSDataType type, const uint16_t *x, const uint16_t *W, int rows,
                   int in, int out) {
    @autoreleasepool {
        IrisDevGraph *g = linear_graph(rows, in, out, type);
        run_graph(g, @[[NSValue valueWithPointer:x], [NSValue valueWithPointer:W]], y);
    }
}

void iris_dev_linear_f32(float *y, const uint16_t *x, const uint16_t *W, int rows, int in,
                         int out) {
    linear(y, MPSDataTypeFloat32, x, W, rows, in, out);
}

void iris_dev_linear_bf16(uint16_t *y, const uint16_t *x, const uint16_t *W, int rows, int in,
                          int out) {
    linear(y, MPSDataTypeBFloat16, x, W, rows, in, out);
}

static IrisDevGraph *sdpa_graph(int seq_q, int seq_k, int heads, int head_dim, float scale,
                                int masked) {
    NSString *key = [NSString stringWithFormat:@"sdpa %d %d %d %d %a %d", seq_q, seq_k, heads,
                                               head_dim, scale, masked];
    IrisDevGraph *g = g_graphs[key];
    if (g) return g;
    g = [IrisDevGraph new];
    g.graph = [MPSGraph new];
    NSArray *qs = @[@1, @(seq_q), @(heads), @(head_dim)], *ks = @[@1, @(seq_k), @(heads), @(head_dim)];
    MPSGraphTensor *q = [g.graph placeholderWithShape:qs dataType:MPSDataTypeBFloat16 name:nil];
    MPSGraphTensor *k = [g.graph placeholderWithShape:ks dataType:MPSDataTypeBFloat16 name:nil];
    MPSGraphTensor *v = [g.graph placeholderWithShape:ks dataType:MPSDataTypeBFloat16 name:nil];
    MPSGraphTensor *qt = [g.graph transposeTensor:q dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *kt = [g.graph transposeTensor:k dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *vt = [g.graph transposeTensor:v dimension:1 withDimension:2 name:nil];
    MPSGraphTensor *o;
    if (masked) {
        NSArray *ms = @[@1, @1, @1, @(seq_k)];
        MPSGraphTensor *m = [g.graph placeholderWithShape:ms dataType:MPSDataTypeBFloat16 name:nil];
        o = [g.graph scaledDotProductAttentionWithQueryTensor:qt keyTensor:kt valueTensor:vt
                                                   maskTensor:m scale:scale name:nil];
        g.inputs = @[q, k, v, m];
        g.shapes = @[qs, ks, ks, ms];
    } else {
        o = [g.graph scaledDotProductAttentionWithQueryTensor:qt keyTensor:kt valueTensor:vt
                                                        scale:scale name:nil];
        g.inputs = @[q, k, v];
        g.shapes = @[qs, ks, ks];
    }
    o = [g.graph transposeTensor:o dimension:1 withDimension:2 name:nil];
    g.output = o;
    g.outShape = qs;
    g.outType = MPSDataTypeBFloat16;
    g_graphs[key] = g;
    return g;
}

/* Attention through MPSGraph's fused SDPA, on contiguous copies of q, k, v
 * when they are strided views; the key mask becomes an additive bias.
 * Supports the non-causal case with one kv head per query head. */
void iris_dev_attention(uint16_t *out, int ldo, const uint16_t *q, const uint16_t *k,
                        const uint16_t *v, int ldi, int seq_q, int seq_k, int heads,
                        int kv_heads, int head_dim, float scale, int causal,
                        const int *key_mask) {
    if (causal || kv_heads != heads) {
        fprintf(stderr, "iris_dev: Metal attention supports only non-causal MHA\n");
        return;
    }
    @autoreleasepool {
        int dim = heads * head_dim;
        int strided = ldi != dim, out_strided = ldo != dim;
        size_t need = (size_t)(strided ? seq_q + 2 * seq_k : 0) * dim +
                      (size_t)(out_strided ? seq_q : 0) * dim + (key_mask ? seq_k : 0);
        if (need > g_attn_scratch_elems) {
            iris_dev_free(g_attn_scratch);
            g_attn_scratch = iris_dev_alloc(need * sizeof(uint16_t));
            g_attn_scratch_elems = g_attn_scratch ? need : 0;
            if (!g_attn_scratch) return;
        }
        uint16_t *s = g_attn_scratch;
        const uint16_t *qc = q, *kc = k, *vc = v;
        if (strided) {
            uint16_t *qs = s, *ks = qs + (size_t)seq_q * dim, *vs = ks + (size_t)seq_k * dim;
            copy_2d(qs, dim, q, ldi, seq_q, dim);
            copy_2d(ks, dim, k, ldi, seq_k, dim);
            copy_2d(vs, dim, v, ldi, seq_k, dim);
            qc = qs;
            kc = ks;
            vc = vs;
            s = vs + (size_t)seq_k * dim;
        }
        uint16_t *oc = out_strided ? s : out;
        if (out_strided) s += (size_t)seq_q * dim;
        NSMutableArray *ptrs = [NSMutableArray arrayWithObjects:[NSValue valueWithPointer:qc],
                                [NSValue valueWithPointer:kc], [NSValue valueWithPointer:vc], nil];
        if (key_mask) {
            id<MTLComputeCommandEncoder> e = begin(p_key_mask_bias);
            arg_ptr(e, s, 0);
            arg_ptr(e, key_mask, 1);
            arg_ulong(e, seq_k, 2);
            end_elementwise(e, seq_k);
            [ptrs addObject:[NSValue valueWithPointer:s]];
        }
        IrisDevGraph *g = sdpa_graph(seq_q, seq_k, heads, head_dim, scale, key_mask != NULL);
        run_graph(g, ptrs, oc);
        if (out_strided) copy_2d(out, ldo, oc, dim, seq_q, dim);
    }
}

/* ========================================================================
 * Convolution Layers (channels-last images)
 * ======================================================================== */

void iris_dev_conv2d(float *out, const float *in, const uint16_t *w, const float *bias,
                     int h, int width, int cin, int cout, int k, int stride, int pad_before,
                     int pad_after) {
    int oh = (h + pad_before + pad_after - k) / stride + 1;
    int ow = (width + pad_before + pad_after - k) / stride + 1;
    int ohw = oh * ow, row_len = k * k * cin;
    if (!g_conv_scratch) g_conv_scratch = iris_dev_alloc(CONV_SCRATCH);
    if (!g_conv_scratch) return;
    int chunk = (int)(CONV_SCRATCH / ((size_t)row_len * sizeof(uint16_t)));
    if (chunk > ohw) chunk = ohw;
    for (int p0 = 0; p0 < ohw; p0 += chunk) {
        int np = ohw - p0 < chunk ? ohw - p0 : chunk;
        size_t n = (size_t)np * row_len;
        @autoreleasepool {
            id<MTLComputeCommandEncoder> e = begin(p_im2col);
            arg_ptr(e, g_conv_scratch, 0);
            arg_ptr(e, in, 1);
            arg_int(e, h, 2);
            arg_int(e, width, 3);
            arg_int(e, cin, 4);
            arg_int(e, k, 5);
            arg_int(e, stride, 6);
            arg_int(e, pad_before, 7);
            arg_int(e, ow, 8);
            arg_int(e, p0, 9);
            arg_ulong(e, n, 10);
            end_elementwise(e, n);
        }
        linear(out + (size_t)p0 * cout, MPSDataTypeFloat32, g_conv_scratch, w, np, row_len, cout);
    }
    if (bias) iris_dev_add_bias(out, bias, ohw, cout);
}

void iris_dev_group_norm(float *out, const float *in, const float *gamma, const float *beta,
                         int hw, int c, int groups, float eps, int silu_act) {
    size_t need = (size_t)groups * GN_BLOCKS + 2 * groups;
    if (need > g_gn_scratch_size) {
        iris_dev_free(g_gn_scratch);
        g_gn_scratch = iris_dev_alloc(need * sizeof(float));
        g_gn_scratch_size = g_gn_scratch ? need : 0;
        if (!g_gn_scratch) return;
    }
    float *part = g_gn_scratch, *stats = g_gn_scratch + (size_t)groups * GN_BLOCKS;
    int cg = c / groups;
    @autoreleasepool {
        for (int pass = 0; pass < 2; pass++) {
            id<MTLComputeCommandEncoder> e = begin(p_gn_partial);
            arg_ptr(e, part, 0);
            arg_ptr(e, in, 1);
            arg_ptr(e, stats, 2);
            arg_int(e, hw, 3);
            arg_int(e, c, 4);
            arg_int(e, cg, 5);
            arg_int(e, pass, 6);
            [e dispatchThreadgroups:MTLSizeMake(groups, GN_BLOCKS, 1)
              threadsPerThreadgroup:MTLSizeMake(ROW_THREADS, 1, 1)];
            [e endEncoding];
            e = begin(p_gn_finalize);
            arg_ptr(e, stats, 0);
            arg_ptr(e, part, 1);
            arg_int(e, groups, 2);
            arg_int(e, GN_BLOCKS, 3);
            arg_float(e, (float)hw * cg, 4);
            arg_float(e, eps, 5);
            arg_int(e, pass, 6);
            end_elementwise(e, groups);
        }
        size_t n = (size_t)hw * c;
        id<MTLComputeCommandEncoder> e = begin(p_gn_apply);
        arg_ptr(e, out, 0);
        arg_ptr(e, in, 1);
        arg_ptr(e, stats, 2);
        arg_ptr(e, gamma, 3);
        arg_ptr(e, beta, 4);
        arg_int(e, c, 5);
        arg_int(e, cg, 6);
        arg_int(e, groups, 7);
        arg_int(e, silu_act, 8);
        arg_ulong(e, n, 9);
        end_elementwise(e, n);
    }
}

void iris_dev_upsample2x(float *out, const float *in, int h, int width, int c) {
    size_t n = (size_t)4 * h * width * c;
    if (!n) return;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> e = begin(p_upsample2x);
        arg_ptr(e, out, 0);
        arg_ptr(e, in, 1);
        arg_int(e, width, 2);
        arg_int(e, c, 3);
        arg_ulong(e, n, 4);
        end_elementwise(e, n);
    }
}
