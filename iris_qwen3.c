/*
 * Qwen3 Text Encoder Implementation
 *
 * Implements Qwen3-4B model for text encoding in Iris image generation.
 * - 36 transformer layers
 * - 2560 hidden dimension
 * - GQA with 32 query heads and 8 KV heads
 * - RoPE positional embeddings
 * - SwiGLU MLP
 */

#include "iris_qwen3.h"
#include "iris_safetensors.h"
#include "iris_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Use BLAS for matrix operations when enabled via Makefile */
#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/* Use Metal for GPU acceleration */
#ifdef USE_METAL
#include "iris_metal.h"
#endif

/* Device backend: the whole encoder on CUDA / ROCm. Metal uses it for Qwen-Image
 * only, the other models keep the native Metal encoder. */
#ifdef IRIS_DEV
#define QWEN3_DEV
#include "iris_dev.h"
#ifdef USE_METAL
#define QWEN3_USE_DEV(model) ((model)->extraction_mode == QWEN3_EXTRACT_QIMAGE)
#else
#define QWEN3_USE_DEV(model) 1
#endif
#endif

/* Minimum matrix size for GPU acceleration.
 * Using 10M threshold keeps text encoder on CPU (Accelerate BLAS), which is
 * faster and avoids GPU memory pressure on 16GB systems. Text encoder weights
 * are only used once per generation, so GPU caching provides no benefit.
 * Fixes issue #9: SIGKILL on 16GB Metal systems during text encoding. */
#define QWEN3_MIN_GPU_ELEMENTS (10 * 1024 * 1024)

/* Maximum number of safetensors shards (Qwen3-8B may have >2) */
#define QWEN3_MAX_SHARDS 16

/* ========================================================================
 * Data Structures
 * ======================================================================== */

typedef struct {
    float *q_proj_weight;     /* [num_heads * head_dim, hidden] = [4096, 2560] */
    float *k_proj_weight;     /* [num_kv_heads * head_dim, hidden] = [1024, 2560] */
    float *v_proj_weight;     /* [num_kv_heads * head_dim, hidden] = [1024, 2560] */
    float *o_proj_weight;     /* [hidden, num_heads * head_dim] = [2560, 4096] */
    float *q_norm_weight;     /* [head_dim] = [128] */
    float *k_norm_weight;     /* [head_dim] = [128] */
    /* BF16 weight pointers (for GPU path) */
    uint16_t *q_proj_weight_bf16;
    uint16_t *k_proj_weight_bf16;
    uint16_t *v_proj_weight_bf16;
    uint16_t *o_proj_weight_bf16;
    uint16_t *q_norm_weight_bf16;  /* [head_dim] = [128] */
    uint16_t *k_norm_weight_bf16;  /* [head_dim] = [128] */
} qwen3_attention_t;

typedef struct {
    float *gate_proj_weight;  /* [intermediate, hidden] = [9728, 2560] */
    float *up_proj_weight;    /* [intermediate, hidden] = [9728, 2560] */
    float *down_proj_weight;  /* [hidden, intermediate] = [2560, 9728] */
    /* BF16 weight pointers (for GPU path) */
    uint16_t *gate_proj_weight_bf16;
    uint16_t *up_proj_weight_bf16;
    uint16_t *down_proj_weight_bf16;
} qwen3_mlp_t;

typedef struct {
    float *input_layernorm_weight;      /* [hidden] */
    float *post_attention_layernorm_weight;  /* [hidden] */
    qwen3_attention_t attn;
    qwen3_mlp_t mlp;
    /* BF16 layer norm weights (for GPU path) - unused currently, kept for future */
    uint16_t *input_layernorm_weight_bf16;
    uint16_t *post_attention_layernorm_weight_bf16;
} qwen3_layer_t;

struct qwen3_model {
    /* Architecture (from config.json) */
    int hidden_size;
    int intermediate_size;
    int num_heads;
    int num_kv_heads;
    int head_dim;
    int vocab_size;
    float rope_theta;
    int extraction_mode;      /* QWEN3_EXTRACT_* */
    int out_layers[QWEN3_MAX_OUTPUT_LAYERS];  /* 0-based layers forming the embeddings */
    int num_out_layers;
    int max_seq_len;          /* QWEN3_MAX_SEQ_LEN, QWEN3_VL_MAX_SEQ_LEN for Qwen3-VL,
                               * QWEN3_QIMAGE_MAX_SEQ_LEN for Qwen-Image */
    char prefix[32];          /* Tensor names: "model." or "model.language_model." */

    /* Embedding layer */
    float *embed_tokens;      /* [vocab_size, hidden] */

    /* Transformer layers */
    qwen3_layer_t *layers;    /* [num_layers] */
    int num_layers;

    /* Final layer norm */
    float *norm_weight;       /* [hidden] */

    /* RoPE tables, filled for each forward pass */
    float *rope_cos;          /* [seq_len, head_dim/2] */
    float *rope_sin;          /* [seq_len, head_dim/2] */
    int mrope_h, mrope_w;     /* mrope_section[1..2], 0 without one */

    /* Images of the prompt being encoded (Qwen-Image references) */
    const qwen3_image_t *images;
    const int *image_rows;    /* first slot row of each image */
    int num_images;

    /* Working memory, grown to the longest sequence seen */
    int work_seq;             /* sequence length it holds */
    float *hidden_state;      /* [seq_len, hidden] */
    float *residual;          /* [seq_len, hidden] */
    float *q_buf;             /* [seq_len, num_heads * head_dim] */
    float *k_buf;             /* [seq_len, num_kv_heads * head_dim] */
    float *v_buf;             /* [seq_len, num_kv_heads * head_dim] */
    float *attn_scores;       /* [seq_len, seq_len], one head at a time */
    float *attn_out;          /* [seq_len, num_heads * head_dim] */
    float *mlp_gate;          /* [seq_len, intermediate] */
    float *mlp_up;            /* [seq_len, intermediate] */
    float *mlp_out;           /* [seq_len, hidden] */
    float *norm_buf;          /* [seq_len, hidden] */

    /* Outputs of out_layers */
    float *layer_outputs[QWEN3_MAX_OUTPUT_LAYERS];  /* [seq_len, hidden] each */

    /* Pre-allocated attention work buffers (avoid per-call allocation) */
    float *attn_q_head;       /* [seq_len, head_dim] */
    float *attn_v_head;       /* [seq_len, head_dim] */
    float *attn_out_head;     /* [seq_len, head_dim] */

    /* Mmap mode: keep safetensors files open, load layer weights on-demand */
    int use_mmap;
    safetensors_file_t *sf_files[QWEN3_MAX_SHARDS];
    int num_sf_files;

    /* BF16 GPU acceleration */
    int use_bf16;

    /* Device copy of the layers (CUDA / ROCm), NULL until first used */
    struct qwen3_dev *dev;
};

/* Forward declarations for mmap streaming mode */
static int load_layer_weights(qwen3_layer_t *layer, safetensors_file_t **files,
                              int num_files, const char *prefix, int layer_idx);
#ifdef USE_METAL
static int load_layer_weights_small_f32(qwen3_layer_t *layer, safetensors_file_t **files,
                                        int num_files, const char *prefix, int layer_idx);
static int load_layer_weights_bf16(qwen3_layer_t *layer, safetensors_file_t **files,
                                   int num_files, const char *prefix, int layer_idx);
#endif
static void free_layer_weights(qwen3_layer_t *layer);
#ifdef QWEN3_DEV
static int qwen3_forward_dev(qwen3_model_t *model, int seq_len, const int *attention_mask,
                             const int *out_layers, int num_out);
static void qwen3_dev_free(qwen3_model_t *model);
#endif

/* ========================================================================
 * Basic Operations
 * ======================================================================== */

static void qwen3_linear(float *y, const float *x, const float *W,
                         int seq_len, int in_dim, int out_dim) {
    /* y[seq, out] = x[seq, in] @ W[out, in]^T */
#ifdef USE_METAL
    /* Use GPU for large matrices */
    size_t matrix_elements = (size_t)seq_len * out_dim;
    if (iris_metal_available() && matrix_elements >= QWEN3_MIN_GPU_ELEMENTS) {
        iris_metal_sgemm_cached(0, 1,  /* no transpose A, transpose B */
                                seq_len, out_dim, in_dim,
                                1.0f,
                                x, in_dim,
                                W, in_dim,
                                0.0f,
                                y, out_dim);
        return;
    }
#endif

#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                seq_len, out_dim, in_dim,
                1.0f, x, in_dim, W, in_dim,
                0.0f, y, out_dim);
#else
    for (int s = 0; s < seq_len; s++) {
        for (int o = 0; o < out_dim; o++) {
            float sum = 0.0f;
            for (int i = 0; i < in_dim; i++) {
                sum += x[s * in_dim + i] * W[o * in_dim + i];
            }
            y[s * out_dim + o] = sum;
        }
    }
#endif
}

static void qwen3_rms_norm(float *out, const float *x, const float *weight,
                           int seq_len, int hidden, float eps) {
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

/* Per-head RMS norm for Q/K normalization */
static void qwen3_head_rms_norm(float *out, const float *x, const float *weight,
                                int seq_len, int num_heads, int head_dim, float eps) {
    for (int s = 0; s < seq_len; s++) {
        for (int h = 0; h < num_heads; h++) {
            const float *x_head = x + s * num_heads * head_dim + h * head_dim;
            float *out_head = out + s * num_heads * head_dim + h * head_dim;

            /* Compute RMS for this head */
            float sum_sq = 0.0f;
            for (int i = 0; i < head_dim; i++) {
                sum_sq += x_head[i] * x_head[i];
            }
            float rms = sqrtf(sum_sq / head_dim + eps);
            float rms_inv = 1.0f / rms;

            /* Normalize and scale */
            for (int i = 0; i < head_dim; i++) {
                out_head[i] = x_head[i] * rms_inv * weight[i];
            }
        }
    }
}

static void qwen3_softmax(float *x, int len) {
    float max_val = x[0];
    for (int i = 1; i < len; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < len; i++) {
        x[i] = fast_expf(x[i] - max_val);
        sum += x[i];
    }

    float inv_sum = 1.0f / sum;
    for (int i = 0; i < len; i++) {
        x[i] *= inv_sum;
    }
}

/* ========================================================================
 * RoPE (Rotary Position Embedding)
 * ======================================================================== */

/* One row of the RoPE tables at multi-axis position (t, h, w). Interleaved
 * MRoPE: every third of the first 3 * mrope_h frequencies follows the height,
 * the next one of the first 3 * mrope_w the width, the others the frame. */
static void qwen3_rope_row(qwen3_model_t *model, int s, int t, int h, int w) {
    int half_dim = model->head_dim / 2;
    for (int i = 0; i < half_dim; i++) {
        float freq = 1.0f / powf(model->rope_theta, (float)(2 * i) / model->head_dim);
        int pos = (i % 3 == 1 && i < 3 * model->mrope_h) ? h :
                  (i % 3 == 2 && i < 3 * model->mrope_w) ? w : t;
        float angle = pos * freq;
        model->rope_cos[s * half_dim + i] = cosf(angle);
        model->rope_sin[s * half_dim + i] = sinf(angle);
    }
}

/* Fill the RoPE tables for one forward pass. Tokens sit at consecutive
 * positions on all axes, with two exceptions. In Flux3 mode padding tokens
 * sit at position 1: that is what the transformers version Flux3 was trained
 * with assigns them, and the Flux3 transformer attends to the padding
 * embeddings. The h x w slots of an image sit at (p, p + row, p + col), and
 * the positions after it continue at p + max(h, w). */
static void qwen3_set_rope_positions(qwen3_model_t *model,
                                     const int *attention_mask, int seq_len) {
    int pad_at_one = model->extraction_mode == QWEN3_EXTRACT_FLUX3 && attention_mask;
    int p = 0, next = 0;

    for (int s = 0; s < seq_len;) {
        if (next < model->num_images && s == model->image_rows[next]) {
            const qwen3_image_t *img = &model->images[next++];
            for (int r = 0; r < img->h; r++)
                for (int c = 0; c < img->w; c++)
                    qwen3_rope_row(model, s++, p, p + r, p + c);
            p += img->h > img->w ? img->h : img->w;
        } else {
            int pos = (pad_at_one && !attention_mask[s]) ? 1 : p;
            qwen3_rope_row(model, s++, pos, pos, pos);
            p++;
        }
    }
}

/* Add deepstack feature k at the image rows of x [seq, hidden]. */
static void qwen3_add_deepstack(const qwen3_model_t *model, float *x, int k) {
    size_t hidden = model->hidden_size;
    for (int i = 0; i < model->num_images; i++) {
        const qwen3_image_t *img = &model->images[i];
        float *row = x + model->image_rows[i] * hidden;
        size_t n = (size_t)img->h * img->w * hidden;
        for (size_t j = 0; j < n; j++) row[j] += img->deep[k][j];
    }
}

/* Apply RoPE rotation to Q and K for all attention heads. Uses split-half
 * rotation (dims [0..63] paired with [64..127]) within GQA: each of the
 * 32 Q heads is rotated independently, while only 8 K heads are rotated
 * (shared across groups of 4 Q heads). Position encoding creates
 * relative-position-dependent attention logit decay. */
static void apply_rope(float *q, float *k, const float *cos_cache, const float *sin_cache,
                       int seq_len, int num_q_heads, int num_kv_heads, int head_dim) {
    int half_dim = head_dim / 2;

    /* Apply RoPE to Q */
    for (int s = 0; s < seq_len; s++) {
        const float *cos_row = cos_cache + s * half_dim;
        const float *sin_row = sin_cache + s * half_dim;

        for (int h = 0; h < num_q_heads; h++) {
            float *q_head = q + s * num_q_heads * head_dim + h * head_dim;

            for (int i = 0; i < half_dim; i++) {
                float x0 = q_head[i];
                float x1 = q_head[i + half_dim];
                float cos_val = cos_row[i];
                float sin_val = sin_row[i];

                q_head[i] = x0 * cos_val - x1 * sin_val;
                q_head[i + half_dim] = x0 * sin_val + x1 * cos_val;
            }
        }
    }

    /* Apply RoPE to K */
    for (int s = 0; s < seq_len; s++) {
        const float *cos_row = cos_cache + s * half_dim;
        const float *sin_row = sin_cache + s * half_dim;

        for (int h = 0; h < num_kv_heads; h++) {
            float *k_head = k + s * num_kv_heads * head_dim + h * head_dim;

            for (int i = 0; i < half_dim; i++) {
                float x0 = k_head[i];
                float x1 = k_head[i + half_dim];
                float cos_val = cos_row[i];
                float sin_val = sin_row[i];

                k_head[i] = x0 * cos_val - x1 * sin_val;
                k_head[i + half_dim] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

/* ========================================================================
 * Attention
 * ======================================================================== */

/* Full attention forward pass for one Qwen3 layer. Implements GQA (Grouped
 * Query Attention): 32 Q heads share 8 KV heads (4:1 ratio for 4B model).
 * Pipeline: Q/K/V projections -> per-head RMSNorm on Q and K -> RoPE ->
 * scaled dot-product attention with causal + padding mask -> output projection.
 * Each Q head attends to its corresponding KV group (h / heads_per_kv). */
static void qwen3_attention_forward(qwen3_model_t *model, qwen3_layer_t *layer,
                                    int seq_len, const int *attention_mask) {
    int num_heads = model->num_heads;
    int num_kv_heads = model->num_kv_heads;
    int head_dim = model->head_dim;
    int hidden = model->hidden_size;
    int kv_dim = num_kv_heads * head_dim;
    int q_dim = num_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Q, K, V projections */
    qwen3_linear(model->q_buf, model->norm_buf, layer->attn.q_proj_weight,
                 seq_len, hidden, q_dim);
    qwen3_linear(model->k_buf, model->norm_buf, layer->attn.k_proj_weight,
                 seq_len, hidden, kv_dim);
    qwen3_linear(model->v_buf, model->norm_buf, layer->attn.v_proj_weight,
                 seq_len, hidden, kv_dim);

    /* Q/K RMS normalization (per-head) */
    qwen3_head_rms_norm(model->q_buf, model->q_buf, layer->attn.q_norm_weight,
                        seq_len, num_heads, head_dim, QWEN3_RMS_NORM_EPS);
    qwen3_head_rms_norm(model->k_buf, model->k_buf, layer->attn.k_norm_weight,
                        seq_len, num_kv_heads, head_dim, QWEN3_RMS_NORM_EPS);

    /* Apply RoPE */
    apply_rope(model->q_buf, model->k_buf, model->rope_cos, model->rope_sin,
               seq_len, num_heads, num_kv_heads, head_dim);

#ifdef USE_METAL
    /* Try GPU-accelerated causal attention for all heads in parallel.
     * The GPU kernel uses both causal masking and attention mask.
     * This ensures exact parity with CPU implementation. */
    if (iris_metal_available()) {
        if (iris_metal_causal_attention(model->attn_out,
                                         model->q_buf, model->k_buf, model->v_buf,
                                         attention_mask,
                                         seq_len, num_heads, num_kv_heads,
                                         head_dim, scale)) {
            /* GPU attention succeeded - skip to output projection */
            goto output_proj;
        }
    }
#endif

    /* CPU fallback: compute attention for each head with GQA
     * Use BLAS for Q@K^T and scores@V matrix multiplications */
    {
        int heads_per_kv = num_heads / num_kv_heads;

        for (int h = 0; h < num_heads; h++) {
            int kv_h = h / heads_per_kv;  /* Which KV head to use */
            float *scores = model->attn_scores;

            /* Q accessed directly with strided lda (avoids copy)
             * Q[s,d] = q_buf[s * q_dim + h * head_dim + d] */
            const float *q_strided = model->q_buf + h * head_dim;

            /* K accessed directly with strided lda + CblasTrans (avoids transpose)
             * K[s,d] = k_buf[s * kv_dim + kv_h * head_dim + d] */
            const float *k_strided = model->k_buf + kv_h * head_dim;

            /* scores = scale * Q @ K^T using strided BLAS */
#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                        seq_len, seq_len, head_dim,
                        scale, q_strided, q_dim, k_strided, kv_dim,
                        0.0f, scores, seq_len);
#else
            /* Fallback: naive matmul */
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        dot += q_strided[i * q_dim + d] * k_strided[j * kv_dim + d];
                    }
                    scores[i * seq_len + j] = dot * scale;
                }
            }
#endif

            /* Apply causal mask and attention mask, then softmax */
            for (int i = 0; i < seq_len; i++) {
                for (int j = 0; j < seq_len; j++) {
                    if (j > i) {
                        scores[i * seq_len + j] = -1e9f;
                    }
                    if (attention_mask && attention_mask[j] == 0) {
                        scores[i * seq_len + j] = -1e9f;
                    }
                }
                qwen3_softmax(scores + i * seq_len, seq_len);
            }

            /* V can be accessed directly with strided lda (avoids copy)
             * V[s,d] = v_buf[s * kv_dim + kv_h * head_dim + d] */
            const float *v_strided = model->v_buf + kv_h * head_dim;

            /* Output can be written directly with strided ldc (avoids copy)
             * out[s,d] = attn_out[s * q_dim + h * head_dim + d] */
            float *out_strided = model->attn_out + h * head_dim;

            /* out = scores @ V using strided BLAS (avoids V copy and output copy)
             * scores: [seq_len, seq_len], V: [seq_len, head_dim] with ldb=kv_dim */
#ifdef USE_BLAS
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        seq_len, head_dim, seq_len,
                        1.0f, scores, seq_len, v_strided, kv_dim,
                        0.0f, out_strided, q_dim);
#else
            for (int i = 0; i < seq_len; i++) {
                for (int d = 0; d < head_dim; d++) {
                    float sum = 0.0f;
                    for (int j = 0; j < seq_len; j++) {
                        sum += scores[i * seq_len + j] * v_strided[j * kv_dim + d];
                    }
                    out_strided[i * q_dim + d] = sum;
                }
            }
#endif
        }
    }

#ifdef USE_METAL
output_proj:
#endif
    /* Output projection */
    qwen3_linear(model->hidden_state, model->attn_out, layer->attn.o_proj_weight,
                 seq_len, q_dim, hidden);
}

/* ========================================================================
 * MLP (SwiGLU)
 * ======================================================================== */

/* SwiGLU MLP: gate = W_gate @ x, up = W_up @ x, out = W_down @ (silu(gate) * up).
 * The gated activation lets the network learn which features to pass through:
 * silu(gate) acts as a learned soft switch on the up-projected features. */
static void qwen3_mlp_forward(qwen3_model_t *model, qwen3_layer_t *layer, int seq_len) {
    int hidden = model->hidden_size;
    int intermediate = model->intermediate_size;

    /* Gate and Up projections */
    qwen3_linear(model->mlp_gate, model->norm_buf, layer->mlp.gate_proj_weight,
                 seq_len, hidden, intermediate);
    qwen3_linear(model->mlp_up, model->norm_buf, layer->mlp.up_proj_weight,
                 seq_len, hidden, intermediate);

    /* SwiGLU: silu(gate) * up - fused for better performance */
    iris_silu_mul(model->mlp_gate, model->mlp_up, seq_len * intermediate);

    /* Down projection */
    qwen3_linear(model->mlp_out, model->mlp_gate, layer->mlp.down_proj_weight,
                 seq_len, intermediate, hidden);
}

/* ========================================================================
 * Transformer Layer
 * ======================================================================== */

/* One Qwen3 transformer layer (pre-norm architecture):
 * RMSNorm -> self-attention -> residual add -> RMSNorm -> SwiGLU MLP ->
 * residual add. Pre-norm (normalizing before each sub-layer rather than
 * after) improves training stability and is standard in modern LLMs. */
static void qwen3_layer_forward(qwen3_model_t *model, qwen3_layer_t *layer,
                                int seq_len, const int *attention_mask) {
    int hidden = model->hidden_size;

    /* Save residual */
    memcpy(model->residual, model->hidden_state, seq_len * hidden * sizeof(float));

    /* Pre-attention LayerNorm */
    qwen3_rms_norm(model->norm_buf, model->hidden_state, layer->input_layernorm_weight,
                   seq_len, hidden, QWEN3_RMS_NORM_EPS);

    /* Self-attention */
    qwen3_attention_forward(model, layer, seq_len, attention_mask);

    /* Residual connection */
    for (int i = 0; i < seq_len * hidden; i++) {
        model->hidden_state[i] += model->residual[i];
    }

    /* Save residual */
    memcpy(model->residual, model->hidden_state, seq_len * hidden * sizeof(float));

    /* Pre-MLP LayerNorm */
    qwen3_rms_norm(model->norm_buf, model->hidden_state, layer->post_attention_layernorm_weight,
                   seq_len, hidden, QWEN3_RMS_NORM_EPS);

    /* MLP */
    qwen3_mlp_forward(model, layer, seq_len);

    /* Residual connection */
    for (int i = 0; i < seq_len * hidden; i++) {
        model->hidden_state[i] = model->residual[i] + model->mlp_out[i];
    }
}

#ifdef USE_METAL
/* ========================================================================
 * BF16 GPU-Accelerated Layer Forward
 * Uses GPU for linear layers, keeps attention/norm on CPU for simplicity.
 * ======================================================================== */

/* Helper to convert f32 array to bf16 GPU tensor */
static iris_gpu_tensor_t f32_to_bf16_tensor(const float *data, int n) {
    iris_gpu_tensor_t f32_tensor = iris_gpu_tensor_create(data, n);
    if (!f32_tensor) return NULL;
    iris_gpu_tensor_t bf16_tensor = iris_gpu_tensor_f32_to_bf16(f32_tensor);
    iris_gpu_tensor_free(f32_tensor);
    return bf16_tensor;
}

/* Helper to read bf16 GPU tensor back to f32 array */
static void bf16_tensor_to_f32(iris_gpu_tensor_t bf16_tensor, float *out) {
    iris_gpu_tensor_t f32_tensor = iris_gpu_tensor_bf16_to_f32(bf16_tensor);
    if (f32_tensor) {
        iris_gpu_tensor_read(f32_tensor, out);
        iris_gpu_tensor_free(f32_tensor);
    }
}

/* Convert bf16 value to f32 */
static inline float bf16_to_f32_val(uint16_t bf16) {
    uint32_t f32_bits = ((uint32_t)bf16) << 16;
    float result;
    memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

/* Helper to create bf16 GPU tensor from bf16 CPU data (for small tensors like norm weights) */
static iris_gpu_tensor_t bf16_ptr_to_bf16_tensor(const uint16_t *bf16_data, int n) {
    /* Convert bf16->f32 on CPU, then f32->bf16 on GPU */
    float *f32_tmp = malloc(n * sizeof(float));
    if (!f32_tmp) return NULL;
    for (int i = 0; i < n; i++) {
        f32_tmp[i] = bf16_to_f32_val(bf16_data[i]);
    }
    iris_gpu_tensor_t result = f32_to_bf16_tensor(f32_tmp, n);
    free(f32_tmp);
    return result;
}

/* GPU-accelerated MLP using bf16 weights */
static void qwen3_mlp_forward_bf16(qwen3_model_t *model, qwen3_layer_t *layer, int seq_len) {
    int hidden = model->hidden_size;
    int intermediate = model->intermediate_size;
    int n = seq_len * intermediate;

    /* Convert input to bf16 tensor on GPU */
    iris_gpu_tensor_t x = f32_to_bf16_tensor(model->norm_buf, seq_len * hidden);
    if (!x) {
        qwen3_mlp_forward(model, layer, seq_len);
        return;
    }

    /* Gate and Up projections on GPU */
    iris_gpu_tensor_t gate = iris_gpu_linear_bf16_native(x, layer->mlp.gate_proj_weight_bf16,
                                                          seq_len, hidden, intermediate);
    iris_gpu_tensor_t up = iris_gpu_linear_bf16_native(x, layer->mlp.up_proj_weight_bf16,
                                                        seq_len, hidden, intermediate);
    iris_gpu_tensor_free(x);

    if (!gate || !up) {
        if (gate) iris_gpu_tensor_free(gate);
        if (up) iris_gpu_tensor_free(up);
        qwen3_mlp_forward(model, layer, seq_len);
        return;
    }

    /* SwiGLU: silu(gate) * up on GPU */
    iris_gpu_silu_mul_bf16(gate, up, n);
    iris_gpu_tensor_free(up);

    /* Down projection on GPU */
    iris_gpu_tensor_t out = iris_gpu_linear_bf16_native(gate, layer->mlp.down_proj_weight_bf16,
                                                         seq_len, intermediate, hidden);
    iris_gpu_tensor_free(gate);

    if (!out) {
        qwen3_mlp_forward(model, layer, seq_len);
        return;
    }

    /* Read result back to CPU */
    bf16_tensor_to_f32(out, model->mlp_out);
    iris_gpu_tensor_free(out);
}

/* GPU-accelerated attention using bf16 weights for projections */
static void qwen3_attention_forward_bf16(qwen3_model_t *model, qwen3_layer_t *layer,
                                          int seq_len, const int *attention_mask) {
    int num_heads = model->num_heads;
    int num_kv_heads = model->num_kv_heads;
    int head_dim = model->head_dim;
    int hidden = model->hidden_size;
    int kv_dim = num_kv_heads * head_dim;
    int q_dim = num_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Convert input to bf16 tensor */
    iris_gpu_tensor_t x = f32_to_bf16_tensor(model->norm_buf, seq_len * hidden);
    if (!x) {
        qwen3_attention_forward(model, layer, seq_len, attention_mask);
        return;
    }

    /* Q, K, V projections on GPU */
    iris_gpu_tensor_t q = iris_gpu_linear_bf16_native(x, layer->attn.q_proj_weight_bf16,
                                                       seq_len, hidden, q_dim);
    iris_gpu_tensor_t k = iris_gpu_linear_bf16_native(x, layer->attn.k_proj_weight_bf16,
                                                       seq_len, hidden, kv_dim);
    iris_gpu_tensor_t v = iris_gpu_linear_bf16_native(x, layer->attn.v_proj_weight_bf16,
                                                       seq_len, hidden, kv_dim);
    iris_gpu_tensor_free(x);

    if (!q || !k || !v) {
        if (q) iris_gpu_tensor_free(q);
        if (k) iris_gpu_tensor_free(k);
        if (v) iris_gpu_tensor_free(v);
        qwen3_attention_forward(model, layer, seq_len, attention_mask);
        return;
    }

    /* Try full bf16 pipeline: Q/K norm, RoPE, and attention all on GPU */
    iris_gpu_tensor_t attn_out = iris_gpu_tensor_alloc_f16(seq_len * q_dim);
    if (attn_out && layer->attn.q_norm_weight_bf16 && layer->attn.k_norm_weight_bf16) {
        /* Get bf16 weight tensors for Q/K norm */
        iris_gpu_tensor_t q_norm_w = bf16_ptr_to_bf16_tensor(layer->attn.q_norm_weight_bf16, head_dim);
        iris_gpu_tensor_t k_norm_w = bf16_ptr_to_bf16_tensor(layer->attn.k_norm_weight_bf16, head_dim);

        if (q_norm_w && k_norm_w) {

            /* Q/K RMS normalization on GPU - separate calls for GQA (different head counts) */
            int q_norm_ok = iris_gpu_head_rms_norm_bf16(q, q_norm_w, seq_len, num_heads, head_dim, QWEN3_RMS_NORM_EPS);
            int k_norm_ok = iris_gpu_head_rms_norm_bf16(k, k_norm_w, seq_len, num_kv_heads, head_dim, QWEN3_RMS_NORM_EPS);

            iris_gpu_tensor_free(q_norm_w);
            iris_gpu_tensor_free(k_norm_w);

            if (q_norm_ok && k_norm_ok) {
                /* Apply RoPE - GPU bf16 */
                iris_gpu_rope_text_bf16(q, k, model->rope_cos, model->rope_sin,
                                         seq_len, num_heads, num_kv_heads, head_dim);

                /* GPU causal attention (bf16) */
                if (iris_gpu_causal_attention_bf16(attn_out, q, k, v, attention_mask,
                                                    seq_len, num_heads, num_kv_heads,
                                                    head_dim, scale)) {
                    iris_gpu_tensor_free(q);
                    iris_gpu_tensor_free(k);
                    iris_gpu_tensor_free(v);

                    /* Output projection on GPU - input already bf16 */
                    iris_gpu_tensor_t out = iris_gpu_linear_bf16_native(attn_out, layer->attn.o_proj_weight_bf16,
                                                                         seq_len, q_dim, hidden);
                    iris_gpu_tensor_free(attn_out);

                    if (out) {
                        bf16_tensor_to_f32(out, model->hidden_state);
                        iris_gpu_tensor_free(out);
                        return;  /* Success - full bf16 pipeline */
                    }
                }
            }
        } else {
            /* q_norm_w or k_norm_w allocation failed */
            if (q_norm_w) iris_gpu_tensor_free(q_norm_w);
            if (k_norm_w) iris_gpu_tensor_free(k_norm_w);
        }
        /* GPU bf16 path failed - free everything and use full CPU fallback */
        iris_gpu_tensor_free(q);
        iris_gpu_tensor_free(k);
        iris_gpu_tensor_free(v);
        iris_gpu_tensor_free(attn_out);
        qwen3_attention_forward(model, layer, seq_len, attention_mask);
        return;
    }

    /* Fallback: No attn_out or no bf16 norm weights - use CPU path */
    if (attn_out) iris_gpu_tensor_free(attn_out);
    bf16_tensor_to_f32(q, model->q_buf);
    bf16_tensor_to_f32(k, model->k_buf);
    bf16_tensor_to_f32(v, model->v_buf);
    iris_gpu_tensor_free(q);
    iris_gpu_tensor_free(k);
    iris_gpu_tensor_free(v);

    /* Q/K RMS normalization (per-head) - CPU */
    qwen3_head_rms_norm(model->q_buf, model->q_buf, layer->attn.q_norm_weight,
                        seq_len, num_heads, head_dim, QWEN3_RMS_NORM_EPS);
    qwen3_head_rms_norm(model->k_buf, model->k_buf, layer->attn.k_norm_weight,
                        seq_len, num_kv_heads, head_dim, QWEN3_RMS_NORM_EPS);

    /* Apply RoPE - CPU */
    apply_rope(model->q_buf, model->k_buf, model->rope_cos, model->rope_sin,
               seq_len, num_heads, num_kv_heads, head_dim);

    /* GPU causal attention (f32) */
    if (!iris_metal_causal_attention(model->attn_out,
                                      model->q_buf, model->k_buf, model->v_buf,
                                      attention_mask,
                                      seq_len, num_heads, num_kv_heads,
                                      head_dim, scale)) {
        qwen3_attention_forward(model, layer, seq_len, attention_mask);
        return;
    }

    /* Output projection on GPU */
    iris_gpu_tensor_t attn = f32_to_bf16_tensor(model->attn_out, seq_len * q_dim);
    if (!attn) {
        qwen3_linear(model->hidden_state, model->attn_out, layer->attn.o_proj_weight,
                     seq_len, q_dim, hidden);
        return;
    }

    iris_gpu_tensor_t out = iris_gpu_linear_bf16_native(attn, layer->attn.o_proj_weight_bf16,
                                                         seq_len, q_dim, hidden);
    iris_gpu_tensor_free(attn);

    if (!out) {
        qwen3_linear(model->hidden_state, model->attn_out, layer->attn.o_proj_weight,
                     seq_len, q_dim, hidden);
        return;
    }

    bf16_tensor_to_f32(out, model->hidden_state);
    iris_gpu_tensor_free(out);
}

/* GPU-accelerated layer forward */
static void qwen3_layer_forward_bf16(qwen3_model_t *model, qwen3_layer_t *layer,
                                      int seq_len, const int *attention_mask) {
    int hidden = model->hidden_size;

    /* Check if we have bf16 weights */
    if (!layer->attn.q_proj_weight_bf16 || !layer->mlp.gate_proj_weight_bf16) {
        qwen3_layer_forward(model, layer, seq_len, attention_mask);
        return;
    }

    /* Save residual */
    memcpy(model->residual, model->hidden_state, seq_len * hidden * sizeof(float));

    /* Pre-attention LayerNorm - CPU (GPU conversion overhead not worth it) */
    qwen3_rms_norm(model->norm_buf, model->hidden_state, layer->input_layernorm_weight,
                   seq_len, hidden, QWEN3_RMS_NORM_EPS);

    /* Self-attention with GPU projections */
    qwen3_attention_forward_bf16(model, layer, seq_len, attention_mask);

    /* Residual connection */
    for (int i = 0; i < seq_len * hidden; i++) {
        model->hidden_state[i] += model->residual[i];
    }

    /* Save residual */
    memcpy(model->residual, model->hidden_state, seq_len * hidden * sizeof(float));

    /* Pre-MLP LayerNorm - CPU (GPU conversion overhead not worth it) */
    qwen3_rms_norm(model->norm_buf, model->hidden_state, layer->post_attention_layernorm_weight,
                   seq_len, hidden, QWEN3_RMS_NORM_EPS);

    /* MLP with GPU */
    qwen3_mlp_forward_bf16(model, layer, seq_len);

    /* Residual connection */
    for (int i = 0; i < seq_len * hidden; i++) {
        model->hidden_state[i] = model->residual[i] + model->mlp_out[i];
    }
}

/* ========================================================================
 * Fully GPU-Resident Forward Pass
 *
 * Keeps hidden state on GPU (bf16) across all layers, eliminating the
 * 72 CPU-GPU syncs (2 per layer × 36 layers) of the mixed path above.
 * Only one GPU sync at the end for all 27 needed layers.
 * ======================================================================== */

/* GPU-only attention: bf16 tensor in, bf16 tensor out.
 * Returns O projection output or NULL on failure. */
static iris_gpu_tensor_t qwen3_attention_gpu(qwen3_model_t *model,
                                              qwen3_layer_t *layer,
                                              iris_gpu_tensor_t norm_out,
                                              int seq_len,
                                              const int *attention_mask) {
    int num_heads = model->num_heads;
    int num_kv_heads = model->num_kv_heads;
    int head_dim = model->head_dim;
    int hidden = model->hidden_size;
    int q_dim = num_heads * head_dim;
    int kv_dim = num_kv_heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Q, K, V projections (bf16 → bf16) */
    iris_gpu_tensor_t q = iris_gpu_linear_bf16_native(norm_out, layer->attn.q_proj_weight_bf16,
                                                       seq_len, hidden, q_dim);
    iris_gpu_tensor_t k = iris_gpu_linear_bf16_native(norm_out, layer->attn.k_proj_weight_bf16,
                                                       seq_len, hidden, kv_dim);
    iris_gpu_tensor_t v = iris_gpu_linear_bf16_native(norm_out, layer->attn.v_proj_weight_bf16,
                                                       seq_len, hidden, kv_dim);
    if (!q || !k || !v) goto fail_qkv;

    /* Q/K RMS normalization on GPU */
    if (layer->attn.q_norm_weight_bf16 && layer->attn.k_norm_weight_bf16) {
        iris_gpu_tensor_t q_norm_w = bf16_ptr_to_bf16_tensor(layer->attn.q_norm_weight_bf16, head_dim);
        iris_gpu_tensor_t k_norm_w = bf16_ptr_to_bf16_tensor(layer->attn.k_norm_weight_bf16, head_dim);
        if (!q_norm_w || !k_norm_w) {
            if (q_norm_w) iris_gpu_tensor_free(q_norm_w);
            if (k_norm_w) iris_gpu_tensor_free(k_norm_w);
            goto fail_qkv;
        }
        iris_gpu_head_rms_norm_bf16(q, q_norm_w, seq_len, num_heads, head_dim, QWEN3_RMS_NORM_EPS);
        iris_gpu_head_rms_norm_bf16(k, k_norm_w, seq_len, num_kv_heads, head_dim, QWEN3_RMS_NORM_EPS);
        iris_gpu_tensor_free(q_norm_w);
        iris_gpu_tensor_free(k_norm_w);
    }

    /* RoPE */
    iris_gpu_rope_text_bf16(q, k, model->rope_cos, model->rope_sin,
                             seq_len, num_heads, num_kv_heads, head_dim);

    /* Causal attention */
    iris_gpu_tensor_t attn_out = iris_gpu_tensor_alloc_f16(seq_len * q_dim);
    if (!attn_out) goto fail_qkv;
    if (!iris_gpu_causal_attention_bf16(attn_out, q, k, v, attention_mask,
                                        seq_len, num_heads, num_kv_heads,
                                        head_dim, scale)) {
        iris_gpu_tensor_free(attn_out);
        goto fail_qkv;
    }
    iris_gpu_tensor_free(q);
    iris_gpu_tensor_free(k);
    iris_gpu_tensor_free(v);

    /* O projection */
    iris_gpu_tensor_t out = iris_gpu_linear_bf16_native(attn_out, layer->attn.o_proj_weight_bf16,
                                                         seq_len, q_dim, hidden);
    iris_gpu_tensor_free(attn_out);
    return out;

fail_qkv:
    if (q) iris_gpu_tensor_free(q);
    if (k) iris_gpu_tensor_free(k);
    if (v) iris_gpu_tensor_free(v);
    return NULL;
}

/* GPU-only MLP: bf16 tensor in, bf16 tensor out.
 * Returns down projection output or NULL on failure. */
static iris_gpu_tensor_t qwen3_mlp_gpu(qwen3_model_t *model, qwen3_layer_t *layer,
                                         iris_gpu_tensor_t norm_out,
                                         int seq_len) {
    int hidden = model->hidden_size;
    int intermediate = model->intermediate_size;

    iris_gpu_tensor_t gate = iris_gpu_linear_bf16_native(norm_out, layer->mlp.gate_proj_weight_bf16,
                                                          seq_len, hidden, intermediate);
    iris_gpu_tensor_t up = iris_gpu_linear_bf16_native(norm_out, layer->mlp.up_proj_weight_bf16,
                                                        seq_len, hidden, intermediate);
    if (!gate || !up) {
        if (gate) iris_gpu_tensor_free(gate);
        if (up) iris_gpu_tensor_free(up);
        return NULL;
    }

    iris_gpu_silu_mul_bf16(gate, up, seq_len * intermediate);
    iris_gpu_tensor_free(up);

    iris_gpu_tensor_t out = iris_gpu_linear_bf16_native(gate, layer->mlp.down_proj_weight_bf16,
                                                         seq_len, intermediate, hidden);
    iris_gpu_tensor_free(gate);
    return out;
}

/* Fully GPU-resident forward pass.
 * hidden_state must already be set (from embedding lookup).
 * Processes layers up to the last extracted one and fills
 * model->layer_outputs[] with the outputs of the extracted layers.
 * Returns 1 on success, 0 on failure. */
/* Fully GPU-resident Qwen3 forward pass. Key optimization: keeps all hidden
 * states on GPU (bf16) across all layers, only reading back the extracted
 * layer outputs at the end. This reduces ~72 CPU-GPU round-trips (2 per
 * layer x 36 layers) down to a single GPU sync. Layer weights are still
 * loaded from mmap on demand and freed after each layer. */
static int qwen3_forward_gpu(qwen3_model_t *model, int seq_len, const int *attention_mask) {
    int hidden = model->hidden_size;
    const int *out_layers = model->out_layers;
    int num_saved = model->num_out_layers;
    int last_layer = out_layers[num_saved - 1];

    /* Upload hidden state to GPU as bf16 */
    iris_gpu_batch_begin();
    iris_gpu_tensor_t hidden_gpu = f32_to_bf16_tensor(model->hidden_state, seq_len * hidden);
    if (!hidden_gpu) {
        iris_gpu_batch_end();
        return 0;
    }

    /* Allocate tensors for saved layer outputs */
    iris_gpu_tensor_t saved[QWEN3_MAX_OUTPUT_LAYERS] = {NULL};
    for (int i = 0; i < num_saved; i++) {
        saved[i] = iris_gpu_tensor_alloc_f16(seq_len * hidden);
        if (!saved[i]) {
            for (int j = 0; j <= i; j++) if (saved[j]) iris_gpu_tensor_free(saved[j]);
            iris_gpu_tensor_free(hidden_gpu);
            iris_gpu_batch_end();
            return 0;
        }
    }

    int ok = 1;

    for (int layer_idx = 0; layer_idx <= last_layer; layer_idx++) {
        qwen3_layer_t *layer = &model->layers[layer_idx];

        /* Load weights on demand (mmap mode) */
        if (model->use_mmap) {
            if (load_layer_weights_small_f32(layer, model->sf_files, model->num_sf_files, model->prefix, layer_idx) != 0) {
                ok = 0; break;
            }
            load_layer_weights_bf16(layer, model->sf_files, model->num_sf_files, model->prefix, layer_idx);
        }

        if (!layer->attn.q_proj_weight_bf16 || !layer->mlp.gate_proj_weight_bf16) {
            ok = 0; break;
        }

        /* Input RMS norm */
        iris_gpu_tensor_t norm_w = f32_to_bf16_tensor(layer->input_layernorm_weight, hidden);
        iris_gpu_tensor_t norm_out = iris_gpu_tensor_alloc_f16(seq_len * hidden);
        if (!norm_w || !norm_out) {
            if (norm_w) iris_gpu_tensor_free(norm_w);
            if (norm_out) iris_gpu_tensor_free(norm_out);
            ok = 0; break;
        }
        iris_gpu_rms_norm_bf16(norm_out, hidden_gpu, norm_w, seq_len, hidden, QWEN3_RMS_NORM_EPS);
        iris_gpu_tensor_free(norm_w);

        /* Attention */
        iris_gpu_tensor_t attn_out = qwen3_attention_gpu(model, layer, norm_out, seq_len, attention_mask);
        iris_gpu_tensor_free(norm_out);
        if (!attn_out) { ok = 0; break; }

        /* Residual: hidden += attn_out */
        iris_gpu_add_bf16(hidden_gpu, hidden_gpu, attn_out, seq_len * hidden);
        iris_gpu_tensor_free(attn_out);

        /* Post-attention RMS norm */
        iris_gpu_tensor_t post_norm_w = f32_to_bf16_tensor(layer->post_attention_layernorm_weight, hidden);
        norm_out = iris_gpu_tensor_alloc_f16(seq_len * hidden);
        if (!post_norm_w || !norm_out) {
            if (post_norm_w) iris_gpu_tensor_free(post_norm_w);
            if (norm_out) iris_gpu_tensor_free(norm_out);
            ok = 0; break;
        }
        iris_gpu_rms_norm_bf16(norm_out, hidden_gpu, post_norm_w, seq_len, hidden, QWEN3_RMS_NORM_EPS);
        iris_gpu_tensor_free(post_norm_w);

        /* MLP */
        iris_gpu_tensor_t mlp_out = qwen3_mlp_gpu(model, layer, norm_out, seq_len);
        iris_gpu_tensor_free(norm_out);
        if (!mlp_out) { ok = 0; break; }

        /* Residual: hidden += mlp_out */
        iris_gpu_add_bf16(hidden_gpu, hidden_gpu, mlp_out, seq_len * hidden);
        iris_gpu_tensor_free(mlp_out);

        /* Free layer weights (mmap mode) */
        if (model->use_mmap) {
            free_layer_weights(layer);
        }

        if (model->num_images && layer_idx < QWEN3_VISION_DEEPSTACK) {
            float *deep = calloc((size_t)seq_len * hidden, sizeof(float));
            iris_gpu_tensor_t t = NULL;
            if (deep) {
                qwen3_add_deepstack(model, deep, layer_idx);
                t = f32_to_bf16_tensor(deep, seq_len * hidden);
                free(deep);
            }
            if (!t) { ok = 0; break; }
            iris_gpu_add_bf16(hidden_gpu, hidden_gpu, t, seq_len * hidden);
            iris_gpu_tensor_free(t);
        }

        /* Save output at extraction layers (GPU blit copy) */
        for (int i = 0; i < num_saved; i++)
            if (layer_idx == out_layers[i])
                iris_gpu_copy_bf16(saved[i], hidden_gpu, seq_len * hidden);

        if (iris_text_progress_callback)
            iris_text_progress_callback(layer_idx, model->num_layers);
    }

    if (!ok) {
        iris_gpu_batch_end();
        for (int i = 0; i < num_saved; i++) if (saved[i]) iris_gpu_tensor_free(saved[i]);
        iris_gpu_tensor_free(hidden_gpu);
        return 0;
    }

    /* Convert saved bf16 → f32 on GPU (still within batch) */
    iris_gpu_tensor_t saved_f32[QWEN3_MAX_OUTPUT_LAYERS] = {NULL};
    for (int i = 0; i < num_saved; i++)
        saved_f32[i] = iris_gpu_tensor_bf16_to_f32(saved[i]);

    /* Execute everything in one GPU sync */
    iris_gpu_batch_end();

    /* Signal full completion for progress display */
    if (iris_text_progress_callback)
        iris_text_progress_callback(model->num_layers - 1, model->num_layers);

    /* Read f32 results to CPU */
    int read_ok = 1;
    for (int i = 0; i < num_saved; i++) {
        if (saved_f32[i]) {
            iris_gpu_tensor_read(saved_f32[i], model->layer_outputs[i]);
            iris_gpu_tensor_free(saved_f32[i]);
        } else {
            read_ok = 0;
        }
        iris_gpu_tensor_free(saved[i]);
    }
    iris_gpu_tensor_free(hidden_gpu);

    return read_ok;
}

#endif /* USE_METAL */

/* ========================================================================
 * Working Memory
 * ======================================================================== */

static void qwen3_free_work(qwen3_model_t *model) {
    float **buffers[] = {
        &model->hidden_state, &model->residual, &model->q_buf, &model->k_buf, &model->v_buf,
        &model->attn_scores, &model->attn_out, &model->mlp_gate, &model->mlp_up,
        &model->mlp_out, &model->norm_buf, &model->attn_q_head, &model->attn_v_head,
        &model->attn_out_head, &model->rope_cos, &model->rope_sin};
    for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); i++) {
        free(*buffers[i]);
        *buffers[i] = NULL;
    }
    for (int i = 0; i < QWEN3_MAX_OUTPUT_LAYERS; i++) {
        free(model->layer_outputs[i]);
        model->layer_outputs[i] = NULL;
    }
    model->work_seq = 0;
}

/* Size the work buffers for seq tokens. Returns 0 if they do not fit. */
static int qwen3_ensure_work(qwen3_model_t *model, int seq) {
    if (seq <= model->work_seq) return 1;
    qwen3_free_work(model);

    size_t bytes = (size_t)seq * sizeof(float);
    size_t hidden = model->hidden_size, inter = model->intermediate_size;
    size_t head_dim = model->head_dim, q_dim = model->num_heads * head_dim;
    size_t kv_dim = model->num_kv_heads * head_dim;
    struct { float **p; size_t size; } buffers[] = {
        {&model->hidden_state, hidden}, {&model->residual, hidden}, {&model->q_buf, q_dim},
        {&model->k_buf, kv_dim}, {&model->v_buf, kv_dim}, {&model->attn_scores, seq},
        {&model->attn_out, q_dim}, {&model->mlp_gate, inter}, {&model->mlp_up, inter},
        {&model->mlp_out, hidden}, {&model->norm_buf, hidden}, {&model->attn_q_head, head_dim},
        {&model->attn_v_head, head_dim}, {&model->attn_out_head, head_dim},
        {&model->rope_cos, head_dim / 2}, {&model->rope_sin, head_dim / 2}};
    int ok = 1;
    for (size_t i = 0; i < sizeof(buffers) / sizeof(buffers[0]); i++) {
        *buffers[i].p = malloc(bytes * buffers[i].size);
        ok = ok && *buffers[i].p;
    }
    for (int i = 0; i < QWEN3_MAX_OUTPUT_LAYERS; i++) {
        model->layer_outputs[i] = malloc(bytes * hidden);
        ok = ok && model->layer_outputs[i];
    }
    if (!ok) {
        fprintf(stderr, "Qwen3: not enough memory for %d tokens\n", seq);
        qwen3_free_work(model);
        return 0;
    }
    model->work_seq = seq;
    return 1;
}

/* ========================================================================
 * Forward Pass
 * ======================================================================== */

/* Main Qwen3 forward pass. Runs embedding lookup then processes through
 * transformer layers, saving the hidden states of the layers selected by
 * the extraction mode (see qwen3_set_extraction_mode()) and concatenating them
 * -> [seq, n*hidden]. Stops early at the last needed extraction layer to
 * skip unnecessary layers of compute. Tries fully GPU-resident path first,
 * falls back to mixed CPU/GPU or pure CPU. */
float *qwen3_forward(qwen3_model_t *model, const int *input_ids,
                     const int *attention_mask, int seq_len) {
    int hidden = model->hidden_size;
    const int *out_layers = model->out_layers;
    int num_out = model->num_out_layers;
    int last_layer = out_layers[num_out - 1];
    float *output;

    if (seq_len > model->max_seq_len || !qwen3_ensure_work(model, seq_len)) return NULL;
    qwen3_set_rope_positions(model, attention_mask, seq_len);

    /* Embedding lookup */
    for (int s = 0; s < seq_len; s++) {
        int token_id = input_ids[s];
        if (token_id >= 0 && token_id < model->vocab_size) {
            memcpy(model->hidden_state + s * hidden,
                   model->embed_tokens + token_id * hidden,
                   hidden * sizeof(float));
        } else {
            /* Unknown token - use zeros */
            memset(model->hidden_state + s * hidden, 0, hidden * sizeof(float));
        }
    }
    for (int i = 0; i < model->num_images; i++) {
        const qwen3_image_t *img = &model->images[i];
        memcpy(model->hidden_state + (size_t)model->image_rows[i] * hidden, img->embeds,
               (size_t)img->h * img->w * hidden * sizeof(float));
    }

    /* Run through transformer layers */
#ifdef QWEN3_DEV
    /* Whole forward pass on the GPU, weights from the mmap */
    if (iris_dev_available() && model->use_mmap && QWEN3_USE_DEV(model) &&
        qwen3_forward_dev(model, seq_len, attention_mask, out_layers, num_out))
        goto concatenate;
#endif
#ifdef USE_METAL
    /* Mapped bf16 weights run only on the GPU, and the native attention ends here. */
    if (model->use_bf16 && seq_len > CAUSAL_ATTENTION_MAX_SEQ) {
        fprintf(stderr, "Qwen3: %d tokens need the GPU device path\n", seq_len);
        return NULL;
    }

    /* Try fully GPU-resident path: 1 sync instead of 72, skips unneeded layers */
    if (model->use_bf16 && iris_metal_available()) {
        if (qwen3_forward_gpu(model, seq_len, attention_mask))
            goto concatenate;
        /* GPU path failed, fall through to mixed CPU/GPU path.
         * hidden_state is unmodified (GPU worked on a copy). */
    }

    /* Start batch mode to reduce GPU sync overhead between layers */
    int batch_mode = model->use_bf16 && iris_metal_available();
    if (batch_mode) {
        iris_gpu_batch_begin();
    }
#endif

    for (int layer_idx = 0; layer_idx <= last_layer; layer_idx++) {
        /* In mmap mode, load layer weights on-demand */
        if (model->use_mmap) {
#ifdef USE_METAL
            if (model->use_bf16) {
                /* Load only small f32 weights (layer norms) + bf16 projection weights */
                if (load_layer_weights_small_f32(&model->layers[layer_idx], model->sf_files, model->num_sf_files, model->prefix, layer_idx) != 0) {
                    fprintf(stderr, "Failed to load layer %d small weights\n", layer_idx);
#ifdef USE_METAL
                    if (batch_mode) iris_gpu_batch_end();
#endif
                    return NULL;
                }
                load_layer_weights_bf16(&model->layers[layer_idx], model->sf_files, model->num_sf_files, model->prefix, layer_idx);
            } else
#endif
            {
                if (load_layer_weights(&model->layers[layer_idx], model->sf_files, model->num_sf_files, model->prefix, layer_idx) != 0) {
                    fprintf(stderr, "Failed to load layer %d weights\n", layer_idx);
                    return NULL;
                }
            }
        }

#ifdef USE_METAL
        if (model->use_bf16 && iris_metal_available()) {
            qwen3_layer_forward_bf16(model, &model->layers[layer_idx], seq_len, attention_mask);
        } else
#endif
        {
            qwen3_layer_forward(model, &model->layers[layer_idx], seq_len, attention_mask);
        }

        /* In mmap mode, free layer weights after use */
        if (model->use_mmap) {
            free_layer_weights(&model->layers[layer_idx]);
        }

        if (layer_idx < QWEN3_VISION_DEEPSTACK)
            qwen3_add_deepstack(model, model->hidden_state, layer_idx);

        /* Save output at extraction layers */
        for (int i = 0; i < num_out; i++)
            if (layer_idx == out_layers[i])
                memcpy(model->layer_outputs[i], model->hidden_state, seq_len * hidden * sizeof(float));

        /* Progress callback */
        if (iris_text_progress_callback)
            iris_text_progress_callback(layer_idx, model->num_layers);
    }

#ifdef USE_METAL
    /* End batch mode */
    if (batch_mode) {
        iris_gpu_batch_end();
    }
#endif

    /* Build output embeddings */
#if defined(USE_METAL) || defined(QWEN3_DEV)
concatenate: (void)0; /* label needs a statement; can't precede a declaration in C */
#endif
    int text_dim = num_out * hidden;
    output = malloc((size_t)seq_len * text_dim * sizeof(float));
    if (!output) return NULL;

    /* Concatenate the extracted layers along the feature axis */
    for (int s = 0; s < seq_len; s++) {
        for (int i = 0; i < num_out; i++) {
            memcpy(output + (size_t)s * text_dim + i * hidden,
                   model->layer_outputs[i] + (size_t)s * hidden,
                   hidden * sizeof(float));
        }
    }

    return output;
}

/* ========================================================================
 * Model Loading
 * ======================================================================== */

/* Helper to load a tensor from safetensors files */
static float *load_tensor(safetensors_file_t **files, int num_files, const char *name) {
    for (int f = 0; f < num_files; f++) {
        const safetensor_t *t = safetensors_find(files[f], name);
        if (t) {
            return safetensors_get_f32(files[f], t);
        }
    }
    fprintf(stderr, "Error: required tensor not found: %s\n", name);
    return NULL;
}

#if defined(USE_METAL) || defined(QWEN3_DEV)
/* Helper to load bf16 tensor directly (zero-copy from mmap region) */
static uint16_t *load_tensor_bf16(safetensors_file_t **files, int num_files, const char *name) {
    for (int f = 0; f < num_files; f++) {
        const safetensor_t *t = safetensors_find(files[f], name);
        if (t && safetensor_is_bf16(t)) {
            return safetensors_get_bf16_direct(files[f], t);
        }
    }
    return NULL;  /* Not found or not bf16 - fall back to f32 */
}
#endif

#ifdef USE_METAL

/* Load only small f32 weights for bf16 path (layer norms and q/k norms) */
static int load_layer_weights_small_f32(qwen3_layer_t *layer, safetensors_file_t **files,
                                        int num_files, const char *prefix, int layer_idx) {
    char name[256];

    /* Input layernorm */
    snprintf(name, sizeof(name), "%slayers.%d.input_layernorm.weight", prefix, layer_idx);
    layer->input_layernorm_weight = load_tensor(files, num_files, name);

    /* Post-attention layernorm */
    snprintf(name, sizeof(name), "%slayers.%d.post_attention_layernorm.weight", prefix, layer_idx);
    layer->post_attention_layernorm_weight = load_tensor(files, num_files, name);

    /* Q/K norm */
    snprintf(name, sizeof(name), "%slayers.%d.self_attn.q_norm.weight", prefix, layer_idx);
    layer->attn.q_norm_weight = load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "%slayers.%d.self_attn.k_norm.weight", prefix, layer_idx);
    layer->attn.k_norm_weight = load_tensor(files, num_files, name);

    return (layer->input_layernorm_weight && layer->post_attention_layernorm_weight &&
            layer->attn.q_norm_weight && layer->attn.k_norm_weight) ? 0 : -1;
}
#endif

static int load_layer_weights(qwen3_layer_t *layer, safetensors_file_t **files,
                              int num_files, const char *prefix, int layer_idx) {
    char name[256];

    /* Input layernorm */
    snprintf(name, sizeof(name), "%slayers.%d.input_layernorm.weight", prefix, layer_idx);
    layer->input_layernorm_weight = load_tensor(files, num_files, name);

    /* Post-attention layernorm */
    snprintf(name, sizeof(name), "%slayers.%d.post_attention_layernorm.weight", prefix, layer_idx);
    layer->post_attention_layernorm_weight = load_tensor(files, num_files, name);

    /* Attention weights */
    snprintf(name, sizeof(name), "%slayers.%d.self_attn.q_proj.weight", prefix, layer_idx);
    layer->attn.q_proj_weight = load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "%slayers.%d.self_attn.k_proj.weight", prefix, layer_idx);
    layer->attn.k_proj_weight = load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "%slayers.%d.self_attn.v_proj.weight", prefix, layer_idx);
    layer->attn.v_proj_weight = load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "%slayers.%d.self_attn.o_proj.weight", prefix, layer_idx);
    layer->attn.o_proj_weight = load_tensor(files, num_files, name);

    /* Q/K norm */
    snprintf(name, sizeof(name), "%slayers.%d.self_attn.q_norm.weight", prefix, layer_idx);
    layer->attn.q_norm_weight = load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "%slayers.%d.self_attn.k_norm.weight", prefix, layer_idx);
    layer->attn.k_norm_weight = load_tensor(files, num_files, name);

    /* MLP weights */
    snprintf(name, sizeof(name), "%slayers.%d.mlp.gate_proj.weight", prefix, layer_idx);
    layer->mlp.gate_proj_weight = load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "%slayers.%d.mlp.up_proj.weight", prefix, layer_idx);
    layer->mlp.up_proj_weight = load_tensor(files, num_files, name);

    snprintf(name, sizeof(name), "%slayers.%d.mlp.down_proj.weight", prefix, layer_idx);
    layer->mlp.down_proj_weight = load_tensor(files, num_files, name);

    /* Check that all required tensors were loaded */
    if (!layer->input_layernorm_weight || !layer->post_attention_layernorm_weight ||
        !layer->attn.q_proj_weight || !layer->attn.k_proj_weight ||
        !layer->attn.v_proj_weight || !layer->attn.o_proj_weight ||
        !layer->attn.q_norm_weight || !layer->attn.k_norm_weight ||
        !layer->mlp.gate_proj_weight || !layer->mlp.up_proj_weight ||
        !layer->mlp.down_proj_weight) {
        return -1;
    }

    return 0;
}

#ifdef USE_METAL
/* Load bf16 weights for a layer (GPU acceleration path).
 * Returns 1 if all bf16 weights loaded successfully, 0 otherwise.
 * bf16 pointers are direct into mmap region - do NOT free them. */
static int load_layer_weights_bf16(qwen3_layer_t *layer, safetensors_file_t **files,
                                    int num_files, const char *prefix, int layer_idx) {
    char name[256];

    /* Attention weights */
    snprintf(name, sizeof(name), "%slayers.%d.self_attn.q_proj.weight", prefix, layer_idx);
    layer->attn.q_proj_weight_bf16 = load_tensor_bf16(files, num_files, name);

    snprintf(name, sizeof(name), "%slayers.%d.self_attn.k_proj.weight", prefix, layer_idx);
    layer->attn.k_proj_weight_bf16 = load_tensor_bf16(files, num_files, name);

    snprintf(name, sizeof(name), "%slayers.%d.self_attn.v_proj.weight", prefix, layer_idx);
    layer->attn.v_proj_weight_bf16 = load_tensor_bf16(files, num_files, name);

    snprintf(name, sizeof(name), "%slayers.%d.self_attn.o_proj.weight", prefix, layer_idx);
    layer->attn.o_proj_weight_bf16 = load_tensor_bf16(files, num_files, name);

    /* Q/K norm weights (small but needed for GPU path) */
    snprintf(name, sizeof(name), "%slayers.%d.self_attn.q_norm.weight", prefix, layer_idx);
    layer->attn.q_norm_weight_bf16 = load_tensor_bf16(files, num_files, name);

    snprintf(name, sizeof(name), "%slayers.%d.self_attn.k_norm.weight", prefix, layer_idx);
    layer->attn.k_norm_weight_bf16 = load_tensor_bf16(files, num_files, name);

    /* MLP weights */
    snprintf(name, sizeof(name), "%slayers.%d.mlp.gate_proj.weight", prefix, layer_idx);
    layer->mlp.gate_proj_weight_bf16 = load_tensor_bf16(files, num_files, name);

    snprintf(name, sizeof(name), "%slayers.%d.mlp.up_proj.weight", prefix, layer_idx);
    layer->mlp.up_proj_weight_bf16 = load_tensor_bf16(files, num_files, name);

    snprintf(name, sizeof(name), "%slayers.%d.mlp.down_proj.weight", prefix, layer_idx);
    layer->mlp.down_proj_weight_bf16 = load_tensor_bf16(files, num_files, name);

    /* Check if all large weights loaded as bf16 */
    return (layer->attn.q_proj_weight_bf16 && layer->attn.k_proj_weight_bf16 &&
            layer->attn.v_proj_weight_bf16 && layer->attn.o_proj_weight_bf16 &&
            layer->mlp.gate_proj_weight_bf16 && layer->mlp.up_proj_weight_bf16 &&
            layer->mlp.down_proj_weight_bf16);
}
#endif

/* Free a single layer's weights (used in mmap streaming mode) */
static void free_layer_weights(qwen3_layer_t *layer) {
    free(layer->input_layernorm_weight);
    free(layer->post_attention_layernorm_weight);
    free(layer->attn.q_proj_weight);
    free(layer->attn.k_proj_weight);
    free(layer->attn.v_proj_weight);
    free(layer->attn.o_proj_weight);
    free(layer->attn.q_norm_weight);
    free(layer->attn.k_norm_weight);
    free(layer->mlp.gate_proj_weight);
    free(layer->mlp.up_proj_weight);
    free(layer->mlp.down_proj_weight);
    /* Note: bf16 pointers are direct to mmap region, don't free them */
    memset(layer, 0, sizeof(*layer));
}

#ifdef QWEN3_DEV
/* ========================================================================
 * GPU Forward (CUDA / ROCm / Metal)
 * ======================================================================== */

/* Device copy of one layer, with q/k/v and gate/up fused into one GEMM each. */
typedef struct {
    uint16_t *w_qkv;              /* [q_dim + 2 * kv_dim, hidden] */
    uint16_t *w_o;                /* [hidden, q_dim] */
    uint16_t *w_gate_up;          /* [2 * intermediate, hidden] */
    uint16_t *w_down;             /* [hidden, intermediate] */
    float *in_norm, *post_norm;   /* [hidden] */
    float *q_norm, *k_norm;       /* [head_dim] */
} qwen3_dev_layer_t;

struct qwen3_dev {
    qwen3_dev_layer_t *layers;    /* [num_layers] */
    int num_loaded;               /* layers [0, num_loaded) are uploaded */
    int seq;                      /* activations sized for seq tokens */
    float *x, *out;               /* [seq, hidden] residual stream, sublayer out */
    uint16_t *norm;               /* [seq, hidden] */
    uint16_t *qkv;                /* [seq, q_dim + 2 * kv_dim] */
    uint16_t *attn;               /* [seq, q_dim] */
    uint16_t *gate_up;            /* [seq, 2 * intermediate] */
    uint16_t *act;                /* [seq, intermediate] */
    float *rope_cos, *rope_sin;   /* [seq, head_dim / 2] */
    int *mask;                    /* [seq] */
};

static void *dev_copy(const void *src, size_t bytes) {
    void *d = iris_dev_alloc(bytes);
    if (d) iris_dev_upload(d, src, bytes);
    return d;
}

/* Upload a small tensor (n elements) converted to f32. */
static float *dev_f32_tensor(qwen3_model_t *model, const char *name, size_t n) {
    float *host = load_tensor(model->sf_files, model->num_sf_files, name);
    if (!host) return NULL;
    float *d = dev_copy(host, n * sizeof(float));
    free(host);
    return d;
}

static int qwen3_dev_load_layer(qwen3_model_t *model, qwen3_dev_layer_t *dl, int idx) {
    size_t hidden = model->hidden_size, inter = model->intermediate_size;
    size_t q_dim = (size_t)model->num_heads * model->head_dim;
    size_t kv_dim = (size_t)model->num_kv_heads * model->head_dim;
    const char *names[7] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                            "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
                            "mlp.down_proj"};
    const uint16_t *w[7];
    char name[256];
    for (int i = 0; i < 7; i++) {
        snprintf(name, sizeof(name), "%slayers.%d.%s.weight", model->prefix, idx, names[i]);
        w[i] = load_tensor_bf16(model->sf_files, model->num_sf_files, name);
        if (!w[i]) return 0;
    }

    size_t es = sizeof(uint16_t);
    dl->w_qkv = iris_dev_alloc((q_dim + 2 * kv_dim) * hidden * es);
    if (dl->w_qkv) {
        iris_dev_upload(dl->w_qkv, w[0], q_dim * hidden * es);
        iris_dev_upload(dl->w_qkv + q_dim * hidden, w[1], kv_dim * hidden * es);
        iris_dev_upload(dl->w_qkv + (q_dim + kv_dim) * hidden, w[2], kv_dim * hidden * es);
    }
    dl->w_gate_up = iris_dev_alloc(2 * inter * hidden * es);
    if (dl->w_gate_up) {
        iris_dev_upload(dl->w_gate_up, w[4], inter * hidden * es);
        iris_dev_upload(dl->w_gate_up + inter * hidden, w[5], inter * hidden * es);
    }
    dl->w_o = dev_copy(w[3], hidden * q_dim * sizeof(uint16_t));
    dl->w_down = dev_copy(w[6], hidden * inter * sizeof(uint16_t));

    snprintf(name, sizeof(name), "%slayers.%d.input_layernorm.weight", model->prefix, idx);
    dl->in_norm = dev_f32_tensor(model, name, hidden);
    snprintf(name, sizeof(name), "%slayers.%d.post_attention_layernorm.weight", model->prefix, idx);
    dl->post_norm = dev_f32_tensor(model, name, hidden);
    snprintf(name, sizeof(name), "%slayers.%d.self_attn.q_norm.weight", model->prefix, idx);
    dl->q_norm = dev_f32_tensor(model, name, model->head_dim);
    snprintf(name, sizeof(name), "%slayers.%d.self_attn.k_norm.weight", model->prefix, idx);
    dl->k_norm = dev_f32_tensor(model, name, model->head_dim);
    return dl->w_qkv && dl->w_gate_up && dl->w_o && dl->w_down &&
           dl->in_norm && dl->post_norm && dl->q_norm && dl->k_norm;
}

static void qwen3_dev_free_activations(struct qwen3_dev *d) {
    iris_dev_free(d->x);
    iris_dev_free(d->out);
    iris_dev_free(d->norm);
    iris_dev_free(d->qkv);
    iris_dev_free(d->attn);
    iris_dev_free(d->gate_up);
    iris_dev_free(d->act);
    iris_dev_free(d->rope_cos);
    iris_dev_free(d->rope_sin);
    iris_dev_free(d->mask);
    d->x = d->out = d->rope_cos = d->rope_sin = NULL;
    d->norm = d->qkv = d->attn = d->gate_up = d->act = NULL;
    d->mask = NULL;
    d->seq = 0;
}

static void qwen3_dev_free(qwen3_model_t *model) {
    struct qwen3_dev *d = model->dev;
    if (!d) return;
    for (int i = 0; i < d->num_loaded; i++) {
        qwen3_dev_layer_t *dl = &d->layers[i];
        iris_dev_free(dl->w_qkv);
        iris_dev_free(dl->w_o);
        iris_dev_free(dl->w_gate_up);
        iris_dev_free(dl->w_down);
        iris_dev_free(dl->in_norm);
        iris_dev_free(dl->post_norm);
        iris_dev_free(dl->q_norm);
        iris_dev_free(dl->k_norm);
    }
    free(d->layers);
    qwen3_dev_free_activations(d);
    free(d);
    model->dev = NULL;
}

/* Upload layers up to last_layer and size the activations for seq tokens. */
static int qwen3_dev_prepare(qwen3_model_t *model, int last_layer, int seq) {
    if (!model->dev) {
        model->dev = calloc(1, sizeof(struct qwen3_dev));
        if (!model->dev) return 0;
        model->dev->layers = calloc(model->num_layers, sizeof(qwen3_dev_layer_t));
        if (!model->dev->layers) return 0;
    }
    struct qwen3_dev *d = model->dev;
    while (d->num_loaded <= last_layer) {
        int ok = qwen3_dev_load_layer(model, &d->layers[d->num_loaded], d->num_loaded);
        d->num_loaded++;   /* free() handles a partially uploaded layer */
        if (!ok) return 0;
    }
    if (seq > d->seq) {
        size_t hidden = model->hidden_size, inter = model->intermediate_size;
        size_t q_dim = (size_t)model->num_heads * model->head_dim;
        size_t kv_dim = (size_t)model->num_kv_heads * model->head_dim;
        size_t half = model->head_dim / 2;
        qwen3_dev_free_activations(d);
        d->x = iris_dev_alloc(seq * hidden * sizeof(float));
        d->out = iris_dev_alloc(seq * hidden * sizeof(float));
        d->norm = iris_dev_alloc(seq * hidden * sizeof(uint16_t));
        d->qkv = iris_dev_alloc(seq * (q_dim + 2 * kv_dim) * sizeof(uint16_t));
        d->attn = iris_dev_alloc(seq * q_dim * sizeof(uint16_t));
        d->gate_up = iris_dev_alloc(seq * 2 * inter * sizeof(uint16_t));
        d->act = iris_dev_alloc(seq * inter * sizeof(uint16_t));
        d->rope_cos = iris_dev_alloc(seq * half * sizeof(float));
        d->rope_sin = iris_dev_alloc(seq * half * sizeof(float));
        d->mask = iris_dev_alloc(seq * sizeof(int));
        if (!d->x || !d->out || !d->norm || !d->qkv || !d->attn || !d->gate_up ||
            !d->act || !d->rope_cos || !d->rope_sin || !d->mask) {
            qwen3_dev_free_activations(d);
            return 0;
        }
        d->seq = seq;
    }
    return 1;
}

/* Forward pass on the GPU. model->hidden_state holds the token embeddings
 * and the RoPE tables are filled; on success the outputs of the extracted
 * layers are in model->layer_outputs. Returns 0 to fall back to the CPU. */
static int qwen3_forward_dev(qwen3_model_t *model, int seq_len, const int *attention_mask,
                             const int *out_layers, int num_out) {
    int last_layer = out_layers[num_out - 1];
    if (!qwen3_dev_prepare(model, last_layer, seq_len)) {
        fprintf(stderr, "Qwen3: GPU allocation failed for %d tokens\n", seq_len);
        qwen3_dev_free(model);
        return 0;
    }
    struct qwen3_dev *d = model->dev;
    int hidden = model->hidden_size, inter = model->intermediate_size, hd = model->head_dim;
    int q_dim = model->num_heads * hd, kv_dim = model->num_kv_heads * hd;
    int ldqkv = q_dim + 2 * kv_dim;
    float eps = QWEN3_RMS_NORM_EPS;

    iris_dev_upload(d->x, model->hidden_state, (size_t)seq_len * hidden * sizeof(float));
    iris_dev_upload(d->rope_cos, model->rope_cos, (size_t)seq_len * (hd / 2) * sizeof(float));
    iris_dev_upload(d->rope_sin, model->rope_sin, (size_t)seq_len * (hd / 2) * sizeof(float));
    if (attention_mask) iris_dev_upload(d->mask, attention_mask, seq_len * sizeof(int));

    for (int l = 0; l <= last_layer; l++) {
        qwen3_dev_layer_t *dl = &d->layers[l];

        /* Self-attention (causal, padding keys masked) */
        iris_dev_rms_norm(d->norm, d->x, dl->in_norm, seq_len, hidden, eps);
        iris_dev_linear_bf16(d->qkv, d->norm, dl->w_qkv, seq_len, hidden, ldqkv);
        iris_dev_head_norm_rope(d->qkv, ldqkv, dl->q_norm, d->rope_cos, d->rope_sin,
                                seq_len, model->num_heads, hd, eps, 1);
        iris_dev_head_norm_rope(d->qkv + q_dim, ldqkv, dl->k_norm, d->rope_cos, d->rope_sin,
                                seq_len, model->num_kv_heads, hd, eps, 1);
        iris_dev_attention(d->attn, q_dim, d->qkv, d->qkv + q_dim, d->qkv + q_dim + kv_dim,
                           ldqkv, seq_len, seq_len, model->num_heads, model->num_kv_heads, hd,
                           1.0f / sqrtf((float)hd), 1, attention_mask ? d->mask : NULL);
        iris_dev_linear_f32(d->out, d->attn, dl->w_o, seq_len, q_dim, hidden);
        iris_dev_residual(d->x, d->out, NULL, seq_len, hidden);

        /* SwiGLU MLP */
        iris_dev_rms_norm(d->norm, d->x, dl->post_norm, seq_len, hidden, eps);
        iris_dev_linear_bf16(d->gate_up, d->norm, dl->w_gate_up, seq_len, hidden, 2 * inter);
        iris_dev_swiglu(d->act, inter, d->gate_up, d->gate_up + inter, 2 * inter, seq_len, inter);
        iris_dev_linear_f32(d->out, d->act, dl->w_down, seq_len, inter, hidden);
        iris_dev_residual(d->x, d->out, NULL, seq_len, hidden);

        if (model->num_images && l < QWEN3_VISION_DEEPSTACK) {
            float *deep = calloc((size_t)seq_len * hidden, sizeof(float));
            if (!deep) return 0;
            qwen3_add_deepstack(model, deep, l);
            iris_dev_upload(d->out, deep, (size_t)seq_len * hidden * sizeof(float));
            free(deep);
            iris_dev_residual(d->x, d->out, NULL, seq_len, hidden);
        }

        for (int i = 0; i < num_out; i++)
            if (l == out_layers[i])
                iris_dev_download(model->layer_outputs[i], d->x,
                                  (size_t)seq_len * hidden * sizeof(float));
        if (iris_text_progress_callback)
            iris_text_progress_callback(l, model->num_layers);
    }
    return 1;
}
#endif /* QWEN3_DEV */


/* Parse text_encoder/config.json to get architecture dimensions.
 * Sets model fields. Returns 0 on success, -1 on failure. */
static int parse_qwen3_config(const char *model_dir, qwen3_model_t *model) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/config.json", model_dir);

    FILE *f = fopen(path, "r");
    if (!f) return -1;

    char buf[8192];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = '\0';
    fclose(f);

    char *p;
    int hidden = 0, intermediate = 0, num_heads = 0, num_kv_heads = 0;
    int head_dim = 0, vocab_size = 0, num_layers = 0;
    float rope_theta = 0;

    /* Qwen3-VL keeps the text model config in "text_config" (saved
     * before "vision_config", whose fields share the same names). */
    char *base = strstr(buf, "\"text_config\"");
    if (!base) base = buf;

    if ((p = strstr(base, "\"hidden_size\""))) {
        if ((p = strchr(p, ':'))) hidden = atoi(p + 1);
    }
    if ((p = strstr(base, "\"intermediate_size\""))) {
        if ((p = strchr(p, ':'))) intermediate = atoi(p + 1);
    }
    if ((p = strstr(base, "\"num_attention_heads\""))) {
        if ((p = strchr(p, ':'))) num_heads = atoi(p + 1);
    }
    if ((p = strstr(base, "\"num_key_value_heads\""))) {
        if ((p = strchr(p, ':'))) num_kv_heads = atoi(p + 1);
    }
    if ((p = strstr(base, "\"head_dim\""))) {
        if ((p = strchr(p, ':'))) head_dim = atoi(p + 1);
    }
    if ((p = strstr(base, "\"vocab_size\""))) {
        if ((p = strchr(p, ':'))) vocab_size = atoi(p + 1);
    }
    if ((p = strstr(base, "\"num_hidden_layers\""))) {
        if ((p = strchr(p, ':'))) num_layers = atoi(p + 1);
    }
    if ((p = strstr(base, "\"rope_theta\""))) {
        if ((p = strchr(p, ':'))) rope_theta = atof(p + 1);
    }
    if ((p = strstr(base, "\"mrope_section\""))) {
        int frame;
        if ((p = strchr(p, '[')))
            sscanf(p, "[ %d , %d , %d", &frame, &model->mrope_h, &model->mrope_w);
    }

    if (hidden <= 0 || num_heads <= 0) return -1;

    model->hidden_size = hidden;
    model->intermediate_size = intermediate > 0 ? intermediate : 9728;
    model->num_heads = num_heads;
    model->num_kv_heads = num_kv_heads > 0 ? num_kv_heads : 8;
    model->head_dim = head_dim > 0 ? head_dim : 128;
    model->vocab_size = vocab_size > 0 ? vocab_size : QWEN3_VOCAB_SIZE;
    model->num_layers = num_layers > 0 ? num_layers : 36;
    model->rope_theta = rope_theta > 0 ? rope_theta : QWEN3_ROPE_THETA;

    return 0;
}

/* Set default Qwen3-4B architecture values. */
static void qwen3_set_defaults(qwen3_model_t *model) {
    model->hidden_size = 2560;
    model->intermediate_size = 9728;
    model->num_heads = 32;
    model->num_kv_heads = 8;
    model->head_dim = 128;
    model->vocab_size = QWEN3_VOCAB_SIZE;
    model->num_layers = 36;
    model->rope_theta = QWEN3_ROPE_THETA;
}

/* Open all safetensors shards dynamically by reading model.safetensors.index.json.
 * Returns number of files opened, 0 on failure. */
int open_safetensors_shards(const char *model_dir,
                            safetensors_file_t **files, int max_files) {
    char path[1024];

    /* First try: read the index JSON to discover shard filenames */
    snprintf(path, sizeof(path), "%s/model.safetensors.index.json", model_dir);
    FILE *f = fopen(path, "r");
    if (f) {
        /* Read the whole index file */
        fseek(f, 0, SEEK_END);
        long fsize = ftell(f);
        fseek(f, 0, SEEK_SET);
        char *buf = malloc(fsize + 1);
        if (!buf) { fclose(f); return 0; }
        fread(buf, 1, fsize, f);
        buf[fsize] = '\0';
        fclose(f);

        /* Collect unique shard filenames from weight_map values.
         * They look like: "model-00001-of-00003.safetensors" */
        char shard_names[QWEN3_MAX_SHARDS][128];
        int num_shards = 0;

        char *p = buf;
        while ((p = strstr(p, "model-")) != NULL) {
            /* Find the end of the filename */
            char *end = strstr(p, ".safetensors");
            if (!end) { p++; continue; }
            end += strlen(".safetensors");

            int len = (int)(end - p);
            if (len >= 128) { p = end; continue; }

            /* Check if we already have this shard */
            char name[128];
            memcpy(name, p, len);
            name[len] = '\0';

            int found = 0;
            for (int i = 0; i < num_shards; i++) {
                if (strcmp(shard_names[i], name) == 0) {
                    found = 1;
                    break;
                }
            }
            if (!found && num_shards < max_files && num_shards < QWEN3_MAX_SHARDS) {
                strcpy(shard_names[num_shards], name);
                num_shards++;
            }
            p = end;
        }
        free(buf);

        if (num_shards > 0) {
            for (int i = 0; i < num_shards; i++) {
                snprintf(path, sizeof(path), "%s/%s", model_dir, shard_names[i]);
                files[i] = safetensors_open(path);
                if (!files[i]) {
                    fprintf(stderr, "qwen3: failed to open shard %s\n", path);
                    for (int j = 0; j < i; j++) safetensors_close(files[j]);
                    return 0;
                }
            }
            return num_shards;
        }
    }

    /* Fallback: try model-00001-of-00002.safetensors pattern */
    char path1[512], path2[512];
    snprintf(path1, sizeof(path1), "%s/model-00001-of-00002.safetensors", model_dir);
    snprintf(path2, sizeof(path2), "%s/model-00002-of-00002.safetensors", model_dir);

    files[0] = safetensors_open(path1);
    files[1] = safetensors_open(path2);
    if (files[0] && files[1]) return 2;

    if (files[0]) safetensors_close(files[0]);
    if (files[1]) safetensors_close(files[1]);

    /* Single file (Qwen-Image 2.1 Turbo from the Hub) */
    snprintf(path1, sizeof(path1), "%s/model.safetensors", model_dir);
    files[0] = safetensors_open(path1);
    return files[0] ? 1 : 0;
}

/* Detect plain Qwen3 vs the text model of Qwen3-VL (Flux3) from the
 * tensor names, setting the name prefix and the maximum sequence length. */
static void qwen3_detect_variant(qwen3_model_t *model,
                                 safetensors_file_t **files, int num_files) {
    strcpy(model->prefix, "model.");
    model->max_seq_len = QWEN3_MAX_SEQ_LEN;
    for (int f = 0; f < num_files; f++) {
        if (safetensors_find(files[f], "model.language_model.embed_tokens.weight")) {
            strcpy(model->prefix, "model.language_model.");
            model->max_seq_len = QWEN3_VL_MAX_SEQ_LEN;
            break;
        }
    }
}

/* Load the token embeddings and the final norm, which every mode keeps
 * resident. Returns 0 on success, -1 on failure. */
static int qwen3_load_globals(qwen3_model_t *model,
                              safetensors_file_t **files, int num_files) {
    char name[256];
    snprintf(name, sizeof(name), "%sembed_tokens.weight", model->prefix);
    model->embed_tokens = load_tensor(files, num_files, name);
    snprintf(name, sizeof(name), "%snorm.weight", model->prefix);
    model->norm_weight = load_tensor(files, num_files, name);
    return (model->embed_tokens && model->norm_weight) ? 0 : -1;
}

/* Eager-mode model loading: reads all weights into RAM upfront. Parses
 * config.json for architecture parameters (hidden size, head counts, etc.),
 * loads all layer weights from safetensors shards, and precomputes RoPE
 * frequency tables. Uses ~16GB RAM for 4B model. Prefer mmap mode for
 * lower memory usage unless weights need repeated random access. */
qwen3_model_t *qwen3_model_load(const char *model_dir) {
    qwen3_model_t *model = calloc(1, sizeof(qwen3_model_t));
    if (!model) return NULL;

    /* Parse config, fall back to defaults */
    if (parse_qwen3_config(model_dir, model) != 0) {
        qwen3_set_defaults(model);
    }

    model->layers = calloc(model->num_layers, sizeof(qwen3_layer_t));
    if (!model->layers) {
        free(model);
        return NULL;
    }

    /* Open safetensors shards dynamically */
    safetensors_file_t *files[QWEN3_MAX_SHARDS];
    int num_files = open_safetensors_shards(model_dir, files, QWEN3_MAX_SHARDS);
    if (num_files == 0) {
        fprintf(stderr, "qwen3_model_load: failed to open safetensors files\n");
        free(model->layers);
        free(model);
        return NULL;
    }

    qwen3_detect_variant(model, files, num_files);
    if (qwen3_load_globals(model, files, num_files) != 0) {
        fprintf(stderr, "qwen3_model_load: failed to load embeddings/final norm\n");
        goto error;
    }

    /* Load layer weights */
    for (int i = 0; i < model->num_layers; i++) {
        if (load_layer_weights(&model->layers[i], files, num_files, model->prefix, i) != 0) {
            fprintf(stderr, "qwen3_model_load: failed to load layer %d\n", i);
            goto error;
        }
    }

    for (int i = 0; i < num_files; i++) safetensors_close(files[i]);

    return model;

error:
    for (int i = 0; i < num_files; i++) safetensors_close(files[i]);
    qwen3_model_free(model);
    return NULL;
}

/* Load model in mmap mode - keeps safetensors files open and loads layer weights
 * on-demand during forward pass. Reduces peak memory from ~16GB to ~2GB. */
/* Memory-mapped loading: keeps safetensors files open and loads layer weights
 * on demand during forward pass. Only embeddings, final norm, and RoPE tables
 * are resident. Dramatically reduces startup time and peak memory (~2GB vs
 * ~16GB). For GPU: also loads bf16 weight pointers for zero-copy GPU upload
 * directly from the mmap region. */
qwen3_model_t *qwen3_model_load_mmap(const char *model_dir) {
    qwen3_model_t *model = calloc(1, sizeof(qwen3_model_t));
    if (!model) return NULL;

    model->use_mmap = 1;

    /* Parse config, fall back to defaults */
    if (parse_qwen3_config(model_dir, model) != 0) {
        qwen3_set_defaults(model);
    }

#ifdef USE_METAL
    /* Enable bf16 GPU acceleration when Metal is available.
     * Set IRIS_QWEN3_NO_BF16=1 to disable for debugging. */
    model->use_bf16 = (iris_metal_available() && !getenv("IRIS_QWEN3_NO_BF16")) ? 1 : 0;
    if (model->use_bf16) {
        if (iris_verbose)
            fprintf(stderr, "Qwen3: bf16 GPU acceleration enabled\n");
    }
#endif
    model->layers = calloc(model->num_layers, sizeof(qwen3_layer_t));
    if (!model->layers) {
        free(model);
        return NULL;
    }

    /* Open safetensors shards dynamically and keep them open */
    model->num_sf_files = open_safetensors_shards(model_dir, model->sf_files, QWEN3_MAX_SHARDS);
    if (model->num_sf_files == 0) {
        fprintf(stderr, "qwen3_model_load_mmap: failed to open safetensors files\n");
        goto error;
    }

    /* Load only embeddings and final norm, layers are loaded on demand */
    qwen3_detect_variant(model, model->sf_files, model->num_sf_files);
    if (qwen3_load_globals(model, model->sf_files, model->num_sf_files) != 0) {
        fprintf(stderr, "qwen3_model_load_mmap: failed to load embeddings/final norm\n");
        goto error;
    }

    /* DON'T load layer weights - they'll be loaded on-demand in forward pass */
    if (iris_verbose)
        fprintf(stderr, "Mmap mode: layer weights will be loaded on-demand\n");

    return model;

error:
    qwen3_model_free(model);
    return NULL;
}

void qwen3_model_free(qwen3_model_t *model) {
    if (!model) return;

    free(model->embed_tokens);
    free(model->norm_weight);

    if (model->layers) {
        for (int i = 0; i < model->num_layers; i++) {
            qwen3_layer_t *layer = &model->layers[i];
            free(layer->input_layernorm_weight);
            free(layer->post_attention_layernorm_weight);
            free(layer->attn.q_proj_weight);
            free(layer->attn.k_proj_weight);
            free(layer->attn.v_proj_weight);
            free(layer->attn.o_proj_weight);
            free(layer->attn.q_norm_weight);
            free(layer->attn.k_norm_weight);
            free(layer->mlp.gate_proj_weight);
            free(layer->mlp.up_proj_weight);
            free(layer->mlp.down_proj_weight);
        }
        free(model->layers);
    }

    qwen3_free_work(model);

#ifdef QWEN3_DEV
    qwen3_dev_free(model);
#endif

    /* Close mmap'd safetensors files if open */
    for (int i = 0; i < model->num_sf_files; i++) {
        if (model->sf_files[i]) safetensors_close(model->sf_files[i]);
    }

    free(model);
}

/* ========================================================================
 * Combined Encoder API
 * ======================================================================== */

/* Flux3 ships the tokenizer inside text_encoder/, Qwen-Image inside processor/. */
static qwen3_tokenizer_t *qwen3_tokenizer_load_dir(const char *model_dir) {
    static const char *tok_dirs[] = {"tokenizer", "text_encoder", "processor"};
    char tok_path[512];
    for (int i = 0; i < 3; i++) {
        snprintf(tok_path, sizeof(tok_path), "%s/%s/tokenizer.json", model_dir, tok_dirs[i]);
        FILE *tf = fopen(tok_path, "r");
        if (tf) {
            fclose(tf);
            break;
        }
    }
    return qwen3_tokenizer_load(tok_path);
}

qwen3_encoder_t *qwen3_encoder_load(const char *model_dir, int use_mmap) {
    qwen3_encoder_t *enc = calloc(1, sizeof(qwen3_encoder_t));
    if (!enc) return NULL;

    enc->tokenizer = qwen3_tokenizer_load_dir(model_dir);
    if (!enc->tokenizer) {
        fprintf(stderr, "qwen3_encoder_load: failed to load tokenizer\n");
        free(enc);
        return NULL;
    }

    /* Load model - use mmap mode if requested (saves ~14GB RAM) */
#if defined(QWEN3_DEV) && !defined(USE_METAL)
    /* The GPU path uploads the bf16 weights straight from the mapped files. */
    if (iris_dev_available()) use_mmap = 1;
#endif
    char model_path[512];
    snprintf(model_path, sizeof(model_path), "%s/text_encoder", model_dir);
    if (use_mmap) {
        enc->model = qwen3_model_load_mmap(model_path);
    } else {
        enc->model = qwen3_model_load(model_path);
    }
    if (!enc->model) {
        fprintf(stderr, "qwen3_encoder_load: failed to load model\n");
        qwen3_tokenizer_free(enc->tokenizer);
        free(enc);
        return NULL;
    }

    return enc;
}

/* Select which hidden states form the embeddings (QWEN3_EXTRACT_*), which
 * determines the embedding size the transformer expects. Must be called
 * before encoding. */
void qwen3_set_extraction_mode(qwen3_encoder_t *enc, int mode) {
    if (!enc || !enc->model) return;
    qwen3_model_t *model = enc->model;
    int *layers = model->out_layers;
    model->extraction_mode = mode;
    /* Python hidden_states[i] is the output after layer i-1. */
    switch (mode) {
    case QWEN3_EXTRACT_ZIMAGE:  /* hidden_states[-2] */
        layers[0] = model->num_layers - 2;
        model->num_out_layers = 1;
        break;
    case QWEN3_EXTRACT_FLUX3:   /* hidden_states[4, 8, ..., 32] */
        for (int i = 0; i < 8; i++) layers[i] = 4 * (i + 1) - 1;
        model->num_out_layers = 8;
        break;
    case QWEN3_EXTRACT_QIMAGE:  /* last layer, before the final norm */
        layers[0] = model->num_layers - 1;
        model->num_out_layers = 1;
        model->max_seq_len = QWEN3_QIMAGE_MAX_SEQ_LEN;
        break;
    default:                    /* hidden_states[9, 18, 27] */
        layers[0] = 8;
        layers[1] = 17;
        layers[2] = 26;
        model->num_out_layers = 3;
    }
}

void qwen3_encoder_free(qwen3_encoder_t *enc) {
    if (!enc) return;
    qwen3_tokenizer_free(enc->tokenizer);
    qwen3_model_free(enc->model);
    free(enc);
}

/* Pad tokens (freed here), run the forward pass and return the extracted
 * embeddings. Qwen-Image keeps only the real tokens after its system block. */
static float *qwen3_encode_tokens(qwen3_encoder_t *enc, int *tokens, int num_tokens,
                                  int num_system, int *out_num_tokens, int *out_seq_len) {
    int mode = enc->model->extraction_mode;
    int vl = mode == QWEN3_EXTRACT_FLUX3 || mode == QWEN3_EXTRACT_QIMAGE;
    int max_len = vl ? enc->model->max_seq_len : QWEN3_MAX_SEQ_LEN;

    /* Qwen3-VL prompts pad to a multiple of 80. For Qwen-Image the padding
     * only follows the real tokens, which the causal encoder keeps
     * unchanged, and is dropped below. */
    int seq_len = max_len;
    if (vl) {
        int m = QWEN3_FLUX3_PAD_MULTIPLE;
        seq_len = (num_tokens + m - 1) / m * m;
        if (seq_len > max_len) seq_len = max_len;
    }

    /* Pad to the sequence length */
    int *attention_mask = malloc(seq_len * sizeof(int));
    int *padded_tokens = qwen3_pad_tokens(tokens, num_tokens, seq_len, attention_mask);
    free(tokens);

    if (!padded_tokens) {
        free(attention_mask);
        return NULL;
    }

    if (out_num_tokens) *out_num_tokens = num_tokens;
    if (out_seq_len) *out_seq_len = seq_len;

    /* Forward pass */
    float *embeddings = qwen3_forward(enc->model, padded_tokens, attention_mask, seq_len);

    free(padded_tokens);
    free(attention_mask);

    /* Qwen-Image: keep the real tokens after the system block. */
    if (embeddings && mode == QWEN3_EXTRACT_QIMAGE) {
        size_t row = (size_t)enc->model->hidden_size;
        num_tokens -= num_system;
        memmove(embeddings, embeddings + num_system * row, num_tokens * row * sizeof(float));
        if (out_num_tokens) *out_num_tokens = num_tokens;
        if (out_seq_len) *out_seq_len = num_tokens;
    }

    return embeddings;
}

/* Main text encoding API. Tokenizes the prompt using Qwen3 chat template
 * (with <think> tags for Flux only, with a system message for Qwen-Image),
 * pads it (to 512 tokens, or for the Qwen3-VL models to the next multiple
 * of 80), runs the forward pass, and returns the extracted embeddings.
 * Also returns the number of real (non-padding) tokens via out_num_tokens,
 * which Z-Image needs for its unpadded attention over text tokens, and the
 * padded length via out_seq_len. Qwen-Image returns only the real tokens
 * after its system block, and both counts are that number. */
float *qwen3_encode_text_ex(qwen3_encoder_t *enc, const char *prompt,
                              int *out_num_tokens, int *out_seq_len) {
    if (!enc || !enc->tokenizer || !enc->model || !prompt) return NULL;

    int mode = enc->model->extraction_mode;
    int vl = mode == QWEN3_EXTRACT_FLUX3 || mode == QWEN3_EXTRACT_QIMAGE;
    int max_len = vl ? enc->model->max_seq_len : QWEN3_MAX_SEQ_LEN;

    /* Tokenize with chat template */
    int num_tokens, num_system = 0;
    int *tokens = (mode == QWEN3_EXTRACT_QIMAGE)
        ? qwen3_tokenize_chat_system(enc->tokenizer, QWEN3_QIMAGE_SYSTEM, prompt,
                                     &num_tokens, &num_system, max_len)
        : qwen3_tokenize_chat(enc->tokenizer, prompt, &num_tokens,
                              max_len, mode != QWEN3_EXTRACT_FLUX);
    if (!tokens) return NULL;
    return qwen3_encode_tokens(enc, tokens, num_tokens, num_system,
                               out_num_tokens, out_seq_len);
}

size_t qwen3_memory(const char *model_dir, int use_mmap) {
    qwen3_model_t m = {0};
    char dir[512];
    snprintf(dir, sizeof(dir), "%s/text_encoder", model_dir);
    if (parse_qwen3_config(dir, &m) != 0) qwen3_set_defaults(&m);
    size_t hidden = m.hidden_size, q_dim = (size_t)m.num_heads * m.head_dim;
    size_t kv_dim = (size_t)m.num_kv_heads * m.head_dim;
    size_t bf16 = m.num_layers * hidden * (2 * q_dim + 2 * kv_dim + 3 * m.intermediate_size) * 2;
    size_t embed = m.vocab_size * hidden * sizeof(float);
#ifdef USE_METAL
    if (!use_mmap) return embed + bf16 * 4;   /* f32, and the GPU's copy of it */
#elif defined(QWEN3_DEV)
    if (iris_dev_available()) use_mmap = 1;   /* see qwen3_encoder_load() */
#endif
    if (!use_mmap) return embed + bf16 * 2;
#ifdef QWEN3_DEV
    if (iris_dev_available()) return embed + bf16;
#endif
    return embed;
}

int qwen3_count_images(const char *model_dir, const char *prompt, const int *slots, int n) {
    qwen3_tokenizer_t *tok = qwen3_tokenizer_load_dir(model_dir);
    int *rows = malloc((n + 1) * sizeof(int));
    int num_tokens = 0, num_system;
    if (tok && rows)
        free(qwen3_tokenize_chat_images(tok, QWEN3_QIMAGE_SYSTEM, prompt, slots, n,
                                        &num_tokens, &num_system, rows));
    free(rows);
    if (tok) qwen3_tokenizer_free(tok);
    return num_tokens;
}

float *qwen3_encode_images(qwen3_encoder_t *enc, const char *prompt,
                           const qwen3_image_t *images, int n,
                           int *out_seq, int *pos) {
    qwen3_model_t *model = enc->model;
    int *slots = malloc(n * sizeof(int)), *rows = malloc(n * sizeof(int));
    int num_tokens, num_system, seq = 0;
    float *emb = NULL;
    if (slots && rows) {
        for (int i = 0; i < n; i++) slots[i] = images[i].h * images[i].w;
        int *tokens = qwen3_tokenize_chat_images(enc->tokenizer, QWEN3_QIMAGE_SYSTEM, prompt,
                                                 slots, n, &num_tokens, &num_system, rows);
        if (num_tokens > model->max_seq_len) {
            fprintf(stderr, "Qwen3: the prompt with its images needs %d tokens, "
                    "the limit is %d\n", num_tokens, model->max_seq_len);
            free(tokens);
        } else {
            model->images = images;
            model->image_rows = rows;
            model->num_images = n;
            emb = qwen3_encode_tokens(enc, tokens, num_tokens, num_system, NULL, &seq);
            model->images = NULL;
            model->image_rows = NULL;
            model->num_images = 0;
        }
    }

    /* The transformer fills the slots with image latents: drop them here. */
    if (emb) {
        size_t row = model->hidden_size;
        int dst = 0, src = 0;
        for (int i = 0; i <= n; i++) {
            int end = i < n ? rows[i] - num_system : seq;
            memmove(emb + dst * row, emb + src * row, (size_t)(end - src) * row * sizeof(float));
            dst += end - src;
            if (i < n) {
                pos[i] = dst;
                src = end + slots[i];
            }
        }
        *out_seq = dst;
    }
    free(slots);
    free(rows);
    return emb;
}

float *qwen3_encode_text(qwen3_encoder_t *enc, const char *prompt) {
    return qwen3_encode_text_ex(enc, prompt, NULL, NULL);
}
