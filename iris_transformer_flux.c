/*
 * FLUX Diffusion Transformer Implementation
 *
 * Flux2Transformer2DModel - Rectified Flow Transformer for image generation.
 *
 * Architecture (klein 4B):
 * - 5 double-stream blocks (MM-DiT: separate image/text processing, joint attention)
 * - 20 single-stream blocks (parallel DiT: fused QKV+FFN)
 * - 24 attention heads, 128 dim per head (3072 hidden)
 * - No bias parameters
 * - SwiGLU activation
 * - RoPE positional embeddings
 * - Shared AdaLN-Zero modulation
 */

#include "iris.h"
#include "iris_kernels.h"
#include "iris_safetensors.h"
#ifdef IRIS_DEV
#include "iris_dev.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

/* External timing counters from iris_sample.c */
extern double iris_timing_transformer_total;
extern double iris_timing_transformer_double;
extern double iris_timing_transformer_single;
extern double iris_timing_transformer_final;

/* Fine-grained profiling for BLAS optimization */
static double prof_single_adaln = 0;
static double prof_single_fused_matmul = 0;
static double prof_single_split = 0;
static double prof_single_qknorm_rope = 0;
static double prof_single_attention = 0;
static double prof_single_swiglu = 0;
static double prof_single_proj_matmul = 0;
static double prof_single_gated_add = 0;

static double prof_get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

void iris_print_blas_profile(void) {
    double total = prof_single_adaln + prof_single_fused_matmul + prof_single_split +
                   prof_single_qknorm_rope + prof_single_attention + prof_single_swiglu +
                   prof_single_proj_matmul + prof_single_gated_add;
    if (total < 1.0) return;
    fprintf(stderr, "\nSingle block breakdown (cumulative):\n");
    fprintf(stderr, "  AdaLN+mod:     %7.1fms (%4.1f%%)\n", prof_single_adaln, 100*prof_single_adaln/total);
    fprintf(stderr, "  Fused QKV+MLP: %7.1fms (%4.1f%%)\n", prof_single_fused_matmul, 100*prof_single_fused_matmul/total);
    fprintf(stderr, "  Split:         %7.1fms (%4.1f%%)\n", prof_single_split, 100*prof_single_split/total);
    fprintf(stderr, "  QKnorm+RoPE:   %7.1fms (%4.1f%%)\n", prof_single_qknorm_rope, 100*prof_single_qknorm_rope/total);
    fprintf(stderr, "  Attention:     %7.1fms (%4.1f%%)\n", prof_single_attention, 100*prof_single_attention/total);
    fprintf(stderr, "  SwiGLU:        %7.1fms (%4.1f%%)\n", prof_single_swiglu, 100*prof_single_swiglu/total);
    fprintf(stderr, "  Proj matmul:   %7.1fms (%4.1f%%)\n", prof_single_proj_matmul, 100*prof_single_proj_matmul/total);
    fprintf(stderr, "  Gated add:     %7.1fms (%4.1f%%)\n", prof_single_gated_add, 100*prof_single_gated_add/total);
    fprintf(stderr, "  Total:         %7.1fms\n", total);
}

void iris_reset_blas_profile(void) {
    prof_single_adaln = prof_single_fused_matmul = prof_single_split = 0;
    prof_single_qknorm_rope = prof_single_attention = prof_single_swiglu = 0;
    prof_single_proj_matmul = prof_single_gated_add = 0;
}

/* Helper to get current time in ms (wall-clock) */
static double tf_get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* Use BLAS for matrix operations when enabled via Makefile */
#ifdef USE_BLAS
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#include <pthread.h>
#include <unistd.h>
#endif

/* Use Metal for GPU acceleration when available */
#ifdef USE_METAL
#include "iris_metal.h"
#endif

/* Enable BF16 pipeline debug logging when IRIS_BF16_DEBUG is set. */
#ifdef USE_METAL
static int bf16_debug_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        enabled = getenv("IRIS_BF16_DEBUG") ? 1 : 0;
    }
    return enabled;
}

#define BF16_DEBUG(...) \
    do { \
        if (bf16_debug_enabled()) { \
            fprintf(stderr, __VA_ARGS__); \
        } \
    } while (0)
#else
#define BF16_DEBUG(...) do { } while (0)
#endif

/* Helper macro for using bf16 linear layer when available.
 * Uses bf16 if w_bf16 is not NULL (GPU), otherwise falls back to f32. */
#define LINEAR_BF16_OR_F32(out, x, w_f32, w_bf16, seq, in_dim, out_dim) \
    do { \
        if ((w_bf16) != NULL) { \
            iris_linear_nobias_bf16((out), (x), (w_bf16), (seq), (in_dim), (out_dim)); \
        } else if ((w_f32) != NULL) { \
            iris_linear_nobias((out), (x), (w_f32), (seq), (in_dim), (out_dim)); \
        } else { \
            fprintf(stderr, "FATAL: Neither bf16 nor f32 weights available\n"); \
            exit(1); \
        } \
    } while(0)

/* Gated add: out += gate * proj, where gate is [hidden] and proj is [seq, hidden]
 * Double loop avoids modulo which prevents vectorization.
 */
static inline void gated_add(float *out, const float *gate, const float *proj,
                             int seq, int hidden) {
    for (int s = 0; s < seq; s++) {
        for (int i = 0; i < hidden; i++) {
            out[s * hidden + i] += gate[i] * proj[s * hidden + i];
        }
    }
}

/* ========================================================================
 * Transformer Data Structures
 * ======================================================================== */

/* AdaLN-Zero modulation parameters (shared across blocks) */
typedef struct {
    float *mod_weight;      /* [hidden * 6] for double, [hidden * 3] for single */
} adaln_t;

/* Double-stream block (MM-DiT style) */
typedef struct {
    /* Image stream - separate Q, K, V weights (f32 and bf16) */
    float *img_q_weight;            /* [hidden, hidden] (f32) */
    float *img_k_weight;            /* [hidden, hidden] (f32) */
    float *img_v_weight;            /* [hidden, hidden] (f32) */
    uint16_t *img_q_weight_bf16;    /* [hidden, hidden] (bf16) */
    uint16_t *img_k_weight_bf16;    /* [hidden, hidden] (bf16) */
    uint16_t *img_v_weight_bf16;    /* [hidden, hidden] (bf16) */
    float *img_norm_q_weight;       /* [head_dim] - QK norm on Q (always f32) */
    float *img_norm_k_weight;       /* [head_dim] - QK norm on K (always f32) */
    float *img_proj_weight;         /* [hidden, hidden] (f32) */
    uint16_t *img_proj_weight_bf16; /* [hidden, hidden] (bf16) */
    float *img_mlp_gate_weight;     /* [mlp_hidden, hidden] (f32) */
    float *img_mlp_up_weight;       /* [mlp_hidden, hidden] (f32) */
    float *img_mlp_down_weight;     /* [hidden, mlp_hidden] (f32) */
    uint16_t *img_mlp_gate_weight_bf16; /* [mlp_hidden, hidden] (bf16) */
    uint16_t *img_mlp_up_weight_bf16;   /* [mlp_hidden, hidden] (bf16) */
    uint16_t *img_mlp_down_weight_bf16; /* [hidden, mlp_hidden] (bf16) */

    /* Text stream - separate Q, K, V weights (f32 and bf16) */
    float *txt_q_weight;            /* [hidden, hidden] (f32) */
    float *txt_k_weight;            /* [hidden, hidden] (f32) */
    float *txt_v_weight;            /* [hidden, hidden] (f32) */
    uint16_t *txt_q_weight_bf16;    /* [hidden, hidden] (bf16) */
    uint16_t *txt_k_weight_bf16;    /* [hidden, hidden] (bf16) */
    uint16_t *txt_v_weight_bf16;    /* [hidden, hidden] (bf16) */
    float *txt_norm_q_weight;       /* [head_dim] - QK norm on Q (always f32) */
    float *txt_norm_k_weight;       /* [head_dim] - QK norm on K (always f32) */
    float *txt_proj_weight;         /* [hidden, hidden] (f32) */
    uint16_t *txt_proj_weight_bf16; /* [hidden, hidden] (bf16) */
    float *txt_mlp_gate_weight;     /* [mlp_hidden, hidden] (f32) */
    float *txt_mlp_up_weight;       /* [mlp_hidden, hidden] (f32) */
    float *txt_mlp_down_weight;     /* [hidden, mlp_hidden] (f32) */
    uint16_t *txt_mlp_gate_weight_bf16; /* [mlp_hidden, hidden] (bf16) */
    uint16_t *txt_mlp_up_weight_bf16;   /* [mlp_hidden, hidden] (bf16) */
    uint16_t *txt_mlp_down_weight_bf16; /* [hidden, mlp_hidden] (bf16) */
} double_block_t;

/* Single-stream block (Parallel DiT style, fused) */
typedef struct {
    /* Fused QKV + FFN input projection - bf16 for GPU, f32 as fallback */
    float *qkv_mlp_weight;          /* [hidden*3 + mlp_hidden*2, hidden] (f32) */
    uint16_t *qkv_mlp_weight_bf16;  /* [hidden*3 + mlp_hidden*2, hidden] (bf16) */
    /* QK normalization - always f32 (small) */
    float *norm_q_weight;           /* [head_dim] */
    float *norm_k_weight;           /* [head_dim] */
    /* Fused attention out + FFN down projection - bf16 for GPU, f32 as fallback */
    float *proj_mlp_weight;         /* [hidden, hidden + mlp_hidden] (f32) */
    uint16_t *proj_mlp_weight_bf16; /* [hidden, hidden + mlp_hidden] (bf16) */
} single_block_t;

/* Timestep embedding MLP
 * FLUX.2-klein uses 256-dim sinusoidal embedding (128 frequencies)
 * linear_1: [hidden, 256] - projects sinusoidal to hidden
 * linear_2: [hidden, hidden] - another linear layer
 */
typedef struct {
    float *fc1_weight;              /* [hidden, 256] */
    float *fc2_weight;              /* [hidden, hidden] */
    int sincos_dim;                 /* 256 for FLUX.2-klein */
} time_embed_t;

/* Full transformer context */
typedef struct iris_transformer_flux {
#ifdef IRIS_DEV
    struct dev_flux *dev;
#endif

    /* Configuration */
    int hidden_size;        /* 3072 */
    int num_heads;          /* 24 */
    int head_dim;           /* 128 */
    int mlp_hidden;         /* hidden * 3 = 9216 */
    int num_double_layers;  /* 5 */
    int num_single_layers;  /* 20 */
    int text_dim;           /* 7680 */
    int latent_channels;    /* 128 */
    float rope_theta;       /* 2000 */
    int rope_dim;           /* 128 */
    int use_bf16;           /* Use bf16 weights (1) or f32 (0) */

    /* Input projections */
    float *img_in_weight;   /* [hidden, latent_channels] */
    float *txt_in_weight;   /* [hidden, text_dim] */
    uint16_t *img_in_weight_bf16;   /* [hidden, latent_channels] (bf16) */
    uint16_t *txt_in_weight_bf16;   /* [hidden, text_dim] (bf16) */

    /* Timestep embedding */
    time_embed_t time_embed;
    float *time_freq;       /* [hidden/2] - precomputed sinusoidal frequencies */

    /* Shared AdaLN modulation (f32 and bf16) */
    float *adaln_double_img_weight;  /* [hidden * 6, hidden] for double block img stream */
    float *adaln_double_txt_weight;  /* [hidden * 6, hidden] for double block txt stream */
    float *adaln_single_weight;      /* [hidden * 3, hidden] for single block */
    uint16_t *adaln_double_img_weight_bf16;  /* (bf16) */
    uint16_t *adaln_double_txt_weight_bf16;  /* (bf16) */
    uint16_t *adaln_single_weight_bf16;      /* (bf16) */

    /* Transformer blocks */
    double_block_t *double_blocks;
    single_block_t *single_blocks;

    /* Final layer */
    float *final_norm_weight;       /* [hidden] (always f32) */
    float *final_proj_weight;       /* [latent_channels, hidden] */
    uint16_t *final_proj_weight_bf16; /* [latent_channels, hidden] (bf16) */

    /* RoPE frequencies (precomputed) */
    float *rope_freqs;              /* [max_seq, head_dim/2, 2] - legacy 1D */
    float *rope_cos;                /* [max_seq, axis_dim] - 2D cos frequencies */
    float *rope_sin;                /* [max_seq, axis_dim] - 2D sin frequencies */
    int max_seq_len;
    int axis_dim;                   /* 32 for FLUX (128 head_dim / 4 axes) */

    /* Working memory */
    float *img_hidden;              /* [max_img_seq, hidden] */
    float *txt_hidden;              /* [max_txt_seq, hidden] */
    float *q, *k, *v;               /* [max_seq, hidden] */
    float *attn_out;                /* [max_seq, hidden] */
    float *mlp_buffer;              /* [max_seq, mlp_hidden] */
    float *work1, *work2;
    size_t work_size;

    /* Pre-allocated attention workspaces to avoid malloc in hot path */
    float *attn_q_t;                /* [max_seq, hidden] transposed Q */
    float *attn_k_t;                /* [max_seq, hidden] transposed K */
    float *attn_v_t;                /* [max_seq, hidden] transposed V */
    float *attn_out_t;              /* [max_seq, hidden] transposed output */
    float *attn_scores;             /* [num_heads, seq, seq] attention scores (BLAS/Metal only) */
    size_t attn_scores_alloc;       /* Currently allocated size in bytes */
    int work_seq_alloc;             /* Currently allocated sequence length for work buffers */
    float *attn_cat_k;              /* [max_seq, hidden] concatenated K */
    float *attn_cat_v;              /* [max_seq, hidden] concatenated V */

    /* Single-block work buffers (pre-allocated to avoid malloc in hot path) */
    float *single_q;                /* [max_seq, hidden] */
    float *single_k;                /* [max_seq, hidden] */
    float *single_v;                /* [max_seq, hidden] */
    float *single_mlp_gate;         /* [max_seq, mlp_hidden] */
    float *single_mlp_up;           /* [max_seq, mlp_hidden] */
    float *single_attn_out;         /* [max_seq, hidden] */
    float *single_concat;           /* [max_seq, hidden + mlp_hidden] */

    /* FFN work buffers (shared by double and single blocks) */
    float *ffn_gate;                /* [max_seq, mlp_hidden] */
    float *ffn_up;                  /* [max_seq, mlp_hidden] */

    /* Double-block work buffers */
    float *t_emb_silu;              /* [hidden] */
    float *double_mod_img;          /* [hidden * 6] */
    float *double_mod_txt;          /* [hidden * 6] */
    float *double_img_attn_out;     /* [max_seq, hidden] */
    float *double_txt_attn_out;     /* [max_seq, hidden] */

    /* Cached 2D RoPE embeddings (to avoid malloc/compute each step) */
    float *cached_img_rope_cos;     /* [img_seq * axis_dim * 4] */
    float *cached_img_rope_sin;
    int cached_img_h;
    int cached_img_w;
    float *cached_ref_rope_cos;     /* [ref_seq * axis_dim * 4] for img2img */
    float *cached_ref_rope_sin;
    int cached_ref_h;
    int cached_ref_w;
    int cached_ref_t_offset;
    float *cached_txt_rope_cos;     /* [txt_seq * head_dim] */
    float *cached_txt_rope_sin;
    int cached_txt_seq;
    /* Combined RoPE cache for img2img (target + reference) */
    float *cached_combined_rope_cos;  /* [(img_seq + ref_seq) * axis_dim * 4] */
    float *cached_combined_rope_sin;
    int cached_combined_img_h;
    int cached_combined_img_w;
    int cached_combined_ref_h;
    int cached_combined_ref_w;
    int cached_combined_t_offset;

    /* Mmap mode: keep safetensors file open, load block weights on-demand */
    int use_mmap;
    #define MAX_TF_SHARDS 4
    safetensors_file_t *sf_files[MAX_TF_SHARDS];
    int num_sf_files;
} iris_transformer_flux_t;

/* ========================================================================
 * Transformer Config Parsing
 * ======================================================================== */

/* Parse transformer/config.json to get architecture dimensions.
 * Returns 0 on success, -1 on failure (caller should use defaults). */
static int parse_transformer_config(const char *model_dir, iris_transformer_flux_t *tf) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/transformer/config.json", model_dir);

    FILE *f = fopen(path, "r");
    if (!f) return -1;

    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    buf[n] = '\0';
    fclose(f);

    /* Simple JSON integer/float extraction.
     * Look for "key": value patterns. */
    char *p;
    int num_heads = 0, head_dim = 0, num_layers = 0, num_single = 0;
    int joint_attention_dim = 0, in_channels = 0;
    float mlp_ratio = 0, rope_theta = 0;

    if ((p = strstr(buf, "\"num_attention_heads\""))) {
        if ((p = strchr(p, ':'))) num_heads = atoi(p + 1);
    }
    if ((p = strstr(buf, "\"attention_head_dim\""))) {
        if ((p = strchr(p, ':'))) head_dim = atoi(p + 1);
    }
    if ((p = strstr(buf, "\"num_layers\""))) {
        if ((p = strchr(p, ':'))) num_layers = atoi(p + 1);
    }
    if ((p = strstr(buf, "\"num_single_layers\""))) {
        if ((p = strchr(p, ':'))) num_single = atoi(p + 1);
    }
    if ((p = strstr(buf, "\"joint_attention_dim\""))) {
        if ((p = strchr(p, ':'))) joint_attention_dim = atoi(p + 1);
    }
    if ((p = strstr(buf, "\"in_channels\""))) {
        if ((p = strchr(p, ':'))) in_channels = atoi(p + 1);
    }
    if ((p = strstr(buf, "\"mlp_ratio\""))) {
        if ((p = strchr(p, ':'))) mlp_ratio = atof(p + 1);
    }
    if ((p = strstr(buf, "\"rope_theta\""))) {
        if ((p = strchr(p, ':'))) rope_theta = atof(p + 1);
    }

    /* Validate: we need at least heads and head_dim */
    if (num_heads <= 0 || head_dim <= 0) return -1;

    tf->num_heads = num_heads;
    tf->head_dim = head_dim;
    tf->hidden_size = num_heads * head_dim;
    tf->mlp_hidden = (int)(tf->hidden_size * (mlp_ratio > 0 ? mlp_ratio : 3.0f));
    tf->num_double_layers = num_layers > 0 ? num_layers : 5;
    tf->num_single_layers = num_single > 0 ? num_single : 20;
    tf->text_dim = joint_attention_dim > 0 ? joint_attention_dim : 7680;
    tf->latent_channels = in_channels > 0 ? in_channels : 128;
    tf->rope_theta = rope_theta > 0 ? rope_theta : 2000.0f;
    tf->rope_dim = head_dim;
    tf->axis_dim = head_dim / 4;  /* 4 RoPE axes */

    return 0;
}

/* Open transformer safetensors shards.
 * Reads diffusion_pytorch_model.safetensors.index.json for shard filenames,
 * falls back to single diffusion_pytorch_model.safetensors.
 * Returns number of files opened (0 on failure). */
static int open_transformer_shards(const char *model_dir,
                                   safetensors_file_t **files, int max_files) {
    char path[1024];
    int num_files = 0;

    /* Try to read index JSON for sharded models */
    snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors.index.json",
             model_dir);
    FILE *fp = fopen(path, "r");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        long len = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        char *json = malloc(len + 1);
        if (json) {
            fread(json, 1, len, fp);
            json[len] = '\0';
            fclose(fp);

            /* Extract unique shard filenames from weight_map values */
            char shard_names[MAX_TF_SHARDS][256];
            int num_shards = 0;
            const char *p = strstr(json, "\"weight_map\"");
            if (p) {
                p = strchr(p, '{');
                if (p) p++;
                while (p && num_shards < max_files) {
                    /* Find next value (shard filename) */
                    const char *colon = strchr(p, ':');
                    if (!colon) break;
                    const char *q1 = strchr(colon, '"');
                    if (!q1) break;
                    q1++;
                    const char *q2 = strchr(q1, '"');
                    if (!q2) break;
                    int slen = (int)(q2 - q1);
                    if (slen > 0 && slen < 256) {
                        char fname[256];
                        memcpy(fname, q1, slen);
                        fname[slen] = '\0';
                        /* Check if already seen */
                        int dup = 0;
                        for (int i = 0; i < num_shards; i++) {
                            if (strcmp(shard_names[i], fname) == 0) { dup = 1; break; }
                        }
                        if (!dup) {
                            strcpy(shard_names[num_shards], fname);
                            num_shards++;
                        }
                    }
                    p = q2 + 1;
                    /* Skip to next key or end */
                    const char *comma = strchr(p, ',');
                    const char *brace = strchr(p, '}');
                    if (brace && (!comma || brace < comma)) break;
                    if (comma) p = comma + 1; else break;
                }
            }
            free(json);

            /* Open each shard - all must succeed */
            for (int i = 0; i < num_shards; i++) {
                snprintf(path, sizeof(path), "%s/transformer/%s", model_dir, shard_names[i]);
                files[num_files] = safetensors_open(path);
                if (files[num_files]) {
                    num_files++;
                } else {
                    fprintf(stderr, "Error: failed to open transformer shard %s\n", shard_names[i]);
                    for (int j = 0; j < num_files; j++) safetensors_close(files[j]);
                    return 0;
                }
            }
            if (num_files > 0) return num_files;
        } else {
            fclose(fp);
        }
    }

    /* Fallback: single file */
    snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors",
             model_dir);
    files[0] = safetensors_open(path);
    if (files[0]) return 1;

    return 0;
}

/* Forward declarations */
void iris_transformer_free_flux(iris_transformer_flux_t *tf);
static int load_double_block_weights(double_block_t *b, safetensors_file_t **files, int num_files, int idx, int h, int mlp, int use_bf16);
static void free_double_block_weights(double_block_t *b);
static int load_single_block_weights(single_block_t *b, safetensors_file_t **files, int num_files, int idx, int h, int mlp, int use_bf16);
static void free_single_block_weights(single_block_t *b);

/* ========================================================================
 * Mmap mode: on-demand weight loading/freeing for blocks
 * ======================================================================== */

/* Helper to get tensor as f32 (used by mmap load functions) */
static float *mmap_get_f32(safetensors_file_t **files, int num_files, const char *name) {
    for (int f = 0; f < num_files; f++) {
        const safetensor_t *t = safetensors_find(files[f], name);
        if (t) return safetensors_get_f32(files[f], t);
    }
    fprintf(stderr, "Error: required tensor %s not found\n", name);
    return NULL;
}

/* Helper to get tensor as bf16 direct pointer (used by mmap load functions)
 * Returns pointer into mmap'd region - caller must NOT free */
static uint16_t *mmap_get_bf16(safetensors_file_t **files, int num_files, const char *name) {
    for (int f = 0; f < num_files; f++) {
        const safetensor_t *t = safetensors_find(files[f], name);
        if (t && safetensor_is_bf16(t)) return safetensors_get_bf16_direct(files[f], t);
    }
    return NULL;
}

/* Load one double block's weights from safetensors files. In mmap mode,
 * returns direct pointers into the memory-mapped file (zero-copy). For the
 * bf16 GPU path, also returns direct bf16 pointers from the safetensors data.
 * The fused FFN weight layout packs gate and up projections into linear_in. */
static int load_double_block_weights(double_block_t *b, safetensors_file_t **files,
                                     int num_files, int idx, int h, int mlp, int use_bf16) {
    char name[256];

    /* Image attention - QK norm weights (always f32) */
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_q.weight", idx);
    b->img_norm_q_weight = mmap_get_f32(files, num_files, name);
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_k.weight", idx);
    b->img_norm_k_weight = mmap_get_f32(files, num_files, name);

    /* Image Q, K, V projections - skip f32 if bf16 available */
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_q.weight", idx);
    if (use_bf16) b->img_q_weight_bf16 = mmap_get_bf16(files, num_files, name);
    if (!use_bf16) b->img_q_weight = mmap_get_f32(files, num_files, name);
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_k.weight", idx);
    if (use_bf16) b->img_k_weight_bf16 = mmap_get_bf16(files, num_files, name);
    if (!use_bf16) b->img_k_weight = mmap_get_f32(files, num_files, name);
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_v.weight", idx);
    if (use_bf16) b->img_v_weight_bf16 = mmap_get_bf16(files, num_files, name);
    if (!use_bf16) b->img_v_weight = mmap_get_f32(files, num_files, name);

    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_out.0.weight", idx);
    if (use_bf16) b->img_proj_weight_bf16 = mmap_get_bf16(files, num_files, name);
    if (!use_bf16) b->img_proj_weight = mmap_get_f32(files, num_files, name);

    /* Image FFN - linear_in contains gate and up fused - skip f32 if bf16 available */
    snprintf(name, sizeof(name), "transformer_blocks.%d.ff.linear_in.weight", idx);
    if (use_bf16) {
        uint16_t *ff_in_bf16 = mmap_get_bf16(files, num_files, name);
        if (ff_in_bf16) {
            /* Direct pointers with offset - no malloc/copy needed */
            b->img_mlp_gate_weight_bf16 = ff_in_bf16;
            b->img_mlp_up_weight_bf16 = ff_in_bf16 + (size_t)mlp * h;
        }
    } else {
        float *ff_in = mmap_get_f32(files, num_files, name);
        if (ff_in) {
            b->img_mlp_gate_weight = malloc(mlp * h * sizeof(float));
            b->img_mlp_up_weight = malloc(mlp * h * sizeof(float));
            memcpy(b->img_mlp_gate_weight, ff_in, mlp * h * sizeof(float));
            memcpy(b->img_mlp_up_weight, ff_in + mlp * h, mlp * h * sizeof(float));
            free(ff_in);
        }
    }

    snprintf(name, sizeof(name), "transformer_blocks.%d.ff.linear_out.weight", idx);
    if (use_bf16) b->img_mlp_down_weight_bf16 = mmap_get_bf16(files, num_files, name);
    if (!use_bf16) b->img_mlp_down_weight = mmap_get_f32(files, num_files, name);

    /* Text stream - QK norm weights (always f32) */
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_added_q.weight", idx);
    b->txt_norm_q_weight = mmap_get_f32(files, num_files, name);
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_added_k.weight", idx);
    b->txt_norm_k_weight = mmap_get_f32(files, num_files, name);

    /* Text Q, K, V projections - skip f32 if bf16 available */
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.add_q_proj.weight", idx);
    if (use_bf16) b->txt_q_weight_bf16 = mmap_get_bf16(files, num_files, name);
    if (!use_bf16) b->txt_q_weight = mmap_get_f32(files, num_files, name);
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.add_k_proj.weight", idx);
    if (use_bf16) b->txt_k_weight_bf16 = mmap_get_bf16(files, num_files, name);
    if (!use_bf16) b->txt_k_weight = mmap_get_f32(files, num_files, name);
    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.add_v_proj.weight", idx);
    if (use_bf16) b->txt_v_weight_bf16 = mmap_get_bf16(files, num_files, name);
    if (!use_bf16) b->txt_v_weight = mmap_get_f32(files, num_files, name);

    snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_add_out.weight", idx);
    if (use_bf16) b->txt_proj_weight_bf16 = mmap_get_bf16(files, num_files, name);
    if (!use_bf16) b->txt_proj_weight = mmap_get_f32(files, num_files, name);

    /* Text FFN - skip f32 if bf16 available */
    snprintf(name, sizeof(name), "transformer_blocks.%d.ff_context.linear_in.weight", idx);
    if (use_bf16) {
        uint16_t *txt_ff_in_bf16 = mmap_get_bf16(files, num_files, name);
        if (txt_ff_in_bf16) {
            /* Direct pointers with offset - no malloc/copy needed */
            b->txt_mlp_gate_weight_bf16 = txt_ff_in_bf16;
            b->txt_mlp_up_weight_bf16 = txt_ff_in_bf16 + (size_t)mlp * h;
        }
    } else {
        float *txt_ff_in = mmap_get_f32(files, num_files, name);
        if (txt_ff_in) {
            b->txt_mlp_gate_weight = malloc(mlp * h * sizeof(float));
            b->txt_mlp_up_weight = malloc(mlp * h * sizeof(float));
            memcpy(b->txt_mlp_gate_weight, txt_ff_in, mlp * h * sizeof(float));
            memcpy(b->txt_mlp_up_weight, txt_ff_in + mlp * h, mlp * h * sizeof(float));
            free(txt_ff_in);
        }
    }

    snprintf(name, sizeof(name), "transformer_blocks.%d.ff_context.linear_out.weight", idx);
    if (use_bf16) b->txt_mlp_down_weight_bf16 = mmap_get_bf16(files, num_files, name);
    if (!use_bf16) b->txt_mlp_down_weight = mmap_get_f32(files, num_files, name);

    return 0;
}

/* Free weights for a single double_block (mmap mode only)
 * Note: bf16 pointers are direct mmap pointers, don't free them */
static void free_double_block_weights(double_block_t *b) {
#ifdef USE_METAL
    /* Invalidate GPU weight cache before freeing CPU pointers.
     * malloc can reuse freed addresses, causing stale cache hits. */
    iris_metal_clear_weight_cache_only();
#endif
    free(b->img_norm_q_weight); b->img_norm_q_weight = NULL;
    free(b->img_norm_k_weight); b->img_norm_k_weight = NULL;
    free(b->img_q_weight); b->img_q_weight = NULL;
    free(b->img_k_weight); b->img_k_weight = NULL;
    free(b->img_v_weight); b->img_v_weight = NULL;
    free(b->img_proj_weight); b->img_proj_weight = NULL;
    free(b->img_mlp_gate_weight); b->img_mlp_gate_weight = NULL;
    free(b->img_mlp_up_weight); b->img_mlp_up_weight = NULL;
    free(b->img_mlp_down_weight); b->img_mlp_down_weight = NULL;
    /* bf16 pointers are direct mmap pointers - just clear, don't free */
    b->img_q_weight_bf16 = NULL;
    b->img_k_weight_bf16 = NULL;
    b->img_v_weight_bf16 = NULL;
    b->img_proj_weight_bf16 = NULL;
    b->img_mlp_gate_weight_bf16 = NULL;
    b->img_mlp_up_weight_bf16 = NULL;
    b->img_mlp_down_weight_bf16 = NULL;
    free(b->txt_norm_q_weight); b->txt_norm_q_weight = NULL;
    free(b->txt_norm_k_weight); b->txt_norm_k_weight = NULL;
    free(b->txt_q_weight); b->txt_q_weight = NULL;
    free(b->txt_k_weight); b->txt_k_weight = NULL;
    free(b->txt_v_weight); b->txt_v_weight = NULL;
    free(b->txt_proj_weight); b->txt_proj_weight = NULL;
    free(b->txt_mlp_gate_weight); b->txt_mlp_gate_weight = NULL;
    free(b->txt_mlp_up_weight); b->txt_mlp_up_weight = NULL;
    free(b->txt_mlp_down_weight); b->txt_mlp_down_weight = NULL;
    /* bf16 pointers are direct mmap pointers - just clear, don't free */
    b->txt_q_weight_bf16 = NULL;
    b->txt_k_weight_bf16 = NULL;
    b->txt_v_weight_bf16 = NULL;
    b->txt_proj_weight_bf16 = NULL;
    b->txt_mlp_gate_weight_bf16 = NULL;
    b->txt_mlp_up_weight_bf16 = NULL;
    b->txt_mlp_down_weight_bf16 = NULL;
}

/* Load one single block's weights from safetensors files, same zero-copy mmap
 * strategy as double blocks. The fused qkv_mlp projection packs [Q, K, V,
 * gate, up] into one weight matrix, enabling a single large matmul instead
 * of five separate ones during inference. */
static int load_single_block_weights(single_block_t *b, safetensors_file_t **files,
                                     int num_files, int idx, int h, int mlp, int use_bf16) {
    char name[256];
    (void)h; (void)mlp;  /* Unused in single block */

    /* QK norm weights (always f32, small) */
    snprintf(name, sizeof(name), "single_transformer_blocks.%d.attn.norm_q.weight", idx);
    b->norm_q_weight = mmap_get_f32(files, num_files, name);
    snprintf(name, sizeof(name), "single_transformer_blocks.%d.attn.norm_k.weight", idx);
    b->norm_k_weight = mmap_get_f32(files, num_files, name);

    /* Fused QKV+MLP input projection - skip f32 if bf16 available */
    snprintf(name, sizeof(name), "single_transformer_blocks.%d.attn.to_qkv_mlp_proj.weight", idx);
    if (use_bf16) b->qkv_mlp_weight_bf16 = mmap_get_bf16(files, num_files, name);
    if (!use_bf16) b->qkv_mlp_weight = mmap_get_f32(files, num_files, name);

    /* Fused attn out + MLP down projection - skip f32 if bf16 available */
    snprintf(name, sizeof(name), "single_transformer_blocks.%d.attn.to_out.weight", idx);
    if (use_bf16) b->proj_mlp_weight_bf16 = mmap_get_bf16(files, num_files, name);
    if (!use_bf16) b->proj_mlp_weight = mmap_get_f32(files, num_files, name);

    return 0;
}

/* Free weights for a single single_block (mmap mode only)
 * Note: bf16 pointers are direct mmap pointers, don't free them */
static void free_single_block_weights(single_block_t *b) {
#ifdef USE_METAL
    iris_metal_clear_weight_cache_only();
#endif
    free(b->norm_q_weight); b->norm_q_weight = NULL;
    free(b->norm_k_weight); b->norm_k_weight = NULL;
    free(b->qkv_mlp_weight); b->qkv_mlp_weight = NULL;
    free(b->proj_mlp_weight); b->proj_mlp_weight = NULL;
    /* bf16 pointers are direct mmap pointers - just clear, don't free */
    b->qkv_mlp_weight_bf16 = NULL;
    b->proj_mlp_weight_bf16 = NULL;
}

/* Free cached mmap weights for all blocks.
 * Called after denoising completes to release memory held across steps. */
void iris_transformer_free_mmap_cache_flux(iris_transformer_flux_t *tf) {
    if (!tf || !tf->use_mmap) return;
    for (int i = 0; i < tf->num_double_layers; i++)
        free_double_block_weights(&tf->double_blocks[i]);
    for (int i = 0; i < tf->num_single_layers; i++)
        free_single_block_weights(&tf->single_blocks[i]);
}

#ifdef USE_METAL
/* Pre-warm bf16 weight buffer cache for all blocks (mmap mode).
 * Loads each block's bf16 mmap pointers and copies weight data to Metal
 * GPU buffers so the first denoising step doesn't pay the cache-miss cost. */
static void warmup_mmap_bf16_buffers(iris_transformer_flux_t *tf) {
    if (!tf->use_mmap || !tf->use_bf16) return;

    int h = tf->hidden_size, mlp = tf->mlp_hidden;
    int fused = h * 3 + mlp * 2;

    /* Input/output projections (already loaded) */
    if (tf->img_in_weight_bf16)
        iris_metal_warmup_bf16_buffer(tf->img_in_weight_bf16,
                                       (size_t)tf->latent_channels * h);
    if (tf->txt_in_weight_bf16)
        iris_metal_warmup_bf16_buffer(tf->txt_in_weight_bf16,
                                       (size_t)tf->text_dim * h);
    if (tf->final_proj_weight_bf16)
        iris_metal_warmup_bf16_buffer(tf->final_proj_weight_bf16,
                                       (size_t)tf->latent_channels * h);

    /* Double blocks */
    for (int i = 0; i < tf->num_double_layers; i++) {
        load_double_block_weights(&tf->double_blocks[i], tf->sf_files, tf->num_sf_files, i, h, mlp, 1);
        double_block_t *b = &tf->double_blocks[i];

        if (b->img_q_weight_bf16)
            iris_metal_warmup_bf16_buffer(b->img_q_weight_bf16, (size_t)h*h);
        if (b->img_k_weight_bf16)
            iris_metal_warmup_bf16_buffer(b->img_k_weight_bf16, (size_t)h*h);
        if (b->img_v_weight_bf16)
            iris_metal_warmup_bf16_buffer(b->img_v_weight_bf16, (size_t)h*h);
        if (b->img_proj_weight_bf16)
            iris_metal_warmup_bf16_buffer(b->img_proj_weight_bf16, (size_t)h*h);
        if (b->img_mlp_gate_weight_bf16)
            iris_metal_warmup_bf16_buffer(b->img_mlp_gate_weight_bf16, (size_t)mlp*h);
        if (b->img_mlp_up_weight_bf16)
            iris_metal_warmup_bf16_buffer(b->img_mlp_up_weight_bf16, (size_t)mlp*h);
        if (b->img_mlp_down_weight_bf16)
            iris_metal_warmup_bf16_buffer(b->img_mlp_down_weight_bf16, (size_t)h*mlp);

        if (b->txt_q_weight_bf16)
            iris_metal_warmup_bf16_buffer(b->txt_q_weight_bf16, (size_t)h*h);
        if (b->txt_k_weight_bf16)
            iris_metal_warmup_bf16_buffer(b->txt_k_weight_bf16, (size_t)h*h);
        if (b->txt_v_weight_bf16)
            iris_metal_warmup_bf16_buffer(b->txt_v_weight_bf16, (size_t)h*h);
        if (b->txt_proj_weight_bf16)
            iris_metal_warmup_bf16_buffer(b->txt_proj_weight_bf16, (size_t)h*h);
        if (b->txt_mlp_gate_weight_bf16)
            iris_metal_warmup_bf16_buffer(b->txt_mlp_gate_weight_bf16, (size_t)mlp*h);
        if (b->txt_mlp_up_weight_bf16)
            iris_metal_warmup_bf16_buffer(b->txt_mlp_up_weight_bf16, (size_t)mlp*h);
        if (b->txt_mlp_down_weight_bf16)
            iris_metal_warmup_bf16_buffer(b->txt_mlp_down_weight_bf16, (size_t)h*mlp);

        free_double_block_weights(&tf->double_blocks[i]);
    }

    /* Single blocks */
    for (int i = 0; i < tf->num_single_layers; i++) {
        load_single_block_weights(&tf->single_blocks[i], tf->sf_files, tf->num_sf_files, i, h, mlp, 1);
        single_block_t *b = &tf->single_blocks[i];

        if (b->qkv_mlp_weight_bf16)
            iris_metal_warmup_bf16_buffer(b->qkv_mlp_weight_bf16, (size_t)fused*h);
        if (b->proj_mlp_weight_bf16)
            iris_metal_warmup_bf16_buffer(b->proj_mlp_weight_bf16, (size_t)h*(h+mlp));

        free_single_block_weights(&tf->single_blocks[i]);
    }
}
#endif

/* ========================================================================
 * RoPE (Rotary Position Embeddings)
 * ======================================================================== */

/* Precompute RoPE frequencies for given positions (1D version) */
static void compute_rope_freqs(float *freqs, int max_seq, int dim, float theta) {
    int half_dim = dim / 2;

    for (int pos = 0; pos < max_seq; pos++) {
        for (int d = 0; d < half_dim; d++) {
            float freq = 1.0f / powf(theta, (float)(2 * d) / (float)dim);
            float angle = (float)pos * freq;
            freqs[pos * half_dim * 2 + d * 2] = cosf(angle);
            freqs[pos * half_dim * 2 + d * 2 + 1] = sinf(angle);
        }
    }
}

/* Build 4-axis RoPE frequency tables for image tokens on a (patch_h x patch_w) grid.
 * Axes are T (temporal), H (height), W (width), L (sequence), each with axes_dim/2
 * frequency pairs. Image tokens use H and W for spatial position encoding so the
 * transformer knows where each image patch is; T and L are set to identity (cos=1,
 * sin=0). The 2D grid positions let joint attention distinguish spatial layout,
 * which is critical for coherent image generation. */
static void compute_rope_2d(float *cos_out, float *sin_out,
                            int patch_h, int patch_w, int axis_dim, float theta) {
    int half_axis = axis_dim / 2;  /* 16 dims per half-axis */
    int seq = patch_h * patch_w;
    (void)seq;  /* Unused but kept for documentation */

    /* Precompute base frequencies on stack (axis_dim is always 32, half_axis=16) */
    float base_freqs[16];
    for (int d = 0; d < half_axis; d++) {
        base_freqs[d] = 1.0f / powf(theta, (float)(2 * d) / (float)axis_dim);
    }

    for (int hy = 0; hy < patch_h; hy++) {
        for (int wx = 0; wx < patch_w; wx++) {
            int pos = hy * patch_w + wx;
            float *cos_p = cos_out + pos * axis_dim * 4;  /* 4 axes * 32 dims each = 128 */
            float *sin_p = sin_out + pos * axis_dim * 4;

            /* Axis 0 (dims 0-31): T position = 0, so cos=1, sin=0 */
            for (int d = 0; d < axis_dim; d++) {
                cos_p[d] = 1.0f;
                sin_p[d] = 0.0f;
            }

            /* Axis 1 (dims 32-63): H position (y/height)
             * Python RoPE stacks [cos, -sin, sin, cos] as 2x2 matrix per freq.
             * For apply_rope: out = [[cos, -sin], [sin, cos]] @ [x0, x1]
             * We store cos/sin per pair and apply_rope_2d handles the rotation.
             */
            for (int d = 0; d < half_axis; d++) {
                float angle_h = (float)hy * base_freqs[d];
                float cos_h = cosf(angle_h);
                float sin_h = sinf(angle_h);
                /* Each frequency contributes to a pair of dimensions */
                cos_p[axis_dim + d * 2] = cos_h;
                cos_p[axis_dim + d * 2 + 1] = cos_h;
                sin_p[axis_dim + d * 2] = sin_h;
                sin_p[axis_dim + d * 2 + 1] = sin_h;
            }

            /* Axis 2 (dims 64-95): W position (x/width) */
            for (int d = 0; d < half_axis; d++) {
                float angle_w = (float)wx * base_freqs[d];
                float cos_w = cosf(angle_w);
                float sin_w = sinf(angle_w);
                cos_p[axis_dim * 2 + d * 2] = cos_w;
                cos_p[axis_dim * 2 + d * 2 + 1] = cos_w;
                sin_p[axis_dim * 2 + d * 2] = sin_w;
                sin_p[axis_dim * 2 + d * 2 + 1] = sin_w;
            }

            /* Axis 3 (dims 96-127): L position = 0, so cos=1, sin=0 */
            for (int d = 0; d < axis_dim; d++) {
                cos_p[axis_dim * 3 + d] = 1.0f;
                sin_p[axis_dim * 3 + d] = 0.0f;
            }
        }
    }
}

/* Compute 2D RoPE frequencies for reference image tokens with T offset
 * Same as compute_rope_2d but with non-zero T coordinate for axis 0.
 * t_offset is typically 10, 20, 30... for multiple reference images.
 */
static void compute_rope_2d_with_t_offset(float *cos_out, float *sin_out,
                                           int patch_h, int patch_w, int axis_dim,
                                           float theta, int t_offset) {
    int half_axis = axis_dim / 2;

    /* Precompute base frequencies */
    float base_freqs[16];
    for (int d = 0; d < half_axis; d++) {
        base_freqs[d] = 1.0f / powf(theta, (float)(2 * d) / (float)axis_dim);
    }

    for (int hy = 0; hy < patch_h; hy++) {
        for (int wx = 0; wx < patch_w; wx++) {
            int pos = hy * patch_w + wx;
            float *cos_p = cos_out + pos * axis_dim * 4;
            float *sin_p = sin_out + pos * axis_dim * 4;

            /* Axis 0 (dims 0-31): T position = t_offset (non-zero for refs) */
            for (int d = 0; d < half_axis; d++) {
                float angle_t = (float)t_offset * base_freqs[d];
                float cos_t = cosf(angle_t);
                float sin_t = sinf(angle_t);
                cos_p[d * 2] = cos_t;
                cos_p[d * 2 + 1] = cos_t;
                sin_p[d * 2] = sin_t;
                sin_p[d * 2 + 1] = sin_t;
            }

            /* Axis 1 (dims 32-63): H position */
            for (int d = 0; d < half_axis; d++) {
                float angle_h = (float)hy * base_freqs[d];
                float cos_h = cosf(angle_h);
                float sin_h = sinf(angle_h);
                cos_p[axis_dim + d * 2] = cos_h;
                cos_p[axis_dim + d * 2 + 1] = cos_h;
                sin_p[axis_dim + d * 2] = sin_h;
                sin_p[axis_dim + d * 2 + 1] = sin_h;
            }

            /* Axis 2 (dims 64-95): W position */
            for (int d = 0; d < half_axis; d++) {
                float angle_w = (float)wx * base_freqs[d];
                float cos_w = cosf(angle_w);
                float sin_w = sinf(angle_w);
                cos_p[axis_dim * 2 + d * 2] = cos_w;
                cos_p[axis_dim * 2 + d * 2 + 1] = cos_w;
                sin_p[axis_dim * 2 + d * 2] = sin_w;
                sin_p[axis_dim * 2 + d * 2 + 1] = sin_w;
            }

            /* Axis 3 (dims 96-127): L position = 0 */
            for (int d = 0; d < axis_dim; d++) {
                cos_p[axis_dim * 3 + d] = 1.0f;
                sin_p[axis_dim * 3 + d] = 0.0f;
            }
        }
    }
}

/* Apply precomputed RoPE rotation to Q or K tensors in-place.
 * Uses consecutive-pair rotation: for each pair (x0, x1), computes
 * (x0*cos - x1*sin, x1*cos + x0*sin). This is equivalent to complex
 * multiplication (x0+i*x1) * (cos+i*sin), encoding positional information
 * directly into the attention logits so that attention scores are
 * position-aware without explicit position tokens. */
static void apply_rope_2d(float *x, const float *cos_freq, const float *sin_freq,
                          int seq, int heads, int head_dim, int axis_dim) {
    (void)axis_dim;  /* head_dim = 128 = 4 * axis_dim (axis_dim = 32) */
    for (int s = 0; s < seq; s++) {
        const float *cos_s = cos_freq + s * head_dim;  /* [128] */
        const float *sin_s = sin_freq + s * head_dim;

        for (int h = 0; h < heads; h++) {
            float *vec = x + (s * heads + h) * head_dim;

            /* Apply rotation to all 128 dims in pairs (0,1), (2,3), ... (126,127) */
            for (int d = 0; d < head_dim; d += 2) {
                float cos_val = cos_s[d];  /* cos[d] == cos[d+1] due to repeat_interleave */
                float sin_val = sin_s[d];
                float x0 = vec[d];
                float x1 = vec[d + 1];
                /* Complex rotation: (x0 + i*x1) * (cos + i*sin) */
                vec[d] = x0 * cos_val - x1 * sin_val;
                vec[d + 1] = x1 * cos_val + x0 * sin_val;
            }
        }
    }
}

/* Build RoPE for text tokens. Only axis 3 (L = sequence position) is active;
 * axes 0-2 (T, H, W) are identity. Text tokens encode sequential order but
 * have no spatial structure, matching their 1D nature. In joint attention,
 * image and text tokens share the same head_dim but use orthogonal RoPE axes,
 * so their positional encodings don't interfere with each other. */
static void compute_rope_text(float *cos_out, float *sin_out,
                              int txt_seq, int axis_dim, float theta) {
    int half_axis = axis_dim / 2;  /* 16 */
    int head_dim = axis_dim * 4;   /* 128 */

    /* Precompute base frequencies on stack (axis_dim is always 32, half_axis=16) */
    float base_freqs[16];
    for (int d = 0; d < half_axis; d++) {
        base_freqs[d] = 1.0f / powf(theta, (float)(2 * d) / (float)axis_dim);
    }

    for (int s = 0; s < txt_seq; s++) {
        float *cos_p = cos_out + s * head_dim;  /* 128 dims */
        float *sin_p = sin_out + s * head_dim;

        /* Axes 0, 1, 2 (dims 0-95): T=H=W=0, so identity */
        for (int d = 0; d < axis_dim * 3; d++) {
            cos_p[d] = 1.0f;
            sin_p[d] = 0.0f;
        }

        /* Axis 3 (dims 96-127): L position = s (sequence index) */
        for (int d = 0; d < half_axis; d++) {
            float angle = (float)s * base_freqs[d];
            float cos_l = cosf(angle);
            float sin_l = sinf(angle);
            cos_p[axis_dim * 3 + d * 2] = cos_l;
            cos_p[axis_dim * 3 + d * 2 + 1] = cos_l;
            sin_p[axis_dim * 3 + d * 2] = sin_l;
            sin_p[axis_dim * 3 + d * 2 + 1] = sin_l;
        }
    }
}

/* ========================================================================
 * Cached RoPE Helpers
 * Cache RoPE embeddings between steps to avoid malloc/compute overhead.
 * ======================================================================== */

/* Get or compute cached image RoPE for target (T=0) */
static void get_cached_img_rope(iris_transformer_flux_t *tf, int patch_h, int patch_w,
                                float **cos_out, float **sin_out) {
    int seq = patch_h * patch_w;
    int axis_dim = tf->axis_dim;
    size_t size = (size_t)seq * axis_dim * 4 * sizeof(float);

    if (tf->cached_img_h == patch_h && tf->cached_img_w == patch_w &&
        tf->cached_img_rope_cos && tf->cached_img_rope_sin) {
        /* Cache hit */
        *cos_out = tf->cached_img_rope_cos;
        *sin_out = tf->cached_img_rope_sin;
        return;
    }

    /* Cache miss - reallocate if needed */
    if (tf->cached_img_rope_cos) free(tf->cached_img_rope_cos);
    if (tf->cached_img_rope_sin) free(tf->cached_img_rope_sin);
    tf->cached_img_rope_cos = (float *)malloc(size);
    tf->cached_img_rope_sin = (float *)malloc(size);
    tf->cached_img_h = patch_h;
    tf->cached_img_w = patch_w;

    compute_rope_2d(tf->cached_img_rope_cos, tf->cached_img_rope_sin,
                    patch_h, patch_w, axis_dim, tf->rope_theta);

    *cos_out = tf->cached_img_rope_cos;
    *sin_out = tf->cached_img_rope_sin;
}

/* Get or compute cached reference image RoPE (with T offset for img2img) */
static void get_cached_ref_rope(iris_transformer_flux_t *tf, int patch_h, int patch_w,
                                int t_offset, float **cos_out, float **sin_out) {
    int seq = patch_h * patch_w;
    int axis_dim = tf->axis_dim;
    size_t size = (size_t)seq * axis_dim * 4 * sizeof(float);

    if (tf->cached_ref_h == patch_h && tf->cached_ref_w == patch_w &&
        tf->cached_ref_t_offset == t_offset &&
        tf->cached_ref_rope_cos && tf->cached_ref_rope_sin) {
        /* Cache hit */
        *cos_out = tf->cached_ref_rope_cos;
        *sin_out = tf->cached_ref_rope_sin;
        return;
    }

    /* Cache miss - reallocate if needed */
    if (tf->cached_ref_rope_cos) free(tf->cached_ref_rope_cos);
    if (tf->cached_ref_rope_sin) free(tf->cached_ref_rope_sin);
    tf->cached_ref_rope_cos = (float *)malloc(size);
    tf->cached_ref_rope_sin = (float *)malloc(size);
    tf->cached_ref_h = patch_h;
    tf->cached_ref_w = patch_w;
    tf->cached_ref_t_offset = t_offset;

    compute_rope_2d_with_t_offset(tf->cached_ref_rope_cos, tf->cached_ref_rope_sin,
                                   patch_h, patch_w, axis_dim, tf->rope_theta, t_offset);

    *cos_out = tf->cached_ref_rope_cos;
    *sin_out = tf->cached_ref_rope_sin;
}

/* Get or compute cached text RoPE */
static void get_cached_txt_rope(iris_transformer_flux_t *tf, int txt_seq,
                                float **cos_out, float **sin_out) {
    int head_dim = tf->head_dim;
    size_t size = (size_t)txt_seq * head_dim * sizeof(float);

    if (tf->cached_txt_seq == txt_seq &&
        tf->cached_txt_rope_cos && tf->cached_txt_rope_sin) {
        /* Cache hit */
        *cos_out = tf->cached_txt_rope_cos;
        *sin_out = tf->cached_txt_rope_sin;
        return;
    }

    /* Cache miss - reallocate if needed */
    if (tf->cached_txt_rope_cos) free(tf->cached_txt_rope_cos);
    if (tf->cached_txt_rope_sin) free(tf->cached_txt_rope_sin);
    tf->cached_txt_rope_cos = (float *)malloc(size);
    tf->cached_txt_rope_sin = (float *)malloc(size);
    tf->cached_txt_seq = txt_seq;

    compute_rope_text(tf->cached_txt_rope_cos, tf->cached_txt_rope_sin,
                      txt_seq, tf->axis_dim, tf->rope_theta);

    *cos_out = tf->cached_txt_rope_cos;
    *sin_out = tf->cached_txt_rope_sin;
}

/* Get or compute cached combined RoPE for img2img (target + reference) */
static void get_cached_combined_rope(iris_transformer_flux_t *tf,
                                     int img_h, int img_w,
                                     int ref_h, int ref_w,
                                     int t_offset,
                                     float **cos_out, float **sin_out) {
    int img_seq = img_h * img_w;
    int ref_seq = ref_h * ref_w;
    int combined_seq = img_seq + ref_seq;
    int axis_dim = tf->axis_dim;
    size_t combined_size = (size_t)combined_seq * axis_dim * 4 * sizeof(float);
    size_t img_size = (size_t)img_seq * axis_dim * 4 * sizeof(float);
    size_t ref_size = (size_t)ref_seq * axis_dim * 4 * sizeof(float);

    if (tf->cached_combined_img_h == img_h &&
        tf->cached_combined_img_w == img_w &&
        tf->cached_combined_ref_h == ref_h &&
        tf->cached_combined_ref_w == ref_w &&
        tf->cached_combined_t_offset == t_offset &&
        tf->cached_combined_rope_cos && tf->cached_combined_rope_sin) {
        /* Cache hit */
        *cos_out = tf->cached_combined_rope_cos;
        *sin_out = tf->cached_combined_rope_sin;
        return;
    }

    /* Cache miss - reallocate if needed */
    if (tf->cached_combined_rope_cos) free(tf->cached_combined_rope_cos);
    if (tf->cached_combined_rope_sin) free(tf->cached_combined_rope_sin);
    tf->cached_combined_rope_cos = (float *)malloc(combined_size);
    tf->cached_combined_rope_sin = (float *)malloc(combined_size);
    tf->cached_combined_img_h = img_h;
    tf->cached_combined_img_w = img_w;
    tf->cached_combined_ref_h = ref_h;
    tf->cached_combined_ref_w = ref_w;
    tf->cached_combined_t_offset = t_offset;

    /* Compute target image RoPE (T=0) */
    float *img_rope_cos, *img_rope_sin;
    get_cached_img_rope(tf, img_h, img_w, &img_rope_cos, &img_rope_sin);

    /* Compute reference image RoPE (T=t_offset) */
    float *ref_rope_cos, *ref_rope_sin;
    get_cached_ref_rope(tf, ref_h, ref_w, t_offset, &ref_rope_cos, &ref_rope_sin);

    /* Concatenate: [target, reference] */
    memcpy(tf->cached_combined_rope_cos, img_rope_cos, img_size);
    memcpy(tf->cached_combined_rope_cos + img_seq * axis_dim * 4, ref_rope_cos, ref_size);
    memcpy(tf->cached_combined_rope_sin, img_rope_sin, img_size);
    memcpy(tf->cached_combined_rope_sin + img_seq * axis_dim * 4, ref_rope_sin, ref_size);

    *cos_out = tf->cached_combined_rope_cos;
    *sin_out = tf->cached_combined_rope_sin;
}

/* ========================================================================
 * Timestep Embedding
 * ======================================================================== */

/* Convert scalar timestep t to a 256-dim sinusoidal embedding using log-spaced
 * frequencies (like positional encoding in the original Transformer paper).
 * Output layout is [cos(all freqs), sin(all freqs)] (flip_sin_to_cos=True).
 * The diffusion model conditions on this to know what noise level it's
 * denoising at the current step. */
static void get_timestep_embedding(float *out, float t, int dim, float max_period) {
    int half = dim / 2;
    float log_max = logf(max_period);

    for (int i = 0; i < half; i++) {
        /* freq = exp(-log(max_period) * i / half_dim) */
        float freq = expf(-log_max * (float)i / (float)half);
        float angle = t * freq;
        out[i] = cosf(angle);           /* cos part first (flip_sin_to_cos=True) */
        out[i + half] = sinf(angle);    /* sin part second */
    }
}

/* Project the 256-dim sinusoidal embedding through an MLP:
 * Linear(256->hidden) + SiLU + Linear(hidden->hidden). The output drives
 * AdaLN modulation in every transformer block -- this is how the model knows
 * which denoising step it's on and adjusts its behavior accordingly. */
static void time_embed_forward(float *out, const float *t_sincos,
                               const time_embed_t *te, int hidden, float *work) {
    /* MLP: fc1 (256->hidden) -> SiLU -> fc2 (hidden->hidden) */
    int sincos_dim = te->sincos_dim;

    /* fc1: [sincos_dim] -> [hidden] */
    iris_linear_nobias(work, t_sincos, te->fc1_weight, 1, sincos_dim, hidden);

    /* SiLU */
    iris_silu(work, hidden);

    /* fc2: [hidden] -> [hidden] */
    iris_linear_nobias(out, work, te->fc2_weight, 1, hidden, hidden);
}

/* ========================================================================
 * AdaLN-Zero Modulation
 * ======================================================================== */

/* Adaptive Layer Normalization: out = (1 + scale) * LayerNorm(x) + shift.
 * The shift and scale come from the timestep embedding, letting each denoising
 * step modulate the normalization differently. This is the key mechanism for
 * timestep conditioning in the transformer -- without it, every step would
 * produce the same output. Scale is centered at 0 (so (1+scale) starts near 1).
 * Uses LayerNorm (not RMSNorm) with elementwise_affine=False. */
static void apply_adaln(float *out, const float *x,
                        const float *shift, const float *scale,
                        int seq, int hidden, float eps) {
    /* Layer Norm (subtract mean, divide by std) + AdaLN modulation
     * Note: Flux2 uses LayerNorm with elementwise_affine=False (no learned weights)
     * Vectorized using Accelerate framework on Apple platforms.
     */
#if defined(__APPLE__) && defined(USE_BLAS)
    /* Vectorized implementation using vDSP */
    for (int s = 0; s < seq; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        /* Compute mean using vDSP_meanv */
        float mean;
        vDSP_meanv(x_row, 1, &mean, hidden);

        /* Compute variance: sum((x - mean)^2) / n */
        /* First subtract mean: temp = x - mean */
        vDSP_vsadd(x_row, 1, &(float){-mean}, out_row, 1, hidden);

        /* Then square: temp = temp^2 */
        vDSP_vsq(out_row, 1, out_row, 1, hidden);

        /* Sum the squares */
        float var_sum;
        vDSP_sve(out_row, 1, &var_sum, hidden);
        float var = var_sum / hidden;
        float std_inv = 1.0f / sqrtf(var + eps);

        /* Apply normalization: out = (x - mean) * std_inv */
        vDSP_vsadd(x_row, 1, &(float){-mean}, out_row, 1, hidden);
        vDSP_vsmul(out_row, 1, &std_inv, out_row, 1, hidden);

        /* Apply modulation: out = (1 + scale) * out + shift
         * Could use vDSP_vma but need temp buffer; scalar loop is fast enough */
        for (int i = 0; i < hidden; i++) {
            out_row[i] = (1.0f + scale[i]) * out_row[i] + shift[i];
        }
    }
#else
    /* Scalar fallback */
    for (int s = 0; s < seq; s++) {
        const float *x_row = x + s * hidden;
        float *out_row = out + s * hidden;

        /* Compute mean */
        float sum = 0.0f;
        for (int i = 0; i < hidden; i++) {
            sum += x_row[i];
        }
        float mean = sum / hidden;

        /* Compute variance */
        float var_sum = 0.0f;
        for (int i = 0; i < hidden; i++) {
            float diff = x_row[i] - mean;
            var_sum += diff * diff;
        }
        float var = var_sum / hidden;
        float std_inv = 1.0f / sqrtf(var + eps);

        /* Apply Layer Norm + AdaLN modulation */
        for (int i = 0; i < hidden; i++) {
            float norm = (x_row[i] - mean) * std_inv;
            out_row[i] = (1.0f + scale[i]) * norm + shift[i];
        }
    }
#endif
}

/* Apply QK normalization (RMSNorm per head)
 * Vectorized using Accelerate framework on Apple platforms. */
static void apply_qk_norm(float *q, float *k,
                          const float *q_weight, const float *k_weight,
                          int seq, int heads, int head_dim, float eps) {
#if defined(__APPLE__) && defined(USE_BLAS)
    /* Vectorized implementation using vDSP */
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < heads; h++) {
            /* Q normalization: x = x * rsqrt(mean(x^2) + eps) * weight */
            float *qh = q + s * heads * head_dim + h * head_dim;
            float sum_sq;
            vDSP_svesq(qh, 1, &sum_sq, head_dim);
            float rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);

            /* Apply: qh = qh * rms_inv * q_weight */
            vDSP_vsmul(qh, 1, &rms_inv, qh, 1, head_dim);
            vDSP_vmul(qh, 1, q_weight, 1, qh, 1, head_dim);

            /* K normalization */
            float *kh = k + s * heads * head_dim + h * head_dim;
            vDSP_svesq(kh, 1, &sum_sq, head_dim);
            rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);

            vDSP_vsmul(kh, 1, &rms_inv, kh, 1, head_dim);
            vDSP_vmul(kh, 1, k_weight, 1, kh, 1, head_dim);
        }
    }
#else
    /* Scalar fallback */
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < heads; h++) {
            /* Q normalization */
            float *qh = q + s * heads * head_dim + h * head_dim;
            float sum_sq = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum_sq += qh[d] * qh[d];
            }
            float rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);
            for (int d = 0; d < head_dim; d++) {
                qh[d] = qh[d] * rms_inv * q_weight[d];
            }

            /* K normalization */
            float *kh = k + s * heads * head_dim + h * head_dim;
            sum_sq = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                sum_sq += kh[d] * kh[d];
            }
            rms_inv = 1.0f / sqrtf(sum_sq / head_dim + eps);
            for (int d = 0; d < head_dim; d++) {
                kh[d] = kh[d] * rms_inv * k_weight[d];
            }
        }
    }
#endif
}

/* ========================================================================
 * Attention Layer
 * ======================================================================== */

/* Multi-head self-attention */

#ifdef USE_METAL
/* Transpose from [seq, heads, head_dim] to [heads, seq, head_dim]
 * Needed for batched attention that processes each head separately */
static void transpose_shd_to_hsd(float *out, const float *in,
                                  int seq, int heads, int head_dim) {
    for (int s = 0; s < seq; s++) {
        for (int h = 0; h < heads; h++) {
            const float *src = in + (s * heads + h) * head_dim;
            float *dst = out + (h * seq + s) * head_dim;
            memcpy(dst, src, head_dim * sizeof(float));
        }
    }
}

/* Transpose from [heads, seq, head_dim] to [seq, heads, head_dim] */
static void transpose_hsd_to_shd(float *out, const float *in,
                                  int seq, int heads, int head_dim) {
    for (int h = 0; h < heads; h++) {
        for (int s = 0; s < seq; s++) {
            const float *src = in + (h * seq + s) * head_dim;
            float *dst = out + (s * heads + h) * head_dim;
            memcpy(dst, src, head_dim * sizeof(float));
        }
    }
}
#endif /* USE_METAL */

/* Ensure attn_scores buffer is large enough for current sequence lengths.
 * Only needed for BLAS/Metal paths - flash attention doesn't use this buffer.
 * Returns 0 on success, -1 on allocation failure.
 */
static int ensure_attn_scores(iris_transformer_flux_t *tf, int img_seq, int txt_seq) {
#if defined(USE_METAL) || defined(USE_BLAS)
    int total_seq = img_seq + txt_seq;
    /* Need space for num_heads * max_seq * max_seq where max_seq = total_seq
     * This covers both self-attention and joint attention needs */
    size_t needed = (size_t)tf->num_heads * total_seq * total_seq * sizeof(float);

    if (tf->attn_scores_alloc < needed) {
        free(tf->attn_scores);
        tf->attn_scores = (float *)malloc(needed);
        if (!tf->attn_scores) {
            tf->attn_scores_alloc = 0;
            return -1;
        }
        tf->attn_scores_alloc = needed;
    }
    return 0;
#else
    (void)tf; (void)img_seq; (void)txt_seq;
    return 0;  /* Flash attention doesn't need attn_scores */
#endif
}

/* References may exceed the manual attention budget. Allocate scores only
 * after the device and native BF16 paths have been tried. */
static int ensure_ref_fallback_scores_flux(iris_transformer_flux_t *tf, int img_seq, int txt_seq) {
#if defined(USE_METAL) || defined(USE_BLAS)
    size_t total_seq = (size_t)img_seq + txt_seq;
    if ((size_t)tf->num_heads * total_seq * total_seq * sizeof(float) >= ((size_t)4 << 30)) {
        fprintf(stderr, "Memory-efficient GPU attention failed; manual attention would exceed 4 GiB\n");
        return -1;
    }
#endif
    return ensure_attn_scores(tf, img_seq, txt_seq);
}

/* Ensure all work buffers are allocated for the given sequence length.
 * Buffers are only reallocated if the current allocation is too small.
 * Returns 0 on success, -1 on allocation failure.
 */
static int ensure_work_buffers(iris_transformer_flux_t *tf, int total_seq) {
    if (tf->work_seq_alloc >= total_seq) {
        return 0;  /* Already have enough space */
    }

    int hidden = tf->hidden_size;
    int mlp = tf->mlp_hidden;

    /* Free old buffers */
    free(tf->img_hidden);
    free(tf->txt_hidden);
    free(tf->work1);
    free(tf->work2);
    free(tf->attn_q_t);
    free(tf->attn_k_t);
    free(tf->attn_v_t);
    free(tf->attn_out_t);
    free(tf->attn_cat_k);
    free(tf->attn_cat_v);
    free(tf->single_q);
    free(tf->single_k);
    free(tf->single_v);
    free(tf->single_mlp_gate);
    free(tf->single_mlp_up);
    free(tf->single_attn_out);
    free(tf->single_concat);
    free(tf->ffn_gate);
    free(tf->ffn_up);
    free(tf->double_img_attn_out);
    free(tf->double_txt_attn_out);

    /* Allocate new buffers */
    tf->img_hidden = (float *)malloc((size_t)total_seq * hidden * sizeof(float));
    tf->txt_hidden = (float *)malloc((size_t)total_seq * hidden * sizeof(float));
    /* work2 needs to hold fused QKV+MLP output: seq * (hidden*3 + mlp*2) + mod_params (hidden*3)
     * fused_dim = hidden*3 + mlp*2 = 3072*3 + 9216*2 = 27648 */
    int fused_dim = hidden * 3 + mlp * 2;
    tf->work_size = (size_t)total_seq * fused_dim * sizeof(float) + (size_t)hidden * 3 * sizeof(float);
    tf->work1 = (float *)malloc((size_t)total_seq * hidden * sizeof(float));
    tf->work2 = (float *)malloc(tf->work_size);
    tf->attn_q_t = (float *)malloc((size_t)total_seq * hidden * sizeof(float));
    tf->attn_k_t = (float *)malloc((size_t)total_seq * hidden * sizeof(float));
    tf->attn_v_t = (float *)malloc((size_t)total_seq * hidden * sizeof(float));
    tf->attn_out_t = (float *)malloc((size_t)total_seq * hidden * sizeof(float));
    tf->attn_cat_k = (float *)malloc((size_t)total_seq * hidden * sizeof(float));
    tf->attn_cat_v = (float *)malloc((size_t)total_seq * hidden * sizeof(float));
    tf->single_q = (float *)malloc((size_t)total_seq * hidden * sizeof(float));
    tf->single_k = (float *)malloc((size_t)total_seq * hidden * sizeof(float));
    tf->single_v = (float *)malloc((size_t)total_seq * hidden * sizeof(float));
    tf->single_mlp_gate = (float *)malloc((size_t)total_seq * mlp * sizeof(float));
    tf->single_mlp_up = (float *)malloc((size_t)total_seq * mlp * sizeof(float));
    tf->single_attn_out = (float *)malloc((size_t)total_seq * hidden * sizeof(float));
    tf->single_concat = (float *)malloc((size_t)total_seq * (hidden + mlp) * sizeof(float));
    tf->ffn_gate = (float *)malloc((size_t)total_seq * mlp * sizeof(float));
    tf->ffn_up = (float *)malloc((size_t)total_seq * mlp * sizeof(float));
    tf->double_img_attn_out = (float *)malloc((size_t)total_seq * hidden * sizeof(float));
    tf->double_txt_attn_out = (float *)malloc((size_t)total_seq * hidden * sizeof(float));

    /* Check all allocations succeeded */
    if (!tf->img_hidden || !tf->txt_hidden || !tf->work1 || !tf->work2 ||
        !tf->attn_q_t || !tf->attn_k_t || !tf->attn_v_t || !tf->attn_out_t ||
        !tf->attn_cat_k || !tf->attn_cat_v ||
        !tf->single_q || !tf->single_k || !tf->single_v ||
        !tf->single_mlp_gate || !tf->single_mlp_up || !tf->single_attn_out ||
        !tf->single_concat || !tf->ffn_gate || !tf->ffn_up ||
        !tf->double_img_attn_out || !tf->double_txt_attn_out) {
        tf->work_seq_alloc = 0;
        return -1;
    }

    tf->work_seq_alloc = total_seq;
    return 0;
}

/* ========================================================================
 * Thread-parallel attention for BLAS path.
 * Per-head sgemm is too small for BLAS internal threading, so we
 * parallelize across heads using pthreads instead.
 * ======================================================================== */

#ifdef USE_BLAS
/* Work descriptor for self-attention (single blocks) */
typedef struct {
    const float *q, *k, *v;
    float *out, *scores;
    int seq, head_dim, hidden;
    float scale;
    int head_start, head_end;
} mha_thread_work_t;

static void *mha_thread_worker(void *arg) {
    mha_thread_work_t *w = (mha_thread_work_t *)arg;
    for (int h = w->head_start; h < w->head_end; h++) {
        const float *qh = w->q + h * w->head_dim;
        const float *kh = w->k + h * w->head_dim;
        const float *vh = w->v + h * w->head_dim;
        float *oh = w->out + h * w->head_dim;
        float *sh = w->scores + (size_t)h * w->seq * w->seq;

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    w->seq, w->seq, w->head_dim,
                    w->scale, qh, w->hidden, kh, w->hidden,
                    0.0f, sh, w->seq);
        iris_softmax_cpu(sh, w->seq, w->seq);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    w->seq, w->head_dim, w->seq,
                    1.0f, sh, w->seq, vh, w->hidden,
                    0.0f, oh, w->hidden);
    }
    return NULL;
}

/* Work descriptor for joint attention (double blocks) */
typedef struct {
    const float *img_q, *txt_q, *cat_k, *cat_v;
    float *img_out, *txt_out, *scores;
    int img_seq, txt_seq, total_seq, head_dim, hidden;
    float scale;
    int head_start, head_end;
} joint_attn_thread_work_t;

static void *joint_attn_thread_worker(void *arg) {
    joint_attn_thread_work_t *w = (joint_attn_thread_work_t *)arg;
    for (int h = w->head_start; h < w->head_end; h++) {
        const float *img_qh = w->img_q + h * w->head_dim;
        const float *txt_qh = w->txt_q + h * w->head_dim;
        const float *kh = w->cat_k + h * w->head_dim;
        const float *vh = w->cat_v + h * w->head_dim;
        float *img_oh = w->img_out + h * w->head_dim;
        float *txt_oh = w->txt_out + h * w->head_dim;
        float *img_sh = w->scores + (size_t)h * w->total_seq * w->total_seq;
        float *txt_sh = img_sh + (size_t)w->img_seq * w->total_seq;

        /* Image attention */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    w->img_seq, w->total_seq, w->head_dim,
                    w->scale, img_qh, w->hidden, kh, w->hidden,
                    0.0f, img_sh, w->total_seq);
        iris_softmax_cpu(img_sh, w->img_seq, w->total_seq);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    w->img_seq, w->head_dim, w->total_seq,
                    1.0f, img_sh, w->total_seq, vh, w->hidden,
                    0.0f, img_oh, w->hidden);

        /* Text attention */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    w->txt_seq, w->total_seq, w->head_dim,
                    w->scale, txt_qh, w->hidden, kh, w->hidden,
                    0.0f, txt_sh, w->total_seq);
        iris_softmax_cpu(txt_sh, w->txt_seq, w->total_seq);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    w->txt_seq, w->head_dim, w->total_seq,
                    1.0f, txt_sh, w->total_seq, vh, w->hidden,
                    0.0f, txt_oh, w->hidden);
    }
    return NULL;
}

/* Get number of threads for head-parallel attention.
 * Uses CPU core count, capped to divide num_heads evenly. */
static int get_attn_num_threads(int heads) {
    static int cached = 0;
    if (cached) return cached;
    int ncpu = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (ncpu < 2) { cached = 1; return 1; }
    if (ncpu > heads) ncpu = heads;
    /* Round down to divide heads evenly */
    while (heads % ncpu != 0) ncpu--;
    cached = ncpu;
    return cached;
}
#endif /* USE_BLAS */

/* Multi-head attention with BLAS optimization
 * Uses pre-allocated workspace buffers from transformer struct
 */
static void mha_forward(float *out, const float *q, const float *k, const float *v,
                        int seq, int heads, int head_dim, iris_transformer_flux_t *tf) {
    float scale = 1.0f / sqrtf((float)head_dim);
    (void)heads; /* hidden = heads * head_dim, but we use tf->hidden_size */

#ifdef USE_METAL
    /* Try fused attention kernel first - operates directly on [seq, hidden] layout
     * This avoids CPU transpose overhead */
    if (iris_metal_attention_fused(out, q, k, v, seq, seq, tf->num_heads, head_dim, scale)) {
        return;  /* Success - no transpose needed */
    }

    /* Try GPU-accelerated batched attention when Metal is available */
    if (iris_metal_available()) {
        /* Use pre-allocated buffers from transformer */
        float *q_t = tf->attn_q_t;
        float *k_t = tf->attn_k_t;
        float *v_t = tf->attn_v_t;
        float *out_t = tf->attn_out_t;
        float *scores = tf->attn_scores;

        /* Transpose to [heads, seq, head_dim] for GPU batched attention */
        transpose_shd_to_hsd(q_t, q, seq, tf->num_heads, head_dim);
        transpose_shd_to_hsd(k_t, k, seq, tf->num_heads, head_dim);
        transpose_shd_to_hsd(v_t, v, seq, tf->num_heads, head_dim);

        iris_metal_attention(out_t, q_t, k_t, v_t, scores,
                             tf->num_heads, seq, seq, head_dim, scale);

        /* Transpose output back to [seq, heads, head_dim] */
        transpose_hsd_to_shd(out, out_t, seq, tf->num_heads, head_dim);
        return;
    }
#endif

    /* CPU fallback: Use BLAS-optimized attention (faster) or flash attention (memory-efficient) */
#ifdef USE_BLAS
    /* BLAS path: thread-parallel per-head attention.
     * Q, K, V are [seq, heads*head_dim] layout. We use lda=hidden to stride
     * over heads, reading head_dim elements per row directly.
     * Per-head sgemm is too small for BLAS internal threading, so we
     * parallelize across heads with pthreads for better core utilization. */
    {
        int hidden = tf->num_heads * head_dim;
        float *scores = tf->attn_scores;
        int nthreads = get_attn_num_threads(tf->num_heads);
        int heads_per_thread = tf->num_heads / nthreads;

        if (nthreads <= 1) {
            /* Serial fallback */
            for (int h = 0; h < tf->num_heads; h++) {
                const float *qh = q + h * head_dim;
                const float *kh = k + h * head_dim;
                const float *vh = v + h * head_dim;
                float *oh = out + h * head_dim;
                float *sh = scores + (size_t)h * seq * seq;

                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            seq, seq, head_dim,
                            scale, qh, hidden, kh, hidden,
                            0.0f, sh, seq);
                iris_softmax(sh, seq, seq);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            seq, head_dim, seq,
                            1.0f, sh, seq, vh, hidden,
                            0.0f, oh, hidden);
            }
        } else {
            pthread_t threads[nthreads];
            mha_thread_work_t work[nthreads];
            int ok[nthreads];
            for (int t = 0; t < nthreads; t++) {
                work[t] = (mha_thread_work_t){
                    .q = q, .k = k, .v = v,
                    .out = out, .scores = scores,
                    .seq = seq, .head_dim = head_dim, .hidden = hidden,
                    .scale = scale,
                    .head_start = t * heads_per_thread,
                    .head_end = (t + 1) * heads_per_thread,
                };
                ok[t] = pthread_create(&threads[t], NULL, mha_thread_worker, &work[t]) == 0;
                if (!ok[t]) mha_thread_worker(&work[t]);
            }
            for (int t = 0; t < nthreads; t++) {
                if (ok[t]) pthread_join(threads[t], NULL);
            }
        }
    }
#else
    /* Generic fallback: Use flash attention (memory-efficient, no transpose needed) */
    iris_flash_attention(out, q, k, v, seq, seq, tf->num_heads, head_dim, scale);
#endif
}

/* Joint attention (for double blocks) - image and text attend to each other
 * Uses pre-allocated workspace buffers from transformer struct
 */
static void joint_attention(float *img_out, float *txt_out,
                            const float *img_q, const float *img_k, const float *img_v,
                            const float *txt_q, const float *txt_k, const float *txt_v,
                            int img_seq, int txt_seq, int heads, int head_dim,
                            iris_transformer_flux_t *tf) {
    int total_seq = img_seq + txt_seq;
    int hidden = heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    /* Use pre-allocated buffers for K/V concatenation */
    float *cat_k = tf->attn_cat_k;
    float *cat_v = tf->attn_cat_v;

    /* Concatenate K, V from both streams in [seq, heads, head_dim] format
     * IMPORTANT: Python (official Flux2) concatenates as [TEXT, IMAGE]
     */
    memcpy(cat_k, txt_k, txt_seq * hidden * sizeof(float));
    memcpy(cat_v, txt_v, txt_seq * hidden * sizeof(float));
    memcpy(cat_k + txt_seq * hidden, img_k, img_seq * hidden * sizeof(float));
    memcpy(cat_v + txt_seq * hidden, img_v, img_seq * hidden * sizeof(float));

#ifdef USE_METAL
    /* Try fused attention kernel first - operates directly on [seq, hidden] layout
     * This avoids CPU transpose overhead */
    if (iris_metal_attention_fused(img_out, img_q, cat_k, cat_v,
                                   img_seq, total_seq, heads, head_dim, scale) &&
        iris_metal_attention_fused(txt_out, txt_q, cat_k, cat_v,
                                   txt_seq, total_seq, heads, head_dim, scale)) {
        return;  /* Success - no transpose needed */
    }

    /* Try GPU-accelerated batched attention when Metal is available */
    if (iris_metal_available()) {
        /* Use pre-allocated buffers for transposed data */
        float *img_q_t = tf->attn_q_t;
        float *txt_q_t = tf->attn_q_t + img_seq * hidden;
        float *cat_k_t = tf->attn_k_t;
        float *cat_v_t = tf->attn_v_t;
        float *img_out_t = tf->attn_out_t;
        float *txt_out_t = tf->attn_out_t + img_seq * hidden;
        float *scores = tf->attn_scores;

        /* Transpose to [heads, seq, head_dim] for GPU batched attention */
        transpose_shd_to_hsd(img_q_t, img_q, img_seq, heads, head_dim);
        transpose_shd_to_hsd(txt_q_t, txt_q, txt_seq, heads, head_dim);
        transpose_shd_to_hsd(cat_k_t, cat_k, total_seq, heads, head_dim);
        transpose_shd_to_hsd(cat_v_t, cat_v, total_seq, heads, head_dim);

        /* Image attention: img_Q @ cat_K^T, softmax, @ cat_V */
        iris_metal_attention(img_out_t, img_q_t, cat_k_t, cat_v_t, scores,
                             heads, img_seq, total_seq, head_dim, scale);
        /* Text attention: txt_Q @ cat_K^T, softmax, @ cat_V */
        iris_metal_attention(txt_out_t, txt_q_t, cat_k_t, cat_v_t, scores,
                             heads, txt_seq, total_seq, head_dim, scale);

        /* Transpose outputs back */
        transpose_hsd_to_shd(img_out, img_out_t, img_seq, heads, head_dim);
        transpose_hsd_to_shd(txt_out, txt_out_t, txt_seq, heads, head_dim);
        return;
    }
#endif

    /* CPU fallback: Use BLAS-optimized attention (faster) or flash attention (memory-efficient) */
#ifdef USE_BLAS
    /* BLAS path: thread-parallel per-head joint attention.
     * All tensors are [seq, heads*head_dim] layout, use lda=hidden for strides.
     * Each head gets its own scores slice for thread safety. */
    {
        float *scores = tf->attn_scores;
        int nthreads = get_attn_num_threads(heads);
        int heads_per_thread = heads / nthreads;

        if (nthreads <= 1) {
            /* Serial fallback */
            for (int h = 0; h < heads; h++) {
                const float *img_qh = img_q + h * head_dim;
                const float *txt_qh = txt_q + h * head_dim;
                const float *kh = cat_k + h * head_dim;
                const float *vh = cat_v + h * head_dim;
                float *img_oh = img_out + h * head_dim;
                float *txt_oh = txt_out + h * head_dim;
                float *img_sh = scores + (size_t)h * total_seq * total_seq;
                float *txt_sh = img_sh + (size_t)img_seq * total_seq;

                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            img_seq, total_seq, head_dim,
                            scale, img_qh, hidden, kh, hidden,
                            0.0f, img_sh, total_seq);
                iris_softmax(img_sh, img_seq, total_seq);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            img_seq, head_dim, total_seq,
                            1.0f, img_sh, total_seq, vh, hidden,
                            0.0f, img_oh, hidden);

                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            txt_seq, total_seq, head_dim,
                            scale, txt_qh, hidden, kh, hidden,
                            0.0f, txt_sh, total_seq);
                iris_softmax(txt_sh, txt_seq, total_seq);
                cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                            txt_seq, head_dim, total_seq,
                            1.0f, txt_sh, total_seq, vh, hidden,
                            0.0f, txt_oh, hidden);
            }
        } else {
            pthread_t threads[nthreads];
            joint_attn_thread_work_t work[nthreads];
            int ok[nthreads];
            for (int t = 0; t < nthreads; t++) {
                work[t] = (joint_attn_thread_work_t){
                    .img_q = img_q, .txt_q = txt_q,
                    .cat_k = cat_k, .cat_v = cat_v,
                    .img_out = img_out, .txt_out = txt_out,
                    .scores = scores,
                    .img_seq = img_seq, .txt_seq = txt_seq,
                    .total_seq = total_seq,
                    .head_dim = head_dim, .hidden = hidden,
                    .scale = scale,
                    .head_start = t * heads_per_thread,
                    .head_end = (t + 1) * heads_per_thread,
                };
                ok[t] = pthread_create(&threads[t], NULL, joint_attn_thread_worker, &work[t]) == 0;
                if (!ok[t]) joint_attn_thread_worker(&work[t]);
            }
            for (int t = 0; t < nthreads; t++) {
                if (ok[t]) pthread_join(threads[t], NULL);
            }
        }
    }
#else
    /* Generic fallback: Use flash attention (memory-efficient, no transpose needed) */
    iris_flash_attention(img_out, img_q, cat_k, cat_v,
                         img_seq, total_seq, heads, head_dim, scale);
    iris_flash_attention(txt_out, txt_q, cat_k, cat_v,
                         txt_seq, total_seq, heads, head_dim, scale);
#endif
}

#ifdef USE_METAL
static iris_gpu_tensor_t bf16_tensor_from_f32(const float *data, int n) {
    iris_gpu_tensor_t f32 = iris_gpu_tensor_create(data, n);
    if (!f32) return NULL;
    iris_gpu_tensor_t bf16 = iris_gpu_tensor_f32_to_bf16(f32);
    if (getenv("IRIS_BF16_SYNC_CONVERT")) {
        iris_gpu_sync();
    }
    iris_gpu_tensor_free(f32);
    return bf16;
}

/* Joint attention on bf16 GPU tensors (img and txt attend to concatenated K/V) */
static int joint_attention_bf16(iris_gpu_tensor_t img_out, iris_gpu_tensor_t txt_out,
                                iris_gpu_tensor_t img_q, iris_gpu_tensor_t img_k, iris_gpu_tensor_t img_v,
                                iris_gpu_tensor_t txt_q, iris_gpu_tensor_t txt_k, iris_gpu_tensor_t txt_v,
                                int img_seq, int txt_seq, int heads, int head_dim) {
    if (!img_out || !txt_out || !img_q || !img_k || !img_v || !txt_q || !txt_k || !txt_v) return 0;
    if (!iris_gpu_tensor_is_f16(img_out) || !iris_gpu_tensor_is_f16(txt_out)) return 0;
    if (!iris_gpu_tensor_is_f16(img_q) || !iris_gpu_tensor_is_f16(img_k) || !iris_gpu_tensor_is_f16(img_v)) return 0;
    if (!iris_gpu_tensor_is_f16(txt_q) || !iris_gpu_tensor_is_f16(txt_k) || !iris_gpu_tensor_is_f16(txt_v)) return 0;

    int total_seq = img_seq + txt_seq;
    int hidden = heads * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim);

    iris_gpu_tensor_t cat_k = iris_gpu_tensor_alloc_f16((size_t)total_seq * hidden);
    iris_gpu_tensor_t cat_v = iris_gpu_tensor_alloc_f16((size_t)total_seq * hidden);
    if (!cat_k || !cat_v) {
        if (cat_k) iris_gpu_tensor_free(cat_k);
        if (cat_v) iris_gpu_tensor_free(cat_v);
        return 0;
    }

    iris_gpu_concat_seq_bf16(cat_k, txt_k, img_k, txt_seq, img_seq, hidden);
    iris_gpu_concat_seq_bf16(cat_v, txt_v, img_v, txt_seq, img_seq, hidden);

    int ok = 1;
    if (!iris_gpu_attention_fused_bf16(img_out, img_q, cat_k, cat_v,
                                       img_seq, total_seq, heads, head_dim, scale)) {
        ok = 0;
    } else if (!iris_gpu_attention_fused_bf16(txt_out, txt_q, cat_k, cat_v,
                                              txt_seq, total_seq, heads, head_dim, scale)) {
        ok = 0;
    }

    iris_gpu_tensor_free(cat_k);
    iris_gpu_tensor_free(cat_v);
    return ok;
}

/* BF16 double-stream block forward (MM-DiT) */
static int double_block_forward_bf16(iris_gpu_tensor_t img_hidden, iris_gpu_tensor_t txt_hidden,
                                     const double_block_t *block,
                                     iris_gpu_tensor_t img_shift1, iris_gpu_tensor_t img_scale1, iris_gpu_tensor_t img_gate1,
                                     iris_gpu_tensor_t img_shift2, iris_gpu_tensor_t img_scale2, iris_gpu_tensor_t img_gate2,
                                     iris_gpu_tensor_t txt_shift1, iris_gpu_tensor_t txt_scale1, iris_gpu_tensor_t txt_gate1,
                                     iris_gpu_tensor_t txt_shift2, iris_gpu_tensor_t txt_scale2, iris_gpu_tensor_t txt_gate2,
                                     const float *img_rope_cos, const float *img_rope_sin,
                                     const float *txt_rope_cos, const float *txt_rope_sin,
                                     int img_seq, int txt_seq, iris_transformer_flux_t *tf) {
    if (!img_hidden || !txt_hidden || !iris_gpu_tensor_is_f16(img_hidden) || !iris_gpu_tensor_is_f16(txt_hidden)) {
        BF16_DEBUG("[BF16] double block: hidden tensors not bf16\n");
        return 0;
    }
    if (!block->img_q_weight_bf16 || !block->img_k_weight_bf16 || !block->img_v_weight_bf16 ||
        !block->txt_q_weight_bf16 || !block->txt_k_weight_bf16 || !block->txt_v_weight_bf16 ||
        !block->img_proj_weight_bf16 || !block->txt_proj_weight_bf16 ||
        !block->img_mlp_gate_weight_bf16 || !block->img_mlp_up_weight_bf16 || !block->img_mlp_down_weight_bf16 ||
        !block->txt_mlp_gate_weight_bf16 || !block->txt_mlp_up_weight_bf16 || !block->txt_mlp_down_weight_bf16) {
        static int missing_logged = 0;
        if (!missing_logged) {
            missing_logged = 1;
            BF16_DEBUG("[BF16] missing bf16 double-block weights:"
                       " img_q=%d img_k=%d img_v=%d img_proj=%d"
                       " img_gate=%d img_up=%d img_down=%d"
                       " txt_q=%d txt_k=%d txt_v=%d txt_proj=%d"
                       " txt_gate=%d txt_up=%d txt_down=%d\n",
                       block->img_q_weight_bf16 != NULL,
                       block->img_k_weight_bf16 != NULL,
                       block->img_v_weight_bf16 != NULL,
                       block->img_proj_weight_bf16 != NULL,
                       block->img_mlp_gate_weight_bf16 != NULL,
                       block->img_mlp_up_weight_bf16 != NULL,
                       block->img_mlp_down_weight_bf16 != NULL,
                       block->txt_q_weight_bf16 != NULL,
                       block->txt_k_weight_bf16 != NULL,
                       block->txt_v_weight_bf16 != NULL,
                       block->txt_proj_weight_bf16 != NULL,
                       block->txt_mlp_gate_weight_bf16 != NULL,
                       block->txt_mlp_up_weight_bf16 != NULL,
                       block->txt_mlp_down_weight_bf16 != NULL);
        }
        return 0;
    }

    int hidden = tf->hidden_size;
    int heads = tf->num_heads;
    int head_dim = tf->head_dim;
    int mlp_hidden = tf->mlp_hidden;
    float eps = 1e-6f;
    int axis_dim = 32;

    iris_gpu_tensor_t img_norm = NULL;
    iris_gpu_tensor_t txt_norm = NULL;
    iris_gpu_tensor_t img_q = NULL, img_k = NULL, img_v = NULL;
    iris_gpu_tensor_t txt_q = NULL, txt_k = NULL, txt_v = NULL;
    iris_gpu_tensor_t img_attn_out = NULL, txt_attn_out = NULL;
    iris_gpu_tensor_t img_proj = NULL, txt_proj = NULL;
    iris_gpu_tensor_t img_gate = NULL, img_up = NULL, img_down = NULL;
    iris_gpu_tensor_t txt_gate = NULL, txt_up = NULL, txt_down = NULL;

    /* AdaLN (shift1/scale1) */
    img_norm = iris_gpu_tensor_alloc_f16((size_t)img_seq * hidden);
    txt_norm = iris_gpu_tensor_alloc_f16((size_t)txt_seq * hidden);
    if (!img_norm || !txt_norm) {
        BF16_DEBUG("[BF16] double block: failed to alloc norm tensors\n");
        goto cleanup;
    }

    iris_gpu_adaln_norm_bf16(img_norm, img_hidden, img_shift1, img_scale1, img_seq, hidden, eps);
    iris_gpu_adaln_norm_bf16(txt_norm, txt_hidden, txt_shift1, txt_scale1, txt_seq, hidden, eps);

    /* QKV projections */
    img_q = iris_gpu_linear_bf16_native(img_norm, block->img_q_weight_bf16, img_seq, hidden, hidden);
    img_k = iris_gpu_linear_bf16_native(img_norm, block->img_k_weight_bf16, img_seq, hidden, hidden);
    img_v = iris_gpu_linear_bf16_native(img_norm, block->img_v_weight_bf16, img_seq, hidden, hidden);

    txt_q = iris_gpu_linear_bf16_native(txt_norm, block->txt_q_weight_bf16, txt_seq, hidden, hidden);
    txt_k = iris_gpu_linear_bf16_native(txt_norm, block->txt_k_weight_bf16, txt_seq, hidden, hidden);
    txt_v = iris_gpu_linear_bf16_native(txt_norm, block->txt_v_weight_bf16, txt_seq, hidden, hidden);

    if (!img_q || !img_k || !img_v || !txt_q || !txt_k || !txt_v) {
        BF16_DEBUG("[BF16] double block: QKV projection failed\n");
        goto cleanup;
    }

    /* QK RMSNorm weights (bf16) */
    iris_gpu_tensor_t img_norm_q = bf16_tensor_from_f32(block->img_norm_q_weight, head_dim);
    iris_gpu_tensor_t img_norm_k = bf16_tensor_from_f32(block->img_norm_k_weight, head_dim);
    iris_gpu_tensor_t txt_norm_q = bf16_tensor_from_f32(block->txt_norm_q_weight, head_dim);
    iris_gpu_tensor_t txt_norm_k = bf16_tensor_from_f32(block->txt_norm_k_weight, head_dim);

    if (!img_norm_q || !img_norm_k || !txt_norm_q || !txt_norm_k) {
        BF16_DEBUG("[BF16] double block: failed to create qk norm tensors\n");
        if (img_norm_q) iris_gpu_tensor_free(img_norm_q);
        if (img_norm_k) iris_gpu_tensor_free(img_norm_k);
        if (txt_norm_q) iris_gpu_tensor_free(txt_norm_q);
        if (txt_norm_k) iris_gpu_tensor_free(txt_norm_k);
        goto cleanup;
    }

    iris_gpu_qk_rms_norm_bf16(img_q, img_k, img_norm_q, img_norm_k, img_seq, heads, head_dim, eps);
    iris_gpu_qk_rms_norm_bf16(txt_q, txt_k, txt_norm_q, txt_norm_k, txt_seq, heads, head_dim, eps);

    iris_gpu_tensor_free(img_norm_q);
    iris_gpu_tensor_free(img_norm_k);
    iris_gpu_tensor_free(txt_norm_q);
    iris_gpu_tensor_free(txt_norm_k);

    /* RoPE */
    iris_gpu_rope_2d_bf16(img_q, img_rope_cos, img_rope_sin, img_seq, heads, head_dim, axis_dim);
    iris_gpu_rope_2d_bf16(img_k, img_rope_cos, img_rope_sin, img_seq, heads, head_dim, axis_dim);
    iris_gpu_rope_2d_bf16(txt_q, txt_rope_cos, txt_rope_sin, txt_seq, heads, head_dim, axis_dim);
    iris_gpu_rope_2d_bf16(txt_k, txt_rope_cos, txt_rope_sin, txt_seq, heads, head_dim, axis_dim);

    /* Joint attention */
    img_attn_out = iris_gpu_tensor_alloc_f16((size_t)img_seq * hidden);
    txt_attn_out = iris_gpu_tensor_alloc_f16((size_t)txt_seq * hidden);
    if (!img_attn_out || !txt_attn_out) {
        BF16_DEBUG("[BF16] double block: failed to alloc attn outputs\n");
        goto cleanup;
    }

    if (!joint_attention_bf16(img_attn_out, txt_attn_out,
                              img_q, img_k, img_v,
                              txt_q, txt_k, txt_v,
                              img_seq, txt_seq, heads, head_dim)) {
        BF16_DEBUG("[BF16] double block: joint attention failed\n");
        goto cleanup;
    }

    /* Projection + residual */
    img_proj = iris_gpu_linear_bf16_native(img_attn_out, block->img_proj_weight_bf16,
                                           img_seq, hidden, hidden);
    txt_proj = iris_gpu_linear_bf16_native(txt_attn_out, block->txt_proj_weight_bf16,
                                           txt_seq, hidden, hidden);
    if (!img_proj || !txt_proj) {
        BF16_DEBUG("[BF16] double block: proj weights failed\n");
        goto cleanup;
    }

    iris_gpu_gated_add_bf16(img_hidden, img_gate1, img_proj, img_seq, hidden);
    iris_gpu_gated_add_bf16(txt_hidden, txt_gate1, txt_proj, txt_seq, hidden);

    /* FFN (AdaLN + SwiGLU) */
    iris_gpu_adaln_norm_bf16(img_norm, img_hidden, img_shift2, img_scale2, img_seq, hidden, eps);
    iris_gpu_adaln_norm_bf16(txt_norm, txt_hidden, txt_shift2, txt_scale2, txt_seq, hidden, eps);

    img_gate = iris_gpu_linear_bf16_native(img_norm, block->img_mlp_gate_weight_bf16,
                                           img_seq, hidden, mlp_hidden);
    img_up = iris_gpu_linear_bf16_native(img_norm, block->img_mlp_up_weight_bf16,
                                         img_seq, hidden, mlp_hidden);
    txt_gate = iris_gpu_linear_bf16_native(txt_norm, block->txt_mlp_gate_weight_bf16,
                                           txt_seq, hidden, mlp_hidden);
    txt_up = iris_gpu_linear_bf16_native(txt_norm, block->txt_mlp_up_weight_bf16,
                                         txt_seq, hidden, mlp_hidden);
    if (!img_gate || !img_up || !txt_gate || !txt_up) {
        BF16_DEBUG("[BF16] double block: FFN up/gate failed\n");
        goto cleanup;
    }

    iris_gpu_silu_mul_bf16(img_gate, img_up, img_seq * mlp_hidden);
    iris_gpu_silu_mul_bf16(txt_gate, txt_up, txt_seq * mlp_hidden);

    img_down = iris_gpu_linear_bf16_native(img_gate, block->img_mlp_down_weight_bf16,
                                           img_seq, mlp_hidden, hidden);
    txt_down = iris_gpu_linear_bf16_native(txt_gate, block->txt_mlp_down_weight_bf16,
                                           txt_seq, mlp_hidden, hidden);
    if (!img_down || !txt_down) {
        BF16_DEBUG("[BF16] double block: FFN down failed\n");
        goto cleanup;
    }

    iris_gpu_gated_add_bf16(img_hidden, img_gate2, img_down, img_seq, hidden);
    iris_gpu_gated_add_bf16(txt_hidden, txt_gate2, txt_down, txt_seq, hidden);

    iris_gpu_tensor_free(img_norm);
    iris_gpu_tensor_free(txt_norm);
    iris_gpu_tensor_free(img_q);
    iris_gpu_tensor_free(img_k);
    iris_gpu_tensor_free(img_v);
    iris_gpu_tensor_free(txt_q);
    iris_gpu_tensor_free(txt_k);
    iris_gpu_tensor_free(txt_v);
    iris_gpu_tensor_free(img_attn_out);
    iris_gpu_tensor_free(txt_attn_out);
    iris_gpu_tensor_free(img_proj);
    iris_gpu_tensor_free(txt_proj);
    iris_gpu_tensor_free(img_gate);
    iris_gpu_tensor_free(img_up);
    iris_gpu_tensor_free(img_down);
    iris_gpu_tensor_free(txt_gate);
    iris_gpu_tensor_free(txt_up);
    iris_gpu_tensor_free(txt_down);
    return 1;

cleanup:
    if (img_norm) iris_gpu_tensor_free(img_norm);
    if (txt_norm) iris_gpu_tensor_free(txt_norm);
    if (img_q) iris_gpu_tensor_free(img_q);
    if (img_k) iris_gpu_tensor_free(img_k);
    if (img_v) iris_gpu_tensor_free(img_v);
    if (txt_q) iris_gpu_tensor_free(txt_q);
    if (txt_k) iris_gpu_tensor_free(txt_k);
    if (txt_v) iris_gpu_tensor_free(txt_v);
    if (img_attn_out) iris_gpu_tensor_free(img_attn_out);
    if (txt_attn_out) iris_gpu_tensor_free(txt_attn_out);
    if (img_proj) iris_gpu_tensor_free(img_proj);
    if (txt_proj) iris_gpu_tensor_free(txt_proj);
    if (img_gate) iris_gpu_tensor_free(img_gate);
    if (img_up) iris_gpu_tensor_free(img_up);
    if (img_down) iris_gpu_tensor_free(img_down);
    if (txt_gate) iris_gpu_tensor_free(txt_gate);
    if (txt_up) iris_gpu_tensor_free(txt_up);
    if (txt_down) iris_gpu_tensor_free(txt_down);
    return 0;
}
#endif

/* ========================================================================
 * SwiGLU FFN
 * ======================================================================== */

/* SwiGLU FFN with optional bf16 weights - uses pre-allocated buffers from tf */
static void swiglu_ffn_bf16(float *out, const float *x,
                            const float *gate_weight, const float *up_weight,
                            const float *down_weight,
                            const uint16_t *gate_weight_bf16,
                            const uint16_t *up_weight_bf16,
                            const uint16_t *down_weight_bf16,
                            int seq, int hidden, int mlp_hidden,
                            iris_transformer_flux_t *tf) {
    /* Use pre-allocated FFN work buffers */
    float *gate = tf->ffn_gate;
    float *up = tf->ffn_up;

    /* Gate and up projections - these are independent, batch them */
    iris_gpu_begin_batch();
    LINEAR_BF16_OR_F32(gate, x, gate_weight, gate_weight_bf16, seq, hidden, mlp_hidden);
    LINEAR_BF16_OR_F32(up, x, up_weight, up_weight_bf16, seq, hidden, mlp_hidden);
    iris_gpu_end_batch();

    /* SiLU(gate) * up - fused for better performance */
    iris_silu_mul(gate, up, seq * mlp_hidden);

    /* Down projection */
    LINEAR_BF16_OR_F32(out, gate, down_weight, down_weight_bf16, seq, mlp_hidden, hidden);

    /* No free - using pre-allocated buffers */
}

/* ========================================================================
 * Double-Stream Block (MM-DiT)
 * ======================================================================== */

/* One MM-DiT (Multi-Modal DiT) double block. Processes image and text as
 * separate streams with their own Q/K/V projections, but performs joint
 * attention (K and V are concatenated from both streams so each modality
 * attends to the other). Each stream gets its own AdaLN modulation with
 * 6 parameters: shift1, scale1, gate1 (pre-attention), shift2, scale2,
 * gate2 (pre-FFN). The gating mechanism controls information flow from
 * the attention and FFN residual paths. */
static void double_block_forward(float *img_hidden, float *txt_hidden,
                                 const double_block_t *block,
                                 const float *img_mod, const float *txt_mod,
                                 const float *img_rope_cos, const float *img_rope_sin,
                                 const float *txt_rope_cos, const float *txt_rope_sin,
                                 int img_seq, int txt_seq,
                                 iris_transformer_flux_t *tf) {
    int hidden = tf->hidden_size;
    int heads = tf->num_heads;
    int head_dim = tf->head_dim;
    int mlp_hidden = tf->mlp_hidden;
    float eps = 1e-6f;

    /* Extract pre-computed modulation parameters */
    const float *img_shift1 = img_mod;
    const float *img_scale1 = img_mod + hidden;
    const float *img_gate1 = img_mod + hidden * 2;
    const float *img_shift2 = img_mod + hidden * 3;
    const float *img_scale2 = img_mod + hidden * 4;
    const float *img_gate2 = img_mod + hidden * 5;

    const float *txt_shift1 = txt_mod;
    const float *txt_scale1 = txt_mod + hidden;
    const float *txt_gate1 = txt_mod + hidden * 2;
    const float *txt_shift2 = txt_mod + hidden * 3;
    const float *txt_scale2 = txt_mod + hidden * 4;
    const float *txt_gate2 = txt_mod + hidden * 5;

    /* Image stream: AdaLN -> QKV -> QK-norm -> RoPE */
    float *img_norm = tf->work1;
    apply_adaln(img_norm, img_hidden, img_shift1, img_scale1, img_seq, hidden, eps);

#ifdef DEBUG_DOUBLE_BLOCK
    static int block_idx = 0;
    if (block_idx == 0) {
        fprintf(stderr, "\n[DBL] img_mod[0,:10] (shift1): ");
        for (int i = 0; i < 10; i++) fprintf(stderr, "%.6f ", img_shift1[i]);
        fprintf(stderr, "\n[DBL] img_mod[0,3072:3082] (scale1): ");
        for (int i = 0; i < 10; i++) fprintf(stderr, "%.6f ", img_scale1[i]);
        fprintf(stderr, "\n[DBL] After AdaLN img_norm[0,0,:10]: ");
        for (int i = 0; i < 10; i++) fprintf(stderr, "%.6f ", img_norm[i]);
        fprintf(stderr, "\n");
    }
#endif

    /* Separate Q, K, V projections (fixes interleaved output bug)
     * Note: These 3 projections are independent - batch them for GPU efficiency */
    float *img_q = tf->work2;
    float *img_k = img_q + img_seq * hidden;
    float *img_v = img_k + img_seq * hidden;

    iris_gpu_begin_batch();
    LINEAR_BF16_OR_F32(img_q, img_norm, block->img_q_weight, block->img_q_weight_bf16,
                       img_seq, hidden, hidden);
    LINEAR_BF16_OR_F32(img_k, img_norm, block->img_k_weight, block->img_k_weight_bf16,
                       img_seq, hidden, hidden);
    LINEAR_BF16_OR_F32(img_v, img_norm, block->img_v_weight, block->img_v_weight_bf16,
                       img_seq, hidden, hidden);
    iris_gpu_end_batch();

    /* Apply QK normalization (per-head RMSNorm) */
    apply_qk_norm(img_q, img_k, block->img_norm_q_weight, block->img_norm_k_weight,
                  img_seq, heads, head_dim, eps);

    /* Apply 2D RoPE to image Q, K (using h, w positions) */
    int axis_dim = 32;
    apply_rope_2d(img_q, img_rope_cos, img_rope_sin, img_seq, heads, head_dim, axis_dim);
    apply_rope_2d(img_k, img_rope_cos, img_rope_sin, img_seq, heads, head_dim, axis_dim);

#ifdef DEBUG_DOUBLE_BLOCK
    if (block_idx == 0) {
        fprintf(stderr, "[DBL] After Q proj img_q[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_q[i]);
        fprintf(stderr, "\n[DBL] After RoPE img_q[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_q[i]);
        fprintf(stderr, "\n");
    }
#endif

    /* Text stream: AdaLN -> QKV -> QK-norm -> RoPE */
    float *txt_norm = img_norm + img_seq * hidden;
    apply_adaln(txt_norm, txt_hidden, txt_shift1, txt_scale1, txt_seq, hidden, eps);

    /* Separate Q, K, V projections for text
     * Note: These 3 projections are independent - batch them for GPU efficiency */
    float *txt_q = img_v + img_seq * hidden;  /* After img_v */
    float *txt_k = txt_q + txt_seq * hidden;
    float *txt_v = txt_k + txt_seq * hidden;

    iris_gpu_begin_batch();
    LINEAR_BF16_OR_F32(txt_q, txt_norm, block->txt_q_weight, block->txt_q_weight_bf16,
                       txt_seq, hidden, hidden);
    LINEAR_BF16_OR_F32(txt_k, txt_norm, block->txt_k_weight, block->txt_k_weight_bf16,
                       txt_seq, hidden, hidden);
    LINEAR_BF16_OR_F32(txt_v, txt_norm, block->txt_v_weight, block->txt_v_weight_bf16,
                       txt_seq, hidden, hidden);
    iris_gpu_end_batch();

    /* Apply QK normalization */
    apply_qk_norm(txt_q, txt_k, block->txt_norm_q_weight, block->txt_norm_k_weight,
                  txt_seq, heads, head_dim, eps);

    /* Apply text RoPE - text tokens have position IDs (0, 0, 0, L) where L is sequence index
     * This applies rotation in axis 3 (dims 96-127)
     */
    apply_rope_2d(txt_q, txt_rope_cos, txt_rope_sin, txt_seq, heads, head_dim, axis_dim);
    apply_rope_2d(txt_k, txt_rope_cos, txt_rope_sin, txt_seq, heads, head_dim, axis_dim);

    /* Joint attention - use pre-allocated buffers */
    float *img_attn_out = tf->double_img_attn_out;
    float *txt_attn_out = tf->double_txt_attn_out;

    joint_attention(img_attn_out, txt_attn_out,
                    img_q, img_k, img_v,
                    txt_q, txt_k, txt_v,
                    img_seq, txt_seq, heads, head_dim, tf);

#ifdef DEBUG_DOUBLE_BLOCK
    if (block_idx == 0) {
        fprintf(stderr, "[DBL] After attn img_attn_out[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_attn_out[i]);
        fprintf(stderr, "\n");
    }
#endif

    /* Project attention output
     * Note: img_proj and txt_proj are independent - batch them for GPU efficiency */
    float *img_proj = tf->work1;
    float *txt_proj = img_proj + img_seq * hidden;

    iris_gpu_begin_batch();
    LINEAR_BF16_OR_F32(img_proj, img_attn_out, block->img_proj_weight, block->img_proj_weight_bf16,
                       img_seq, hidden, hidden);
    LINEAR_BF16_OR_F32(txt_proj, txt_attn_out, block->txt_proj_weight, block->txt_proj_weight_bf16,
                       txt_seq, hidden, hidden);
    iris_gpu_end_batch();

#ifdef DEBUG_DOUBLE_BLOCK
    if (block_idx == 0) {
        fprintf(stderr, "[DBL] After proj img_proj[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_proj[i]);
        fprintf(stderr, "\n");
        fprintf(stderr, "[DBL] gate1[0:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_gate1[i]);
        fprintf(stderr, "\n");
    }
#endif

    /* Apply gate and add residual - use vectorized helper */
    gated_add(img_hidden, img_gate1, img_proj, img_seq, hidden);
    gated_add(txt_hidden, txt_gate1, txt_proj, txt_seq, hidden);

#ifdef DEBUG_DOUBLE_BLOCK
    if (block_idx == 0) {
        fprintf(stderr, "[DBL] After attn residual img_hidden[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_hidden[i]);
        fprintf(stderr, "\n");
    }
#endif

    /* FFN for image */
    apply_adaln(img_norm, img_hidden, img_shift2, img_scale2, img_seq, hidden, eps);

#ifdef DEBUG_DOUBLE_BLOCK
    if (block_idx == 0) {
        fprintf(stderr, "[DBL] FFN input (after AdaLN) img_norm[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_norm[i]);
        fprintf(stderr, "\n");
    }
#endif

    swiglu_ffn_bf16(img_proj, img_norm,
                    block->img_mlp_gate_weight, block->img_mlp_up_weight,
                    block->img_mlp_down_weight,
                    block->img_mlp_gate_weight_bf16, block->img_mlp_up_weight_bf16,
                    block->img_mlp_down_weight_bf16,
                    img_seq, hidden, mlp_hidden, tf);

#ifdef DEBUG_DOUBLE_BLOCK
    if (block_idx == 0) {
        fprintf(stderr, "[DBL] FFN output img_proj[0,0,:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_proj[i]);
        fprintf(stderr, "\n");
        fprintf(stderr, "[DBL] gate2[0:5]: ");
        for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_gate2[i]);
        fprintf(stderr, "\n");
    }
#endif

    gated_add(img_hidden, img_gate2, img_proj, img_seq, hidden);

#ifdef DEBUG_DOUBLE_BLOCK
    fprintf(stderr, "[DBL%d] After FFN residual img_hidden[0,0,:5]: ", block_idx);
    for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_hidden[i]);
    fprintf(stderr, "\n");
#endif

    /* FFN for text */
    apply_adaln(txt_norm, txt_hidden, txt_shift2, txt_scale2, txt_seq, hidden, eps);
    swiglu_ffn_bf16(txt_proj, txt_norm,
                    block->txt_mlp_gate_weight, block->txt_mlp_up_weight,
                    block->txt_mlp_down_weight,
                    block->txt_mlp_gate_weight_bf16, block->txt_mlp_up_weight_bf16,
                    block->txt_mlp_down_weight_bf16,
                    txt_seq, hidden, mlp_hidden, tf);
    gated_add(txt_hidden, txt_gate2, txt_proj, txt_seq, hidden);

    /* No free - using pre-allocated buffers */

#ifdef DEBUG_DOUBLE_BLOCK
    block_idx++;
#endif
}

/* ========================================================================
 * Single-Stream Block (Parallel DiT)
 * ======================================================================== */

#ifdef USE_METAL
/* GPU-optimized single block forward using persistent GPU tensors
 * Keeps activations on GPU throughout the block to minimize memory transfers.
 * Returns 1 if GPU path was used, 0 to fall back to CPU path.
 */
static int single_block_forward_gpu(float *hidden, const single_block_t *block,
                                    const float *t_emb, const float *adaln_weight,
                                    const float *img_rope_cos, const float *img_rope_sin,
                                    const float *txt_rope_cos, const float *txt_rope_sin,
                                    int seq, int img_offset, iris_transformer_flux_t *tf) {
    /* Check if GPU tensors are available */
    if (!iris_metal_available() || !iris_metal_shaders_available()) return 0;
    if (block->qkv_mlp_weight_bf16 == NULL) return 0;  /* Need bf16 weights */

    int h_size = tf->hidden_size;
    int heads = tf->num_heads;
    int head_dim = tf->head_dim;
    int mlp_hidden = tf->mlp_hidden;
    float eps = 1e-6f;

    /* === Phase 1: AdaLN modulation (small, keep on CPU) === */
    int mod_size = h_size * 3;
    float *t_emb_silu = tf->t_emb_silu;
    for (int i = 0; i < h_size; i++) {
        float x = t_emb[i];
        t_emb_silu[i] = x / (1.0f + expf(-x));
    }
    /* mod_params goes after fused_out in work2 buffer. Must match single_block_forward. */
    int fused_dim = h_size * 3 + mlp_hidden * 2;
    float *mod_params = tf->work2 + seq * fused_dim;
    iris_linear_nobias(mod_params, t_emb_silu, adaln_weight, 1, h_size, mod_size);

    float *shift = mod_params;
    float *scale = mod_params + h_size;
    float *gate = mod_params + h_size * 2;

    /* === Phase 2: Create GPU tensors and enter batch mode === */
    iris_gpu_batch_begin();

    /* Create input tensor on GPU */
    iris_gpu_tensor_t hidden_gpu = iris_gpu_tensor_create(hidden, seq * h_size);
    if (!hidden_gpu) {
        iris_gpu_batch_end();
        return 0;
    }

    /* Allocate output tensors */
    iris_gpu_tensor_t norm_gpu = iris_gpu_tensor_alloc(seq * h_size);
    iris_gpu_tensor_t fused_gpu = iris_gpu_tensor_alloc(seq * fused_dim);
    iris_gpu_tensor_t q_gpu = iris_gpu_tensor_alloc(seq * h_size);
    iris_gpu_tensor_t k_gpu = iris_gpu_tensor_alloc(seq * h_size);
    iris_gpu_tensor_t v_gpu = iris_gpu_tensor_alloc(seq * h_size);
    iris_gpu_tensor_t gate_gpu = iris_gpu_tensor_alloc(seq * mlp_hidden);
    iris_gpu_tensor_t up_gpu = iris_gpu_tensor_alloc(seq * mlp_hidden);
    iris_gpu_tensor_t attn_out_gpu = iris_gpu_tensor_alloc(seq * h_size);
    iris_gpu_tensor_t concat_gpu = iris_gpu_tensor_alloc(seq * (h_size + mlp_hidden));
    iris_gpu_tensor_t proj_out_gpu = iris_gpu_tensor_alloc(seq * h_size);

    if (!norm_gpu || !fused_gpu || !q_gpu || !k_gpu || !v_gpu ||
        !gate_gpu || !up_gpu || !attn_out_gpu || !concat_gpu || !proj_out_gpu) {
        /* Cleanup and fall back */
        if (hidden_gpu) iris_gpu_tensor_free(hidden_gpu);
        if (norm_gpu) iris_gpu_tensor_free(norm_gpu);
        if (fused_gpu) iris_gpu_tensor_free(fused_gpu);
        if (q_gpu) iris_gpu_tensor_free(q_gpu);
        if (k_gpu) iris_gpu_tensor_free(k_gpu);
        if (v_gpu) iris_gpu_tensor_free(v_gpu);
        if (gate_gpu) iris_gpu_tensor_free(gate_gpu);
        if (up_gpu) iris_gpu_tensor_free(up_gpu);
        if (attn_out_gpu) iris_gpu_tensor_free(attn_out_gpu);
        if (concat_gpu) iris_gpu_tensor_free(concat_gpu);
        if (proj_out_gpu) iris_gpu_tensor_free(proj_out_gpu);
        iris_gpu_batch_end();
        return 0;
    }

    /* === Phase 3: AdaLN normalization on GPU === */
    iris_gpu_adaln_norm(norm_gpu, hidden_gpu, shift, scale, seq, h_size, eps);

    /* === Phase 4: Fused QKV + MLP projection on GPU === */
    iris_gpu_tensor_t fused_result = iris_gpu_linear_bf16(norm_gpu,
                                                          block->qkv_mlp_weight_bf16,
                                                          seq, h_size, fused_dim);
    if (!fused_result) {
        /* Cleanup and fall back */
        iris_gpu_tensor_free(hidden_gpu);
        iris_gpu_tensor_free(norm_gpu);
        iris_gpu_tensor_free(fused_gpu);
        iris_gpu_tensor_free(q_gpu);
        iris_gpu_tensor_free(k_gpu);
        iris_gpu_tensor_free(v_gpu);
        iris_gpu_tensor_free(gate_gpu);
        iris_gpu_tensor_free(up_gpu);
        iris_gpu_tensor_free(attn_out_gpu);
        iris_gpu_tensor_free(concat_gpu);
        iris_gpu_tensor_free(proj_out_gpu);
        iris_gpu_batch_end();
        return 0;
    }

    /* === Phase 5: Split fused output on GPU === */
    iris_gpu_split_qkv_mlp(fused_result, q_gpu, k_gpu, v_gpu, gate_gpu, up_gpu,
                           seq, h_size, mlp_hidden);

    /* === Phase 6+7: Fused QK RMSNorm + unified RoPE on GPU === */
    iris_gpu_qk_norm_rope_unified(q_gpu, k_gpu,
                                   block->norm_q_weight, block->norm_k_weight,
                                   txt_rope_cos, txt_rope_sin,
                                   img_rope_cos, img_rope_sin,
                                   seq, img_offset, heads, head_dim, eps);

    /* === Phase 8: Self-attention on GPU === */
    /* Try fused attention (seq <= 1024), then bf16 MPS attention (any seq) */
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    if (!iris_gpu_attention_fused(attn_out_gpu, q_gpu, k_gpu, v_gpu,
                                  seq, seq, heads, head_dim, attn_scale)) {
        /* Fused attention failed (seq > 1024) - use bf16 MPS attention */
        iris_gpu_batch_end();
        iris_gpu_attention_bf16(attn_out_gpu, q_gpu, k_gpu, v_gpu,
                                seq, seq, heads, head_dim, attn_scale);
        iris_gpu_batch_begin();
    }

    /* === Phase 9: SwiGLU on GPU === */
    iris_gpu_silu_mul(gate_gpu, up_gpu, seq * mlp_hidden);

    /* === Phase 10: Concat attention + MLP outputs on GPU === */
    iris_gpu_concat_attn_mlp(attn_out_gpu, gate_gpu, concat_gpu, seq, h_size, mlp_hidden);

    /* === Phase 11: Final projection on GPU === */
    /* Free pre-allocated tensor since linear returns a new one */
    iris_gpu_tensor_free(proj_out_gpu);
    proj_out_gpu = iris_gpu_linear_bf16(concat_gpu, block->proj_mlp_weight_bf16,
                                        seq, h_size + mlp_hidden, h_size);
    if (!proj_out_gpu) {
        /* Fall back to CPU projection */
        iris_gpu_batch_end();
        float *concat_cpu = tf->single_concat;
        iris_gpu_tensor_read(concat_gpu, concat_cpu);
        float *proj_out_cpu = tf->work1;
        iris_linear_nobias_bf16(proj_out_cpu, concat_cpu, block->proj_mlp_weight_bf16,
                                seq, h_size + mlp_hidden, h_size);
        gated_add(hidden, gate, proj_out_cpu, seq, h_size);
        goto cleanup;
    }

    /* === Phase 12: Gated add residual on GPU === */
    iris_gpu_gated_add(hidden_gpu, gate, proj_out_gpu, seq, h_size);

    /* === Phase 13: Sync and copy result back === */
    iris_gpu_batch_end();
    iris_gpu_tensor_read(hidden_gpu, hidden);

cleanup:
    /* === Cleanup GPU tensors === */
    iris_gpu_tensor_free(hidden_gpu);
    iris_gpu_tensor_free(norm_gpu);
    iris_gpu_tensor_free(fused_result);
    iris_gpu_tensor_free(q_gpu);
    iris_gpu_tensor_free(k_gpu);
    iris_gpu_tensor_free(v_gpu);
    iris_gpu_tensor_free(gate_gpu);
    iris_gpu_tensor_free(up_gpu);
    iris_gpu_tensor_free(attn_out_gpu);
    iris_gpu_tensor_free(concat_gpu);
    iris_gpu_tensor_free(proj_out_gpu);

    return 1;  /* GPU path succeeded */
}

/* ========================================================================
 * Scratch buffers for GPU-chained single block (f32 path)
 * Pre-allocating these avoids 10 GPU tensor alloc/frees per block (200 per step)
 * ======================================================================== */
typedef struct {
    int seq;
    int hidden;
    int mlp_hidden;
    int fused_dim;
    iris_gpu_tensor_t norm;
    iris_gpu_tensor_t fused;
    iris_gpu_tensor_t q;
    iris_gpu_tensor_t k;
    iris_gpu_tensor_t v;
    iris_gpu_tensor_t gate_mlp;
    iris_gpu_tensor_t up;
    iris_gpu_tensor_t attn_out;
    iris_gpu_tensor_t concat;
    iris_gpu_tensor_t proj_out;
} single_block_gpu_scratch_t;

static void single_block_gpu_scratch_free(single_block_gpu_scratch_t *s) {
    if (!s) return;
    if (s->norm) iris_gpu_tensor_free(s->norm);
    if (s->fused) iris_gpu_tensor_free(s->fused);
    if (s->q) iris_gpu_tensor_free(s->q);
    if (s->k) iris_gpu_tensor_free(s->k);
    if (s->v) iris_gpu_tensor_free(s->v);
    if (s->gate_mlp) iris_gpu_tensor_free(s->gate_mlp);
    if (s->up) iris_gpu_tensor_free(s->up);
    if (s->attn_out) iris_gpu_tensor_free(s->attn_out);
    if (s->concat) iris_gpu_tensor_free(s->concat);
    if (s->proj_out) iris_gpu_tensor_free(s->proj_out);
    memset(s, 0, sizeof(*s));
}

static int single_block_gpu_scratch_init(single_block_gpu_scratch_t *s,
                                         int seq, int hidden, int mlp_hidden) {
    if (!s || seq <= 0 || hidden <= 0 || mlp_hidden <= 0) return 0;
    memset(s, 0, sizeof(*s));

    s->seq = seq;
    s->hidden = hidden;
    s->mlp_hidden = mlp_hidden;
    s->fused_dim = hidden * 3 + mlp_hidden * 2;

    /* Allocate all f32 scratch buffers */
    s->norm = iris_gpu_tensor_alloc((size_t)seq * hidden);
    s->fused = iris_gpu_tensor_alloc((size_t)seq * s->fused_dim);
    s->q = iris_gpu_tensor_alloc((size_t)seq * hidden);
    s->k = iris_gpu_tensor_alloc((size_t)seq * hidden);
    s->v = iris_gpu_tensor_alloc((size_t)seq * hidden);
    s->gate_mlp = iris_gpu_tensor_alloc((size_t)seq * mlp_hidden);
    s->up = iris_gpu_tensor_alloc((size_t)seq * mlp_hidden);
    s->attn_out = iris_gpu_tensor_alloc((size_t)seq * hidden);
    s->concat = iris_gpu_tensor_alloc((size_t)seq * (hidden + mlp_hidden));
    s->proj_out = iris_gpu_tensor_alloc((size_t)seq * hidden);

    if (!s->norm || !s->fused || !s->q || !s->k || !s->v ||
        !s->gate_mlp || !s->up || !s->attn_out || !s->concat || !s->proj_out) {
        single_block_gpu_scratch_free(s);
        return 0;
    }

    return 1;
}

/* GPU-chained single block: operates on a GPU tensor that persists across blocks.
 * Unlike single_block_forward_gpu(), this does NOT copy hidden to/from CPU.
 * The hidden_gpu tensor must be pre-allocated and will be modified in-place.
 * Caller must be in batch mode (iris_gpu_batch_begin called).
 * The shift/scale/gate modulation parameters must be pre-computed by the caller
 * (they are the same for all 20 single blocks within a step).
 * Optional scratch parameter: if provided, reuses pre-allocated buffers to avoid
 * per-block allocation overhead (10 allocations per block -> 0).
 * Returns 1 if GPU path was used, 0 to fall back to CPU path.
 */
static int single_block_forward_gpu_chained(iris_gpu_tensor_t hidden_gpu,
                                            const single_block_t *block,
                                            const float *shift, const float *scale, const float *gate,
                                            const float *img_rope_cos, const float *img_rope_sin,
                                            const float *txt_rope_cos, const float *txt_rope_sin,
                                            int seq, int img_offset, iris_transformer_flux_t *tf,
                                            single_block_gpu_scratch_t *scratch) {
    /* Check if GPU tensors are available */
    if (!iris_metal_available() || !iris_metal_shaders_available()) return 0;
    if (block->qkv_mlp_weight_bf16 == NULL) return 0;  /* Need bf16 weights */
    if (!hidden_gpu) return 0;

    int h_size = tf->hidden_size;
    int heads = tf->num_heads;
    int head_dim = tf->head_dim;
    int mlp_hidden = tf->mlp_hidden;
    int fused_dim = h_size * 3 + mlp_hidden * 2;  /* QKV + gate + up */
    float eps = 1e-6f;

    int own_tensors = (scratch == NULL);

    /* Either reuse scratch buffers (fast path) or allocate per-call (fallback) */
    iris_gpu_tensor_t norm_gpu = NULL;
    iris_gpu_tensor_t fused_result = NULL;
    iris_gpu_tensor_t q_gpu = NULL;
    iris_gpu_tensor_t k_gpu = NULL;
    iris_gpu_tensor_t v_gpu = NULL;
    iris_gpu_tensor_t gate_gpu = NULL;
    iris_gpu_tensor_t up_gpu = NULL;
    iris_gpu_tensor_t attn_out_gpu = NULL;
    iris_gpu_tensor_t concat_gpu = NULL;
    iris_gpu_tensor_t proj_out_gpu = NULL;

    if (scratch) {
        /* Validate scratch dimensions match */
        if (scratch->seq != seq || scratch->hidden != h_size ||
            scratch->mlp_hidden != mlp_hidden || scratch->fused_dim != fused_dim) {
            return 0;
        }
        norm_gpu = scratch->norm;
        fused_result = scratch->fused;
        q_gpu = scratch->q;
        k_gpu = scratch->k;
        v_gpu = scratch->v;
        gate_gpu = scratch->gate_mlp;
        up_gpu = scratch->up;
        attn_out_gpu = scratch->attn_out;
        concat_gpu = scratch->concat;
        proj_out_gpu = scratch->proj_out;

        if (!norm_gpu || !fused_result || !q_gpu || !k_gpu || !v_gpu ||
            !gate_gpu || !up_gpu || !attn_out_gpu || !concat_gpu || !proj_out_gpu) {
            return 0;
        }
    } else {
        /* Allocate intermediate GPU tensors */
        norm_gpu = iris_gpu_tensor_alloc(seq * h_size);
        q_gpu = iris_gpu_tensor_alloc(seq * h_size);
        k_gpu = iris_gpu_tensor_alloc(seq * h_size);
        v_gpu = iris_gpu_tensor_alloc(seq * h_size);
        gate_gpu = iris_gpu_tensor_alloc(seq * mlp_hidden);
        up_gpu = iris_gpu_tensor_alloc(seq * mlp_hidden);
        attn_out_gpu = iris_gpu_tensor_alloc(seq * h_size);
        concat_gpu = iris_gpu_tensor_alloc(seq * (h_size + mlp_hidden));

        if (!norm_gpu || !q_gpu || !k_gpu || !v_gpu ||
            !gate_gpu || !up_gpu || !attn_out_gpu || !concat_gpu) {
            /* Cleanup and fall back */
            if (norm_gpu) iris_gpu_tensor_free(norm_gpu);
            if (q_gpu) iris_gpu_tensor_free(q_gpu);
            if (k_gpu) iris_gpu_tensor_free(k_gpu);
            if (v_gpu) iris_gpu_tensor_free(v_gpu);
            if (gate_gpu) iris_gpu_tensor_free(gate_gpu);
            if (up_gpu) iris_gpu_tensor_free(up_gpu);
            if (attn_out_gpu) iris_gpu_tensor_free(attn_out_gpu);
            if (concat_gpu) iris_gpu_tensor_free(concat_gpu);
            return 0;
        }
    }

    /* === Phase 3: AdaLN normalization on GPU === */
    iris_gpu_adaln_norm(norm_gpu, hidden_gpu, shift, scale, seq, h_size, eps);

    /* === Phase 4: Fused QKV + MLP projection on GPU === */
    if (scratch) {
        if (!iris_gpu_linear_bf16_into(fused_result, norm_gpu,
                                       block->qkv_mlp_weight_bf16,
                                       seq, h_size, fused_dim)) {
            return 0;
        }
    } else {
        fused_result = iris_gpu_linear_bf16(norm_gpu,
                                            block->qkv_mlp_weight_bf16,
                                            seq, h_size, fused_dim);
        if (!fused_result) {
            return 0;
        }
    }

    /* === Phase 5: Split fused output on GPU === */
    iris_gpu_split_qkv_mlp(fused_result, q_gpu, k_gpu, v_gpu, gate_gpu, up_gpu,
                           seq, h_size, mlp_hidden);

    /* === Phase 6+7: Fused QK RMSNorm + unified RoPE on GPU === */
    iris_gpu_qk_norm_rope_unified(q_gpu, k_gpu,
                                   block->norm_q_weight, block->norm_k_weight,
                                   txt_rope_cos, txt_rope_sin,
                                   img_rope_cos, img_rope_sin,
                                   seq, img_offset, heads, head_dim, eps);

    /* === Phase 8: Self-attention on GPU === */
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    if (!iris_gpu_attention_fused(attn_out_gpu, q_gpu, k_gpu, v_gpu,
                                  seq, seq, heads, head_dim, attn_scale)) {
        /* Fall back to CPU attention - need to sync and copy */
        iris_gpu_sync();
        float *q_cpu = tf->single_q;
        float *k_cpu = tf->single_k;
        float *v_cpu = tf->single_v;
        iris_gpu_tensor_read(q_gpu, q_cpu);
        iris_gpu_tensor_read(k_gpu, k_cpu);
        iris_gpu_tensor_read(v_gpu, v_cpu);
        float *attn_out_cpu = tf->single_attn_out;
        mha_forward(attn_out_cpu, q_cpu, k_cpu, v_cpu, seq, heads, head_dim, tf);
        memcpy(iris_gpu_tensor_data(attn_out_gpu), attn_out_cpu, seq * h_size * sizeof(float));
    }

    /* === Phase 9: SwiGLU on GPU === */
    iris_gpu_silu_mul(gate_gpu, up_gpu, seq * mlp_hidden);

    /* === Phase 10: Concat attention + MLP outputs on GPU === */
    iris_gpu_concat_attn_mlp(attn_out_gpu, gate_gpu, concat_gpu, seq, h_size, mlp_hidden);

    /* === Phase 11: Final projection on GPU === */
    if (scratch) {
        if (!iris_gpu_linear_bf16_into(proj_out_gpu, concat_gpu,
                                       block->proj_mlp_weight_bf16,
                                       seq, h_size + mlp_hidden, h_size)) {
            return 0;
        }
    } else {
        proj_out_gpu = iris_gpu_linear_bf16(concat_gpu, block->proj_mlp_weight_bf16,
                                            seq, h_size + mlp_hidden, h_size);
        if (!proj_out_gpu) {
            /* Fall back to CPU projection - need to sync to read data */
            iris_gpu_sync();
            float *concat_cpu = tf->single_concat;
            iris_gpu_tensor_read(concat_gpu, concat_cpu);
            float *proj_out_cpu = tf->work1;
            iris_linear_nobias_bf16(proj_out_cpu, concat_cpu, block->proj_mlp_weight_bf16,
                                    seq, h_size + mlp_hidden, h_size);
            /* Read hidden back, apply gated add on CPU, write back */
            float *hidden_cpu = tf->work2;  /* Reuse work2 since mod_params is done */
            iris_gpu_tensor_read(hidden_gpu, hidden_cpu);
            gated_add(hidden_cpu, gate, proj_out_cpu, seq, h_size);
            memcpy(iris_gpu_tensor_data(hidden_gpu), hidden_cpu, seq * h_size * sizeof(float));
            goto cleanup;
        }
    }

    /* === Phase 12: Gated add residual on GPU (modifies hidden_gpu in-place) === */
    iris_gpu_gated_add(hidden_gpu, gate, proj_out_gpu, seq, h_size);

    /* Note: We do NOT sync here - GPU work will be committed when
     * iris_gpu_tensor_read is called at the end of all single blocks */

    if (!own_tensors) return 1;  /* Using scratch - don't free */

    iris_gpu_tensor_free(proj_out_gpu);

cleanup:
    /* Free all tensors allocated in this function (only when not using scratch) */
    if (fused_result) iris_gpu_tensor_free(fused_result);
    if (norm_gpu) iris_gpu_tensor_free(norm_gpu);
    if (q_gpu) iris_gpu_tensor_free(q_gpu);
    if (k_gpu) iris_gpu_tensor_free(k_gpu);
    if (v_gpu) iris_gpu_tensor_free(v_gpu);
    if (gate_gpu) iris_gpu_tensor_free(gate_gpu);
    if (up_gpu) iris_gpu_tensor_free(up_gpu);
    if (attn_out_gpu) iris_gpu_tensor_free(attn_out_gpu);
    if (concat_gpu) iris_gpu_tensor_free(concat_gpu);

    return 1;  /* GPU path succeeded */
}

/* ========================================================================
 * BF16 Native Single Block Forward
 *
 * This function processes a single block entirely in bf16. All intermediate
 * tensors stay in bf16 with
 * f32 accumulation only inside compute shaders for numerical stability.
 *
 * Parameters:
 * - hidden_gpu: bf16 GPU tensor [seq, hidden], modified in-place
 * - shift_bf16, scale_bf16, gate_bf16: pre-computed modulation (bf16 tensors)
 * - norm_q_weight_bf16, norm_k_weight_bf16: QK norm weights (bf16 tensors)
 *
 * Returns 1 on success, 0 to fall back to f32 path.
 * ======================================================================== */
typedef struct {
    int seq;
    int hidden;
    int mlp_hidden;
    int fused_dim;
    iris_gpu_tensor_t norm;
    iris_gpu_tensor_t fused;
    iris_gpu_tensor_t q;
    iris_gpu_tensor_t k;
    iris_gpu_tensor_t v;
    iris_gpu_tensor_t gate_mlp;
    iris_gpu_tensor_t up;
    iris_gpu_tensor_t attn_out;
    iris_gpu_tensor_t concat;
    iris_gpu_tensor_t proj_out;
} single_block_bf16_scratch_t;

static void single_block_bf16_scratch_free(single_block_bf16_scratch_t *s) {
    if (!s) return;
    if (s->norm) iris_gpu_tensor_free(s->norm);
    if (s->fused) iris_gpu_tensor_free(s->fused);
    if (s->q) iris_gpu_tensor_free(s->q);
    if (s->k) iris_gpu_tensor_free(s->k);
    if (s->v) iris_gpu_tensor_free(s->v);
    if (s->gate_mlp) iris_gpu_tensor_free(s->gate_mlp);
    if (s->up) iris_gpu_tensor_free(s->up);
    if (s->attn_out) iris_gpu_tensor_free(s->attn_out);
    if (s->concat) iris_gpu_tensor_free(s->concat);
    if (s->proj_out) iris_gpu_tensor_free(s->proj_out);
    memset(s, 0, sizeof(*s));
}

static int single_block_bf16_scratch_init(single_block_bf16_scratch_t *s,
                                         int seq, int hidden, int mlp_hidden) {
    if (!s || seq <= 0 || hidden <= 0 || mlp_hidden <= 0) return 0;
    memset(s, 0, sizeof(*s));

    s->seq = seq;
    s->hidden = hidden;
    s->mlp_hidden = mlp_hidden;
    s->fused_dim = hidden * 3 + mlp_hidden * 2;

    s->norm = iris_gpu_tensor_alloc_f16((size_t)seq * hidden);
    s->fused = iris_gpu_tensor_alloc_f16((size_t)seq * s->fused_dim);
    s->q = iris_gpu_tensor_alloc_f16((size_t)seq * hidden);
    s->k = iris_gpu_tensor_alloc_f16((size_t)seq * hidden);
    s->v = iris_gpu_tensor_alloc_f16((size_t)seq * hidden);
    s->gate_mlp = iris_gpu_tensor_alloc_f16((size_t)seq * mlp_hidden);
    s->up = iris_gpu_tensor_alloc_f16((size_t)seq * mlp_hidden);
    s->attn_out = iris_gpu_tensor_alloc_f16((size_t)seq * hidden);
    s->concat = iris_gpu_tensor_alloc_f16((size_t)seq * (hidden + mlp_hidden));
    s->proj_out = iris_gpu_tensor_alloc_f16((size_t)seq * hidden);

    if (!s->norm || !s->fused || !s->q || !s->k || !s->v ||
        !s->gate_mlp || !s->up || !s->attn_out || !s->concat || !s->proj_out) {
        single_block_bf16_scratch_free(s);
        return 0;
    }

    return 1;
}

static int single_block_forward_bf16(iris_gpu_tensor_t hidden_gpu,
                                      const single_block_t *block,
                                      iris_gpu_tensor_t shift_bf16,
                                      iris_gpu_tensor_t scale_bf16,
                                      iris_gpu_tensor_t gate_bf16,
                                      iris_gpu_tensor_t norm_q_weight_bf16,
                                      iris_gpu_tensor_t norm_k_weight_bf16,
                                      const float *img_rope_cos, const float *img_rope_sin,
                                      const float *txt_rope_cos, const float *txt_rope_sin,
                                      int seq, int img_offset, iris_transformer_flux_t *tf,
                                      single_block_bf16_scratch_t *scratch) {
    /* Check if bf16 pipeline is available */
    if (!iris_bf16_pipeline_available()) return 0;
    if (!hidden_gpu || !iris_gpu_tensor_is_f16(hidden_gpu)) return 0;
    if (block->qkv_mlp_weight_bf16 == NULL) return 0;

    int h_size = tf->hidden_size;
    int heads = tf->num_heads;
    int head_dim = tf->head_dim;
    int mlp_hidden = tf->mlp_hidden;
    int fused_dim = h_size * 3 + mlp_hidden * 2;
    float eps = 1e-6f;
    int axis_dim = 32;

    int own_tensors = (scratch == NULL);

    /* Either reuse a scratch set (fast path) or allocate per-call (fallback). */
    iris_gpu_tensor_t norm_gpu = NULL;
    iris_gpu_tensor_t fused_result = NULL;
    iris_gpu_tensor_t q_gpu = NULL;
    iris_gpu_tensor_t k_gpu = NULL;
    iris_gpu_tensor_t v_gpu = NULL;
    iris_gpu_tensor_t gate_mlp_gpu = NULL;
    iris_gpu_tensor_t up_gpu = NULL;
    iris_gpu_tensor_t attn_out_gpu = NULL;
    iris_gpu_tensor_t concat_gpu = NULL;
    iris_gpu_tensor_t proj_out_gpu = NULL;

    if (scratch) {
        if (scratch->seq != seq || scratch->hidden != h_size ||
            scratch->mlp_hidden != mlp_hidden || scratch->fused_dim != fused_dim) {
            return 0;
        }
        norm_gpu = scratch->norm;
        fused_result = scratch->fused;
        q_gpu = scratch->q;
        k_gpu = scratch->k;
        v_gpu = scratch->v;
        gate_mlp_gpu = scratch->gate_mlp;
        up_gpu = scratch->up;
        attn_out_gpu = scratch->attn_out;
        concat_gpu = scratch->concat;
        proj_out_gpu = scratch->proj_out;

        if (!norm_gpu || !fused_result || !q_gpu || !k_gpu || !v_gpu ||
            !gate_mlp_gpu || !up_gpu || !attn_out_gpu || !concat_gpu || !proj_out_gpu) {
            return 0;
        }
    } else {
        norm_gpu = iris_gpu_tensor_alloc_f16((size_t)seq * h_size);
        q_gpu = iris_gpu_tensor_alloc_f16((size_t)seq * h_size);
        k_gpu = iris_gpu_tensor_alloc_f16((size_t)seq * h_size);
        v_gpu = iris_gpu_tensor_alloc_f16((size_t)seq * h_size);
        gate_mlp_gpu = iris_gpu_tensor_alloc_f16((size_t)seq * mlp_hidden);
        up_gpu = iris_gpu_tensor_alloc_f16((size_t)seq * mlp_hidden);
        attn_out_gpu = iris_gpu_tensor_alloc_f16((size_t)seq * h_size);
        concat_gpu = iris_gpu_tensor_alloc_f16((size_t)seq * (h_size + mlp_hidden));

        if (!norm_gpu || !q_gpu || !k_gpu || !v_gpu ||
            !gate_mlp_gpu || !up_gpu || !attn_out_gpu || !concat_gpu) {
            goto cleanup;
        }
    }

    /* === Phase 1: AdaLN normalization (bf16) === */
    iris_gpu_adaln_norm_bf16(norm_gpu, hidden_gpu, shift_bf16, scale_bf16, seq, h_size, eps);

    /* === Phase 2: Fused QKV + MLP projection (bf16 native) === */
    if (scratch) {
        if (!iris_gpu_linear_bf16_native_into(fused_result, norm_gpu,
                                              block->qkv_mlp_weight_bf16,
                                              seq, h_size, fused_dim)) {
            return 0;
        }
    } else {
        fused_result = iris_gpu_linear_bf16_native(norm_gpu,
                                                   block->qkv_mlp_weight_bf16,
                                                   seq, h_size, fused_dim);
        if (!fused_result) goto cleanup;
    }

    /* === Phase 3: Split fused output (bf16) === */
    iris_gpu_split_qkv_mlp_bf16(fused_result, q_gpu, k_gpu, v_gpu, gate_mlp_gpu, up_gpu,
                                seq, h_size, mlp_hidden);

    /* === Phase 4: QK RMSNorm (bf16) === */
    iris_gpu_qk_rms_norm_bf16(q_gpu, k_gpu, norm_q_weight_bf16, norm_k_weight_bf16,
                              seq, heads, head_dim, eps);

    /* === Phase 5: Apply unified RoPE (bf16 tensors, f32 frequencies) === */
    iris_gpu_rope_unified_bf16(q_gpu, k_gpu,
                               txt_rope_cos, txt_rope_sin,
                               img_rope_cos, img_rope_sin,
                               seq, img_offset, heads, head_dim, axis_dim);

    /* === Phase 6: Self-attention (native bf16 fused SDPA) === */
    float attn_scale = 1.0f / sqrtf((float)head_dim);
    if (!iris_gpu_attention_fused_bf16(attn_out_gpu, q_gpu, k_gpu, v_gpu,
                                       seq, seq, heads, head_dim, attn_scale)) {
        goto cleanup;
    }

    /* === Phase 7: SwiGLU (bf16) === */
    iris_gpu_silu_mul_bf16(gate_mlp_gpu, up_gpu, seq * mlp_hidden);

    /* === Phase 8: Concat attention + MLP outputs (bf16) === */
    iris_gpu_concat_attn_mlp_bf16(attn_out_gpu, gate_mlp_gpu, concat_gpu, seq, h_size, mlp_hidden);

    /* === Phase 9: Final projection (bf16 native) === */
    if (scratch) {
        if (!iris_gpu_linear_bf16_native_into(proj_out_gpu, concat_gpu,
                                              block->proj_mlp_weight_bf16,
                                              seq, h_size + mlp_hidden, h_size)) {
            return 0;
        }
    } else {
        proj_out_gpu = iris_gpu_linear_bf16_native(concat_gpu,
                                                   block->proj_mlp_weight_bf16,
                                                   seq, h_size + mlp_hidden, h_size);
        if (!proj_out_gpu) goto cleanup;
    }

    /* === Phase 10: Gated add residual (bf16) === */
    iris_gpu_gated_add_bf16(hidden_gpu, gate_bf16, proj_out_gpu, seq, h_size);

    if (!own_tensors) return 1;

    /* Owned tensors: release intermediates before returning. */
    if (norm_gpu) iris_gpu_tensor_free(norm_gpu);
    if (fused_result) iris_gpu_tensor_free(fused_result);
    if (q_gpu) iris_gpu_tensor_free(q_gpu);
    if (k_gpu) iris_gpu_tensor_free(k_gpu);
    if (v_gpu) iris_gpu_tensor_free(v_gpu);
    if (gate_mlp_gpu) iris_gpu_tensor_free(gate_mlp_gpu);
    if (up_gpu) iris_gpu_tensor_free(up_gpu);
    if (attn_out_gpu) iris_gpu_tensor_free(attn_out_gpu);
    if (concat_gpu) iris_gpu_tensor_free(concat_gpu);
    if (proj_out_gpu) iris_gpu_tensor_free(proj_out_gpu);
    return 1;

cleanup:
    if (!own_tensors) return 0;

    if (norm_gpu) iris_gpu_tensor_free(norm_gpu);
    if (fused_result) iris_gpu_tensor_free(fused_result);
    if (q_gpu) iris_gpu_tensor_free(q_gpu);
    if (k_gpu) iris_gpu_tensor_free(k_gpu);
    if (v_gpu) iris_gpu_tensor_free(v_gpu);
    if (gate_mlp_gpu) iris_gpu_tensor_free(gate_mlp_gpu);
    if (up_gpu) iris_gpu_tensor_free(up_gpu);
    if (attn_out_gpu) iris_gpu_tensor_free(attn_out_gpu);
    if (concat_gpu) iris_gpu_tensor_free(concat_gpu);
    if (proj_out_gpu) iris_gpu_tensor_free(proj_out_gpu);

    return 0;
}

/* GPU-accelerated forward pass using bf16 precision on Metal.
 * Key optimization: runs ALL blocks (double + single) inside a single Metal
 * command buffer batch, eliminating GPU sync points between blocks. Pre-computes
 * per-step modulation vectors (shift/scale/gate) for all blocks in one pass
 * before entering the block loop. For img2img, img_seq includes reference
 * tokens but only extract_seq target tokens are returned as output. */
static float *iris_transformer_forward_bf16_flux(iris_transformer_flux_t *tf,
                                            const float *img_transposed, int img_seq,
                                            int extract_seq,
                                            const float *txt_emb, int txt_seq,
                                            const float *t_emb,
                                            const float *img_rope_cos, const float *img_rope_sin,
                                            const float *txt_rope_cos, const float *txt_rope_sin) {
#ifdef USE_METAL
    iris_metal_rope_cache_begin();
#endif
    int hidden = tf->hidden_size;
    int head_dim = tf->head_dim;
    int mlp_hidden = tf->mlp_hidden;
    int total_seq = img_seq + txt_seq;
    int channels = tf->latent_channels;

    float *output = NULL;
    float *output_nlc = NULL;

    iris_gpu_tensor_t img_in_f32 = NULL;
    iris_gpu_tensor_t txt_in_f32 = NULL;
    iris_gpu_tensor_t img_in_bf16 = NULL;
    iris_gpu_tensor_t txt_in_bf16 = NULL;
    iris_gpu_tensor_t img_hidden = NULL;
    iris_gpu_tensor_t txt_hidden = NULL;
    iris_gpu_tensor_t concat_hidden = NULL;
    iris_gpu_tensor_t img_hidden_final = NULL;
    iris_gpu_tensor_t final_norm = NULL;
    iris_gpu_tensor_t output_bf16 = NULL;
    iris_gpu_tensor_t output_f32 = NULL;

    iris_gpu_tensor_t img_shift1 = NULL, img_scale1 = NULL, img_gate1 = NULL;
    iris_gpu_tensor_t img_shift2 = NULL, img_scale2 = NULL, img_gate2 = NULL;
    iris_gpu_tensor_t txt_shift1 = NULL, txt_scale1 = NULL, txt_gate1 = NULL;
    iris_gpu_tensor_t txt_shift2 = NULL, txt_scale2 = NULL, txt_gate2 = NULL;
    iris_gpu_tensor_t single_shift = NULL, single_scale = NULL, single_gate = NULL;
    iris_gpu_tensor_t final_shift = NULL, final_scale = NULL;
    double step_start = 0.0;
    double step_time = 0.0;

    if (!tf->img_in_weight_bf16 || !tf->txt_in_weight_bf16 ||
        !tf->final_proj_weight_bf16 || !tf->adaln_single_weight ||
        !tf->adaln_double_img_weight || !tf->adaln_double_txt_weight) {
        BF16_DEBUG("[BF16] missing required weights for bf16 path\n");
        return NULL;
    }

    /* Input projections to bf16 */
    img_in_f32 = iris_gpu_tensor_create(img_transposed, img_seq * channels);
    txt_in_f32 = iris_gpu_tensor_create(txt_emb, txt_seq * tf->text_dim);
    if (!img_in_f32 || !txt_in_f32) {
        BF16_DEBUG("[BF16] failed to create input f32 GPU tensors\n");
        goto cleanup;
    }

    img_in_bf16 = iris_gpu_tensor_f32_to_bf16(img_in_f32);
    txt_in_bf16 = iris_gpu_tensor_f32_to_bf16(txt_in_f32);
    iris_gpu_tensor_free(img_in_f32);
    iris_gpu_tensor_free(txt_in_f32);
    img_in_f32 = NULL;
    txt_in_f32 = NULL;
    if (!img_in_bf16 || !txt_in_bf16) {
        BF16_DEBUG("[BF16] failed to convert inputs to bf16\n");
        goto cleanup;
    }

    img_hidden = iris_gpu_linear_bf16_native(img_in_bf16, tf->img_in_weight_bf16,
                                             img_seq, channels, hidden);
    txt_hidden = iris_gpu_linear_bf16_native(txt_in_bf16, tf->txt_in_weight_bf16,
                                             txt_seq, tf->text_dim, hidden);
    iris_gpu_tensor_free(img_in_bf16);
    iris_gpu_tensor_free(txt_in_bf16);
    img_in_bf16 = NULL;
    txt_in_bf16 = NULL;
    if (!img_hidden || !txt_hidden) {
        BF16_DEBUG("[BF16] input projection failed\n");
        goto cleanup;
    }

    /* Precompute ALL modulation parameters on CPU before starting the GPU batch.
     * All modulation only depends on t_emb_silu and fixed weight matrices,
     * not on any GPU results, so we can compute everything upfront. */
    step_start = tf_get_time_ms();
    for (int i = 0; i < hidden; i++) {
        float x = t_emb[i];
        tf->t_emb_silu[i] = x / (1.0f + expf(-x));
    }
    iris_linear_nobias(tf->double_mod_img, tf->t_emb_silu, tf->adaln_double_img_weight,
                       1, hidden, hidden * 6);
    iris_linear_nobias(tf->double_mod_txt, tf->t_emb_silu, tf->adaln_double_txt_weight,
                       1, hidden, hidden * 6);
    int mod_size = hidden * 3;
    int fused_dim = hidden * 3 + mlp_hidden * 2;
    float *mod_params = tf->work2 + total_seq * fused_dim;
    iris_linear_nobias(mod_params, tf->t_emb_silu, tf->adaln_single_weight, 1, hidden, mod_size);
    float *final_mod = tf->double_mod_img;  /* Reuse buffer (double_mod_img not needed after its tensors are created) */

    /* Pre-allocate all GPU buffers needed for the step */
    concat_hidden = iris_gpu_tensor_alloc_f16((size_t)total_seq * hidden);
    img_hidden_final = iris_gpu_tensor_alloc_f16((size_t)extract_seq * hidden);
    final_norm = iris_gpu_tensor_alloc_f16((size_t)extract_seq * hidden);
    if (!concat_hidden || !img_hidden_final || !final_norm) {
        BF16_DEBUG("[BF16] failed to allocate step buffers\n");
        goto cleanup;
    }

    single_block_bf16_scratch_t single_scratch;
    int single_scratch_ok = single_block_bf16_scratch_init(&single_scratch, total_seq, hidden, mlp_hidden);
    if (!single_scratch_ok) {
        BF16_DEBUG("[BF16] failed to allocate single-block scratch tensors\n");
        goto cleanup;
    }

    /* === MONOLITHIC GPU BATCH: all blocks + concat + slice + final in ONE command buffer ===
     * This eliminates 4 waitUntilCompleted sync points per step (was 5 batches, now 1). */
    iris_gpu_batch_begin();

    /* Convert modulation params to bf16 GPU tensors (inside batch) */
    img_shift1 = bf16_tensor_from_f32(tf->double_mod_img, hidden);
    img_scale1 = bf16_tensor_from_f32(tf->double_mod_img + hidden, hidden);
    img_gate1 = bf16_tensor_from_f32(tf->double_mod_img + hidden * 2, hidden);
    img_shift2 = bf16_tensor_from_f32(tf->double_mod_img + hidden * 3, hidden);
    img_scale2 = bf16_tensor_from_f32(tf->double_mod_img + hidden * 4, hidden);
    img_gate2 = bf16_tensor_from_f32(tf->double_mod_img + hidden * 5, hidden);

    txt_shift1 = bf16_tensor_from_f32(tf->double_mod_txt, hidden);
    txt_scale1 = bf16_tensor_from_f32(tf->double_mod_txt + hidden, hidden);
    txt_gate1 = bf16_tensor_from_f32(tf->double_mod_txt + hidden * 2, hidden);
    txt_shift2 = bf16_tensor_from_f32(tf->double_mod_txt + hidden * 3, hidden);
    txt_scale2 = bf16_tensor_from_f32(tf->double_mod_txt + hidden * 4, hidden);
    txt_gate2 = bf16_tensor_from_f32(tf->double_mod_txt + hidden * 5, hidden);

    single_shift = bf16_tensor_from_f32(mod_params, hidden);
    single_scale = bf16_tensor_from_f32(mod_params + hidden, hidden);
    single_gate = bf16_tensor_from_f32(mod_params + hidden * 2, hidden);

    if (!img_shift1 || !img_scale1 || !img_gate1 || !img_shift2 || !img_scale2 || !img_gate2 ||
        !txt_shift1 || !txt_scale1 || !txt_gate1 || !txt_shift2 || !txt_scale2 || !txt_gate2 ||
        !single_shift || !single_scale || !single_gate) {
        BF16_DEBUG("[BF16] failed to create modulation tensors\n");
        iris_gpu_batch_end();
        single_block_bf16_scratch_free(&single_scratch);
        goto cleanup;
    }

    /* Double-stream blocks */
    for (int i = 0; i < tf->num_double_layers; i++) {
        if (tf->use_mmap) {
            load_double_block_weights(&tf->double_blocks[i], tf->sf_files, tf->num_sf_files, i,
                                      tf->hidden_size, tf->mlp_hidden, tf->use_bf16);
        }
        if (!double_block_forward_bf16(img_hidden, txt_hidden,
                                       &tf->double_blocks[i],
                                       img_shift1, img_scale1, img_gate1,
                                       img_shift2, img_scale2, img_gate2,
                                       txt_shift1, txt_scale1, txt_gate1,
                                       txt_shift2, txt_scale2, txt_gate2,
                                       img_rope_cos, img_rope_sin,
                                       txt_rope_cos, txt_rope_sin,
                                       img_seq, txt_seq, tf)) {
            BF16_DEBUG("[BF16] double block %d failed\n", i);
            iris_gpu_batch_end();
            single_block_bf16_scratch_free(&single_scratch);
            goto cleanup;
        }
        if (tf->use_mmap) {
            free_double_block_weights(&tf->double_blocks[i]);
        }
        if (iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, i, tf->num_double_layers);
    }
    /* Concatenate text and image for single blocks */
    iris_gpu_concat_seq_bf16(concat_hidden, txt_hidden, img_hidden, txt_seq, img_seq, hidden);

    /* Single-stream blocks */
    for (int i = 0; i < tf->num_single_layers; i++) {
        if (tf->use_mmap) {
            load_single_block_weights(&tf->single_blocks[i], tf->sf_files, tf->num_sf_files, i,
                                      tf->hidden_size, tf->mlp_hidden, tf->use_bf16);
        }

        iris_gpu_tensor_t norm_q = bf16_tensor_from_f32(tf->single_blocks[i].norm_q_weight, head_dim);
        iris_gpu_tensor_t norm_k = bf16_tensor_from_f32(tf->single_blocks[i].norm_k_weight, head_dim);
        if (!norm_q || !norm_k ||
            !single_block_forward_bf16(concat_hidden, &tf->single_blocks[i],
                                       single_shift, single_scale, single_gate,
                                       norm_q, norm_k,
                                       img_rope_cos, img_rope_sin,
                                       txt_rope_cos, txt_rope_sin,
                                       total_seq, txt_seq, tf, &single_scratch)) {
            if (norm_q) iris_gpu_tensor_free(norm_q);
            if (norm_k) iris_gpu_tensor_free(norm_k);
            BF16_DEBUG("[BF16] single block %d failed\n", i);
            iris_gpu_batch_end();
            single_block_bf16_scratch_free(&single_scratch);
            goto cleanup;
        }
        iris_gpu_tensor_free(norm_q);
        iris_gpu_tensor_free(norm_k);

        if (tf->use_mmap) {
            free_single_block_weights(&tf->single_blocks[i]);
        }
        if (iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_SINGLE_BLOCK, i, tf->num_single_layers);
    }
    single_block_bf16_scratch_free(&single_scratch);

    /* Slice image portion for final layer */
    iris_gpu_slice_seq_bf16(img_hidden_final, concat_hidden, extract_seq, hidden, txt_seq);

    /* Final layer: compute modulation (reuse double_mod_img buffer, already consumed) */
    iris_linear_nobias(final_mod, tf->t_emb_silu, tf->final_norm_weight, 1, hidden, hidden * 2);
    final_scale = bf16_tensor_from_f32(final_mod, hidden);
    final_shift = bf16_tensor_from_f32(final_mod + hidden, hidden);
    if (!final_scale || !final_shift) {
        BF16_DEBUG("[BF16] failed to create final modulation tensors\n");
        iris_gpu_batch_end();
        goto cleanup;
    }

    iris_gpu_adaln_norm_bf16(final_norm, img_hidden_final, final_shift, final_scale,
                             extract_seq, hidden, 1e-6f);

    output_bf16 = iris_gpu_linear_bf16_native(final_norm, tf->final_proj_weight_bf16,
                                              extract_seq, hidden, channels);
    if (!output_bf16) {
        BF16_DEBUG("[BF16] final projection failed\n");
        iris_gpu_batch_end();
        goto cleanup;
    }

    output_f32 = iris_gpu_tensor_bf16_to_f32(output_bf16);
    if (!output_f32) {
        BF16_DEBUG("[BF16] failed to convert output to f32\n");
        iris_gpu_batch_end();
        goto cleanup;
    }

    iris_gpu_batch_end();

    output_nlc = (float *)malloc((size_t)extract_seq * channels * sizeof(float));
    if (!output_nlc) {
        BF16_DEBUG("[BF16] failed to allocate output_nlc\n");
        goto cleanup;
    }
    iris_gpu_tensor_read(output_f32, output_nlc);

    output = (float *)malloc((size_t)extract_seq * channels * sizeof(float));
    if (!output) {
        BF16_DEBUG("[BF16] failed to allocate output\n");
        goto cleanup;
    }
    for (int pos = 0; pos < extract_seq; pos++) {
        for (int c = 0; c < channels; c++) {
            output[c * extract_seq + pos] = output_nlc[pos * channels + c];
        }
    }
    step_time = tf_get_time_ms() - step_start;
    iris_timing_transformer_total += step_time;
    if (iris_substep_callback)
        iris_substep_callback(IRIS_SUBSTEP_FINAL_LAYER, 0, 1);

cleanup:
    if (img_in_f32) iris_gpu_tensor_free(img_in_f32);
    if (txt_in_f32) iris_gpu_tensor_free(txt_in_f32);
    if (img_in_bf16) iris_gpu_tensor_free(img_in_bf16);
    if (txt_in_bf16) iris_gpu_tensor_free(txt_in_bf16);
    if (img_hidden) iris_gpu_tensor_free(img_hidden);
    if (txt_hidden) iris_gpu_tensor_free(txt_hidden);
    if (concat_hidden) iris_gpu_tensor_free(concat_hidden);
    if (img_hidden_final) iris_gpu_tensor_free(img_hidden_final);
    if (final_norm) iris_gpu_tensor_free(final_norm);
    if (output_bf16) iris_gpu_tensor_free(output_bf16);
    if (output_f32) iris_gpu_tensor_free(output_f32);

    if (img_shift1) iris_gpu_tensor_free(img_shift1);
    if (img_scale1) iris_gpu_tensor_free(img_scale1);
    if (img_gate1) iris_gpu_tensor_free(img_gate1);
    if (img_shift2) iris_gpu_tensor_free(img_shift2);
    if (img_scale2) iris_gpu_tensor_free(img_scale2);
    if (img_gate2) iris_gpu_tensor_free(img_gate2);

    if (txt_shift1) iris_gpu_tensor_free(txt_shift1);
    if (txt_scale1) iris_gpu_tensor_free(txt_scale1);
    if (txt_gate1) iris_gpu_tensor_free(txt_gate1);
    if (txt_shift2) iris_gpu_tensor_free(txt_shift2);
    if (txt_scale2) iris_gpu_tensor_free(txt_scale2);
    if (txt_gate2) iris_gpu_tensor_free(txt_gate2);

    if (single_shift) iris_gpu_tensor_free(single_shift);
    if (single_scale) iris_gpu_tensor_free(single_scale);
    if (single_gate) iris_gpu_tensor_free(single_gate);
    if (final_shift) iris_gpu_tensor_free(final_shift);
    if (final_scale) iris_gpu_tensor_free(final_scale);

    if (output_nlc) free(output_nlc);

    return output;
}
#endif /* USE_METAL */

/* Single-stream DiT block operating on concatenated [text, image] tokens.
 * Unlike double blocks, text and image share the same self-attention and are
 * processed as one sequence. Uses a fused projection that outputs [Q, K, V,
 * gate, up] in one matmul for efficiency. Text and image portions get
 * different RoPE (text: axis 3, image: axes 1,2). After attention, the SwiGLU
 * MLP output is concatenated with the attention output before the output
 * projection -- this parallel attention+MLP design saves a serial step. */
static void single_block_forward(float *hidden, const single_block_t *block,
                                 const float *t_emb, const float *adaln_weight,
                                 const float *img_rope_cos, const float *img_rope_sin,
                                 const float *txt_rope_cos, const float *txt_rope_sin,
                                 int seq, int img_offset, iris_transformer_flux_t *tf) {
    /* seq = total_seq (txt + img)
     * img_offset = txt_seq (where image starts in the [txt, img] concatenation)
     */
    int h_size = tf->hidden_size;
    int heads = tf->num_heads;
    int head_dim = tf->head_dim;
    int mlp_hidden = tf->mlp_hidden;
    int fused_dim = h_size * 3 + mlp_hidden * 2;  /* QKV + gate + up */
    int img_seq = seq - img_offset;  /* Number of image tokens */
    float eps = 1e-6f;

    /* Compute AdaLN parameters (3: shift, scale, gate)
     * adaln_weight is [hidden*3, hidden], t_emb is [hidden]
     * FLUX applies SiLU to t_emb before the modulation projection
     */
    int mod_size = h_size * 3;
    double _t0 = prof_get_time();

    /* Apply SiLU to t_emb for modulation - use pre-allocated buffer */
    float *t_emb_silu = tf->t_emb_silu;
    for (int i = 0; i < h_size; i++) {
        float x = t_emb[i];
        t_emb_silu[i] = x / (1.0f + expf(-x));
    }

    /* Use end of work2 for mod_params (3*hidden = 9216 floats)
     * fused_out uses seq * fused_dim floats, place mod_params after */
    float *mod_params = tf->work2 + seq * fused_dim;
    iris_linear_nobias(mod_params, t_emb_silu, adaln_weight, 1, h_size, mod_size);

    float *shift = mod_params;
    float *scale = mod_params + h_size;
    float *gate = mod_params + h_size * 2;

    /* Norm */
    float *norm = tf->work1;
    apply_adaln(norm, hidden, shift, scale, seq, h_size, eps);
    double _t1 = prof_get_time();
    prof_single_adaln += _t1 - _t0;

    /* Fused QKV + FFN input projection
     * Output: [seq, fused_dim] where fused_dim = [Q, K, V, gate, up]
     * Layout per position: [3072 Q, 3072 K, 3072 V, 9216 gate, 9216 up] = 27648 total
     */
    float *fused_out = tf->work2;
    LINEAR_BF16_OR_F32(fused_out, norm, block->qkv_mlp_weight, block->qkv_mlp_weight_bf16,
                       seq, h_size, fused_dim);
    double _t2 = prof_get_time();
    prof_single_fused_matmul += _t2 - _t1;

    /* Split outputs: use pre-allocated buffers
     * Each position has [Q, K, V, gate, up] concatenated
     */
    float *q = tf->single_q;
    float *k = tf->single_k;
    float *v = tf->single_v;
    float *mlp_gate = tf->single_mlp_gate;
    float *mlp_up = tf->single_mlp_up;

    for (int s = 0; s < seq; s++) {
        float *row = fused_out + s * fused_dim;
        memcpy(q + s * h_size, row, h_size * sizeof(float));
        memcpy(k + s * h_size, row + h_size, h_size * sizeof(float));
        memcpy(v + s * h_size, row + h_size * 2, h_size * sizeof(float));
        memcpy(mlp_gate + s * mlp_hidden, row + h_size * 3, mlp_hidden * sizeof(float));
        memcpy(mlp_up + s * mlp_hidden, row + h_size * 3 + mlp_hidden, mlp_hidden * sizeof(float));
    }

    double _t3 = prof_get_time();
    prof_single_split += _t3 - _t2;

    /* Apply QK normalization */
    apply_qk_norm(q, k, block->norm_q_weight, block->norm_k_weight,
                  seq, heads, head_dim, eps);

    /* Apply RoPE: layout is [txt, img]
     * - Text portion (0 to img_offset-1): RoPE in axis 3 (L dimension)
     * - Image portion (img_offset to seq-1): 2D RoPE based on H/W positions
     */
    int axis_dim = 32;
    int txt_seq = img_offset;

    /* Text portion: apply RoPE in axis 3 (L dimension = sequence position) */
    apply_rope_2d(q, txt_rope_cos, txt_rope_sin, txt_seq, heads, head_dim, axis_dim);
    apply_rope_2d(k, txt_rope_cos, txt_rope_sin, txt_seq, heads, head_dim, axis_dim);

    /* Image portion: apply 2D RoPE starting at img_offset */
    float *img_q = q + img_offset * h_size;
    float *img_k = k + img_offset * h_size;
    apply_rope_2d(img_q, img_rope_cos, img_rope_sin, img_seq, heads, head_dim, axis_dim);
    apply_rope_2d(img_k, img_rope_cos, img_rope_sin, img_seq, heads, head_dim, axis_dim);

    double _t4 = prof_get_time();
    prof_single_qknorm_rope += _t4 - _t3;

    /* Self-attention - use pre-allocated buffer */
    float *attn_out = tf->single_attn_out;
    mha_forward(attn_out, q, k, v, seq, heads, head_dim, tf);
    double _t5 = prof_get_time();
    prof_single_attention += _t5 - _t4;

    /* SwiGLU: silu(gate) * up - fused for better performance */
    iris_silu_mul(mlp_gate, mlp_up, seq * mlp_hidden);

    double _t6 = prof_get_time();
    prof_single_swiglu += _t6 - _t5;

    /* Fused output projection: [attn_out, mlp_out] -> hidden
     * proj_mlp_weight: [hidden, hidden + mlp_hidden]
     * Use pre-allocated concat buffer
     */
    float *concat = tf->single_concat;
    for (int s = 0; s < seq; s++) {
        memcpy(concat + s * (h_size + mlp_hidden),
               attn_out + s * h_size, h_size * sizeof(float));
        memcpy(concat + s * (h_size + mlp_hidden) + h_size,
               mlp_gate + s * mlp_hidden, mlp_hidden * sizeof(float));
    }

    float *proj_out = tf->work1;
    LINEAR_BF16_OR_F32(proj_out, concat, block->proj_mlp_weight, block->proj_mlp_weight_bf16,
                       seq, h_size + mlp_hidden, h_size);

    double _t7 = prof_get_time();
    prof_single_proj_matmul += _t7 - _t6;

    /* Apply gate and add residual - use vectorized helper */
    gated_add(hidden, gate, proj_out, seq, h_size);
    double _t8 = prof_get_time();
    prof_single_gated_add += _t8 - _t7;

    /* No free - using pre-allocated buffers */
}

typedef struct {
    const float *latent;
    int h, w;
    int t_offset;
} iris_ref_t;

#ifdef IRIS_DEV
/* ========================================================================
 * GPU Forward (CUDA / ROCm / Metal)
 * ======================================================================== */

typedef struct {
    uint16_t *qkv, *proj, *ff_in, *ff_out;
    float *norm_q, *norm_k;
} dev_stream_flux_t;

typedef struct {
    uint16_t *w_in, *w_out;
    float *norm_q, *norm_k;
} dev_single_flux_t;

struct dev_flux {
    dev_stream_flux_t *streams;  /* [double_layers, 2]: text, image */
    dev_single_flux_t *single;
    uint16_t *img_in, *txt_in, *t_lin1, *t_lin2;
    uint16_t *mod_txt, *mod_img, *mod_single, *mod_final, *proj_final;
    float *vec;                 /* sinusoidal, hidden, temb, txt/img/single/final mods */
    uint16_t *vec_bf16;         /* sinusoidal and timestep MLP scratch */

    int seq;
    float *x, *out;             /* f32 residual and projected output */
    uint16_t *norm, *h, *cat;
    float *rope_cos, *rope_sin; /* [seq, head_dim/2] */
    size_t in_elems;
    float *in_f32;
    uint16_t *in_bf16;
};

/* Assemble vertical concatenations directly from the mmapped bf16 tensors. */
static uint16_t *dev_rows_flux(const uint16_t *const *parts, const size_t *counts, int n) {
    size_t total = 0;
    for (int i = 0; i < n; i++) {
        if (!parts[i]) return NULL;
        total += counts[i];
    }
    uint16_t *out = iris_dev_alloc(total * sizeof(uint16_t));
    if (!out) return NULL;
    size_t off = 0;
    for (int i = 0; i < n; i++) {
        iris_dev_upload(out + off, parts[i], counts[i] * sizeof(uint16_t));
        off += counts[i];
    }
    return out;
}

static float *dev_f32_flux(const float *src, size_t n) {
    if (!src) return NULL;
    float *out = iris_dev_alloc(n * sizeof(float));
    if (out) iris_dev_upload(out, src, n * sizeof(float));
    return out;
}

static uint16_t *dev_weight_flux(iris_transformer_flux_t *tf, const char *name, size_t n) {
    const uint16_t *p = mmap_get_bf16(tf->sf_files, tf->num_sf_files, name);
    return dev_rows_flux(&p, &n, 1);
}

static int dev_load_stream_flux(dev_stream_flux_t *s, const uint16_t *q,
                                const uint16_t *k, const uint16_t *v,
                                const uint16_t *proj, const uint16_t *gate,
                                const uint16_t *up, const uint16_t *down,
                                const float *nq, const float *nk,
                                int dim, int mlp, int hd) {
    size_t dd = (size_t)dim * dim, md = (size_t)mlp * dim;
    const uint16_t *qkv[] = {q, k, v}, *ff[] = {gate, up};
    size_t qkv_n[] = {dd, dd, dd}, ff_n[] = {md, md};
    s->qkv = dev_rows_flux(qkv, qkv_n, 3);
    s->proj = dev_rows_flux(&proj, &dd, 1);
    s->ff_in = dev_rows_flux(ff, ff_n, 2);
    s->ff_out = dev_rows_flux(&down, &md, 1);
    s->norm_q = dev_f32_flux(nq, hd);
    s->norm_k = dev_f32_flux(nk, hd);
    return s->qkv && s->proj && s->ff_in && s->ff_out && s->norm_q && s->norm_k;
}

static void dev_free_activations_flux(struct dev_flux *d) {
    iris_dev_free(d->x); d->x = NULL;
    iris_dev_free(d->out); d->out = NULL;
    iris_dev_free(d->norm); d->norm = NULL;
    iris_dev_free(d->h); d->h = NULL;
    iris_dev_free(d->cat); d->cat = NULL;
    iris_dev_free(d->rope_cos); d->rope_cos = NULL;
    iris_dev_free(d->rope_sin); d->rope_sin = NULL;
    d->seq = 0;
}

static void dev_free_flux(iris_transformer_flux_t *tf) {
    struct dev_flux *d = tf->dev;
    if (!d) return;
    if (d->streams) {
        for (int i = 0; i < 2 * tf->num_double_layers; i++) {
            dev_stream_flux_t *s = &d->streams[i];
            iris_dev_free(s->qkv);
            iris_dev_free(s->proj);
            iris_dev_free(s->ff_in);
            iris_dev_free(s->ff_out);
            iris_dev_free(s->norm_q);
            iris_dev_free(s->norm_k);
        }
        free(d->streams);
    }
    if (d->single) {
        for (int i = 0; i < tf->num_single_layers; i++) {
            iris_dev_free(d->single[i].w_in);
            iris_dev_free(d->single[i].w_out);
            iris_dev_free(d->single[i].norm_q);
            iris_dev_free(d->single[i].norm_k);
        }
        free(d->single);
    }
    iris_dev_free(d->img_in);
    iris_dev_free(d->txt_in);
    iris_dev_free(d->t_lin1);
    iris_dev_free(d->t_lin2);
    iris_dev_free(d->mod_txt);
    iris_dev_free(d->mod_img);
    iris_dev_free(d->mod_single);
    iris_dev_free(d->mod_final);
    iris_dev_free(d->proj_final);
    iris_dev_free(d->vec);
    iris_dev_free(d->vec_bf16);
    dev_free_activations_flux(d);
    iris_dev_free(d->in_f32);
    iris_dev_free(d->in_bf16);
    free(d);
    tf->dev = NULL;
}

static int dev_load_flux(iris_transformer_flux_t *tf) {
#ifdef USE_METAL
    /* The manual-SDPA override must also bypass the newer fused device path. */
    if (getenv("IRIS_USE_MANUAL_SDPA") ||
        !iris_dev_has_memory_efficient_attention()) return 0;
#endif
    struct dev_flux *d = calloc(1, sizeof(*d));
    tf->dev = d;
    if (!d) goto fail;
    size_t dim = tf->hidden_size, mlp = tf->mlp_hidden;
    int tc = tf->time_embed.sincos_dim;
    d->img_in = dev_weight_flux(tf, "x_embedder.weight", dim * tf->latent_channels);
    d->txt_in = dev_weight_flux(tf, "context_embedder.weight", dim * tf->text_dim);
    d->t_lin1 = dev_weight_flux(tf, "time_guidance_embed.timestep_embedder.linear_1.weight", dim * tc);
    d->t_lin2 = dev_weight_flux(tf, "time_guidance_embed.timestep_embedder.linear_2.weight", dim * dim);
    d->mod_txt = dev_weight_flux(tf, "double_stream_modulation_txt.linear.weight", 6 * dim * dim);
    d->mod_img = dev_weight_flux(tf, "double_stream_modulation_img.linear.weight", 6 * dim * dim);
    d->mod_single = dev_weight_flux(tf, "single_stream_modulation.linear.weight", 3 * dim * dim);
    d->mod_final = dev_weight_flux(tf, "norm_out.linear.weight", 2 * dim * dim);
    d->proj_final = dev_weight_flux(tf, "proj_out.weight", tf->latent_channels * dim);
    if (!d->img_in || !d->txt_in || !d->t_lin1 || !d->t_lin2 || !d->mod_txt ||
        !d->mod_img || !d->mod_single || !d->mod_final || !d->proj_final) goto fail;

    d->streams = calloc(2 * tf->num_double_layers, sizeof(*d->streams));
    d->single = calloc(tf->num_single_layers, sizeof(*d->single));
    if ((!d->streams && tf->num_double_layers) || (!d->single && tf->num_single_layers)) goto fail;
    for (int i = 0; i < tf->num_double_layers; i++) {
        double_block_t b = {0};
        load_double_block_weights(&b, tf->sf_files, tf->num_sf_files, i, dim, mlp, 1);
        int ok = dev_load_stream_flux(&d->streams[2*i],
            b.txt_q_weight_bf16, b.txt_k_weight_bf16, b.txt_v_weight_bf16,
            b.txt_proj_weight_bf16, b.txt_mlp_gate_weight_bf16, b.txt_mlp_up_weight_bf16,
            b.txt_mlp_down_weight_bf16, b.txt_norm_q_weight, b.txt_norm_k_weight,
            dim, mlp, tf->head_dim);
        if (ok) ok = dev_load_stream_flux(&d->streams[2*i+1],
            b.img_q_weight_bf16, b.img_k_weight_bf16, b.img_v_weight_bf16,
            b.img_proj_weight_bf16, b.img_mlp_gate_weight_bf16, b.img_mlp_up_weight_bf16,
            b.img_mlp_down_weight_bf16, b.img_norm_q_weight, b.img_norm_k_weight,
            dim, mlp, tf->head_dim);
        free_double_block_weights(&b);
        if (!ok) goto fail;
    }
    for (int i = 0; i < tf->num_single_layers; i++) {
        single_block_t b = {0};
        load_single_block_weights(&b, tf->sf_files, tf->num_sf_files, i, dim, mlp, 1);
        dev_single_flux_t *s = &d->single[i];
        const uint16_t *in = b.qkv_mlp_weight_bf16, *out = b.proj_mlp_weight_bf16;
        size_t ni = (3 * dim + 2 * mlp) * dim, no = dim * (dim + mlp);
        s->w_in = dev_rows_flux(&in, &ni, 1);
        s->w_out = dev_rows_flux(&out, &no, 1);
        s->norm_q = dev_f32_flux(b.norm_q_weight, tf->head_dim);
        s->norm_k = dev_f32_flux(b.norm_k_weight, tf->head_dim);
        free_single_block_weights(&b);
        if (!s->w_in || !s->w_out || !s->norm_q || !s->norm_k) goto fail;
    }
    d->vec = iris_dev_alloc((tc + 19 * dim) * sizeof(float));
    d->vec_bf16 = iris_dev_alloc((tc + dim) * sizeof(uint16_t));
    if (!d->vec || !d->vec_bf16) goto fail;
    iris_dev_sync();
    return 1;
fail:
    fprintf(stderr, "FLUX.2: GPU weight upload failed, using the CPU\n");
    dev_free_flux(tf);
    return 0;
}

static int dev_ensure_flux(iris_transformer_flux_t *tf, int seq, size_t in_elems) {
    struct dev_flux *d = tf->dev;
    size_t dim = tf->hidden_size, mlp = tf->mlp_hidden, half = tf->head_dim / 2;
    if (seq > d->seq) {
        dev_free_activations_flux(d);
        d->x = iris_dev_alloc((size_t)seq * dim * sizeof(float));
        size_t out_dim = dim > (size_t)tf->latent_channels ? dim : (size_t)tf->latent_channels;
        d->out = iris_dev_alloc((size_t)seq * out_dim * sizeof(float));
        d->norm = iris_dev_alloc((size_t)seq * dim * sizeof(uint16_t));
        d->h = iris_dev_alloc((size_t)seq * (3 * dim + 2 * mlp) * sizeof(uint16_t));
        d->cat = iris_dev_alloc((size_t)seq * (dim + mlp) * sizeof(uint16_t));
        d->rope_cos = iris_dev_alloc((size_t)seq * half * sizeof(float));
        d->rope_sin = iris_dev_alloc((size_t)seq * half * sizeof(float));
        if (!d->x || !d->out || !d->norm || !d->h || !d->cat || !d->rope_cos || !d->rope_sin) return 0;
        d->seq = seq;
    }
    if (in_elems > d->in_elems) {
        iris_dev_free(d->in_f32);
        iris_dev_free(d->in_bf16);
        d->in_f32 = iris_dev_alloc(in_elems * sizeof(float));
        d->in_bf16 = iris_dev_alloc(in_elems * sizeof(uint16_t));
        if (!d->in_f32 || !d->in_bf16) return 0;
        d->in_elems = in_elems;
    }
    return 1;
}

static void dev_embed_flux(iris_transformer_flux_t *tf, float *out, const float *in,
                            const uint16_t *w, int seq, int channels) {
    struct dev_flux *d = tf->dev;
    size_t n = (size_t)seq * channels;
    iris_dev_upload(d->in_f32, in, n * sizeof(float));
    iris_dev_f32_to_bf16(d->in_bf16, d->in_f32, n);
    iris_dev_linear_f32(out, d->in_bf16, w, seq, channels, tf->hidden_size);
}

/* All three public entry points share the same [TEXT, IMAGE, REFS] layout. */
static float *forward_dev_flux(iris_transformer_flux_t *tf,
                               const float *img, int img_h, int img_w,
                               const iris_ref_t *refs, int num_refs,
                               const float *txt, int txt_seq, float timestep) {
    struct dev_flux *d = tf->dev;
    int dim = tf->hidden_size, mlp = tf->mlp_hidden, hd = tf->head_dim, half = hd / 2;
    int channels = tf->latent_channels, img_seq = img_h * img_w, images = img_seq;
    for (int r = 0; r < num_refs; r++) images += refs[r].h * refs[r].w;
    int seq = txt_seq + images;
    size_t in_elems = (size_t)images * channels;
    if (in_elems < (size_t)txt_seq * tf->text_dim) in_elems = (size_t)txt_seq * tf->text_dim;
    if (!dev_ensure_flux(tf, seq, in_elems)) goto fail;

    /* Host layout conversion and the exact CPU RoPE tables, compacted to pairs. */
    float *tokens = malloc((size_t)images * channels * sizeof(float));
    float *cos = malloc((size_t)seq * hd * sizeof(float));
    float *sin = malloc((size_t)seq * hd * sizeof(float));
    float *output = malloc((size_t)img_seq * channels * sizeof(float));
    if (!tokens || !cos || !sin || !output) {
        free(tokens); free(cos); free(sin); free(output);
        goto fail;
    }
    compute_rope_text(cos, sin, txt_seq, tf->axis_dim, tf->rope_theta);
    compute_rope_2d(cos + (size_t)txt_seq * hd, sin + (size_t)txt_seq * hd,
                    img_h, img_w, tf->axis_dim, tf->rope_theta);
    int off = 0;
    for (int r = -1; r < num_refs; r++) {
        const float *src = r < 0 ? img : refs[r].latent;
        int n = r < 0 ? img_seq : refs[r].h * refs[r].w;
        for (int s = 0; s < n; s++)
            for (int c = 0; c < channels; c++)
                tokens[(size_t)(off + s) * channels + c] = src[(size_t)c * n + s];
        if (r >= 0)
            compute_rope_2d_with_t_offset(cos + (size_t)(txt_seq + off) * hd,
                sin + (size_t)(txt_seq + off) * hd, refs[r].h, refs[r].w,
                tf->axis_dim, tf->rope_theta, refs[r].t_offset);
        off += n;
    }
    for (size_t i = 0; i < (size_t)seq * half; i++) {
        cos[i] = cos[2*i];
        sin[i] = sin[2*i];
    }
    iris_dev_upload(d->rope_cos, cos, (size_t)seq * half * sizeof(float));
    iris_dev_upload(d->rope_sin, sin, (size_t)seq * half * sizeof(float));
    free(cos); free(sin);
    dev_embed_flux(tf, d->x, txt, d->txt_in, txt_seq, tf->text_dim);
    dev_embed_flux(tf, d->x + (size_t)txt_seq * dim, tokens, d->img_in, images, channels);

    int tc = tf->time_embed.sincos_dim;
    float sin_emb[256];
    get_timestep_embedding(sin_emb, timestep * 1000.0f, tc, 10000.0f);
    iris_dev_upload(d->vec, sin_emb, tc * sizeof(float));
    iris_dev_f32_to_bf16(d->vec_bf16, d->vec, tc);
    float *hidden = d->vec + tc, *temb = hidden + dim;
    float *mod_txt = temb + dim, *mod_img = mod_txt + 6 * dim;
    float *mod_single = mod_img + 6 * dim, *fin = mod_single + 3 * dim;
    uint16_t *act = d->vec_bf16 + tc;
    iris_dev_linear_f32(hidden, d->vec_bf16, d->t_lin1, 1, tc, dim);
    iris_dev_silu(hidden, dim);
    iris_dev_f32_to_bf16(act, hidden, dim);
    iris_dev_linear_f32(temb, act, d->t_lin2, 1, dim, dim);
    iris_dev_silu(temb, dim);
    iris_dev_f32_to_bf16(act, temb, dim);
    iris_dev_linear_f32(mod_txt, act, d->mod_txt, 1, dim, 6 * dim);
    iris_dev_linear_f32(mod_img, act, d->mod_img, 1, dim, 6 * dim);
    iris_dev_linear_f32(mod_single, act, d->mod_single, 1, dim, 3 * dim);
    iris_dev_linear_f32(fin, act, d->mod_final, 1, dim, 2 * dim);
    iris_dev_sync();
    double start = tf_get_time_ms();

    for (int l = 0; l < tf->num_double_layers; l++) {
        /* Two QKV GEMMs write contiguous rows of the joint attention input. */
        for (int m = 0; m < 2; m++) {
            dev_stream_flux_t *b = &d->streams[2*l + m];
            int n = m ? images : txt_seq, first = m ? txt_seq : 0;
            size_t xoff = (size_t)first * dim;
            float *mod = m ? mod_img : mod_txt;
            uint16_t *qkv = d->h + 3 * xoff;
            iris_dev_layer_norm_mod(d->norm, d->x + xoff, mod, mod + dim, n, dim, 1e-6f);
            iris_dev_linear_bf16(qkv, d->norm, b->qkv, n, dim, 3 * dim);
            iris_dev_head_norm_rope(qkv, 3 * dim, b->norm_q,
                d->rope_cos + (size_t)first * half, d->rope_sin + (size_t)first * half,
                n, tf->num_heads, hd, 1e-6f, 0);
            iris_dev_head_norm_rope(qkv + dim, 3 * dim, b->norm_k,
                d->rope_cos + (size_t)first * half, d->rope_sin + (size_t)first * half,
                n, tf->num_heads, hd, 1e-6f, 0);
        }
        iris_dev_attention(d->cat, dim, d->h, d->h + dim, d->h + 2 * dim, 3 * dim,
            seq, seq, tf->num_heads, tf->num_heads, hd, 1.0f / sqrtf(hd), 0, NULL);
        /* Attention residual must precede the separately normalized FFN. */
        for (int m = 0; m < 2; m++) {
            dev_stream_flux_t *b = &d->streams[2*l + m];
            int n = m ? images : txt_seq, first = m ? txt_seq : 0;
            size_t xoff = (size_t)first * dim;
            float *mod = m ? mod_img : mod_txt;
            iris_dev_linear_f32(d->out, d->cat + xoff, b->proj, n, dim, dim);
            iris_dev_residual(d->x + xoff, d->out, mod + 2 * dim, n, dim);
        }
        for (int m = 0; m < 2; m++) {
            dev_stream_flux_t *b = &d->streams[2*l + m];
            int n = m ? images : txt_seq, first = m ? txt_seq : 0;
            float *x = d->x + (size_t)first * dim, *mod = m ? mod_img : mod_txt;
            iris_dev_layer_norm_mod(d->norm, x, mod + 3 * dim, mod + 4 * dim, n, dim, 1e-6f);
            iris_dev_linear_bf16(d->h, d->norm, b->ff_in, n, dim, 2 * mlp);
            iris_dev_swiglu(d->cat, mlp, d->h, d->h + mlp, 2 * mlp, n, mlp);
            iris_dev_linear_f32(d->out, d->cat, b->ff_out, n, mlp, dim);
            iris_dev_residual(x, d->out, mod + 5 * dim, n, dim);
        }
        if (iris_substep_callback) iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, l, tf->num_double_layers);
    }
    iris_dev_sync();
    double double_time = tf_get_time_ms() - start;
    start = tf_get_time_ms();
    for (int l = 0; l < tf->num_single_layers; l++) {
        dev_single_flux_t *b = &d->single[l];
        int ldh = 3 * dim + 2 * mlp, ldc = dim + mlp;
        iris_dev_layer_norm_mod(d->norm, d->x, mod_single, mod_single + dim, seq, dim, 1e-6f);
        iris_dev_linear_bf16(d->h, d->norm, b->w_in, seq, dim, ldh);
        iris_dev_head_norm_rope(d->h, ldh, b->norm_q, d->rope_cos, d->rope_sin,
            seq, tf->num_heads, hd, 1e-6f, 0);
        iris_dev_head_norm_rope(d->h + dim, ldh, b->norm_k, d->rope_cos, d->rope_sin,
            seq, tf->num_heads, hd, 1e-6f, 0);
        iris_dev_attention(d->cat, ldc, d->h, d->h + dim, d->h + 2 * dim, ldh,
            seq, seq, tf->num_heads, tf->num_heads, hd, 1.0f / sqrtf(hd), 0, NULL);
        iris_dev_swiglu(d->cat + dim, ldc, d->h + 3 * dim, d->h + 3 * dim + mlp, ldh, seq, mlp);
        iris_dev_linear_f32(d->out, d->cat, b->w_out, seq, ldc, dim);
        iris_dev_residual(d->x, d->out, mod_single + 2 * dim, seq, dim);
        if (iris_substep_callback) iris_substep_callback(IRIS_SUBSTEP_SINGLE_BLOCK, l, tf->num_single_layers);
    }
    iris_dev_sync();
    double single_time = tf_get_time_ms() - start;
    start = tf_get_time_ms();
    /* Final modulation is (scale, shift); only target image rows reach output. */
    iris_dev_layer_norm_mod(d->norm, d->x + (size_t)txt_seq * dim, fin + dim, fin, img_seq, dim, 1e-6f);
    iris_dev_linear_f32(d->out, d->norm, d->proj_final, img_seq, dim, channels);
    iris_dev_download(tokens, d->out, (size_t)img_seq * channels * sizeof(float));
    for (int s = 0; s < img_seq; s++)
        for (int c = 0; c < channels; c++)
            output[(size_t)c * img_seq + s] = tokens[(size_t)s * channels + c];
    free(tokens);
    double final_time = tf_get_time_ms() - start;
    iris_timing_transformer_double += double_time;
    iris_timing_transformer_single += single_time;
    iris_timing_transformer_final += final_time;
    iris_timing_transformer_total += double_time + single_time + final_time;
    if (iris_substep_callback) iris_substep_callback(IRIS_SUBSTEP_FINAL_LAYER, 0, 1);
    return output;
fail:
    fprintf(stderr, "FLUX.2: GPU workspace allocation failed, using the CPU\n");
    dev_free_flux(tf);
    return NULL;
}
#endif /* IRIS_DEV */

/* ========================================================================
 * Full Transformer Forward Pass
 * ======================================================================== */

/* Top-level transformer entry point for one denoising step. Transposes input
 * from NCHW to NLC layout, computes timestep embedding, then runs double
 * blocks (joint img/txt attention) followed by single blocks (fused stream).
 * Routes to the GPU bf16 path when Metal is available. RoPE tables are cached
 * across calls since they only depend on image dimensions (not timestep). */
float *iris_transformer_forward_flux(iris_transformer_flux_t *tf,
                                const float *img_latent, int img_h, int img_w,
                                const float *txt_emb, int txt_seq,
                                float timestep) {
#ifdef IRIS_DEV
    if (tf->dev) {
        float *out = forward_dev_flux(tf, img_latent, img_h, img_w, NULL, 0,
                                       txt_emb, txt_seq, timestep);
        if (out) return out;
    }
#endif
    int hidden = tf->hidden_size;
    int img_seq = img_h * img_w;

    /* Ensure work buffers are sized for actual sequence length */
    int total_seq = img_seq + txt_seq;
    if (ensure_work_buffers(tf, total_seq) < 0) {
        fprintf(stderr, "Failed to allocate work buffers for seq_len=%d\n", total_seq);
        return NULL;
    }
    if (ensure_attn_scores(tf, img_seq, txt_seq) < 0) {
        fprintf(stderr, "Failed to allocate attention scores buffer\n");
        return NULL;
    }

    /* Get timestep embedding
     * FLUX.2-klein uses 256-dim sinusoidal (128 frequencies), not hidden_size
     * Use t_emb_silu as work buffer (it's not used until double blocks)
     */
    int sincos_dim = tf->time_embed.sincos_dim;
    float *t_emb = (float *)malloc(hidden * sizeof(float));
    float t_sincos[256];  /* sincos_dim is always 256 */
    get_timestep_embedding(t_sincos, timestep * 1000.0f, sincos_dim, 10000.0f);
    time_embed_forward(t_emb, t_sincos, &tf->time_embed, hidden, tf->t_emb_silu);

    /* Get cached 2D RoPE frequencies for image tokens */
    float *img_rope_cos, *img_rope_sin;
    get_cached_img_rope(tf, img_h, img_w, &img_rope_cos, &img_rope_sin);

    /* Get cached text RoPE frequencies */
    float *txt_rope_cos, *txt_rope_sin;
    get_cached_txt_rope(tf, txt_seq, &txt_rope_cos, &txt_rope_sin);

    /* Transpose input from NCHW [channels, h, w] to NLC [seq, channels] format
     * Input: img_latent[c * img_seq + pos] for channel c at position pos
     * Output: transposed[pos * channels + c]
     */
    int channels = tf->latent_channels;
    float *img_transposed = (float *)malloc(img_seq * channels * sizeof(float));
    for (int pos = 0; pos < img_seq; pos++) {
        for (int c = 0; c < channels; c++) {
            img_transposed[pos * channels + c] = img_latent[c * img_seq + pos];
        }
    }

#ifdef USE_METAL
    /* With direct mmap pointers, the bf16 pipeline now works correctly in mmap mode.
     * Cache entries are stable (pointers point into mmap region) so no collision. */
    if (iris_metal_available() && iris_bf16_pipeline_available() && tf->use_bf16) {
        float *bf16_output = iris_transformer_forward_bf16_flux(tf, img_transposed, img_seq,
                                                           img_seq, /* extract_seq = img_seq for txt2img */
                                                           txt_emb, txt_seq, t_emb,
                                                           img_rope_cos, img_rope_sin,
                                                           txt_rope_cos, txt_rope_sin);
        if (bf16_output) {
            free(img_transposed);
            free(t_emb);
            /* RoPE buffers are cached in transformer struct - don't free */
            return bf16_output;
        } else {
            BF16_DEBUG("[BF16] bf16 pipeline failed, falling back\n");
        }
    } else {
        BF16_DEBUG("[BF16] gate metal=%d bf16_pipeline=%d use_mmap=%d use_bf16=%d\n",
                   iris_metal_available(), iris_bf16_pipeline_available(),
                   tf->use_mmap, tf->use_bf16);
    }
#endif

    /* Project image latent to hidden */
    float *img_hidden = tf->img_hidden;
    LINEAR_BF16_OR_F32(img_hidden, img_transposed, tf->img_in_weight, tf->img_in_weight_bf16,
                       img_seq, tf->latent_channels, hidden);
    free(img_transposed);

    /* Project text embeddings to hidden */
    float *txt_hidden = tf->txt_hidden;
    LINEAR_BF16_OR_F32(txt_hidden, txt_emb, tf->txt_in_weight, tf->txt_in_weight_bf16,
                       txt_seq, tf->text_dim, hidden);

#ifdef DEBUG_TRANSFORMER
    /* Debug: print intermediate values for comparison with Python */
    fprintf(stderr, "\n[DEBUG] t_emb first 10: ");
    for (int i = 0; i < 10; i++) fprintf(stderr, "%.6f ", t_emb[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "[DEBUG] img_hidden[0,0,:5] (img_proj): ");
    for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", img_hidden[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "[DEBUG] txt_hidden[0,0,:5] (txt_proj): ");
    for (int i = 0; i < 5; i++) fprintf(stderr, "%.6f ", txt_hidden[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "[DEBUG] img_rope_cos[0, :10]: ");
    for (int i = 0; i < 10; i++) fprintf(stderr, "%.6f ", img_rope_cos[i]);
    fprintf(stderr, "\n");

    fprintf(stderr, "[DEBUG] txt_rope_cos[10, 96:106]: ");
    for (int i = 96; i < 106; i++) fprintf(stderr, "%.6f ", txt_rope_cos[10 * head_dim + i]);
    fprintf(stderr, "\n");
#endif

    /* Double-stream blocks */
    double double_start = tf_get_time_ms();

    /* Pre-compute AdaLN modulation ONCE for all 5 double blocks.
     * t_emb and adaln weights are the same for all blocks within a step. */
    int double_mod_size = hidden * 6;  /* shift1, scale1, gate1, shift2, scale2, gate2 */
    for (int j = 0; j < hidden; j++) {
        float x = t_emb[j];
        tf->t_emb_silu[j] = x / (1.0f + expf(-x));
    }
    iris_linear_nobias(tf->double_mod_img, tf->t_emb_silu, tf->adaln_double_img_weight,
                       1, hidden, double_mod_size);
    iris_linear_nobias(tf->double_mod_txt, tf->t_emb_silu, tf->adaln_double_txt_weight,
                       1, hidden, double_mod_size);

    for (int i = 0; i < tf->num_double_layers; i++) {
        /* In mmap mode, load block weights on-demand and free after use */
        if (tf->use_mmap && tf->double_blocks[i].img_q_weight == NULL
                         && tf->double_blocks[i].img_q_weight_bf16 == NULL) {
            load_double_block_weights(&tf->double_blocks[i], tf->sf_files, tf->num_sf_files, i,
                                      tf->hidden_size, tf->mlp_hidden, tf->use_bf16);
        }
        double_block_forward(img_hidden, txt_hidden,
                             &tf->double_blocks[i],
                             tf->double_mod_img, tf->double_mod_txt,
                             img_rope_cos, img_rope_sin,
                             txt_rope_cos, txt_rope_sin,
                             img_seq, txt_seq, tf);
        if (tf->use_mmap) free_double_block_weights(&tf->double_blocks[i]);
        if (iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, i, tf->num_double_layers);
#ifdef DEBUG_TRANSFORMER
        if (i == 0) {
            fprintf(stderr, "\n[DEBUG] After double block 0:\n");
            fprintf(stderr, "[DEBUG] img_hidden[0,0,:10]: ");
            for (int d = 0; d < 10; d++) fprintf(stderr, "%.6f ", img_hidden[d]);
            fprintf(stderr, "\n");
            float sum = 0, sum_sq = 0;
            for (int d = 0; d < img_seq * hidden; d++) {
                sum += img_hidden[d];
                sum_sq += img_hidden[d] * img_hidden[d];
            }
            float mean = sum / (img_seq * hidden);
            float std = sqrtf(sum_sq / (img_seq * hidden) - mean * mean);
            fprintf(stderr, "[DEBUG] img_hidden mean=%.6f, std=%.6f\n", mean, std);
        }
#endif
    }

    double double_time = tf_get_time_ms() - double_start;

    /* Concatenate text and image for single-stream blocks
     * Python uses [txt, img] order for concatenation
     */
    float *concat_hidden = (float *)malloc(total_seq * hidden * sizeof(float));
    memcpy(concat_hidden, txt_hidden, txt_seq * hidden * sizeof(float));
    memcpy(concat_hidden + txt_seq * hidden, img_hidden,
           img_seq * hidden * sizeof(float));

    /* Single-stream blocks */
    double single_start = tf_get_time_ms();

#ifdef USE_METAL
    /* Try BF16 native path first */
    int bf16_path_ok = 0;
    int gpu_chained_ok = 0;
    iris_gpu_tensor_t concat_hidden_gpu = NULL;

    /* BF16 native single block path - currently disabled.
     * MPS GEMM doesn't support bf16, so our custom bf16 linear kernel is ~4x slower than
     * MPS SGEMM with f16 weights. To achieve higher bf16 performance, we would need
     * highly optimized custom Metal matmul kernels.
     * The f32 path with f16 weights and pre-warmed caches is currently faster. */
    if (0 && iris_metal_available() && iris_bf16_pipeline_available() && !tf->use_mmap && tf->use_bf16) {
        /* Create f32 GPU tensor first, then convert to bf16 */
        iris_gpu_tensor_t hidden_f32 = iris_gpu_tensor_create(concat_hidden, total_seq * hidden);
        if (hidden_f32) {
            iris_gpu_tensor_t hidden_bf16 = iris_gpu_tensor_f32_to_bf16(hidden_f32);
            iris_gpu_tensor_free(hidden_f32);

            if (hidden_bf16) {
                iris_gpu_tensor_set_persistent(hidden_bf16, 1);
                bf16_path_ok = 1;

                /* Pre-compute AdaLN modulation and convert to bf16 */
                int mod_size = hidden * 3;
                for (int j = 0; j < hidden; j++) {
                    float x = t_emb[j];
                    tf->t_emb_silu[j] = x / (1.0f + expf(-x));
                }
                int fused_dim = hidden * 3 + tf->mlp_hidden * 2;
                float *mod_params = tf->work2 + total_seq * fused_dim;
                iris_linear_nobias(mod_params, tf->t_emb_silu, tf->adaln_single_weight, 1, hidden, mod_size);

                /* Convert modulation to bf16 GPU tensors */
                iris_gpu_tensor_t shift_f32 = iris_gpu_tensor_create(mod_params, hidden);
                iris_gpu_tensor_t scale_f32 = iris_gpu_tensor_create(mod_params + hidden, hidden);
                iris_gpu_tensor_t gate_f32 = iris_gpu_tensor_create(mod_params + hidden * 2, hidden);

                iris_gpu_tensor_t shift_bf16 = shift_f32 ? iris_gpu_tensor_f32_to_bf16(shift_f32) : NULL;
                iris_gpu_tensor_t scale_bf16 = scale_f32 ? iris_gpu_tensor_f32_to_bf16(scale_f32) : NULL;
                iris_gpu_tensor_t gate_bf16 = gate_f32 ? iris_gpu_tensor_f32_to_bf16(gate_f32) : NULL;

                if (shift_f32) iris_gpu_tensor_free(shift_f32);
                if (scale_f32) iris_gpu_tensor_free(scale_f32);
                if (gate_f32) iris_gpu_tensor_free(gate_f32);

                if (shift_bf16 && scale_bf16 && gate_bf16) {
                    iris_gpu_batch_begin();

                    /* Process all single blocks in bf16 */
                    for (int i = 0; i < tf->num_single_layers && bf16_path_ok; i++) {
                        /* Convert norm weights to bf16 tensors for this block */
                        iris_gpu_tensor_t norm_q_f32 = iris_gpu_tensor_create(
                            tf->single_blocks[i].norm_q_weight, tf->head_dim);
                        iris_gpu_tensor_t norm_k_f32 = iris_gpu_tensor_create(
                            tf->single_blocks[i].norm_k_weight, tf->head_dim);
                        iris_gpu_tensor_t norm_q_bf16 = norm_q_f32 ? iris_gpu_tensor_f32_to_bf16(norm_q_f32) : NULL;
                        iris_gpu_tensor_t norm_k_bf16 = norm_k_f32 ? iris_gpu_tensor_f32_to_bf16(norm_k_f32) : NULL;

                        if (norm_q_f32) iris_gpu_tensor_free(norm_q_f32);
                        if (norm_k_f32) iris_gpu_tensor_free(norm_k_f32);

                        if (!norm_q_bf16 || !norm_k_bf16 ||
                            !single_block_forward_bf16(hidden_bf16, &tf->single_blocks[i],
                                                       shift_bf16, scale_bf16, gate_bf16,
                                                       norm_q_bf16, norm_k_bf16,
                                                       img_rope_cos, img_rope_sin,
                                                       txt_rope_cos, txt_rope_sin,
                                                       total_seq, txt_seq, tf, NULL)) {
                            bf16_path_ok = 0;
                            iris_gpu_batch_end();
                            /* Convert back to f32 for fallback */
                            iris_gpu_tensor_t result_f32 = iris_gpu_tensor_bf16_to_f32(hidden_bf16);
                            if (result_f32) {
                                iris_gpu_tensor_read(result_f32, concat_hidden);
                                iris_gpu_tensor_free(result_f32);
                            }
                        }

                        if (norm_q_bf16) iris_gpu_tensor_free(norm_q_bf16);
                        if (norm_k_bf16) iris_gpu_tensor_free(norm_k_bf16);

                        if (iris_substep_callback)
                            iris_substep_callback(IRIS_SUBSTEP_SINGLE_BLOCK, i, tf->num_single_layers);
                    }

                    if (bf16_path_ok) {
                        iris_gpu_batch_end();
                        /* Convert final result back to f32 */
                        iris_gpu_tensor_t result_f32 = iris_gpu_tensor_bf16_to_f32(hidden_bf16);
                        if (result_f32) {
                            iris_gpu_tensor_read(result_f32, concat_hidden);
                            iris_gpu_tensor_free(result_f32);
                        }
                    }
                } else {
                    bf16_path_ok = 0;
                }

                if (shift_bf16) iris_gpu_tensor_free(shift_bf16);
                if (scale_bf16) iris_gpu_tensor_free(scale_bf16);
                if (gate_bf16) iris_gpu_tensor_free(gate_bf16);
                iris_gpu_tensor_free(hidden_bf16);
            }
        }
    }

    /* Fall back to f32 GPU-chained path if bf16 path not used or failed */
    if (!bf16_path_ok && iris_metal_available() && iris_metal_shaders_available() && !tf->use_mmap) {
        /* Create persistent GPU tensor for hidden state */
        concat_hidden_gpu = iris_gpu_tensor_create(concat_hidden, total_seq * hidden);
        if (concat_hidden_gpu) {
            iris_gpu_tensor_set_persistent(concat_hidden_gpu, 1);
            gpu_chained_ok = 1;

            /* Pre-compute AdaLN modulation ONCE for all 20 single blocks.
             * t_emb and adaln_single_weight are the same for all blocks within a step,
             * so the output (shift, scale, gate) is identical. Computing this 20x was wasteful. */
            int mod_size = hidden * 3;
            for (int j = 0; j < hidden; j++) {
                float x = t_emb[j];
                tf->t_emb_silu[j] = x / (1.0f + expf(-x));
            }
            int fused_dim = hidden * 3 + tf->mlp_hidden * 2;
            float *mod_params = tf->work2 + total_seq * fused_dim;  /* Place after fused_out buffer */
            iris_linear_nobias(mod_params, tf->t_emb_silu, tf->adaln_single_weight, 1, hidden, mod_size);
            float *precomputed_shift = mod_params;
            float *precomputed_scale = mod_params + hidden;
            float *precomputed_gate = mod_params + hidden * 2;

            /* Start batch mode OUTSIDE the loop so all 20 blocks share the same
             * command buffer. This eliminates the sync between blocks. */
            iris_gpu_batch_begin();

            /* Allocate scratch buffers once for all 20 single blocks.
             * This eliminates 200 GPU tensor alloc/frees per denoising step. */
            single_block_gpu_scratch_t scratch;
            int scratch_ok = single_block_gpu_scratch_init(&scratch, total_seq, hidden, tf->mlp_hidden);

            /* Process all single blocks with GPU tensor chaining */
            for (int i = 0; i < tf->num_single_layers && gpu_chained_ok; i++) {
                if (!single_block_forward_gpu_chained(concat_hidden_gpu, &tf->single_blocks[i],
                                                      precomputed_shift, precomputed_scale, precomputed_gate,
                                                      img_rope_cos, img_rope_sin,
                                                      txt_rope_cos, txt_rope_sin,
                                                      total_seq, txt_seq, tf,
                                                      scratch_ok ? &scratch : NULL)) {
                    /* GPU chained path failed, need to fall back */
                    gpu_chained_ok = 0;
                    /* End batch mode before falling back */
                    iris_gpu_batch_end();
                    /* Read current state back to CPU to continue with CPU/old GPU path */
                    iris_gpu_tensor_read(concat_hidden_gpu, concat_hidden);
                }
                if (iris_substep_callback)
                    iris_substep_callback(IRIS_SUBSTEP_SINGLE_BLOCK, i, tf->num_single_layers);
            }

            /* Free scratch buffers */
            if (scratch_ok) single_block_gpu_scratch_free(&scratch);

            if (gpu_chained_ok) {
                /* End batch mode - commits all GPU work and waits for completion */
                iris_gpu_batch_end();
                /* Read final result back to CPU */
                iris_gpu_tensor_read(concat_hidden_gpu, concat_hidden);
            }

            iris_gpu_tensor_free(concat_hidden_gpu);
            concat_hidden_gpu = NULL;
        }
    }

    /* Fall back to per-block GPU/CPU path if both bf16 and f32 chained paths failed */
    if (!bf16_path_ok && !gpu_chained_ok) {
#endif
        for (int i = 0; i < tf->num_single_layers; i++) {
            /* In mmap mode, load block weights on-demand and free after use */
            if (tf->use_mmap && tf->single_blocks[i].qkv_mlp_weight == NULL
                             && tf->single_blocks[i].qkv_mlp_weight_bf16 == NULL) {
                load_single_block_weights(&tf->single_blocks[i], tf->sf_files, tf->num_sf_files, i,
                                          tf->hidden_size, tf->mlp_hidden, tf->use_bf16);
            }
#ifdef USE_METAL
            /* Try GPU-optimized path first */
            if (!single_block_forward_gpu(concat_hidden, &tf->single_blocks[i],
                                          t_emb, tf->adaln_single_weight,
                                          img_rope_cos, img_rope_sin,
                                          txt_rope_cos, txt_rope_sin,
                                          total_seq, txt_seq, tf))
#endif
            {
                /* Fall back to CPU path */
                single_block_forward(concat_hidden, &tf->single_blocks[i],
                                     t_emb, tf->adaln_single_weight,
                                     img_rope_cos, img_rope_sin,
                                     txt_rope_cos, txt_rope_sin,
                                     total_seq, txt_seq, tf);  /* txt_seq is the offset to image */
            }
            if (tf->use_mmap) free_single_block_weights(&tf->single_blocks[i]);
            if (iris_substep_callback)
                iris_substep_callback(IRIS_SUBSTEP_SINGLE_BLOCK, i, tf->num_single_layers);

#ifdef DEBUG_SINGLE_BLOCK
            if (i == 0 || i == 9 || i == 19) {
                /* Print image portion (starts at txt_seq offset) */
                float *img_part = concat_hidden + txt_seq * hidden;
                fprintf(stderr, "[SGL%d] img_hidden[0,0,:5]: ", i);
                for (int d = 0; d < 5; d++) fprintf(stderr, "%.6f ", img_part[d]);
                fprintf(stderr, "\n");
            }
#endif
        }
#ifdef USE_METAL
    }
#endif

    double single_time = tf_get_time_ms() - single_start;

    /* Extract image hidden states (image is after text) */
    memcpy(img_hidden, concat_hidden + txt_seq * hidden, img_seq * hidden * sizeof(float));
    free(concat_hidden);

#ifdef DEBUG_FINAL_LAYER
    fprintf(stderr, "[FINAL] Before final layer img_hidden[0,0,:5]: ");
    for (int d = 0; d < 5; d++) fprintf(stderr, "%.6f ", img_hidden[d]);
    fprintf(stderr, "\n");
#endif

    /* Final layer: AdaLN modulation -> project to latent channels
     * norm_out.linear.weight is [6144, 3072] = [shift, scale] projection
     * Apply SiLU to t_emb before modulation projection (FLUX architecture)
     */
    double final_start = tf_get_time_ms();
    /* Reuse pre-allocated t_emb_silu buffer */
    for (int i = 0; i < hidden; i++) {
        float x = t_emb[i];
        tf->t_emb_silu[i] = x / (1.0f + expf(-x));
    }

    /* Reuse double_mod_img buffer for final_mod (needs hidden*2, has hidden*6) */
    float *final_mod = tf->double_mod_img;
    iris_linear_nobias(final_mod, tf->t_emb_silu, tf->final_norm_weight, 1, hidden, hidden * 2);

    /* Python: scale, shift = mod.chunk(2, dim=1) - scale is first half, shift is second half */
    float *final_scale = final_mod;
    float *final_shift = final_mod + hidden;

    float *final_norm = tf->work1;
    apply_adaln(final_norm, img_hidden, final_shift, final_scale, img_seq, hidden, 1e-6f);

    float *output_nlc = (float *)malloc(img_seq * tf->latent_channels * sizeof(float));
    LINEAR_BF16_OR_F32(output_nlc, final_norm, tf->final_proj_weight, tf->final_proj_weight_bf16,
                       img_seq, hidden, tf->latent_channels);

    /* Transpose output from NLC [seq, channels] to NCHW [channels, h, w] format
     * Input: output_nlc[pos * channels + c]
     * Output: output[c * img_seq + pos]
     */
    float *output = (float *)malloc(img_seq * tf->latent_channels * sizeof(float));
    for (int pos = 0; pos < img_seq; pos++) {
        for (int c = 0; c < channels; c++) {
            output[c * img_seq + pos] = output_nlc[pos * channels + c];
        }
    }
    free(output_nlc);

    free(t_emb);
    /* RoPE buffers are cached in the transformer and freed in iris_transformer_free_flux(). */

    double final_time = tf_get_time_ms() - final_start;

    /* Update global timing counters */
    iris_timing_transformer_double += double_time;
    iris_timing_transformer_single += single_time;
    iris_timing_transformer_final += final_time;
    iris_timing_transformer_total += double_time + single_time + final_time;

    if (iris_substep_callback)
        iris_substep_callback(IRIS_SUBSTEP_FINAL_LAYER, 0, 1);

#ifdef USE_METAL
    /* Ensure all GPU operations complete before returning.
     * This prevents issues where subsequent denoising steps
     * start before previous step's GPU work is fully done. */
    iris_gpu_sync();
#endif

    return output;
}

/* ========================================================================
 * Transformer Forward with Reference Image Tokens
 * ======================================================================== */

/*
 * Extended transformer forward that accepts reference image tokens.
 * Reference tokens are concatenated to target image tokens with separate
 * RoPE positions (T=t_offset instead of T=0).
 *
 * This implements FLUX.2's in-context conditioning for img2img.
 *
 * Parameters:
 *   tf          - Transformer model
 *   img_latent  - Target image latent (to be generated), NCHW format
 *   img_h/w     - Target image patch dimensions
 *   ref_latent  - Reference image latent (conditioning), NCHW format, or NULL
 *   ref_h/w     - Reference image patch dimensions (ignored if ref_latent is NULL)
 *   t_offset    - T coordinate for reference tokens (typically 10)
 *   txt_emb     - Text embeddings
 *   txt_seq     - Text sequence length
 *   timestep    - Current denoising timestep
 *
 * Returns:
 *   Output latent for target image only (not reference), NCHW format.
 *   Caller must free.
 */
float *iris_transformer_forward_refs_flux(iris_transformer_flux_t *tf,
                                          const float *img_latent, int img_h, int img_w,
                                          const float *ref_latent, int ref_h, int ref_w,
                                          int t_offset,
                                          const float *txt_emb, int txt_seq,
                                          float timestep) {
    /* If no reference, delegate to regular function */
    if (ref_latent == NULL) {
        return iris_transformer_forward_flux(tf, img_latent, img_h, img_w,
                                        txt_emb, txt_seq, timestep);
    }

#ifdef IRIS_DEV
    if (tf->dev) {
        iris_ref_t ref = {ref_latent, ref_h, ref_w, t_offset};
        float *out = forward_dev_flux(tf, img_latent, img_h, img_w, &ref, 1,
                                       txt_emb, txt_seq, timestep);
        if (out) return out;
    }
#endif
    int hidden = tf->hidden_size;
    int img_seq = img_h * img_w;
    int ref_seq = ref_h * ref_w;
    int combined_img_seq = img_seq + ref_seq;  /* Target + reference */
    int channels = tf->latent_channels;

    /* Ensure work buffers are sized for combined sequence */
    int total_seq = combined_img_seq + txt_seq;
    if (ensure_work_buffers(tf, total_seq) < 0) {
        fprintf(stderr, "Failed to allocate work buffers for seq_len=%d\n", total_seq);
        return NULL;
    }

    /* Get timestep embedding */
    int sincos_dim = tf->time_embed.sincos_dim;
    float *t_emb = (float *)malloc(hidden * sizeof(float));
    float t_sincos[256];
    get_timestep_embedding(t_sincos, timestep * 1000.0f, sincos_dim, 10000.0f);
    time_embed_forward(t_emb, t_sincos, &tf->time_embed, hidden, tf->t_emb_silu);

    /* Get cached combined RoPE for img2img (target + reference) */
    float *combined_rope_cos, *combined_rope_sin;
    get_cached_combined_rope(tf, img_h, img_w, ref_h, ref_w, t_offset,
                             &combined_rope_cos, &combined_rope_sin);

    /* Get cached text RoPE */
    float *txt_rope_cos, *txt_rope_sin;
    get_cached_txt_rope(tf, txt_seq, &txt_rope_cos, &txt_rope_sin);

    /* Transpose and concatenate image latents: [target, reference] */
    float *combined_transposed = (float *)malloc(combined_img_seq * channels * sizeof(float));

    /* Target image */
    for (int pos = 0; pos < img_seq; pos++) {
        for (int c = 0; c < channels; c++) {
            combined_transposed[pos * channels + c] = img_latent[c * img_seq + pos];
        }
    }
    /* Reference image */
    for (int pos = 0; pos < ref_seq; pos++) {
        for (int c = 0; c < channels; c++) {
            combined_transposed[(img_seq + pos) * channels + c] = ref_latent[c * ref_seq + pos];
        }
    }

#ifdef USE_METAL
    /* Try BF16 GPU-accelerated path for img2img.
     * Pass combined_img_seq as img_seq (full sequence including reference),
     * but only extract img_seq (target) tokens at the end. */
    if (iris_metal_available() && iris_bf16_pipeline_available() && tf->use_bf16) {
        float *bf16_output = iris_transformer_forward_bf16_flux(tf, combined_transposed, combined_img_seq,
                                                           img_seq, /* extract_seq = target only */
                                                           txt_emb, txt_seq, t_emb,
                                                           combined_rope_cos, combined_rope_sin,
                                                           txt_rope_cos, txt_rope_sin);
        if (bf16_output) {
            free(combined_transposed);
            free(t_emb);
            /* RoPE buffers are cached in transformer struct - don't free */
            return bf16_output;
        } else {
            BF16_DEBUG("[BF16] bf16 pipeline failed for refs, falling back\n");
        }
    }
#endif

    if (ensure_ref_fallback_scores_flux(tf, combined_img_seq, txt_seq) < 0) {
        free(combined_transposed);
        free(t_emb);
        return NULL;
    }

    /* Project combined image latent to hidden */
    float *combined_hidden = (float *)malloc(combined_img_seq * hidden * sizeof(float));
    LINEAR_BF16_OR_F32(combined_hidden, combined_transposed, tf->img_in_weight, tf->img_in_weight_bf16,
                       combined_img_seq, tf->latent_channels, hidden);
    free(combined_transposed);

    /* Project text embeddings to hidden */
    float *txt_hidden = tf->txt_hidden;
    LINEAR_BF16_OR_F32(txt_hidden, txt_emb, tf->txt_in_weight, tf->txt_in_weight_bf16,
                       txt_seq, tf->text_dim, hidden);

    /* Pre-compute AdaLN modulation for double blocks */
    int double_mod_size = hidden * 6;  /* shift1, scale1, gate1, shift2, scale2, gate2 */
    for (int j = 0; j < hidden; j++) {
        float x = t_emb[j];
        tf->t_emb_silu[j] = x / (1.0f + expf(-x));
    }
    iris_linear_nobias(tf->double_mod_img, tf->t_emb_silu, tf->adaln_double_img_weight,
                       1, hidden, double_mod_size);
    iris_linear_nobias(tf->double_mod_txt, tf->t_emb_silu, tf->adaln_double_txt_weight,
                       1, hidden, double_mod_size);

    /* Double blocks - process combined image with text */
    for (int i = 0; i < tf->num_double_layers; i++) {
        if (tf->use_mmap) {
            load_double_block_weights(&tf->double_blocks[i], tf->sf_files, tf->num_sf_files, i,
                                      tf->hidden_size, tf->mlp_hidden, tf->use_bf16);
        }
        double_block_forward(combined_hidden, txt_hidden,
                             &tf->double_blocks[i],
                             tf->double_mod_img, tf->double_mod_txt,
                             combined_rope_cos, combined_rope_sin,
                             txt_rope_cos, txt_rope_sin,
                             combined_img_seq, txt_seq, tf);
        if (tf->use_mmap) {
            free_double_block_weights(&tf->double_blocks[i]);
            /* With direct mmap pointers for bf16, no need to clear caches. */
        }
        if (iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, i, tf->num_double_layers);
    }

    /* Concatenate for single blocks: [txt, combined_img] */
    float *concat_hidden = (float *)malloc(total_seq * hidden * sizeof(float));
    memcpy(concat_hidden, txt_hidden, txt_seq * hidden * sizeof(float));
    memcpy(concat_hidden + txt_seq * hidden, combined_hidden, combined_img_seq * hidden * sizeof(float));
    free(combined_hidden);

    /* Single blocks */
    for (int i = 0; i < tf->num_single_layers; i++) {
        if (tf->use_mmap) {
            load_single_block_weights(&tf->single_blocks[i], tf->sf_files, tf->num_sf_files, i,
                                      tf->hidden_size, tf->mlp_hidden, tf->use_bf16);
        }
        single_block_forward(concat_hidden, &tf->single_blocks[i],
                             t_emb, tf->adaln_single_weight,
                             combined_rope_cos, combined_rope_sin,
                             txt_rope_cos, txt_rope_sin,
                             total_seq, txt_seq, tf);
        if (tf->use_mmap) {
            free_single_block_weights(&tf->single_blocks[i]);
            /* With direct mmap pointers for bf16, no need to clear caches. */
        }
        if (iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_SINGLE_BLOCK, i, tf->num_single_layers);
    }

    /* Extract ONLY target image hidden states (first img_seq tokens after txt) */
    float *img_hidden = (float *)malloc(img_seq * hidden * sizeof(float));
    memcpy(img_hidden, concat_hidden + txt_seq * hidden, img_seq * hidden * sizeof(float));
    free(concat_hidden);

    /* Final layer - only for target image tokens */
    for (int i = 0; i < hidden; i++) {
        float x = t_emb[i];
        tf->t_emb_silu[i] = x / (1.0f + expf(-x));
    }

    float *final_mod = tf->double_mod_img;
    iris_linear_nobias(final_mod, tf->t_emb_silu, tf->final_norm_weight, 1, hidden, hidden * 2);

    float *final_scale = final_mod;
    float *final_shift = final_mod + hidden;

    float *final_norm = tf->work1;
    apply_adaln(final_norm, img_hidden, final_shift, final_scale, img_seq, hidden, 1e-6f);
    free(img_hidden);

    float *output_nlc = (float *)malloc(img_seq * tf->latent_channels * sizeof(float));
    LINEAR_BF16_OR_F32(output_nlc, final_norm, tf->final_proj_weight, tf->final_proj_weight_bf16,
                       img_seq, hidden, tf->latent_channels);

    /* Transpose output from NLC to NCHW */
    float *output = (float *)malloc(img_seq * tf->latent_channels * sizeof(float));
    for (int pos = 0; pos < img_seq; pos++) {
        for (int c = 0; c < channels; c++) {
            output[c * img_seq + pos] = output_nlc[pos * channels + c];
        }
    }
    free(output_nlc);

    free(t_emb);
    /* RoPE buffers are cached in the transformer and freed in iris_transformer_free_flux(). */

    if (iris_substep_callback)
        iris_substep_callback(IRIS_SUBSTEP_FINAL_LAYER, 0, 1);

#ifdef USE_METAL
    iris_gpu_sync();
#endif

    return output;
}

/* ========================================================================
 * Transformer Forward with Multiple Reference Images
 * ======================================================================== */

/*
 * Extended transformer forward that accepts multiple reference images.
 * Each reference gets a different T offset in RoPE positioning.
 */
float *iris_transformer_forward_multirefs_flux(iris_transformer_flux_t *tf,
                                                const float *img_latent, int img_h, int img_w,
                                                const iris_ref_t *refs, int num_refs,
                                                const float *txt_emb, int txt_seq,
                                                float timestep) {
    /* No references - delegate to regular forward */
    if (refs == NULL || num_refs == 0) {
        return iris_transformer_forward_flux(tf, img_latent, img_h, img_w,
                                        txt_emb, txt_seq, timestep);
    }

    /* Single reference - use optimized path */
    if (num_refs == 1) {
        return iris_transformer_forward_refs_flux(tf, img_latent, img_h, img_w,
                                                  refs[0].latent, refs[0].h, refs[0].w,
                                                  refs[0].t_offset,
                                                  txt_emb, txt_seq, timestep);
    }

#ifdef IRIS_DEV
    if (tf->dev) {
        float *out = forward_dev_flux(tf, img_latent, img_h, img_w, refs, num_refs,
                                       txt_emb, txt_seq, timestep);
        if (out) return out;
    }
#endif
    int hidden = tf->hidden_size;
    int img_seq = img_h * img_w;
    int axis_dim = 32;
    int channels = tf->latent_channels;

    /* Calculate total reference sequence length */
    int total_ref_seq = 0;
    for (int r = 0; r < num_refs; r++) {
        total_ref_seq += refs[r].h * refs[r].w;
    }

    int combined_img_seq = img_seq + total_ref_seq;
    int total_seq = combined_img_seq + txt_seq;

    /* Ensure work buffers */
    if (ensure_work_buffers(tf, total_seq) < 0) {
        fprintf(stderr, "Failed to allocate work buffers for seq_len=%d\n", total_seq);
        return NULL;
    }

    /* Get timestep embedding */
    int sincos_dim = tf->time_embed.sincos_dim;
    float *t_emb = (float *)malloc(hidden * sizeof(float));
    float t_sincos[256];
    get_timestep_embedding(t_sincos, timestep * 1000.0f, sincos_dim, 10000.0f);
    time_embed_forward(t_emb, t_sincos, &tf->time_embed, hidden, tf->t_emb_silu);

    /* Allocate combined RoPE arrays */
    float *combined_rope_cos = (float *)malloc(combined_img_seq * axis_dim * 4 * sizeof(float));
    float *combined_rope_sin = (float *)malloc(combined_img_seq * axis_dim * 4 * sizeof(float));

    /* Compute RoPE for target image (T=0) */
    compute_rope_2d(combined_rope_cos, combined_rope_sin, img_h, img_w, axis_dim, tf->rope_theta);

    /* Compute RoPE for each reference and append */
    int rope_offset = img_seq * axis_dim * 4;
    for (int r = 0; r < num_refs; r++) {
        int ref_seq = refs[r].h * refs[r].w;
        compute_rope_2d_with_t_offset(combined_rope_cos + rope_offset,
                                      combined_rope_sin + rope_offset,
                                      refs[r].h, refs[r].w,
                                      axis_dim, tf->rope_theta, refs[r].t_offset);
        rope_offset += ref_seq * axis_dim * 4;
    }

    /* Get cached text RoPE */
    float *txt_rope_cos, *txt_rope_sin;
    get_cached_txt_rope(tf, txt_seq, &txt_rope_cos, &txt_rope_sin);

    /* Transpose and concatenate all image latents */
    float *combined_transposed = (float *)malloc(combined_img_seq * channels * sizeof(float));

    /* Target image */
    for (int pos = 0; pos < img_seq; pos++) {
        for (int c = 0; c < channels; c++) {
            combined_transposed[pos * channels + c] = img_latent[c * img_seq + pos];
        }
    }

    /* Reference images */
    int trans_offset = img_seq;
    for (int r = 0; r < num_refs; r++) {
        int ref_seq = refs[r].h * refs[r].w;
        for (int pos = 0; pos < ref_seq; pos++) {
            for (int c = 0; c < channels; c++) {
                combined_transposed[(trans_offset + pos) * channels + c] =
                    refs[r].latent[c * ref_seq + pos];
            }
        }
        trans_offset += ref_seq;
    }

#ifdef USE_METAL
    /* Try BF16 GPU-accelerated path for multi-ref img2img. */
    if (iris_metal_available() && iris_bf16_pipeline_available() && tf->use_bf16) {
        float *bf16_output = iris_transformer_forward_bf16_flux(tf, combined_transposed, combined_img_seq,
                                                           img_seq, /* extract_seq = target only */
                                                           txt_emb, txt_seq, t_emb,
                                                           combined_rope_cos, combined_rope_sin,
                                                           txt_rope_cos, txt_rope_sin);
        if (bf16_output) {
            free(combined_transposed);
            free(t_emb);
            free(combined_rope_cos);
            free(combined_rope_sin);
            /* txt_rope is cached - don't free */
            return bf16_output;
        } else {
            BF16_DEBUG("[BF16] bf16 pipeline failed for multi-refs, falling back\n");
        }
    }
#endif

    if (ensure_ref_fallback_scores_flux(tf, combined_img_seq, txt_seq) < 0) {
        free(combined_transposed);
        free(t_emb);
        free(combined_rope_cos);
        free(combined_rope_sin);
        return NULL;
    }

    /* Project combined image latent to hidden */
    float *combined_hidden = (float *)malloc(combined_img_seq * hidden * sizeof(float));
    LINEAR_BF16_OR_F32(combined_hidden, combined_transposed, tf->img_in_weight, tf->img_in_weight_bf16,
                       combined_img_seq, tf->latent_channels, hidden);
    free(combined_transposed);

    /* Project text embeddings to hidden */
    float *txt_hidden = tf->txt_hidden;
    LINEAR_BF16_OR_F32(txt_hidden, txt_emb, tf->txt_in_weight, tf->txt_in_weight_bf16,
                       txt_seq, tf->text_dim, hidden);

    /* Pre-compute AdaLN modulation */
    int double_mod_size = hidden * 6;
    for (int j = 0; j < hidden; j++) {
        float x = t_emb[j];
        tf->t_emb_silu[j] = x / (1.0f + expf(-x));
    }
    iris_linear_nobias(tf->double_mod_img, tf->t_emb_silu, tf->adaln_double_img_weight,
                       1, hidden, double_mod_size);
    iris_linear_nobias(tf->double_mod_txt, tf->t_emb_silu, tf->adaln_double_txt_weight,
                       1, hidden, double_mod_size);

    /* Double blocks */
    for (int i = 0; i < tf->num_double_layers; i++) {
        if (tf->use_mmap) {
            load_double_block_weights(&tf->double_blocks[i], tf->sf_files, tf->num_sf_files, i,
                                      tf->hidden_size, tf->mlp_hidden, tf->use_bf16);
        }
        double_block_forward(combined_hidden, txt_hidden,
                             &tf->double_blocks[i],
                             tf->double_mod_img, tf->double_mod_txt,
                             combined_rope_cos, combined_rope_sin,
                             txt_rope_cos, txt_rope_sin,
                             combined_img_seq, txt_seq, tf);
        if (tf->use_mmap) {
            free_double_block_weights(&tf->double_blocks[i]);
        }
        if (iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_DOUBLE_BLOCK, i, tf->num_double_layers);
    }

    /* Concatenate for single blocks */
    float *concat_hidden = (float *)malloc(total_seq * hidden * sizeof(float));
    memcpy(concat_hidden, txt_hidden, txt_seq * hidden * sizeof(float));
    memcpy(concat_hidden + txt_seq * hidden, combined_hidden, combined_img_seq * hidden * sizeof(float));
    free(combined_hidden);

    /* Single blocks */
    for (int i = 0; i < tf->num_single_layers; i++) {
        if (tf->use_mmap) {
            load_single_block_weights(&tf->single_blocks[i], tf->sf_files, tf->num_sf_files, i,
                                      tf->hidden_size, tf->mlp_hidden, tf->use_bf16);
        }
        single_block_forward(concat_hidden, &tf->single_blocks[i],
                             t_emb, tf->adaln_single_weight,
                             combined_rope_cos, combined_rope_sin,
                             txt_rope_cos, txt_rope_sin,
                             total_seq, txt_seq, tf);
        if (tf->use_mmap) {
            free_single_block_weights(&tf->single_blocks[i]);
        }
        if (iris_substep_callback)
            iris_substep_callback(IRIS_SUBSTEP_SINGLE_BLOCK, i, tf->num_single_layers);
    }

    /* Extract ONLY target image hidden states */
    float *img_hidden = (float *)malloc(img_seq * hidden * sizeof(float));
    memcpy(img_hidden, concat_hidden + txt_seq * hidden, img_seq * hidden * sizeof(float));
    free(concat_hidden);

    /* Final layer */
    for (int i = 0; i < hidden; i++) {
        float x = t_emb[i];
        tf->t_emb_silu[i] = x / (1.0f + expf(-x));
    }

    float *final_mod = tf->double_mod_img;
    iris_linear_nobias(final_mod, tf->t_emb_silu, tf->final_norm_weight, 1, hidden, hidden * 2);

    float *final_scale = final_mod;
    float *final_shift = final_mod + hidden;

    float *final_norm = tf->work1;
    apply_adaln(final_norm, img_hidden, final_shift, final_scale, img_seq, hidden, 1e-6f);
    free(img_hidden);

    float *output_nlc = (float *)malloc(img_seq * tf->latent_channels * sizeof(float));
    LINEAR_BF16_OR_F32(output_nlc, final_norm, tf->final_proj_weight, tf->final_proj_weight_bf16,
                       img_seq, hidden, tf->latent_channels);

    /* Transpose output from NLC to NCHW */
    float *output = (float *)malloc(img_seq * tf->latent_channels * sizeof(float));
    for (int pos = 0; pos < img_seq; pos++) {
        for (int c = 0; c < channels; c++) {
            output[c * img_seq + pos] = output_nlc[pos * channels + c];
        }
    }
    free(output_nlc);

    free(t_emb);
    free(combined_rope_cos);
    free(combined_rope_sin);
    /* Text RoPE buffers are cached in the transformer and freed in iris_transformer_free_flux(). */

    if (iris_substep_callback)
        iris_substep_callback(IRIS_SUBSTEP_FINAL_LAYER, 0, 1);

#ifdef USE_METAL
    iris_gpu_sync();
#endif

    return output;
}

/* ========================================================================
 * Transformer Loading
 * ======================================================================== */

static float *read_floats(FILE *f, int count) {
    float *data = (float *)malloc(count * sizeof(float));
    if (!data) return NULL;
    if (fread(data, sizeof(float), count, f) != (size_t)count) {
        free(data);
        return NULL;
    }
    return data;
}

iris_transformer_flux_t *iris_transformer_load_flux(FILE *f) {
    iris_transformer_flux_t *tf = calloc(1, sizeof(iris_transformer_flux_t));
    if (!tf) return NULL;

    /* Read config */
    uint32_t config[10];
    if (fread(config, sizeof(uint32_t), 10, f) != 10) goto error;

    tf->hidden_size = config[0];
    tf->num_heads = config[1];
    tf->head_dim = config[2];
    tf->mlp_hidden = config[3];
    tf->num_double_layers = config[4];
    tf->num_single_layers = config[5];
    tf->text_dim = config[6];
    tf->latent_channels = config[7];
    tf->max_seq_len = config[8];
    tf->rope_dim = config[9];

    float rope_theta;
    if (fread(&rope_theta, sizeof(float), 1, f) != 1) goto error;
    tf->rope_theta = rope_theta;
    tf->axis_dim = 32;  /* RoPE axis dimension (head_dim = 128 = 4 * axis_dim) */

    /* Read input projections */
    tf->img_in_weight = read_floats(f, tf->hidden_size * tf->latent_channels);
    tf->txt_in_weight = read_floats(f, tf->hidden_size * tf->text_dim);

    /* Read time embedding (binary format - deprecated, use safetensors) */
    tf->time_embed.sincos_dim = 256;  /* Match safetensors model */
    tf->time_embed.fc1_weight = read_floats(f, tf->hidden_size * 256);
    tf->time_embed.fc2_weight = read_floats(f, tf->hidden_size * tf->hidden_size);

    /* Read double blocks (binary format - deprecated, use safetensors) */
    tf->double_blocks = calloc(tf->num_double_layers, sizeof(double_block_t));
    for (int i = 0; i < tf->num_double_layers; i++) {
        double_block_t *b = &tf->double_blocks[i];
        int h = tf->hidden_size;
        int mlp = tf->mlp_hidden;
        int head_dim = tf->head_dim;

        /* QK norm weights (per head) */
        b->img_norm_q_weight = read_floats(f, head_dim);
        b->img_norm_k_weight = read_floats(f, head_dim);
        b->img_q_weight = read_floats(f, h * h);
        b->img_k_weight = read_floats(f, h * h);
        b->img_v_weight = read_floats(f, h * h);
        b->img_proj_weight = read_floats(f, h * h);
        b->img_mlp_gate_weight = read_floats(f, mlp * h);
        b->img_mlp_up_weight = read_floats(f, mlp * h);
        b->img_mlp_down_weight = read_floats(f, h * mlp);

        b->txt_norm_q_weight = read_floats(f, head_dim);
        b->txt_norm_k_weight = read_floats(f, head_dim);
        b->txt_q_weight = read_floats(f, h * h);
        b->txt_k_weight = read_floats(f, h * h);
        b->txt_v_weight = read_floats(f, h * h);
        b->txt_proj_weight = read_floats(f, h * h);
        b->txt_mlp_gate_weight = read_floats(f, mlp * h);
        b->txt_mlp_up_weight = read_floats(f, mlp * h);
        b->txt_mlp_down_weight = read_floats(f, h * mlp);
    }

    /* Read single blocks (binary format - deprecated, use safetensors) */
    tf->single_blocks = calloc(tf->num_single_layers, sizeof(single_block_t));
    for (int i = 0; i < tf->num_single_layers; i++) {
        single_block_t *b = &tf->single_blocks[i];
        int h = tf->hidden_size;
        int mlp = tf->mlp_hidden;
        int head_dim = tf->head_dim;

        b->norm_q_weight = read_floats(f, head_dim);
        b->norm_k_weight = read_floats(f, head_dim);
        b->qkv_mlp_weight = read_floats(f, (h * 3 + mlp * 2) * h);
        b->proj_mlp_weight = read_floats(f, h * (h + mlp));
    }

    /* Read final layer */
    tf->final_norm_weight = read_floats(f, tf->hidden_size);
    tf->final_proj_weight = read_floats(f, tf->latent_channels * tf->hidden_size);

    /* Precompute RoPE frequencies */
    tf->rope_freqs = (float *)malloc(tf->max_seq_len * tf->head_dim * sizeof(float));
    compute_rope_freqs(tf->rope_freqs, tf->max_seq_len, tf->head_dim, tf->rope_theta);

    /* Work buffers are dynamically allocated in forward() based on actual sequence
     * length. This avoids 8.4GB pre-allocation that was causing OOM on 16GB systems. */
    int hidden = tf->hidden_size;
    tf->img_hidden = NULL;
    tf->txt_hidden = NULL;
    tf->work_size = 0;
    tf->work1 = NULL;
    tf->work2 = NULL;
    tf->attn_q_t = NULL;
    tf->attn_k_t = NULL;
    tf->attn_v_t = NULL;
    tf->attn_out_t = NULL;
    tf->attn_scores = NULL;
    tf->attn_scores_alloc = 0;
    tf->work_seq_alloc = 0;
    tf->attn_cat_k = NULL;
    tf->attn_cat_v = NULL;
    tf->single_q = NULL;
    tf->single_k = NULL;
    tf->single_v = NULL;
    tf->single_mlp_gate = NULL;
    tf->single_mlp_up = NULL;
    tf->single_attn_out = NULL;
    tf->single_concat = NULL;
    tf->ffn_gate = NULL;
    tf->ffn_up = NULL;
    tf->double_img_attn_out = NULL;
    tf->double_txt_attn_out = NULL;

    /* Small constant-size buffers */
    tf->t_emb_silu = (float *)malloc(hidden * sizeof(float));
    tf->double_mod_img = (float *)malloc(hidden * 6 * sizeof(float));
    tf->double_mod_txt = (float *)malloc(hidden * 6 * sizeof(float));

    if (!tf->t_emb_silu || !tf->double_mod_img || !tf->double_mod_txt) {
        goto error;
    }

    return tf;

error:
    iris_transformer_free_flux(tf);
    return NULL;
}

void iris_transformer_free_flux(iris_transformer_flux_t *tf) {
    if (!tf) return;
#ifdef IRIS_DEV
    dev_free_flux(tf);
#endif

    /* In mmap mode, bf16 pointers point into the mmap'd file region and must
     * NOT be freed. Clean up any cached block weights first, then only NULL
     * the bf16 pointers (don't free them). */
    if (tf->use_mmap) {
        iris_transformer_free_mmap_cache_flux(tf);
    }

    free(tf->img_in_weight);
    free(tf->txt_in_weight);
    free(tf->img_in_weight_bf16);
    free(tf->txt_in_weight_bf16);
    free(tf->time_embed.fc1_weight);
    free(tf->time_embed.fc2_weight);

    if (tf->double_blocks) {
        for (int i = 0; i < tf->num_double_layers; i++) {
            double_block_t *b = &tf->double_blocks[i];
            free(b->img_norm_q_weight);
            free(b->img_norm_k_weight);
            free(b->img_q_weight);
            free(b->img_k_weight);
            free(b->img_v_weight);
            free(b->img_proj_weight);
            free(b->img_mlp_gate_weight);
            free(b->img_mlp_up_weight);
            free(b->img_mlp_down_weight);
            free(b->txt_norm_q_weight);
            free(b->txt_norm_k_weight);
            free(b->txt_q_weight);
            free(b->txt_k_weight);
            free(b->txt_v_weight);
            free(b->txt_proj_weight);
            free(b->txt_mlp_gate_weight);
            free(b->txt_mlp_up_weight);
            free(b->txt_mlp_down_weight);
            if (!tf->use_mmap) {
                free(b->img_q_weight_bf16);
                free(b->img_k_weight_bf16);
                free(b->img_v_weight_bf16);
                free(b->img_proj_weight_bf16);
                free(b->img_mlp_gate_weight_bf16);
                free(b->img_mlp_up_weight_bf16);
                free(b->img_mlp_down_weight_bf16);
                free(b->txt_q_weight_bf16);
                free(b->txt_k_weight_bf16);
                free(b->txt_v_weight_bf16);
                free(b->txt_proj_weight_bf16);
                free(b->txt_mlp_gate_weight_bf16);
                free(b->txt_mlp_up_weight_bf16);
                free(b->txt_mlp_down_weight_bf16);
            }
        }
        free(tf->double_blocks);
    }

    if (tf->single_blocks) {
        for (int i = 0; i < tf->num_single_layers; i++) {
            single_block_t *b = &tf->single_blocks[i];
            free(b->norm_q_weight);
            free(b->norm_k_weight);
            free(b->qkv_mlp_weight);
            free(b->proj_mlp_weight);
            if (!tf->use_mmap) {
                free(b->qkv_mlp_weight_bf16);
                free(b->proj_mlp_weight_bf16);
            }
        }
        free(tf->single_blocks);
    }

    free(tf->final_norm_weight);
    free(tf->final_proj_weight);
    free(tf->final_proj_weight_bf16);
    free(tf->rope_freqs);
    free(tf->img_hidden);
    free(tf->txt_hidden);
    free(tf->work1);
    free(tf->work2);
    free(tf->adaln_double_img_weight);
    free(tf->adaln_double_txt_weight);
    free(tf->adaln_single_weight);
    free(tf->adaln_double_img_weight_bf16);
    free(tf->adaln_double_txt_weight_bf16);
    free(tf->adaln_single_weight_bf16);

    /* Free attention workspace buffers */
    free(tf->attn_q_t);
    free(tf->attn_k_t);
    free(tf->attn_v_t);
    free(tf->attn_out_t);
    free(tf->attn_scores);
    free(tf->attn_cat_k);
    free(tf->attn_cat_v);

    /* Free single-block work buffers */
    free(tf->single_q);
    free(tf->single_k);
    free(tf->single_v);
    free(tf->single_mlp_gate);
    free(tf->single_mlp_up);
    free(tf->single_attn_out);
    free(tf->single_concat);

    /* Free FFN work buffers */
    free(tf->ffn_gate);
    free(tf->ffn_up);

    /* Free double-block work buffers */
    free(tf->t_emb_silu);
    free(tf->double_mod_img);
    free(tf->double_mod_txt);
    free(tf->double_img_attn_out);
    free(tf->double_txt_attn_out);

    /* Free cached RoPE buffers */
    free(tf->cached_img_rope_cos);
    free(tf->cached_img_rope_sin);
    free(tf->cached_ref_rope_cos);
    free(tf->cached_ref_rope_sin);
    free(tf->cached_txt_rope_cos);
    free(tf->cached_txt_rope_sin);
    free(tf->cached_combined_rope_cos);
    free(tf->cached_combined_rope_sin);

    /* Close safetensors files if in mmap mode */
    if (tf->use_mmap) {
        for (int i = 0; i < tf->num_sf_files; i++) {
            if (tf->sf_files[i]) safetensors_close(tf->sf_files[i]);
        }
        tf->num_sf_files = 0;
    }

    free(tf);
}

/* ========================================================================
 * Safetensors Loading
 * ======================================================================== */

static float *get_sf_tensor_tf(safetensors_file_t **files, int num_files, const char *name) {
    for (int f = 0; f < num_files; f++) {
        const safetensor_t *t = safetensors_find(files[f], name);
        if (t) return safetensors_get_f32(files[f], t);
    }
    fprintf(stderr, "Error: required tensor %s not found\n", name);
    return NULL;
}

/* Get tensor as bf16 (for GPU acceleration) */
static uint16_t *get_sf_tensor_bf16(safetensors_file_t **files, int num_files, const char *name) {
    for (int f = 0; f < num_files; f++) {
        const safetensor_t *t = safetensors_find(files[f], name);
        if (t) {
            if (!safetensor_is_bf16(t)) return NULL;
            return safetensors_get_bf16(files[f], t);
        }
    }
    return NULL;  /* Not found - bf16 is optional */
}

#ifdef USE_METAL
/* Warm up bf16→f16 cache for all weight tensors.
 * This converts bf16 weights to f16 during model loading so it doesn't happen
 * during the first inference step. This shifts ~5s of warmup from first step
 * to model loading, resulting in consistent per-step timing.
 */
static void warmup_bf16_weights(iris_transformer_flux_t *tf) {
    if (!tf->use_bf16) return;
    if (!iris_metal_available()) return;

    int h = tf->hidden_size;
    int mlp = tf->mlp_hidden;
    int text_dim = tf->text_dim;
    int latent_ch = tf->latent_channels;

    /* Input projections */
    if (tf->img_in_weight_bf16)
        iris_metal_warmup_bf16(tf->img_in_weight_bf16, (size_t)h * latent_ch);
    if (tf->txt_in_weight_bf16)
        iris_metal_warmup_bf16(tf->txt_in_weight_bf16, (size_t)h * text_dim);

    /* Double blocks */
    for (int i = 0; i < tf->num_double_layers; i++) {
        double_block_t *b = &tf->double_blocks[i];
        /* Image stream */
        if (b->img_q_weight_bf16)
            iris_metal_warmup_bf16(b->img_q_weight_bf16, (size_t)h * h);
        if (b->img_k_weight_bf16)
            iris_metal_warmup_bf16(b->img_k_weight_bf16, (size_t)h * h);
        if (b->img_v_weight_bf16)
            iris_metal_warmup_bf16(b->img_v_weight_bf16, (size_t)h * h);
        if (b->img_proj_weight_bf16)
            iris_metal_warmup_bf16(b->img_proj_weight_bf16, (size_t)h * h);
        if (b->img_mlp_gate_weight_bf16)
            iris_metal_warmup_bf16(b->img_mlp_gate_weight_bf16, (size_t)mlp * h);
        if (b->img_mlp_up_weight_bf16)
            iris_metal_warmup_bf16(b->img_mlp_up_weight_bf16, (size_t)mlp * h);
        if (b->img_mlp_down_weight_bf16)
            iris_metal_warmup_bf16(b->img_mlp_down_weight_bf16, (size_t)h * mlp);
        /* Text stream */
        if (b->txt_q_weight_bf16)
            iris_metal_warmup_bf16(b->txt_q_weight_bf16, (size_t)h * h);
        if (b->txt_k_weight_bf16)
            iris_metal_warmup_bf16(b->txt_k_weight_bf16, (size_t)h * h);
        if (b->txt_v_weight_bf16)
            iris_metal_warmup_bf16(b->txt_v_weight_bf16, (size_t)h * h);
        if (b->txt_proj_weight_bf16)
            iris_metal_warmup_bf16(b->txt_proj_weight_bf16, (size_t)h * h);
        if (b->txt_mlp_gate_weight_bf16)
            iris_metal_warmup_bf16(b->txt_mlp_gate_weight_bf16, (size_t)mlp * h);
        if (b->txt_mlp_up_weight_bf16)
            iris_metal_warmup_bf16(b->txt_mlp_up_weight_bf16, (size_t)mlp * h);
        if (b->txt_mlp_down_weight_bf16)
            iris_metal_warmup_bf16(b->txt_mlp_down_weight_bf16, (size_t)h * mlp);
    }

    /* Single blocks */
    int fused_dim = h * 3 + mlp * 2;  /* Q, K, V, gate, up */
    for (int i = 0; i < tf->num_single_layers; i++) {
        single_block_t *b = &tf->single_blocks[i];
        if (b->qkv_mlp_weight_bf16)
            iris_metal_warmup_bf16(b->qkv_mlp_weight_bf16, (size_t)fused_dim * h);
        if (b->proj_mlp_weight_bf16)
            iris_metal_warmup_bf16(b->proj_mlp_weight_bf16, (size_t)h * (h + mlp));
    }

    /* Final projection */
    if (tf->final_proj_weight_bf16)
        iris_metal_warmup_bf16(tf->final_proj_weight_bf16, (size_t)latent_ch * h);
}
#endif /* USE_METAL */

#ifdef IRIS_DEV
iris_transformer_flux_t *iris_transformer_load_safetensors_mmap_flux(const char *model_dir);
#endif

iris_transformer_flux_t *iris_transformer_load_safetensors_flux(const char *model_dir) {
#ifdef IRIS_DEV
    /* Keep the mmaps for device upload and on-demand CPU fallback. */
    if (iris_dev_available()) return iris_transformer_load_safetensors_mmap_flux(model_dir);
#endif
    iris_transformer_flux_t *tf = calloc(1, sizeof(iris_transformer_flux_t));
    if (!tf) return NULL;

    char name[256];

    /* Parse config from transformer/config.json, fall back to 4B defaults */
    if (parse_transformer_config(model_dir, tf) != 0) {
        tf->hidden_size = 3072;
        tf->num_heads = 24;
        tf->head_dim = 128;
        tf->mlp_hidden = 9216;
        tf->num_double_layers = 5;
        tf->num_single_layers = 20;
        tf->text_dim = 7680;
        tf->latent_channels = 128;
        tf->rope_theta = 2000.0f;
        tf->rope_dim = 128;
        tf->axis_dim = 32;
    }
    tf->max_seq_len = 52000;

    /* Open safetensors shards */
    safetensors_file_t *files[MAX_TF_SHARDS];
    int num_files = open_transformer_shards(model_dir, files, MAX_TF_SHARDS);
    if (num_files == 0) {
        fprintf(stderr, "iris_transformer_load: failed to open safetensors files\n");
        free(tf);
        return NULL;
    }

    /* Enable bf16 mode if Metal GPU is available */
#ifdef USE_METAL
    tf->use_bf16 = iris_metal_available();
    if (tf->use_bf16) {
        if (iris_verbose)
            fprintf(stderr, "Using bf16 weights for GPU acceleration\n");
    }
#else
    tf->use_bf16 = 0;
#endif

    int h = tf->hidden_size;
    int mlp = tf->mlp_hidden;

    /* Input projections */
    if (tf->use_bf16) {
        tf->img_in_weight_bf16 = get_sf_tensor_bf16(files, num_files, "x_embedder.weight");
        tf->txt_in_weight_bf16 = get_sf_tensor_bf16(files, num_files, "context_embedder.weight");
    } else {
        tf->img_in_weight = get_sf_tensor_tf(files, num_files, "x_embedder.weight");
        tf->txt_in_weight = get_sf_tensor_tf(files, num_files, "context_embedder.weight");
    }

    /* Time embedding */
    tf->time_embed.sincos_dim = 256;
    tf->time_embed.fc1_weight = get_sf_tensor_tf(files, num_files,
        "time_guidance_embed.timestep_embedder.linear_1.weight");
    tf->time_embed.fc2_weight = get_sf_tensor_tf(files, num_files,
        "time_guidance_embed.timestep_embedder.linear_2.weight");

    /* Modulation weights - these are always needed in f32 for CPU modulation computation */
    tf->adaln_double_img_weight = get_sf_tensor_tf(files, num_files,
        "double_stream_modulation_img.linear.weight");
    tf->adaln_double_txt_weight = get_sf_tensor_tf(files, num_files,
        "double_stream_modulation_txt.linear.weight");
    tf->adaln_single_weight = get_sf_tensor_tf(files, num_files,
        "single_stream_modulation.linear.weight");

    /* Double blocks */
    tf->double_blocks = calloc(tf->num_double_layers, sizeof(double_block_t));
    for (int i = 0; i < tf->num_double_layers; i++) {
        double_block_t *b = &tf->double_blocks[i];

        /* Image attention - QK norm weights (always f32) */
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_q.weight", i);
        b->img_norm_q_weight = get_sf_tensor_tf(files, num_files, name);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_k.weight", i);
        b->img_norm_k_weight = get_sf_tensor_tf(files, num_files, name);

        /* Image Q, K, V projections (separate) */
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_q.weight", i);
        if (tf->use_bf16) b->img_q_weight_bf16 = get_sf_tensor_bf16(files, num_files, name);
        else b->img_q_weight = get_sf_tensor_tf(files, num_files, name);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_k.weight", i);
        if (tf->use_bf16) b->img_k_weight_bf16 = get_sf_tensor_bf16(files, num_files, name);
        else b->img_k_weight = get_sf_tensor_tf(files, num_files, name);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_v.weight", i);
        if (tf->use_bf16) b->img_v_weight_bf16 = get_sf_tensor_bf16(files, num_files, name);
        else b->img_v_weight = get_sf_tensor_tf(files, num_files, name);

        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_out.0.weight", i);
        if (tf->use_bf16) b->img_proj_weight_bf16 = get_sf_tensor_bf16(files, num_files, name);
        else b->img_proj_weight = get_sf_tensor_tf(files, num_files, name);

        /* Image FFN - linear_in contains gate and up fused */
        snprintf(name, sizeof(name), "transformer_blocks.%d.ff.linear_in.weight", i);
        if (tf->use_bf16) {
            uint16_t *ff_in_bf16 = get_sf_tensor_bf16(files, num_files, name);
            if (ff_in_bf16) {
                b->img_mlp_gate_weight_bf16 = malloc(mlp * h * sizeof(uint16_t));
                b->img_mlp_up_weight_bf16 = malloc(mlp * h * sizeof(uint16_t));
                memcpy(b->img_mlp_gate_weight_bf16, ff_in_bf16, mlp * h * sizeof(uint16_t));
                memcpy(b->img_mlp_up_weight_bf16, ff_in_bf16 + mlp * h, mlp * h * sizeof(uint16_t));
                free(ff_in_bf16);
            }
        } else {
            float *ff_in = get_sf_tensor_tf(files, num_files, name);
            if (ff_in) {
                b->img_mlp_gate_weight = malloc(mlp * h * sizeof(float));
                b->img_mlp_up_weight = malloc(mlp * h * sizeof(float));
                memcpy(b->img_mlp_gate_weight, ff_in, mlp * h * sizeof(float));
                memcpy(b->img_mlp_up_weight, ff_in + mlp * h, mlp * h * sizeof(float));
                free(ff_in);
            }
        }

        snprintf(name, sizeof(name), "transformer_blocks.%d.ff.linear_out.weight", i);
        if (tf->use_bf16) b->img_mlp_down_weight_bf16 = get_sf_tensor_bf16(files, num_files, name);
        else b->img_mlp_down_weight = get_sf_tensor_tf(files, num_files, name);

        /* Text stream - QK norm weights (always f32) */
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_added_q.weight", i);
        b->txt_norm_q_weight = get_sf_tensor_tf(files, num_files, name);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.norm_added_k.weight", i);
        b->txt_norm_k_weight = get_sf_tensor_tf(files, num_files, name);

        /* Text Q, K, V projections (separate) */
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.add_q_proj.weight", i);
        if (tf->use_bf16) b->txt_q_weight_bf16 = get_sf_tensor_bf16(files, num_files, name);
        else b->txt_q_weight = get_sf_tensor_tf(files, num_files, name);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.add_k_proj.weight", i);
        if (tf->use_bf16) b->txt_k_weight_bf16 = get_sf_tensor_bf16(files, num_files, name);
        else b->txt_k_weight = get_sf_tensor_tf(files, num_files, name);
        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.add_v_proj.weight", i);
        if (tf->use_bf16) b->txt_v_weight_bf16 = get_sf_tensor_bf16(files, num_files, name);
        else b->txt_v_weight = get_sf_tensor_tf(files, num_files, name);

        snprintf(name, sizeof(name), "transformer_blocks.%d.attn.to_add_out.weight", i);
        if (tf->use_bf16) b->txt_proj_weight_bf16 = get_sf_tensor_bf16(files, num_files, name);
        else b->txt_proj_weight = get_sf_tensor_tf(files, num_files, name);

        snprintf(name, sizeof(name), "transformer_blocks.%d.ff_context.linear_in.weight", i);
        if (tf->use_bf16) {
            uint16_t *txt_ff_in_bf16 = get_sf_tensor_bf16(files, num_files, name);
            if (txt_ff_in_bf16) {
                b->txt_mlp_gate_weight_bf16 = malloc(mlp * h * sizeof(uint16_t));
                b->txt_mlp_up_weight_bf16 = malloc(mlp * h * sizeof(uint16_t));
                memcpy(b->txt_mlp_gate_weight_bf16, txt_ff_in_bf16, mlp * h * sizeof(uint16_t));
                memcpy(b->txt_mlp_up_weight_bf16, txt_ff_in_bf16 + mlp * h, mlp * h * sizeof(uint16_t));
                free(txt_ff_in_bf16);
            }
        } else {
            float *txt_ff_in = get_sf_tensor_tf(files, num_files, name);
            if (txt_ff_in) {
                b->txt_mlp_gate_weight = malloc(mlp * h * sizeof(float));
                b->txt_mlp_up_weight = malloc(mlp * h * sizeof(float));
                memcpy(b->txt_mlp_gate_weight, txt_ff_in, mlp * h * sizeof(float));
                memcpy(b->txt_mlp_up_weight, txt_ff_in + mlp * h, mlp * h * sizeof(float));
                free(txt_ff_in);
            }
        }

        snprintf(name, sizeof(name), "transformer_blocks.%d.ff_context.linear_out.weight", i);
        if (tf->use_bf16) b->txt_mlp_down_weight_bf16 = get_sf_tensor_bf16(files, num_files, name);
        else b->txt_mlp_down_weight = get_sf_tensor_tf(files, num_files, name);
    }

    /* Single blocks */
    tf->single_blocks = calloc(tf->num_single_layers, sizeof(single_block_t));
    for (int i = 0; i < tf->num_single_layers; i++) {
        single_block_t *b = &tf->single_blocks[i];

        /* QK norm weights (always f32, small) */
        snprintf(name, sizeof(name), "single_transformer_blocks.%d.attn.norm_q.weight", i);
        b->norm_q_weight = get_sf_tensor_tf(files, num_files, name);
        snprintf(name, sizeof(name), "single_transformer_blocks.%d.attn.norm_k.weight", i);
        b->norm_k_weight = get_sf_tensor_tf(files, num_files, name);

        /* Major linear weights - load bf16 or f32 based on mode */
        snprintf(name, sizeof(name), "single_transformer_blocks.%d.attn.to_qkv_mlp_proj.weight", i);
        if (tf->use_bf16) b->qkv_mlp_weight_bf16 = get_sf_tensor_bf16(files, num_files, name);
        else b->qkv_mlp_weight = get_sf_tensor_tf(files, num_files, name);

        snprintf(name, sizeof(name), "single_transformer_blocks.%d.attn.to_out.weight", i);
        if (tf->use_bf16) b->proj_mlp_weight_bf16 = get_sf_tensor_bf16(files, num_files, name);
        else b->proj_mlp_weight = get_sf_tensor_tf(files, num_files, name);
    }

    /* Final layer */
    tf->final_norm_weight = get_sf_tensor_tf(files, num_files, "norm_out.linear.weight");
    if (tf->use_bf16) {
        tf->final_proj_weight_bf16 = get_sf_tensor_bf16(files, num_files, "proj_out.weight");
    } else {
        tf->final_proj_weight = get_sf_tensor_tf(files, num_files, "proj_out.weight");
    }

    /* Close safetensors files (non-mmap: data already copied) */
    for (int i = 0; i < num_files; i++) safetensors_close(files[i]);

    /* Precompute RoPE frequencies */
    tf->rope_freqs = malloc(tf->max_seq_len * tf->head_dim * sizeof(float));
    if (tf->rope_freqs) {
        compute_rope_freqs(tf->rope_freqs, tf->max_seq_len, tf->head_dim, tf->rope_theta);
    }

    /* Work buffers are dynamically allocated in forward() based on actual sequence
     * length. This avoids the 8.4GB pre-allocation that was causing OOM on 16GB systems. */
    int hidden = tf->hidden_size;
    tf->img_hidden = NULL;
    tf->txt_hidden = NULL;
    tf->work_size = 0;
    tf->work1 = NULL;
    tf->work2 = NULL;
    tf->attn_q_t = NULL;
    tf->attn_k_t = NULL;
    tf->attn_v_t = NULL;
    tf->attn_out_t = NULL;
    tf->attn_scores = NULL;
    tf->attn_scores_alloc = 0;
    tf->work_seq_alloc = 0;
    tf->attn_cat_k = NULL;
    tf->attn_cat_v = NULL;
    tf->single_q = NULL;
    tf->single_k = NULL;
    tf->single_v = NULL;
    tf->single_mlp_gate = NULL;
    tf->single_mlp_up = NULL;
    tf->single_attn_out = NULL;
    tf->single_concat = NULL;
    tf->ffn_gate = NULL;
    tf->ffn_up = NULL;
    tf->double_img_attn_out = NULL;
    tf->double_txt_attn_out = NULL;

    /* Small constant-size buffers - always allocate */
    tf->t_emb_silu = malloc(hidden * sizeof(float));
    tf->double_mod_img = malloc(hidden * 6 * sizeof(float));
    tf->double_mod_txt = malloc(hidden * 6 * sizeof(float));

    if (!tf->t_emb_silu || !tf->double_mod_img || !tf->double_mod_txt) {
        iris_transformer_free_flux(tf);
        return NULL;
    }

#ifdef USE_METAL
    /* Pre-warm bf16→f16 cache to avoid conversion overhead on first inference step */
    warmup_bf16_weights(tf);
#endif

    return tf;
}

/* Load transformer in mmap mode - only load small weights, keep files open for block loading */
iris_transformer_flux_t *iris_transformer_load_safetensors_mmap_flux(const char *model_dir) {
    iris_transformer_flux_t *tf = calloc(1, sizeof(iris_transformer_flux_t));
    if (!tf) return NULL;

    /* Parse config from transformer/config.json, fall back to 4B defaults */
    if (parse_transformer_config(model_dir, tf) != 0) {
        tf->hidden_size = 3072;
        tf->num_heads = 24;
        tf->head_dim = 128;
        tf->mlp_hidden = 9216;
        tf->num_double_layers = 5;
        tf->num_single_layers = 20;
        tf->text_dim = 7680;
        tf->latent_channels = 128;
        tf->rope_theta = 2000.0f;
        tf->rope_dim = 128;
        tf->axis_dim = 32;
    }
    tf->max_seq_len = 52000;  /* Support up to 1792x1792 */

    /* Open safetensors shards and keep them open for on-demand loading */
    tf->use_mmap = 1;
    tf->num_sf_files = open_transformer_shards(model_dir, tf->sf_files, MAX_TF_SHARDS);
    if (tf->num_sf_files == 0) {
        fprintf(stderr, "iris_transformer_load_mmap: failed to open safetensors files\n");
        free(tf);
        return NULL;
    }

    /* Enable bf16 mode if Metal GPU is available */
#ifdef USE_METAL
    tf->use_bf16 = iris_metal_available();
    if (tf->use_bf16) {
        if (iris_verbose)
            fprintf(stderr, "Using bf16 weights for GPU acceleration (mmap mode)\n");
    }
#else
    tf->use_bf16 = 0;
#endif

    safetensors_file_t **files = tf->sf_files;
    int num_files = tf->num_sf_files;

    /* Input projections - always load (small) */
    tf->img_in_weight = get_sf_tensor_tf(files, num_files, "x_embedder.weight");
    tf->txt_in_weight = get_sf_tensor_tf(files, num_files, "context_embedder.weight");
    if (tf->use_bf16) {
        tf->img_in_weight_bf16 = get_sf_tensor_bf16(files, num_files, "x_embedder.weight");
        tf->txt_in_weight_bf16 = get_sf_tensor_bf16(files, num_files, "context_embedder.weight");
    }

    /* Time embedding - always load (small) */
    tf->time_embed.sincos_dim = 256;
    tf->time_embed.fc1_weight = get_sf_tensor_tf(files, num_files,
        "time_guidance_embed.timestep_embedder.linear_1.weight");
    tf->time_embed.fc2_weight = get_sf_tensor_tf(files, num_files,
        "time_guidance_embed.timestep_embedder.linear_2.weight");

    /* Modulation weights - always load */
    tf->adaln_double_img_weight = get_sf_tensor_tf(files, num_files,
        "double_stream_modulation_img.linear.weight");
    tf->adaln_double_txt_weight = get_sf_tensor_tf(files, num_files,
        "double_stream_modulation_txt.linear.weight");
    tf->adaln_single_weight = get_sf_tensor_tf(files, num_files,
        "single_stream_modulation.linear.weight");
    if (tf->use_bf16) {
        tf->adaln_double_img_weight_bf16 = get_sf_tensor_bf16(files, num_files,
            "double_stream_modulation_img.linear.weight");
        tf->adaln_double_txt_weight_bf16 = get_sf_tensor_bf16(files, num_files,
            "double_stream_modulation_txt.linear.weight");
        tf->adaln_single_weight_bf16 = get_sf_tensor_bf16(files, num_files,
            "single_stream_modulation.linear.weight");
    }

    /* Allocate empty block arrays - weights loaded on-demand */
    tf->double_blocks = calloc(tf->num_double_layers, sizeof(double_block_t));
    tf->single_blocks = calloc(tf->num_single_layers, sizeof(single_block_t));

    /* Final layer - always load (small) */
    tf->final_norm_weight = get_sf_tensor_tf(files, num_files, "norm_out.linear.weight");
    tf->final_proj_weight = get_sf_tensor_tf(files, num_files, "proj_out.weight");
    if (tf->use_bf16) {
        tf->final_proj_weight_bf16 = get_sf_tensor_bf16(files, num_files, "proj_out.weight");
    }

    /* Precompute RoPE frequencies */
    tf->rope_freqs = malloc(tf->max_seq_len * tf->head_dim * sizeof(float));
    if (tf->rope_freqs) {
        compute_rope_freqs(tf->rope_freqs, tf->max_seq_len, tf->head_dim, tf->rope_theta);
    }

    /* Work buffers - dynamically allocated in forward() */
    int hidden = tf->hidden_size;
    tf->img_hidden = NULL;
    tf->txt_hidden = NULL;
    tf->work_size = 0;
    tf->work1 = NULL;
    tf->work2 = NULL;
    tf->attn_q_t = NULL;
    tf->attn_k_t = NULL;
    tf->attn_v_t = NULL;
    tf->attn_out_t = NULL;
    tf->attn_scores = NULL;
    tf->attn_scores_alloc = 0;
    tf->work_seq_alloc = 0;
    tf->attn_cat_k = NULL;
    tf->attn_cat_v = NULL;
    tf->single_q = NULL;
    tf->single_k = NULL;
    tf->single_v = NULL;
    tf->single_mlp_gate = NULL;
    tf->single_mlp_up = NULL;
    tf->single_attn_out = NULL;
    tf->single_concat = NULL;
    tf->ffn_gate = NULL;
    tf->ffn_up = NULL;
    tf->double_img_attn_out = NULL;
    tf->double_txt_attn_out = NULL;

    /* Small constant-size buffers - always allocate */
    tf->t_emb_silu = malloc(hidden * sizeof(float));
    tf->double_mod_img = malloc(hidden * 6 * sizeof(float));
    tf->double_mod_txt = malloc(hidden * 6 * sizeof(float));

    if (!tf->t_emb_silu || !tf->double_mod_img || !tf->double_mod_txt) {
        iris_transformer_free_flux(tf);
        return NULL;
    }

#ifdef IRIS_DEV
    if (iris_dev_available()) dev_load_flux(tf);
#endif

#ifdef USE_METAL
    /* Pre-warm bf16 weight buffer cache: copy all block weights from mmap
     * to Metal GPU buffers. This shifts ~1s of first-step overhead to model
     * loading, making all denoising steps consistently fast. Not needed
     * when the device forward owns its own copy of the weights. */
    if (!tf->dev) warmup_mmap_bf16_buffers(tf);
#endif

    return tf;
}
