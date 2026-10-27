/*
 * Iris Main Implementation
 *
 * Main entry point for the Iris inference engine.
 * Ties together all components: tokenizer, text encoder, VAE, transformer, sampling.
 */

#include "iris.h"
#include "iris_kernels.h"
#include "iris_safetensors.h"
#include "iris_qwen3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/stat.h>
#include <limits.h>

#ifdef USE_METAL
#include "iris_metal.h"
#endif
#ifdef IRIS_DEV
#include "iris_dev.h"
#endif

/* ========================================================================
 * Forward Declarations for Internal Types
 * ======================================================================== */

typedef struct iris_tokenizer iris_tokenizer;
typedef struct iris_vae iris_vae_t;
typedef struct iris_transformer_flux iris_transformer_flux_t;

/* Internal function declarations */
extern iris_tokenizer *iris_tokenizer_load(const char *path);
extern void iris_tokenizer_free(iris_tokenizer *tok);
extern int *iris_tokenize(iris_tokenizer *tok, const char *text,
                          int *num_tokens, int max_len);

extern iris_vae_t *iris_vae_load(FILE *f);
extern iris_vae_t *iris_vae_load_safetensors(safetensors_file_t *sf);
extern iris_vae_t *iris_vae_load_safetensors_ex(safetensors_file_t *sf,
                                                  int z_channels,
                                                  float scaling_factor,
                                                  float shift_factor,
                                                  int allow_large_images);
extern int iris_vae_reserve(iris_vae_t *vae, int batch, int H, int W);
extern void iris_vae_free(iris_vae_t *vae);
extern float *iris_vae_encode(iris_vae_t *vae, const float *img,
                              int batch, int H, int W, int *out_h, int *out_w);
extern iris_image *iris_vae_decode(iris_vae_t *vae, const float *latent,
                                   int batch, int latent_h, int latent_w);
extern float *iris_image_to_tensor(const iris_image *img);

extern iris_transformer_flux_t *iris_transformer_load_flux(FILE *f);
extern iris_transformer_flux_t *iris_transformer_load_safetensors_flux(const char *model_dir);
extern iris_transformer_flux_t *iris_transformer_load_safetensors_mmap_flux(const char *model_dir);
extern void iris_transformer_free_flux(iris_transformer_flux_t *tf);
extern float *iris_transformer_forward_flux(iris_transformer_flux_t *tf,
                                        const float *img_latent, int img_h, int img_w,
                                        const float *txt_emb, int txt_seq,
                                        float timestep);

extern float *iris_sample_euler_flux(void *transformer, void *text_encoder,
                                float *z, int batch, int channels, int h, int w,
                                const float *text_emb, int text_seq,
                                const float *schedule, int num_steps,
                                void (*progress_callback)(int step, int total));
extern float *iris_sample_euler_refs_flux(void *transformer, void *text_encoder,
                                          float *z, int batch, int channels, int h, int w,
                                          const float *ref_latent, int ref_h, int ref_w,
                                          int t_offset,
                                          const float *text_emb, int text_seq,
                                          const float *schedule, int num_steps,
                                          void (*progress_callback)(int step, int total));

/* Multi-reference support */
typedef struct {
    const float *latent;
    int h, w;
    int t_offset;
} iris_ref_t;

extern float *iris_sample_euler_multirefs_flux(void *transformer, void *text_encoder,
                                                float *z, int batch, int channels, int h, int w,
                                                const iris_ref_t *refs, int num_refs,
                                                const float *text_emb, int text_seq,
                                                const float *schedule, int num_steps,
                                                void (*progress_callback)(int step, int total));

/* CFG sampling (for base model) */
extern float *iris_sample_euler_cfg_flux(void *transformer, void *text_encoder,
                                     float *z, int batch, int channels, int h, int w,
                                     const float *text_emb_cond, int text_seq_cond,
                                     const float *text_emb_uncond, int text_seq_uncond,
                                     float guidance_scale,
                                     const float *schedule, int num_steps,
                                     void (*progress_callback)(int step, int total));
extern float *iris_sample_euler_cfg_refs_flux(void *transformer, void *text_encoder,
                                               float *z, int batch, int channels, int h, int w,
                                               const float *ref_latent, int ref_h, int ref_w,
                                               int t_offset,
                                               const float *text_emb_cond, int text_seq_cond,
                                               const float *text_emb_uncond, int text_seq_uncond,
                                               float guidance_scale,
                                               const float *schedule, int num_steps,
                                               void (*progress_callback)(int step, int total));
extern float *iris_sample_euler_cfg_multirefs_flux(void *transformer, void *text_encoder,
                                                     float *z, int batch, int channels, int h, int w,
                                                     const iris_ref_t *refs, int num_refs,
                                                     const float *text_emb_cond, int text_seq_cond,
                                                     const float *text_emb_uncond, int text_seq_uncond,
                                                     float guidance_scale,
                                                     const float *schedule, int num_steps,
                                                     void (*progress_callback)(int step, int total));

extern float *iris_schedule_linear(int num_steps);
extern float *iris_schedule_power(int num_steps, float alpha);
extern float *iris_schedule_flux(int num_steps, int image_seq_len);
extern float *iris_schedule_zimage(int num_steps, int image_seq_len);
extern float *iris_schedule_flux3(int num_steps);
extern float *iris_init_noise(int batch, int channels, int h, int w, int64_t seed);

/* FLUX.3 transformer and sampling */
typedef struct iris_transformer_flux3 iris_transformer_flux3_t;
extern iris_transformer_flux3_t *iris_transformer_load_flux3(const char *model_dir);
extern void iris_transformer_free_flux3(iris_transformer_flux3_t *tf);
extern float *iris_sample_euler_flux3(void *transformer,
                                      const float *z, int channels, int h, int w,
                                      const float *text_emb, int text_seq,
                                      const float *uncond_emb, int uncond_seq,
                                      const float *const *refs, const int *ref_h,
                                      const int *ref_w, int num_refs,
                                      const float *schedule, int num_steps,
                                      float guidance,
                                      void (*progress_callback)(int step, int total));

/* Z-Image transformer and sampling */
typedef struct zi_transformer zi_transformer_t;
extern zi_transformer_t *zi_transformer_load_safetensors(const char *model_dir,
                                                           int dim, int n_heads,
                                                           int n_layers, int n_refiner,
                                                           int cap_feat_dim, int in_channels,
                                                           int patch_size, float rope_theta,
                                                           const int *axes_dims);
extern void iris_transformer_free_zimage(zi_transformer_t *tf);
extern float *iris_sample_euler_zimage(void *transformer,
                                        float *z, int batch, int channels, int h, int w,
                                        int patch_size,
                                        const float *cap_feats, int cap_seq,
                                        const float *schedule, int num_steps,
                                        void (*progress_callback)(int step, int total));

/* Return schedule for Flux models based on params.
 * Default is shifted sigmoid; overrides: linear, power, flowmatch. */
static float *iris_selected_schedule(const iris_params *p, int image_seq_len) {
    switch (p->schedule) {
    case IRIS_SCHEDULE_LINEAR:    return iris_schedule_linear(p->num_steps);
    case IRIS_SCHEDULE_POWER:     return iris_schedule_power(p->num_steps, p->power_alpha);
    case IRIS_SCHEDULE_FLOWMATCH: return iris_schedule_zimage(p->num_steps, image_seq_len);
    default:                      return iris_schedule_flux(p->num_steps, image_seq_len);
    }
}

/* Return schedule for FLUX.3 models based on params.
 * Default is FlowMatch with FLUX.3's large static shift; overrides: linear,
 * power, sigmoid, flowmatch (Z-Image's shift). */
static float *iris_selected_flux3_schedule(const iris_params *p, int image_seq_len) {
    switch (p->schedule) {
    case IRIS_SCHEDULE_LINEAR:    return iris_schedule_linear(p->num_steps);
    case IRIS_SCHEDULE_POWER:     return iris_schedule_power(p->num_steps, p->power_alpha);
    case IRIS_SCHEDULE_SIGMOID:   return iris_schedule_flux(p->num_steps, image_seq_len);
    case IRIS_SCHEDULE_FLOWMATCH: return iris_schedule_zimage(p->num_steps, image_seq_len);
    default:                      return iris_schedule_flux3(p->num_steps);
    }
}

/* Return schedule for Z-Image models based on params.
 * Default is FlowMatch Euler; overrides: linear, power, sigmoid. */
static float *iris_selected_zimage_schedule(const iris_params *p, int image_seq_len) {
    switch (p->schedule) {
    case IRIS_SCHEDULE_LINEAR:  return iris_schedule_linear(p->num_steps);
    case IRIS_SCHEDULE_POWER:   return iris_schedule_power(p->num_steps, p->power_alpha);
    case IRIS_SCHEDULE_SIGMOID: return iris_schedule_flux(p->num_steps, image_seq_len);
    default:                    return iris_schedule_zimage(p->num_steps, image_seq_len);
    }
}

/* ========================================================================
 * Text Encoder (Qwen3)
 * ======================================================================== */

/* Qwen3 text encoder is implemented in iris_qwen3.c */

/* ========================================================================
 * Main Context Structure
 * ======================================================================== */

struct iris_ctx {
    /* Components */
    iris_tokenizer *tokenizer;
    qwen3_encoder_t *qwen3_encoder;
    iris_vae_t *vae;
    iris_transformer_flux_t *transformer;
    zi_transformer_t *zi_transformer;
    iris_transformer_flux3_t *flux3_transformer;

    /* Configuration */
    int max_width;
    int max_height;
    int allow_large_images;
    int default_steps;
    float default_guidance;
    int is_distilled;  /* 1 = distilled (4-step), 0 = base (50-step CFG) */
    int text_dim;      /* Text embedding dimension (7680 for 4B, varies for 9B) */
    int is_non_commercial; /* 1 if model has non-commercial license (9B) */
    int num_heads;     /* Transformer attention heads (24 for 4B, 32 for 9B) */
    int is_zimage;     /* 1 = Z-Image S3-DiT, 0 = Flux MMDiT */
    int is_flux3;      /* 1 = FLUX.3 Image Klein */

    /* Z-Image specific config (from transformer/config.json) */
    int zi_dim;            /* Hidden dim (3840) */
    int zi_n_layers;       /* Main transformer layers (30) */
    int zi_n_refiner;      /* Noise/context refiner layers (2) */
    int zi_cap_feat_dim;   /* Caption feature dim (2560) */
    int zi_in_channels;    /* VAE latent channels (16) */
    int zi_patch_size;     /* Spatial patch size (2) */
    float zi_rope_theta;   /* RoPE theta (256.0) */
    int zi_axes_dims[3];   /* RoPE axis dims [32, 48, 48] */
    int zi_latent_channels;/* Patchified latent channels (64 = 16*2*2) */

    /* VAE config (read from vae/config.json) */
    int vae_z_channels;    /* Latent channels before patchify (32 Flux, 16 Z-Image) */
    float vae_scaling;     /* Scaling factor (0.3611 for Z-Image, 0 = use batch norm) */
    float vae_shift;       /* Shift factor (0.1159 for Z-Image, 0 = use batch norm) */

    /* Model info */
    char model_name[64];
    char model_version[32];
    char model_dir[512];  /* For reloading text encoder if released */

    /* Memory mode */
    int use_mmap;  /* Use mmap for text encoder (lower memory, slower) */
};

/* Global error message */
static char g_error_msg[256] = {0};

const char *iris_get_error(void) {
    return g_error_msg;
}

void iris_set_step_image_callback(iris_ctx *ctx, iris_step_image_cb_t callback) {
    iris_step_image_callback = callback;
    iris_step_image_vae = callback ? ctx->vae : NULL;
}

static void set_error(const char *msg) {
    strncpy(g_error_msg, msg, sizeof(g_error_msg) - 1);
    g_error_msg[sizeof(g_error_msg) - 1] = '\0';
}

int iris_max_image_dimension(const iris_ctx *ctx) {
    return ctx ? ctx->max_width : IRIS_VAE_MAX_DIM;
}

static void image_size_error(const iris_ctx *ctx) {
    char msg[160];
    snprintf(msg, sizeof(msg), "Image dimensions exceed maximum (%dx%d)%s",
             ctx->max_width, ctx->max_height,
             ctx->allow_large_images ? "" : "; use --allow-large-images to opt in");
    set_error(msg);
}

/* Reserve output scratch before denoising. References can grow it further
 * during encoding; buffers are shared and reused, not added per reference. */
static int prepare_vae(iris_ctx *ctx, int width, int height) {
    if (!ctx->allow_large_images) return 1;
    if (width <= 0 || height <= 0 || width > ctx->max_width ||
        height > ctx->max_height ||
        (size_t)width * height > (size_t)INT_MAX / 256) {
        set_error("Image dimensions exceed supported VAE tensor size");
        return 0;
    }
    if (!iris_vae_reserve(ctx->vae, 1, height, width)) {
        set_error("Out of memory");
        return 0;
    }
    return 1;
}

/* ========================================================================
 * Model Loading from HuggingFace-style directory with safetensors files
 * ======================================================================== */

static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

/* Main model loading entry point. Parses model_index.json to auto-detect
 * the model type (Flux vs Z-Image, distilled vs base), then reads
 * transformer/config.json and vae/config.json for architecture parameters
 * (hidden dim, heads, layers, etc.). Only the VAE (~300MB) is loaded
 * eagerly; the text encoder and transformer are deferred to generation
 * time so they can be swapped in/out on memory-constrained systems. */
iris_ctx *iris_load_dir(const char *model_dir) {
    return iris_load_dir_ex(model_dir, 0);
}

iris_ctx *iris_load_dir_ex(const char *model_dir, int allow_large_images) {
    char path[1024];

    iris_ctx *ctx = calloc(1, sizeof(iris_ctx));
    if (!ctx) {
        set_error("Out of memory");
        return NULL;
    }

    /* Set defaults - max 1792x1792 (requires ~18GB VAE work buffers) */
    ctx->allow_large_images = allow_large_images != 0;
    ctx->max_width = allow_large_images ? 4096 : IRIS_VAE_MAX_DIM;
    ctx->max_height = ctx->max_width;
    strncpy(ctx->model_version, "1.0", sizeof(ctx->model_version) - 1);
    strncpy(ctx->model_dir, model_dir, sizeof(ctx->model_dir) - 1);

    /* Autodetect model type from model_index.json.
     * Distilled model has "is_distilled": true, base model does not.
     * Z-Image has "_class_name": "ZImagePipeline". */
    ctx->is_distilled = 1;  /* Default to distilled */
    ctx->is_zimage = 0;
    snprintf(path, sizeof(path), "%s/model_index.json", model_dir);
    if (file_exists(path)) {
        FILE *f = fopen(path, "r");
        if (f) {
            char buf[4096];
            size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            buf[n] = '\0';
            fclose(f);
            /* Check for Z-Image pipeline */
            if (strstr(buf, "ZImagePipeline") || strstr(buf, "Z-Image")) {
                ctx->is_zimage = 1;
            }
            /* If "is_distilled" is present and true, it's distilled.
             * If absent, it's the base model. */
            if (!strstr(buf, "\"is_distilled\": true") &&
                !strstr(buf, "\"is_distilled\":true")) {
                ctx->is_distilled = 0;
            }
        }
    }

    /* Read transformer/config.json to determine model size and architecture. */
    int num_heads = 24;  /* default 4B */
    ctx->text_dim = 7680;  /* default 4B: 3 * 2560 */
    snprintf(path, sizeof(path), "%s/transformer/config.json", model_dir);
    if (file_exists(path)) {
        FILE *f = fopen(path, "r");
        if (f) {
            char buf[8192];
            size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            buf[n] = '\0';
            fclose(f);
            char *p;
            if ((p = strstr(buf, "\"num_attention_heads\""))) {
                if ((p = strchr(p, ':'))) num_heads = atoi(p + 1);
            }
            int joint_dim = 0;
            if ((p = strstr(buf, "\"joint_attention_dim\""))) {
                if ((p = strchr(p, ':'))) joint_dim = atoi(p + 1);
            }
            if (joint_dim > 0) ctx->text_dim = joint_dim;

            /* Z-Image autodetection: look for Z-Image-specific fields */
            if ((p = strstr(buf, "\"cap_feat_dim\""))) {
                ctx->is_zimage = 1;
            }

            /* FLUX.3: step-distilled checkpoints are flow maps */
            if (strstr(buf, "Flux3Transformer2DModel")) {
                ctx->is_flux3 = 1;
                ctx->is_distilled = 0;
                if ((p = strstr(buf, "\"flow_map\"")) && (p = strchr(p, ':'))) {
                    p += 1 + strspn(p + 1, " ");
                    ctx->is_distilled = strncmp(p, "true", 4) == 0;
                }
            }

            /* Parse Z-Image config if detected */
            if (ctx->is_zimage) {
                /* dim (hidden size) */
                ctx->zi_dim = 3840;
                if ((p = strstr(buf, "\"dim\""))) {
                    char *colon = strchr(p, ':');
                    if (colon) ctx->zi_dim = atoi(colon + 1);
                }
                num_heads = ctx->zi_dim / 128;  /* head_dim = 128 */

                /* n_layers */
                ctx->zi_n_layers = 30;
                if ((p = strstr(buf, "\"n_layers\""))) {
                    char *colon = strchr(p, ':');
                    if (colon) ctx->zi_n_layers = atoi(colon + 1);
                }

                /* n_refiner_layers */
                ctx->zi_n_refiner = 2;
                if ((p = strstr(buf, "\"n_refiner_layers\""))) {
                    char *colon = strchr(p, ':');
                    if (colon) ctx->zi_n_refiner = atoi(colon + 1);
                }

                /* cap_feat_dim */
                ctx->zi_cap_feat_dim = 2560;
                if ((p = strstr(buf, "\"cap_feat_dim\""))) {
                    char *colon = strchr(p, ':');
                    if (colon) ctx->zi_cap_feat_dim = atoi(colon + 1);
                }

                /* in_channels */
                ctx->zi_in_channels = 16;
                if ((p = strstr(buf, "\"in_channels\""))) {
                    char *colon = strchr(p, ':');
                    if (colon) ctx->zi_in_channels = atoi(colon + 1);
                }

                /* patch_size */
                ctx->zi_patch_size = 2;
                if ((p = strstr(buf, "\"patch_size\""))) {
                    char *colon = strchr(p, ':');
                    if (colon) ctx->zi_patch_size = atoi(colon + 1);
                }

                /* rope_theta */
                ctx->zi_rope_theta = 256.0f;
                if ((p = strstr(buf, "\"rope_theta\""))) {
                    char *colon = strchr(p, ':');
                    if (colon) ctx->zi_rope_theta = atof(colon + 1);
                }

                /* axes_dims - parse JSON array [32, 48, 48] */
                ctx->zi_axes_dims[0] = 32;
                ctx->zi_axes_dims[1] = 48;
                ctx->zi_axes_dims[2] = 48;
                if ((p = strstr(buf, "\"axes_dims\""))) {
                    char *bracket = strchr(p, '[');
                    if (bracket) {
                        ctx->zi_axes_dims[0] = atoi(bracket + 1);
                        char *comma1 = strchr(bracket, ',');
                        if (comma1) {
                            ctx->zi_axes_dims[1] = atoi(comma1 + 1);
                            char *comma2 = strchr(comma1 + 1, ',');
                            if (comma2) {
                                ctx->zi_axes_dims[2] = atoi(comma2 + 1);
                            }
                        }
                    }
                }

                /* Derived values */
                ctx->zi_latent_channels = ctx->zi_in_channels *
                                          ctx->zi_patch_size * ctx->zi_patch_size;
                ctx->text_dim = ctx->zi_cap_feat_dim;  /* 2560 for Z-Image */
            }
        }
    }

    /* Read vae/config.json for Z-Image scaling/shift factors */
    ctx->vae_z_channels = IRIS_VAE_Z_CHANNELS;  /* default: 32 */
    ctx->vae_scaling = 0.0f;
    ctx->vae_shift = 0.0f;
    snprintf(path, sizeof(path), "%s/vae/config.json", model_dir);
    if (file_exists(path)) {
        FILE *f = fopen(path, "r");
        if (f) {
            char buf[4096];
            size_t n = fread(buf, 1, sizeof(buf) - 1, f);
            buf[n] = '\0';
            fclose(f);
            char *p;
            if ((p = strstr(buf, "\"latent_channels\""))) {
                char *colon = strchr(p, ':');
                if (colon) {
                    int lc = atoi(colon + 1);
                    if (lc > 0) ctx->vae_z_channels = lc;
                }
            }
            if ((p = strstr(buf, "\"scaling_factor\""))) {
                char *colon = strchr(p, ':');
                if (colon) ctx->vae_scaling = atof(colon + 1);
            }
            if ((p = strstr(buf, "\"shift_factor\""))) {
                char *colon = strchr(p, ':');
                if (colon) ctx->vae_shift = atof(colon + 1);
            }
        }
    }

    /* Determine model variant name based on architecture. */
    if (ctx->is_zimage) {
        int hidden_size = ctx->zi_dim;
        const char *size_label = "6B";  /* Z-Image-Turbo is 6B */
        if (hidden_size != 3840) {
            size_label = (hidden_size > 3840) ? "large" : "small";
        }
        ctx->is_non_commercial = 0;  /* Z-Image is Apache 2.0 */
        ctx->num_heads = num_heads;
        ctx->default_steps = 9;       /* 8 NFE = 9 scheduler steps */
        ctx->default_guidance = 0.0f; /* No CFG for Z-Image-Turbo */
        ctx->is_distilled = 1;        /* Treat as distilled (no CFG) */
        snprintf(ctx->model_name, sizeof(ctx->model_name),
                 "Z-Image-Turbo-%s", size_label);
    } else if (ctx->is_flux3) {
        ctx->num_heads = num_heads;
        ctx->is_non_commercial = 1;   /* FLUX Kommunity License */
        ctx->default_steps = ctx->is_distilled ? 16 : 50;
        ctx->default_guidance = ctx->is_distilled ? 0.0f : 3.5f;
        snprintf(ctx->model_name, sizeof(ctx->model_name),
                 ctx->is_distilled ? "FLUX.3-image-klein" : "FLUX.3-image-klein-base");
    } else {
        int hidden_size = num_heads * 128;  /* head_dim is always 128 */
        const char *size_label = (hidden_size > 3072) ? "9B" : "4B";
        ctx->is_non_commercial = (hidden_size > 3072) ? 1 : 0;
        ctx->num_heads = num_heads;

        if (ctx->is_distilled) {
            ctx->default_steps = 4;
            ctx->default_guidance = 1.0f;
            snprintf(ctx->model_name, sizeof(ctx->model_name),
                     "FLUX.2-klein-%s", size_label);
        } else {
            ctx->default_steps = 50;
            ctx->default_guidance = 4.0f;
            snprintf(ctx->model_name, sizeof(ctx->model_name),
                     "FLUX.2-klein-base-%s", size_label);
        }
    }

    /* Load VAE only at startup (~300MB).
     * Transformer and text encoder are loaded on-demand during generation
     * to support systems with limited RAM (e.g., 16GB). */
    snprintf(path, sizeof(path), "%s/vae/diffusion_pytorch_model.safetensors", model_dir);
    if (file_exists(path)) {
        safetensors_file_t *sf = safetensors_open(path);
        if (sf) {
            ctx->vae = iris_vae_load_safetensors_ex(sf,
                ctx->vae_z_channels, ctx->vae_scaling, ctx->vae_shift,
                ctx->allow_large_images);
            safetensors_close(sf);
        }
    }

    /* Verify VAE is loaded */
    if (!ctx->vae) {
        set_error("Failed to load VAE - cannot generate images");
        iris_free(ctx);
        return NULL;
    }

    /* Verify transformer dir exists (will be loaded on-demand) */
    snprintf(path, sizeof(path), "%s/transformer/config.json", model_dir);
    if (!file_exists(path)) {
        /* Fallback: check for single safetensors file */
        snprintf(path, sizeof(path), "%s/transformer/diffusion_pytorch_model.safetensors", model_dir);
        if (!file_exists(path)) {
            set_error("Transformer model not found (missing config.json and safetensors)");
            iris_free(ctx);
            return NULL;
        }
    }
    /* Text encoder and transformer are loaded on-demand to reduce peak memory. */

    /* Initialize RNG */
    iris_rng_seed((uint64_t)time(NULL));

    return ctx;
}

void iris_free(iris_ctx *ctx) {
    if (!ctx) return;

    iris_tokenizer_free(ctx->tokenizer);
    qwen3_encoder_free(ctx->qwen3_encoder);
    iris_vae_free(ctx->vae);
    iris_transformer_free_flux(ctx->transformer);
    iris_transformer_free_zimage(ctx->zi_transformer);
    iris_transformer_free_flux3(ctx->flux3_transformer);

    free(ctx);
}

void iris_set_mmap(iris_ctx *ctx, int enable) {
    if (ctx) ctx->use_mmap = enable;
}

int iris_is_distilled(iris_ctx *ctx) {
    return ctx ? ctx->is_distilled : 1;
}

int iris_is_zimage(iris_ctx *ctx) {
    return ctx ? ctx->is_zimage : 0;
}

int iris_is_flux3(iris_ctx *ctx) {
    return ctx ? ctx->is_flux3 : 0;
}

int iris_default_steps(iris_ctx *ctx) {
    return ctx ? ctx->default_steps : 4;
}

float iris_default_guidance(iris_ctx *ctx) {
    return ctx ? ctx->default_guidance : 1.0f;
}

void iris_set_base_mode(iris_ctx *ctx) {
    /* FLUX.3 checkpoints declare whether they are flow maps (distilled). */
    if (!ctx || ctx->is_flux3) return;
    ctx->is_distilled = 0;
    ctx->default_steps = 50;
    ctx->default_guidance = 4.0f;
    const char *size_label = ctx->is_non_commercial ? "9B" : "4B";
    snprintf(ctx->model_name, sizeof(ctx->model_name),
             "FLUX.2-klein-base-%s", size_label);
}

/* Free the Qwen3 text encoder (~4-8GB) to make room for the transformer.
 * The encoder and transformer can't coexist in memory on most machines,
 * so this is called after text encoding and before denoising. On Metal,
 * also resets all GPU state (weight caches, pools) to avoid stale data
 * when the transformer loads into the same memory regions. */
void iris_release_text_encoder(iris_ctx *ctx) {
    if (!ctx || !ctx->qwen3_encoder) return;

    qwen3_encoder_free(ctx->qwen3_encoder);
    ctx->qwen3_encoder = NULL;

#ifdef USE_METAL
    /* Reset all GPU state to ensure clean slate for transformer.
     * This clears weight caches, activation pools, and pending commands. */
    iris_metal_reset();
#endif
}

/* Lazy-load the Flux transformer from safetensors files. Deferred to
 * generation time because the text encoder must be freed first -- both
 * are too large to fit in memory simultaneously. Once loaded, the
 * transformer persists across generations (no reload per image). */
static int iris_load_transformer_if_needed(iris_ctx *ctx) {
    if (ctx->transformer) return 1;  /* Already loaded */

    if (iris_phase_callback) iris_phase_callback("Loading FLUX.2 transformer", 0);
    if (ctx->use_mmap) {
        ctx->transformer = iris_transformer_load_safetensors_mmap_flux(ctx->model_dir);
    } else {
        ctx->transformer = iris_transformer_load_safetensors_flux(ctx->model_dir);
    }
    if (iris_phase_callback) iris_phase_callback("Loading FLUX.2 transformer", 1);

    if (!ctx->transformer) {
        set_error("Failed to load transformer");
        return 0;
    }
    return 1;
}

/* Load FLUX.3 transformer on-demand if not already loaded */
static int iris_load_flux3_transformer_if_needed(iris_ctx *ctx) {
    if (ctx->flux3_transformer) return 1;

    if (iris_phase_callback) iris_phase_callback("Loading FLUX.3 transformer", 0);
    ctx->flux3_transformer = iris_transformer_load_flux3(ctx->model_dir);
    if (iris_phase_callback) iris_phase_callback("Loading FLUX.3 transformer", 1);

    if (!ctx->flux3_transformer) {
        set_error("Failed to load FLUX.3 transformer");
        return 0;
    }
    return 1;
}

/* Load Z-Image transformer on-demand if not already loaded */
static int iris_load_zimage_transformer_if_needed(iris_ctx *ctx) {
    if (ctx->zi_transformer) return 1;  /* Already loaded */

    if (iris_phase_callback) iris_phase_callback("Loading Z-Image transformer", 0);
    ctx->zi_transformer = zi_transformer_load_safetensors(
        ctx->model_dir,
        ctx->zi_dim, ctx->zi_dim / 128, ctx->zi_n_layers, ctx->zi_n_refiner,
        ctx->zi_cap_feat_dim, ctx->zi_in_channels, ctx->zi_patch_size,
        ctx->zi_rope_theta, ctx->zi_axes_dims);
    if (iris_phase_callback) iris_phase_callback("Loading Z-Image transformer", 1);

    if (!ctx->zi_transformer) {
        set_error("Failed to load Z-Image transformer");
        return 0;
    }
    return 1;
}

/* Get transformer for debugging */
void *iris_get_transformer(iris_ctx *ctx) {
    if (!ctx) return NULL;
    if (ctx->is_zimage) return ctx->zi_transformer;
    if (ctx->is_flux3) return ctx->flux3_transformer;
    return ctx->transformer;
}

/* ========================================================================
 * Text Encoding
 * ======================================================================== */

/* Run the prompt through Qwen3 to produce text embeddings. For Flux models,
 * hidden states from layers 8, 17, 26 are concatenated to form [512, text_dim].
 * For Z-Image, takes hidden_states[-2] and reports the real (unpadded) token
 * count via out_seq_len. The returned embedding buffer is still max-seq padded;
 * Z-Image consumes only the first out_seq_len tokens. */
float *iris_encode_text(iris_ctx *ctx, const char *prompt, int *out_seq_len) {
    if (!ctx || !prompt) {
        *out_seq_len = 0;
        return NULL;
    }

    /* Load encoder if not already loaded */
    if (!ctx->qwen3_encoder && ctx->model_dir[0]) {
        if (iris_phase_callback) iris_phase_callback("Loading Qwen3 encoder", 0);
        ctx->qwen3_encoder = qwen3_encoder_load(ctx->model_dir, ctx->use_mmap);
        if (iris_phase_callback) iris_phase_callback("Loading Qwen3 encoder", 1);
        if (!ctx->qwen3_encoder) {
            fprintf(stderr, "Warning: Failed to load Qwen3 text encoder\n");
        }
    }

    if (!ctx->qwen3_encoder) {
        if (ctx->is_zimage || ctx->is_flux3) {
            /* The sequence length depends on the prompt's token count. */
            *out_seq_len = 0;
            set_error("Qwen3 text encoder unavailable");
            return NULL;
        }
        /* Flux fallback: return zero padded embeddings. */
        *out_seq_len = QWEN3_MAX_SEQ_LEN;
        return (float *)calloc(QWEN3_MAX_SEQ_LEN * ctx->text_dim, sizeof(float));
    }

    /* Set extraction mode: which hidden states form the embeddings */
    qwen3_set_extraction_mode(ctx->qwen3_encoder,
        ctx->is_zimage ? QWEN3_EXTRACT_ZIMAGE :
        ctx->is_flux3 ? QWEN3_EXTRACT_FLUX3 : QWEN3_EXTRACT_FLUX);

    /* Encode text using Qwen3 */
    if (iris_phase_callback) iris_phase_callback("encoding text", 0);

    int num_real_tokens = 0, padded_len = 0;
    float *embeddings = qwen3_encode_text_ex(ctx->qwen3_encoder, prompt,
                                               &num_real_tokens, &padded_len);
    if (iris_phase_callback) iris_phase_callback("encoding text", 1);

    /* Z-Image uses only the real tokens of the padded embedding buffer,
     * Flux (512) and Flux3 (multiple of 80) attend to the padding too. */
    *out_seq_len = ctx->is_zimage ? num_real_tokens : padded_len;
    return embeddings;
}

/* ========================================================================
 * Z-Image Generation
 * ======================================================================== */

/* Z-Image txt2img pipeline with pre-computed embeddings. Unlike Flux which
 * works in post-patchification space [128, H/16, W/16], Z-Image initializes
 * noise at pre-patchification dimensions [16, H/8, W/8] and the transformer
 * operates there. After denoising (8 NFE from 9 scheduler steps, where the
 * last step is a no-op because sigma_min=0), the output is patchified to
 * [64, H/16, W/16] for VAE decode. */
static iris_image *iris_generate_zimage_with_embeddings(iris_ctx *ctx,
                                                          const float *text_emb,
                                                          int text_seq,
                                                          const iris_params *p_in) {
    if (!ctx || !text_emb || text_seq <= 0) {
        set_error("Invalid context or embeddings");
        return NULL;
    }

    iris_params p;
    if (p_in) {
        p = *p_in;
    } else {
        p = (iris_params)IRIS_PARAMS_DEFAULT;
    }

    /* Validate dimensions */
    if (p.width <= 0) p.width = 1024;   /* Z-Image default: 1024x1024 */
    if (p.height <= 0) p.height = 1024;
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;

    /* Ensure dimensions are divisible by 16 */
    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;
    if (p.width < 64) p.width = 64;
    if (p.height < 64) p.height = 64;
    if (p.width > ctx->max_width || p.height > ctx->max_height) {
        image_size_error(ctx);
        return NULL;
    }
    if (!prepare_vae(ctx, p.width, p.height)) return NULL;

    /* Release text encoder to free memory before loading transformer */
    iris_release_text_encoder(ctx);

    /* Load Z-Image transformer on-demand (persistent across generations). */
    if (!iris_load_zimage_transformer_if_needed(ctx)) {
        return NULL;
    }

    /* Z-Image latent dimensions:
     * The transformer works at pre-patchification: [in_ch, H/8, W/8]
     * where in_ch=16, and patchification happens inside the transformer.
     * VAE decode expects post-patchification: [latent_ch, H/16, W/16]
     * where latent_ch = in_ch * ps * ps = 64. */
    int ps = ctx->zi_patch_size;  /* 2 */
    int pre_h = p.height / 8;    /* H/8: pre-patchification spatial */
    int pre_w = p.width / 8;
    int in_ch = ctx->zi_in_channels;  /* 16 */
    int post_h = pre_h / ps;     /* H/16: post-patchification spatial */
    int post_w = pre_w / ps;
    int image_seq_len = post_h * post_w;

    /* Initialize noise at pre-patchification dimensions: [in_ch, H/8, W/8] */
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = iris_init_noise(1, in_ch, pre_h, pre_w, seed);

    /* Get Z-Image schedule (default FlowMatch; linear/power if explicitly requested). */
    float *schedule = iris_selected_zimage_schedule(&p, image_seq_len);

    /* Sample using Z-Image Euler method.
     * The transformer takes [in_ch, pre_h, pre_w] and returns same shape. */
    float *denoised = iris_sample_euler_zimage(
        ctx->zi_transformer, z, 1, in_ch, pre_h, pre_w,
        ps,
        text_emb, text_seq,
        schedule, p.num_steps,
        NULL
    );

    free(z);
    free(schedule);

    if (!denoised) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Patchify transformer output for VAE decode:
     * [1, in_ch, H/8, W/8] -> [1, latent_ch, H/16, W/16]
     * where latent_ch = in_ch * ps * ps = 64 */
    int latent_ch = in_ch * ps * ps;
    float *latent = (float *)malloc(latent_ch * post_h * post_w * sizeof(float));
    iris_patchify(latent, denoised, 1, in_ch, pre_h, pre_w, ps);
    free(denoised);

    /* Decode latent to image */
    iris_image *img = NULL;
    if (ctx->vae) {
        if (iris_phase_callback) iris_phase_callback("decoding image", 0);
        img = iris_vae_decode(ctx->vae, latent, 1, post_h, post_w);
        if (iris_phase_callback) iris_phase_callback("decoding image", 1);
    }

    free(latent);
    return img;
}

static iris_image *iris_generate_zimage(iris_ctx *ctx, const char *prompt,
                                          const iris_params *p_in) {
    /* Encode text (Z-Image mode: extraction mode 1, single layer) */
    int text_seq;
    float *text_emb = iris_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        set_error("Failed to encode prompt");
        return NULL;
    }

    iris_image *img = iris_generate_zimage_with_embeddings(ctx, text_emb, text_seq, p_in);
    free(text_emb);
    return img;
}

/* ========================================================================
 * FLUX.3 Generation
 * ======================================================================== */

/* FLUX.3 txt2img from pre-computed embeddings: denoise a noise latent
 * [128, H/16, W/16] with the FLUX.3 Euler sampler (CFG when uncond_emb is
 * given) and decode it with the (FLUX.2) VAE. */
static iris_image *iris_generate_flux3_with_embeddings(iris_ctx *ctx,
                                                       const float *text_emb, int text_seq,
                                                       const float *uncond_emb, int uncond_seq,
                                                       const iris_params *params) {
    iris_params p = params ? *params : (iris_params)IRIS_PARAMS_DEFAULT;
    if (p.width <= 0) p.width = IRIS_DEFAULT_WIDTH;
    if (p.height <= 0) p.height = IRIS_DEFAULT_HEIGHT;
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;
    float guidance = (p.guidance > 0) ? p.guidance : ctx->default_guidance;

    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;
    if (p.width < 64) p.width = 64;
    if (p.height < 64) p.height = 64;
    if (p.width > ctx->max_width || p.height > ctx->max_height) {
        image_size_error(ctx);
        return NULL;
    }
    if (!prepare_vae(ctx, p.width, p.height)) return NULL;

    iris_release_text_encoder(ctx);
    if (!iris_load_flux3_transformer_if_needed(ctx)) return NULL;

    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = iris_init_noise(1, IRIS_LATENT_CHANNELS, latent_h, latent_w, seed);
    float *schedule = iris_selected_flux3_schedule(&p, latent_h * latent_w);

    float *latent = iris_sample_euler_flux3(ctx->flux3_transformer, z,
                                            IRIS_LATENT_CHANNELS, latent_h, latent_w,
                                            text_emb, text_seq, uncond_emb, uncond_seq,
                                            NULL, NULL, NULL, 0,
                                            schedule, p.num_steps, guidance, NULL);
    free(z);
    free(schedule);
    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    if (iris_phase_callback) iris_phase_callback("decoding image", 0);
    iris_image *img = iris_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
    if (iris_phase_callback) iris_phase_callback("decoding image", 1);
    free(latent);
    return img;
}

/* FLUX.3 reference image preparation, following the official image
 * processor. One reference is kept whole: downscaled so that its canvas
 * (its size rounded up to the 16-pixel token grid) fits under 4 megapixels,
 * then reflect-padded to the canvas; crop receives the box of the image in
 * the canvas. Each of several references is downscaled under 1 megapixel
 * and center-cropped to the grid. Sides are capped at the VAE maximum
 * (the official cap is 4096). */
#define FLUX3_REF_AREA_SINGLE (4 * 1024 * 1024)
#define FLUX3_REF_AREA_MULTI (1024 * 1024)

static int flux3_canvas_side(int side, int max_dim) {
    int c = (side + 15) / 16 * 16;
    if (c < 64) c = 64;
    return c > max_dim ? max_dim : c;
}

/* Index into [0, n) for a padded coordinate: numpy "reflect" (edge pixel
 * not repeated), or "symmetric" (repeated) for tiny images. */
static int flux3_pad_index(int i, int n, int symmetric) {
    if (n == 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = symmetric ? -i - 1 : -i;
        if (i >= n) i = symmetric ? 2 * n - 1 - i : 2 * (n - 1) - i;
    }
    return i;
}

static iris_image *flux3_prepare_reference(const iris_image *img, int single, int crop[4], int max_dim) {
    int w = img->width, h = img->height, ch = img->channels;
    int nw = w, nh = h;
    double scale;

    if (single) {
        scale = sqrt((double)FLUX3_REF_AREA_SINGLE / ((double)w * h));
        if (scale > 1.0) scale = 1.0;
        if (scale > (double)max_dim / (w > h ? w : h))
            scale = (double)max_dim / (w > h ? w : h);
        for (;;) {
            nw = (int)lround(w * scale);
            nh = (int)lround(h * scale);
            if (nw < 1) nw = 1;
            if (nh < 1) nh = 1;
            if ((long)flux3_canvas_side(nw, max_dim) * flux3_canvas_side(nh, max_dim) <= FLUX3_REF_AREA_SINGLE) break;
            scale *= 0.995;
        }
    } else {
        scale = 1.0;
        if ((long)w * h > FLUX3_REF_AREA_MULTI) scale = sqrt((double)FLUX3_REF_AREA_MULTI / ((double)w * h));
        if (scale > (double)max_dim / (w > h ? w : h))
            scale = (double)max_dim / (w > h ? w : h);
        if (scale < 1.0) {
            nw = (int)(w * scale);
            nh = (int)(h * scale);
        }
        if (nw < 16) nw = 16;
        if (nh < 16) nh = 16;
    }
    iris_image *src = (nw == w && nh == h) ? NULL : iris_image_resize(img, nw, nh);
    const iris_image *s = src ? src : img;
    if (src == NULL && (nw != w || nh != h)) return NULL;

    /* Target size and the source box inside it. */
    int tw, th, left, top;
    if (single) {
        tw = flux3_canvas_side(nw, max_dim);
        th = flux3_canvas_side(nh, max_dim);
        left = (tw - nw) / 2;
        top = (th - nh) / 2;
        crop[0] = left;
        crop[1] = top;
        crop[2] = left + nw;
        crop[3] = top + nh;
    } else {
        tw = nw - nw % 16;
        th = nh - nh % 16;
        left = -((nw - tw) / 2);
        top = -((nh - th) / 2);
    }
    iris_image *out = iris_image_create(tw, th, 3);
    if (out) {
        int symmetric = (nw < nh ? nw : nh) <= 16;
        for (int y = 0; y < th; y++) {
            int sy = flux3_pad_index(y - top, nh, symmetric);
            for (int x = 0; x < tw; x++) {
                int sx = flux3_pad_index(x - left, nw, symmetric);
                const uint8_t *px = s->data + ((size_t)sy * nw + sx) * ch;
                uint8_t *o = out->data + ((size_t)y * tw + x) * 3;
                for (int c = 0; c < 3; c++) o[c] = px[ch >= 3 ? c : 0];
            }
        }
    }
    iris_image_free(src);
    return out;
}

/* FLUX.3 image-conditioned generation: the prepared references are
 * VAE-encoded and passed to the transformer as extra tokens. Without an
 * explicit size the output takes the first reference's size and, for a
 * single padded reference, is cropped back to the image. */
static iris_image *iris_generate_flux3_refs(iris_ctx *ctx, const char *prompt,
                                            const iris_image **refs, int num_refs,
                                            const iris_params *params) {
    iris_params p = params ? *params : (iris_params)IRIS_PARAMS_DEFAULT;
    int size_given = p.width > 0 && p.height > 0;
    int crop[4] = {0, 0, 0, 0};
    float **lat = calloc(num_refs, sizeof(float *));
    int *lh = calloc(num_refs, sizeof(int)), *lw = calloc(num_refs, sizeof(int));
    iris_image **prepared = NULL;
    iris_image *img = NULL;
    float *text_emb = NULL, *uncond_emb = NULL;
    if (!lat || !lh || !lw) goto done;

    if (ctx->allow_large_images) {
        /* Plan one capacity for output and prepared references before encoding.
         * Only the largest area matters because the VAE processes them in turn. */
        int max_w = 0, max_h = 0;
        if (size_given) {
            p.width = p.width / 16 * 16;
            p.height = p.height / 16 * 16;
            if (p.width < 64) p.width = 64;
            if (p.height < 64) p.height = 64;
            if (p.width > ctx->max_width || p.height > ctx->max_height) {
                image_size_error(ctx);
                goto done;
            }
            max_w = p.width;
            max_h = p.height;
        }
        prepared = calloc(num_refs, sizeof(*prepared));
        if (!prepared) goto done;
        for (int i = 0; i < num_refs; i++) {
            prepared[i] = flux3_prepare_reference(refs[i], num_refs == 1, crop,
                                                  ctx->max_width);
            if (!prepared[i]) {
                set_error("Failed to prepare reference image");
                goto done;
            }
            if ((size_t)prepared[i]->width * prepared[i]->height > (size_t)max_w * max_h) {
                max_w = prepared[i]->width;
                max_h = prepared[i]->height;
            }
        }
        if (!prepare_vae(ctx, max_w, max_h)) goto done;
    }

    if (iris_phase_callback) iris_phase_callback("encoding reference images", 0);
    for (int i = 0; i < num_refs; i++) {
        iris_image *prep = prepared ? prepared[i] :
            flux3_prepare_reference(refs[i], num_refs == 1, crop, ctx->max_width);
        if (!prep) {
            set_error("Failed to prepare reference image");
            goto done;
        }
        float *tensor = iris_image_to_tensor(prep);
        lat[i] = iris_vae_encode(ctx->vae, tensor, 1, prep->height, prep->width, &lh[i], &lw[i]);
        free(tensor);
        iris_image_free(prep);
        if (prepared) prepared[i] = NULL;
        if (!lat[i]) {
            set_error("Failed to encode reference image");
            goto done;
        }
    }
    if (iris_phase_callback) iris_phase_callback("encoding reference images", 1);

    if (!size_given) {
        p.width = lw[0] * 16;
        p.height = lh[0] * 16;
    }
    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;
    if (p.width < 64) p.width = 64;
    if (p.height < 64) p.height = 64;
    if (p.width > ctx->max_width || p.height > ctx->max_height) {
        image_size_error(ctx);
        goto done;
    }
    if (!prepare_vae(ctx, p.width, p.height)) goto done;
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;
    float guidance = (p.guidance > 0) ? p.guidance : ctx->default_guidance;

    int text_seq, uncond_seq = 0;
    text_emb = iris_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        set_error("Failed to encode prompt");
        goto done;
    }
    if (!ctx->is_distilled) {
        uncond_emb = iris_encode_text(ctx, "", &uncond_seq);
        if (!uncond_emb) {
            set_error("Failed to encode empty prompt for CFG");
            goto done;
        }
    }
    iris_release_text_encoder(ctx);
    if (!iris_load_flux3_transformer_if_needed(ctx)) goto done;

    int latent_h = p.height / 16, latent_w = p.width / 16;
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = iris_init_noise(1, IRIS_LATENT_CHANNELS, latent_h, latent_w, seed);
    float *schedule = iris_selected_flux3_schedule(&p, latent_h * latent_w);
    float *latent = iris_sample_euler_flux3(ctx->flux3_transformer, z,
                                            IRIS_LATENT_CHANNELS, latent_h, latent_w,
                                            text_emb, text_seq, uncond_emb, uncond_seq,
                                            (const float *const *)lat, lh, lw, num_refs,
                                            schedule, p.num_steps, guidance, NULL);
    free(z);
    free(schedule);
    if (!latent) {
        set_error("Sampling failed");
        goto done;
    }
    if (iris_phase_callback) iris_phase_callback("decoding image", 0);
    img = iris_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
    if (iris_phase_callback) iris_phase_callback("decoding image", 1);
    free(latent);

    /* A single padded reference: crop the output back to the image. */
    if (img && num_refs == 1 && !size_given &&
        (crop[2] - crop[0] != img->width || crop[3] - crop[1] != img->height)) {
        iris_image *c = iris_image_create(crop[2] - crop[0], crop[3] - crop[1], img->channels);
        if (c) {
            for (int y = 0; y < c->height; y++)
                memcpy(c->data + (size_t)y * c->width * c->channels,
                       img->data + ((size_t)(y + crop[1]) * img->width + crop[0]) * img->channels,
                       (size_t)c->width * c->channels);
            iris_image_free(img);
            img = c;
        }
    }

done:
    if (prepared) {
        for (int i = 0; i < num_refs; i++) iris_image_free(prepared[i]);
        free(prepared);
    }
    if (lat)
        for (int i = 0; i < num_refs; i++) free(lat[i]);
    free(lat);
    free(lh);
    free(lw);
    free(text_emb);
    free(uncond_emb);
    return img;
}

/* FLUX.3 txt2img: encode the prompt (and, for base models, the empty
 * prompt used by CFG), then generate from the embeddings. */
static iris_image *iris_generate_flux3(iris_ctx *ctx, const char *prompt,
                                       const iris_params *params) {
    int text_seq, uncond_seq = 0;
    float *text_emb = iris_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        set_error("Failed to encode prompt");
        return NULL;
    }
    float *uncond_emb = NULL;
    if (!ctx->is_distilled) {
        uncond_emb = iris_encode_text(ctx, "", &uncond_seq);
        if (!uncond_emb) {
            free(text_emb);
            set_error("Failed to encode empty prompt for CFG");
            return NULL;
        }
    }
    iris_image *img = iris_generate_flux3_with_embeddings(ctx, text_emb, text_seq,
                                                          uncond_emb, uncond_seq, params);
    free(text_emb);
    free(uncond_emb);
    return img;
}

/* ========================================================================
 * Image Generation
 * ======================================================================== */

/* Main text-to-image entry point. Routes to Z-Image or Flux pipeline based
 * on model type. For Flux: encodes text via Qwen3, frees the encoder, loads
 * the transformer, initializes Gaussian noise in latent space, then runs
 * Euler ODE denoising (4 steps distilled / 50 steps base with CFG) followed
 * by VAE decode. For base models, an empty-prompt encoding is also produced
 * for Classifier-Free Guidance (two sequential transformer passes per step). */
iris_image *iris_generate(iris_ctx *ctx, const char *prompt,
                          const iris_params *params) {
    if (!ctx || !prompt) {
        set_error("Invalid context or prompt");
        return NULL;
    }

    /* Route to Z-Image / FLUX.3 pipelines if appropriate */
    if (ctx->is_zimage) {
        return iris_generate_zimage(ctx, prompt, params);
    }
    if (ctx->is_flux3) {
        return iris_generate_flux3(ctx, prompt, params);
    }

    /* Use defaults if params is NULL */
    iris_params p;
    if (params) {
        p = *params;
    } else {
        p = (iris_params)IRIS_PARAMS_DEFAULT;
    }

    /* Validate dimensions */
    if (p.width <= 0) p.width = IRIS_DEFAULT_WIDTH;
    if (p.height <= 0) p.height = IRIS_DEFAULT_HEIGHT;
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;
    float guidance = (p.guidance > 0) ? p.guidance : ctx->default_guidance;

    /* Ensure dimensions are divisible by 16 */
    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;
    if (p.width < 64) p.width = 64;
    if (p.height < 64) p.height = 64;
    if (p.width > ctx->max_width || p.height > ctx->max_height) {
        image_size_error(ctx);
        return NULL;
    }
    if (!prepare_vae(ctx, p.width, p.height)) return NULL;

    /* Encode text (and unconditioned text for CFG in base model) */
    int text_seq;
    float *text_emb = iris_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        set_error("Failed to encode prompt");
        return NULL;
    }

    float *text_emb_uncond = NULL;
    int text_seq_uncond = 0;
    if (!ctx->is_distilled) {
        text_emb_uncond = iris_encode_text(ctx, "", &text_seq_uncond);
        if (!text_emb_uncond) {
            free(text_emb);
            set_error("Failed to encode empty prompt for CFG");
            return NULL;
        }
    }

    /* Release text encoder to free ~8GB before loading transformer */
    iris_release_text_encoder(ctx);

    /* Load transformer now (after text encoder is freed to reduce peak memory) */
    if (!iris_load_transformer_if_needed(ctx)) {
        free(text_emb);
        free(text_emb_uncond);
        return NULL;
    }

    /* Compute latent dimensions */
    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;

    /* Initialize noise */
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = iris_init_noise(1, IRIS_LATENT_CHANNELS, latent_h, latent_w, seed);

    /* Get schedule */
    float *schedule = iris_selected_schedule(&p, image_seq_len);

    /* Sample */
    float *latent;
    if (ctx->is_distilled) {
        latent = iris_sample_euler_flux(
            ctx->transformer, ctx->qwen3_encoder,
            z, 1, IRIS_LATENT_CHANNELS, latent_h, latent_w,
            text_emb, text_seq,
            schedule, p.num_steps,
            NULL
        );
    } else {
        latent = iris_sample_euler_cfg_flux(
            ctx->transformer, ctx->qwen3_encoder,
            z, 1, IRIS_LATENT_CHANNELS, latent_h, latent_w,
            text_emb, text_seq,
            text_emb_uncond, text_seq_uncond,
            guidance,
            schedule, p.num_steps,
            NULL
        );
    }

    free(z);
    free(schedule);
    free(text_emb);
    free(text_emb_uncond);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode latent to image */
    iris_image *img = NULL;
    if (ctx->vae) {
        if (iris_phase_callback) iris_phase_callback("decoding image", 0);
        img = iris_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (iris_phase_callback) iris_phase_callback("decoding image", 1);
    }

    free(latent);

    return img;
}

/* ========================================================================
 * Generation with Pre-computed Embeddings
 * ======================================================================== */

/* Generate from pre-computed text embeddings, skipping the Qwen3 encoding
 * step. Useful for the embedding cache (repeat prompts without re-encoding).
 * Only supports distilled models and Z-Image -- base model CFG requires two
 * separate embeddings (conditioned + empty prompt) which this API doesn't
 * provide, so it would produce incorrect results. */
iris_image *iris_generate_with_embeddings(iris_ctx *ctx,
                                           const float *text_emb, int text_seq,
                                           const iris_params *params) {
    if (!ctx || !text_emb) {
        set_error("Invalid context or embeddings");
        return NULL;
    }

    if (ctx->is_zimage) {
        return iris_generate_zimage_with_embeddings(ctx, text_emb, text_seq, params);
    }

    /* This API only supports the distilled (non-CFG) sampler since the
     * caller provides a single embedding.  Warn if used with a base model
     * because results will be incorrect without CFG. */
    if (!ctx->is_distilled) {
        fprintf(stderr, "Warning: iris_generate_with_embeddings() does not "
                        "support CFG. Use iris_generate() for base models.\n");
    }

    if (ctx->is_flux3) {
        return iris_generate_flux3_with_embeddings(ctx, text_emb, text_seq,
                                                   NULL, 0, params);
    }

    /* Load transformer if not already loaded */
    if (!iris_load_transformer_if_needed(ctx)) {
        return NULL;
    }

    iris_params p;
    if (params) {
        p = *params;
    } else {
        p = (iris_params)IRIS_PARAMS_DEFAULT;
    }

    /* Validate dimensions */
    if (p.width <= 0) p.width = IRIS_DEFAULT_WIDTH;
    if (p.height <= 0) p.height = IRIS_DEFAULT_HEIGHT;
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;

    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;
    if (p.width < 64) p.width = 64;
    if (p.height < 64) p.height = 64;
    if (p.width > ctx->max_width || p.height > ctx->max_height) {
        image_size_error(ctx);
        return NULL;
    }
    if (!prepare_vae(ctx, p.width, p.height)) return NULL;

    /* Compute latent dimensions */
    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;

    /* Initialize noise */
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = iris_init_noise(1, IRIS_LATENT_CHANNELS, latent_h, latent_w, seed);

    /* Get schedule */
    float *schedule = iris_selected_schedule(&p, image_seq_len);

    /* Sample - note: pre-computed embeddings only support distilled path.
     * CFG requires two embeddings which the caller doesn't provide. */
    float *latent = iris_sample_euler_flux(
        ctx->transformer, ctx->qwen3_encoder,
        z, 1, IRIS_LATENT_CHANNELS, latent_h, latent_w,
        text_emb, text_seq,
        schedule, p.num_steps,
        NULL  /* progress_callback */
    );

    free(z);
    free(schedule);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode latent to image */
    iris_image *img = NULL;
    if (ctx->vae) {
        if (iris_phase_callback) iris_phase_callback("decoding image", 0);
        img = iris_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (iris_phase_callback) iris_phase_callback("decoding image", 1);
    } else {
        set_error("No VAE loaded");
        free(latent);
        return NULL;
    }

    free(latent);
    return img;
}

/* Generate with external embeddings and external noise */
iris_image *iris_generate_with_embeddings_and_noise(iris_ctx *ctx,
                                                     const float *text_emb, int text_seq,
                                                     const float *noise, int noise_size,
                                                     const iris_params *params) {
    if (!ctx || !text_emb || !noise) {
        set_error("Invalid context, embeddings, or noise");
        return NULL;
    }

    if (ctx->is_zimage || ctx->is_flux3) {
        set_error("This model does not support the external embedding/noise "
                  "generation API. Use iris_generate() with a prompt.");
        return NULL;
    }

    /* Load transformer if not already loaded */
    if (!iris_load_transformer_if_needed(ctx)) {
        return NULL;
    }

    iris_params p;
    if (params) {
        p = *params;
    } else {
        p = (iris_params)IRIS_PARAMS_DEFAULT;
    }

    /* Validate dimensions */
    if (p.width <= 0) p.width = IRIS_DEFAULT_WIDTH;
    if (p.height <= 0) p.height = IRIS_DEFAULT_HEIGHT;
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;

    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;
    if (p.width < 64) p.width = 64;
    if (p.height < 64) p.height = 64;
    if (p.width > ctx->max_width || p.height > ctx->max_height) {
        image_size_error(ctx);
        return NULL;
    }
    if (!prepare_vae(ctx, p.width, p.height)) return NULL;

    /* Compute latent dimensions */
    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;
    int expected_noise_size = IRIS_LATENT_CHANNELS * latent_h * latent_w;

    if (noise_size != expected_noise_size) {
        char err[256];
        snprintf(err, sizeof(err), "Noise size mismatch: got %d, expected %d",
                 noise_size, expected_noise_size);
        set_error(err);
        return NULL;
    }

    /* Copy external noise */
    float *z = (float *)malloc(expected_noise_size * sizeof(float));
    memcpy(z, noise, expected_noise_size * sizeof(float));

    /* Get schedule */
    float *schedule = iris_selected_schedule(&p, image_seq_len);

    /* Sample */
    float *latent = iris_sample_euler_flux(
        ctx->transformer, ctx->qwen3_encoder,
        z, 1, IRIS_LATENT_CHANNELS, latent_h, latent_w,
        text_emb, text_seq,
        schedule, p.num_steps,
        NULL  /* progress_callback */
    );

    free(z);
    free(schedule);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode latent to image */
    iris_image *img = NULL;
    if (ctx->vae) {
        if (iris_phase_callback) iris_phase_callback("decoding image", 0);
        img = iris_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (iris_phase_callback) iris_phase_callback("decoding image", 1);
    } else {
        set_error("No VAE loaded");
        free(latent);
        return NULL;
    }

    free(latent);
    return img;
}

/* ========================================================================
 * Attention Memory Budget
 * ======================================================================== */

/* 4 GB — MPSTemporaryNDArray hard limit. */
#define ATTENTION_MAX_BYTES ((size_t)4ULL << 30)

/* Compute worst-case attention matrix size in bytes.
 * All image dimensions are in pixels (multiples of 16).
 * ref_dims is [h0, w0, h1, w1, ...] in pixels. */
static size_t attention_bytes(int num_heads,
                              int out_h, int out_w,
                              const int *ref_dims, int num_refs,
                              int txt_seq)
{
    size_t total_seq = (size_t)(out_h / 16) * (out_w / 16);
    for (int i = 0; i < num_refs; i++)
        total_seq += (size_t)(ref_dims[i*2] / 16) * (ref_dims[i*2+1] / 16);
    total_seq += txt_seq;
    return (size_t)num_heads * total_seq * total_seq * sizeof(float);
}

/* Shrink reference pixel dimensions so manual attention fits under 4 GiB.
 * ref_dims: [h0, w0, h1, w1, ...] in pixels, modified in-place.
 * Returns -1 if the output cannot fit, 1 if shrunk, or 0 if unchanged. */
static int fit_refs_for_attention(int num_heads,
                                  int out_h, int out_w,
                                  int *ref_dims, int num_refs,
                                  int txt_seq)
{
#ifdef IRIS_DEV
    if (iris_dev_has_memory_efficient_attention() &&
        !getenv("IRIS_USE_MANUAL_SDPA")) return 0;
#endif
#ifdef USE_METAL
    if (iris_metal_has_memory_efficient_attention()) return 0;
#endif
    size_t min_seq = (size_t)(out_h / 16) * (out_w / 16) + txt_seq + num_refs;
    if ((size_t)num_heads * min_seq * min_seq * sizeof(float) >= ATTENTION_MAX_BYTES) {
        set_error("Output exceeds the 4 GiB attention budget even with minimum references; "
                  "reduce output dimensions or use memory-efficient GPU attention "
                  "(unset IRIS_USE_MANUAL_SDPA)");
        return -1;
    }
    if (attention_bytes(num_heads, out_h, out_w,
                        ref_dims, num_refs, txt_seq) < ATTENTION_MAX_BYTES)
        return 0;

    int shrunk = 0;
    for (;;) {
        /* Find reference with the most latent tokens. */
        int best = -1;
        size_t best_tok = 0;
        for (int i = 0; i < num_refs; i++) {
            size_t tok = (size_t)(ref_dims[i*2] / 16) *
                         (ref_dims[i*2+1] / 16);
            if (tok > best_tok) { best_tok = tok; best = i; }
        }
        if (best < 0 || best_tok <= 1) break;  /* can't shrink further */

        /* Scale both dimensions by 0.9, round down to multiple of 16. */
        int h = (int)(ref_dims[best*2] * 0.9f) / 16 * 16;
        int w = (int)(ref_dims[best*2+1] * 0.9f) / 16 * 16;
        if (h < 16) h = 16;
        if (w < 16) w = 16;

        /* No progress — already at minimum. */
        if (h == ref_dims[best*2] && w == ref_dims[best*2+1]) break;

        ref_dims[best*2]   = h;
        ref_dims[best*2+1] = w;
        shrunk = 1;

        if (attention_bytes(num_heads, out_h, out_w,
                            ref_dims, num_refs, txt_seq) < ATTENTION_MAX_BYTES)
            break;
    }
    return shrunk;
}

/* ========================================================================
 * Image-to-Image Generation
 * ======================================================================== */

/* Image-to-image generation via in-context conditioning. The reference image
 * is VAE-encoded into latent tokens with a RoPE T offset (T=10), while the
 * target starts from pure noise (T=0). Both are concatenated and fed to the
 * transformer, which attends to reference tokens via joint attention -- this
 * is fundamentally different from traditional img2img that adds noise to the
 * encoded image. References are dynamically resized if the resulting attention
 * matrix would exceed the 4GB MPS memory limit. */
iris_image *iris_img2img(iris_ctx *ctx, const char *prompt,
                         const iris_image *input, const iris_params *params) {
    if (!ctx || !prompt || !input) {
        set_error("Invalid parameters");
        return NULL;
    }
    if (ctx->is_zimage) {
        set_error("img2img is not supported for Z-Image");
        return NULL;
    }
    if (ctx->is_flux3) return iris_generate_flux3_refs(ctx, prompt, &input, 1, params);

    iris_params p;
    if (params) {
        p = *params;
    } else {
        p = (iris_params)IRIS_PARAMS_DEFAULT;
    }

    /* Use input image dimensions if not specified */
    if (p.width <= 0) p.width = input->width;
    if (p.height <= 0) p.height = input->height;

    /* Clamp to VAE max dimensions, preserving aspect ratio */
    if (p.width > ctx->max_width || p.height > ctx->max_height) {
        float scale = (float)ctx->max_width /
                      (p.width > p.height ? p.width : p.height);
        p.width = (int)(p.width * scale);
        p.height = (int)(p.height * scale);
    }

    /* Ensure divisible by 16 */
    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;
    /* Preserve reference resolution independently of the requested output. */
    int ref_w = input->width, ref_h = input->height;
    if (ref_w > ctx->max_width || ref_h > ctx->max_height) {
        float scale = (float)ctx->max_width / (ref_w > ref_h ? ref_w : ref_h);
        ref_w = (int)(ref_w * scale);
        ref_h = (int)(ref_h * scale);
    }
    ref_w = ref_w / 16 * 16;
    ref_h = ref_h / 16 * 16;
    if (ref_w < 16) ref_w = 16;
    if (ref_h < 16) ref_h = 16;
    {
        int ref_dims[2] = { ref_h, ref_w };
        int fit = fit_refs_for_attention(ctx->num_heads, p.height, p.width,
                                         ref_dims, 1, IRIS_MAX_SEQ_LEN);
        if (fit < 0) return NULL;
        if (fit > 0) {
            fprintf(stderr, "Note: reference image resized from %dx%d to %dx%d "
                    "(GPU attention memory limit)\n",
                    ref_w, ref_h, ref_dims[1], ref_dims[0]);
            ref_h = ref_dims[0];
            ref_w = ref_dims[1];
        }
    }
    if (iris_phase_callback)
        fprintf(stderr, "Reference image: %dx%d (output %dx%d)\n",
                ref_w, ref_h, p.width, p.height);
    if ((size_t)ref_w * ref_h > (size_t)p.width * p.height) {
        if (!prepare_vae(ctx, ref_w, ref_h)) return NULL;
    } else if (!prepare_vae(ctx, p.width, p.height)) return NULL;

    /* Resize input if needed */
    iris_image *resized = NULL;
    const iris_image *img_to_use = input;
    if (input->width != ref_w || input->height != ref_h) {
        resized = iris_image_resize(input, ref_w, ref_h);
        if (!resized) {
            set_error("Failed to resize input image");
            return NULL;
        }
        img_to_use = resized;
    }

    /* Resolve steps and guidance */
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;
    float guidance = (p.guidance > 0) ? p.guidance : ctx->default_guidance;

    /* Encode text */
    int text_seq;
    float *text_emb = iris_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        if (resized) iris_image_free(resized);
        set_error("Failed to encode prompt");
        return NULL;
    }

    float *text_emb_uncond = NULL;
    int text_seq_uncond = 0;
    if (!ctx->is_distilled) {
        text_emb_uncond = iris_encode_text(ctx, "", &text_seq_uncond);
        if (!text_emb_uncond) {
            free(text_emb);
            if (resized) iris_image_free(resized);
            set_error("Failed to encode empty prompt for CFG");
            return NULL;
        }
    }

    /* Release text encoder to free ~8GB before loading transformer */
    iris_release_text_encoder(ctx);

    /* Load transformer now (after text encoder is freed to reduce peak memory) */
    if (!iris_load_transformer_if_needed(ctx)) {
        free(text_emb);
        free(text_emb_uncond);
        if (resized) iris_image_free(resized);
        return NULL;
    }

    /* Encode image to latent */
    if (iris_phase_callback) iris_phase_callback("encoding reference image", 0);
    float *img_tensor = iris_image_to_tensor(img_to_use);
    if (resized) iris_image_free(resized);

    int latent_h, latent_w;
    float *img_latent = NULL;

    if (ctx->vae) {
        img_latent = iris_vae_encode(ctx->vae, img_tensor, 1,
                                     ref_h, ref_w, &latent_h, &latent_w);
    } else {
        /* Placeholder if no VAE */
        latent_h = ref_h / 16;
        latent_w = ref_w / 16;
        img_latent = (float *)calloc(IRIS_LATENT_CHANNELS * latent_h * latent_w, sizeof(float));
    }

    free(img_tensor);
    if (iris_phase_callback) iris_phase_callback("encoding reference image", 1);

    if (!img_latent) {
        free(text_emb);
        free(text_emb_uncond);
        set_error("Failed to encode image");
        return NULL;
    }

    /*
     * FLUX.2 img2img uses in-context conditioning:
     * - Reference image is encoded to latent with T offset in RoPE (T=10)
     * - Target image starts from pure noise (T=0)
     * - Both are concatenated as tokens in the transformer
     * - Model attends to reference via joint attention
     * - Only target tokens are output
     *
     * This is fundamentally different from traditional img2img that adds
     * noise directly to the encoded image.
     */
    int num_steps = p.num_steps;
    int out_lat_h = p.height / 16;
    int out_lat_w = p.width / 16;
    int image_seq_len = out_lat_h * out_lat_w;  /* For schedule calculation */

    /* Get schedule */
    float *schedule = iris_selected_schedule(&p, image_seq_len);

    /* Initialize target latent with pure noise */
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = iris_init_noise(1, IRIS_LATENT_CHANNELS, out_lat_h, out_lat_w, seed);

    /* Reference image latent is img_latent, with T offset = 10 */
    int t_offset = 10;

    /* Sample using in-context conditioning */
    float *latent;
    if (ctx->is_distilled) {
        latent = iris_sample_euler_refs_flux(
            ctx->transformer, ctx->qwen3_encoder,
            z, 1, IRIS_LATENT_CHANNELS, out_lat_h, out_lat_w,
            img_latent, latent_h, latent_w,
            t_offset,
            text_emb, text_seq,
            schedule, num_steps,
            NULL
        );
    } else {
        latent = iris_sample_euler_cfg_refs_flux(
            ctx->transformer, ctx->qwen3_encoder,
            z, 1, IRIS_LATENT_CHANNELS, out_lat_h, out_lat_w,
            img_latent, latent_h, latent_w,
            t_offset,
            text_emb, text_seq,
            text_emb_uncond, text_seq_uncond,
            guidance,
            schedule, num_steps,
            NULL
        );
    }

    free(z);
    free(img_latent);
    free(schedule);
    free(text_emb);
    free(text_emb_uncond);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode */
    iris_image *result = NULL;
    if (ctx->vae) {
        if (iris_phase_callback) iris_phase_callback("decoding image", 0);
        result = iris_vae_decode(ctx->vae, latent, 1, out_lat_h, out_lat_w);
        if (iris_phase_callback) iris_phase_callback("decoding image", 1);
    }

    free(latent);
    return result;
}

/* ========================================================================
 * Multi-Reference Generation
 * ======================================================================== */

/* Multi-reference image generation dispatcher. Zero refs routes to txt2img,
 * one ref to the optimized single-reference img2img path. For multiple refs,
 * each reference is VAE-encoded with a distinct RoPE T offset (10, 20, 30...)
 * so the transformer can distinguish them spatially. All reference latents
 * participate in joint attention alongside the noised target tokens. */
iris_image *iris_multiref(iris_ctx *ctx, const char *prompt,
                          const iris_image **refs, int num_refs,
                          const iris_params *params) {
    if (!ctx || !prompt) {
        set_error("Invalid parameters");
        return NULL;
    }
    if (ctx->is_zimage) {
        set_error("multi-reference img2img is not supported for Z-Image");
        return NULL;
    }
    if (ctx->is_flux3 && refs && num_refs > 0)
        return iris_generate_flux3_refs(ctx, prompt, refs, num_refs, params);

    /* No references - text-to-image */
    if (!refs || num_refs == 0) {
        return iris_generate(ctx, prompt, params);
    }

    /* Single reference - use optimized path */
    if (num_refs == 1) {
        return iris_img2img(ctx, prompt, refs[0], params);
    }

    iris_params p;
    if (params) {
        p = *params;
    } else {
        p = (iris_params)IRIS_PARAMS_DEFAULT;
    }

    /* Use first reference dimensions if not specified */
    if (p.width <= 0) p.width = refs[0]->width;
    if (p.height <= 0) p.height = refs[0]->height;

    /* Clamp to VAE max dimensions */
    if (p.width > ctx->max_width || p.height > ctx->max_height) {
        float scale = (float)ctx->max_width /
                      (p.width > p.height ? p.width : p.height);
        p.width = (int)(p.width * scale);
        p.height = (int)(p.height * scale);
    }

    p.width = (p.width / 16) * 16;
    p.height = (p.height / 16) * 16;

    /* Resolve steps and guidance */
    if (p.num_steps <= 0) p.num_steps = ctx->default_steps;
    float guidance = (p.guidance > 0) ? p.guidance : ctx->default_guidance;

    /* Build reference pixel dimensions, clamped and rounded to 16. */
    int *ref_pixel_dims = (int *)malloc(num_refs * 2 * sizeof(int));
    for (int i = 0; i < num_refs; i++) {
        int rh = (refs[i]->height / 16) * 16;
        int rw = (refs[i]->width / 16) * 16;
        if (rh > ctx->max_height) rh = ctx->max_height;
        if (rw > ctx->max_width) rw = ctx->max_width;
        if (rh < 16) rh = 16;
        if (rw < 16) rw = 16;
        ref_pixel_dims[i*2]   = rh;
        ref_pixel_dims[i*2+1] = rw;
    }

    /* Reject impossible manual budgets before encoding or loading weights. */
    int fit = fit_refs_for_attention(ctx->num_heads, p.height, p.width,
                                     ref_pixel_dims, num_refs, IRIS_MAX_SEQ_LEN);
    if (fit < 0) {
        free(ref_pixel_dims);
        return NULL;
    }
    if (fit > 0) {
        fprintf(stderr,
                "Note: reference images resized to fit GPU attention "
                "memory limit\n");
    }

    int max_w = p.width, max_h = p.height;
    for (int i = 0; i < num_refs; i++) {
        int rh = ref_pixel_dims[i*2], rw = ref_pixel_dims[i*2+1];
        if ((size_t)rh * rw > (size_t)max_h * max_w) { max_h = rh; max_w = rw; }
    }
    if (!prepare_vae(ctx, max_w, max_h)) {
        free(ref_pixel_dims);
        return NULL;
    }

    /* Encode text */
    int text_seq;
    float *text_emb = iris_encode_text(ctx, prompt, &text_seq);
    if (!text_emb) {
        free(ref_pixel_dims);
        set_error("Failed to encode prompt");
        return NULL;
    }

    float *text_emb_uncond = NULL;
    int text_seq_uncond = 0;
    if (!ctx->is_distilled) {
        text_emb_uncond = iris_encode_text(ctx, "", &text_seq_uncond);
        if (!text_emb_uncond) {
            free(text_emb);
            free(ref_pixel_dims);
            set_error("Failed to encode empty prompt for CFG");
            return NULL;
        }
    }

    iris_release_text_encoder(ctx);

    if (!iris_load_transformer_if_needed(ctx)) {
        free(text_emb);
        free(text_emb_uncond);
        free(ref_pixel_dims);
        return NULL;
    }

    /* Encode all reference images */
    iris_ref_t *ref_latents = (iris_ref_t *)malloc(num_refs * sizeof(iris_ref_t));
    float **ref_data = (float **)malloc(num_refs * sizeof(float *));
    iris_image **resized_imgs = (iris_image **)calloc(num_refs, sizeof(iris_image *));

    for (int i = 0; i < num_refs; i++) {
        const iris_image *ref = refs[i];
        const iris_image *img_to_use = ref;

        int ref_h = ref_pixel_dims[i*2];
        int ref_w = ref_pixel_dims[i*2+1];

        /* Resize only if dimensions differ from original */
        if (ref->width != ref_w || ref->height != ref_h) {
            resized_imgs[i] = iris_image_resize(ref, ref_w, ref_h);
            if (!resized_imgs[i]) {
                for (int j = 0; j < i; j++) {
                    free(ref_data[j]);
                    if (resized_imgs[j]) iris_image_free(resized_imgs[j]);
                }
                free(ref_latents);
                free(ref_data);
                free(resized_imgs);
                free(ref_pixel_dims);
                free(text_emb);
                free(text_emb_uncond);
                set_error("Failed to resize reference image");
                return NULL;
            }
            img_to_use = resized_imgs[i];
        }

        /* Encode to latent at reference's own size */
        float *tensor = iris_image_to_tensor(img_to_use);
        int lat_h, lat_w;
        ref_data[i] = iris_vae_encode(ctx->vae, tensor, 1,
                                       img_to_use->height, img_to_use->width,
                                       &lat_h, &lat_w);
        free(tensor);

        if (!ref_data[i]) {
            for (int j = 0; j < i; j++) {
                free(ref_data[j]);
                if (resized_imgs[j]) iris_image_free(resized_imgs[j]);
            }
            if (resized_imgs[i]) iris_image_free(resized_imgs[i]);
            free(ref_latents);
            free(ref_data);
            free(resized_imgs);
            free(ref_pixel_dims);
            free(text_emb);
            free(text_emb_uncond);
            set_error("Failed to encode reference image");
            return NULL;
        }

        ref_latents[i].latent = ref_data[i];
        ref_latents[i].h = lat_h;
        ref_latents[i].w = lat_w;
        ref_latents[i].t_offset = 10 * (i + 1);  /* 10, 20, 30, ... */
    }

    /* Free resized images (latents are now encoded) */
    for (int i = 0; i < num_refs; i++) {
        if (resized_imgs[i]) iris_image_free(resized_imgs[i]);
    }
    free(resized_imgs);
    free(ref_pixel_dims);

    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;

    float *schedule = iris_selected_schedule(&p, image_seq_len);
    int64_t seed = (p.seed < 0) ? (int64_t)time(NULL) : p.seed;
    float *z = iris_init_noise(1, IRIS_LATENT_CHANNELS, latent_h, latent_w, seed);

    /* Sample with multi-reference conditioning */
    float *latent;
    if (ctx->is_distilled) {
        latent = iris_sample_euler_multirefs_flux(
            ctx->transformer, ctx->qwen3_encoder,
            z, 1, IRIS_LATENT_CHANNELS, latent_h, latent_w,
            ref_latents, num_refs,
            text_emb, text_seq,
            schedule, p.num_steps,
            NULL
        );
    } else {
        latent = iris_sample_euler_cfg_multirefs_flux(
            ctx->transformer, ctx->qwen3_encoder,
            z, 1, IRIS_LATENT_CHANNELS, latent_h, latent_w,
            ref_latents, num_refs,
            text_emb, text_seq,
            text_emb_uncond, text_seq_uncond,
            guidance,
            schedule, p.num_steps,
            NULL
        );
    }

    /* Cleanup */
    free(z);
    for (int i = 0; i < num_refs; i++) {
        free(ref_data[i]);
    }
    free(ref_data);
    free(ref_latents);
    free(schedule);
    free(text_emb);
    free(text_emb_uncond);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode */
    iris_image *result = NULL;
    if (ctx->vae) {
        if (iris_phase_callback) iris_phase_callback("decoding image", 0);
        result = iris_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (iris_phase_callback) iris_phase_callback("decoding image", 1);
    }

    free(latent);
    return result;
}

/* ========================================================================
 * Utility Functions
 * ======================================================================== */

void iris_set_seed(int64_t seed) {
    iris_rng_seed((uint64_t)seed);
}

const char *iris_model_info(iris_ctx *ctx) {
    static char info[256];
    const char *type;
    if (!ctx) {
        return "No model loaded";
    }
    if (ctx->is_zimage) type = "zimage";
    else type = ctx->is_distilled ? "distilled" : "base";
    snprintf(info, sizeof(info), "%s v%s (%s, %d steps, guidance %.1f)",
             ctx->model_name, ctx->model_version,
             type,
             ctx->default_steps, ctx->default_guidance);
    return info;
}

int iris_text_dim(iris_ctx *ctx) {
    return ctx ? ctx->text_dim : 7680;
}

const char *iris_license_notice(iris_ctx *ctx) {
    if (!ctx || !ctx->is_non_commercial) return NULL;
    if (ctx->is_flux3) {
        return "This model is released under the FLUX Kommunity License:\n"
               "non-commercial use, plus commercial use of the outputs for\n"
               "users with less than US$5M gross annualized revenue.\n"
               "https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/main/LICENSE.md";
    }
    return "This model is released under a NON COMMERCIAL LICENSE.\n"
           "The output can only be used under the terms of the\n"
           "FLUX non-commercial license:\n"
           "https://huggingface.co/black-forest-labs/FLUX.2-klein-9B/blob/main/LICENSE.md";
}

int iris_is_non_commercial(iris_ctx *ctx) {
    return ctx ? ctx->is_non_commercial : 0;
}

/* ========================================================================
 * Low-level API
 * ======================================================================== */

/* Public API: VAE-encode an RGB image to latent space. Converts the image
 * to a float tensor, runs the VAE encoder, and returns the latent buffer
 * with dimensions in out_h/out_w (each 1/16 of the pixel dimensions). */
float *iris_encode_image(iris_ctx *ctx, const iris_image *img,
                         int *out_h, int *out_w) {
    if (!ctx || !img || !ctx->vae) {
        *out_h = *out_w = 0;
        return NULL;
    }

    float *tensor = iris_image_to_tensor(img);
    if (!tensor) return NULL;

    float *latent = iris_vae_encode(ctx->vae, tensor, 1,
                                    img->height, img->width, out_h, out_w);
    free(tensor);
    return latent;
}

/* Public API: VAE-decode a latent tensor back to an RGB image.
 * Latent dimensions are 1/16 of the output pixel dimensions. */
iris_image *iris_decode_latent(iris_ctx *ctx, const float *latent,
                               int latent_h, int latent_w) {
    if (!ctx || !latent || !ctx->vae) return NULL;
    if (iris_phase_callback) iris_phase_callback("decoding image", 0);
    iris_image *img = iris_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
    if (iris_phase_callback) iris_phase_callback("decoding image", 1);
    return img;
}

float *iris_denoise_step(iris_ctx *ctx, const float *z, float t,
                         const float *text_emb, int text_len,
                         int latent_h, int latent_w) {
    if (!ctx || !z || !text_emb) return NULL;

    /* Load transformer if not already loaded */
    if (!iris_load_transformer_if_needed(ctx)) {
        return NULL;
    }

    return iris_transformer_forward_flux(ctx->transformer,
                                    z, latent_h, latent_w,
                                    text_emb, text_len, t);
}

/* Debug function: img2img with external inputs from Python */
iris_image *iris_img2img_debug_py(iris_ctx *ctx, const iris_params *params) {
    if (!ctx) {
        set_error("Invalid context");
        return NULL;
    }

    iris_params p;
    if (params) {
        p = *params;
    } else {
        p = (iris_params)IRIS_PARAMS_DEFAULT;
    }

    /* Load Python's noise */
    FILE *f_noise = fopen("/tmp/py_noise.bin", "rb");
    if (!f_noise) {
        set_error("Cannot open /tmp/py_noise.bin");
        return NULL;
    }
    fseek(f_noise, 0, SEEK_END);
    int noise_size = ftell(f_noise) / sizeof(float);
    fseek(f_noise, 0, SEEK_SET);
    float *noise = (float *)malloc(noise_size * sizeof(float));
    fread(noise, sizeof(float), noise_size, f_noise);
    fclose(f_noise);
    fprintf(stderr, "[DEBUG] Loaded noise: %d floats\n", noise_size);

    /* Load Python's ref_latent */
    FILE *f_ref = fopen("/tmp/py_ref_latent.bin", "rb");
    if (!f_ref) {
        free(noise);
        set_error("Cannot open /tmp/py_ref_latent.bin");
        return NULL;
    }
    fseek(f_ref, 0, SEEK_END);
    int ref_size = ftell(f_ref) / sizeof(float);
    fseek(f_ref, 0, SEEK_SET);
    float *ref_latent = (float *)malloc(ref_size * sizeof(float));
    fread(ref_latent, sizeof(float), ref_size, f_ref);
    fclose(f_ref);
    fprintf(stderr, "[DEBUG] Loaded ref_latent: %d floats\n", ref_size);

    /* Load Python's text_emb */
    FILE *f_txt = fopen("/tmp/py_text_emb.bin", "rb");
    if (!f_txt) {
        free(noise);
        free(ref_latent);
        set_error("Cannot open /tmp/py_text_emb.bin");
        return NULL;
    }
    fseek(f_txt, 0, SEEK_END);
    int txt_size = ftell(f_txt) / sizeof(float);
    fseek(f_txt, 0, SEEK_SET);
    float *text_emb = (float *)malloc(txt_size * sizeof(float));
    fread(text_emb, sizeof(float), txt_size, f_txt);
    fclose(f_txt);
    int text_seq = 512;
    fprintf(stderr, "[DEBUG] Loaded text_emb: %d floats (%d x %d)\n",
            txt_size, text_seq, txt_size / text_seq);

    /* Load transformer */
    if (!iris_load_transformer_if_needed(ctx)) {
        free(noise);
        free(ref_latent);
        free(text_emb);
        return NULL;
    }

    /* Dimensions */
    int latent_h = p.height / 16;
    int latent_w = p.width / 16;
    int image_seq_len = latent_h * latent_w;

    /* Get schedule */
    float *schedule = iris_selected_schedule(&p, image_seq_len);

    /* Sample with refs */
    float *latent = iris_sample_euler_refs_flux(
        ctx->transformer, NULL,
        noise, 1, IRIS_LATENT_CHANNELS, latent_h, latent_w,
        ref_latent, latent_h, latent_w,
        10,  /* t_offset */
        text_emb, text_seq,
        schedule, p.num_steps,
        NULL  /* progress_callback */
    );

    free(noise);
    free(ref_latent);
    free(schedule);
    free(text_emb);

    if (!latent) {
        set_error("Sampling failed");
        return NULL;
    }

    /* Decode */
    iris_image *result = NULL;
    if (ctx->vae) {
        if (iris_phase_callback) iris_phase_callback("decoding image", 0);
        result = iris_vae_decode(ctx->vae, latent, 1, latent_h, latent_w);
        if (iris_phase_callback) iris_phase_callback("decoding image", 1);
    }

    free(latent);
    return result;
}
