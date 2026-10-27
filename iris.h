/*
 * Iris - C Image Generation Engine
 *
 * A dependency-free C inference engine for image synthesis models.
 * Supports FLUX.2 Klein and Z-Image-Turbo model families.
 *
 * Usage:
 *   iris_ctx *ctx = iris_load_dir("path/to/model");
 *   if (!ctx) { handle error }
 *
 *   iris_params params = IRIS_PARAMS_DEFAULT;
 *   iris_image *img = iris_generate(ctx, "a cat sitting on a rainbow", &params);
 *   iris_image_save(img, "output.png");
 *   iris_image_free(img);
 *   iris_free(ctx);
 */

#ifndef IRIS_H
#define IRIS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Configuration Constants
 * ======================================================================== */

/* Model architecture constants (same across model sizes) */
#define IRIS_LATENT_CHANNELS    128  /* Flux: 32*2*2, Z-Image: 16*2*2=64 */

/* VAE architecture */
#define IRIS_VAE_Z_CHANNELS     32   /* Flux default; Z-Image uses 16 */
#define IRIS_VAE_BASE_CH        128
#define IRIS_VAE_CH_MULT_0      1
#define IRIS_VAE_CH_MULT_1      2
#define IRIS_VAE_CH_MULT_2      4
#define IRIS_VAE_CH_MULT_3      4
#define IRIS_VAE_NUM_RES        2
#define IRIS_VAE_GROUPS         32
#define IRIS_VAE_MAX_DIM        1792  /* Max image dimension for VAE */

/* Tokenizer */
#define IRIS_MAX_SEQ_LEN        512
#define IRIS_VOCAB_HASH_SIZE    150001

/* Sampling */
#define IRIS_MAX_STEPS          256

/* ========================================================================
 * Opaque Types
 * ======================================================================== */

typedef struct iris_ctx iris_ctx;
typedef struct iris_image iris_image;
typedef struct iris_tokenizer iris_tokenizer;

/* ========================================================================
 * Image Structure
 * ======================================================================== */

struct iris_image {
    int width;
    int height;
    int channels;       /* 3 for RGB, 4 for RGBA */
    uint8_t *data;      /* Row-major, channel-interleaved */
};

/* ========================================================================
 * Generation Parameters
 * ======================================================================== */

/* Schedule type: 0 = model default (sigmoid for Flux, flowmatch for Z-Image) */
enum {
    IRIS_SCHEDULE_DEFAULT   = 0,
    IRIS_SCHEDULE_LINEAR    = 1,
    IRIS_SCHEDULE_POWER     = 2,
    IRIS_SCHEDULE_SIGMOID   = 3,  /* Flux shifted sigmoid */
    IRIS_SCHEDULE_FLOWMATCH = 4,  /* Z-Image FlowMatch Euler */
};

typedef struct {
    int width;              /* Output width (default: 256) */
    int height;             /* Output height (default: 256) */
    int num_steps;          /* Inference steps (default: 4 distilled, 50 base) */
    int64_t seed;           /* Random seed (-1 for random) */
    float guidance;         /* CFG guidance scale (0 = auto from model type) */
    int schedule;           /* Schedule type (IRIS_SCHEDULE_*) */
    float power_alpha;      /* Exponent for power schedule (default: 2.0) */
} iris_params;

/* Default parameters */
#define IRIS_DEFAULT_WIDTH  256
#define IRIS_DEFAULT_HEIGHT 256
#define IRIS_PARAMS_DEFAULT { IRIS_DEFAULT_WIDTH, IRIS_DEFAULT_HEIGHT, 0, -1, 0.0f, IRIS_SCHEDULE_DEFAULT, 2.0f }

/* ========================================================================
 * Core API
 * ======================================================================== */

/*
 * Load model from HuggingFace-style directory containing safetensors files.
 * Directory should contain: vae/, transformer/, tokenizer/ subdirectories.
 * Returns NULL on error.
 */
iris_ctx *iris_load_dir(const char *model_dir);

/* Opt in to images beyond 1792 pixels per side (up to 4096 per side,
 * subject to tensor size limits). VAE buffers are allocated on demand.
 * Passing zero preserves iris_load_dir() behavior. */
iris_ctx *iris_load_dir_ex(const char *model_dir, int allow_large_images);

/* Current per-side image limit, also used by the interactive CLI. */
int iris_max_image_dimension(const iris_ctx *ctx);

/*
 * Free model and all associated resources.
 */
void iris_free(iris_ctx *ctx);

/*
 * Release the text encoder to free ~8GB of memory.
 * Call this after encoding if you don't need to encode more prompts.
 * The encoder will be reloaded automatically if needed for a new prompt.
 */
void iris_release_text_encoder(iris_ctx *ctx);

/*
 * Enable mmap mode for text encoder (--mmap).
 * Uses memory-mapped bf16 weights directly instead of converting to f32.
 * Reduces memory usage from ~16GB to ~8GB but is slower due to on-the-fly conversion.
 * Call this after iris_load_dir() and before first generation.
 */
void iris_set_mmap(iris_ctx *ctx, int enable);

/*
 * Check if model is distilled (4-step) or base (50-step with CFG).
 * Returns 1 for distilled, 0 for base.
 */
int iris_is_distilled(iris_ctx *ctx);

/*
 * Check if model is Z-Image (S3-DiT architecture).
 * Returns 1 for Z-Image, 0 for Flux.
 */
int iris_is_zimage(iris_ctx *ctx);

/*
 * Check if model is FLUX.3 (Image Klein).
 */
int iris_is_flux3(iris_ctx *ctx);

/*
 * Default number of steps and guidance scale for the loaded model.
 */
int iris_default_steps(iris_ctx *ctx);
float iris_default_guidance(iris_ctx *ctx);

/*
 * Force base model mode (overrides autodetection).
 * Call after iris_load_dir() if model_index.json is missing.
 */
void iris_set_base_mode(iris_ctx *ctx);

/*
 * Text-to-image generation.
 * Returns newly allocated image, caller must free with iris_image_free().
 * Returns NULL on error.
 */
iris_image *iris_generate(iris_ctx *ctx, const char *prompt,
                          const iris_params *params);

/*
 * Image-to-image generation.
 * Takes an input image and modifies it according to the prompt.
 * Uses in-context conditioning: the reference image is passed as additional
 * tokens that the model attends to during generation.
 */
iris_image *iris_img2img(iris_ctx *ctx, const char *prompt,
                         const iris_image *input, const iris_params *params);

/*
 * Multi-reference generation (up to 4 reference images for klein).
 */
iris_image *iris_multiref(iris_ctx *ctx, const char *prompt,
                          const iris_image **refs, int num_refs,
                          const iris_params *params);

/*
 * Debug: img2img using Python's exact inputs from /tmp/py_*.bin files.
 * Used for comparing C and Python implementations.
 */
iris_image *iris_img2img_debug_py(iris_ctx *ctx, const iris_params *params);

/*
 * Text-to-image generation with pre-computed embeddings.
 * text_emb: float array of shape [text_seq, text_dim]
 * text_seq: number of text tokens (typically 512)
 */
iris_image *iris_generate_with_embeddings(iris_ctx *ctx,
                                           const float *text_emb, int text_seq,
                                           const iris_params *params);

/*
 * Generate image with external embeddings and external noise.
 * For testing and debugging to match Python exactly.
 * noise: [latent_channels, height/16, width/16] in NCHW format
 * noise_size: total number of floats in noise array
 */
iris_image *iris_generate_with_embeddings_and_noise(iris_ctx *ctx,
                                                     const float *text_emb, int text_seq,
                                                     const float *noise, int noise_size,
                                                     const iris_params *params);

/* ========================================================================
 * Image I/O
 * ======================================================================== */

/*
 * Load image from file (PNG or PPM).
 * Returns NULL on error.
 */
iris_image *iris_image_load(const char *path);

/*
 * Save image to file (format determined by extension).
 * Supports: .png, .ppm
 * Returns 0 on success, -1 on error.
 */
int iris_image_save(const iris_image *img, const char *path);

/*
 * Save image to PNG with seed embedded as metadata.
 * The seed is stored in a tEXt chunk with keyword "iris:seed".
 * Returns 0 on success, -1 on error.
 */
int iris_image_save_with_seed(const iris_image *img, const char *path, int64_t seed);

/*
 * Create a new image with given dimensions.
 */
iris_image *iris_image_create(int width, int height, int channels);

/*
 * Free image memory.
 */
void iris_image_free(iris_image *img);

/*
 * Resize image (Lanczos-3, antialiased when downscaling).
 */
iris_image *iris_image_resize(const iris_image *img, int new_width, int new_height);

/* ========================================================================
 * Utility Functions
 * ======================================================================== */

/*
 * Set random seed for reproducible generation.
 */
void iris_set_seed(int64_t seed);

/*
 * Get model info string.
 */
const char *iris_model_info(iris_ctx *ctx);

/*
 * Get text embedding dimension (7680 for 4B, varies by model).
 */
int iris_text_dim(iris_ctx *ctx);

/*
 * Check if model has non-commercial license (e.g., 9B model).
 */
int iris_is_non_commercial(iris_ctx *ctx);

/*
 * License notice to show the user for models with usage restrictions,
 * or NULL for permissively licensed models.
 */
const char *iris_license_notice(iris_ctx *ctx);

/*
 * Get last error message.
 */
const char *iris_get_error(void);

/*
 * Set step image callback to receive decoded images after each denoising step.
 * Useful for visualizing the generation process.
 * Pass NULL to disable. The callback receives images that must NOT be freed.
 */
typedef void (*iris_step_image_cb_t)(int step, int total, const iris_image *img);
void iris_set_step_image_callback(iris_ctx *ctx, iris_step_image_cb_t callback);

/* ========================================================================
 * Advanced / Low-level API
 * ======================================================================== */

/*
 * Encode image to latent space using VAE encoder.
 * Returns latent tensor [1, 128, H/16, W/16].
 * Caller must free() the returned pointer.
 */
float *iris_encode_image(iris_ctx *ctx, const iris_image *img,
                         int *out_h, int *out_w);

/*
 * Decode latent to image using VAE decoder.
 */
iris_image *iris_decode_latent(iris_ctx *ctx, const float *latent,
                               int latent_h, int latent_w);

/*
 * Encode text prompt to embeddings.
 * Returns embedding tensor [1, seq_len, 7680].
 * Caller must free() the returned pointer.
 */
float *iris_encode_text(iris_ctx *ctx, const char *prompt, int *out_seq_len);

/*
 * Run single denoising step.
 * z: current latent [1, 128, H, W]
 * t: timestep (0.0 to 1.0)
 * text_emb: text embeddings
 * Returns velocity prediction.
 */
float *iris_denoise_step(iris_ctx *ctx, const float *z, float t,
                         const float *text_emb, int text_len,
                         int latent_h, int latent_w);

#ifdef __cplusplus
}
#endif

#endif /* IRIS_H */
