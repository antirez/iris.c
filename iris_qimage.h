/*
 * Qwen-Image 2.1 (base and Turbo): internal API shared by iris.c and the
 * Qwen-Image components (scheduler, transformer, VAE).
 */

#ifndef IRIS_QIMAGE_H
#define IRIS_QIMAGE_H

#include "iris.h"

/* Upper bound on Turbo's sample_sigmas in model_index.json. */
#define QIMAGE_MAX_SIGMAS 64

/* Latent channels (VAE z_dim, transformer in/out channels). */
#define IRIS_QIMAGE_LATENT_CHANNELS 64

/* Image size limits. The model recommends up to 2752x1536 and 1536x2752;
 * the VAE decoder needs about 22 GB of GPU activations at 4 MP. */
#define IRIS_QIMAGE_MAX_DIM    2816
#define IRIS_QIMAGE_MAX_PIXELS (2752 * 1536)

/* Reference images are prepared at this area (1024 x 1024). */
#define IRIS_QIMAGE_REF_AREA (1024 * 1024)

/* ========================================================================
 * Reference preparation (iris_image.c)
 * ======================================================================== */

/* Size of a width x height reference at about area pixels, on a 32-pixel grid. */
void iris_ref_size_qimage(int width, int height, int area, int *ref_w, int *ref_h);

/* RGBA copy of img at its reference size. */
iris_image *iris_ref_prepare_qimage(const iris_image *img, int area);

/* Vision encoder RGB copy of a prepared reference, over white. */
iris_image *iris_ref_vision_qimage(const iris_image *rgba);

/* VAE input of a prepared reference: [4, h, w] in [-1, 1] (caller frees). */
float *iris_ref_pixels_qimage(const iris_image *rgba);

/* ========================================================================
 * Scheduler (iris_sample.c)
 * ======================================================================== */

/* FlowMatchEulerDiscreteScheduler settings from scheduler_config.json,
 * with the diffusers defaults for absent keys. */
typedef struct {
    int use_dynamic_shifting;
    float base_seq_len, max_seq_len;    /* base/max_image_seq_len */
    float base_shift, max_shift;
    float shift;                        /* static shift (no dynamic shifting) */
    float shift_terminal;               /* 0 = no stretch */
    int linear_time_shift;              /* time_shift_type "linear" */
} iris_qimage_sched_t;

void iris_qimage_sched_load(iris_qimage_sched_t *s, const char *model_dir);

/* Schedule of num_steps + 1 sigmas (terminal 0), the way
 * FlowMatchEulerDiscreteScheduler.set_timesteps(sigmas=...) processes
 * them. sigmas is Turbo's sample_sigmas grid (num_steps values), or NULL
 * for linspace(1, 1/num_steps, num_steps). image_seq_len is the target
 * token count (H/16 * W/16). */
float *iris_schedule_qimage(const iris_qimage_sched_t *s, const float *sigmas,
                            int num_steps, int image_seq_len);

/* ========================================================================
 * Transformer (iris_transformer_qimage.c)
 * ======================================================================== */

typedef struct iris_transformer_qimage iris_transformer_qimage_t;

iris_transformer_qimage_t *iris_transformer_load_qimage(const char *model_dir);
void iris_transformer_free_qimage(iris_transformer_qimage_t *tf);

/* Estimated peak memory of a prefill of rows prefix tokens, text_rows of them
 * text, from the config alone. On a GPU it is device memory, the weights
 * included. */
size_t iris_transformer_memory_qimage(const char *model_dir, int rows, int text_rows);

/* Free the prefix and the activations. The next step needs a new prefill. */
void iris_transformer_release_prefix_qimage(iris_transformer_qimage_t *tf);

/* Latent channels the transformer reads and predicts (64). */
int iris_transformer_channels_qimage(const iris_transformer_qimage_t *tf);

/* A reference image in the prefix: its latent tokens sit after the first
 * pos text rows. latent: normalized [channels, h * w]. */
typedef struct {
    int pos;
    int h, w;
    const float *latent;
} iris_qimage_ref_t;

/* Run the prefix through every block once and keep its keys and values:
 * the text rows txt [txt_seq, context_in_dim] (without image slots) with
 * the n reference images interleaved. The prefix never attends to the
 * target and is modulated from t = 0, so it serves every step (and every
 * generation with the same inputs). Returns 1 on success. */
int iris_transformer_prefill_qimage(iris_transformer_qimage_t *tf,
                                    const float *txt, int txt_seq,
                                    const iris_qimage_ref_t *refs, int n);

/* Velocity [channels, h * w] of the target latent x [channels, h * w] at
 * sigma, attending to the prefix from the last prefill (caller frees). */
float *iris_transformer_forward_qimage(iris_transformer_qimage_t *tf,
                                       const float *x, int h, int w, float sigma);

/* ========================================================================
 * VAE (iris_vae_qimage.c)
 * ======================================================================== */

typedef struct iris_vae_qimage iris_vae_qimage_t;

iris_vae_qimage_t *iris_vae_load_qimage(const char *model_dir);
void iris_vae_free_qimage(iris_vae_qimage_t *vae);

/* Allocate what a GPU decode of an h x w latent needs. Returns 0 if it does
 * not fit, 1 if it does or there is no GPU. */
int iris_vae_reserve_qimage(iris_vae_qimage_t *vae, int h, int w);

/* GPU activations of such a decode, 0 without a GPU. */
size_t iris_vae_memory_qimage(const iris_vae_qimage_t *vae, int h, int w);

/* Free the GPU activations and the encoder, leaving the decoder's weights
 * loaded. A later encode loads the encoder again. */
void iris_vae_release_qimage(iris_vae_qimage_t *vae);

/* Encode an image [in_channels, h, w] in [-1, 1] (h, w multiples of 16) into
 * a normalized latent [z_dim, h / 16, w / 16] (caller frees). */
float *iris_vae_encode_qimage(iris_vae_qimage_t *vae, const float *rgba, int h, int w);

/* Decode a normalized latent [z_dim, h, w] (as the transformer produces
 * it) into an RGBA image of 16h x 16w pixels. */
iris_image *iris_vae_decode_qimage(iris_vae_qimage_t *vae, const float *latent,
                                   int h, int w);

/* ========================================================================
 * Generation (iris.c)
 * ======================================================================== */

/* Generate with n reference images prepared at ref_area. noise (noise_size
 * floats), if not NULL, replaces the seed's; out_latent, if not NULL, receives
 * the final latent (caller frees). Parity runs use these three arguments;
 * iris_img2img() and iris_multiref() pass IRIS_QIMAGE_REF_AREA and nothing. */
iris_image *iris_generate_refs_qimage(iris_ctx *ctx, const char *prompt,
                                      const iris_image **refs, int n, int ref_area,
                                      const float *noise, int noise_size,
                                      const iris_params *params, float **out_latent);

/* ========================================================================
 * Sampler (iris_sample.c)
 * ======================================================================== */

/* Euler sampler: z += (sigma_next - sigma) * v. The text prefix must have
 * been prefilled. preview_vae, if not NULL, decodes step previews for the
 * step image callback. Returns the final latent (caller frees). */
float *iris_sample_euler_qimage(iris_transformer_qimage_t *tf,
                                const float *z, int channels, int h, int w,
                                const float *schedule, int num_steps,
                                iris_vae_qimage_t *preview_vae);

#endif /* IRIS_QIMAGE_H */
