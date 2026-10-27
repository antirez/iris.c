/*
 * Iris Sampling Implementation
 *
 * Rectified Flow sampling for image generation.
 * Uses Euler method for ODE integration.
 */

#include "iris.h"
#include "iris_kernels.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#ifdef USE_METAL
#include "iris_metal.h"
#endif

/* Timing utilities for performance analysis - use wall-clock time */
static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* Cumulative timing for denoising breakdown */
double iris_timing_transformer_total = 0.0;
double iris_timing_transformer_double = 0.0;
double iris_timing_transformer_single = 0.0;
double iris_timing_transformer_final = 0.0;
double iris_timing_zi_total = 0.0;
double iris_timing_zi_embeddings = 0.0;
double iris_timing_zi_noise_refiner = 0.0;
double iris_timing_zi_context_refiner = 0.0;
double iris_timing_zi_main_blocks = 0.0;
double iris_timing_zi_final = 0.0;

void iris_reset_timing(void) {
    iris_timing_transformer_total = 0.0;
    iris_timing_transformer_double = 0.0;
    iris_timing_transformer_single = 0.0;
    iris_timing_transformer_final = 0.0;
    iris_timing_zi_total = 0.0;
    iris_timing_zi_embeddings = 0.0;
    iris_timing_zi_noise_refiner = 0.0;
    iris_timing_zi_context_refiner = 0.0;
    iris_timing_zi_main_blocks = 0.0;
    iris_timing_zi_final = 0.0;
}

/* ========================================================================
 * Timestep Schedules
 * ======================================================================== */

/*
 * Linear timestep schedule from 1.0 to 0.0
 * Returns array of num_steps+1 values: [1.0, ..., 0.0]
 */
float *iris_schedule_linear(int num_steps) {
    float *schedule = (float *)malloc((num_steps + 1) * sizeof(float));
    for (int i = 0; i <= num_steps; i++) {
        schedule[i] = 1.0f - (float)i / (float)num_steps;
    }
    return schedule;
}

/*
 * Power schedule: denser steps at the start (high noise), sparser at the end.
 * schedule[i] = 1 - (i/n)^alpha
 * alpha=1.0 is linear, alpha=2.0 is quadratic, etc.
 */
float *iris_schedule_power(int num_steps, float alpha) {
    float *schedule = (float *)malloc((num_steps + 1) * sizeof(float));
    for (int i = 0; i <= num_steps; i++) {
        float t = (float)i / (float)num_steps;
        schedule[i] = 1.0f - powf(t, alpha);
    }
    return schedule;
}

/*
 * Shifted sigmoid schedule (better for flow matching)
 * shift controls where the inflection point is
 */
float *iris_schedule_sigmoid(int num_steps, float shift) {
    float *schedule = (float *)malloc((num_steps + 1) * sizeof(float));

    for (int i = 0; i <= num_steps; i++) {
        float t = (float)i / (float)num_steps;
        /* Shifted sigmoid: more steps at the end */
        float x = (t - 0.5f) * 10.0f + shift;
        schedule[i] = 1.0f - 1.0f / (1.0f + expf(-x));
    }

    /* Ensure endpoints */
    schedule[0] = 1.0f;
    schedule[num_steps] = 0.0f;

    return schedule;
}

/*
 * Resolution-dependent schedule (as used in FLUX.2)
 * Higher resolutions use more steps at the start
 */
float *iris_schedule_resolution(int num_steps, int height, int width) {
    float *schedule = (float *)malloc((num_steps + 1) * sizeof(float));

    /* Compute shift based on resolution */
    int pixels = height * width;
    float shift = 0.0f;
    if (pixels >= 1024 * 1024) {
        shift = 1.0f;  /* High res: more early steps */
    } else if (pixels >= 512 * 512) {
        shift = 0.5f;
    }

    for (int i = 0; i <= num_steps; i++) {
        float t = (float)i / (float)num_steps;
        /* Apply shift */
        t = powf(t, 1.0f + shift * 0.5f);
        schedule[i] = 1.0f - t;
    }

    return schedule;
}

/*
 * FLUX.2 official schedule with empirical mu calculation
 * Matches Python's get_schedule() function from official flux2 code
 */
/* Compute the empirical shift parameter mu for the resolution-dependent
 * noise schedule. The constants a1, b1, a2, b2 are fitted from the Flux
 * training distribution and control how the SNR schedule adapts to different
 * image resolutions. Higher resolution images need more denoising steps
 * at high noise levels. Interpolates between two linear fits based on
 * step count, with a cutoff at 4300 tokens. */
static float compute_empirical_mu(int image_seq_len, int num_steps) {
    const float a1 = 8.73809524e-05f, b1 = 1.89833333f;
    const float a2 = 0.00016927f, b2 = 0.45666666f;

    if (image_seq_len > 4300) {
        return a2 * image_seq_len + b2;
    }

    float m_200 = a2 * image_seq_len + b2;
    float m_10 = a1 * image_seq_len + b1;

    float a = (m_200 - m_10) / 190.0f;
    float b = m_200 - 200.0f * a;
    return a * num_steps + b;
}

/* Apply exponential SNR (Signal-to-Noise Ratio) shift to a timestep.
 * Maps t in [0,1] through t/(t + (1-t)*exp(-mu)), shifting the schedule
 * toward more time spent at higher noise levels. The boundary guards
 * (t<=0, t>=1) prevent division by zero. */
static float generalized_time_snr_shift(float t, float mu, float sigma) {
    /* t / (1 - t) with exp(mu) shift */
    if (t <= 0.0f) return 0.0f;
    if (t >= 1.0f) return 1.0f;
    return expf(mu) / (expf(mu) + powf(1.0f / t - 1.0f, sigma));
}

float *iris_schedule_flux(int num_steps, int image_seq_len) {
    float *schedule = (float *)malloc((num_steps + 1) * sizeof(float));
    float mu = compute_empirical_mu(image_seq_len, num_steps);

    for (int i = 0; i <= num_steps; i++) {
        float t = 1.0f - (float)i / (float)num_steps;  /* Linear from 1 to 0 */
        schedule[i] = generalized_time_snr_shift(t, mu, 1.0f);
    }

    return schedule;
}

/*
 * Z-Image schedule: FlowMatchEulerDiscreteScheduler with static shift.
 *
 * Matches diffusers FlowMatchEulerDiscreteScheduler.set_timesteps()
 * for use_dynamic_shifting=False, invert_sigmas=False.
 *
 * Diffusers behavior (with num_train_timesteps=1000, shift=3.0):
 * 1) Build training sigmas from [1.0 .. 1/1000], then shift once.
 * 2) For inference, linspace from sigma_max to sigma_min (already shifted).
 * 3) Shift again.
 * 4) Append terminal sigma=0.
 *
 * Returns array of num_steps+1 sigma values.
 * The sampling loop uses these as sigmas directly with Euler:
 *   dt = sigma_next - sigma (negative for denoising)
 *   The transformer timestep = (1 - sigma), range [0, 1]
 */
float *iris_schedule_zimage(int num_steps, int image_seq_len) {
    (void)image_seq_len;  /* Not used for static shift */
    float *schedule = (float *)malloc((num_steps + 1) * sizeof(float));
    const float shift = 3.0f;
    const float sigma_max = 1.0f;
    const float sigma_train_min = 1.0f / 1000.0f;  /* num_train_timesteps=1000 */
    const float sigma_min = shift * sigma_train_min /
                            (1.0f + (shift - 1.0f) * sigma_train_min);

    /* Diffusers set_timesteps(): linspace(sigma_max, sigma_min, num_steps),
     * then apply static shift one more time. */
    for (int i = 0; i < num_steps; i++) {
        float u = (num_steps > 1) ? (float)i / (float)(num_steps - 1) : 0.0f;
        float raw = sigma_max + (sigma_min - sigma_max) * u;
        schedule[i] = shift * raw / (1.0f + (shift - 1.0f) * raw);
    }
    /* Diffusers appends terminal sigma=0 (invert_sigmas=False). */
    schedule[num_steps] = 0.0f;

    return schedule;
}

/*
 * FLUX.3 schedule: the pipeline passes sigmas = linspace(1, 1/n, n) to
 * FlowMatch Euler with a static shift of 6.93 (which keeps the noise level
 * high for most of the run), then the terminal sigma 0 is appended.
 */
float *iris_schedule_flux3(int num_steps) {
    float *schedule = (float *)malloc((num_steps + 1) * sizeof(float));
    const float shift = 6.93f;

    for (int i = 0; i < num_steps; i++) {
        float u = (num_steps > 1) ? (float)i / (float)(num_steps - 1) : 0.0f;
        float raw = 1.0f + (1.0f / num_steps - 1.0f) * u;
        schedule[i] = shift * raw / (1.0f + (shift - 1.0f) * raw);
    }
    schedule[num_steps] = 0.0f;

    return schedule;
}

/* ========================================================================
 * Euler Sampler for Rectified Flow
 * ======================================================================== */

/*
 * Single Euler step:
 * z_next = z_t + (t_next - t_curr) * v(z_t, t_curr)
 *
 * Where v is the velocity predicted by the model.
 * In rectified flow: v = (z_data - z_noise) at timestep t
 */

typedef struct iris_transformer_flux iris_transformer_flux_t;
typedef struct iris_vae iris_vae_t;

/* Free cached mmap weights after denoising */
extern void iris_transformer_free_mmap_cache_flux(iris_transformer_flux_t *tf);

/* Forward declarations */
extern float *iris_transformer_forward_flux(iris_transformer_flux_t *tf,
                                       const float *img_latent, int img_h, int img_w,
                                       const float *txt_emb, int txt_seq,
                                       float timestep);

/* Z-Image transformer */
typedef struct zi_transformer zi_transformer_t;
extern float *iris_transformer_forward_zimage(zi_transformer_t *tf,
                                      const float *latent, int latent_h, int latent_w,
                                      float timestep,
                                      const float *cap_feats, int cap_seq_len);

/* Forward declaration for in-context conditioning (img2img) */
extern float *iris_transformer_forward_refs_flux(iris_transformer_flux_t *tf,
                                                 const float *img_latent, int img_h, int img_w,
                                                 const float *ref_latent, int ref_h, int ref_w,
                                                 int t_offset,
                                                 const float *txt_emb, int txt_seq,
                                                 float timestep);

/* Forward declaration for multi-reference conditioning */
typedef struct {
    const float *latent;  /* Reference latent in NCHW format */
    int h, w;             /* Latent dimensions */
    int t_offset;         /* RoPE T coordinate (10, 20, 30, ...) */
} iris_ref_t;

extern float *iris_transformer_forward_multirefs_flux(iris_transformer_flux_t *tf,
                                                       const float *img_latent, int img_h, int img_w,
                                                       const iris_ref_t *refs, int num_refs,
                                                       const float *txt_emb, int txt_seq,
                                                       float timestep);

/* VAE decode for step image callback */
extern iris_image *iris_vae_decode(iris_vae_t *vae, const float *latent,
                                   int batch, int latent_h, int latent_w);
extern void iris_image_free(iris_image *img);

/* Euler ODE sampler for Flux distilled models (txt2img). Iterates
 * z_{n+1} = z_n + dt * v(z_n, t) where v is the velocity predicted by
 * the transformer. The schedule provides sigma values from 1 to 0;
 * the timestep passed to the transformer equals sigma, which it scales
 * internally by 1000 for the sinusoidal embedding. */
float *iris_sample_euler_flux(void *transformer, void *text_encoder,
                         float *z, int batch, int channels, int h, int w,
                         const float *text_emb, int text_seq,
                         const float *schedule, int num_steps,
                         void (*progress_callback)(int step, int total)) {
    (void)text_encoder;  /* Reserved for future use */
    iris_transformer_flux_t *tf = (iris_transformer_flux_t *)transformer;
    int latent_size = batch * channels * h * w;

    /* Working buffers */
    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    float *v_cond = NULL;

    iris_copy(z_curr, z, latent_size);

    /* Reset timing counters */
    iris_reset_timing();
    double total_denoising_start = get_time_ms();
    double step_times[IRIS_MAX_STEPS];

    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];
        float t_next = schedule[step + 1];
        float dt = t_next - t_curr;  /* Negative for denoising */

        double step_start = get_time_ms();

        /* Notify step start */
        if (iris_step_callback)
            iris_step_callback(step + 1, num_steps);

        /* Predict velocity with conditioning */
        v_cond = iris_transformer_forward_flux(tf, z_curr, h, w,
                                          text_emb, text_seq, t_curr);

        /* Euler step: z_next = z_curr + dt * v */
        iris_axpy(z_curr, dt, v_cond, latent_size);

        free(v_cond);

        step_times[step] = get_time_ms() - step_start;

        if (progress_callback) {
            progress_callback(step + 1, num_steps);
        }

        /* Step image callback - decode and display intermediate result */
        if (iris_step_image_callback && iris_step_image_vae && step + 1 < num_steps) {
            iris_image *img = iris_vae_decode((iris_vae_t *)iris_step_image_vae,
                                              z_curr, 1, h, w);
            if (img) {
                iris_step_image_callback(step + 1, num_steps, img);
                iris_image_free(img);
            }
        }
    }

    /* Print timing summary */
    if (iris_verbose) {
        double total_denoising = get_time_ms() - total_denoising_start;
        fprintf(stderr, "\nDenoising timing breakdown:\n");
        for (int step = 0; step < num_steps; step++) {
            fprintf(stderr, "  Step %d: %.1f ms\n", step + 1, step_times[step]);
        }
        fprintf(stderr, "  Total denoising: %.1f ms (%.2f s)\n", total_denoising, total_denoising / 1000.0);
        if (iris_timing_transformer_double > 0 || iris_timing_transformer_single > 0) {
            fprintf(stderr, "  Transformer breakdown:\n");
            fprintf(stderr, "    Double blocks: %.1f ms (%.1f%%)\n",
                    iris_timing_transformer_double, 100.0 * iris_timing_transformer_double / iris_timing_transformer_total);
            fprintf(stderr, "    Single blocks: %.1f ms (%.1f%%)\n",
                    iris_timing_transformer_single, 100.0 * iris_timing_transformer_single / iris_timing_transformer_total);
            fprintf(stderr, "    Final layer:   %.1f ms (%.1f%%)\n",
                    iris_timing_transformer_final, 100.0 * iris_timing_transformer_final / iris_timing_transformer_total);
            fprintf(stderr, "    Total:         %.1f ms\n", iris_timing_transformer_total);
            /* Print fine-grained single block profile if available */
            extern void iris_print_blas_profile(void);
            iris_print_blas_profile();
        }
    }

    iris_transformer_free_mmap_cache_flux(tf);
    return z_curr;
}

/* Euler sampler for Z-Image models. Key differences from Flux: the
 * transformer timestep is (1-sigma) instead of sigma*1000, the velocity
 * is negated in the update step (z += dt * (-v)), and the latent operates
 * at pre-patchification dimensions [channels, H/8, W/8]. The schedule
 * provides sigma values from ~1 to 0 with a static shift of 3.0. */
float *iris_sample_euler_zimage(void *transformer,
                                 float *z, int batch, int channels, int h, int w,
                                 int patch_size,
                                 const float *cap_feats, int cap_seq,
                                 const float *schedule, int num_steps,
                                 void (*progress_callback)(int step, int total)) {
    zi_transformer_t *tf = (zi_transformer_t *)transformer;
    int latent_size = batch * channels * h * w;

    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    iris_copy(z_curr, z, latent_size);

    iris_reset_timing();
    double total_denoising_start = get_time_ms();
    double step_times[IRIS_MAX_STEPS];
    float *step_latent = NULL;
    int step_h = h;
    int step_w = w;

    if (iris_step_image_callback && iris_step_image_vae && num_steps > 1 &&
        patch_size > 1) {
        if (h % patch_size == 0 && w % patch_size == 0) {
            int step_ch = channels * patch_size * patch_size;
            step_h = h / patch_size;
            step_w = w / patch_size;
            int step_latent_size = batch * step_ch * step_h * step_w;
            step_latent = (float *)malloc(step_latent_size * sizeof(float));
        } else if (iris_verbose) {
            fprintf(stderr, "Warning: disabling zImage step previews (invalid patch size)\n");
        }
    }

    for (int step = 0; step < num_steps; step++) {
        float sigma = schedule[step];
        float sigma_next = schedule[step + 1];
        float dt = sigma_next - sigma;  /* Negative for denoising */

        /* The transformer timestep is (1 - sigma), matching the Python pipeline:
         * pipeline: timestep = (1000 - t_scheduler) / 1000 = 1 - sigma
         * transformer: t = timestep * t_scale (1000) -> sinusoidal_input */
        float timestep = 1.0f - sigma;

        double step_start = get_time_ms();

        if (iris_step_callback)
            iris_step_callback(step + 1, num_steps);

        /* Run Z-Image transformer */
        float *model_out = iris_transformer_forward_zimage(tf, z_curr, h, w,
                                                    timestep,
                                                    cap_feats, cap_seq);

        /* Euler step: z_next = z + dt * (-model_output) */
        for (int i = 0; i < latent_size; i++) {
            z_curr[i] += dt * (-model_out[i]);
        }

        free(model_out);

        step_times[step] = get_time_ms() - step_start;

        if (progress_callback)
            progress_callback(step + 1, num_steps);

        /* Step image callback */
        if (iris_step_image_callback && iris_step_image_vae && step + 1 < num_steps) {
            const float *decode_latent = z_curr;
            int decode_h = h;
            int decode_w = w;

            if (patch_size > 1) {
                if (!step_latent) continue;
                iris_patchify(step_latent, z_curr, batch, channels, h, w, patch_size);
                decode_latent = step_latent;
                decode_h = step_h;
                decode_w = step_w;
            }

            iris_image *img = iris_vae_decode((iris_vae_t *)iris_step_image_vae,
                                              decode_latent, 1, decode_h, decode_w);
            if (img) {
                iris_step_image_callback(step + 1, num_steps, img);
                iris_image_free(img);
            }
        }
    }

    if (iris_verbose) {
        double total_denoising = get_time_ms() - total_denoising_start;
        fprintf(stderr, "\nDenoising timing breakdown (Z-Image):\n");
        for (int step = 0; step < num_steps; step++) {
            fprintf(stderr, "  Step %d: %.1f ms\n", step + 1, step_times[step]);
        }
        fprintf(stderr, "  Total denoising: %.1f ms (%.2f s)\n", total_denoising, total_denoising / 1000.0);
        if (iris_timing_zi_total > 0.0) {
            fprintf(stderr, "  Transformer breakdown (Z-Image GPU):\n");
            fprintf(stderr, "    Embeddings+RoPE: %.1f ms (%.1f%%)\n",
                    iris_timing_zi_embeddings,
                    100.0 * iris_timing_zi_embeddings / iris_timing_zi_total);
            fprintf(stderr, "    Noise refiner:   %.1f ms (%.1f%%)\n",
                    iris_timing_zi_noise_refiner,
                    100.0 * iris_timing_zi_noise_refiner / iris_timing_zi_total);
            fprintf(stderr, "    Context refiner: %.1f ms (%.1f%%)\n",
                    iris_timing_zi_context_refiner,
                    100.0 * iris_timing_zi_context_refiner / iris_timing_zi_total);
            fprintf(stderr, "    Main blocks:     %.1f ms (%.1f%%)\n",
                    iris_timing_zi_main_blocks,
                    100.0 * iris_timing_zi_main_blocks / iris_timing_zi_total);
            fprintf(stderr, "    Final layer:     %.1f ms (%.1f%%)\n",
                    iris_timing_zi_final,
                    100.0 * iris_timing_zi_final / iris_timing_zi_total);
            fprintf(stderr, "    Total:           %.1f ms\n", iris_timing_zi_total);
        }
    }

    free(step_latent);
    return z_curr;
}

/* Euler sampler for single-reference img2img. The reference image is
 * VAE-encoded and concatenated with the noised target as extra tokens.
 * A RoPE T offset (default 10) distinguishes reference from target in
 * the positional encoding. The transformer attends to both via joint
 * attention, implementing in-context conditioning. */
float *iris_sample_euler_refs_flux(void *transformer, void *text_encoder,
                                   float *z, int batch, int channels, int h, int w,
                                   const float *ref_latent, int ref_h, int ref_w,
                                   int t_offset,
                                   const float *text_emb, int text_seq,
                                   const float *schedule, int num_steps,
                                   void (*progress_callback)(int step, int total)) {
    (void)text_encoder;  /* Reserved for future use */
    iris_transformer_flux_t *tf = (iris_transformer_flux_t *)transformer;
    int latent_size = batch * channels * h * w;

    /* Working buffer */
    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    iris_copy(z_curr, z, latent_size);

    /* Reset timing counters */
    iris_reset_timing();
    double total_denoising_start = get_time_ms();
    double step_times[IRIS_MAX_STEPS];

    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];
        float t_next = schedule[step + 1];
        float dt = t_next - t_curr;

        double step_start = get_time_ms();

        /* Notify step start */
        if (iris_step_callback)
            iris_step_callback(step + 1, num_steps);

        /* Predict velocity with reference image conditioning */
        float *v = iris_transformer_forward_refs_flux(tf,
                                                      z_curr, h, w,
                                                      ref_latent, ref_h, ref_w,
                                                      t_offset,
                                                      text_emb, text_seq,
                                                      t_curr);

        if (!v) {
            free(z_curr);
            return NULL;
        }

        /* Euler step: z_next = z_curr + dt * v */
        iris_axpy(z_curr, dt, v, latent_size);

        free(v);

        step_times[step] = get_time_ms() - step_start;

        if (progress_callback) {
            progress_callback(step + 1, num_steps);
        }

        /* Step image callback - decode and display intermediate result */
        if (iris_step_image_callback && iris_step_image_vae && step + 1 < num_steps) {
            iris_image *img = iris_vae_decode((iris_vae_t *)iris_step_image_vae,
                                              z_curr, 1, h, w);
            if (img) {
                iris_step_image_callback(step + 1, num_steps, img);
                iris_image_free(img);
            }
        }
    }

    /* Print timing summary */
    if (iris_verbose) {
        double total_denoising = get_time_ms() - total_denoising_start;
        fprintf(stderr, "\nDenoising timing breakdown (img2img with refs):\n");
        for (int step = 0; step < num_steps; step++) {
            fprintf(stderr, "  Step %d: %.1f ms\n", step + 1, step_times[step]);
        }
        fprintf(stderr, "  Total denoising: %.1f ms (%.2f s)\n", total_denoising, total_denoising / 1000.0);
    }

    iris_transformer_free_mmap_cache_flux(tf);
    return z_curr;
}

/*
 * Sample using Euler method with multiple reference images.
 * Each reference gets a different T offset in RoPE (10, 20, 30, ...).
 */
float *iris_sample_euler_multirefs_flux(void *transformer, void *text_encoder,
                                         float *z, int batch, int channels, int h, int w,
                                         const iris_ref_t *refs, int num_refs,
                                         const float *text_emb, int text_seq,
                                         const float *schedule, int num_steps,
                                         void (*progress_callback)(int step, int total)) {
    (void)text_encoder;
    iris_transformer_flux_t *tf = (iris_transformer_flux_t *)transformer;
    int latent_size = batch * channels * h * w;

    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    iris_copy(z_curr, z, latent_size);

    iris_reset_timing();
    double total_denoising_start = get_time_ms();
    double step_times[IRIS_MAX_STEPS];

    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];
        float t_next = schedule[step + 1];
        float dt = t_next - t_curr;

        double step_start = get_time_ms();

        if (iris_step_callback)
            iris_step_callback(step + 1, num_steps);

        /* Predict velocity with multiple reference images */
        float *v = iris_transformer_forward_multirefs_flux(tf,
                                                            z_curr, h, w,
                                                            refs, num_refs,
                                                            text_emb, text_seq,
                                                            t_curr);

        if (!v) {
            free(z_curr);
            return NULL;
        }

        /* Euler step */
        iris_axpy(z_curr, dt, v, latent_size);
        free(v);

        step_times[step] = get_time_ms() - step_start;

        if (progress_callback)
            progress_callback(step + 1, num_steps);

        if (iris_step_image_callback && iris_step_image_vae && step + 1 < num_steps) {
            iris_image *img = iris_vae_decode((iris_vae_t *)iris_step_image_vae,
                                              z_curr, 1, h, w);
            if (img) {
                iris_step_image_callback(step + 1, num_steps, img);
                iris_image_free(img);
            }
        }
    }

    if (iris_verbose) {
        double total_denoising = get_time_ms() - total_denoising_start;
        fprintf(stderr, "\nDenoising timing breakdown (multi-ref, %d refs):\n", num_refs);
        for (int step = 0; step < num_steps; step++) {
            fprintf(stderr, "  Step %d: %.1f ms\n", step + 1, step_times[step]);
        }
        fprintf(stderr, "  Total denoising: %.1f ms (%.2f s)\n", total_denoising, total_denoising / 1000.0);
    }

    iris_transformer_free_mmap_cache_flux(tf);
    return z_curr;
}

/* ========================================================================
 * CFG (Classifier-Free Guidance) Samplers for Base Model
 *
 * These run the transformer twice per step: once with empty text (uncond)
 * and once with the real prompt (cond), then combine:
 *   v = v_uncond + guidance_scale * (v_cond - v_uncond)
 * ======================================================================== */

/* Euler sampler with Classifier-Free Guidance for Flux base models.
 * Each step runs the transformer twice: once with empty prompt (unconditional)
 * and once with real prompt (conditional). Combined as
 * v = v_uncond + guidance_scale * (v_cond - v_uncond), which steers
 * generation toward the prompt at the cost of 2x compute per step. */
float *iris_sample_euler_cfg_flux(void *transformer, void *text_encoder,
                              float *z, int batch, int channels, int h, int w,
                              const float *text_emb_cond, int text_seq_cond,
                              const float *text_emb_uncond, int text_seq_uncond,
                              float guidance_scale,
                              const float *schedule, int num_steps,
                              void (*progress_callback)(int step, int total)) {
    (void)text_encoder;
    iris_transformer_flux_t *tf = (iris_transformer_flux_t *)transformer;
    int latent_size = batch * channels * h * w;

    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    iris_copy(z_curr, z, latent_size);

    iris_reset_timing();
    double total_denoising_start = get_time_ms();
    double step_times[IRIS_MAX_STEPS];

    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];
        float t_next = schedule[step + 1];
        float dt = t_next - t_curr;

        double step_start = get_time_ms();

        if (iris_step_callback)
            iris_step_callback(step + 1, num_steps);

        /* Unconditioned prediction */
        float *v_uncond = iris_transformer_forward_flux(tf, z_curr, h, w,
                                                    text_emb_uncond, text_seq_uncond,
                                                    t_curr);

        /* Conditioned prediction */
        float *v_cond = iris_transformer_forward_flux(tf, z_curr, h, w,
                                                  text_emb_cond, text_seq_cond,
                                                  t_curr);

        /* CFG combine: v = v_uncond + scale * (v_cond - v_uncond) */
        for (int i = 0; i < latent_size; i++) {
            float v = v_uncond[i] + guidance_scale * (v_cond[i] - v_uncond[i]);
            z_curr[i] += dt * v;
        }

        free(v_uncond);
        free(v_cond);

        step_times[step] = get_time_ms() - step_start;

        if (progress_callback)
            progress_callback(step + 1, num_steps);

        if (iris_step_image_callback && iris_step_image_vae && step + 1 < num_steps) {
            iris_image *img = iris_vae_decode((iris_vae_t *)iris_step_image_vae,
                                              z_curr, 1, h, w);
            if (img) {
                iris_step_image_callback(step + 1, num_steps, img);
                iris_image_free(img);
            }
        }
    }

    if (iris_verbose) {
        double total_denoising = get_time_ms() - total_denoising_start;
        fprintf(stderr, "\nDenoising timing breakdown (CFG, guidance=%.1f):\n", guidance_scale);
        for (int step = 0; step < num_steps; step++) {
            fprintf(stderr, "  Step %d: %.1f ms\n", step + 1, step_times[step]);
        }
        fprintf(stderr, "  Total denoising: %.1f ms (%.2f s)\n", total_denoising, total_denoising / 1000.0);
    }

    iris_transformer_free_mmap_cache_flux(tf);
    return z_curr;
}

/*
 * Euler sampler with CFG and single reference image (img2img).
 */
float *iris_sample_euler_cfg_refs_flux(void *transformer, void *text_encoder,
                                        float *z, int batch, int channels, int h, int w,
                                        const float *ref_latent, int ref_h, int ref_w,
                                        int t_offset,
                                        const float *text_emb_cond, int text_seq_cond,
                                        const float *text_emb_uncond, int text_seq_uncond,
                                        float guidance_scale,
                                        const float *schedule, int num_steps,
                                        void (*progress_callback)(int step, int total)) {
    (void)text_encoder;
    iris_transformer_flux_t *tf = (iris_transformer_flux_t *)transformer;
    int latent_size = batch * channels * h * w;

    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    iris_copy(z_curr, z, latent_size);

    iris_reset_timing();
    double total_denoising_start = get_time_ms();
    double step_times[IRIS_MAX_STEPS];

    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];
        float t_next = schedule[step + 1];
        float dt = t_next - t_curr;

        double step_start = get_time_ms();

        if (iris_step_callback)
            iris_step_callback(step + 1, num_steps);

        /* Unconditioned prediction (with ref) */
        float *v_uncond = iris_transformer_forward_refs_flux(tf,
                              z_curr, h, w,
                              ref_latent, ref_h, ref_w, t_offset,
                              text_emb_uncond, text_seq_uncond, t_curr);

        /* Conditioned prediction (with ref) */
        float *v_cond = iris_transformer_forward_refs_flux(tf,
                            z_curr, h, w,
                            ref_latent, ref_h, ref_w, t_offset,
                            text_emb_cond, text_seq_cond, t_curr);

        if (!v_uncond || !v_cond) {
            free(v_uncond);
            free(v_cond);
            free(z_curr);
            return NULL;
        }

        /* CFG combine */
        for (int i = 0; i < latent_size; i++) {
            float v = v_uncond[i] + guidance_scale * (v_cond[i] - v_uncond[i]);
            z_curr[i] += dt * v;
        }

        free(v_uncond);
        free(v_cond);

        step_times[step] = get_time_ms() - step_start;

        if (progress_callback)
            progress_callback(step + 1, num_steps);

        if (iris_step_image_callback && iris_step_image_vae && step + 1 < num_steps) {
            iris_image *img = iris_vae_decode((iris_vae_t *)iris_step_image_vae,
                                              z_curr, 1, h, w);
            if (img) {
                iris_step_image_callback(step + 1, num_steps, img);
                iris_image_free(img);
            }
        }
    }

    if (iris_verbose) {
        double total_denoising = get_time_ms() - total_denoising_start;
        fprintf(stderr, "\nDenoising timing breakdown (CFG img2img, guidance=%.1f):\n", guidance_scale);
        for (int step = 0; step < num_steps; step++) {
            fprintf(stderr, "  Step %d: %.1f ms\n", step + 1, step_times[step]);
        }
        fprintf(stderr, "  Total denoising: %.1f ms (%.2f s)\n", total_denoising, total_denoising / 1000.0);
    }

    iris_transformer_free_mmap_cache_flux(tf);
    return z_curr;
}

/*
 * Euler sampler with CFG and multiple reference images.
 */
float *iris_sample_euler_cfg_multirefs_flux(void *transformer, void *text_encoder,
                                              float *z, int batch, int channels, int h, int w,
                                              const iris_ref_t *refs, int num_refs,
                                              const float *text_emb_cond, int text_seq_cond,
                                              const float *text_emb_uncond, int text_seq_uncond,
                                              float guidance_scale,
                                              const float *schedule, int num_steps,
                                              void (*progress_callback)(int step, int total)) {
    (void)text_encoder;
    iris_transformer_flux_t *tf = (iris_transformer_flux_t *)transformer;
    int latent_size = batch * channels * h * w;

    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    iris_copy(z_curr, z, latent_size);

    iris_reset_timing();
    double total_denoising_start = get_time_ms();
    double step_times[IRIS_MAX_STEPS];

    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];
        float t_next = schedule[step + 1];
        float dt = t_next - t_curr;

        double step_start = get_time_ms();

        if (iris_step_callback)
            iris_step_callback(step + 1, num_steps);

        /* Unconditioned prediction (with refs) */
        float *v_uncond = iris_transformer_forward_multirefs_flux(tf,
                              z_curr, h, w,
                              refs, num_refs,
                              text_emb_uncond, text_seq_uncond, t_curr);

        /* Conditioned prediction (with refs) */
        float *v_cond = iris_transformer_forward_multirefs_flux(tf,
                            z_curr, h, w,
                            refs, num_refs,
                            text_emb_cond, text_seq_cond, t_curr);

        if (!v_uncond || !v_cond) {
            free(v_uncond);
            free(v_cond);
            free(z_curr);
            return NULL;
        }

        /* CFG combine */
        for (int i = 0; i < latent_size; i++) {
            float v = v_uncond[i] + guidance_scale * (v_cond[i] - v_uncond[i]);
            z_curr[i] += dt * v;
        }

        free(v_uncond);
        free(v_cond);

        step_times[step] = get_time_ms() - step_start;

        if (progress_callback)
            progress_callback(step + 1, num_steps);

        if (iris_step_image_callback && iris_step_image_vae && step + 1 < num_steps) {
            iris_image *img = iris_vae_decode((iris_vae_t *)iris_step_image_vae,
                                              z_curr, 1, h, w);
            if (img) {
                iris_step_image_callback(step + 1, num_steps, img);
                iris_image_free(img);
            }
        }
    }

    if (iris_verbose) {
        double total_denoising = get_time_ms() - total_denoising_start;
        fprintf(stderr, "\nDenoising timing breakdown (CFG multi-ref, %d refs, guidance=%.1f):\n",
                num_refs, guidance_scale);
        for (int step = 0; step < num_steps; step++) {
            fprintf(stderr, "  Step %d: %.1f ms\n", step + 1, step_times[step]);
        }
        fprintf(stderr, "  Total denoising: %.1f ms (%.2f s)\n", total_denoising, total_denoising / 1000.0);
    }

    iris_transformer_free_mmap_cache_flux(tf);
    return z_curr;
}

/* ========================================================================
 * FLUX.3 Sampler
 * ======================================================================== */

typedef struct iris_transformer_flux3 iris_transformer_flux3_t;
extern float *iris_transformer_forward_flux3(iris_transformer_flux3_t *tf,
                                             const float *img, int img_h, int img_w,
                                             const float *txt, int txt_seq,
                                             const float *const *refs, const int *ref_h,
                                             const int *ref_w, int num_refs,
                                             float t, float t_target);

/* Euler sampler for FLUX.3: z_{n+1} = z_n + (sigma_{n+1} - sigma_n) * v.
 * The transformer gets sigma as timestep; flow-map (distilled) models also
 * get the sigma the step jumps to. With uncond_emb (base models) each step
 * runs the transformer twice for classifier-free guidance:
 * v = v_uncond + guidance * (v_cond - v_uncond). Reference latents, if any,
 * condition every pass (in-context conditioning). */
float *iris_sample_euler_flux3(void *transformer,
                               const float *z, int channels, int h, int w,
                               const float *text_emb, int text_seq,
                               const float *uncond_emb, int uncond_seq,
                               const float *const *refs, const int *ref_h,
                               const int *ref_w, int num_refs,
                               const float *schedule, int num_steps,
                               float guidance,
                               void (*progress_callback)(int step, int total)) {
    iris_transformer_flux3_t *tf = (iris_transformer_flux3_t *)transformer;
    int latent_size = channels * h * w;

    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    iris_copy(z_curr, z, latent_size);

    double total_start = get_time_ms();
    double step_times[IRIS_MAX_STEPS];

    for (int step = 0; step < num_steps; step++) {
        float sigma = schedule[step];
        float sigma_next = schedule[step + 1];
        double step_start = get_time_ms();

        if (iris_step_callback)
            iris_step_callback(step + 1, num_steps);

        float *v = iris_transformer_forward_flux3(tf, z_curr, h, w, text_emb, text_seq,
                                                  refs, ref_h, ref_w, num_refs,
                                                  sigma, sigma_next);
        if (v && uncond_emb) {
            float *v_uncond = iris_transformer_forward_flux3(tf, z_curr, h, w,
                                                             uncond_emb, uncond_seq,
                                                             refs, ref_h, ref_w, num_refs,
                                                             sigma, sigma_next);
            if (!v_uncond) {
                free(v);
                v = NULL;
            } else {
                for (int i = 0; i < latent_size; i++)
                    v[i] = v_uncond[i] + guidance * (v[i] - v_uncond[i]);
                free(v_uncond);
            }
        }
        if (!v) {
            free(z_curr);
            return NULL;
        }

        iris_axpy(z_curr, sigma_next - sigma, v, latent_size);
        free(v);

        if (step < IRIS_MAX_STEPS) step_times[step] = get_time_ms() - step_start;

        if (progress_callback)
            progress_callback(step + 1, num_steps);

        if (iris_step_image_callback && iris_step_image_vae && step + 1 < num_steps) {
            iris_image *img = iris_vae_decode((iris_vae_t *)iris_step_image_vae,
                                              z_curr, 1, h, w);
            if (img) {
                iris_step_image_callback(step + 1, num_steps, img);
                iris_image_free(img);
            }
        }
    }

    if (iris_verbose) {
        fprintf(stderr, "\nDenoising timing breakdown (FLUX.3):\n");
        for (int step = 0; step < num_steps && step < IRIS_MAX_STEPS; step++)
            fprintf(stderr, "  Step %d: %.1f ms\n", step + 1, step_times[step]);
        double total = get_time_ms() - total_start;
        fprintf(stderr, "  Total denoising: %.1f ms (%.2f s)\n", total, total / 1000.0);
    }

    return z_curr;
}

/*
 * Sample using Euler method with stochastic noise injection.
 * This can help with diversity and quality.
 */
float *iris_sample_euler_ancestral(void *transformer,
                                   float *z, int batch, int channels, int h, int w,
                                   const float *text_emb, int text_seq,
                                   const float *schedule, int num_steps,
                                   float eta,
                                   void (*progress_callback)(int step, int total)) {
    iris_transformer_flux_t *tf = (iris_transformer_flux_t *)transformer;
    int latent_size = batch * channels * h * w;

    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    float *noise = (float *)malloc(latent_size * sizeof(float));

    iris_copy(z_curr, z, latent_size);

    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];
        float t_next = schedule[step + 1];
        float dt = t_next - t_curr;

        /* Predict velocity */
        float *v = iris_transformer_forward_flux(tf, z_curr, h, w,
                                            text_emb, text_seq, t_curr);

        /* Euler step */
        iris_axpy(z_curr, dt, v, latent_size);

        /* Add noise (ancestral sampling) */
        if (eta > 0 && step < num_steps - 1) {
            float sigma = eta * sqrtf(fabsf(dt));
            iris_randn(noise, latent_size);
            iris_axpy(z_curr, sigma, noise, latent_size);
        }

        free(v);

        if (progress_callback) {
            progress_callback(step + 1, num_steps);
        }

        /* Step image callback - decode and display intermediate result */
        if (iris_step_image_callback && iris_step_image_vae && step + 1 < num_steps) {
            iris_image *img = iris_vae_decode((iris_vae_t *)iris_step_image_vae,
                                              z_curr, 1, h, w);
            if (img) {
                iris_step_image_callback(step + 1, num_steps, img);
                iris_image_free(img);
            }
        }
    }

    free(noise);
    iris_transformer_free_mmap_cache_flux(tf);
    return z_curr;
}

/* ========================================================================
 * Heun Sampler (2nd order)
 * ======================================================================== */

/*
 * Heun's method (improved Euler):
 * 1. Predict: z_pred = z_t + dt * v(z_t, t)
 * 2. Correct: z_next = z_t + dt/2 * (v(z_t, t) + v(z_pred, t+dt))
 */
float *iris_sample_heun(void *transformer,
                        float *z, int batch, int channels, int h, int w,
                        const float *text_emb, int text_seq,
                        const float *schedule, int num_steps,
                        void (*progress_callback)(int step, int total)) {
    iris_transformer_flux_t *tf = (iris_transformer_flux_t *)transformer;
    int latent_size = batch * channels * h * w;

    float *z_curr = (float *)malloc(latent_size * sizeof(float));
    float *z_pred = (float *)malloc(latent_size * sizeof(float));

    iris_copy(z_curr, z, latent_size);

    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];
        float t_next = schedule[step + 1];
        float dt = t_next - t_curr;

        /* First velocity estimate */
        float *v1 = iris_transformer_forward_flux(tf, z_curr, h, w,
                                             text_emb, text_seq, t_curr);

        /* Predict next state */
        iris_copy(z_pred, z_curr, latent_size);
        iris_axpy(z_pred, dt, v1, latent_size);

        /* Second velocity estimate (only if not last step) */
        if (step < num_steps - 1) {
            float *v2 = iris_transformer_forward_flux(tf, z_pred, h, w,
                                                 text_emb, text_seq, t_next);

            /* Heun correction: z_next = z_curr + dt/2 * (v1 + v2) */
            for (int i = 0; i < latent_size; i++) {
                z_curr[i] += 0.5f * dt * (v1[i] + v2[i]);
            }

            free(v2);
        } else {
            /* Last step: just use Euler */
            iris_axpy(z_curr, dt, v1, latent_size);
        }

        free(v1);

        if (progress_callback) {
            progress_callback(step + 1, num_steps);
        }

        /* Step image callback - decode and display intermediate result */
        if (iris_step_image_callback && iris_step_image_vae && step + 1 < num_steps) {
            iris_image *img = iris_vae_decode((iris_vae_t *)iris_step_image_vae,
                                              z_curr, 1, h, w);
            if (img) {
                iris_step_image_callback(step + 1, num_steps, img);
                iris_image_free(img);
            }
        }
    }

    free(z_pred);
    iris_transformer_free_mmap_cache_flux(tf);
    return z_curr;
}

/* ========================================================================
 * Latent Noise Initialization
 * ======================================================================== */

/* Generate initial noise for the denoising process. Uses a fixed 112x112
 * noise patch tiled/cropped to the target size, ensuring seed-reproducible
 * results regardless of output resolution. Without this, the same seed
 * would produce different images at different sizes. For targets at or
 * above 112x112 latent dims, noise is generated directly. */
#define NOISE_MAX_LATENT_DIM 112  /* 1792/16 = 112 */

float *iris_init_noise(int batch, int channels, int h, int w, int64_t seed) {
    int target_size = batch * channels * h * w;
    float *noise = (float *)malloc(target_size * sizeof(float));

    if (seed >= 0) {
        iris_rng_seed((uint64_t)seed);
    }

    /* If target is max size or larger, just generate directly */
    if (h >= NOISE_MAX_LATENT_DIM && w >= NOISE_MAX_LATENT_DIM) {
        iris_randn(noise, target_size);
        return noise;
    }

    /* Generate noise at max latent size */
    int max_h = NOISE_MAX_LATENT_DIM;
    int max_w = NOISE_MAX_LATENT_DIM;
    int max_size = batch * channels * max_h * max_w;
    float *max_noise = (float *)malloc(max_size * sizeof(float));
    iris_randn(max_noise, max_size);

    /* Subsample to target size using nearest-neighbor */
    for (int b = 0; b < batch; b++) {
        for (int c = 0; c < channels; c++) {
            for (int ty = 0; ty < h; ty++) {
                for (int tx = 0; tx < w; tx++) {
                    /* Map target position to source position */
                    int sy = ty * max_h / h;
                    int sx = tx * max_w / w;

                    int src_idx = ((b * channels + c) * max_h + sy) * max_w + sx;
                    int dst_idx = ((b * channels + c) * h + ty) * w + tx;
                    noise[dst_idx] = max_noise[src_idx];
                }
            }
        }
    }

    free(max_noise);
    return noise;
}

/* ========================================================================
 * Full Generation Pipeline
 * ======================================================================== */

/* Complete text-to-image pipeline in latent space. Orchestrates:
 * text encoding -> noise initialization -> schedule computation -> Euler
 * sampling -> returns denoised latent (caller handles VAE decode).
 * Routes to the appropriate sampler variant based on model type. */
typedef struct iris_ctx iris_ctx;

/* Forward declaration */
extern iris_ctx *iris_get_ctx(void);

float *iris_generate_latent(void *ctx_ptr,
                            const float *text_emb, int text_seq,
                            int height, int width,
                            int num_steps,
                            int64_t seed,
                            void (*progress_callback)(int step, int total)) {
    /* Compute latent dimensions */
    int latent_h = height / 16;
    int latent_w = width / 16;
    int channels = IRIS_LATENT_CHANNELS;

    /* Initialize noise */
    float *z = iris_init_noise(1, channels, latent_h, latent_w, seed);

    /* Get schedule (4 steps for klein distilled) */
    float *schedule = iris_schedule_linear(num_steps);

    /* Sample (FLUX.2-klein is guidance-distilled, no CFG needed) */
    float *latent = iris_sample_euler_flux(ctx_ptr, NULL,
                                      z, 1, channels, latent_h, latent_w,
                                      text_emb, text_seq,
                                      schedule, num_steps,
                                      progress_callback);

    free(z);
    free(schedule);

    return latent;
}

/* ========================================================================
 * Legacy Progress Callback (for backwards compatibility)
 * ======================================================================== */

/* Legacy callback for step-level progress (called from sampling loop) */
void (*iris_progress_callback)(int, int) = NULL;

void iris_set_progress_callback(void (*callback)(int, int)) {
    iris_progress_callback = callback;
}
