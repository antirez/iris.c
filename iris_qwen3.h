/*
 * Qwen3 Text Encoder for Iris
 *
 * Implements the Qwen3 text encoder that produces embeddings for image
 * generation. Supports multiple model sizes (architecture read from config.json).
 */

#ifndef IRIS_QWEN3_H
#define IRIS_QWEN3_H

#include <stddef.h>
#include <stdint.h>
#include "iris_safetensors.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Architecture Constants
 * ======================================================================== */

/* Fixed constants (same across model sizes) */
#define QWEN3_VOCAB_SIZE       151936
#define QWEN3_MAX_SEQ_LEN      512
#define QWEN3_VL_MAX_SEQ_LEN   1280  /* Flux3 (Qwen3-VL): 16 windows of 80 tokens */
#define QWEN3_QIMAGE_MAX_SEQ_LEN 17280  /* Qwen-Image: 16 references at 1 MP and a prompt */
#define QWEN3_FLUX3_PAD_MULTIPLE 80  /* Flux3 pads the prompt to a multiple of this */
#define QWEN3_RMS_NORM_EPS     1e-6f
#define QWEN3_ROPE_THETA       1000000.0f

/* Maximum number of hidden-state layers concatenated into the embeddings
 * (Flux3 stacks 8 of them). */
#define QWEN3_MAX_OUTPUT_LAYERS 8

/* Extraction modes, see qwen3_set_extraction_mode(). */
#define QWEN3_EXTRACT_FLUX     0
#define QWEN3_EXTRACT_ZIMAGE   1
#define QWEN3_EXTRACT_FLUX3    2
#define QWEN3_EXTRACT_QIMAGE   3

/* Qwen-Image 2.1 system message; its tokens are dropped from the output. */
#define QWEN3_QIMAGE_SYSTEM "Comprehend and analyze the provided prompt."

/* ========================================================================
 * Opaque Types
 * ======================================================================== */

typedef struct qwen3_model qwen3_model_t;
typedef struct qwen3_tokenizer qwen3_tokenizer_t;

#define QWEN3_VISION_DEEPSTACK 3

/* Vision tower output of one image, on its grid of h x w merged blocks. */
typedef struct {
    int h, w;
    const float *embeds;                        /* [h * w, hidden] */
    const float *deep[QWEN3_VISION_DEEPSTACK];  /* [h * w, hidden] each */
} qwen3_image_t;

/* ========================================================================
 * Tokenizer API
 * ======================================================================== */

/*
 * Load tokenizer from HuggingFace tokenizer.json file.
 */
qwen3_tokenizer_t *qwen3_tokenizer_load(const char *tokenizer_json_path);

/*
 * Free tokenizer resources.
 */
void qwen3_tokenizer_free(qwen3_tokenizer_t *tok);

/*
 * Tokenize text with chat template.
 * Format: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
 * skip_think_tags: when non-zero, omit <think>...</think> suffix (used by Z-Image mode)
 */
int *qwen3_tokenize_chat(qwen3_tokenizer_t *tok, const char *prompt,
                         int *num_tokens, int max_len, int skip_think_tags);

/*
 * Tokenize with a system message and no <think> block (Qwen-Image 2.1):
 * <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
 * num_system receives the number of system-block tokens.
 */
int *qwen3_tokenize_chat_system(qwen3_tokenizer_t *tok, const char *system,
                                const char *prompt, int *num_tokens,
                                int *num_system, int max_len);

/*
 * Same template with images in front of the prompt (Qwen-Image references):
 * <|im_start|>user\n<image1><|vision_start|>{pad * slots[0]}<|vision_end|>
 * then " <image2>..." for later images, then the prompt. image_rows[i]
 * receives the first pad row of image i.
 */
int *qwen3_tokenize_chat_images(qwen3_tokenizer_t *tok, const char *system,
                                const char *prompt, const int *slots, int n,
                                int *num_tokens, int *num_system, int *image_rows);

/*
 * Pad tokens to max_len with PAD token.
 * Returns new array, caller must free original.
 */
int *qwen3_pad_tokens(int *tokens, int num_tokens, int max_len, int *attention_mask);

/*
 * Get token string by ID.
 */
const char *qwen3_get_token(qwen3_tokenizer_t *tok, int id);

/*
 * Get token ID by string.
 */
int qwen3_get_id(qwen3_tokenizer_t *tok, const char *token);

/* ========================================================================
 * Model API
 * ======================================================================== */

/*
 * Load Qwen3 model weights from HuggingFace model directory.
 * Directory should contain model-00001-of-00002.safetensors, etc.
 * Also accepts the text model of a Qwen3-VL checkpoint (Flux3), whose
 * tensors live under "model.language_model.".
 */
qwen3_model_t *qwen3_model_load(const char *model_dir);

/*
 * Free model resources.
 */
void qwen3_model_free(qwen3_model_t *model);

/*
 * Run forward pass to generate text embeddings.
 *
 * input_ids: Token IDs [seq_len]
 * attention_mask: Attention mask [seq_len] (1 for real tokens, 0 for padding)
 * seq_len: Length of input sequences
 *
 * Returns: Embedding array [seq_len, text_dim] (caller must free)
 * Concatenates the hidden states selected by the extraction mode.
 */
float *qwen3_forward(qwen3_model_t *model,
                     const int *input_ids,
                     const int *attention_mask,
                     int seq_len);

/* ========================================================================
 * Combined Text Encoder API
 * ======================================================================== */

typedef struct qwen3_encoder {
    qwen3_tokenizer_t *tokenizer;
    qwen3_model_t *model;
} qwen3_encoder_t;

/*
 * Load complete text encoder (tokenizer + model).
 * model_dir should contain text_encoder/ and the tokenizer, either in
 * tokenizer/, (Flux3) inside text_encoder/ or (Qwen-Image) in processor/.
 * use_mmap: if true, use memory-mapped bf16 weights (saves ~8GB, slower inference)
 */
qwen3_encoder_t *qwen3_encoder_load(const char *model_dir, int use_mmap);

/*
 * Set extraction mode (Python hidden_states[] indices):
 *   QWEN3_EXTRACT_FLUX:   hidden_states[9,18,27] concatenated -> [seq, 3*hidden]
 *   QWEN3_EXTRACT_ZIMAGE: hidden_states[-2] only -> [seq, hidden]
 *   QWEN3_EXTRACT_FLUX3:  hidden_states[4,8,...,32] -> [seq, 8*hidden]
 *   QWEN3_EXTRACT_QIMAGE: last layer before the final norm -> [seq, hidden],
 *                         system-block tokens dropped, padding not returned
 * Flux and Z-Image pad prompts to 512 tokens; Flux3 pads to a multiple of
 * 80 tokens and places the padding at RoPE position 1, like the
 * transformers version it was trained with.
 */
void qwen3_set_extraction_mode(qwen3_encoder_t *enc, int mode);

/*
 * Free encoder resources.
 */
void qwen3_encoder_free(qwen3_encoder_t *enc);

/*
 * Encode text prompt to embeddings.
 * Returns: Embedding array [seq_len, text_dim] (caller must free), where
 * seq_len is 512 except in Flux3 mode (see qwen3_encode_text_ex) and
 * text_dim is hidden_size times the number of extracted layers.
 */
float *qwen3_encode_text(qwen3_encoder_t *enc, const char *prompt);

/*
 * Encode text prompt to embeddings, returning token counts.
 * Same as qwen3_encode_text but also returns the number of real
 * (non-padding) tokens in *out_num_tokens and the padded sequence
 * length (rows of the returned array) in *out_seq_len. Both may be NULL.
 * Returns: Embedding array [seq_len, text_dim] (caller must free)
 */
float *qwen3_encode_text_ex(qwen3_encoder_t *enc, const char *prompt,
                              int *out_num_tokens, int *out_seq_len);

/*
 * Bytes of the Qwen-Image encoder's weights in the memory the run is limited
 * by, from the config: the f32 token embeddings, and the layers as bf16 on a
 * GPU, as f32 without mmap, not at all as clean mapped pages. The model
 * directory is the pipeline's.
 */
size_t qwen3_memory(const char *model_dir, int use_mmap);

/*
 * Encoder tokens of a Qwen-Image prompt with n images of slots[i] tokens,
 * from the tokenizer alone (no encoder load). Returns 0 on failure.
 */
int qwen3_count_images(const char *model_dir, const char *prompt, const int *slots, int n);

/*
 * Encode a Qwen-Image prompt with n reference images, in
 * QWEN3_EXTRACT_QIMAGE mode. Returns the text rows [*out_seq, hidden] without
 * the image slots (caller frees); pos[i] receives the number of text rows
 * before image i. Returns NULL above the encoder's token limit.
 */
float *qwen3_encode_images(qwen3_encoder_t *enc, const char *prompt,
                           const qwen3_image_t *images, int n,
                           int *out_seq, int *pos);

/*
 * Open the safetensors shards of model_dir (index, or a single file).
 * Returns the number of files opened, 0 on failure.
 */
int open_safetensors_shards(const char *model_dir,
                            safetensors_file_t **files, int max_files);

/* ========================================================================
 * Vision Tower (Qwen3-VL, iris_qwen3_vision.c)
 * ======================================================================== */

typedef struct qwen3_vision qwen3_vision_t;

/*
 * Open the model.visual.* weights of a text_encoder/ directory (shards or a
 * single file). The weights stay mmapped; GPU builds upload them on the
 * first image.
 */
qwen3_vision_t *qwen3_vision_load(const char *text_encoder_dir);

void qwen3_vision_free(qwen3_vision_t *vision);

/*
 * Encode one image. rgb: [3, h, w], values 0..255; h and w multiples of 32.
 * *embeds receives the main embeddings and deep[k] the deepstack features,
 * each [tokens, 4096] in raster order of the merged blocks (caller frees).
 * Returns tokens = (h / 32) * (w / 32), or 0 on failure.
 */
int qwen3_vision_encode(qwen3_vision_t *vision, const float *rgb, int h, int w,
                        float **embeds, float *deep[QWEN3_VISION_DEEPSTACK]);

#ifdef __cplusplus
}
#endif

#endif /* IRIS_QWEN3_H */
