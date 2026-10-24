/* ========================================================================
 * Embedding Cache with 4-bit Quantization
 *
 * Implements a simple single-entry cache for CLI prompt embeddings.
 * Uses 4-bit block quantization to reduce memory from 15.7 MB to ~2 MB.
 * ======================================================================== */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "embcache.h"

/* ========================================================================
 * Hash Function (FNV-1a)
 * ======================================================================== */

static uint64_t hash_string(const char *str) {
    uint64_t hash = 14695981039346656037ULL;  /* FNV offset basis */
    while (*str) {
        hash ^= (uint8_t)*str++;
        hash *= 1099511628211ULL;  /* FNV prime */
    }
    return hash;
}

/* ========================================================================
 * 4-bit Quantization
 * ======================================================================== */

/* Quantize f32 embeddings to 4-bit with block-wise min/max scaling.
 * Each block of 32 values is mapped to [0, 15] using per-block scale and
 * min offset. Gives ~8x compression for caching text embeddings across
 * repeated prompts, with minimal quality loss since the diffusion model
 * is robust to small embedding perturbations. */
emb_quantized_t *emb_quantize_4bit(const float *data, int num_elements) {
    if (!data || num_elements <= 0) return NULL;

    emb_quantized_t *q = calloc(1, sizeof(emb_quantized_t));
    if (!q) return NULL;

    q->num_elements = num_elements;
    q->num_blocks = (num_elements + EMB_QUANT_BLOCK_SIZE - 1) / EMB_QUANT_BLOCK_SIZE;

    /* Allocate storage */
    size_t packed_size = (num_elements + 1) / 2;  /* 2 values per byte */
    q->data = calloc(packed_size, 1);
    q->scales = malloc(q->num_blocks * sizeof(float));
    q->offsets = malloc(q->num_blocks * sizeof(float));

    if (!q->data || !q->scales || !q->offsets) {
        emb_quantized_free(q);
        return NULL;
    }

    /* Quantize each block */
    for (int b = 0; b < q->num_blocks; b++) {
        int start = b * EMB_QUANT_BLOCK_SIZE;
        int end = start + EMB_QUANT_BLOCK_SIZE;
        if (end > num_elements) end = num_elements;

        /* Find min/max in block */
        float min_val = data[start];
        float max_val = data[start];
        for (int i = start + 1; i < end; i++) {
            if (data[i] < min_val) min_val = data[i];
            if (data[i] > max_val) max_val = data[i];
        }

        /* Store scale and offset */
        float range = max_val - min_val;
        if (range < 1e-10f) range = 1e-10f;  /* Avoid division by zero */
        q->scales[b] = range;
        q->offsets[b] = min_val;

        /* Quantize values to 4-bit (0-15) */
        float inv_scale = 15.0f / range;
        for (int i = start; i < end; i++) {
            float normalized = (data[i] - min_val) * inv_scale;
            int quantized = (int)(normalized + 0.5f);  /* Round */
            if (quantized < 0) quantized = 0;
            if (quantized > 15) quantized = 15;

            /* Pack into bytes (2 values per byte) */
            int byte_idx = i / 2;
            if (i % 2 == 0) {
                q->data[byte_idx] = (q->data[byte_idx] & 0xF0) | (quantized & 0x0F);
            } else {
                q->data[byte_idx] = (q->data[byte_idx] & 0x0F) | ((quantized & 0x0F) << 4);
            }
        }
    }

    return q;
}

float *emb_dequantize_4bit(const emb_quantized_t *q) {
    if (!q || !q->data) return NULL;

    float *data = malloc(q->num_elements * sizeof(float));
    if (!data) return NULL;

    /* Dequantize each block */
    for (int b = 0; b < q->num_blocks; b++) {
        int start = b * EMB_QUANT_BLOCK_SIZE;
        int end = start + EMB_QUANT_BLOCK_SIZE;
        if (end > q->num_elements) end = q->num_elements;

        float scale = q->scales[b] / 15.0f;
        float offset = q->offsets[b];

        for (int i = start; i < end; i++) {
            /* Unpack from bytes */
            int byte_idx = i / 2;
            int quantized;
            if (i % 2 == 0) {
                quantized = q->data[byte_idx] & 0x0F;
            } else {
                quantized = (q->data[byte_idx] >> 4) & 0x0F;
            }

            /* Dequantize */
            data[i] = quantized * scale + offset;
        }
    }

    return data;
}

void emb_quantized_free(emb_quantized_t *q) {
    if (!q) return;
    free(q->data);
    free(q->scales);
    free(q->offsets);
    free(q);
}

/* ========================================================================
 * Multi-Slot Cache
 *
 * Uses EMB_CACHE_SLOTS independent slots indexed by hash(prompt) % SLOTS.
 * Multiple slots allow caching both real prompt and empty prompt (for CFG),
 * plus handles prompt iteration workflows without evicting either entry.
 * ======================================================================== */

#define EMB_CACHE_SLOTS 4

static emb_cache_entry_t g_cache[EMB_CACHE_SLOTS];
static int g_cache_initialized = 0;

static void clear_slot(int slot) {
    free(g_cache[slot].prompt);
    emb_quantized_free(g_cache[slot].emb);
    memset(&g_cache[slot], 0, sizeof(g_cache[slot]));
}

void emb_cache_init(void) {
    if (g_cache_initialized) return;
    memset(g_cache, 0, sizeof(g_cache));
    g_cache_initialized = 1;
}

/* Store quantized text embeddings keyed by prompt string. Each prompt hashes
 * to a fixed slot; only that slot is evicted when a new prompt maps to it. */
void emb_cache_store(const char *prompt, const float *embedding, int num_elements) {
    if (!prompt || !embedding || num_elements <= 0) return;
    if (!g_cache_initialized) emb_cache_init();

    uint64_t hash = hash_string(prompt);
    int slot = (int)(hash % EMB_CACHE_SLOTS);

    clear_slot(slot);

    g_cache[slot].prompt = strdup(prompt);
    g_cache[slot].hash = hash;
    g_cache[slot].emb = emb_quantize_4bit(embedding, num_elements);

    if (!g_cache[slot].prompt || !g_cache[slot].emb) {
        clear_slot(slot);
    }
}

float *emb_cache_lookup_ex(const char *prompt, int *num_elements) {
    if (!prompt || !g_cache_initialized) return NULL;

    uint64_t hash = hash_string(prompt);
    int slot = (int)(hash % EMB_CACHE_SLOTS);

    if (!g_cache[slot].prompt || !g_cache[slot].emb) return NULL;
    if (hash != g_cache[slot].hash) return NULL;
    if (strcmp(prompt, g_cache[slot].prompt) != 0) return NULL;

    if (num_elements) *num_elements = g_cache[slot].emb->num_elements;
    return emb_dequantize_4bit(g_cache[slot].emb);
}

float *emb_cache_lookup(const char *prompt) {
    return emb_cache_lookup_ex(prompt, NULL);
}

int emb_cache_has(const char *prompt) {
    if (!prompt || !g_cache_initialized) return 0;

    uint64_t hash = hash_string(prompt);
    int slot = (int)(hash % EMB_CACHE_SLOTS);

    if (!g_cache[slot].prompt) return 0;
    if (hash != g_cache[slot].hash) return 0;
    return strcmp(prompt, g_cache[slot].prompt) == 0;
}

void emb_cache_clear(void) {
    if (!g_cache_initialized) return;
    for (int i = 0; i < EMB_CACHE_SLOTS; i++)
        clear_slot(i);
}

void emb_cache_free(void) {
    emb_cache_clear();
    g_cache_initialized = 0;
}

void emb_cache_stats(int *num_entries, size_t *memory_used) {
    int entries = 0;
    size_t mem = 0;

    if (g_cache_initialized) {
        for (int i = 0; i < EMB_CACHE_SLOTS; i++) {
            if (g_cache[i].emb) {
                entries++;
                size_t packed_size = (g_cache[i].emb->num_elements + 1) / 2;
                mem += packed_size;
                mem += g_cache[i].emb->num_blocks * sizeof(float) * 2;
                if (g_cache[i].prompt)
                    mem += strlen(g_cache[i].prompt) + 1;
            }
        }
    }

    if (num_entries) *num_entries = entries;
    if (memory_used) *memory_used = mem;
}
