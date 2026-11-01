/*
 * Qwen3 Tokenizer Implementation
 *
 * BPE (Byte Pair Encoding) tokenizer for Qwen3 text encoder.
 * Loads directly from HuggingFace tokenizer.json format.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <limits.h>
#include "iris_kernels.h"
#include "iris_nfc_table.h"

/* ========================================================================
 * Configuration
 * ======================================================================== */

#define QWEN3_VOCAB_SIZE 151936
#define QWEN3_MAX_TOKEN_LEN 256
#define QWEN3_MAX_SEQ_LEN 512
#define QWEN3_HASH_SIZE 300007  /* Prime > 2 * vocab_size */

/* Special token IDs */
#define QWEN3_PAD_ID 151643      /* <|endoftext|> */
#define QWEN3_IM_START_ID 151644 /* <|im_start|> */
#define QWEN3_VISION_START_ID 151652 /* <|vision_start|> */
#define QWEN3_VISION_END_ID 151653   /* <|vision_end|> */
#define QWEN3_IMAGE_PAD_ID 151655    /* <|image_pad|> */
#define QWEN3_IM_END_ID 151645   /* <|im_end|> */
#define QWEN3_THINK_START_ID 151667 /* <think> */
#define QWEN3_THINK_END_ID 151668   /* </think> */

/* ========================================================================
 * Data Structures
 * ======================================================================== */

typedef struct {
    char *token;
    int id;
} vocab_entry_t;

typedef struct {
    char *left;
    char *right;
    int rank;  /* Lower rank = higher priority (merge first) */
} bpe_merge_t;

typedef struct qwen3_tokenizer {
    /* Vocabulary: id -> token string */
    char **vocab;
    int vocab_size;

    /* Hash table: token string -> id */
    vocab_entry_t *vocab_hash;
    int hash_size;

    /* BPE merges */
    bpe_merge_t *merges;
    int num_merges;

    /* Merge rank lookup: "left right" -> rank */
    int *merge_ranks;  /* Hash table: hash("left right") -> rank, or -1 */

    /* added_tokens ids: matched verbatim in the text, before anything else */
    int *added_ids;
    int num_added;
} qwen3_tokenizer_t;

/* ========================================================================
 * Byte-Level BPE Encoding Table
 * ======================================================================== */

/*
 * GPT-2/Qwen style byte-to-unicode mapping.
 * Bytes 33-126 and 161-172 and 174-255 map to themselves.
 * Other bytes (0-32, 127-160, 173) map to 256+i for uniqueness.
 */
static int byte_to_unicode[256];
static int unicode_to_byte[512];
static int byte_encoder_initialized = 0;

static void init_byte_encoder(void) {
    if (byte_encoder_initialized) return;

    /* Printable ASCII and extended Latin */
    for (int i = 33; i <= 126; i++) {
        byte_to_unicode[i] = i;
        unicode_to_byte[i] = i;
    }
    for (int i = 161; i <= 172; i++) {
        byte_to_unicode[i] = i;
        unicode_to_byte[i] = i;
    }
    for (int i = 174; i <= 255; i++) {
        byte_to_unicode[i] = i;
        unicode_to_byte[i] = i;
    }

    /* Map remaining bytes to 256+ range */
    int offset = 256;
    for (int i = 0; i < 256; i++) {
        if (byte_to_unicode[i] == 0 && i != 33) {  /* 33 maps to itself */
            byte_to_unicode[i] = offset;
            unicode_to_byte[offset] = i;
            offset++;
        }
    }
    /* Fix: byte 0 should also be mapped */
    byte_to_unicode[0] = 256;
    unicode_to_byte[256] = 0;

    byte_encoder_initialized = 1;
}

/* Encode a byte to its unicode character (UTF-8) */
static int encode_byte_to_utf8(unsigned char b, char *out) {
    init_byte_encoder();
    int cp = byte_to_unicode[b];
    if (cp < 128) {
        out[0] = (char)cp;
        return 1;
    } else if (cp < 2048) {
        out[0] = (char)(0xC0 | (cp >> 6));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    /* Shouldn't reach here for byte-level BPE */
    out[0] = '?';
    return 1;
}


/* ========================================================================
 * Hash Functions
 * ======================================================================== */

static unsigned int hash_string(const char *str) {
    unsigned int hash = 2166136261u;
    while (*str) {
        hash ^= (unsigned char)*str++;
        hash *= 16777619u;
    }
    return hash;
}

static void vocab_hash_insert(vocab_entry_t *table, int hash_size,
                              const char *token, int id) {
    unsigned int h = hash_string(token) % hash_size;
    int probes = 0;
    while (table[h].token != NULL && probes < hash_size) {
        if (strcmp(table[h].token, token) == 0) {
            return;  /* Already exists */
        }
        h = (h + 1) % hash_size;
        probes++;
    }
    if (probes < hash_size) {
        table[h].token = strdup(token);
        table[h].id = id;
    }
}

static int vocab_hash_lookup(const vocab_entry_t *table, int hash_size,
                             const char *token) {
    unsigned int h = hash_string(token) % hash_size;
    int probes = 0;
    while (table[h].token != NULL && probes < hash_size) {
        if (strcmp(table[h].token, token) == 0) {
            return table[h].id;
        }
        h = (h + 1) % hash_size;
        probes++;
    }
    return -1;
}

/* ========================================================================
 * JSON Parsing Helpers (minimal, just for tokenizer.json)
 * ======================================================================== */

/* Skip whitespace */
static const char *skip_ws(const char *p) {
    while (*p && (*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')) p++;
    return p;
}

/* Parse a JSON string, returns allocated string, advances *p past closing quote */
static char *parse_json_string(const char **pp) {
    const char *p = *pp;
    if (*p != '"') return NULL;
    p++;

    /* Find string length first (handling escapes) */
    const char *start = p;
    int len = 0;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) {
            p += 2;
            len++;
        } else {
            p++;
            len++;
        }
    }

    /* Allocate and copy */
    char *result = malloc(len + 1);
    if (!result) return NULL;

    p = start;
    int i = 0;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1]) {
            p++;
            switch (*p) {
                case 'n': result[i++] = '\n'; break;
                case 'r': result[i++] = '\r'; break;
                case 't': result[i++] = '\t'; break;
                case '\\': result[i++] = '\\'; break;
                case '"': result[i++] = '"'; break;
                case 'u': {
                    /* Parse \uXXXX */
                    if (p[1] && p[2] && p[3] && p[4]) {
                        char hex[5] = {p[1], p[2], p[3], p[4], 0};
                        int cp = (int)strtol(hex, NULL, 16);
                        p += 4;
                        /* Encode as UTF-8 */
                        if (cp < 0x80) {
                            result[i++] = (char)cp;
                        } else if (cp < 0x800) {
                            result[i++] = (char)(0xC0 | (cp >> 6));
                            result[i++] = (char)(0x80 | (cp & 0x3F));
                            len++;  /* Need more space */
                        } else {
                            result[i++] = (char)(0xE0 | (cp >> 12));
                            result[i++] = (char)(0x80 | ((cp >> 6) & 0x3F));
                            result[i++] = (char)(0x80 | (cp & 0x3F));
                            len += 2;
                        }
                    }
                    break;
                }
                default: result[i++] = *p; break;
            }
            p++;
        } else {
            result[i++] = *p++;
        }
    }
    result[i] = '\0';

    if (*p == '"') p++;
    *pp = p;
    return result;
}

/* Parse a JSON integer */
static int parse_json_int(const char **pp) {
    const char *p = *pp;
    int neg = 0;
    if (*p == '-') { neg = 1; p++; }
    int val = 0;
    while (*p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    *pp = p;
    return neg ? -val : val;
}

/* Skip a JSON value (string, number, object, array, bool, null) */
static const char *skip_json_value(const char *p) {
    p = skip_ws(p);
    if (*p == '"') {
        p++;
        while (*p && *p != '"') {
            if (*p == '\\' && p[1]) p += 2;
            else p++;
        }
        if (*p == '"') p++;
    } else if (*p == '{') {
        int depth = 1;
        p++;
        while (*p && depth > 0) {
            if (*p == '{') depth++;
            else if (*p == '}') depth--;
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\' && p[1]) p += 2;
                    else p++;
                }
            }
            p++;
        }
    } else if (*p == '[') {
        int depth = 1;
        p++;
        while (*p && depth > 0) {
            if (*p == '[') depth++;
            else if (*p == ']') depth--;
            else if (*p == '"') {
                p++;
                while (*p && *p != '"') {
                    if (*p == '\\' && p[1]) p += 2;
                    else p++;
                }
            }
            p++;
        }
    } else {
        /* number, bool, null */
        while (*p && *p != ',' && *p != '}' && *p != ']' &&
               *p != ' ' && *p != '\n' && *p != '\r' && *p != '\t') p++;
    }
    return p;
}

/* ========================================================================
 * Tokenizer Loading
 * ======================================================================== */

qwen3_tokenizer_t *qwen3_tokenizer_load(const char *tokenizer_json_path) {
    /* Read file */
    FILE *f = fopen(tokenizer_json_path, "rb");
    if (!f) {
        fprintf(stderr, "qwen3_tokenizer_load: cannot open %s\n", tokenizer_json_path);
        return NULL;
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    char *json = malloc(size + 1);
    if (!json) {
        fclose(f);
        return NULL;
    }
    fread(json, 1, size, f);
    json[size] = '\0';
    fclose(f);

    qwen3_tokenizer_t *tok = calloc(1, sizeof(qwen3_tokenizer_t));
    if (!tok) {
        free(json);
        return NULL;
    }

    tok->hash_size = QWEN3_HASH_SIZE;
    tok->vocab_hash = calloc(tok->hash_size, sizeof(vocab_entry_t));
    if (!tok->vocab_hash) {
        free(tok);
        free(json);
        return NULL;
    }

    /* Parse vocabulary from "model": { "vocab": { ... } } */
    const char *p = strstr(json, "\"model\"");
    if (!p) {
        fprintf(stderr, "qwen3_tokenizer_load: no model section\n");
        goto error;
    }

    p = strstr(p, "\"vocab\"");
    if (!p) {
        fprintf(stderr, "qwen3_tokenizer_load: no vocab section\n");
        goto error;
    }

    /* Skip to opening brace */
    p = strchr(p, '{');
    if (!p) goto error;
    p++;

    /* Count vocab entries first */
    int vocab_count = 0;
    const char *count_p = p;
    int depth = 1;
    while (*count_p && depth > 0) {
        if (*count_p == '{') depth++;
        else if (*count_p == '}') depth--;
        else if (*count_p == '"' && depth == 1) {
            vocab_count++;
            /* Skip the string */
            count_p++;
            while (*count_p && *count_p != '"') {
                if (*count_p == '\\' && count_p[1]) count_p += 2;
                else count_p++;
            }
        }
        count_p++;
    }
    /* vocab_count is the number of string keys (values are integers, not strings) */

    tok->vocab_size = vocab_count;
    tok->vocab = calloc(vocab_count + 1000, sizeof(char *));  /* Extra for added_tokens */
    if (!tok->vocab) goto error;

    /* Parse vocab entries */
    p = skip_ws(p);
    int max_id = 0;
    while (*p && *p != '}') {
        if (*p == '"') {
            char *token = parse_json_string(&p);
            p = skip_ws(p);
            if (*p == ':') p++;
            p = skip_ws(p);
            int id = parse_json_int(&p);

            if (token && id >= 0 && id < vocab_count + 1000) {
                tok->vocab[id] = token;
                vocab_hash_insert(tok->vocab_hash, tok->hash_size, token, id);
                if (id > max_id) max_id = id;
            } else {
                free(token);
            }

            p = skip_ws(p);
            if (*p == ',') p++;
            p = skip_ws(p);
        } else {
            p++;
        }
    }

    /* Parse merges from "model": { "merges": [ ... ] }
     * Merges are pairs like [["Ġ", "Ġ"], ["ĠĠ", "ĠĠ"], ...] or, in the
     * older format (Qwen3-VL), strings like ["Ġ Ġ", "ĠĠ ĠĠ", ...].
     */
    p = strstr(json, "\"merges\"");
    if (!p) {
        fprintf(stderr, "qwen3_tokenizer_load: no merges section\n");
        goto error;
    }

    p = strchr(p, '[');
    if (!p) goto error;
    p++;

    /* Count merges by counting the '[' or '"' elements at depth 1 */
    int merge_count = 0;
    count_p = p;
    depth = 1;
    while (*count_p && depth > 0) {
        if (*count_p == '[') {
            if (depth == 1) merge_count++;
            depth++;
        } else if (*count_p == ']') {
            depth--;
        } else if (*count_p == '"') {
            if (depth == 1) merge_count++;
            count_p++;
            while (*count_p && *count_p != '"') {
                if (*count_p == '\\' && count_p[1]) count_p += 2;
                else count_p++;
            }
        }
        if (*count_p) count_p++;
    }

    tok->num_merges = merge_count;
    tok->merges = calloc(merge_count, sizeof(bpe_merge_t));
    tok->merge_ranks = calloc(tok->hash_size, sizeof(int));
    if (!tok->merges || !tok->merge_ranks) goto error;

    /* Initialize merge_ranks to -1 */
    for (int i = 0; i < tok->hash_size; i++) {
        tok->merge_ranks[i] = -1;
    }

    /* Parse merges */
    p = skip_ws(p);
    int merge_idx = 0;
    while (*p && *p != ']' && merge_idx < merge_count) {
        if (*p == '[' || *p == '"') {
            char *left = NULL;
            char *right = NULL;
            if (*p == '"') {
                /* "left right": tokens never contain a literal space */
                char *pair = parse_json_string(&p);
                char *sp = pair ? strchr(pair, ' ') : NULL;
                if (sp) {
                    *sp = '\0';
                    left = strdup(pair);
                    right = strdup(sp + 1);
                }
                free(pair);
            } else {
                p++;
                p = skip_ws(p);

                /* Parse left string */
                if (*p == '"') {
                    left = parse_json_string(&p);
                }
                p = skip_ws(p);
                if (*p == ',') p++;
                p = skip_ws(p);

                /* Parse right string */
                if (*p == '"') {
                    right = parse_json_string(&p);
                }

                /* Skip to closing ] */
                while (*p && *p != ']') p++;
                if (*p == ']') p++;
            }

            if (left && right) {
                tok->merges[merge_idx].left = left;
                tok->merges[merge_idx].right = right;
                tok->merges[merge_idx].rank = merge_idx;

                /* Add to merge rank lookup: hash "left right" */
                int len1 = strlen(left);
                int len2 = strlen(right);
                char *key = malloc(len1 + len2 + 2);
                memcpy(key, left, len1);
                key[len1] = ' ';
                memcpy(key + len1 + 1, right, len2);
                key[len1 + len2 + 1] = '\0';

                unsigned int h = hash_string(key) % tok->hash_size;
                int probes = 0;
                while (tok->merge_ranks[h] != -1 && probes < tok->hash_size) {
                    h = (h + 1) % tok->hash_size;
                    probes++;
                }
                if (probes < tok->hash_size) {
                    tok->merge_ranks[h] = merge_idx;
                }
                free(key);
            } else {
                free(left);
                free(right);
            }

            merge_idx++;

            p = skip_ws(p);
            if (*p == ',') p++;
            p = skip_ws(p);
        } else {
            p++;
        }
    }

    /* Parse added_tokens for special tokens */
    p = strstr(json, "\"added_tokens\"");
    if (p) {
        p = strchr(p, '[');
        if (p) {
            p++;
            while (*p && *p != ']') {
                if (*p == '{') {
                    /* Parse added token object */
                    p++;
                    char *content = NULL;
                    int id = -1;

                    while (*p && *p != '}') {
                        p = skip_ws(p);
                        if (*p == '"') {
                            char *key = parse_json_string(&p);
                            p = skip_ws(p);
                            if (*p == ':') p++;
                            p = skip_ws(p);

                            if (key && strcmp(key, "content") == 0 && *p == '"') {
                                content = parse_json_string(&p);
                            } else if (key && strcmp(key, "id") == 0) {
                                id = parse_json_int(&p);
                            } else {
                                p = skip_json_value(p);
                            }
                            free(key);
                        }
                        p = skip_ws(p);
                        if (*p == ',') p++;
                    }

                    if (content && id >= 0) {
                        if (id < vocab_count + 1000) {
                            if (!tok->vocab[id]) {
                                tok->vocab[id] = content;
                                vocab_hash_insert(tok->vocab_hash, tok->hash_size, content, id);
                                if (id > max_id) max_id = id;
                                content = NULL;  /* Don't free */
                            }
                            tok->added_ids = realloc(tok->added_ids,
                                                     (tok->num_added + 1) * sizeof(int));
                            tok->added_ids[tok->num_added++] = id;
                        }
                    }
                    free(content);

                    if (*p == '}') p++;
                }
                p = skip_ws(p);
                if (*p == ',') p++;
            }
        }
    }

    tok->vocab_size = max_id + 1;

    free(json);

    if (iris_verbose)
        fprintf(stderr, " Qwen3 tokenizer loaded (%d vocab)\n",
                tok->vocab_size);

    return tok;

error:
    free(json);
    if (tok) {
        free(tok->vocab_hash);
        free(tok->vocab);
        free(tok->merges);
        free(tok->merge_ranks);
        free(tok);
    }
    return NULL;
}

void qwen3_tokenizer_free(qwen3_tokenizer_t *tok) {
    if (!tok) return;

    if (tok->vocab) {
        for (int i = 0; i < tok->vocab_size; i++) {
            free(tok->vocab[i]);
        }
        free(tok->vocab);
    }

    if (tok->vocab_hash) {
        for (int i = 0; i < tok->hash_size; i++) {
            free(tok->vocab_hash[i].token);
        }
        free(tok->vocab_hash);
    }

    if (tok->merges) {
        for (int i = 0; i < tok->num_merges; i++) {
            free(tok->merges[i].left);
            free(tok->merges[i].right);
        }
        free(tok->merges);
    }

    free(tok->merge_ranks);
    free(tok->added_ids);
    free(tok);
}

/* ========================================================================
 * Merge Rank Lookup
 * ======================================================================== */

static int get_merge_rank(qwen3_tokenizer_t *tok, const char *left, const char *right) {
    /* Build "left right" string */
    int len1 = strlen(left);
    int len2 = strlen(right);
    char *key = malloc(len1 + len2 + 2);
    if (!key) return -1;
    memcpy(key, left, len1);
    key[len1] = ' ';
    memcpy(key + len1 + 1, right, len2);
    key[len1 + len2 + 1] = '\0';

    /* Lookup in hash table */
    unsigned int h = hash_string(key) % tok->hash_size;
    int probes = 0;
    while (tok->merge_ranks[h] != -1 && probes < tok->hash_size) {
        int rank = tok->merge_ranks[h];
        if (rank >= 0 && rank < tok->num_merges) {
            /* Check if this is the right merge */
            if (strcmp(tok->merges[rank].left, left) == 0 &&
                strcmp(tok->merges[rank].right, right) == 0) {
                free(key);
                return rank;
            }
        }
        h = (h + 1) % tok->hash_size;
        probes++;
    }

    free(key);
    return -1;
}

/* ========================================================================
 * BPE Tokenization
 * ======================================================================== */

/* Token list node for BPE */
typedef struct token_node {
    char *text;
    struct token_node *next;
} token_node_t;

static token_node_t *create_node(const char *text) {
    token_node_t *node = malloc(sizeof(token_node_t));
    if (node) {
        node->text = strdup(text);
        node->next = NULL;
    }
    return node;
}

static void free_token_list(token_node_t *head) {
    while (head) {
        token_node_t *next = head->next;
        free(head->text);
        free(head);
        head = next;
    }
}

/* Core BPE (Byte-Pair Encoding) algorithm. Starts with character-level
 * tokens, then iteratively merges the adjacent pair with the lowest rank
 * (highest priority) until no more merges apply. The merge ranks come from
 * the tokenizer's trained vocabulary. This produces the subword tokenization
 * that Qwen3 expects as input. */
static token_node_t *bpe_encode_word(qwen3_tokenizer_t *tok, const char *word) {
    int len = strlen(word);
    if (len == 0) return NULL;

    /* Start with character-level tokens */
    token_node_t *head = NULL;
    token_node_t *tail = NULL;

    const char *p = word;
    while (*p) {
        /* Get one UTF-8 character */
        int char_len = 1;
        unsigned char c = (unsigned char)*p;
        if ((c & 0xE0) == 0xC0) char_len = 2;
        else if ((c & 0xF0) == 0xE0) char_len = 3;
        else if ((c & 0xF8) == 0xF0) char_len = 4;

        char buf[8];
        memcpy(buf, p, char_len);
        buf[char_len] = '\0';

        token_node_t *node = create_node(buf);
        if (!head) head = node;
        else tail->next = node;
        tail = node;

        p += char_len;
    }

    /* Apply BPE merges */
    int changed = 1;
    while (changed) {
        changed = 0;

        /* Find best merge (lowest rank) */
        int best_rank = tok->num_merges + 1;
        token_node_t *best_node = NULL;

        for (token_node_t *node = head; node && node->next; node = node->next) {
            int rank = get_merge_rank(tok, node->text, node->next->text);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_node = node;
            }
        }

        /* Apply best merge */
        if (best_node) {
            /* Merge best_node and best_node->next */
            int len1 = strlen(best_node->text);
            int len2 = strlen(best_node->next->text);
            char *merged = malloc(len1 + len2 + 1);
            memcpy(merged, best_node->text, len1);
            memcpy(merged + len1, best_node->next->text, len2);
            merged[len1 + len2] = '\0';

            free(best_node->text);
            best_node->text = merged;

            token_node_t *to_free = best_node->next;
            best_node->next = to_free->next;
            free(to_free->text);
            free(to_free);

            changed = 1;
        }
    }

    return head;
}

/* Convert text to byte-level encoding */
static char *text_to_bytes(const char *text) {
    init_byte_encoder();
    int len = strlen(text);

    /* Allocate worst case: each byte could become 2 UTF-8 bytes */
    char *result = malloc(len * 2 + 1);
    if (!result) return NULL;

    int j = 0;
    for (int i = 0; i < len; i++) {
        j += encode_byte_to_utf8((unsigned char)text[i], result + j);
    }
    result[j] = '\0';

    return result;
}

/* ========================================================================
 * Unicode normalization (NFC), the tokenizer's normalizer
 * ======================================================================== */

/* Decode the UTF-8 character at s, storing its length (1 for invalid bytes). */
static uint32_t utf8_decode(const char *s, int *len) {
    const unsigned char *u = (const unsigned char *)s;
    int n = u[0] < 0x80 ? 1 : (u[0] & 0xE0) == 0xC0 ? 2 : (u[0] & 0xF0) == 0xE0 ? 3 :
            (u[0] & 0xF8) == 0xF0 ? 4 : 1;
    uint32_t c = n == 1 ? u[0] : u[0] & (0x7F >> n);
    for (int i = 1; i < n; i++) {
        if ((u[i] & 0xC0) != 0x80) { *len = 1; return u[0]; }
        c = (c << 6) | (u[i] & 0x3F);
    }
    *len = n;
    return c;
}

static int utf8_encode(uint32_t c, char *out) {
    if (c < 0x80) {
        out[0] = (char)c;
        return 1;
    }
    int n = c < 0x800 ? 2 : c < 0x10000 ? 3 : 4;
    for (int i = n - 1; i > 0; i--, c >>= 6) out[i] = (char)(0x80 | (c & 0x3F));
    out[0] = (char)(((0xFF00 >> n) & 0xFF) | c);
    return n;
}

static int nfc_combining_class(uint32_t c) {
    int lo = 0, hi = (int)(sizeof(nfc_class) / sizeof(nfc_class[0])) - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (c < nfc_class[mid].first) hi = mid - 1;
        else if (c > nfc_class[mid].last) lo = mid + 1;
        else return nfc_class[mid].ccc;
    }
    return 0;
}

/* Hangul syllables are algorithmic: S = S0 + (L * NV + V) * NT + T. */
#define NFC_S0 0xAC00
#define NFC_L0 0x1100
#define NFC_V0 0x1161
#define NFC_T0 0x11A7
#define NFC_NV 21
#define NFC_NT 28
#define NFC_NS (19 * NFC_NV * NFC_NT)

/* Append the full canonical decomposition of c to out[n], return the new n. */
static int nfc_decompose(uint32_t c, uint32_t *out, int n) {
    if (c - NFC_S0 < NFC_NS) {
        uint32_t s = c - NFC_S0;
        out[n++] = NFC_L0 + s / (NFC_NV * NFC_NT);
        out[n++] = NFC_V0 + s / NFC_NT % NFC_NV;
        if (s % NFC_NT) out[n++] = NFC_T0 + s % NFC_NT;
        return n;
    }
    int lo = 0, hi = (int)(sizeof(nfc_decomp) / sizeof(nfc_decomp[0])) - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (c < nfc_decomp[mid].cp) hi = mid - 1;
        else if (c > nfc_decomp[mid].cp) lo = mid + 1;
        else {
            n = nfc_decompose(nfc_decomp[mid].first, out, n);
            return nfc_decomp[mid].second ? nfc_decompose(nfc_decomp[mid].second, out, n) : n;
        }
    }
    out[n++] = c;
    return n;
}

/* Primary composite of a + b, 0 if none. No second character is below U+0300,
 * which keeps plain text from scanning the table. */
static uint32_t nfc_compose(uint32_t a, uint32_t b) {
    if (a - NFC_L0 < 19 && b - NFC_V0 < NFC_NV)
        return NFC_S0 + ((a - NFC_L0) * NFC_NV + (b - NFC_V0)) * NFC_NT;
    if (a - NFC_S0 < NFC_NS && (a - NFC_S0) % NFC_NT == 0 && b - NFC_T0 - 1 < NFC_NT - 1)
        return a + (b - NFC_T0);
    if (b < 0x300) return 0;
    for (size_t i = 0; i < sizeof(nfc_decomp) / sizeof(nfc_decomp[0]); i++)
        if (nfc_decomp[i].composes && nfc_decomp[i].first == a && nfc_decomp[i].second == b)
            return nfc_decomp[i].cp;
    return 0;
}

/* NFC of the first len bytes of text. Caller frees. */
static char *nfc_normalize(const char *text, size_t len) {
    /* One byte per character at worst, each decomposing to at most 4. */
    uint32_t *cp = malloc((len + 1) * 4 * sizeof(uint32_t));
    char *out = malloc(len * 4 + 1);
    int n = 0, step;
    for (size_t i = 0; i < len; i += step) n = nfc_decompose(utf8_decode(text + i, &step), cp, n);

    /* Canonical order: sort each run of combining marks by class. */
    for (int i = 1; i < n; i++) {
        uint32_t c = cp[i];
        int k = nfc_combining_class(c), j = i;
        for (; k && j > 0 && nfc_combining_class(cp[j - 1]) > k; j--) cp[j] = cp[j - 1];
        cp[j] = c;
    }

    /* Compose each character into the last starter unless a character in
     * between has a class of 0 or at least its own. */
    int kept = 0, starter = -1, last = -1;
    for (int i = 0; i < n; i++) {
        int k = nfc_combining_class(cp[i]);
        uint32_t c = starter >= 0 && (last < 0 || last < k) ? nfc_compose(cp[starter], cp[i]) : 0;
        if (c) {
            cp[starter] = c;
            continue;
        }
        if (k == 0) starter = kept;
        last = k ? k : -1;
        cp[kept++] = cp[i];
    }

    size_t bytes = 0;
    for (int i = 0; i < kept; i++) bytes += utf8_encode(cp[i], out + bytes);
    out[bytes] = '\0';
    free(cp);
    return out;
}

/* ========================================================================
 * Pre-tokenization (Qwen2 regex split)
 * ======================================================================== */

/* \p{N}: exact for ASCII and Latin-1 (superscripts, fractions). */
static int cp_is_number(uint32_t c) {
    return (c >= '0' && c <= '9') || c == 0xB2 || c == 0xB3 || c == 0xB9 ||
           (c >= 0xBC && c <= 0xBE) || (c >= 0xFF10 && c <= 0xFF19);
}

/* \p{L}: exact for ASCII and Latin-1. Above, combining marks and the
 * punctuation and symbol blocks found in prompts (dashes, quotes, ellipsis,
 * arrows, math, CJK and fullwidth punctuation, emoji) are not letters, the
 * rest is. */
static int cp_is_letter(uint32_t c) {
    if (c < 0x80) return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z');
    if (c < 0x100) return c == 0xAA || c == 0xB5 || c == 0xBA ||
                          (c >= 0xC0 && c != 0xD7 && c != 0xF7);
    if (nfc_combining_class(c)) return 0;
    if (c >= 0xFF00 && c <= 0xFF65)
        return (c >= 0xFF21 && c <= 0xFF3A) || (c >= 0xFF41 && c <= 0xFF5A);
    return !((c >= 0x2000 && c <= 0x2BFF) || (c >= 0x3000 && c <= 0x303F) ||
             (c >= 0xFE30 && c <= 0xFE6F) || (c >= 0x1F000 && c <= 0x1FAFF));
}

/* \s */
static int cp_is_space(uint32_t c) {
    return c == ' ' || (c >= 0x09 && c <= 0x0D) || c == 0x85 || c == 0xA0 ||
           c == 0x1680 || (c >= 0x2000 && c <= 0x200A) || c == 0x2028 ||
           c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
}

static int cp_is_newline(uint32_t c) { return c == '\r' || c == '\n'; }

/* Length of the match at s of the Qwen2 pre-tokenizer pattern
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
 *    ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
 * trying the alternatives in order, as the regex engine does. */
static int pretoken_len(const char *s) {
    int n, n2;
    uint32_t c = utf8_decode(s, &n);

    /* Contractions */
    if (c == '\'') {
        char a = tolower((unsigned char)s[1]), b = a ? tolower((unsigned char)s[2]) : 0;
        if ((a == 'r' || a == 'v') && b == 'e') return 3;
        if (a == 'l' && b == 'l') return 3;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
    }

    /* Letters, with an optional leading character that is not a letter,
     * a number or a newline */
    int p = 0;
    if (!cp_is_letter(c) && !cp_is_number(c) && !cp_is_newline(c) &&
        cp_is_letter(utf8_decode(s + n, &n2)))
        p = n;
    if (p || cp_is_letter(c)) {
        while (s[p] && cp_is_letter(utf8_decode(s + p, &n2))) p += n2;
        return p;
    }

    /* One number */
    if (cp_is_number(c)) return n;

    /* Punctuation run, with an optional leading space and trailing newlines */
    p = c == ' ' ? 1 : 0;
    int q = p;
    while (s[q]) {
        uint32_t d = utf8_decode(s + q, &n2);
        if (cp_is_space(d) || cp_is_letter(d) || cp_is_number(d)) break;
        q += n2;
    }
    if (q > p) {
        while (s[q] == '\r' || s[q] == '\n') q++;
        return q;
    }

    /* Whitespace: up to its last newline, or all but the last character
     * when followed by a non-space, or all of it */
    int end = 0, last_nl = -1, last_start = 0;
    while (s[end]) {
        uint32_t d = utf8_decode(s + end, &n2);
        if (!cp_is_space(d)) break;
        if (cp_is_newline(d)) last_nl = end + n2;
        last_start = end;
        end += n2;
    }
    if (end == 0) return n;   /* not reached: every character matches above */
    if (last_nl > 0) return last_nl;
    if (!s[end] || last_start == 0) return end;
    return last_start;
}

static char **pretokenize(const char *text, int *num_chunks) {
    int capacity = 64;
    char **chunks = malloc(capacity * sizeof(char *));
    int count = 0;

    for (const char *p = text; *p; ) {
        int len = pretoken_len(p);
        char *chunk = malloc(len + 1);
        memcpy(chunk, p, len);
        chunk[len] = '\0';
        if (count >= capacity) {
            capacity *= 2;
            chunks = realloc(chunks, capacity * sizeof(char *));
        }
        chunks[count++] = chunk;
        p += len;
    }

    *num_chunks = count;
    return chunks;
}

/* ========================================================================
 * Main Tokenization API
 * ======================================================================== */

/* Append n token ids to the growable array tokens[*total], *capacity. */
static int *append_tokens(int *tokens, int *total, int *capacity, const int *ids, int n) {
    if (*total + n > *capacity) {
        while (*total + n > *capacity) *capacity *= 2;
        tokens = realloc(tokens, *capacity * sizeof(int));
    }
    memcpy(tokens + *total, ids, n * sizeof(int));
    *total += n;
    return tokens;
}

/* Normalize, pre-tokenize and BPE the len bytes of text between added tokens. */
static int *tokenize_piece(qwen3_tokenizer_t *tok, const char *text, size_t len, int *tokens,
                           int *total, int *capacity, int max_len) {
    char *normal = nfc_normalize(text, len);
    int num_chunks;
    char **chunks = pretokenize(normal, &num_chunks);
    free(normal);

    for (int c = 0; c < num_chunks; c++) {
        /* Convert to byte-level encoding, apply BPE, convert to token IDs */
        char *byte_text = text_to_bytes(chunks[c]);
        token_node_t *bpe_tokens = byte_text ? bpe_encode_word(tok, byte_text) : NULL;
        for (token_node_t *node = bpe_tokens; node && *total < max_len; node = node->next) {
            int id = vocab_hash_lookup(tok->vocab_hash, tok->hash_size, node->text);
            if (id >= 0) tokens = append_tokens(tokens, total, capacity, &id, 1);
        }
        free_token_list(bpe_tokens);
        free(byte_text);
        free(chunks[c]);
    }
    free(chunks);
    return tokens;
}

/* Length of the longest added token at s (0 if none), and its id. */
static size_t added_token_at(const qwen3_tokenizer_t *tok, const char *s, int *id) {
    size_t best = 0;
    for (int i = 0; i < tok->num_added; i++) {
        const char *content = tok->vocab[tok->added_ids[i]];
        size_t len = strlen(content);
        if (len > best && strncmp(s, content, len) == 0) {
            best = len;
            *id = tok->added_ids[i];
        }
    }
    return best;
}

/*
 * Tokenize text to token IDs, as the reference does: added tokens split the
 * text and stay as they are, the pieces between them go through NFC, the
 * pre-tokenizer and BPE.
 * Returns array of token IDs, caller must free.
 */
int *qwen3_tokenize(qwen3_tokenizer_t *tok, const char *text,
                    int *num_tokens, int max_len) {
    if (max_len <= 0) max_len = QWEN3_MAX_SEQ_LEN;

    int capacity = 256;
    int *tokens = malloc(capacity * sizeof(int));
    int total = 0;

    while (*text) {
        const char *end = text;
        size_t added = 0;
        int id = 0;
        while (*end && !(added = added_token_at(tok, end, &id))) end++;
        tokens = tokenize_piece(tok, text, (size_t)(end - text), tokens, &total, &capacity,
                                max_len);
        if (added && total < max_len) tokens = append_tokens(tokens, &total, &capacity, &id, 1);
        text = end + added;
    }

    *num_tokens = total;
    return tokens;
}

/* Apply the Qwen3 chat template and tokenize the result. Template:
 * <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
 * For Flux, also appends <think>\n\n</think>\n\n to match the training
 * template that triggers direct generation. For Z-Image the think tags
 * are skipped (controlled by skip_think_tags). */
int *qwen3_tokenize_chat(qwen3_tokenizer_t *tok, const char *prompt,
                         int *num_tokens, int max_len, int skip_think_tags) {
    if (max_len <= 0) max_len = QWEN3_MAX_SEQ_LEN;

    /* Tokenize everything and cut at the end, so a truncation can say how
     * many tokens were lost. */
    int limit = max_len;
    max_len = INT_MAX;

    int capacity = 256;
    int *tokens = malloc(capacity * sizeof(int));
    int total = 0;

    /* Add special tokens and text */
    /* <|im_start|> */
    tokens[total++] = QWEN3_IM_START_ID;

    /* "user\n" */
    int n;
    int *user_tokens = qwen3_tokenize(tok, "user\n", &n, max_len - total);
    for (int i = 0; i < n && total < max_len; i++) {
        if (total >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(int));
        }
        tokens[total++] = user_tokens[i];
    }
    free(user_tokens);

    /* prompt */
    int *prompt_tokens = qwen3_tokenize(tok, prompt, &n, max_len - total);
    for (int i = 0; i < n && total < max_len; i++) {
        if (total >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(int));
        }
        tokens[total++] = prompt_tokens[i];
    }
    free(prompt_tokens);

    /* <|im_end|>\n */
    if (total < max_len) {
        if (total >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(int));
        }
        tokens[total++] = QWEN3_IM_END_ID;
    }
    int *newline_tokens = qwen3_tokenize(tok, "\n", &n, max_len - total);
    for (int i = 0; i < n && total < max_len; i++) {
        if (total >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(int));
        }
        tokens[total++] = newline_tokens[i];
    }
    free(newline_tokens);

    /* <|im_start|> */
    if (total < max_len) {
        if (total >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(int));
        }
        tokens[total++] = QWEN3_IM_START_ID;
    }

    /* "assistant\n" */
    int *asst_tokens = qwen3_tokenize(tok, "assistant\n", &n, max_len - total);
    for (int i = 0; i < n && total < max_len; i++) {
        if (total >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(int));
        }
        tokens[total++] = asst_tokens[i];
    }
    free(asst_tokens);

    /* <think>\n\n</think>\n\n — only for Flux, not Z-Image. */
    if (!skip_think_tags) {
        if (total < max_len) {
            if (total >= capacity) {
                capacity *= 2;
                tokens = realloc(tokens, capacity * sizeof(int));
            }
            tokens[total++] = QWEN3_THINK_START_ID;
        }
        int *think_newlines = qwen3_tokenize(tok, "\n\n", &n, max_len - total);
        for (int i = 0; i < n && total < max_len; i++) {
            if (total >= capacity) {
                capacity *= 2;
                tokens = realloc(tokens, capacity * sizeof(int));
            }
            tokens[total++] = think_newlines[i];
        }
        free(think_newlines);

        if (total < max_len) {
            if (total >= capacity) {
                capacity *= 2;
                tokens = realloc(tokens, capacity * sizeof(int));
            }
            tokens[total++] = QWEN3_THINK_END_ID;
        }

        think_newlines = qwen3_tokenize(tok, "\n\n", &n, max_len - total);
        for (int i = 0; i < n && total < max_len; i++) {
            if (total >= capacity) {
                capacity *= 2;
                tokens = realloc(tokens, capacity * sizeof(int));
            }
            tokens[total++] = think_newlines[i];
        }
        free(think_newlines);
    }

    if (total > limit) {
        /* Leading newline: the one-shot CLI is mid-line on "Encoding text...". */
        fprintf(stderr, "\nWarning: prompt truncated from %d to %d tokens "
                "(text encoder limit)\n", total, limit);
        total = limit;
    }

    *num_tokens = total;
    return tokens;
}

/* Tokenize the raw template (Qwen-Image 2.1, no <think> block)
 *   <|im_start|>system\n{system}<|im_end|>\n
 *   <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n
 * Each role header is tokenized together with its text, as the reference
 * tokenizes the whole string (the special tokens split it). num_system
 * receives the length of the system block, which callers drop from the
 * encoder output. An empty prompt becomes " ". Over max_len tokens the
 * end of the prompt is cut, keeping the template tail. */
int *qwen3_tokenize_chat_system(qwen3_tokenizer_t *tok, const char *system,
                                const char *prompt, int *num_tokens,
                                int *num_system, int max_len) {
    static const int im_start = QWEN3_IM_START_ID, im_end = QWEN3_IM_END_ID;
    if (!prompt[0]) prompt = " ";

    size_t sys_len = strlen(system), prompt_len = strlen(prompt);
    char *sys_text = malloc(sys_len + 8);
    char *user_text = malloc(prompt_len + 8);
    sprintf(sys_text, "system\n%s", system);
    sprintf(user_text, "user\n%s", prompt);

    int n_sys, n_user, n_nl, n_asst;
    int *sys_ids = qwen3_tokenize(tok, sys_text, &n_sys, INT_MAX);
    int *user_ids = qwen3_tokenize(tok, user_text, &n_user, INT_MAX);
    int *nl_ids = qwen3_tokenize(tok, "\n", &n_nl, INT_MAX);
    int *asst_ids = qwen3_tokenize(tok, "assistant\n", &n_asst, INT_MAX);
    free(sys_text);
    free(user_text);

    /* Fixed parts around the user text: the system block, the user
     * <|im_start|>, then <|im_end|>\n<|im_start|>assistant\n. */
    int fixed = (2 + n_sys + n_nl) + 1 + (1 + n_nl + 1 + n_asst);
    int total_full = fixed + n_user;
    if (total_full > max_len) {
        int keep = max_len - fixed;
        if (keep < 1) keep = 1;
        fprintf(stderr, "\nWarning: prompt truncated from %d to %d tokens "
                "(text encoder limit)\n", total_full, fixed + keep);
        n_user = keep;
    }

    int capacity = 256, total = 0;
    int *tokens = malloc(capacity * sizeof(int));
    tokens = append_tokens(tokens, &total, &capacity, &im_start, 1);
    tokens = append_tokens(tokens, &total, &capacity, sys_ids, n_sys);
    tokens = append_tokens(tokens, &total, &capacity, &im_end, 1);
    tokens = append_tokens(tokens, &total, &capacity, nl_ids, n_nl);
    *num_system = total;
    tokens = append_tokens(tokens, &total, &capacity, &im_start, 1);
    tokens = append_tokens(tokens, &total, &capacity, user_ids, n_user);
    tokens = append_tokens(tokens, &total, &capacity, &im_end, 1);
    tokens = append_tokens(tokens, &total, &capacity, nl_ids, n_nl);
    tokens = append_tokens(tokens, &total, &capacity, &im_start, 1);
    tokens = append_tokens(tokens, &total, &capacity, asst_ids, n_asst);

    free(sys_ids);
    free(user_ids);
    free(nl_ids);
    free(asst_ids);
    *num_tokens = total;
    return tokens;
}

/* qwen3_tokenize_chat_system with images in the user turn. Each piece between
 * special tokens is tokenized on its own, <imageN> is literal text and the
 * prompt no longer joins the role header. */
int *qwen3_tokenize_chat_images(qwen3_tokenizer_t *tok, const char *system,
                                const char *prompt, const int *slots, int n,
                                int *num_tokens, int *num_system, int *image_rows) {
    static const int im_start = QWEN3_IM_START_ID, im_end = QWEN3_IM_END_ID;
    static const int vision_start = QWEN3_VISION_START_ID, vision_end = QWEN3_VISION_END_ID;
    static const int image_pad = QWEN3_IMAGE_PAD_ID;
    if (!prompt[0]) prompt = " ";

    char *sys_text = malloc(strlen(system) + 8);
    sprintf(sys_text, "system\n%s", system);
    int n_sys, n_prompt, n_nl, n_asst;
    int *sys_ids = qwen3_tokenize(tok, sys_text, &n_sys, INT_MAX);
    int *prompt_ids = qwen3_tokenize(tok, prompt, &n_prompt, INT_MAX);
    int *nl_ids = qwen3_tokenize(tok, "\n", &n_nl, INT_MAX);
    int *asst_ids = qwen3_tokenize(tok, "assistant\n", &n_asst, INT_MAX);
    free(sys_text);

    int capacity = 256, total = 0;
    int *tokens = malloc(capacity * sizeof(int));
    tokens = append_tokens(tokens, &total, &capacity, &im_start, 1);
    tokens = append_tokens(tokens, &total, &capacity, sys_ids, n_sys);
    tokens = append_tokens(tokens, &total, &capacity, &im_end, 1);
    tokens = append_tokens(tokens, &total, &capacity, nl_ids, n_nl);
    *num_system = total;
    tokens = append_tokens(tokens, &total, &capacity, &im_start, 1);
    for (int i = 0; i < n; i++) {
        char label[32];
        int n_label;
        snprintf(label, sizeof(label), "%s<image%d>", i ? " " : "user\n", i + 1);
        int *label_ids = qwen3_tokenize(tok, label, &n_label, INT_MAX);
        tokens = append_tokens(tokens, &total, &capacity, label_ids, n_label);
        free(label_ids);
        tokens = append_tokens(tokens, &total, &capacity, &vision_start, 1);
        image_rows[i] = total;
        for (int s = 0; s < slots[i]; s++)
            tokens = append_tokens(tokens, &total, &capacity, &image_pad, 1);
        tokens = append_tokens(tokens, &total, &capacity, &vision_end, 1);
    }
    tokens = append_tokens(tokens, &total, &capacity, prompt_ids, n_prompt);
    tokens = append_tokens(tokens, &total, &capacity, &im_end, 1);
    tokens = append_tokens(tokens, &total, &capacity, nl_ids, n_nl);
    tokens = append_tokens(tokens, &total, &capacity, &im_start, 1);
    tokens = append_tokens(tokens, &total, &capacity, asst_ids, n_asst);

    free(sys_ids);
    free(prompt_ids);
    free(nl_ids);
    free(asst_ids);
    *num_tokens = total;
    return tokens;
}

/*
 * Pad token sequence to max_len with PAD tokens.
 * Returns new array, caller must free original.
 */
int *qwen3_pad_tokens(int *tokens, int num_tokens, int max_len, int *attention_mask) {
    int *padded = malloc(max_len * sizeof(int));
    if (!padded) return NULL;

    for (int i = 0; i < max_len; i++) {
        if (i < num_tokens) {
            padded[i] = tokens[i];
            if (attention_mask) attention_mask[i] = 1;
        } else {
            padded[i] = QWEN3_PAD_ID;
            if (attention_mask) attention_mask[i] = 0;
        }
    }

    return padded;
}

/* ========================================================================
 * Debug / Utility Functions
 * ======================================================================== */

/* Get token string by ID */
const char *qwen3_get_token(qwen3_tokenizer_t *tok, int id) {
    if (!tok || id < 0 || id >= tok->vocab_size) return NULL;
    return tok->vocab[id];
}

/* Get token ID by string */
int qwen3_get_id(qwen3_tokenizer_t *tok, const char *token) {
    if (!tok || !token) return -1;
    return vocab_hash_lookup(tok->vocab_hash, tok->hash_size, token);
}
