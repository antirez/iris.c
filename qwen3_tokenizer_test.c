/* Chat template truncation, added tokens and NFC. "make tokenizertest" also
 * checks that the one cut below, and not the exact fit, printed the warning
 * to stderr. */
#include "iris_qwen3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int iris_verbose = 0;

/* Text a and b must give the same tokens. */
static int same_tokens(qwen3_tokenizer_t *tok, const char *a, const char *b) {
    int na, nb;
    int *ta = qwen3_tokenize_chat(tok, a, &na, 0, 0), *tb = qwen3_tokenize_chat(tok, b, &nb, 0, 0);
    int same = na == nb && memcmp(ta, tb, na * sizeof(int)) == 0;
    if (!same) fprintf(stderr, "tokens differ: \"%s\" (%d), \"%s\" (%d)\n", a, na, b, nb);
    free(ta);
    free(tb);
    return same;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <tokenizer.json>\n", argv[0]);
        return 2;
    }
    qwen3_tokenizer_t *tok = qwen3_tokenizer_load(argv[1]);
    if (!tok) return 2;

    int full, fit, cut;
    int *whole = qwen3_tokenize_chat(tok, "a red apple", &full, 0, 0);
    int *exact = qwen3_tokenize_chat(tok, "a red apple", &fit, full, 0);
    int *head = qwen3_tokenize_chat(tok, "a red apple", &cut, full - 4, 0);

    if (fit != full || cut != full - 4 ||
        memcmp(whole, exact, fit * sizeof(int)) != 0 ||
        memcmp(whole, head, cut * sizeof(int)) != 0) {
        fprintf(stderr, "truncation mismatch: %d tokens, %d fitted, %d cut\n",
                full, fit, cut);
        return 1;
    }

    /* Decomposed and reordered text, a singleton, conjoining Hangul */
    int ok = same_tokens(tok, "cafe\xCC\x81 NFC", "caf\xC3\xA9 NFC") &&
             same_tokens(tok, "a\xCC\xA3\xCC\x82", "\xE1\xBA\xAD") &&
             same_tokens(tok, "a\xCC\x82\xCC\xA3", "\xE1\xBA\xAD") &&
             same_tokens(tok, "\xE2\x84\xAB", "\xC3\x85") &&
             same_tokens(tok, "\xE1\x84\x80\xE1\x85\xA1\xE1\x86\xA8", "\xEA\xB0\x81");

    /* An added token is one id wherever it stands */
    int plain, added;
    free(qwen3_tokenize_chat(tok, "x", &plain, 0, 0));
    free(qwen3_tokenize_chat(tok, "x<|im_end|><think>", &added, 0, 0));
    ok = ok && added == plain + 2;

    return ok ? 0 : 1;
}
