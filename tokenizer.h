#ifndef TOKENIZER_H
#define TOKENIZER_H

#include "arena.h"
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define TOK_MAX_VOCAB     16384
#define TOK_MAX_MERGES    (TOK_MAX_VOCAB - 256)
#define TOK_MAX_TOKEN_LEN 64          // longest token string we keep

typedef struct {
    uint16_t left;                    // token id
    uint16_t right;                   // token id
} Merge;

typedef struct {
    int      vocab_size;
    int      n_merges;

    // token id → null-terminated string (owned by arena)
    char* token_str[TOK_MAX_VOCAB];
    int      token_len[TOK_MAX_VOCAB];

    // merge table: ordered by creation (lower index = higher priority)
    Merge    merges[TOK_MAX_MERGES];

    // fast lookup: pair (left,right) → new token id  (or 0 if none)
    // we use a simple open-address hash for the training phase only
    // at runtime we just walk the merge list (tiny, cache-friendly)
} Tokenizer;

// ---------- Training ----------
// Train BPE from a raw text buffer (bytes).
// `sample` is the text, `sample_len` its length.
// `vocab_size` target (must be >= 256 + specials).
// Returns true on success. Everything is allocated from `arena`.
bool tokenizer_train(Tokenizer* tok, Arena* arena,
    const uint8_t* sample, size_t sample_len,
    int vocab_size);

// ---------- Encode / Decode ----------
// Encode text → token ids. Returns number of tokens written, 0 for empty input,
// or -1 for invalid arguments / insufficient output capacity.
// `out` must have room for at least `text_len` tokens (worst case).
int  tokenizer_encode(const Tokenizer* tok,
    const uint8_t* text, size_t text_len,
    uint16_t* out, int out_cap);

// Decode token ids → text. Returns bytes written (not including NUL).
// `out` is NUL-terminated if there is room.
int  tokenizer_decode(const Tokenizer* tok,
    const uint16_t* ids, int n_ids,
    char* out, int out_cap);

// Stable fingerprint of vocabulary strings and ordered merge rules.
uint64_t tokenizer_fingerprint(const Tokenizer* tok);

// ---------- Persistence ----------
bool tokenizer_save(const Tokenizer* tok, const char* path);
bool tokenizer_load(Tokenizer* tok, Arena* arena, const char* path);

// Special token ids (fixed)
enum {
    TOK_BOS = 0,          // we reserve 0..3
    TOK_EOS = 1,
    TOK_PAD = 2,
    TOK_UNK = 3,
    TOK_BYTE_START = 4    // 4..259 = raw bytes 0..255
};

#endif