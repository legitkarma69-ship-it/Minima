#include "tokenizer.h"
#include "util.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

// ------------------------------------------------------------------
// Internal helpers
// ------------------------------------------------------------------

static void add_token(Tokenizer* tok, Arena* arena, const char* s, int len) {
    if (tok->vocab_size >= TOK_MAX_VOCAB) die("vocab overflow");
    char* copy = arena_alloc(arena, len + 1);
    if (!copy) die("arena OOM in tokenizer");
    memcpy(copy, s, len);
    copy[len] = 0;
    tok->token_str[tok->vocab_size] = copy;
    tok->token_len[tok->vocab_size] = len;
    tok->vocab_size++;
}

// Very small open-address hash for pair counting during training
typedef struct {
    uint32_t key;     // (left << 16) | right
    int32_t  count;
    int32_t  new_id;  // filled after merge decision
} PairSlot;

#define PAIR_HASH_SIZE  (1 << 18)   // 256 k slots – enough for 8k vocab

static uint32_t pair_hash(uint16_t left, uint16_t right) {
    uint32_t k = ((uint32_t)left << 16) | right;
    k *= 0x9e3779b9u;
    return k & (PAIR_HASH_SIZE - 1);
}

// ------------------------------------------------------------------
// Training
// ------------------------------------------------------------------

bool tokenizer_train(Tokenizer* tok, Arena* arena,
    const uint8_t* sample, size_t sample_len,
    int target_vocab) {

    if (target_vocab < 260 || target_vocab > TOK_MAX_VOCAB)
        die("vocab_size must be 260..%d", TOK_MAX_VOCAB);

    memset(tok, 0, sizeof(*tok));

    // 1. Special tokens
    add_token(tok, arena, "<bos>", 5);
    add_token(tok, arena, "<eos>", 5);
    add_token(tok, arena, "<pad>", 5);
    add_token(tok, arena, "<unk>", 5);

    // 2. Raw bytes 0..255
    for (int b = 0; b < 256; b++) {
        char tmp[8];
        int len = snprintf(tmp, sizeof(tmp), "<0x%02x>", b);
        // We actually store the single byte for speed later
        // but keep a printable name for debugging.
        // For the real token we store the byte itself.
        char byte[2] = { (char)b, 0 };
        add_token(tok, arena, byte, 1);
        (void)len; // silence unused
    }

    // At this point vocab_size == 260 (4 specials + 256 bytes)

    // 3. Work buffer: we keep a list of token ids for the whole sample
    //    (this is the most memory-hungry part of training)
    size_t max_tokens = sample_len + 16;
    uint16_t* ids = arena_alloc(arena, max_tokens * sizeof(uint16_t));
    if (!ids) die("arena OOM for training ids");

    // Initial: every byte becomes its own token
    size_t n = 0;
    for (size_t i = 0; i < sample_len; i++) {
        ids[n++] = (uint16_t)(TOK_BYTE_START + sample[i]);
    }

    // 4. Pair counting table (temporary, lives in arena)
    PairSlot* table = arena_alloc_zero(arena, PAIR_HASH_SIZE * sizeof(PairSlot));
    if (!table) die("arena OOM for pair table");

    int merges_needed = target_vocab - tok->vocab_size;

    for (int m = 0; m < merges_needed; m++) {
        // Clear counts
        for (int i = 0; i < PAIR_HASH_SIZE; i++) {
            table[i].count = 0;
            table[i].key = 0;
        }

        // Count pairs
        for (size_t i = 0; i + 1 < n; i++) {
            uint16_t a = ids[i];
            uint16_t b = ids[i + 1];

            // Keep every learned token within TOK_MAX_TOKEN_LEN.
            int len_a = tok->token_len[a];
            int len_b = tok->token_len[b];
            if (len_a <= 0 || len_b <= 0 ||
                len_a > TOK_MAX_TOKEN_LEN - len_b)
                continue;

            uint32_t h = pair_hash(a, b);
            // Linear probing must either find this pair or an empty slot.
            // A bounded 32-slot search silently loses counts under collisions.
            bool inserted = false;
            const uint32_t key = ((uint32_t)a << 16) | b;
            for (uint32_t probe = 0; probe < PAIR_HASH_SIZE; probe++) {
                uint32_t slot = (h + probe) & (PAIR_HASH_SIZE - 1);
                if (table[slot].count == 0) {
                    table[slot].key = key;
                    table[slot].count = 1;
                    inserted = true;
                    break;
                }
                if (table[slot].key == key) {
                    if (table[slot].count == INT32_MAX)
                        die("BPE pair count overflow");
                    table[slot].count++;
                    inserted = true;
                    break;
                }
            }
            if (!inserted)
                die("BPE pair hash table is full; increase PAIR_HASH_SIZE");
        }

        // Find most frequent pair
        int best_count = 0;
        uint16_t best_left = 0, best_right = 0;
        for (int i = 0; i < PAIR_HASH_SIZE; i++) {
            if (table[i].count > best_count) {
                best_count = table[i].count;
                best_left = (uint16_t)(table[i].key >> 16);
                best_right = (uint16_t)(table[i].key & 0xffff);
            }
        }

        if (best_count < 2) break;   // nothing left to merge

        // Create new token
        int new_id = tok->vocab_size;
        // Build the string by concatenating the two children
        int len1 = tok->token_len[best_left];
        int len2 = tok->token_len[best_right];
        char* new_str = arena_alloc(arena, (size_t)len1 + (size_t)len2 + 1);
        if (!new_str) die("arena OOM for merged token");
        memcpy(new_str, tok->token_str[best_left], (size_t)len1);
        memcpy(new_str + len1, tok->token_str[best_right], (size_t)len2);
        new_str[len1 + len2] = 0;
        tok->token_str[new_id] = new_str;
        tok->token_len[new_id] = len1 + len2;
        tok->vocab_size++;

        // Record merge
        tok->merges[tok->n_merges].left = best_left;
        tok->merges[tok->n_merges].right = best_right;
        tok->n_merges++;

        // Replace all occurrences of the pair in the id stream
        size_t w = 0;
        for (size_t r = 0; r < n; ) {
            if (r + 1 < n && ids[r] == best_left && ids[r + 1] == best_right) {
                ids[w++] = (uint16_t)new_id;
                r += 2;
            }
            else {
                ids[w++] = ids[r++];
            }
        }
        n = w;

        if ((m + 1) % 500 == 0)
            printf("merge %d / %d   tokens left in sample: %zu\n",
                m + 1, merges_needed, n);
    }

    printf("BPE finished. vocab = %d  merges = %d\n",
        tok->vocab_size, tok->n_merges);
    return true;
}

// ------------------------------------------------------------------
// Encode
// ------------------------------------------------------------------

int tokenizer_encode(const Tokenizer* tok,
    const uint8_t* text, size_t text_len,
    uint16_t* out, int out_cap) {

    if (text_len == 0 || out_cap <= 0) return 0;

    // First pass: bytes → token ids
    int n = 0;
    for (size_t i = 0; i < text_len && n < out_cap; i++) {
        out[n++] = (uint16_t)(TOK_BYTE_START + text[i]);
    }

    // Apply merges in order (classic BPE)
    // Because the merge list is small we just rescanning is fine
    // for training batch sizes.
    for (int m = 0; m < tok->n_merges; m++) {
        uint16_t left = tok->merges[m].left;
        uint16_t right = tok->merges[m].right;
        uint16_t newid = (uint16_t)(260 + m);   // specials + bytes = 260

        int w = 0;
        for (int r = 0; r < n; ) {
            if (r + 1 < n && out[r] == left && out[r + 1] == right) {
                out[w++] = newid;
                r += 2;
            }
            else {
                out[w++] = out[r++];
            }
        }
        n = w;
    }

    return n;
}

// ------------------------------------------------------------------
// Decode
// ------------------------------------------------------------------

int tokenizer_decode(const Tokenizer* tok,
    const uint16_t* ids, int n_ids,
    char* out, int out_cap) {

    int pos = 0;
    for (int i = 0; i < n_ids; i++) {
        uint16_t id = ids[i];
        if (id >= tok->vocab_size) continue;

        int len = tok->token_len[id];
        if (pos + len >= out_cap) break;

        memcpy(out + pos, tok->token_str[id], len);
        pos += len;
    }
    if (pos < out_cap) out[pos] = 0;
    return pos;
}

// ------------------------------------------------------------------
// Save / Load (simple binary format)
// ------------------------------------------------------------------

bool tokenizer_save(const Tokenizer* tok, const char* path) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;

    // header
    uint32_t magic = 0x42504531;          // "BPE1"
    fwrite(&magic, 4, 1, f);
    fwrite(&tok->vocab_size, 4, 1, f);
    fwrite(&tok->n_merges, 4, 1, f);

    // token strings
    for (int i = 0; i < tok->vocab_size; i++) {
        uint16_t len = (uint16_t)tok->token_len[i];
        fwrite(&len, 2, 1, f);
        fwrite(tok->token_str[i], 1, len, f);
    }

    // merges
    fwrite(tok->merges, sizeof(Merge), tok->n_merges, f);

    fclose(f);
    return true;
}

bool tokenizer_load(Tokenizer* tok, Arena* arena, const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    uint32_t magic;
    if (fread(&magic, 4, 1, f) != 1 || magic != 0x42504531) {
        fclose(f);
        return false;
    }

    memset(tok, 0, sizeof(*tok));
    fread(&tok->vocab_size, 4, 1, f);
    fread(&tok->n_merges, 4, 1, f);

    for (int i = 0; i < tok->vocab_size; i++) {
        uint16_t len;
        fread(&len, 2, 1, f);
        char* s = arena_alloc(arena, len + 1);
        fread(s, 1, len, f);
        s[len] = 0;
        tok->token_str[i] = s;
        tok->token_len[i] = len;
    }

    fread(tok->merges, sizeof(Merge), tok->n_merges, f);
    fclose(f);
    return true;
}