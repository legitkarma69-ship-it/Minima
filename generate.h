#ifndef GENERATE_H
#define GENERATE_H

#include "model.h"
#include "tokenizer.h"
#include "arena.h"
#include "util.h"
#include <stdint.h>

// Simple KV cache for autoregressive generation
typedef struct {
    int max_seq;
    int n_layers;
    int n_heads;
    int head_dim;
    int dim;

    // [n_layers, max_seq, dim] for K and V (we store the full projected K/V)
    float* k_cache;   // [n_layers, max_seq, dim]
    float* v_cache;   // [n_layers, max_seq, dim]
    int    pos;       // current length
} KVCache;

bool kvcache_init(KVCache* cache, Arena* arena, const Model* model, int max_seq);
void kvcache_reset(KVCache* cache);

// Verify that incremental KV-cache inference matches a full causal
// forward pass for the final token in a sequence.
bool generation_check_kv_cache(Model* model, KVCache* cache, Arena* temp_arena,
    const uint16_t* tokens, int seq, float tolerance);

bool generation_inspect_next_tokens(Model* model, const Tokenizer* tok,
    KVCache* cache, const uint16_t* prompt_ids, int prompt_len, int top_n);

// Generate `n_new` tokens starting from the prompt.
// Writes the generated token ids into `out_ids` (including the prompt).
// Returns total number of tokens written.
int generate(Model* model, Tokenizer* tok, KVCache* cache,
    const uint16_t* prompt_ids, int prompt_len,
    uint16_t* out_ids, int out_cap,
    int n_new,
    float temperature, int top_k,
    Rng* rng);

int generate_ex(Model* model, Tokenizer* tok, KVCache* cache,
    const uint16_t* prompt_ids, int prompt_len,
    uint16_t* out_ids, int out_cap, int n_new,
    float temperature, int top_k, float top_p,
    float repetition_penalty, Rng* rng);

#endif