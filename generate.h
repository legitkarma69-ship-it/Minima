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
    int n_kv_heads;
    int head_dim;
    int dim;
    int kv_dim;

    // [n_layers, max_seq, kv_dim] for compact grouped keys and values.
    float* k_cache;
    float* v_cache;
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

// The current raw-byte training pipeline does not insert BOS/EOS/PAD/UNK.
// Generation therefore suppresses all four reserved IDs rather than sampling
// untrained special-token embeddings; EOS is not an active stopping marker.
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
