#ifndef MODEL_H
#define MODEL_H

#include "arena.h"
#include "config.h"
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    // Hyper-parameters (copied from config for convenience)
    int vocab_size;
    int dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int head_dim;
    int kv_dim;
    int qkv_dim;
    int ffn_dim;
    int max_seq;

    // ---------- Weights (all pointers into the arena) ----------
    float* tok_emb;          // [vocab_size, dim]
    // Per-layer (contiguous blocks)
    float* attn_qkv;         // [n_layers, qkv_dim, dim] (Q + compact grouped K/V)
    float* attn_proj;        // [n_layers, dim, dim]
    float* ffn_up;           // [n_layers, ffn_dim, dim]
    float* ffn_gate;         // [n_layers, ffn_dim, dim]   (SwiGLU)
    float* ffn_down;         // [n_layers, dim, ffn_dim]
    float* rms1_scale;       // [n_layers, dim]
    float* rms2_scale;       // [n_layers, dim]

    float* rms_final_scale;  // [dim]

    // Output head – we tie it to tok_emb (no extra storage)
    // float *lm_head;       // intentionally absent

    // Total bytes occupied by weights
    size_t weight_bytes;
} Model;

// Allocate all weight tensors from the arena and initialise them.
// Returns true on success.
bool model_init(Model* m, Arena* arena, uint64_t seed, int vocab_size);
bool model_init_config(Model* m, Arena* arena, uint64_t seed, const ModelConfig* config);

/* Shared invariants used by initialization, forward/backward, and training. */
bool model_config_validate(const ModelConfig* config);
bool model_validate(const Model* model);

// Convenience: number of parameters
size_t model_param_count(const Model* m);

// Debug print
void model_summary(const Model* m);

#endif
