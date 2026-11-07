#ifndef FORWARD_H
#define FORWARD_H

#include "model.h"
#include "arena.h"
#include <stdint.h>

// Forward-pass caches and backward-pass scratch storage.
// All buffers are allocated from the caller's temporary arena.
typedef struct {
    // Final residual stream and reusable normalization workspace.
    float* x;               // [batch * seq, dim]
    float* x_norm;          // [batch * seq, dim]

    // Per-layer forward caches.
    float* qkv;             // [layers, batch * seq, 3 * dim]
    float* attn_out;        // [layers, batch * seq, dim]
    float* ffn_buf;         // [layers, batch * seq, ffn_dim], SwiGLU output
    float* up_buf;          // [layers, batch * seq, ffn_dim], up projection
    float* gate_buf;        // [layers, batch * seq, ffn_dim], gate projection
    float* resid_attn;      // [layers, batch * seq, dim], input to attention block
    float* resid_ffn;       // [layers, batch * seq, dim], input to FFN block

    // Final output.
    float* logits;          // [batch * seq, vocab]

    // Backward-pass scratch buffers. These never alias forward caches.
    float* dx;              // [batch * seq, dim]
    float* dx_tmp;          // [batch * seq, dim]
    float* dx_norm;         // [batch * seq, dim]
    float* dattn;          // [batch * seq, dim]
    float* dqkv;            // [batch * seq, 3 * dim]
    float* dffn;            // [batch * seq, ffn_dim]
    float* dup;             // [batch * seq, ffn_dim]
    float* dgate;           // [batch * seq, ffn_dim]

    // Reusable sequence-sized attention softmax scratch (forward and backward).
    float* attention_scores; // [seq]
} Activations;

// Allocate all activation and gradient-workspace buffers.
// The caller must reset/reallocate the temp arena between training steps.
bool activations_init(Activations* act, Arena* temp_arena,
    int batch, int seq, const Model* model);

// Shared scalar kernels used by both forward and backward passes.
void rmsnorm(float* out, const float* x, const float* scale,
    int dim, float eps);
void matmul(float* out, const float* x, const float* w,
    int n, int d_in, int d_out);
float silu(float x);

// Full forward pass.
// tokens: [batch, seq] (uint16_t token ids)
// Returns logits [batch * seq, vocab], or NULL for invalid dimensions/input.
float* model_forward(Model* model, Activations* act,
    const uint16_t* tokens, int batch, int seq);

#endif
