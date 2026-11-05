#ifndef FORWARD_H
#define FORWARD_H

#include "model.h"
#include "arena.h"
#include <stdint.h>

// Temporary activation storage for one forward pass.
// Allocated once from a temp arena and reused.
typedef struct {

        float* x;           // residual stream [batch*seq*dim]  – kept after forward
        float* x_norm;      // after each RMSNorm
        float* qkv;         // [batch*seq*3*dim]
        float* attn_out;
        float* ffn_buf;     // up-projection result
        float* gate_buf;    // gate projection (new)
        float* logits;
        float* resid_attn;  // [n_layers, batch*seq*dim]  input to attention block
        float* resid_ffn;   // [n_layers, batch*seq*dim]  input to FFN block
    } Activations;


// Allocate all activation buffers for a given batch size and sequence length.
// Call once at startup (or when batch/seq changes).
bool activations_init(Activations* act, Arena* temp_arena,
    int batch, int seq, const Model* model);

// Full forward pass.
// tokens: [batch, seq]  (uint16_t token ids)
// Returns pointer to logits [batch * seq * vocab] (also stored in act->logits)
float* model_forward(Model* model, Activations* act,
    const uint16_t* tokens, int batch, int seq);

#endif