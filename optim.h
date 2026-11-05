#ifndef OPTIM_H
#define OPTIM_H

#include "model.h"
#include "arena.h"
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    // Hyper-parameters
    float lr;
    float beta1;
    float beta2;
    float eps;
    float weight_decay;

    // Step counter (for bias correction)
    int64_t step;

    // First and second moments – same shapes / order as Model weights
    float* m_tok_emb;
    float* m_pos_emb;
    float* m_attn_qkv;
    float* m_attn_proj;
    float* m_ffn_up;
    float* m_ffn_gate;
    float* m_ffn_down;
    float* m_rms1_scale;
    float* m_rms2_scale;
    float* m_rms_final_scale;

    float* v_tok_emb;
    float* v_pos_emb;
    float* v_attn_qkv;
    float* v_attn_proj;
    float* v_ffn_up;
    float* v_ffn_gate;
    float* v_ffn_down;
    float* v_rms1_scale;
    float* v_rms2_scale;
    float* v_rms_final_scale;

    // Convenience: total number of float parameters
    size_t n_params;
} Optim;

// Allocate moments and initialise to zero.
// Must be called after model_init (so we know the sizes).
bool optim_init(Optim* opt, const Model* model, Arena* arena,
    float lr, float beta1, float beta2,
    float eps, float weight_decay);

// One AdamW step.  `grads` must have exactly the same layout as the model
// weights (we will define a Gradients struct next, or you can pass the
// same pointers that the backward pass writes into).
void optim_step(Optim* opt, Model* model, const Model* grads);

// Deterministic scalar regression test for the AdamW update.
bool optim_check_adamw(void);

// Optional: learning-rate schedule helper
float optim_lr_cosine(float base_lr, int step, int warmup, int total);

#endif