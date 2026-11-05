#include "optim.h"
#include "util.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

// ------------------------------------------------------------------
// Internal: allocate a moment buffer of `count` floats and zero it
// ------------------------------------------------------------------
static float* alloc_moment(Arena* arena, size_t count) {
    float* p = arena_alloc_zero(arena, count * sizeof(float));
    if (!p) die("optim moment allocation failed");
    return p;
}

// ------------------------------------------------------------------
// Core AdamW update for a single contiguous parameter block
// ------------------------------------------------------------------
static void adamw_update(float* param, float* m, float* v, const float* grad,
    size_t n, float lr, float beta1, float beta2,
    float eps, float weight_decay, float bias1, float bias2) {

    float lr_t = lr * sqrtf(bias2) / bias1;   // bias-corrected learning rate

    for (size_t i = 0; i < n; i++) {
        float g = grad[i];

        // weight decay (AdamW style – applied to the parameter)
        param[i] -= lr * weight_decay * param[i];

        // moments
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

        // update
        float m_hat = m[i];           // already scaled by lr_t below
        float v_hat = v[i];
        param[i] -= lr_t * m_hat / (sqrtf(v_hat) + eps);
    }
}

// ------------------------------------------------------------------
// Public API
// ------------------------------------------------------------------

bool optim_init(Optim* opt, const Model* model, Arena* arena,
    float lr, float beta1, float beta2,
    float eps, float weight_decay) {

    memset(opt, 0, sizeof(*opt));

    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->weight_decay = weight_decay;
    opt->step = 0;

    // Sizes (must match model_init exactly)
    size_t tok_n = (size_t)model->vocab_size * model->dim;
    size_t pos_n = (size_t)model->max_seq * model->dim;
    size_t qkv_n = (size_t)model->n_layers * 3 * model->dim * model->dim;
    size_t proj_n = (size_t)model->n_layers * model->dim * model->dim;
    size_t up_n = (size_t)model->n_layers * model->ffn_dim * model->dim;
    size_t gate_n = (size_t)model->n_layers * model->ffn_dim * model->dim;
    size_t down_n = (size_t)model->n_layers * model->dim * model->ffn_dim;
    size_t rms_n = (size_t)model->n_layers * model->dim;
    size_t final_n = model->dim;

    // Allocate first moments
    opt->m_tok_emb = alloc_moment(arena, tok_n);
    opt->m_pos_emb = alloc_moment(arena, pos_n);
    opt->m_attn_qkv = alloc_moment(arena, qkv_n);
    opt->m_attn_proj = alloc_moment(arena, proj_n);
    opt->m_ffn_up = alloc_moment(arena, up_n);
    opt->m_ffn_gate = alloc_moment(arena, gate_n);
    opt->m_ffn_down = alloc_moment(arena, down_n);
    opt->m_rms1_scale = alloc_moment(arena, rms_n);
    opt->m_rms2_scale = alloc_moment(arena, rms_n);
    opt->m_rms_final_scale = alloc_moment(arena, final_n);

    // Allocate second moments
    opt->v_tok_emb = alloc_moment(arena, tok_n);
    opt->v_pos_emb = alloc_moment(arena, pos_n);
    opt->v_attn_qkv = alloc_moment(arena, qkv_n);
    opt->v_attn_proj = alloc_moment(arena, proj_n);
    opt->v_ffn_up = alloc_moment(arena, up_n);
    opt->v_ffn_gate = alloc_moment(arena, gate_n);
    opt->v_ffn_down = alloc_moment(arena, down_n);
    opt->v_rms1_scale = alloc_moment(arena, rms_n);
    opt->v_rms2_scale = alloc_moment(arena, rms_n);
    opt->v_rms_final_scale = alloc_moment(arena, final_n);

    opt->n_params = tok_n + pos_n + qkv_n + proj_n + up_n + gate_n +
        down_n + rms_n * 2 + final_n;

    return true;
}

void optim_step(Optim* opt, Model* model, const Model* grads) {
    opt->step++;

    float b1 = opt->beta1;
    float b2 = opt->beta2;
    float bias1 = 1.0f - powf(b1, (float)opt->step);
    float bias2 = 1.0f - powf(b2, (float)opt->step);

    // Helper macro to keep the call sites short
#define UPDATE(field, count) \
        adamw_update(model->field, opt->m_##field, opt->v_##field, \
                     grads->field, (count), \
                     opt->lr, b1, b2, opt->eps, opt->weight_decay, \
                     bias1, bias2)

    size_t tok_n = (size_t)model->vocab_size * model->dim;
    size_t pos_n = (size_t)model->max_seq * model->dim;
    size_t qkv_n = (size_t)model->n_layers * 3 * model->dim * model->dim;
    size_t proj_n = (size_t)model->n_layers * model->dim * model->dim;
    size_t up_n = (size_t)model->n_layers * model->ffn_dim * model->dim;
    size_t gate_n = (size_t)model->n_layers * model->ffn_dim * model->dim;
    size_t down_n = (size_t)model->n_layers * model->dim * model->ffn_dim;
    size_t rms_n = (size_t)model->n_layers * model->dim;

    UPDATE(tok_emb, tok_n);
    UPDATE(pos_emb, pos_n);
    UPDATE(attn_qkv, qkv_n);
    UPDATE(attn_proj, proj_n);
    UPDATE(ffn_up, up_n);
    UPDATE(ffn_gate, gate_n);
    UPDATE(ffn_down, down_n);
    UPDATE(rms1_scale, rms_n);
    UPDATE(rms2_scale, rms_n);
    UPDATE(rms_final_scale, model->dim);

#undef UPDATE
}

// Simple cosine decay with linear warmup
float optim_lr_cosine(float base_lr, int step, int warmup, int total) {
    if (step < warmup)
        return base_lr * (float)step / (float)warmup;

    float progress = (float)(step - warmup) / (float)(total - warmup);
    if (progress > 1.0f) progress = 1.0f;
    return base_lr * 0.5f * (1.0f + cosf(3.14159265f * progress));
}