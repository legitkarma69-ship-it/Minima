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

    for (size_t i = 0; i < n; i++) {
        const float g = grad[i];

        // Decoupled weight decay (AdamW).
        param[i] *= 1.0f - lr * weight_decay;

        // Exponential moving averages of gradient and squared gradient.
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

        // Standard bias correction. Keep epsilon outside sqrt, after
        // correcting the second moment.
        const float m_hat = m[i] / bias1;
        const float v_hat = v[i] / bias2;
        param[i] -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

// Small scalar regression test for bias correction and decoupled decay.
bool optim_check_adamw(void) {
    float actual_param = 1.25f;
    float actual_m = 0.0f;
    float actual_v = 0.0f;
    const float gradients[] = { 0.5f, -0.25f, 0.125f };
    const float lr = 0.01f, beta1 = 0.9f, beta2 = 0.999f;
    const float eps = 1e-8f, weight_decay = 0.02f;
    float expected_param = actual_param;
    float expected_m = 0.0f, expected_v = 0.0f;

    for (int step = 1; step <= 3; step++) {
        const float g = gradients[step - 1];
        const float bias1 = 1.0f - powf(beta1, (float)step);
        const float bias2 = 1.0f - powf(beta2, (float)step);

        adamw_update(&actual_param, &actual_m, &actual_v, &g, 1,
            lr, beta1, beta2, eps, weight_decay, bias1, bias2);

        expected_m = beta1 * expected_m + (1.0f - beta1) * g;
        expected_v = beta2 * expected_v + (1.0f - beta2) * g * g;
        expected_param *= 1.0f - lr * weight_decay;
        expected_param -= lr * (expected_m / bias1) /
            (sqrtf(expected_v / bias2) + eps);

        if (!isfinite(actual_param) || !isfinite(actual_m) ||
            !isfinite(actual_v) ||
            fabsf(actual_param - expected_param) > 1e-7f ||
            fabsf(actual_m - expected_m) > 1e-7f ||
            fabsf(actual_v - expected_v) > 1e-7f)
            return false;
    }

    return true;
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
    size_t qkv_n = (size_t)model->n_layers * model->qkv_dim * model->dim;
    size_t proj_n = (size_t)model->n_layers * model->dim * model->dim;
    size_t up_n = (size_t)model->n_layers * model->ffn_dim * model->dim;
    size_t gate_n = (size_t)model->n_layers * model->ffn_dim * model->dim;
    size_t down_n = (size_t)model->n_layers * model->dim * model->ffn_dim;
    size_t rms_n = (size_t)model->n_layers * model->dim;
    size_t final_n = model->dim;

    // Allocate first moments
    opt->m_tok_emb = alloc_moment(arena, tok_n);
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
    opt->v_attn_qkv = alloc_moment(arena, qkv_n);
    opt->v_attn_proj = alloc_moment(arena, proj_n);
    opt->v_ffn_up = alloc_moment(arena, up_n);
    opt->v_ffn_gate = alloc_moment(arena, gate_n);
    opt->v_ffn_down = alloc_moment(arena, down_n);
    opt->v_rms1_scale = alloc_moment(arena, rms_n);
    opt->v_rms2_scale = alloc_moment(arena, rms_n);
    opt->v_rms_final_scale = alloc_moment(arena, final_n);

    opt->n_params = tok_n + qkv_n + proj_n + up_n + gate_n +
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
    size_t qkv_n = (size_t)model->n_layers * model->qkv_dim * model->dim;
    size_t proj_n = (size_t)model->n_layers * model->dim * model->dim;
    size_t up_n = (size_t)model->n_layers * model->ffn_dim * model->dim;
    size_t gate_n = (size_t)model->n_layers * model->ffn_dim * model->dim;
    size_t down_n = (size_t)model->n_layers * model->dim * model->ffn_dim;
    size_t rms_n = (size_t)model->n_layers * model->dim;

    UPDATE(tok_emb, tok_n);
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
float optim_lr_cosine(float base_lr, int64_t step, int64_t warmup, int64_t total) {
    if (!isfinite(base_lr) || base_lr < 0.0f || total <= 0 || step < 0)
        return 0.0f;

    if (step <= 0) return 0.0f;
    if (total == 1) return step == 1 ? base_lr : 0.0f;
    if (warmup < 0) warmup = 0;
    if (warmup >= total) warmup = total - 1;
    if (step >= total) return 0.0f;

    if (warmup > 0 && step < warmup)
        return base_lr * (float)step / (float)warmup;

    float progress = (float)(step - warmup) / (float)(total - warmup);
    if (progress < 0.0f) progress = 0.0f;
    if (progress > 1.0f) progress = 1.0f;
    return base_lr * 0.5f * (1.0f + cosf(3.14159265f * progress));
}

/* A split invocation must produce exactly the same LR at every global step. */
bool optim_check_lr_schedule(void) {
    const float base = 3e-4f;
    const int64_t total = 10000, warmup = 100;
    float uninterrupted[10000];
    for (int64_t step = 1; step <= total; ++step)
        uninterrupted[step - 1] = optim_lr_cosine(base, step, warmup, total);
    for (int64_t split = 1; split < total; split += 137) {
        for (int64_t step = split + 1; step <= total; ++step)
            if (optim_lr_cosine(base, step, warmup, total) != uninterrupted[step - 1])
                return false;
    }
    return uninterrupted[0] > 0.0f &&
        fabsf(uninterrupted[warmup - 1] - base) <= 1e-8f &&
        uninterrupted[total - 1] == 0.0f &&
        optim_lr_cosine(base, total + 1, warmup, total) == 0.0f;
}
