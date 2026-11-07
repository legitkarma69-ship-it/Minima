#include "forward.h"
#include "util.h"
#include <math.h>
#include <string.h>
#include <stddef.h>
#include <limits.h>

// ------------------------------------------------------------------
// Basic kernels
// ------------------------------------------------------------------

void rmsnorm(float* out, const float* x, const float* scale,
    int dim, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++)
        ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / (float)dim + eps);

    for (int i = 0; i < dim; i++)
        out[i] = x[i] * inv_rms * scale[i];
}

void matmul(float* out, const float* x, const float* w,
    int n, int d_in, int d_out) {
    // x: [n, d_in], w: [d_out, d_in], out: [n, d_out]
    /* Each output row is independent; keep the inner accumulation order fixed. */
    int i;
    #pragma omp parallel for schedule(static) if(n >= 32)
    for (i = 0; i < n; i++) {
        for (int j = 0; j < d_out; j++) {
            float sum = 0.0f;
            const float* row = w + (size_t)j * d_in;
            const float* xi = x + (size_t)i * d_in;
            for (int k = 0; k < d_in; k++)
                sum += xi[k] * row[k];
            out[(size_t)i * d_out + j] = sum;
        }
    }
}

static void matmul_add(float* out, const float* x, const float* w,
    int n, int d_in, int d_out) {
    int i;
    #pragma omp parallel for schedule(static) if(n >= 32)
    for (i = 0; i < n; i++) {
        for (int j = 0; j < d_out; j++) {
            float sum = 0.0f;
            const float* row = w + (size_t)j * d_in;
            const float* xi = x + (size_t)i * d_in;
            for (int k = 0; k < d_in; k++)
                sum += xi[k] * row[k];
            out[(size_t)i * d_out + j] += sum;
        }
    }
}

float silu(float x) {
    return x / (1.0f + expf(-x));
}

// ------------------------------------------------------------------
// Causal multi-head attention
// ------------------------------------------------------------------

static void attention(float* out, const float* qkv, float* scores,
    int batch, int seq, int dim, int n_heads, int head_dim) {

    const float scale = 1.0f / sqrtf((float)head_dim);

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < n_heads; h++) {
            // qkv token layout: [Q | K | V], each section has dim values.
            const float* q_base = qkv + (size_t)b * seq * 3 * dim + h * head_dim;
            const float* k_base = q_base + dim;
            const float* v_base = k_base + dim;

            for (int qi = 0; qi < seq; qi++) {
                float max_score = -1.0e30f;

                for (int kj = 0; kj <= qi; kj++) {
                    float dot = 0.0f;
                    const float* q = q_base + (size_t)qi * 3 * dim;
                    const float* k = k_base + (size_t)kj * 3 * dim;
                    for (int d = 0; d < head_dim; d++)
                        dot += q[d] * k[d];
                    scores[kj] = dot * scale;
                    if (scores[kj] > max_score) max_score = scores[kj];
                }

                float sum = 0.0f;
                for (int kj = 0; kj <= qi; kj++) {
                    scores[kj] = expf(scores[kj] - max_score);
                    sum += scores[kj];
                }
                const float inv_sum = 1.0f / sum;
                for (int kj = 0; kj <= qi; kj++)
                    scores[kj] *= inv_sum;

                float* out_h = out + (size_t)b * seq * dim +
                    (size_t)qi * dim + h * head_dim;
                for (int d = 0; d < head_dim; d++)
                    out_h[d] = 0.0f;

                for (int kj = 0; kj <= qi; kj++) {
                    const float* v = v_base + (size_t)kj * 3 * dim;
                    const float weight = scores[kj];
                    for (int d = 0; d < head_dim; d++)
                        out_h[d] += weight * v[d];
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// Activation allocation
// ------------------------------------------------------------------

bool activations_init(Activations* act, Arena* temp_arena,
    int batch, int seq, const Model* model) {

    if (!act || !temp_arena || batch <= 0 || seq <= 0 || batch > INT_MAX / seq ||
        !model_validate(model) || seq > model->max_seq)
        return false;

    memset(act, 0, sizeof(*act));

    const size_t n = (size_t)batch * (size_t)seq;
    const size_t dim = (size_t)model->dim;
    const size_t ffn = (size_t)model->ffn_dim;
    const size_t vocab = (size_t)model->vocab_size;
    const size_t layers = (size_t)model->n_layers;

    act->x = arena_alloc(temp_arena, n * dim * sizeof(float));
    act->x_norm = arena_alloc(temp_arena, n * dim * sizeof(float));

    act->qkv = arena_alloc(temp_arena, layers * n * 3 * dim * sizeof(float));
    act->attn_out = arena_alloc(temp_arena, layers * n * dim * sizeof(float));
    act->ffn_buf = arena_alloc(temp_arena, layers * n * ffn * sizeof(float));
    act->up_buf = arena_alloc(temp_arena, layers * n * ffn * sizeof(float));
    act->gate_buf = arena_alloc(temp_arena, layers * n * ffn * sizeof(float));
    act->resid_attn = arena_alloc(temp_arena, layers * n * dim * sizeof(float));
    act->resid_ffn = arena_alloc(temp_arena, layers * n * dim * sizeof(float));

    act->logits = arena_alloc(temp_arena, n * vocab * sizeof(float));

    act->dx = arena_alloc(temp_arena, n * dim * sizeof(float));
    act->dx_tmp = arena_alloc(temp_arena, n * dim * sizeof(float));
    act->dx_norm = arena_alloc(temp_arena, n * dim * sizeof(float));
    act->dattn = arena_alloc(temp_arena, n * dim * sizeof(float));
    act->dqkv = arena_alloc(temp_arena, n * 3 * dim * sizeof(float));
    act->dffn = arena_alloc(temp_arena, n * ffn * sizeof(float));
    act->dup = arena_alloc(temp_arena, n * ffn * sizeof(float));
    act->dgate = arena_alloc(temp_arena, n * ffn * sizeof(float));
    act->attention_scores = arena_alloc(temp_arena, (size_t)seq * sizeof(float));

    return act->x && act->x_norm && act->qkv && act->attn_out &&
        act->ffn_buf && act->up_buf && act->gate_buf &&
        act->resid_attn && act->resid_ffn && act->logits &&
        act->dx && act->dx_tmp && act->dx_norm && act->dattn &&
        act->dqkv && act->dffn && act->dup && act->dgate &&
        act->attention_scores;
}

// ------------------------------------------------------------------
// Full forward pass
// ------------------------------------------------------------------

float* model_forward(Model* model, Activations* act,
    const uint16_t* tokens, int batch, int seq) {

    if (!model_validate(model) || !act || !tokens ||
        batch <= 0 || seq <= 0 || batch > INT_MAX / seq ||
        seq > model->max_seq ||
        !act->x || !act->x_norm || !act->qkv || !act->attn_out ||
        !act->up_buf || !act->gate_buf || !act->ffn_buf ||
        !act->resid_attn || !act->resid_ffn || !act->logits ||
        !act->attention_scores)
        return NULL;

    const int dim = model->dim;
    const int n_layers = model->n_layers;
    const int n_heads = model->n_heads;
    const int head_dim = model->head_dim;
    const int ffn_dim = model->ffn_dim;
    const int vocab = model->vocab_size;
    const size_t n = (size_t)batch * seq;
    const size_t layer_stride = n;

    // Token + learned position embeddings.
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < seq; t++) {
            const uint16_t id = tokens[b * seq + t];
            if (id >= vocab) return NULL;

            float* dst = act->x + ((size_t)b * seq + t) * dim;
            const float* tok = model->tok_emb + (size_t)id * dim;
            const float* pos = model->pos_emb + (size_t)t * dim;
            for (int d = 0; d < dim; d++)
                dst[d] = tok[d] + pos[d];
        }
    }

    for (int layer = 0; layer < n_layers; layer++) {
        const size_t layer_offset = (size_t)layer * layer_stride;
        float* qkv = act->qkv + layer_offset * 3 * dim;
        float* attn_out = act->attn_out + layer_offset * dim;
        float* up = act->up_buf + layer_offset * ffn_dim;
        float* gate = act->gate_buf + layer_offset * ffn_dim;
        float* ffn_out = act->ffn_buf + layer_offset * ffn_dim;
        float* resid_attn = act->resid_attn + layer_offset * dim;
        float* resid_ffn = act->resid_ffn + layer_offset * dim;

        // Save the exact input to this layer's attention block.
        memcpy(resid_attn, act->x, n * dim * sizeof(float));

        // Attention pre-norm and QKV projection.
        for (size_t i = 0; i < n; i++)
            rmsnorm(act->x_norm + i * dim, resid_attn + i * dim,
                model->rms1_scale + (size_t)layer * dim, dim, 1e-5f);

        const float* w_qkv = model->attn_qkv + (size_t)layer * 3 * dim * dim;
        matmul(qkv, act->x_norm, w_qkv, (int)n, dim, 3 * dim);

        attention(attn_out, qkv, act->attention_scores,
            batch, seq, dim, n_heads, head_dim);

        const float* w_proj = model->attn_proj + (size_t)layer * dim * dim;
        matmul_add(act->x, attn_out, w_proj, (int)n, dim, dim);

        // Save the input to this layer's FFN block (after attention residual).
        memcpy(resid_ffn, act->x, n * dim * sizeof(float));

        // FFN pre-norm and SwiGLU.
        for (size_t i = 0; i < n; i++)
            rmsnorm(act->x_norm + i * dim, resid_ffn + i * dim,
                model->rms2_scale + (size_t)layer * dim, dim, 1e-5f);

        const float* w_up = model->ffn_up + (size_t)layer * ffn_dim * dim;
        const float* w_gate = model->ffn_gate + (size_t)layer * ffn_dim * dim;
        const float* w_down = model->ffn_down + (size_t)layer * dim * ffn_dim;

        matmul(up, act->x_norm, w_up, (int)n, dim, ffn_dim);
        matmul(gate, act->x_norm, w_gate, (int)n, dim, ffn_dim);

        for (size_t i = 0; i < n * (size_t)ffn_dim; i++)
            ffn_out[i] = silu(gate[i]) * up[i];

        matmul_add(act->x, ffn_out, w_down, (int)n, ffn_dim, dim);
    }

    // Final RMSNorm and tied-embedding language-model head.
    for (size_t i = 0; i < n; i++)
        rmsnorm(act->x_norm + i * dim, act->x + i * dim,
            model->rms_final_scale, dim, 1e-5f);

    matmul(act->logits, act->x_norm, model->tok_emb,
        (int)n, dim, vocab);

    return act->logits;
}
