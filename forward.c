#include "forward.h"
#include "util.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

// ------------------------------------------------------------------
// Basic kernels (scalar for clarity – replace with SIMD later)
// ------------------------------------------------------------------

void rmsnorm(float* out, const float* x, const float* scale,
    int dim, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++)
        ss += x[i] * x[i];
    ss = 1.0f / sqrtf(ss / dim + eps);

    for (int i = 0; i < dim; i++)
        out[i] = x[i] * ss * scale[i];
}

void matmul(float* out, const float* x, const float* w,
    int n, int d_in, int d_out) {
    // x: [n, d_in]   w: [d_out, d_in]   out: [n, d_out]
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d_out; j++) {
            float sum = 0.0f;
            const float* row = w + j * d_in;
            const float* xi = x + i * d_in;
            for (int k = 0; k < d_in; k++)
                sum += xi[k] * row[k];
            out[i * d_out + j] = sum;
        }
    }
}

static void matmul_add(float* out, const float* x, const float* w,
    int n, int d_in, int d_out) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d_out; j++) {
            float sum = 0.0f;
            const float* row = w + j * d_in;
            const float* xi = x + i * d_in;
            for (int k = 0; k < d_in; k++)
                sum += xi[k] * row[k];
            out[i * d_out + j] += sum;
        }
    }
}

float silu(float x) {
    return x / (1.0f + expf(-x));
}

// ------------------------------------------------------------------
// Attention (causal, multi-head)
// ------------------------------------------------------------------

static void attention(float* out, const float* qkv,
    int batch, int seq, int dim, int n_heads, int head_dim) {

    // qkv is [batch, seq, 3*dim]
    // We compute attention per head and write into out [batch, seq, dim]

    float scale = 1.0f / sqrtf((float)head_dim);

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < n_heads; h++) {
            // pointers for this head
            // Q/K/V layout inside qkv: [Q | K | V] each of size dim
            const float* q_base = qkv + b * seq * 3 * dim + h * head_dim;
            const float* k_base = q_base + dim;
            const float* v_base = k_base + dim;

            for (int qi = 0; qi < seq; qi++) {
                // compute attention scores for query qi against all keys 0..qi
                float scores[512];          // MODEL_MAX_SEQ is 256, safe
                float max_score = -1e30f;

                for (int kj = 0; kj <= qi; kj++) {
                    float dot = 0.0f;
                    const float* q = q_base + qi * 3 * dim;
                    const float* k = k_base + kj * 3 * dim;
                    for (int d = 0; d < head_dim; d++)
                        dot += q[d] * k[d];
                    scores[kj] = dot * scale;
                    if (scores[kj] > max_score) max_score = scores[kj];
                }

                // softmax
                float sum = 0.0f;
                for (int kj = 0; kj <= qi; kj++) {
                    scores[kj] = expf(scores[kj] - max_score);
                    sum += scores[kj];
                }
                float inv_sum = 1.0f / sum;
                for (int kj = 0; kj <= qi; kj++)
                    scores[kj] *= inv_sum;

                // weighted sum of values
                float* out_h = out + b * seq * dim + qi * dim + h * head_dim;
                for (int d = 0; d < head_dim; d++)
                    out_h[d] = 0.0f;

                for (int kj = 0; kj <= qi; kj++) {
                    const float* v = v_base + kj * 3 * dim;
                    float s = scores[kj];
                    for (int d = 0; d < head_dim; d++)
                        out_h[d] += s * v[d];
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// Activations allocation
// ------------------------------------------------------------------

bool activations_init(Activations* act, Arena* temp_arena,
    int batch, int seq, const Model* model) {
    size_t n = (size_t)batch * seq;
    size_t dim = model->dim;
    size_t ffn = model->ffn_dim;
    size_t vocab = model->vocab_size;

    act->x = arena_alloc(temp_arena, n * dim * sizeof(float));
    act->x_norm = arena_alloc(temp_arena, n * dim * sizeof(float));
    act->qkv = arena_alloc(temp_arena, n * 3 * dim * sizeof(float));
    act->attn_out = arena_alloc(temp_arena, n * dim * sizeof(float));
    act->ffn_buf = arena_alloc(temp_arena, n * ffn * sizeof(float));
    act->logits = arena_alloc(temp_arena, n * vocab * sizeof(float));

    if (!act->x || !act->x_norm || !act->qkv || !act->attn_out ||
        !act->ffn_buf || !act->logits)
        return false;
    return true;
}

// ------------------------------------------------------------------
// Full forward
// ------------------------------------------------------------------

float* model_forward(Model* model, Activations* act,
    const uint16_t* tokens, int batch, int seq) {

    int dim = model->dim;
    int n_layers = model->n_layers;
    int n_heads = model->n_heads;
    int head_dim = model->head_dim;
    int ffn_dim = model->ffn_dim;
    int vocab = model->vocab_size;
    size_t n = (size_t)batch * seq;

    // 1. Token + position embedding
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < seq; t++) {
            uint16_t id = tokens[b * seq + t];
            float* dst = act->x + (b * seq + t) * dim;
            const float* tok = model->tok_emb + id * dim;
            const float* pos = model->pos_emb + t * dim;
            for (int d = 0; d < dim; d++)
                dst[d] = tok[d] + pos[d];
        }
    }

    // 2. Transformer layers
    for (int layer = 0; layer < n_layers; layer++) {
        // ----- Attention block -----
        // RMSNorm
        for (size_t i = 0; i < n; i++)
            rmsnorm(act->x_norm + i * dim,
                act->x + i * dim,
                model->rms1_scale + layer * dim,
                dim, 1e-5f);

        // QKV projection
        const float* w_qkv = model->attn_qkv + layer * 3 * dim * dim;
        matmul(act->qkv, act->x_norm, w_qkv, (int)n, dim, 3 * dim);

        // Attention
        memset(act->attn_out, 0, n * dim * sizeof(float));
        attention(act->attn_out, act->qkv, batch, seq, dim, n_heads, head_dim);

        // Output projection + residual
        const float* w_proj = model->attn_proj + layer * dim * dim;
        matmul_add(act->x, act->attn_out, w_proj, (int)n, dim, dim);

        // ----- FFN block -----
        for (size_t i = 0; i < n; i++)
            rmsnorm(act->x_norm + i * dim,
                act->x + i * dim,
                model->rms2_scale + layer * dim,
                dim, 1e-5f);

        // SwiGLU: up and gate
        const float* w_up = model->ffn_up + layer * ffn_dim * dim;
        const float* w_gate = model->ffn_gate + layer * ffn_dim * dim;
        const float* w_down = model->ffn_down + layer * dim * ffn_dim;

        // up
        matmul(act->ffn_buf, act->x_norm, w_up, (int)n, dim, ffn_dim);

        // gate + silu * up  (in-place on ffn_buf)
        // We need a temporary for the gate; reuse logits buffer carefully
        // or allocate a second ffn buffer if you prefer. For simplicity:
        float* gate = act->logits;   // temporary reuse (logits not needed yet)
        matmul(gate, act->x_norm, w_gate, (int)n, dim, ffn_dim);

        for (size_t i = 0; i < n * ffn_dim; i++)
            act->ffn_buf[i] = silu(gate[i]) * act->ffn_buf[i];

        // down + residual
        matmul_add(act->x, act->ffn_buf, w_down, (int)n, ffn_dim, dim);
    }

    // 3. Final RMSNorm
    for (size_t i = 0; i < n; i++)
        rmsnorm(act->x_norm + i * dim,
            act->x + i * dim,
            model->rms_final_scale,
            dim, 1e-5f);

    // 4. Logits (weight-tied)
    // lm_head is tok_emb^T  →  matmul with tok_emb as [vocab, dim]
    matmul(act->logits, act->x_norm, model->tok_emb, (int)n, dim, vocab);

    return act->logits;
}