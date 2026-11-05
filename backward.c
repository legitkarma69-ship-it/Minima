#include "backward.h"
#include "util.h"
#include <math.h>
#include <string.h>
#include <stddef.h>

// ------------------------------------------------------------------
// Gradient storage
// ------------------------------------------------------------------

void gradients_zero(Gradients* g, const Model* model) {
    const size_t tok_n = (size_t)model->vocab_size * model->dim;
    const size_t pos_n = (size_t)model->max_seq * model->dim;
    const size_t qkv_n = (size_t)model->n_layers * 3 * model->dim * model->dim;
    const size_t proj_n = (size_t)model->n_layers * model->dim * model->dim;
    const size_t up_n = (size_t)model->n_layers * model->ffn_dim * model->dim;
    const size_t down_n = (size_t)model->n_layers * model->dim * model->ffn_dim;
    const size_t rms_n = (size_t)model->n_layers * model->dim;

    memset(g->tok_emb, 0, tok_n * sizeof(float));
    memset(g->pos_emb, 0, pos_n * sizeof(float));
    memset(g->attn_qkv, 0, qkv_n * sizeof(float));
    memset(g->attn_proj, 0, proj_n * sizeof(float));
    memset(g->ffn_up, 0, up_n * sizeof(float));
    memset(g->ffn_gate, 0, up_n * sizeof(float));
    memset(g->ffn_down, 0, down_n * sizeof(float));
    memset(g->rms1_scale, 0, rms_n * sizeof(float));
    memset(g->rms2_scale, 0, rms_n * sizeof(float));
    memset(g->rms_final_scale, 0, (size_t)model->dim * sizeof(float));
}

bool gradients_init(Gradients* g, Arena* arena, const Model* model) {
    (void)model;
    // Model and Gradients intentionally have identical parameter layouts.
    return model_init(g, arena, 0);
}

// ------------------------------------------------------------------
// Cross-entropy
// ------------------------------------------------------------------

float loss_cross_entropy(const float* logits, const uint16_t* targets,
    float* dlogits, int batch, int seq, int vocab) {

    if (!logits || !targets || !dlogits ||
        batch <= 0 || seq <= 0 || vocab <= 0)
        return NAN;

    const int count = batch * seq;
    float loss = 0.0f;

    for (int i = 0; i < count; i++) {
        const float* row = logits + (size_t)i * vocab;
        const int target = targets[i];
        if (target < 0 || target >= vocab) return NAN;

        float max_logit = row[0];
        for (int v = 1; v < vocab; v++)
            if (row[v] > max_logit) max_logit = row[v];

        float sum_exp = 0.0f;
        for (int v = 0; v < vocab; v++)
            sum_exp += expf(row[v] - max_logit);

        loss += logf(sum_exp) - (row[target] - max_logit);

        float* drow = dlogits + (size_t)i * vocab;
        for (int v = 0; v < vocab; v++)
            drow[v] = expf(row[v] - max_logit) / sum_exp;
        drow[target] -= 1.0f;
    }

    const float inv_count = 1.0f / (float)count;
    for (size_t i = 0; i < (size_t)count * vocab; i++)
        dlogits[i] *= inv_count;

    return loss * inv_count;
}

// ------------------------------------------------------------------
// RMSNorm backward over a batch of independent vectors
// ------------------------------------------------------------------

static void rmsnorm_backward_batch(float* dx, float* dscale,
    const float* x, const float* scale, const float* dout,
    size_t rows, int dim, float eps) {

    for (size_t row = 0; row < rows; row++) {
        const float* xr = x + row * dim;
        const float* dr = dout + row * dim;
        float* dxr = dx + row * dim;

        float ss = 0.0f;
        for (int i = 0; i < dim; i++)
            ss += xr[i] * xr[i];

        const float inv_rms = 1.0f / sqrtf(ss / (float)dim + eps);
        float dot = 0.0f;

        for (int i = 0; i < dim; i++) {
            dscale[i] += dr[i] * xr[i] * inv_rms;
            dot += dr[i] * scale[i] * xr[i];
        }

        const float correction = dot / (float)dim;
        const float inv_rms3 = inv_rms * inv_rms * inv_rms;

        for (int i = 0; i < dim; i++) {
            const float direct = dr[i] * scale[i] * inv_rms;
            const float norm_term = xr[i] * inv_rms3 * correction;
            dxr[i] = direct - norm_term;
        }
    }
}

// ------------------------------------------------------------------
// Matrix multiplication backward
// out = x * W^T, W shape [d_out, d_in]
// dx is accumulated into; dw is accumulated into.
// ------------------------------------------------------------------

static void matmul_backward(float* dx, float* dw,
    const float* x, const float* w, const float* dout,
    int n, int d_in, int d_out) {

    for (int i = 0; i < n; i++) {
        for (int k = 0; k < d_in; k++) {
            float sum = 0.0f;
            for (int j = 0; j < d_out; j++)
                sum += dout[(size_t)i * d_out + j] * w[(size_t)j * d_in + k];
            dx[(size_t)i * d_in + k] += sum;
        }
    }

    for (int j = 0; j < d_out; j++) {
        for (int k = 0; k < d_in; k++) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++)
                sum += dout[(size_t)i * d_out + j] * x[(size_t)i * d_in + k];
            dw[(size_t)j * d_in + k] += sum;
        }
    }
}

static float sigmoid(float x) {
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = expf(x);
    return z / (1.0f + z);
}

static float silu_derivative(float x) {
    const float s = sigmoid(x);
    return s * (1.0f + x * (1.0f - s));
}

// ------------------------------------------------------------------
// Causal multi-head attention backward
// ------------------------------------------------------------------

static void attention_backward(float* dqkv, const float* qkv,
    const float* dout_attn, int batch, int seq, int dim,
    int n_heads, int head_dim) {

    const float scale = 1.0f / sqrtf((float)head_dim);
    float scores[512];
    float dscores[512];

    for (int b = 0; b < batch; b++) {
        for (int h = 0; h < n_heads; h++) {
            const float* q_base = qkv + (size_t)b * seq * 3 * dim + h * head_dim;
            const float* k_base = q_base + dim;
            const float* v_base = k_base + dim;

            float* dq_base = dqkv + (size_t)b * seq * 3 * dim + h * head_dim;
            float* dk_base = dq_base + dim;
            float* dv_base = dk_base + dim;

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

                const float* do_h = dout_attn +
                    (size_t)b * seq * dim + (size_t)qi * dim + h * head_dim;

                // Value gradients.
                for (int kj = 0; kj <= qi; kj++) {
                    float* dv = dv_base + (size_t)kj * 3 * dim;
                    for (int d = 0; d < head_dim; d++)
                        dv[d] += scores[kj] * do_h[d];
                }

                // Derivative with respect to attention probabilities.
                for (int kj = 0; kj <= qi; kj++) {
                    const float* v = v_base + (size_t)kj * 3 * dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++)
                        dot += do_h[d] * v[d];
                    dscores[kj] = dot;
                }

                float weighted_sum = 0.0f;
                for (int kj = 0; kj <= qi; kj++)
                    weighted_sum += dscores[kj] * scores[kj];

                // Softmax Jacobian followed by QK^T derivative.
                for (int kj = 0; kj <= qi; kj++) {
                    const float ds = scores[kj] *
                        (dscores[kj] - weighted_sum) * scale;

                    float* dq = dq_base + (size_t)qi * 3 * dim;
                    float* dk = dk_base + (size_t)kj * 3 * dim;
                    const float* q = q_base + (size_t)qi * 3 * dim;
                    const float* k = k_base + (size_t)kj * 3 * dim;

                    for (int d = 0; d < head_dim; d++) {
                        dq[d] += ds * k[d];
                        dk[d] += ds * q[d];
                    }
                }
            }
        }
    }
}

// ------------------------------------------------------------------
// Full backward pass
// ------------------------------------------------------------------

void model_backward(Model* model, Gradients* grads, Activations* act,
    const float* dlogits, const uint16_t* tokens, int batch, int seq) {

    if (!model || !grads || !act || !dlogits || !tokens ||
        batch <= 0 || seq <= 0 || seq > model->max_seq || seq > 512)
        return;

    const int dim = model->dim;
    const int n_layers = model->n_layers;
    const int n_heads = model->n_heads;
    const int head_dim = model->head_dim;
    const int ffn_dim = model->ffn_dim;
    const int vocab = model->vocab_size;
    const size_t n = (size_t)batch * seq;
    const size_t dim_bytes = n * dim * sizeof(float);
    const size_t ffn_count = n * (size_t)ffn_dim;

    float* dx = act->dx;
    float* dx_tmp = act->dx_tmp;
    float* dx_norm = act->dx_norm;

    // Final language-model head. The output matrix is the token embedding
    // matrix transposed, so both paths contribute to tok_emb gradients.
    memset(dx_norm, 0, dim_bytes);
    for (size_t i = 0; i < n; i++) {
        const float* dl = dlogits + i * vocab;
        const float* normalized = act->x_norm + i * dim;

        for (int v = 0; v < vocab; v++) {
            const float g = dl[v];
            if (g == 0.0f) continue;

            float* embedding_grad = grads->tok_emb + (size_t)v * dim;
            const float* embedding = model->tok_emb + (size_t)v * dim;

            for (int d = 0; d < dim; d++) {
                dx_norm[i * dim + d] += g * embedding[d];
                embedding_grad[d] += g * normalized[d];
            }
        }
    }

    // Final RMSNorm.
    memset(dx, 0, dim_bytes);
    rmsnorm_backward_batch(dx, grads->rms_final_scale,
        act->x, model->rms_final_scale, dx_norm, n, dim, 1e-5f);

    // Reverse transformer layers.
    for (int layer = n_layers - 1; layer >= 0; layer--) {
        const size_t layer_offset = (size_t)layer * n;

        const float* x_ffn = act->resid_ffn + layer_offset * dim;
        const float* up = act->up_buf + layer_offset * ffn_dim;
        const float* gate = act->gate_buf + layer_offset * ffn_dim;
        const float* ffn_out = act->ffn_buf + layer_offset * ffn_dim;

        const float* w_down = model->ffn_down +
            (size_t)layer * dim * ffn_dim;
        const float* w_up = model->ffn_up +
            (size_t)layer * ffn_dim * dim;
        const float* w_gate = model->ffn_gate +
            (size_t)layer * ffn_dim * dim;

        // FFN down projection: d(ffn_out) = d(output) * W_down.
        memset(act->dffn, 0, ffn_count * sizeof(float));
        matmul_backward(act->dffn,
            grads->ffn_down + (size_t)layer * dim * ffn_dim,
            ffn_out, w_down, dx, (int)n, ffn_dim, dim);

        // SwiGLU: ffn_out = up * SiLU(gate).
        for (size_t i = 0; i < ffn_count; i++) {
            const float s = silu(gate[i]);
            act->dup[i] = act->dffn[i] * s;
            act->dgate[i] = act->dffn[i] * up[i] *
                silu_derivative(gate[i]);
        }

        // Backpropagate both FFN projections into the normalized input.
        memset(dx_norm, 0, dim_bytes);
        matmul_backward(dx_norm,
            grads->ffn_up + (size_t)layer * ffn_dim * dim,
            act->x_norm, w_up, act->dup, (int)n, dim, ffn_dim);
        matmul_backward(dx_norm,
            grads->ffn_gate + (size_t)layer * ffn_dim * dim,
            act->x_norm, w_gate, act->dgate, (int)n, dim, ffn_dim);

        // RMSNorm-2 backward, then add the FFN residual identity gradient.
        rmsnorm_backward_batch(dx_tmp,
            grads->rms2_scale + (size_t)layer * dim,
            x_ffn, model->rms2_scale + (size_t)layer * dim,
            dx_norm, n, dim, 1e-5f);
        for (size_t i = 0; i < n * (size_t)dim; i++)
            dx_tmp[i] += dx[i];

        // Swap the two working gradient buffers.
        {
            float* swap = dx;
            dx = dx_tmp;
            dx_tmp = swap;
        }

        // Attention block.
        const float* x_attn = act->resid_attn + layer_offset * dim;
        const float* qkv = act->qkv + layer_offset * 3 * dim;
        const float* attn_out = act->attn_out + layer_offset * dim;
        const float* w_proj = model->attn_proj +
            (size_t)layer * dim * dim;
        const float* w_qkv = model->attn_qkv +
            (size_t)layer * 3 * dim * dim;

        // Recompute the normalized attention input.
        for (size_t i = 0; i < n; i++)
            rmsnorm(act->x_norm + i * dim, x_attn + i * dim,
                model->rms1_scale + (size_t)layer * dim, dim, 1e-5f);

        // Output projection backward.
        memset(act->dattn, 0, dim_bytes);
        matmul_backward(act->dattn,
            grads->attn_proj + (size_t)layer * dim * dim,
            attn_out, w_proj, dx, (int)n, dim, dim);

        // Attention backward uses the preserved forward QKV cache.
        memset(act->dqkv, 0, n * 3 * (size_t)dim * sizeof(float));
        attention_backward(act->dqkv, qkv, act->dattn,
            batch, seq, dim, n_heads, head_dim);

        // QKV projection backward.
        memset(dx_norm, 0, dim_bytes);
        matmul_backward(dx_norm,
            grads->attn_qkv + (size_t)layer * 3 * dim * dim,
            act->x_norm, w_qkv, act->dqkv, (int)n, dim, 3 * dim);

        // RMSNorm-1 backward, then add the attention residual gradient.
        rmsnorm_backward_batch(dx_tmp,
            grads->rms1_scale + (size_t)layer * dim,
            x_attn, model->rms1_scale + (size_t)layer * dim,
            dx_norm, n, dim, 1e-5f);
        for (size_t i = 0; i < n * (size_t)dim; i++)
            dx_tmp[i] += dx[i];

        {
            float* swap = dx;
            dx = dx_tmp;
            dx_tmp = swap;
        }
    }

    // Embedding and learned-position gradients.
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < seq; t++) {
            const uint16_t id = tokens[b * seq + t];
            if (id >= vocab) continue;

            const float* row_grad = dx + ((size_t)b * seq + t) * dim;
            float* token_grad = grads->tok_emb + (size_t)id * dim;
            float* position_grad = grads->pos_emb + (size_t)t * dim;

            for (int d = 0; d < dim; d++) {
                token_grad[d] += row_grad[d];
                position_grad[d] += row_grad[d];
            }
        }
    }
}
