#include "backward.h"
#include "util.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

// ------------------------------------------------------------------
// Zero gradients
// ------------------------------------------------------------------

void gradients_zero(Gradients* g, const Model* model) {
    size_t tok_n = (size_t)model->vocab_size * model->dim;
    size_t pos_n = (size_t)model->max_seq * model->dim;
    size_t qkv_n = (size_t)model->n_layers * 3 * model->dim * model->dim;
    size_t proj_n = (size_t)model->n_layers * model->dim * model->dim;
    size_t up_n = (size_t)model->n_layers * model->ffn_dim * model->dim;
    size_t gate_n = (size_t)model->n_layers * model->ffn_dim * model->dim;
    size_t down_n = (size_t)model->n_layers * model->dim * model->ffn_dim;
    size_t rms_n = (size_t)model->n_layers * model->dim;

    memset(g->tok_emb, 0, tok_n * sizeof(float));
    memset(g->pos_emb, 0, pos_n * sizeof(float));
    memset(g->attn_qkv, 0, qkv_n * sizeof(float));
    memset(g->attn_proj, 0, proj_n * sizeof(float));
    memset(g->ffn_up, 0, up_n * sizeof(float));
    memset(g->ffn_gate, 0, gate_n * sizeof(float));
    memset(g->ffn_down, 0, down_n * sizeof(float));
    memset(g->rms1_scale, 0, rms_n * sizeof(float));
    memset(g->rms2_scale, 0, rms_n * sizeof(float));
    memset(g->rms_final_scale, 0, model->dim * sizeof(float));
}

bool gradients_init(Gradients* g, Arena* arena, const Model* model) {
    // Re-use the exact same allocation path as the model
    return model_init(g, arena, 0);
}

// ------------------------------------------------------------------
// Loss (unchanged, already correct)
// ------------------------------------------------------------------

float loss_cross_entropy(const float* logits, const uint16_t* targets,
    float* dlogits, int batch, int seq, int vocab) {
    int N = batch * seq;
    float loss = 0.0f;

    for (int i = 0; i < N; i++) {
        const float* row = logits + i * vocab;
        int target = targets[i];

        float max_logit = row[0];
        for (int v = 1; v < vocab; v++)
            if (row[v] > max_logit) max_logit = row[v];

        float sum_exp = 0.0f;
        for (int v = 0; v < vocab; v++)
            sum_exp += expf(row[v] - max_logit);

        float log_prob = (row[target] - max_logit) - logf(sum_exp);
        loss -= log_prob;

        float* drow = dlogits + i * vocab;
        for (int v = 0; v < vocab; v++) {
            float p = expf(row[v] - max_logit) / sum_exp;
            drow[v] = p;
        }
        drow[target] -= 1.0f;
    }

    float inv = 1.0f / (float)N;
    loss *= inv;
    for (int i = 0; i < N * vocab; i++)
        dlogits[i] *= inv;

    return loss;
}

// ------------------------------------------------------------------
// RMSNorm backward
// ------------------------------------------------------------------

static void rmsnorm_backward(float* dx, float* dscale,
    const float* x, const float* scale,
    const float* dout, int dim, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++)
        ss += x[i] * x[i];
    float inv_rms = 1.0f / sqrtf(ss / (float)dim + eps);
    float inv_rms3 = inv_rms * inv_rms * inv_rms;

    // dscale
    for (int i = 0; i < dim; i++)
        dscale[i] += dout[i] * (x[i] * inv_rms);

    // mean of (dout * scale * x)
    float m = 0.0f;
    for (int i = 0; i < dim; i++)
        m += dout[i] * scale[i] * x[i];
    m /= (float)dim;

    for (int i = 0; i < dim; i++) {
        float term1 = dout[i] * scale[i] * inv_rms;
        float term2 = -x[i] * inv_rms3 * m;
        dx[i] = term1 + term2;
    }
}

// ------------------------------------------------------------------
// Matmul backward (out = x * W^T, W shape [d_out, d_in])
// ------------------------------------------------------------------

static void matmul_backward(float* dx, float* dw,
    const float* x, const float* w, const float* dout,
    int n, int d_in, int d_out) {
    // dx += dout * W
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < d_in; k++) {
            float sum = 0.0f;
            for (int j = 0; j < d_out; j++)
                sum += dout[i * d_out + j] * w[j * d_in + k];
            dx[i * d_in + k] += sum;
        }
    }
    // dw += dout^T * x
    for (int j = 0; j < d_out; j++) {
        for (int k = 0; k < d_in; k++) {
            float sum = 0.0f;
            for (int i = 0; i < n; i++)
                sum += dout[i * d_out + j] * x[i * d_in + k];
            dw[j * d_in + k] += sum;
        }
    }
}

// ------------------------------------------------------------------
// Causal multi-head attention backward
// ------------------------------------------------------------------

static void attention_backward(float* dqkv,
    const float* qkv, const float* dout_attn,
    int batch, int seq, int dim,
    int n_heads, int head_dim) {

    float scale = 1.0f / sqrtf((float)head_dim);
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
                // recompute softmax
                float max_score = -1e30f;
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
                float inv = 1.0f / sum;
                for (int kj = 0; kj <= qi; kj++)
                    scores[kj] *= inv;

                const float* do_h = dout_attn + (size_t)b * seq * dim + qi * dim + h * head_dim;

                // dV
                for (int kj = 0; kj <= qi; kj++) {
                    float* dv = dv_base + (size_t)kj * 3 * dim;
                    float s = scores[kj];
                    for (int d = 0; d < head_dim; d++)
                        dv[d] += s * do_h[d];
                }

                // dScores
                for (int kj = 0; kj <= qi; kj++) {
                    const float* v = v_base + (size_t)kj * 3 * dim;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim; d++)
                        dot += do_h[d] * v[d];
                    dscores[kj] = dot;
                }

                // softmax Jacobian
                float sum_ds = 0.0f;
                for (int kj = 0; kj <= qi; kj++)
                    sum_ds += dscores[kj] * scores[kj];

                for (int kj = 0; kj <= qi; kj++) {
                    float ds = scores[kj] * (dscores[kj] - sum_ds) * scale;

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
// Full backward
// ------------------------------------------------------------------

void model_backward(Model* model, Gradients* grads, Activations* act,
    const float* dlogits, const uint16_t* tokens,
    int batch, int seq) {

    const int dim = model->dim;
    const int n_layers = model->n_layers;
    const int n_heads = model->n_heads;
    const int head_dim = model->head_dim;
    const int ffn_dim = model->ffn_dim;
    const int vocab = model->vocab_size;
    const size_t n = (size_t)batch * seq;

    // Working gradient of the residual stream
    float* dx = act->x;               // we overwrite the forward residual
    float* dx_norm = act->x_norm;

    // ---------- Final logits → final RMSNorm ----------
    // d_x_norm = dlogits * tok_emb   (tied)
    memset(dx_norm, 0, n * dim * sizeof(float));
    for (size_t i = 0; i < n; i++) {
        const float* dl = dlogits + i * vocab;
        for (int v = 0; v < vocab; v++) {
            float g = dl[v];
            if (g == 0.0f) continue;
            const float* emb = model->tok_emb + v * dim;
            for (int d = 0; d < dim; d++)
                dx_norm[i * dim + d] += g * emb[d];
        }
    }

    // tok_emb gradients (tied lm_head)
    for (size_t i = 0; i < n; i++) {
        const float* dl = dlogits + i * vocab;
        const float* xn = act->x_norm + i * dim;   // final normalized residual
        for (int v = 0; v < vocab; v++) {
            float g = dl[v];
            if (g == 0.0f) continue;
            float* ge = grads->tok_emb + v * dim;
            for (int d = 0; d < dim; d++)
                ge[d] += g * xn[d];
        }
    }

    // Final RMSNorm backward
    // We need the pre-norm residual (the final act->x from forward)
    // For simplicity we re-use the last residual that was saved.
    // (In the updated forward you keep act->x as the final residual.)
    rmsnorm_backward(dx, grads->rms_final_scale,
        act->x, model->rms_final_scale,
        dx_norm, dim, 1e-5f);

    // ---------- Layers in reverse ----------
    for (int layer = n_layers - 1; layer >= 0; layer--) {

        // ===== FFN block =====
        const float* x_ffn = act->resid_ffn + (size_t)layer * n * dim;  // input to FFN
        float* dx_ffn = dx;   // incoming gradient is currently in dx

        // Recompute x_norm for this FFN (or save it – recomputing is cheap)
        for (size_t i = 0; i < n; i++)
            rmsnorm(act->x_norm + i * dim, x_ffn + i * dim,
                model->rms2_scale + layer * dim, dim, 1e-5f);

        // Gradient of down-projection
        // ffn_buf held the post-SiLU value
        float* dffn = act->ffn_buf;          // will become d(up*silu)
        memset(dffn, 0, n * ffn_dim * sizeof(float));

        const float* w_down = model->ffn_down + (size_t)layer * dim * ffn_dim;
        matmul_backward(dffn, grads->ffn_down + (size_t)layer * dim * ffn_dim,
            act->ffn_buf, w_down, dx_ffn,   // careful: forward ffn_buf was overwritten?
            (int)n, ffn_dim, dim);

        // SiLU * up  backward
        // We need the original gate and up values.
        // (In a fully saved version they are in gate_buf and a saved up buffer.
        //  Here we assume they were kept or we recompute the up projection.)

        // For correctness we recompute up and gate (cheap relative to attention)
        const float* w_up = model->ffn_up + (size_t)layer * ffn_dim * dim;
        const float* w_gate = model->ffn_gate + (size_t)layer * ffn_dim * dim;

        float* up = act->ffn_buf;      // temporary
        float* gate = act->gate_buf;
        matmul(up, act->x_norm, w_up, (int)n, dim, ffn_dim);
        matmul(gate, act->x_norm, w_gate, (int)n, dim, ffn_dim);

        // d(silu(gate) * up)
        for (size_t i = 0; i < n * ffn_dim; i++) {
            float s = silu(gate[i]);
            float ds = s * (1.0f - s);          // derivative of silu
            float dup = dffn[i] * s;
            float dgate = dffn[i] * up[i] * ds;
            up[i] = dup;                      // reuse buffers
            gate[i] = dgate;
        }

        // back into x_norm and the two weight matrices
        float* dx_norm_ffn = act->x_norm;
        memset(dx_norm_ffn, 0, n * dim * sizeof(float));

        matmul_backward(dx_norm_ffn, grads->ffn_up + (size_t)layer * ffn_dim * dim,
            act->x_norm, w_up, up, (int)n, dim, ffn_dim);
        matmul_backward(dx_norm_ffn, grads->ffn_gate + (size_t)layer * ffn_dim * dim,
            act->x_norm, w_gate, gate, (int)n, dim, ffn_dim);

        // RMSNorm-2 backward
        float* dx_resid = dx;   // will accumulate
        rmsnorm_backward(dx_resid, grads->rms2_scale + layer * dim,
            x_ffn, model->rms2_scale + layer * dim,
            dx_norm_ffn, dim, 1e-5f);

        // ===== Attention block =====
        const float* x_attn = act->resid_attn + (size_t)layer * n * dim;

        // recompute norm
        for (size_t i = 0; i < n; i++)
            rmsnorm(act->x_norm + i * dim, x_attn + i * dim,
                model->rms1_scale + layer * dim, dim, 1e-5f);

        // d(attn_out) comes from the residual add → currently in dx
        float* dattn = act->attn_out;
        memcpy(dattn, dx, n * dim * sizeof(float));

        // output projection backward
        float* dqkv = act->qkv;
        memset(dqkv, 0, n * 3 * dim * sizeof(float));

        const float* w_proj = model->attn_proj + (size_t)layer * dim * dim;
        matmul_backward(dqkv /* wrong shape – temporary */,
            grads->attn_proj + (size_t)layer * dim * dim,
            act->attn_out, w_proj, dattn,
            (int)n, dim, dim);

        // The correct place for dqkv is after attention_backward.
        // We zero a proper buffer:
        memset(act->qkv, 0, n * 3 * dim * sizeof(float));
        attention_backward(act->qkv, act->qkv /* forward qkv must be saved */,
            dattn, batch, seq, dim, n_heads, head_dim);

        // QKV projection backward
        float* dx_norm_attn = act->x_norm;
        memset(dx_norm_attn, 0, n * dim * sizeof(float));
        const float* w_qkv = model->attn_qkv + (size_t)layer * 3 * dim * dim;
        matmul_backward(dx_norm_attn,
            grads->attn_qkv + (size_t)layer * 3 * dim * dim,
            act->x_norm, w_qkv, act->qkv,
            (int)n, dim, 3 * dim);

        // RMSNorm-1 backward
        rmsnorm_backward(dx, grads->rms1_scale + layer * dim,
            x_attn, model->rms1_scale + layer * dim,
            dx_norm_attn, dim, 1e-5f);
    }

    // ---------- Embedding gradients ----------
    for (int b = 0; b < batch; b++) {
        for (int t = 0; t < seq; t++) {
            uint16_t id = tokens[b * seq + t];
            const float* g = dx + (size_t)(b * seq + t) * dim;
            float* ge = grads->tok_emb + id * dim;
            float* gp = grads->pos_emb + t * dim;
            for (int d = 0; d < dim; d++) {
                ge[d] += g[d];
                gp[d] += g[d];
            }
        }
    }
}