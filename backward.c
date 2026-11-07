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
    if (!g || !arena || !arena->base || arena->used > arena->size ||
        !model_validate(model))
        return false;

    const size_t dim = (size_t)model->dim;
    const size_t layers = (size_t)model->n_layers;
    const size_t ffn = (size_t)model->ffn_dim;
    const size_t expected_params =
        (size_t)model->vocab_size * dim +
        (size_t)model->max_seq * dim +
        layers * 3 * dim * dim +
        layers * dim * dim +
        layers * ffn * dim * 2 +
        layers * dim * ffn +
        layers * dim * 2 + dim;
    if (expected_params > SIZE_MAX / sizeof(float) ||
        model->weight_bytes != expected_params * sizeof(float))
        return false;

    const size_t arena_used_before = arena->used;
    const size_t arena_peak_before = arena->peak;
    memset(g, 0, sizeof(*g));
    g->vocab_size = model->vocab_size;
    g->dim = model->dim;
    g->n_layers = model->n_layers;
    g->n_heads = model->n_heads;
    g->head_dim = model->head_dim;
    g->ffn_dim = model->ffn_dim;
    g->max_seq = model->max_seq;
    g->weight_bytes = model->weight_bytes;

    const size_t tok_n = (size_t)model->vocab_size * dim;
    const size_t pos_n = (size_t)model->max_seq * dim;
    const size_t qkv_n = layers * 3 * dim * dim;
    const size_t proj_n = layers * dim * dim;
    const size_t up_n = layers * ffn * dim;
    const size_t down_n = layers * dim * ffn;
    const size_t rms_n = layers * dim;

    /* Gradient buffers need only zero-filled storage. Calling model_init_config
       here used to generate Gaussian weights and then immediately erase them. */
    g->tok_emb = (float*)arena_alloc_zero(arena, tok_n * sizeof(float));
    g->pos_emb = (float*)arena_alloc_zero(arena, pos_n * sizeof(float));
    g->attn_qkv = (float*)arena_alloc_zero(arena, qkv_n * sizeof(float));
    g->attn_proj = (float*)arena_alloc_zero(arena, proj_n * sizeof(float));
    g->ffn_up = (float*)arena_alloc_zero(arena, up_n * sizeof(float));
    g->ffn_gate = (float*)arena_alloc_zero(arena, up_n * sizeof(float));
    g->ffn_down = (float*)arena_alloc_zero(arena, down_n * sizeof(float));
    g->rms1_scale = (float*)arena_alloc_zero(arena, rms_n * sizeof(float));
    g->rms2_scale = (float*)arena_alloc_zero(arena, rms_n * sizeof(float));
    g->rms_final_scale =
        (float*)arena_alloc_zero(arena, dim * sizeof(float));

    if (!gradients_check_layout(g, model, arena)) {
        arena->used = arena_used_before;
        arena->peak = arena_peak_before;
        memset(g, 0, sizeof(*g));
        return false;
    }
    return true;
}

/*
 * Verify that every gradient tensor has enough contiguous storage for the
 * corresponding runtime model tensor. The arena bound catches an undersized
 * final tensor; adjacent-pointer checks catch undersized earlier tensors.
 */
bool gradients_check_layout(const Gradients* g, const Model* model,
    const Arena* arena) {
    if (!g || !model || !arena || !arena->base ||
        g->vocab_size != model->vocab_size ||
        g->dim != model->dim ||
        g->n_layers != model->n_layers ||
        g->n_heads != model->n_heads ||
        g->head_dim != model->head_dim ||
        g->ffn_dim != model->ffn_dim ||
        g->max_seq != model->max_seq ||
        g->weight_bytes != model->weight_bytes ||
        model->dim <= 0 || model->n_layers <= 0 ||
        model->ffn_dim <= 0 || model->vocab_size <= 0 ||
        model->max_seq <= 0)
        return false;

    float* tensors[] = {
        g->tok_emb, g->pos_emb, g->attn_qkv, g->attn_proj,
        g->ffn_up, g->ffn_gate, g->ffn_down,
        g->rms1_scale, g->rms2_scale, g->rms_final_scale
    };
    const size_t counts[] = {
        (size_t)model->vocab_size * model->dim,
        (size_t)model->max_seq * model->dim,
        (size_t)model->n_layers * 3 * model->dim * model->dim,
        (size_t)model->n_layers * model->dim * model->dim,
        (size_t)model->n_layers * model->ffn_dim * model->dim,
        (size_t)model->n_layers * model->ffn_dim * model->dim,
        (size_t)model->n_layers * model->dim * model->ffn_dim,
        (size_t)model->n_layers * model->dim,
        (size_t)model->n_layers * model->dim,
        (size_t)model->dim
    };

    const uintptr_t arena_begin = (uintptr_t)arena->base;
    if (arena->used > UINTPTR_MAX - arena_begin)
        return false;
    const uintptr_t arena_end = arena_begin + arena->used;

    size_t expected_bytes = 0;
    for (size_t i = 0; i < sizeof(tensors) / sizeof(tensors[0]); ++i) {
        if (!tensors[i] || counts[i] > SIZE_MAX / sizeof(float))
            return false;
        const uintptr_t begin = (uintptr_t)tensors[i];
        const size_t bytes = counts[i] * sizeof(float);
        if (begin < arena_begin || begin > arena_end ||
            bytes > (size_t)(arena_end - begin))
            return false;
        if (i + 1 < sizeof(tensors) / sizeof(tensors[0])) {
            const uintptr_t next = (uintptr_t)tensors[i + 1];
            if (next < begin || bytes > (size_t)(next - begin))
                return false;
        }
        if (expected_bytes > SIZE_MAX - bytes)
            return false;
        expected_bytes += bytes;
    }

    return expected_bytes == g->weight_bytes;
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
    const float* dout_attn, float* scores, int batch, int seq, int dim,
    int n_heads, int head_dim) {

    const float scale = 1.0f / sqrtf((float)head_dim);

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

                /* Recompute the probability derivative from V to avoid a
                   second sequence-sized scratch array. */
                float weighted_sum = 0.0f;
                for (int kj = 0; kj <= qi; kj++) {
                    const float* v = v_base + (size_t)kj * 3 * dim;
                    float value_dot = 0.0f;
                    for (int d = 0; d < head_dim; d++)
                        value_dot += do_h[d] * v[d];
                    weighted_sum += scores[kj] * value_dot;

                    float* dv = dv_base + (size_t)kj * 3 * dim;
                    for (int d = 0; d < head_dim; d++)
                        dv[d] += scores[kj] * do_h[d];
                }

                for (int kj = 0; kj <= qi; kj++) {
                    const float* v = v_base + (size_t)kj * 3 * dim;
                    float value_dot = 0.0f;
                    for (int d = 0; d < head_dim; d++)
                        value_dot += do_h[d] * v[d];
                    const float ds = scores[kj] *
                        (value_dot - weighted_sum) * scale;

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

    if (!model_validate(model) || !grads || !act || !dlogits || !tokens ||
        batch <= 0 || seq <= 0 || seq > model->max_seq ||
        !act->x || !act->x_norm || !act->qkv || !act->attn_out ||
        !act->up_buf || !act->gate_buf || !act->ffn_buf ||
        !act->resid_attn || !act->resid_ffn || !act->dx ||
        !act->dx_tmp || !act->dx_norm || !act->dattn ||
        !act->dqkv || !act->dffn || !act->dup || !act->dgate ||
        !act->attention_scores)
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

        // Recompute this layer's FFN-normalized input. x_norm was reused
        // by later layers and by the final normalization during forward.
        for (size_t i = 0; i < n; i++)
            rmsnorm(act->x_norm + i * dim, x_ffn + i * dim,
                model->rms2_scale + (size_t)layer * dim, dim, 1e-5f);

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
            act->attention_scores, batch, seq, dim, n_heads, head_dim);

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
