#include "generate.h"
#include "util.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

// ------------------------------------------------------------------
// KV cache
// ------------------------------------------------------------------

bool kvcache_init(KVCache* cache, Arena* arena, const Model* model, int max_seq) {
    memset(cache, 0, sizeof(*cache));
    cache->max_seq = max_seq;
    cache->n_layers = model->n_layers;
    cache->n_heads = model->n_heads;
    cache->head_dim = model->head_dim;
    cache->dim = model->dim;

    size_t bytes = (size_t)model->n_layers * max_seq * model->dim * sizeof(float);
    cache->k_cache = arena_alloc(arena, bytes);
    cache->v_cache = arena_alloc(arena, bytes);
    if (!cache->k_cache || !cache->v_cache) return false;

    cache->pos = 0;
    return true;
}

void kvcache_reset(KVCache* cache) {
    cache->pos = 0;
}

// ------------------------------------------------------------------
// Single-token forward with KV cache (much cheaper than full forward)
// ------------------------------------------------------------------

static void rmsnorm_inplace(float* x, const float* scale, int dim, float eps) {
    float ss = 0.0f;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / dim + eps);
    for (int i = 0; i < dim; i++)
        x[i] = x[i] * inv * scale[i];
}

static void matvec(float* out, const float* x, const float* w, int d_in, int d_out) {
    for (int j = 0; j < d_out; j++) {
        float sum = 0.0f;
        const float* row = w + j * d_in;
        for (int k = 0; k < d_in; k++)
            sum += x[k] * row[k];
        out[j] = sum;
    }
}

// Forward one token, updating the KV cache. Returns logits for the new token.
static void forward_one(Model* model, KVCache* cache,
    uint16_t token_id, float* x, float* logits) {

    int dim = model->dim;
    int n_heads = model->n_heads;
    int head_dim = model->head_dim;
    int ffn_dim = model->ffn_dim;
    int pos = cache->pos;

    // Token + position embedding
    const float* tok = model->tok_emb + token_id * dim;
    const float* pos_emb = model->pos_emb + pos * dim;
    for (int d = 0; d < dim; d++)
        x[d] = tok[d] + pos_emb[d];

    // Temporary buffers (stack is fine for this tiny model)
    float x_norm[512];
    float qkv[3 * 512];
    float attn_out[512];
    float ffn_buf[1024];
    float gate_buf[1024];

    for (int layer = 0; layer < model->n_layers; layer++) {
        // ----- Attention -----
        memcpy(x_norm, x, dim * sizeof(float));
        rmsnorm_inplace(x_norm, model->rms1_scale + layer * dim, dim, 1e-5f);

        // QKV
        const float* w_qkv = model->attn_qkv + (size_t)layer * 3 * dim * dim;
        matvec(qkv, x_norm, w_qkv, dim, 3 * dim);

        // Split Q, K, V and write K/V into cache
        float* k_cache_l = cache->k_cache + (size_t)layer * cache->max_seq * dim;
        float* v_cache_l = cache->v_cache + (size_t)layer * cache->max_seq * dim;

        // store current K and V
        memcpy(k_cache_l + pos * dim, qkv + dim, dim * sizeof(float));
        memcpy(v_cache_l + pos * dim, qkv + 2 * dim, dim * sizeof(float));

        // Attention (only the new query against all keys 0..pos)
        memset(attn_out, 0, dim * sizeof(float));
        float scale = 1.0f / sqrtf((float)head_dim);

        for (int h = 0; h < n_heads; h++) {
            const float* q = qkv + h * head_dim;
            float scores[512];
            float max_score = -1e30f;

            for (int kj = 0; kj <= pos; kj++) {
                const float* k = k_cache_l + kj * dim + h * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++)
                    dot += q[d] * k[d];
                scores[kj] = dot * scale;
                if (scores[kj] > max_score) max_score = scores[kj];
            }

            float sum = 0.0f;
            for (int kj = 0; kj <= pos; kj++) {
                scores[kj] = expf(scores[kj] - max_score);
                sum += scores[kj];
            }
            float inv = 1.0f / sum;
            for (int kj = 0; kj <= pos; kj++)
                scores[kj] *= inv;

            float* out_h = attn_out + h * head_dim;
            for (int kj = 0; kj <= pos; kj++) {
                const float* v = v_cache_l + kj * dim + h * head_dim;
                float s = scores[kj];
                for (int d = 0; d < head_dim; d++)
                    out_h[d] += s * v[d];
            }
        }

        // Output projection + residual
        const float* w_proj = model->attn_proj + (size_t)layer * dim * dim;
        float proj[512];
        matvec(proj, attn_out, w_proj, dim, dim);
        for (int d = 0; d < dim; d++)
            x[d] += proj[d];

        // ----- FFN -----
        memcpy(x_norm, x, dim * sizeof(float));
        rmsnorm_inplace(x_norm, model->rms2_scale + layer * dim, dim, 1e-5f);

        const float* w_up = model->ffn_up + (size_t)layer * ffn_dim * dim;
        const float* w_gate = model->ffn_gate + (size_t)layer * ffn_dim * dim;
        const float* w_down = model->ffn_down + (size_t)layer * dim * ffn_dim;

        matvec(ffn_buf, x_norm, w_up, dim, ffn_dim);
        matvec(gate_buf, x_norm, w_gate, dim, ffn_dim);

        for (int i = 0; i < ffn_dim; i++) {
            float s = gate_buf[i] / (1.0f + expf(-gate_buf[i])); // silu
            ffn_buf[i] = s * ffn_buf[i];
        }

        float down[512];
        matvec(down, ffn_buf, w_down, ffn_dim, dim);
        for (int d = 0; d < dim; d++)
            x[d] += down[d];
    }

    // Final norm
    rmsnorm_inplace(x, model->rms_final_scale, dim, 1e-5f);

    // Logits (tied)
    matvec(logits, x, model->tok_emb, dim, model->vocab_size);

    cache->pos++;
}

// ------------------------------------------------------------------
// Sampling helpers
// ------------------------------------------------------------------

static int sample_top_k(const float* logits, int vocab, float temperature,
    int top_k, Rng* rng) {
    // Find top-k
    int idx[256];          // enough for our small smoke-test vocab
    float vals[256];
    int n = 0;

    for (int i = 0; i < vocab; i++) {
        float v = logits[i] / temperature;
        if (n < top_k) {
            idx[n] = i;
            vals[n] = v;
            n++;
        }
        else {
            // replace the current minimum
            int min_i = 0;
            for (int j = 1; j < top_k; j++)
                if (vals[j] < vals[min_i]) min_i = j;
            if (v > vals[min_i]) {
                vals[min_i] = v;
                idx[min_i] = i;
            }
        }
    }

    // Softmax over the top-k
    float max_v = vals[0];
    for (int i = 1; i < n; i++)
        if (vals[i] > max_v) max_v = vals[i];

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        vals[i] = expf(vals[i] - max_v);
        sum += vals[i];
    }

    float r = rng_f32(rng) * sum;
    for (int i = 0; i < n; i++) {
        r -= vals[i];
        if (r <= 0.0f) return idx[i];
    }
    return idx[n - 1];
}

// ------------------------------------------------------------------
// Public generate
// ------------------------------------------------------------------

int generate(Model* model, Tokenizer* tok, KVCache* cache,
    const uint16_t* prompt_ids, int prompt_len,
    uint16_t* out_ids, int out_cap,
    int n_new,
    float temperature, int top_k,
    Rng* rng) {

    if (prompt_len + n_new > out_cap) n_new = out_cap - prompt_len;
    if (n_new <= 0) return prompt_len;

    kvcache_reset(cache);

    float x[512];
    float logits[8192];   // safe upper bound

    // Prefill the prompt
    for (int i = 0; i < prompt_len; i++) {
        out_ids[i] = prompt_ids[i];
        forward_one(model, cache, prompt_ids[i], x, logits);
    }

    // Autoregressive loop
    for (int i = 0; i < n_new; i++) {
        int next = sample_top_k(logits, model->vocab_size, temperature, top_k, rng);
        out_ids[prompt_len + i] = (uint16_t)next;
        forward_one(model, cache, (uint16_t)next, x, logits);
    }

    return prompt_len + n_new;
}