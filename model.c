#include "model.h"
#include "util.h"
#include <string.h>
#include <math.h>
#include <stdio.h>

// ------------------------------------------------------------------
// Helpers
// ------------------------------------------------------------------

static float* alloc_weights(Arena* arena, size_t count) {
    float* p = arena_alloc(arena, count * sizeof(float));
    if (!p) die("model weight allocation failed – increase arena size");
    return p;
}

// Simple Gaussian via Box-Muller (good enough for init)
static float randn(Rng* rng) {
    float u1 = rng_f32(rng);
    float u2 = rng_f32(rng);
    if (u1 < 1e-7f) u1 = 1e-7f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
}

// Fill with N(0, std)
static void fill_normal(float* p, size_t n, float std, Rng* rng) {
    for (size_t i = 0; i < n; i++)
        p[i] = randn(rng) * std;
}

// Fill with zeros
static void fill_zero(float* p, size_t n) {
    memset(p, 0, n * sizeof(float));
}

// ------------------------------------------------------------------
// Public API
// ------------------------------------------------------------------

bool model_init_config(Model* m, Arena* arena, uint64_t seed, const ModelConfig* config) {
    if (!m || !arena || !config ||
        config->vocab_size < 260 || config->vocab_size > 8192 ||
        config->dim <= 0 || config->dim > 512 ||
        config->n_layers <= 0 || config->n_layers > 16 ||
        config->n_heads <= 0 || config->n_heads > 32 ||
        config->dim % config->n_heads != 0 ||
        config->ffn_dim <= 0 || config->ffn_dim > 1024 ||
        config->max_seq < 32 || config->max_seq > 512)
        return false;
    int vocab_size=config->vocab_size;

    memset(m, 0, sizeof(*m));

    m->vocab_size = vocab_size;
    m->dim = config->dim;
    m->n_layers = config->n_layers;
    m->n_heads = config->n_heads;
    m->head_dim = config->dim / config->n_heads;
    m->ffn_dim = config->ffn_dim;
    m->max_seq = config->max_seq;

    Rng rng;
    rng_seed(&rng, seed);

    // ----- Token + position embeddings -----
    m->tok_emb = alloc_weights(arena, (size_t)m->vocab_size * m->dim);
    m->pos_emb = alloc_weights(arena, (size_t)m->max_seq * m->dim);

    // Standard embedding init: N(0, 0.02)
    fill_normal(m->tok_emb, (size_t)m->vocab_size * m->dim, 0.02f, &rng);
    fill_normal(m->pos_emb, (size_t)m->max_seq * m->dim, 0.02f, &rng);

    // ----- Attention & FFN weights -----
    size_t qkv_count = (size_t)m->n_layers * 3 * m->dim * m->dim;
    size_t proj_count = (size_t)m->n_layers * m->dim * m->dim;
    size_t up_count = (size_t)m->n_layers * m->ffn_dim * m->dim;
    size_t gate_count = (size_t)m->n_layers * m->ffn_dim * m->dim;
    size_t down_count = (size_t)m->n_layers * m->dim * m->ffn_dim;
    size_t rms_count = (size_t)m->n_layers * m->dim;

    m->attn_qkv = alloc_weights(arena, qkv_count);
    m->attn_proj = alloc_weights(arena, proj_count);
    m->ffn_up = alloc_weights(arena, up_count);
    m->ffn_gate = alloc_weights(arena, gate_count);
    m->ffn_down = alloc_weights(arena, down_count);
    m->rms1_scale = alloc_weights(arena, rms_count);
    m->rms2_scale = alloc_weights(arena, rms_count);
    m->rms_final_scale = alloc_weights(arena, m->dim);

    // GPT-2 style init: N(0, 0.02) for most matrices
    // Residual projections get a smaller std so the residual stream
    // does not blow up at the beginning of training.
    float std_main = 0.02f;
    float std_res = 0.02f / sqrtf((float)m->n_layers);

    fill_normal(m->attn_qkv, qkv_count, std_main, &rng);
    fill_normal(m->attn_proj, proj_count, std_res, &rng);
    fill_normal(m->ffn_up, up_count, std_main, &rng);
    fill_normal(m->ffn_gate, gate_count, std_main, &rng);
    fill_normal(m->ffn_down, down_count, std_res, &rng);

    // RMSNorm scales start at 1.0
    for (size_t i = 0; i < rms_count; i++) {
        m->rms1_scale[i] = 1.0f;
        m->rms2_scale[i] = 1.0f;
    }
    for (int i = 0; i < m->dim; i++)
        m->rms_final_scale[i] = 1.0f;

    // Record total weight footprint
    m->weight_bytes =
        ((size_t)m->vocab_size * m->dim +
            (size_t)m->max_seq * m->dim +
            qkv_count + proj_count + up_count + gate_count + down_count +
            rms_count * 2 + m->dim) * sizeof(float);

    return true;
}

bool model_init(Model* m, Arena* arena, uint64_t seed, int vocab_size) {
    ModelConfig config = MODEL_CONFIG_DEFAULT;
    config.vocab_size = vocab_size;
    return model_init_config(m, arena, seed, &config);
}

size_t model_param_count(const Model* m) {
    return m->weight_bytes / sizeof(float);
}

void model_summary(const Model* m) {
    printf("Model summary\n");
    printf("  vocab_size : %d\n", m->vocab_size);
    printf("  dim        : %d\n", m->dim);
    printf("  n_layers   : %d\n", m->n_layers);
    printf("  n_heads    : %d\n", m->n_heads);
    printf("  head_dim   : %d\n", m->head_dim);
    printf("  ffn_dim    : %d\n", m->ffn_dim);
    printf("  max_seq    : %d\n", m->max_seq);
    printf("  parameters : %zu (%.2f M)\n",
        model_param_count(m),
        model_param_count(m) / 1e6);
    printf("  weight mem : %.2f MB\n", m->weight_bytes / (1024.0 * 1024.0));
    printf("  AdamW state: %.2f MB (two moment buffers)\n",
        (2.0 * (double)m->weight_bytes) / (1024.0 * 1024.0));
    printf("  gradients  : %.2f MB (one parameter-sized buffer)\n",
        (double)m->weight_bytes / (1024.0 * 1024.0));
    printf("  train core : %.2f MB (weights + AdamW + gradients; excludes activations)\n",
        (4.0 * (double)m->weight_bytes) / (1024.0 * 1024.0));
}