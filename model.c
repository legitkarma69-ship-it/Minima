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

bool model_config_validate(const ModelConfig* config) {
    return config &&
        config->vocab_size >= MODEL_MIN_VOCAB_SIZE &&
        config->vocab_size <= MODEL_MAX_VOCAB_SIZE &&
        config->dim > 0 && config->dim <= MODEL_MAX_DIM &&
        config->n_layers > 0 && config->n_layers <= MODEL_MAX_LAYERS &&
        config->n_heads > 0 && config->n_heads <= MODEL_MAX_HEADS &&
        config->dim % config->n_heads == 0 &&
        config->ffn_dim > 0 && config->ffn_dim <= MODEL_MAX_FFN_DIM &&
        config->max_seq >= MODEL_MIN_SEQ &&
        config->max_seq <= MODEL_MAX_SEQ;
}

bool model_validate(const Model* model) {
    if (!model) return false;
    const ModelConfig config = {
        model->vocab_size, model->dim, model->n_layers,
        model->n_heads, model->ffn_dim, model->max_seq
    };
    if (!model_config_validate(&config) ||
        model->head_dim != model->dim / model->n_heads ||
        !model->tok_emb || !model->pos_emb ||
        !model->attn_qkv || !model->attn_proj ||
        !model->ffn_up || !model->ffn_gate || !model->ffn_down ||
        !model->rms1_scale || !model->rms2_scale ||
        !model->rms_final_scale)
        return false;

    const size_t dim = (size_t)model->dim;
    const size_t layers = (size_t)model->n_layers;
    const size_t ffn = (size_t)model->ffn_dim;
    const size_t params =
        (size_t)model->vocab_size * dim +
        (size_t)model->max_seq * dim +
        layers * 3 * dim * dim +
        layers * dim * dim +
        layers * ffn * dim * 2 +
        layers * dim * ffn +
        layers * dim * 2 + dim;
    return params <= SIZE_MAX / sizeof(float) &&
        model->weight_bytes == params * sizeof(float);
}

// ------------------------------------------------------------------
// Public API
// ------------------------------------------------------------------

bool model_init_config(Model* m, Arena* arena, uint64_t seed, const ModelConfig* config) {
    if (!m || !arena || !model_config_validate(config))
        return false;
    const int vocab_size = config->vocab_size;

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