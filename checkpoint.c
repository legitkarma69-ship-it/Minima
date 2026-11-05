#include "checkpoint.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define CHECKPOINT_MAGIC "MINIMA01"
#define CHECKPOINT_VERSION 1u

static bool write_bytes(FILE* f, const void* data, size_t size, size_t count) {
    return fwrite(data, size, count, f) == count;
}

static bool read_bytes(FILE* f, void* data, size_t size, size_t count) {
    return fread(data, size, count, f) == count;
}

static bool write_floats(FILE* f, const float* p, size_t n) {
    return p && write_bytes(f, p, sizeof(float), n);
}

static bool read_floats(FILE* f, float* p, size_t n) {
    return p && read_bytes(f, p, sizeof(float), n);
}

static size_t model_parameter_count(const Model* m) {
    size_t layers = (size_t)m->n_layers;
    size_t dim = (size_t)m->dim;
    size_t ffn = (size_t)m->ffn_dim;
    return (size_t)m->vocab_size * dim +
        (size_t)m->max_seq * dim +
        layers * (3 * dim * dim + dim * dim +
            ffn * dim + ffn * dim + dim * ffn +
            2 * dim) + dim;
}

#define MODEL_ARRAYS(F) \
    F(tok_emb, (size_t)m->vocab_size * m->dim); \
    F(pos_emb, (size_t)m->max_seq * m->dim); \
    F(attn_qkv, (size_t)m->n_layers * 3 * m->dim * m->dim); \
    F(attn_proj, (size_t)m->n_layers * m->dim * m->dim); \
    F(ffn_up, (size_t)m->n_layers * m->ffn_dim * m->dim); \
    F(ffn_gate, (size_t)m->n_layers * m->ffn_dim * m->dim); \
    F(ffn_down, (size_t)m->n_layers * m->dim * m->ffn_dim); \
    F(rms1_scale, (size_t)m->n_layers * m->dim); \
    F(rms2_scale, (size_t)m->n_layers * m->dim); \
    F(rms_final_scale, (size_t)m->dim)

#define OPT_ARRAYS(F) \
    F(tok_emb, (size_t)m->vocab_size * m->dim); \
    F(pos_emb, (size_t)m->max_seq * m->dim); \
    F(attn_qkv, (size_t)m->n_layers * 3 * m->dim * m->dim); \
    F(attn_proj, (size_t)m->n_layers * m->dim * m->dim); \
    F(ffn_up, (size_t)m->n_layers * m->ffn_dim * m->dim); \
    F(ffn_gate, (size_t)m->n_layers * m->ffn_dim * m->dim); \
    F(ffn_down, (size_t)m->n_layers * m->dim * m->ffn_dim); \
    F(rms1_scale, (size_t)m->n_layers * m->dim); \
    F(rms2_scale, (size_t)m->n_layers * m->dim); \
    F(rms_final_scale, (size_t)m->dim)

#define WRITE_MODEL(name, count) do { \
    if (!write_floats(f, m->name, (count))) goto fail; \
} while (0)

#define WRITE_OPT_MOMENTS(name, count) do { \
    if (!write_floats(f, opt->m_##name, (count))) goto fail; \
    if (!write_floats(f, opt->v_##name, (count))) goto fail; \
} while (0)

bool checkpoint_save(const char* path, const Model* model, const Optim* opt) {
    if (!path || !model || !opt || !model->tok_emb ||
        !model->pos_emb || !model->attn_qkv || !model->attn_proj ||
        !model->ffn_up || !model->ffn_gate || !model->ffn_down ||
        !model->rms1_scale || !model->rms2_scale ||
        !model->rms_final_scale || opt->n_params != model_parameter_count(model))
        return false;

    FILE* f = fopen(path, "wb");
    if (!f) return false;

    const char magic[8] = CHECKPOINT_MAGIC;
    uint32_t version = CHECKPOINT_VERSION;
    uint32_t dims[6] = {
        (uint32_t)model->vocab_size, (uint32_t)model->dim,
        (uint32_t)model->n_layers, (uint32_t)model->n_heads,
        (uint32_t)model->ffn_dim, (uint32_t)model->max_seq
    };
    uint64_t params = (uint64_t)model_parameter_count(model);
    int64_t step = opt->step;

    if (!write_bytes(f, magic, 1, sizeof(magic)) ||
        !write_bytes(f, &version, sizeof(version), 1) ||
        !write_bytes(f, dims, sizeof(uint32_t), 6) ||
        !write_bytes(f, &params, sizeof(params), 1) ||
        !write_bytes(f, &step, sizeof(step), 1) ||
        !write_bytes(f, &opt->lr, sizeof(float), 1) ||
        !write_bytes(f, &opt->beta1, sizeof(float), 1) ||
        !write_bytes(f, &opt->beta2, sizeof(float), 1) ||
        !write_bytes(f, &opt->eps, sizeof(float), 1) ||
        !write_bytes(f, &opt->weight_decay, sizeof(float), 1))
        goto fail;

    MODEL_ARRAYS(WRITE_MODEL);
    OPT_ARRAYS(WRITE_OPT_MOMENTS);

    if (fclose(f) != 0) return false;
    return true;

fail:
    fclose(f);
    return false;
}

#define READ_MODEL(name, count) do { \
    if (!read_floats(f, model->name, (count))) goto fail; \
} while (0)

#define READ_OPT_MOMENTS(name, count) do { \
    if (!read_floats(f, opt->m_##name, (count))) goto fail; \
    if (!read_floats(f, opt->v_##name, (count))) goto fail; \
} while (0)

bool checkpoint_load(const char* path, Model* model, Optim* opt) {
    if (!path || !model || !opt || !model->tok_emb ||
        !model->pos_emb || !model->attn_qkv || !model->attn_proj ||
        !model->ffn_up || !model->ffn_gate || !model->ffn_down ||
        !model->rms1_scale || !model->rms2_scale ||
        !model->rms_final_scale)
        return false;

    FILE* f = fopen(path, "rb");
    if (!f) return false;

    char magic[8];
    uint32_t version = 0, dims[6] = { 0 };
    uint64_t params = 0;
    int64_t step = 0;
    float lr, beta1, beta2, eps, weight_decay;

    if (!read_bytes(f, magic, 1, sizeof(magic)) ||
        memcmp(magic, CHECKPOINT_MAGIC, sizeof(magic)) != 0 ||
        !read_bytes(f, &version, sizeof(version), 1) ||
        version != CHECKPOINT_VERSION ||
        !read_bytes(f, dims, sizeof(uint32_t), 6) ||
        !read_bytes(f, &params, sizeof(params), 1) ||
        !read_bytes(f, &step, sizeof(step), 1) ||
        !read_bytes(f, &lr, sizeof(float), 1) ||
        !read_bytes(f, &beta1, sizeof(float), 1) ||
        !read_bytes(f, &beta2, sizeof(float), 1) ||
        !read_bytes(f, &eps, sizeof(float), 1) ||
        !read_bytes(f, &weight_decay, sizeof(float), 1))
        goto fail;

    if (dims[0] != (uint32_t)model->vocab_size ||
        dims[1] != (uint32_t)model->dim ||
        dims[2] != (uint32_t)model->n_layers ||
        dims[3] != (uint32_t)model->n_heads ||
        dims[4] != (uint32_t)model->ffn_dim ||
        dims[5] != (uint32_t)model->max_seq ||
        params != (uint64_t)model_parameter_count(model) ||
        params != (uint64_t)opt->n_params ||
        step < 0 || !isfinite(lr) || lr <= 0.0f ||
        !isfinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f ||
        !isfinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f ||
        !isfinite(eps) || eps <= 0.0f ||
        !isfinite(weight_decay) || weight_decay < 0.0f)
        goto fail;

    MODEL_ARRAYS(READ_MODEL);
    OPT_ARRAYS(READ_OPT_MOMENTS);

    // Reject truncated files and unexpected trailing data.
    if (fgetc(f) != EOF || ferror(f)) goto fail;

    opt->step = step;
    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->weight_decay = weight_decay;

    if (fclose(f) != 0) return false;
    return true;

fail:
    fclose(f);
    return false;
}
