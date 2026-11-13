#include "checkpoint.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#define CHECKPOINT_MAGIC_V4 "MINIMA04"
#define CHECKPOINT_MAGIC_V5 "MINIMA05"
#define CHECKPOINT_VERSION_V4 4u
#define CHECKPOINT_VERSION_V5 5u

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
        layers * ((size_t)m->qkv_dim * dim + dim * dim +
            ffn * dim + ffn * dim + dim * ffn +
            2 * dim) + dim;
}

#define MODEL_ARRAYS(F) \
    F(tok_emb, (size_t)m->vocab_size * m->dim); \
    F(attn_qkv, (size_t)m->n_layers * m->qkv_dim * m->dim); \
    F(attn_proj, (size_t)m->n_layers * m->dim * m->dim); \
    F(ffn_up, (size_t)m->n_layers * m->ffn_dim * m->dim); \
    F(ffn_gate, (size_t)m->n_layers * m->ffn_dim * m->dim); \
    F(ffn_down, (size_t)m->n_layers * m->dim * m->ffn_dim); \
    F(rms1_scale, (size_t)m->n_layers * m->dim); \
    F(rms2_scale, (size_t)m->n_layers * m->dim); \
    F(rms_final_scale, (size_t)m->dim)

#define OPT_ARRAYS(F) \
    F(tok_emb, (size_t)m->vocab_size * m->dim); \
    F(attn_qkv, (size_t)m->n_layers * m->qkv_dim * m->dim); \
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
    if (opt->type == OPTIMIZER_ADAMW && \
        !write_floats(f, opt->v_##name, (count))) goto fail; \
} while (0)

#define OPT_SECOND_MOMENTS_VALID(o) \
    ((o)->type != OPTIMIZER_ADAMW || \
     ((o)->v_tok_emb && (o)->v_attn_qkv && (o)->v_attn_proj && \
      (o)->v_ffn_up && (o)->v_ffn_gate && (o)->v_ffn_down && \
      (o)->v_rms1_scale && (o)->v_rms2_scale && (o)->v_rms_final_scale))

bool checkpoint_save(const char* path, const Model* model, const Optim* opt,
    const Rng* rng, uint64_t tokenizer_hash) {
    const Model* m = model;
    if (!path || !model || !opt || !rng ||
        (opt->type != OPTIMIZER_ADAMW && opt->type != OPTIMIZER_LION) ||
        !opt->m_tok_emb || !opt->m_attn_qkv || !opt->m_attn_proj ||
        !opt->m_ffn_up || !opt->m_ffn_gate || !opt->m_ffn_down ||
        !opt->m_rms1_scale || !opt->m_rms2_scale ||
        !opt->m_rms_final_scale || !OPT_SECOND_MOMENTS_VALID(opt) ||
        !model->tok_emb ||
        !model->attn_qkv || !model->attn_proj ||
        !model->ffn_up || !model->ffn_gate || !model->ffn_down ||
        !model->rms1_scale || !model->rms2_scale ||
        !model->rms_final_scale || opt->n_params != model_parameter_count(model) ||
        !isfinite(opt->schedule_base_lr) || opt->schedule_base_lr <= 0.0f ||
        opt->schedule_total_steps < 1 || opt->step < 0 ||
        opt->step > opt->schedule_total_steps || opt->schedule_warmup_steps < 0 ||
        opt->schedule_warmup_steps >= opt->schedule_total_steps ||
        (opt->schedule_type != 0 && opt->schedule_type != 1))
        return false;

    char temp_path[1024];
    const int path_len = snprintf(temp_path, sizeof(temp_path), "%s.tmp", path);
    if (path_len < 0 || (size_t)path_len >= sizeof(temp_path))
        return false;

    FILE* f = fopen(temp_path, "wb");
    if (!f) return false;

    const char magic[8] = CHECKPOINT_MAGIC_V5;
    uint32_t version = CHECKPOINT_VERSION_V5;
    uint32_t dims[6] = {
        (uint32_t)model->vocab_size, (uint32_t)model->dim,
        (uint32_t)model->n_layers, (uint32_t)model->n_heads,
        (uint32_t)model->ffn_dim, (uint32_t)model->max_seq
    };
    uint64_t params = (uint64_t)model_parameter_count(model);
    int64_t step = opt->step;
    uint64_t rng_state = rng->state;
    const int32_t optimizer_type = (int32_t)opt->type;

    if (!write_bytes(f, magic, 1, sizeof(magic)) ||
        !write_bytes(f, &version, sizeof(version), 1) ||
        !write_bytes(f, dims, sizeof(uint32_t), 6) ||
        !write_bytes(f, &params, sizeof(params), 1) ||
        !write_bytes(f, &step, sizeof(step), 1) ||
        !write_bytes(f, &rng_state, sizeof(rng_state), 1) ||
        !write_bytes(f, &tokenizer_hash, sizeof(tokenizer_hash), 1) ||
        !write_bytes(f, &opt->lr, sizeof(float), 1) ||
        !write_bytes(f, &opt->beta1, sizeof(float), 1) ||
        !write_bytes(f, &opt->beta2, sizeof(float), 1) ||
        !write_bytes(f, &opt->eps, sizeof(float), 1) ||
        !write_bytes(f, &opt->weight_decay, sizeof(float), 1) ||
        !write_bytes(f, &opt->schedule_base_lr, sizeof(float), 1) ||
        !write_bytes(f, &opt->schedule_total_steps, sizeof(int64_t), 1) ||
        !write_bytes(f, &opt->schedule_warmup_steps, sizeof(int64_t), 1) ||
        !write_bytes(f, &opt->schedule_type, sizeof(int32_t), 1) ||
        !write_bytes(f, &optimizer_type, sizeof(optimizer_type), 1))
        goto fail;

    MODEL_ARRAYS(WRITE_MODEL);
    OPT_ARRAYS(WRITE_OPT_MOMENTS);

    if (fflush(f) != 0) goto fail;
    if (fclose(f) != 0) {
        remove(temp_path);
        return false;
    }

#ifdef _WIN32
    if (!MoveFileExA(temp_path, path,
        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        remove(temp_path);
        return false;
    }
#else
    if (rename(temp_path, path) != 0) {
        remove(temp_path);
        return false;
    }
#endif
    return true;

fail:
    fclose(f);
    remove(temp_path);
    return false;
}

bool checkpoint_read_optimizer_type(const char* path, OptimizerType* type_out) {
    if (!path || !type_out) return false;
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    char magic[8];
    uint32_t version = 0;
    bool ok = read_bytes(f, magic, 1, sizeof(magic)) &&
        read_bytes(f, &version, sizeof(version), 1);
    OptimizerType type = OPTIMIZER_ADAMW;
    if (ok && version == CHECKPOINT_VERSION_V4) {
        ok = memcmp(magic, CHECKPOINT_MAGIC_V4, sizeof(magic)) == 0;
    } else if (ok && version == CHECKPOINT_VERSION_V5) {
        ok = memcmp(magic, CHECKPOINT_MAGIC_V5, sizeof(magic)) == 0;
        /* Version 5 appends the optimizer identifier after the 112-byte
           version-4 header. */
        int32_t stored_type = -1;
        if (ok && fseek(f, 112L, SEEK_SET) == 0 &&
            read_bytes(f, &stored_type, sizeof(stored_type), 1) &&
            (stored_type == OPTIMIZER_ADAMW || stored_type == OPTIMIZER_LION))
            type = (OptimizerType)stored_type;
        else
            ok = false;
    } else {
        ok = false;
    }

    if (fclose(f) != 0) ok = false;
    if (ok) *type_out = type;
    return ok;
}

#define STAGE_MODEL(name, count) do { \
    staged_model.name = cursor; cursor += (count); \
} while (0)

#define STAGE_OPT_MOMENTS(name, count) do { \
    staged_opt.m_##name = cursor; cursor += (count); \
    if (staged_opt.type == OPTIMIZER_ADAMW) { \
        staged_opt.v_##name = cursor; cursor += (count); \
    } else { \
        staged_opt.v_##name = NULL; \
    } \
} while (0)

#define READ_MODEL(name, count) do { \
    if (!read_floats(f, staged_model.name, (count))) goto fail; \
} while (0)

#define READ_OPT_MOMENTS(name, count) do { \
    if (!read_floats(f, staged_opt.m_##name, (count))) goto fail; \
    if (staged_opt.type == OPTIMIZER_ADAMW && \
        !read_floats(f, staged_opt.v_##name, (count))) goto fail; \
} while (0)

#define COMMIT_MODEL(name, count) do { \
    memcpy(model->name, staged_model.name, (count) * sizeof(float)); \
} while (0)

#define COMMIT_OPT_MOMENTS(name, count) do { \
    memcpy(opt->m_##name, staged_opt.m_##name, (count) * sizeof(float)); \
    if (staged_opt.type == OPTIMIZER_ADAMW) \
        memcpy(opt->v_##name, staged_opt.v_##name, (count) * sizeof(float)); \
} while (0)

bool checkpoint_load(const char* path, Model* model, Optim* opt,
    Rng* rng, uint64_t tokenizer_hash) {
    if (!path || !model || !opt || !rng ||
        (opt->type != OPTIMIZER_ADAMW && opt->type != OPTIMIZER_LION) ||
        !model->tok_emb || !model->attn_qkv ||
        !model->attn_proj || !model->ffn_up || !model->ffn_gate ||
        !model->ffn_down || !model->rms1_scale || !model->rms2_scale ||
        !model->rms_final_scale || opt->n_params != model_parameter_count(model) ||
        !opt->m_tok_emb || !opt->m_attn_qkv ||
        !opt->m_attn_proj || !opt->m_ffn_up || !opt->m_ffn_gate ||
        !opt->m_ffn_down || !opt->m_rms1_scale || !opt->m_rms2_scale ||
        !opt->m_rms_final_scale || !OPT_SECOND_MOMENTS_VALID(opt))
        return false;

    const size_t param_count = model_parameter_count(model);
    const size_t staged_arrays = opt->type == OPTIMIZER_ADAMW ? 3u : 2u;
    if (param_count > SIZE_MAX / staged_arrays / sizeof(float))
        return false;
    const size_t staged_floats = param_count * staged_arrays;
    float* staging = (float*)malloc(staged_floats * sizeof(float));
    if (!staging) return false;

    Model staged_model = *model;
    Optim staged_opt = *opt;
    float* cursor = staging;
    Model* m = &staged_model;
    MODEL_ARRAYS(STAGE_MODEL);
    OPT_ARRAYS(STAGE_OPT_MOMENTS);

    FILE* f = fopen(path, "rb");
    if (!f) { free(staging); return false; }

    char magic[8];
    uint32_t version = 0, dims[6] = { 0 };
    uint64_t checkpoint_params = 0;
    int64_t step = 0;
    uint64_t rng_state = 0, saved_tokenizer_hash = 0;
    float lr, beta1, beta2, eps, weight_decay, schedule_base_lr;
    int64_t schedule_total_steps = 0, schedule_warmup_steps = 0;
    int32_t schedule_type = -1;
    int32_t saved_optimizer_type = OPTIMIZER_ADAMW;

    if (!read_bytes(f, magic, 1, sizeof(magic)) ||
        !read_bytes(f, &version, sizeof(version), 1))
        goto fail;
    if (!((version == CHECKPOINT_VERSION_V4 &&
            memcmp(magic, CHECKPOINT_MAGIC_V4, sizeof(magic)) == 0) ||
          (version == CHECKPOINT_VERSION_V5 &&
            memcmp(magic, CHECKPOINT_MAGIC_V5, sizeof(magic)) == 0)))
        goto fail;

    if (
        !read_bytes(f, dims, sizeof(uint32_t), 6) ||
        !read_bytes(f, &checkpoint_params, sizeof(checkpoint_params), 1) ||
        !read_bytes(f, &step, sizeof(step), 1) ||
        !read_bytes(f, &rng_state, sizeof(rng_state), 1) ||
        !read_bytes(f, &saved_tokenizer_hash, sizeof(saved_tokenizer_hash), 1) ||
        !read_bytes(f, &lr, sizeof(float), 1) ||
        !read_bytes(f, &beta1, sizeof(float), 1) ||
        !read_bytes(f, &beta2, sizeof(float), 1) ||
        !read_bytes(f, &eps, sizeof(float), 1) ||
        !read_bytes(f, &weight_decay, sizeof(float), 1) ||
        !read_bytes(f, &schedule_base_lr, sizeof(float), 1) ||
        !read_bytes(f, &schedule_total_steps, sizeof(int64_t), 1) ||
        !read_bytes(f, &schedule_warmup_steps, sizeof(int64_t), 1) ||
        !read_bytes(f, &schedule_type, sizeof(int32_t), 1))
        goto fail;
    if (version == CHECKPOINT_VERSION_V5 &&
        !read_bytes(f, &saved_optimizer_type, sizeof(saved_optimizer_type), 1))
        goto fail;
    if (saved_optimizer_type != (int32_t)opt->type) goto fail;

    if (dims[0] != (uint32_t)model->vocab_size ||
        dims[1] != (uint32_t)model->dim ||
        dims[2] != (uint32_t)model->n_layers ||
        dims[3] != (uint32_t)model->n_heads ||
        dims[4] != (uint32_t)model->ffn_dim ||
        dims[5] != (uint32_t)model->max_seq ||
        checkpoint_params != (uint64_t)model_parameter_count(model) ||
        checkpoint_params != (uint64_t)opt->n_params ||
        saved_tokenizer_hash != tokenizer_hash ||
        step < 0 || !isfinite(lr) || lr < 0.0f ||
        !isfinite(beta1) || beta1 < 0.0f || beta1 >= 1.0f ||
        !isfinite(beta2) || beta2 < 0.0f || beta2 >= 1.0f ||
        !isfinite(eps) || eps <= 0.0f ||
        !isfinite(weight_decay) || weight_decay < 0.0f ||
        !isfinite(schedule_base_lr) || schedule_base_lr <= 0.0f ||
        schedule_total_steps < 1 || step > schedule_total_steps ||
        schedule_warmup_steps < 0 || schedule_warmup_steps >= schedule_total_steps ||
        (schedule_type != 0 && schedule_type != 1))
        goto fail;

    MODEL_ARRAYS(READ_MODEL);
    OPT_ARRAYS(READ_OPT_MOMENTS);

    // Reject truncated files, unexpected trailing bytes, and non-finite state.
    if (fgetc(f) != EOF || ferror(f)) goto fail;
    for (size_t i = 0; i < staged_floats; ++i)
        if (!isfinite(staging[i])) goto fail;

    if (fclose(f) != 0) { free(staging); return false; }

    // Commit only after the entire file has been read and validated.
    m = &staged_model;
    MODEL_ARRAYS(COMMIT_MODEL);
    OPT_ARRAYS(COMMIT_OPT_MOMENTS);
    opt->step = step;
    rng->state = rng_state ? rng_state : 0x123456789abcdef0ULL;
    opt->lr = lr;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->eps = eps;
    opt->weight_decay = weight_decay;
    opt->schedule_base_lr = schedule_base_lr;
    opt->schedule_total_steps = schedule_total_steps;
    opt->schedule_warmup_steps = schedule_warmup_steps;
    opt->schedule_type = schedule_type;
    opt->type = (OptimizerType)saved_optimizer_type;
    free(staging);
    return true;

fail:
    fclose(f);
    free(staging);
    return false;
}
