#include "train.h"

#include <limits.h>
#include <math.h>

static bool validation_fail(const char** reason_out, const char* reason) {
    if (reason_out) *reason_out = reason;
    return false;
}

#define REQUIRE_MODEL_TENSORS(m) \
    ((m)->tok_emb && (m)->pos_emb && (m)->attn_qkv && \
     (m)->attn_proj && (m)->ffn_up && (m)->ffn_gate && \
     (m)->ffn_down && (m)->rms1_scale && (m)->rms2_scale && \
     (m)->rms_final_scale)

#define REQUIRE_OPT_TENSORS(o) \
    ((o)->m_tok_emb && (o)->m_pos_emb && (o)->m_attn_qkv && \
     (o)->m_attn_proj && (o)->m_ffn_up && (o)->m_ffn_gate && \
     (o)->m_ffn_down && (o)->m_rms1_scale && (o)->m_rms2_scale && \
     (o)->m_rms_final_scale && (o)->v_tok_emb && (o)->v_pos_emb && \
     (o)->v_attn_qkv && (o)->v_attn_proj && (o)->v_ffn_up && \
     (o)->v_ffn_gate && (o)->v_ffn_down && (o)->v_rms1_scale && \
     (o)->v_rms2_scale && (o)->v_rms_final_scale)

bool train_run_validate_inputs(const Model* model, const Optim* opt,
    const Gradients* grads, const Arena* temp, const uint16_t* all_ids,
    int train_tokens, int validation_tokens, const Rng* rng,
    TrainRunOptions options, const char** reason_out) {
    if (reason_out) *reason_out = NULL;

    if (!model || !opt || !grads || !temp || !all_ids || !rng)
        return validation_fail(reason_out, "required model, optimizer, gradient, arena, token, or RNG pointer is null");
    if (!temp->base || temp->size == 0 || temp->used > temp->size)
        return validation_fail(reason_out, "temporary arena is not initialized or has an invalid cursor");
    if (!options.metrics_path || !options.metrics_path[0])
        return validation_fail(reason_out, "metrics CSV path is empty");
    if (options.steps < 0)
        return validation_fail(reason_out, "training step count cannot be negative");
    if (options.log_interval <= 0 || options.validation_interval <= 0 ||
        options.checkpoint_interval <= 0)
        return validation_fail(reason_out, "log, validation, and checkpoint intervals must be positive");
    if (options.validation_batches <= 0)
        return validation_fail(reason_out, "validation batch count must be positive");
    if (options.batch_size <= 0 || options.sequence_length <= 0)
        return validation_fail(reason_out, "batch size and sequence length must be positive");

    if (model->vocab_size < 260 || model->vocab_size > 8192 ||
        model->dim <= 0 || model->dim > 512 ||
        model->n_layers <= 0 || model->n_layers > 16 ||
        model->n_heads <= 0 || model->n_heads > 32 ||
        model->dim % model->n_heads != 0 ||
        model->head_dim != model->dim / model->n_heads ||
        model->ffn_dim <= 0 || model->ffn_dim > 1024 ||
        model->max_seq < 32 || model->max_seq > 512 ||
        options.sequence_length > model->max_seq ||
        model->weight_bytes == 0 ||
        model->weight_bytes % sizeof(float) != 0)
        return validation_fail(reason_out, "model dimensions or weight footprint are invalid");
    if (!REQUIRE_MODEL_TENSORS(model))
        return validation_fail(reason_out, "one or more model weight tensors are null");

    if (grads->vocab_size != model->vocab_size ||
        grads->dim != model->dim ||
        grads->n_layers != model->n_layers ||
        grads->n_heads != model->n_heads ||
        grads->head_dim != model->head_dim ||
        grads->ffn_dim != model->ffn_dim ||
        grads->max_seq != model->max_seq ||
        grads->weight_bytes != model->weight_bytes ||
        !REQUIRE_MODEL_TENSORS(grads))
        return validation_fail(reason_out, "gradient tensors do not match the model layout");

    if (!REQUIRE_OPT_TENSORS(opt) ||
        opt->n_params != model_param_count(model))
        return validation_fail(reason_out, "optimizer moment buffers do not match the model parameter count");

    if (!isfinite(opt->lr) || opt->lr <= 0.0f ||
        !isfinite(opt->beta1) || opt->beta1 < 0.0f || opt->beta1 >= 1.0f ||
        !isfinite(opt->beta2) || opt->beta2 < 0.0f || opt->beta2 >= 1.0f ||
        !isfinite(opt->eps) || opt->eps <= 0.0f ||
        !isfinite(opt->weight_decay) || opt->weight_decay < 0.0f)
        return validation_fail(reason_out, "AdamW hyperparameters are invalid");

    if (!isfinite(opt->schedule_base_lr) || opt->schedule_base_lr <= 0.0f ||
        opt->schedule_total_steps <= 0 ||
        opt->schedule_warmup_steps < 0 ||
        opt->schedule_warmup_steps > opt->schedule_total_steps ||
        opt->step < 0 || opt->step > opt->schedule_total_steps ||
        (opt->schedule_type != 0 && opt->schedule_type != 1))
        return validation_fail(reason_out, "learning-rate schedule state is invalid");
    if (!options.diagnostics_only && options.steps > 0 &&
        (int64_t)options.steps > opt->schedule_total_steps - opt->step)
        return validation_fail(reason_out, "requested training steps exceed the remaining learning-rate schedule");

    if ((size_t)options.batch_size > SIZE_MAX /
            (size_t)options.sequence_length)
        return validation_fail(reason_out, "batch token count overflows size_t");
    const size_t batch_tokens =
        (size_t)options.batch_size * (size_t)options.sequence_length;
    if (batch_tokens > (size_t)INT_MAX - 1 ||
        batch_tokens > SIZE_MAX / (size_t)model->vocab_size / sizeof(float) ||
        batch_tokens > SIZE_MAX / sizeof(uint16_t))
        return validation_fail(reason_out, "training scratch dimensions overflow");
    if (train_tokens < (int)batch_tokens + 1 ||
        validation_tokens < (int)batch_tokens + 1)
        return validation_fail(reason_out, "training and validation splits must each contain at least batch_size * sequence_length + 1 tokens");
    if ((int64_t)train_tokens + validation_tokens > INT_MAX)
        return validation_fail(reason_out, "combined training and validation token count overflows supported indexing");

    if ((size_t)options.validation_batches >
        SIZE_MAX / (size_t)options.batch_size)
        return validation_fail(reason_out, "validation sample count overflows size_t");

    return true;
}
