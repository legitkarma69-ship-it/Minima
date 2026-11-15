#include "train.h"

#include <limits.h>
#include <math.h>

static bool validation_fail(const char** reason_out, const char* reason) {
    if (reason_out) *reason_out = reason;
    return false;
}

#define REQUIRE_MODEL_TENSORS(m) \
    ((m)->tok_emb && (m)->attn_qkv && \
     (m)->attn_proj && (m)->ffn_up && (m)->ffn_gate && \
     (m)->ffn_down && (m)->rms1_scale && (m)->rms2_scale && \
     (m)->rms_final_scale)

#define REQUIRE_OPT_FIRST_MOMENTS(o) \
    ((o)->m_tok_emb && (o)->m_attn_qkv && \
     (o)->m_attn_proj && (o)->m_ffn_up && (o)->m_ffn_gate && \
     (o)->m_ffn_down && (o)->m_rms1_scale && (o)->m_rms2_scale && \
     (o)->m_rms_final_scale)

#define REQUIRE_OPT_SECOND_MOMENTS(o) \
    ((o)->type != OPTIMIZER_ADAMW || ((o)->v_tok_emb && \
     (o)->v_attn_qkv && (o)->v_attn_proj && (o)->v_ffn_up && \
     (o)->v_ffn_gate && (o)->v_ffn_down && (o)->v_rms1_scale && \
     (o)->v_rms2_scale && (o)->v_rms_final_scale))

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
    if (options.corpus_mix != CORPUS_MIX_PROPORTIONAL &&
        options.corpus_mix != CORPUS_MIX_SQRT &&
        options.corpus_mix != CORPUS_MIX_EQUAL)
        return validation_fail(reason_out, "corpus mixture mode is invalid");
    if (options.corpus_source_count > TRAIN_MAX_SOURCES ||
        (options.corpus_source_count > 0 && !options.corpus_sources))
        return validation_fail(reason_out, "corpus source metadata is invalid");
    if (options.batch_size <= 0 || options.sequence_length <= 0)
        return validation_fail(reason_out, "batch size and sequence length must be positive");
    if (!isfinite(options.grad_clip_norm) || options.grad_clip_norm < 0.0f)
        return validation_fail(reason_out, "gradient clipping norm must be finite and non-negative");

    if (!model_validate(model) ||
        options.sequence_length > model->max_seq)
        return validation_fail(reason_out, "model dimensions or weight footprint are invalid");
    if (!REQUIRE_MODEL_TENSORS(model))
        return validation_fail(reason_out, "one or more model weight tensors are null");

    if (grads->vocab_size != model->vocab_size ||
        grads->dim != model->dim ||
        grads->n_layers != model->n_layers ||
        grads->n_heads != model->n_heads ||
        grads->n_kv_heads != model->n_kv_heads ||
        grads->head_dim != model->head_dim ||
        grads->kv_dim != model->kv_dim ||
        grads->qkv_dim != model->qkv_dim ||
        grads->ffn_dim != model->ffn_dim ||
        grads->max_seq != model->max_seq ||
        grads->weight_bytes != model->weight_bytes ||
        !REQUIRE_MODEL_TENSORS(grads))
        return validation_fail(reason_out, "gradient tensors do not match the model layout");

    if ((opt->type != OPTIMIZER_ADAMW && opt->type != OPTIMIZER_LION) ||
        !REQUIRE_OPT_FIRST_MOMENTS(opt) ||
        !REQUIRE_OPT_SECOND_MOMENTS(opt) ||
        opt->n_params != model_param_count(model))
        return validation_fail(reason_out, "optimizer state does not match the model parameter count or optimizer type");

    if (!isfinite(opt->lr) || opt->lr <= 0.0f ||
        !isfinite(opt->beta1) || opt->beta1 < 0.0f || opt->beta1 >= 1.0f ||
        !isfinite(opt->beta2) || opt->beta2 < 0.0f || opt->beta2 >= 1.0f ||
        !isfinite(opt->eps) || opt->eps <= 0.0f ||
        !isfinite(opt->weight_decay) || opt->weight_decay < 0.0f)
        return validation_fail(reason_out, "optimizer hyperparameters are invalid");

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
    const int minimum_split_tokens = options.corpus_source_count > 0
        ? options.sequence_length + 1 : (int)batch_tokens + 1;
    if (train_tokens < minimum_split_tokens ||
        validation_tokens < minimum_split_tokens)
        return validation_fail(reason_out, "training and validation data do not contain enough tokens for the configured sequence windows");
    if ((int64_t)train_tokens + validation_tokens > INT_MAX)
        return validation_fail(reason_out, "combined training and validation token count overflows supported indexing");

    if (options.corpus_source_count > 0) {
        int64_t source_train_total = 0;
        int64_t source_validation_total = 0;
        const int minimum_source_tokens = options.sequence_length + 1;
        for (size_t i = 0; i < options.corpus_source_count; ++i) {
            const TrainSourceSpan* source = &options.corpus_sources[i];
            if (source->train_start < 0 || source->validation_start < 0 ||
                source->train_tokens < minimum_source_tokens ||
                source->validation_tokens < minimum_source_tokens)
                return validation_fail(reason_out, "each corpus source must contain valid train and held-out windows");
            if ((int64_t)source->train_start + source->train_tokens >
                    (int64_t)train_tokens + validation_tokens ||
                (int64_t)source->validation_start + source->validation_tokens >
                    (int64_t)train_tokens + validation_tokens)
                return validation_fail(reason_out, "corpus source range exceeds the token array");
            source_train_total += source->train_tokens;
            source_validation_total += source->validation_tokens;
        }
        if (source_train_total != train_tokens ||
            source_validation_total != validation_tokens)
            return validation_fail(reason_out, "corpus source ranges do not match aggregate split sizes");
    }

    if ((size_t)options.validation_batches >
        SIZE_MAX / (size_t)options.batch_size)
        return validation_fail(reason_out, "validation sample count overflows size_t");
    const size_t validation_sample_count =
        (size_t)options.validation_batches * (size_t)options.batch_size;
    const size_t validation_span =
        (size_t)validation_tokens - (size_t)options.sequence_length - 1;
    if (validation_sample_count > 1 &&
        validation_span > SIZE_MAX / (validation_sample_count - 1))
        return validation_fail(reason_out, "validation window indexing overflows size_t");

    return true;
}
