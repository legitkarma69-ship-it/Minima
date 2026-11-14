#include "train.h"

#include "checkpoint.h"
#include "forward.h"

#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static bool stop_file_exists(const char* path) {
    if (!path || !path[0]) return false;
    FILE* f=fopen(path,"rb"); if(!f)return false; fclose(f); return true;
}
static bool consume_file_signal(const char* path) {
    if(!stop_file_exists(path))return false;
    remove(path);return true;
}

typedef struct {
    size_t source_index;
    double cumulative_weight;
    double weight;
} SourceChoice;

typedef struct {
    SourceChoice choices[TRAIN_MAX_SOURCES];
    size_t count;
    double total_weight;
} SourcePlan;

static const char* corpus_mix_name(CorpusMixMode mode) {
    switch (mode) {
    case CORPUS_MIX_SQRT: return "sqrt";
    case CORPUS_MIX_EQUAL: return "equal";
    default: return "proportional";
    }
}

static bool source_plan_init(SourcePlan* plan, const TrainSourceSpan* sources,
    size_t source_count, CorpusMixMode mode, bool validation_set, int seq_len) {
    if (!plan || !sources || source_count == 0 ||
        source_count > TRAIN_MAX_SOURCES) return false;
    memset(plan, 0, sizeof(*plan));
    for (size_t i = 0; i < source_count; ++i) {
        const int sample_length = sources[i].train_tokens;
        const int range_length = validation_set
            ? sources[i].validation_tokens : sources[i].train_tokens;
        if (range_length < seq_len + 1 || sample_length < seq_len + 1) continue;
        /* Keep validation's source proportions identical to the training
           mixture even when a short source needs a larger held-out fraction. */
        const double available_windows = (double)(sample_length - seq_len);
        double weight = mode == CORPUS_MIX_EQUAL ? 1.0 :
            (mode == CORPUS_MIX_SQRT ? sqrt(available_windows) : available_windows);
        if (!(weight > 0.0) || !isfinite(weight)) continue;
        plan->total_weight += weight;
        plan->choices[plan->count++] = (SourceChoice){
            .source_index = i,
            .cumulative_weight = plan->total_weight,
            .weight = weight
        };
    }
    return plan->count > 0 && plan->total_weight > 0.0 &&
        isfinite(plan->total_weight);
}

static size_t source_plan_select(const SourcePlan* plan, double point) {
    if (point < 0.0) point = 0.0;
    if (point >= 1.0) point = 0.9999999999999999;
    const double target = point * plan->total_weight;
    for (size_t i = 0; i < plan->count; ++i)
        if (target < plan->choices[i].cumulative_weight)
            return plan->choices[i].source_index;
    return plan->choices[plan->count - 1].source_index;
}


static float evaluate_loss(Model* model, Arena* temp,
    const uint16_t* all_ids, const TrainSourceSpan* sources,
    size_t source_count, CorpusMixMode mix, bool validation_set,
    int validation_batches, int batch_size, int seq_len) {

    if (!model || !temp || !all_ids ||
        !sources || source_count == 0 || validation_batches < 1 || batch_size < 1)
        return NAN;

    arena_reset(temp);

    Activations act;
    if (!activations_init(&act, temp, batch_size, seq_len, model))
        return NAN;

    float* dlogits = arena_alloc(temp,
        (size_t)batch_size * seq_len * model->vocab_size * sizeof(float));
    uint16_t* batch_tokens = arena_alloc(temp,
        (size_t)batch_size * seq_len * sizeof(uint16_t));
    uint16_t* batch_targets = arena_alloc(temp,
        (size_t)batch_size * seq_len * sizeof(uint16_t));
    const size_t sample_count = (size_t)validation_batches * (size_t)batch_size;
    size_t* sample_sources = arena_alloc(temp, sample_count * sizeof(size_t));

    if (!dlogits || !batch_tokens || !batch_targets || !sample_sources)
        return NAN;

    SourcePlan plan;
    if (!source_plan_init(&plan, sources, source_count, mix,
            validation_set, seq_len)) return NAN;
    size_t source_counts[TRAIN_MAX_SOURCES] = { 0 };
    size_t source_seen[TRAIN_MAX_SOURCES] = { 0 };
    for (size_t i = 0; i < sample_count; ++i) {
        const size_t source_index = source_plan_select(&plan,
            ((double)i + 0.5) / (double)sample_count);
        sample_sources[i] = source_index;
        source_counts[source_index]++;
    }

    double loss_sum = 0.0;

    // Evaluate fixed windows spread across held-out pieces, with the same
    // configured source mixture used for training.
    for (int eval_batch = 0; eval_batch < validation_batches; eval_batch++) {
        for (int b = 0; b < batch_size; b++) {
            const size_t sample_index = (size_t)eval_batch * batch_size + b;
            const size_t source_index = sample_sources[sample_index];
            const TrainSourceSpan* source = &sources[source_index];
            const int range_start = validation_set
                ? source->validation_start : source->train_start;
            const int range_length = validation_set
                ? source->validation_tokens : source->train_tokens;
            const size_t span = (size_t)range_length - (size_t)seq_len - 1;
            const size_t count = source_counts[source_index];
            const size_t ordinal = source_seen[source_index]++;
            const size_t offset = count > 1
                ? (span * ordinal) / (count - 1) : span / 2;
            const int start = range_start + (int)offset;

            for (int t = 0; t < seq_len; t++) {
                batch_tokens[b * seq_len + t] = all_ids[start + t];
                batch_targets[b * seq_len + t] = all_ids[start + t + 1];
            }
        }

        float* logits = model_forward(model, &act,
            batch_tokens, batch_size, seq_len);
        if (!logits) return NAN;

        const float loss = loss_cross_entropy(logits, batch_targets,
            dlogits, batch_size, seq_len, model->vocab_size);
        if (!isfinite(loss)) return NAN;
        loss_sum += loss;
    }

    return (float)(loss_sum / validation_batches);
}


void train_run(Model* model, Optim* opt, Gradients* grads, Arena* temp,
    const uint16_t* all_ids, int train_tokens, int validation_tokens,
    Rng* rng, uint64_t tokenizer_hash, TrainRunOptions options) {
    // ------------------------------------------------------------------
    // 7. Continue training; optimizer step and RNG state are global
    // ------------------------------------------------------------------
    const char* validation_error = NULL;
    if (!train_run_validate_inputs(model, opt, grads, temp, all_ids,
            train_tokens, validation_tokens, rng, options, &validation_error))
        die("train_run input validation failed: %s",
            validation_error ? validation_error : "unknown validation error");

    const TrainSourceSpan fallback_source = {
        .name = "corpus",
        .train_start = 0,
        .train_tokens = train_tokens,
        .validation_start = train_tokens,
        .validation_tokens = validation_tokens
    };
    if (options.corpus_source_count == 0) {
        options.corpus_sources = &fallback_source;
        options.corpus_source_count = 1;
    }
    SourcePlan training_plan;
    if (!source_plan_init(&training_plan, options.corpus_sources,
            options.corpus_source_count, options.corpus_mix, false,
            options.sequence_length))
        die("no corpus source has enough tokens for the configured sequence length");

    Activations act;
    uint16_t* batch_tokens;
    uint16_t* batch_targets;
    float* dlogits;

    const int64_t initial_step = opt->step;
    if (!options.diagnostics_only && options.steps > 0 &&
        (int64_t)options.steps > opt->schedule_total_steps - opt->step)
        die("requested %d steps exceed the remaining LR schedule horizon (%lld steps remain)",
            options.steps, (long long)(opt->schedule_total_steps - opt->step));

    printf(" step | train loss | valid loss | learning rate | grad norm mean/max, clipped steps\n");
    printf("------+------------+------------+--------------+--------------------------------\n");
    FILE* metrics=fopen(options.metrics_path,"a");
    if (!metrics) die("could not open metrics CSV: %s",options.metrics_path);
    if (fseek(metrics,0,SEEK_END)!=0) die("could not seek metrics CSV");
    if (ftell(metrics)==0) { fprintf(metrics,"step,train_loss,validation_loss,learning_rate,train_perplexity,validation_perplexity,elapsed_seconds\n"); fflush(metrics); }
    const clock_t training_started=clock();
    printf("Metrics CSV: %s\n",options.metrics_path);
    printf("Settings: lr=%.7g schedule=%s warmup=%lld total_steps=%lld log=%d validation_every=%d val_batches=%d checkpoint=%d seed=%llu tokenizer_sample=%zu sampling=%s corpus_mix=%s grad_clip=%.3g batch=%d seq=%d\n",
        opt->schedule_base_lr,opt->schedule_type==0?"cosine":"constant",(long long)opt->schedule_warmup_steps,(long long)opt->schedule_total_steps,options.log_interval,options.validation_interval,options.validation_batches,
        options.checkpoint_interval,(unsigned long long)options.seed,options.tokenizer_sample_bytes,
        options.tokenizer_spread?"spread":"prefix",corpus_mix_name(options.corpus_mix),options.grad_clip_norm,options.batch_size,options.sequence_length);
    printf("Training data mixture (%s source weighting):\n", corpus_mix_name(options.corpus_mix));
    const double planned_training_tokens = options.diagnostics_only ? 0.0 :
        (double)options.steps * options.batch_size * options.sequence_length;
    for (size_t i = 0; i < training_plan.count; ++i) {
        const SourceChoice* choice = &training_plan.choices[i];
        const TrainSourceSpan* source = &options.corpus_sources[choice->source_index];
        const double share = choice->weight / training_plan.total_weight;
        const double estimated_passes = source->train_tokens > 0
            ? planned_training_tokens * share / source->train_tokens : 0.0;
        printf("  %s: %.1f%% (~%.1f source passes this segment)\n",
            source->name ? source->name : "corpus", 100.0 * share,
            estimated_passes);
    }
    printf("Estimated activations + logits scratch: %.2f MiB; KV cache: %.2f MiB\n",
        (double)options.activation_bytes/(1024.0*1024.0),
        2.0*model->n_layers*model->max_seq*model->kv_dim*sizeof(float)/(1024.0*1024.0));

    double train_loss_sum = 0.0;
    int train_loss_count = 0;
    double gradient_norm_sum = 0.0;
    double gradient_norm_max = 0.0;
    int gradient_norm_count = 0;
    int clipped_steps = 0;
    float last_validation_loss=NAN;

    if (options.diagnostics_only) {
        printf("Diagnostics-only mode: optimizer updates skipped.\n");
    }
    else for (int local_step = 0; local_step < options.steps; local_step++) {
        arena_reset(temp);
        if (!activations_init(&act, temp, options.batch_size, options.sequence_length, model))
            die("activation allocation failed");

        dlogits = arena_alloc(temp,
            (size_t)options.batch_size * options.sequence_length * model->vocab_size * sizeof(float));
        batch_tokens = arena_alloc(temp, options.batch_size * options.sequence_length * sizeof(uint16_t));
        batch_targets = arena_alloc(temp, options.batch_size * options.sequence_length * sizeof(uint16_t));
        if (!dlogits || !batch_tokens || !batch_targets)
            die("training scratch allocation failed");

        for (int b = 0; b < options.batch_size; b++) {
            const size_t source_index = source_plan_select(&training_plan,
                (double)rng_f32(rng));
            const TrainSourceSpan* source = &options.corpus_sources[source_index];
            const int max_start = source->train_tokens - options.sequence_length - 1;
            const int start = source->train_start + rng_int(rng, 0, max_start);
            for (int t = 0; t < options.sequence_length; t++) {
                batch_tokens[b * options.sequence_length + t] = all_ids[start + t];
                batch_targets[b * options.sequence_length + t] = all_ids[start + t + 1];
            }
        }

        float* logits = model_forward(model, &act,
            batch_tokens, options.batch_size, options.sequence_length);
        if (!logits) die("model_forward failed");

        const float loss = loss_cross_entropy(logits, batch_targets, dlogits,
            options.batch_size, options.sequence_length, model->vocab_size);
        if (!isfinite(loss)) die("training loss became non-finite");

        gradients_zero(grads, model);
        model_backward(model, grads, &act, dlogits,
            batch_tokens, options.batch_size, options.sequence_length);
        bool gradient_was_clipped = false;
        const double gradient_norm = gradients_clip_global_norm(grads, model,
            options.grad_clip_norm, &gradient_was_clipped);
        if (!isfinite(gradient_norm))
            die("gradient norm became non-finite; optimizer update was skipped");
        gradient_norm_sum += gradient_norm;
        if (gradient_norm > gradient_norm_max) gradient_norm_max = gradient_norm;
        gradient_norm_count++;
        if (gradient_was_clipped) clipped_steps++;

        // Use the next absolute optimizer step, independent of process restarts.
        const int64_t scheduled_step = opt->step + 1;
        opt->lr = opt->schedule_type == 1 ? opt->schedule_base_lr :
            optim_lr_cosine(opt->schedule_base_lr, scheduled_step,
                opt->schedule_warmup_steps, opt->schedule_total_steps);
        optim_step(opt, model, grads);
        const bool manual_save_requested=consume_file_signal(options.save_file);
        if (opt->step % options.checkpoint_interval == 0 || manual_save_requested) {
            if (!checkpoint_save("minima.model", model, opt, rng, tokenizer_hash))
                die("checkpoint save failed");
            printf("Checkpoint saved at optimizer step %lld\n",(long long)opt->step);
        }

        train_loss_sum += loss;
        train_loss_count++;

        if ((local_step + 1) % options.log_interval == 0 ||
            local_step == options.steps - 1 ||
            stop_file_exists(options.stop_file)) {
            if ((opt->step % options.validation_interval)==0 ||
                local_step==options.steps-1 || stop_file_exists(options.stop_file) ||
                !isfinite(last_validation_loss)) {
                last_validation_loss=evaluate_loss(model,temp,all_ids,
                    options.corpus_sources,options.corpus_source_count,
                    options.corpus_mix,true,options.validation_batches,
                    options.batch_size,options.sequence_length);
                if(!isfinite(last_validation_loss))die("validation loss evaluation failed");
            }
            const float validation_loss=last_validation_loss;

            const double average_train_loss =
                train_loss_sum / (double)train_loss_count;

            const double elapsed=(double)(clock()-training_started)/CLOCKS_PER_SEC;
            const double mean_gradient_norm = gradient_norm_count > 0
                ? gradient_norm_sum / gradient_norm_count : 0.0;
            printf("%5lld | %10.4f | %10.4f | %.7f | grad %.3f/%.3f clipped %d/%d\n",
                (long long)opt->step,(float)average_train_loss,validation_loss,
                opt->lr,mean_gradient_norm,gradient_norm_max,clipped_steps,
                gradient_norm_count);
            fprintf(metrics,"%lld,%.7f,%.7f,%.9g,%.7f,%.7f,%.3f\n",
                (long long)opt->step,(float)average_train_loss,validation_loss,opt->lr,
                expf((float)average_train_loss),expf(validation_loss),elapsed);
            fflush(metrics);
            train_loss_sum=0.0; train_loss_count=0;
            gradient_norm_sum=0.0; gradient_norm_max=0.0;
            gradient_norm_count=0; clipped_steps=0;
            if (stop_file_exists(options.stop_file)) {
                printf("Cooperative stop requested; saving checkpoint and stopping after optimizer step %lld.\n",
                    (long long)opt->step);
                break;
            }
        }
    }

    const int64_t final_step = opt->step;
    if (!options.diagnostics_only && (final_step == initial_step || final_step % options.checkpoint_interval != 0))
        if (!checkpoint_save("minima.model",model,opt,rng,tokenizer_hash)) die("final checkpoint save failed");
    const double training_seconds=(double)(clock()-training_started)/CLOCKS_PER_SEC;
    fclose(metrics);
    printf("\nTraining segment complete: %lld -> %lld\n",
        (long long)initial_step, (long long)final_step);

    const float final_train_loss = evaluate_loss(model, temp, all_ids,
        options.corpus_sources, options.corpus_source_count, options.corpus_mix,
        false, options.validation_batches, options.batch_size,
        options.sequence_length);
    const float final_validation_loss = evaluate_loss(model, temp, all_ids,
        options.corpus_sources, options.corpus_source_count, options.corpus_mix,
        true, options.validation_batches, options.batch_size,
        options.sequence_length);
    if (!isfinite(final_train_loss) || !isfinite(final_validation_loss))
        die("final loss diagnostics are unavailable");

    printf("\nFinal evaluation loss and perplexity (fixed, evenly spaced windows)\n");
    printf("  train loss      : %.4f\n", final_train_loss);
    printf("  train perplexity: %.2f\n", expf(final_train_loss));
    printf("  valid loss      : %.4f\n", final_validation_loss);
    printf("  valid perplexity: %.2f\n", expf(final_validation_loss));
    printf("  training duration: %.2f seconds\n",training_seconds);
    printf("  total optimizer steps: %lld\n",(long long)opt->step);
    printf("  checkpoint status: %s (minima.model)\n",
        options.diagnostics_only ? "unchanged" : "saved");


}
