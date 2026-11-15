#ifndef TRAIN_H
#define TRAIN_H

#include "arena.h"
#include "backward.h"
#include "model.h"
#include "optim.h"
#include "util.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define TRAIN_MAX_SOURCES 32u

typedef enum {
    CORPUS_MIX_PROPORTIONAL = 0,
    CORPUS_MIX_SQRT = 1,
    CORPUS_MIX_EQUAL = 2
} CorpusMixMode;

typedef struct {
    const char* name;
    int train_start;
    int train_tokens;
    int validation_start;
    int validation_tokens;
} TrainSourceSpan;

typedef struct {
    int steps;
    int log_interval;
    int validation_batches;
    int validation_interval;
    int checkpoint_interval;
    int batch_size;
    int sequence_length;
    float grad_clip_norm;
    uint64_t seed;
    size_t tokenizer_sample_bytes;
    bool tokenizer_spread;
    bool diagnostics_only;
    CorpusMixMode corpus_mix;
    const TrainSourceSpan* corpus_sources;
    size_t corpus_source_count;
    size_t activation_bytes;
    const char* metrics_path;
    const char* stop_file;
    const char* save_file;
} TrainRunOptions;

/* Validate the public training boundary before allocating scratch or
   using interval modulo operations. reason_out receives a static message. */
bool train_run_validate_inputs(const Model* model, const Optim* opt,
    const Gradients* grads, const Arena* temp, const uint16_t* all_ids,
    int train_tokens, int validation_tokens, const Rng* rng,
    TrainRunOptions options, const char** reason_out);

/* Run the configured training segment, including validation, metrics, and saves. */
void train_run(Model* model, Optim* opt, Gradients* grads, Arena* temp,
    const uint16_t* all_ids, int train_tokens, int validation_tokens,
    Rng* rng, uint64_t tokenizer_hash, TrainRunOptions options);

#endif
