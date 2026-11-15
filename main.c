#include "arena.h"
#include "util.h"
#include "tokenizer.h"
#include "model.h"
#include "optim.h"
#include "forward.h"
#include "backward.h"
#include "gradcheck.h"
#include "generate.h"
#include "corpus.h"
#include "mapped_buffer.h"
#include "checkpoint.h"
#include "config.h"
#include "train.h"
#include "chat.h"
#include "diagnostics.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define DEFAULT_BATCH_SIZE 2
#define DEFAULT_SEQUENCE_LENGTH 128
#define DEFAULT_STEPS 10000
#define DEFAULT_TOTAL_STEPS 100000
#define DEFAULT_LR 2e-4f
#define DEFAULT_LION_LR 3e-5f
#define DEFAULT_LION_WEIGHT_DECAY 0.1f
#define DEFAULT_WARMUP_STEPS 3000
#define DEFAULT_LOG_INTERVAL 1000
#define DEFAULT_VALIDATION_BATCHES 16
#define DEFAULT_CHECKPOINT_INTERVAL 1000
#define DEFAULT_TOKENIZER_SAMPLE_BYTES (4u * 1024u * 1024u)
#define DEFAULT_RAM_BUDGET_MB 10240
#define DEFAULT_THREADS 12

typedef struct {
    float learning_rate;
    float weight_decay;
    float grad_clip_norm;
    int total_steps;
    bool total_steps_explicit;
    int warmup_steps, log_interval, validation_batches, validation_interval, checkpoint_interval;
    int batch_size, sequence_length;
    uint64_t seed;
    size_t tokenizer_sample_bytes;
    int threads;
    int ram_budget_mb;
    const char* metrics_path;
    const char* stop_file;
    const char* save_file;
    float temperature, top_p, repetition_penalty;
    int top_k, max_new_tokens;
    bool chat_mode;
    ModelConfig model;
    const char* lr_schedule;
    CorpusMixMode corpus_mix;
    OptimizerType optimizer;
    bool optimizer_explicit;
    bool learning_rate_explicit;
    bool weight_decay_explicit;
    bool tokenizer_spread;
    bool tokenizer_only;
    bool use_existing_tokenizer;
} TrainOptions;
