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
#include <io.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#define DEFAULT_BATCH_SIZE 4
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

typedef struct {
    float learning_rate;
    float weight_decay;
    float beta1, beta2, epsilon;
    float grad_clip_norm;
    int total_steps;
    bool total_steps_explicit;
    int warmup_steps, log_interval, validation_batches, validation_interval, checkpoint_interval, checkpoint_retention;
    int batch_size, sequence_length;
    uint64_t seed;
    size_t tokenizer_sample_bytes;
    int threads;
    int memory_budget_gb;
    const char* metrics_path;
    const char* stop_file;
    const char* pause_file;
    const char* source_weights;
    const char* stop_sequence;
    const char* save_file;
    float temperature, top_p, repetition_penalty;
    float min_p, frequency_penalty, presence_penalty;
    int top_k, max_new_tokens, min_new_tokens, no_repeat_ngram;
    uint64_t sampling_seed;
    bool chat_mode;
    ModelConfig model;
    const char* lr_schedule;
    CorpusMixMode corpus_mix;
    OptimizerType optimizer;
    bool optimizer_explicit;
    bool learning_rate_explicit;
    bool weight_decay_explicit;
    bool beta1_explicit, beta2_explicit, epsilon_explicit;
    bool tokenizer_spread;
    bool tokenizer_only;
    bool use_existing_tokenizer;
} TrainOptions;

static long parse_long_value(const char* name, const char* value, long minv, long maxv) {
    char* end = NULL; errno = 0;
    long v = strtol(value, &end, 10);
    if (errno || !end || *end || v < minv || v > maxv)
        die("%s has invalid value: %s", name, value);
    return v;
}
static float parse_float_value(const char* name, const char* value, float minv, float maxv) {
    char* end = NULL; errno = 0;
    float v = strtof(value, &end);
    if (errno || !end || *end || !isfinite(v) || v < minv || v > maxv)
        die("%s has invalid value: %s", name, value);
    return v;
}

static uint64_t hash_bytes(uint64_t hash, const void* data, size_t length) {
    const uint8_t* bytes = (const uint8_t*)data;
    for (size_t i = 0; i < length; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static void print_usage(const char* exe) {
    printf("Usage: %s [options] [corpus.txt ...]\n", exe);
    printf("  --resume --steps N --total-steps N --lr FLOAT --warmup N --log-interval N\n");
    printf("  Defaults: 10,000 steps in a 100,000-step schedule; warmup 3,000\n");
    printf("  --validation-batches N --validation-interval N --checkpoint-interval N --seed N\n");
    printf("  --tokenizer-sample-bytes N --vocab-size N --dim N --layers N\n");
    printf("  --heads N --ffn-dim N --max-seq N --batch-size N --sequence-length N\n");
    printf("  --lr-schedule cosine|constant --tokenizer-sampling spread|prefix\n");
    printf("  --corpus-mix proportional|sqrt|equal (default: sqrt)\n");
    printf("  --optimizer adamw|lion (Lion uses less optimizer memory; tune --lr)\n");
    printf("  --weight-decay FLOAT --beta1 FLOAT --beta2 FLOAT --epsilon FLOAT\n");
    printf("  --memory-budget-gb N (aggregate arena + tracked heap budget; 0 disables)\n");
    printf("  --grad-clip FLOAT (global gradient norm; 0 disables clipping)\n");
    printf("  --temperature F --top-k N --top-p F --min-p F --repetition-penalty F\n");
    printf("  --frequency-penalty F --presence-penalty F --no-repeat-ngram N\n");
    printf("  --min-new-tokens N --sampling-seed N --stop-sequence TEXT\n");
    printf("  --max-new-tokens N --chat (requires --resume) --metrics PATH\n");
    printf("  --checkpoint-retention N (0-20 archived checkpoints; default 3)\n");
    printf("  --stop-file PATH --pause-file PATH --save-file PATH (cooperative training controls)\n");
    printf("  --source-weights CSV (positive multipliers, one per non-empty corpus file)\n");
    printf("  --retrain-tokenizer-only (rebuild minima.bpe from corpus and exit)\n");
    printf("  --use-existing-tokenizer (fresh model using a previously rebuilt minima.bpe)\n");
    printf("  --diagnostics-only --help\n");
}

static bool parse_source_weights_csv(const char* csv, double* weights,
    size_t capacity, size_t* count_out) {
    if (!csv || !weights || !count_out || capacity == 0) return false;
    const char* cursor = csv;
    size_t count = 0;
    while (*cursor) {
        while (*cursor == ' ' || *cursor == '\t') ++cursor;
        if (count >= capacity) return false;
        errno = 0;
        char* end = NULL;
        const double value = strtod(cursor, &end);
        if (end == cursor || errno == ERANGE || !isfinite(value) ||
            value < 0.001 || value > 1000.0) return false;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\t') ++cursor;
        weights[count++] = value;
        if (!*cursor) break;
        if (*cursor != ',') return false;
        ++cursor;
        if (!*cursor) return false;
    }
    if (count == 0) return false;
    *count_out = count;
    return true;
}

static void write_json_string(FILE* file, const char* value) {
    fputc('"', file);
    if (value) {
        for (const unsigned char* p = (const unsigned char*)value; *p; ++p) {
            switch (*p) {
            case '"': fputs("\\\"", file); break;
            case '\\': fputs("\\\\", file); break;
            case '\n': fputs("\\n", file); break;
            case '\r': fputs("\\r", file); break;
            case '\t': fputs("\\t", file); break;
            default:
                if (*p < 0x20) fprintf(file, "\\u%04x", (unsigned int)*p);
                else fputc(*p, file);
                break;
            }
        }
    }
    fputc('"', file);
}

static const char* corpus_mix_name(CorpusMixMode mode) {
    switch (mode) {
    case CORPUS_MIX_PROPORTIONAL: return "proportional";
    case CORPUS_MIX_EQUAL: return "equal";
    case CORPUS_MIX_SQRT:
    default: return "sqrt";
    }
}

static bool write_run_metadata(const char* metrics_path,
    const TrainOptions* options, const Optim* optimizer,
    const TrainSourceSpan* sources, size_t source_count,
    int requested_steps, bool resumed, uint64_t tokenizer_hash) {
    if (!metrics_path || !metrics_path[0] || !options || !optimizer ||
        source_count > TRAIN_MAX_SOURCES || (source_count && !sources))
        return false;

    char metadata_path[4096];
    if (_snprintf_s(metadata_path, sizeof(metadata_path), _TRUNCATE,
            "%s.meta.json", metrics_path) < 0)
        return false;
    FILE* file = fopen(metadata_path, "wb");
    if (!file) return false;

    const char* optimizer_name = optimizer->type == OPTIMIZER_LION ? "lion" : "adamw";
    const char* schedule_name = optimizer->schedule_type == 1 ? "constant" : "cosine";
    const time_t started = time(NULL);
    int threads = options->threads;
#ifdef _OPENMP
    if (threads == 0) threads = omp_get_max_threads();
#endif

    bool ok = true;
#define META_PRINT(...) do { if (ok && fprintf(file, __VA_ARGS__) < 0) ok = false; } while (0)
    META_PRINT("{\n  \"schema_version\": 1,\n  \"started_unix\": %lld,\n  \"metrics_path\": ",
        (long long)started);
    write_json_string(file, metrics_path);
    META_PRINT(",\n  \"tokenizer_hash\": \"%016llx\",\n  \"resumed\": %s,\n",
        (unsigned long long)tokenizer_hash, resumed ? "true" : "false");
    META_PRINT("  \"model\": {\"vocab_size\": %d, \"dim\": %d, \"layers\": %d, \"heads\": %d, \"ffn_dim\": %d, \"max_seq\": %d},\n",
        options->model.vocab_size, options->model.dim, options->model.n_layers,
        options->model.n_heads, options->model.ffn_dim, options->model.max_seq);
    META_PRINT("  \"training\": {\"optimizer\": \"%s\", \"learning_rate\": %.9g, \"weight_decay\": %.9g, \"beta1\": %.9g, \"beta2\": %.9g, \"epsilon\": %.9g, \"grad_clip_norm\": %.9g, \"lr_schedule\": \"%s\", \"initial_step\": %lld, \"requested_steps\": %d, \"total_steps\": %lld, \"warmup_steps\": %lld, \"batch_size\": %d, \"sequence_length\": %d, \"seed\": %llu, \"threads\": %d, \"memory_budget_gb\": %d, \"checkpoint_retention\": %d, \"log_interval\": %d, \"validation_batches\": %d, \"validation_interval\": %d, \"checkpoint_interval\": %d, \"tokenizer_sample_bytes\": %zu, \"tokenizer_spread\": %s},\n",
        optimizer_name, optimizer->schedule_base_lr, optimizer->weight_decay,
        optimizer->beta1, optimizer->beta2, optimizer->eps, options->grad_clip_norm,
        schedule_name, (long long)optimizer->step, requested_steps,
        (long long)optimizer->schedule_total_steps,
        (long long)optimizer->schedule_warmup_steps, options->batch_size,
        options->sequence_length, (unsigned long long)options->seed, threads,
        options->memory_budget_gb, options->checkpoint_retention,
        options->log_interval, options->validation_batches,
        options->validation_interval, options->checkpoint_interval,
        options->tokenizer_sample_bytes, options->tokenizer_spread ? "true" : "false");
    META_PRINT("  \"data\": {\"mix\": \"%s\", \"source_count\": %zu, \"sources\": [",
        corpus_mix_name(options->corpus_mix), source_count);
    for (size_t i = 0; ok && i < source_count; ++i) {
        if (i) META_PRINT(",");
        META_PRINT("{\"name\": ");
        write_json_string(file, sources[i].name);
        META_PRINT(", \"weight\": %.9g, \"train_tokens\": %d, \"validation_tokens\": %d}",
            sources[i].weight, sources[i].train_tokens, sources[i].validation_tokens);
    }
    META_PRINT("]}\n}\n");
#undef META_PRINT
    if (ok && fflush(file) != 0) ok = false;
    if (ok && _commit(_fileno(file)) != 0) ok = false;
    if (fclose(file) != 0) ok = false;
    if (!ok) { remove(metadata_path); return false; }
    printf("Run metadata: %s\n", metadata_path);
    return true;
}

static int parse_arguments(int argc, char** argv, bool* resume,
    bool* diagnostics_only, int* steps, const char** corpus_paths, TrainOptions* o) {
    int path_count = 0;
    *resume = false; *diagnostics_only = false; *steps = DEFAULT_STEPS;
    ModelConfig default_model=MODEL_CONFIG_DEFAULT;
    o->model=default_model;
    o->batch_size=DEFAULT_BATCH_SIZE; o->sequence_length=DEFAULT_SEQUENCE_LENGTH;
    o->learning_rate=DEFAULT_LR; o->warmup_steps=DEFAULT_WARMUP_STEPS;
    o->weight_decay=0.01f; o->weight_decay_explicit=false;
    o->beta1=0.9f; o->beta2=0.95f; o->epsilon=1e-8f;
    o->beta1_explicit=false; o->beta2_explicit=false; o->epsilon_explicit=false;
    o->grad_clip_norm=1.0f;
    o->log_interval=DEFAULT_LOG_INTERVAL; o->validation_batches=DEFAULT_VALIDATION_BATCHES;
    o->validation_interval=DEFAULT_LOG_INTERVAL;
    o->checkpoint_interval=DEFAULT_CHECKPOINT_INTERVAL; o->checkpoint_retention=3; o->seed=1234;
    o->tokenizer_sample_bytes=DEFAULT_TOKENIZER_SAMPLE_BYTES;
    o->threads=0; o->memory_budget_gb=0;
    o->metrics_path="minima_metrics.csv"; o->stop_file=NULL; o->pause_file=NULL; o->source_weights=NULL; o->save_file=NULL;
    o->temperature=0.8f; o->top_k=40; o->top_p=0.9f;
    o->min_p=0.0f; o->frequency_penalty=0.0f; o->presence_penalty=0.0f;
    o->min_new_tokens=0; o->no_repeat_ngram=0; o->sampling_seed=0; o->stop_sequence=NULL;
    o->repetition_penalty=1.1f; o->max_new_tokens=32; o->chat_mode=false;
    o->lr_schedule="cosine"; o->tokenizer_spread=true;
    o->corpus_mix=CORPUS_MIX_SQRT;
    o->optimizer=OPTIMIZER_ADAMW; o->optimizer_explicit=false;
    o->learning_rate_explicit=false;
    o->total_steps=DEFAULT_TOTAL_STEPS; o->total_steps_explicit=false;
    o->tokenizer_only=false; o->use_existing_tokenizer=false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
        else if (strcmp(argv[i], "--resume") == 0) {
            *resume = true;
        }
        else if (strcmp(argv[i], "--diagnostics-only") == 0) {
            *diagnostics_only = true;
        }
        else if (strcmp(argv[i], "--retrain-tokenizer-only") == 0) {
            o->tokenizer_only = true;
        }
        else if (strcmp(argv[i], "--use-existing-tokenizer") == 0) {
            o->use_existing_tokenizer = true;
        }
        else if (strcmp(argv[i], "--chat") == 0) o->chat_mode=true;
        else if (strcmp(argv[i], "--steps") == 0 ||
            strcmp(argv[i], "--warmup") == 0 ||
            strcmp(argv[i], "--total-steps") == 0 ||
            strcmp(argv[i], "--log-interval") == 0 ||
            strcmp(argv[i], "--validation-batches") == 0 ||
            strcmp(argv[i], "--validation-interval") == 0 ||
            strcmp(argv[i], "--checkpoint-interval") == 0 ||
            strcmp(argv[i], "--checkpoint-retention") == 0 ||
            strcmp(argv[i], "--seed") == 0 ||
            strcmp(argv[i], "--tokenizer-sample-bytes") == 0 ||
            strcmp(argv[i], "--threads") == 0 ||
            strcmp(argv[i], "--memory-budget-gb") == 0 ||
            strcmp(argv[i], "--vocab-size") == 0 ||
            strcmp(argv[i], "--dim") == 0 ||
            strcmp(argv[i], "--layers") == 0 ||
            strcmp(argv[i], "--heads") == 0 ||
            strcmp(argv[i], "--ffn-dim") == 0 ||
            strcmp(argv[i], "--max-seq") == 0 ||
            strcmp(argv[i], "--batch-size") == 0 ||
            strcmp(argv[i], "--sequence-length") == 0) {
            const char* opt=argv[i];
            if (++i >= argc) die("%s requires a value", opt);
            long lo=1, hi=INT_MAX;
            if(strcmp(opt,"--validation-batches")==0)hi=4096;
            if(strcmp(opt,"--checkpoint-retention")==0) { lo=0; hi=20; }
            if (strcmp(opt,"--seed")==0) { lo=0; hi=LONG_MAX; }
            if (strcmp(opt,"--tokenizer-sample-bytes")==0) { lo=4096; hi=4194304; }
            if (strcmp(opt,"--threads")==0) { lo=0; hi=256; }
            if (strcmp(opt,"--memory-budget-gb")==0) { lo=0; hi=64; }
            if (strcmp(opt,"--vocab-size")==0) { lo=260; hi=MODEL_MAX_VOCAB_SIZE; }
            if (strcmp(opt,"--dim")==0) { lo=1; hi=512; }
            if (strcmp(opt,"--layers")==0) { lo=1; hi=16; }
            if (strcmp(opt,"--heads")==0) { lo=1; hi=32; }
            if (strcmp(opt,"--ffn-dim")==0) { lo=1; hi=1024; }
            if (strcmp(opt,"--max-seq")==0) { lo=32; hi=512; }
            if (strcmp(opt,"--batch-size")==0) { lo=1; hi=16; }
            if (strcmp(opt,"--sequence-length")==0) { lo=8; hi=512; }
            long v=parse_long_value(opt,argv[i],lo,hi);
            if (strcmp(opt,"--steps")==0) *steps=(int)v;
            else if (strcmp(opt,"--total-steps")==0) { o->total_steps=(int)v; o->total_steps_explicit=true; }
            else if (strcmp(opt,"--warmup")==0) o->warmup_steps=(int)v;
            else if (strcmp(opt,"--log-interval")==0) o->log_interval=(int)v;
            else if (strcmp(opt,"--validation-batches")==0) o->validation_batches=(int)v;
            else if (strcmp(opt,"--validation-interval")==0) o->validation_interval=(int)v;
            else if (strcmp(opt,"--checkpoint-interval")==0) o->checkpoint_interval=(int)v;
            else if (strcmp(opt,"--checkpoint-retention")==0) o->checkpoint_retention=(int)v;
            else if (strcmp(opt,"--seed")==0) o->seed=(uint64_t)v;
            else if (strcmp(opt,"--vocab-size")==0) o->model.vocab_size=(int)v;
            else if (strcmp(opt,"--dim")==0) o->model.dim=(int)v;
            else if (strcmp(opt,"--layers")==0) o->model.n_layers=(int)v;
            else if (strcmp(opt,"--heads")==0) o->model.n_heads=(int)v;
            else if (strcmp(opt,"--ffn-dim")==0) o->model.ffn_dim=(int)v;
            else if (strcmp(opt,"--max-seq")==0) o->model.max_seq=(int)v;
            else if (strcmp(opt,"--batch-size")==0) o->batch_size=(int)v;
            else if (strcmp(opt,"--sequence-length")==0) o->sequence_length=(int)v;
            else if (strcmp(opt,"--threads")==0) o->threads=(int)v;
            else if (strcmp(opt,"--memory-budget-gb")==0) o->memory_budget_gb=(int)v;
            else o->tokenizer_sample_bytes=(size_t)v;
        }
        else if (strcmp(argv[i], "--top-k") == 0 || strcmp(argv[i], "--max-new-tokens") == 0 ||
            strcmp(argv[i], "--min-new-tokens") == 0 || strcmp(argv[i], "--no-repeat-ngram") == 0 ||
            strcmp(argv[i], "--sampling-seed") == 0) {
            const char* option=argv[i]; if(++i>=argc)die("%s requires a value",option);
            long minv = (strcmp(option,"--min-new-tokens")==0 || strcmp(option,"--no-repeat-ngram")==0 || strcmp(option,"--sampling-seed")==0) ? 0 : 1;
            long maxv = strcmp(option,"--top-k")==0 ? 256 :
                (strcmp(option,"--no-repeat-ngram")==0 ? 16 :
                 (strcmp(option,"--sampling-seed")==0 ? LONG_MAX : 512));
            long v=parse_long_value(option,argv[i],minv,maxv);
            if(strcmp(option,"--top-k")==0)o->top_k=(int)v;
            else if(strcmp(option,"--max-new-tokens")==0)o->max_new_tokens=(int)v;
            else if(strcmp(option,"--min-new-tokens")==0)o->min_new_tokens=(int)v;
            else if(strcmp(option,"--no-repeat-ngram")==0)o->no_repeat_ngram=(int)v;
            else o->sampling_seed=(uint64_t)v;
        }
        else if (strcmp(argv[i], "--temperature") == 0 ||
            strcmp(argv[i], "--top-p") == 0 ||
            strcmp(argv[i], "--repetition-penalty") == 0 ||
            strcmp(argv[i], "--min-p") == 0 ||
            strcmp(argv[i], "--frequency-penalty") == 0 ||
            strcmp(argv[i], "--presence-penalty") == 0) {
            const char* option=argv[i];if(++i>=argc)die("%s requires a value",option);
            if(strcmp(option,"--temperature")==0)o->temperature=parse_float_value(option,argv[i],0.05f,5.0f);
            else if(strcmp(option,"--top-p")==0)o->top_p=parse_float_value(option,argv[i],0.01f,1.0f);
            else if(strcmp(option,"--repetition-penalty")==0)o->repetition_penalty=parse_float_value(option,argv[i],1.0f,3.0f);
            else if(strcmp(option,"--min-p")==0)o->min_p=parse_float_value(option,argv[i],0.0f,1.0f);
            else if(strcmp(option,"--frequency-penalty")==0)o->frequency_penalty=parse_float_value(option,argv[i],-2.0f,2.0f);
            else o->presence_penalty=parse_float_value(option,argv[i],-2.0f,2.0f);
        }
        else if (strcmp(argv[i], "--lr-schedule") == 0) {
            if(++i>=argc)die("--lr-schedule requires cosine or constant");
            if(strcmp(argv[i],"cosine")!=0 && strcmp(argv[i],"constant")!=0)
                die("--lr-schedule must be cosine or constant");
            o->lr_schedule=argv[i];
        }
        else if (strcmp(argv[i], "--corpus-mix") == 0) {
            if (++i >= argc) die("--corpus-mix requires proportional, sqrt, or equal");
            if (strcmp(argv[i], "proportional") == 0)
                o->corpus_mix = CORPUS_MIX_PROPORTIONAL;
            else if (strcmp(argv[i], "sqrt") == 0)
                o->corpus_mix = CORPUS_MIX_SQRT;
            else if (strcmp(argv[i], "equal") == 0)
                o->corpus_mix = CORPUS_MIX_EQUAL;
            else
                die("--corpus-mix must be proportional, sqrt, or equal");
        }
        else if (strcmp(argv[i], "--optimizer") == 0) {
            if (++i >= argc) die("--optimizer requires adamw or lion");
            if (strcmp(argv[i], "adamw") == 0)
                o->optimizer = OPTIMIZER_ADAMW;
            else if (strcmp(argv[i], "lion") == 0)
                o->optimizer = OPTIMIZER_LION;
            else
                die("--optimizer must be adamw or lion");
            o->optimizer_explicit = true;
        }
        else if (strcmp(argv[i], "--tokenizer-sampling") == 0) {
            if(++i>=argc)die("--tokenizer-sampling requires spread or prefix");
            if(strcmp(argv[i],"spread")!=0 && strcmp(argv[i],"prefix")!=0)
                die("--tokenizer-sampling must be spread or prefix");
            o->tokenizer_spread=(strcmp(argv[i],"spread")==0);
        }
        else if (strcmp(argv[i], "--lr") == 0) {
            if (++i >= argc) die("--lr requires a value");
            o->learning_rate=parse_float_value("--lr",argv[i],1e-7f,1.0f);
            o->learning_rate_explicit=true;
        }
        else if (strcmp(argv[i], "--grad-clip") == 0) {
            if (++i >= argc) die("--grad-clip requires a value");
            o->grad_clip_norm=parse_float_value("--grad-clip",argv[i],0.0f,1000.0f);
        }
        else if (strcmp(argv[i], "--weight-decay") == 0) {
            if (++i >= argc) die("--weight-decay requires a value");
            o->weight_decay=parse_float_value("--weight-decay",argv[i],0.0f,1.0f);
            o->weight_decay_explicit=true;
        }
        else if (strcmp(argv[i], "--beta1") == 0 ||
            strcmp(argv[i], "--beta2") == 0 ||
            strcmp(argv[i], "--epsilon") == 0) {
            const char* option = argv[i];
            if (++i >= argc) die("%s requires a value", option);
            if (strcmp(option, "--beta1") == 0) {
                o->beta1 = parse_float_value(option, argv[i], 0.0f, 0.999999f);
                o->beta1_explicit = true;
            } else if (strcmp(option, "--beta2") == 0) {
                o->beta2 = parse_float_value(option, argv[i], 0.0f, 0.999999f);
                o->beta2_explicit = true;
            } else {
                o->epsilon = parse_float_value(option, argv[i], 1e-12f, 1e-2f);
                o->epsilon_explicit = true;
            }
        }
        else if (strcmp(argv[i], "--metrics") == 0 || strcmp(argv[i], "--stop-file") == 0 || strcmp(argv[i], "--pause-file") == 0 || strcmp(argv[i], "--source-weights") == 0 || strcmp(argv[i], "--stop-sequence") == 0 || strcmp(argv[i], "--save-file") == 0) {
            const char* option=argv[i];
            if (++i >= argc || !argv[i][0]) die("%s requires a value",option);
            if (strcmp(option,"--metrics")==0) o->metrics_path=argv[i];
            else if(strcmp(option,"--stop-file")==0)o->stop_file=argv[i];
            else if(strcmp(option,"--pause-file")==0)o->pause_file=argv[i];
            else if(strcmp(option,"--source-weights")==0)o->source_weights=argv[i];
            else if(strcmp(option,"--stop-sequence")==0)o->stop_sequence=argv[i];
            else o->save_file=argv[i];
        }
        else if (argv[i][0] == '-') {
            die("unknown option: %s (use --help)", argv[i]);
        }
        else {
            if (path_count >= 32)
                die("too many corpus paths (maximum 32)");
            corpus_paths[path_count++] = argv[i];
        }
    }
    if (o->chat_mode && !*resume) die("--chat requires --resume");
    if (o->tokenizer_only && (*resume || o->chat_mode || *diagnostics_only || o->use_existing_tokenizer))
        die("--retrain-tokenizer-only cannot be combined with resume, chat, diagnostics-only, or --use-existing-tokenizer");
    if (o->use_existing_tokenizer && (*resume || o->chat_mode || *diagnostics_only))
        die("--use-existing-tokenizer is for fresh training only");
    if (o->min_new_tokens > o->max_new_tokens)
        die("--min-new-tokens cannot exceed --max-new-tokens");

    if (o->optimizer == OPTIMIZER_LION) {
        if (!o->beta1_explicit) o->beta1 = 0.95f;
        if (!o->beta2_explicit) o->beta2 = 0.98f;
        if (!o->learning_rate_explicit) o->learning_rate = DEFAULT_LION_LR;
        if (!o->weight_decay_explicit)
            o->weight_decay = DEFAULT_LION_WEIGHT_DECAY;
    }
    if (*diagnostics_only) {
        if (!*resume)
            die("--diagnostics-only requires --resume");
        *steps = 0;
    }
    return path_count;
}

int main(int argc, char** argv) {
    /* Keep redirected GUI output responsive instead of fully buffered. */
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    bool resume = false;
    bool diagnostics_only = false;
    int steps_to_run = DEFAULT_STEPS;
    const char* user_paths[32];
    TrainOptions options;
    const int user_path_count = parse_arguments(argc, argv,
        &resume, &diagnostics_only, &steps_to_run, user_paths, &options);
    OptimizerType optimizer_type = options.optimizer;
    if (resume && !options.optimizer_explicit &&
        !checkpoint_read_optimizer_type("minima.model", &optimizer_type))
        die("could not read optimizer type from checkpoint; specify --optimizer adamw|lion");
#ifdef _OPENMP
    omp_set_dynamic(0);
    if (options.threads > 0) omp_set_num_threads(options.threads);
    printf("CPU parallelism: OpenMP enabled, %d worker threads\n", omp_get_max_threads());
#else
    if (options.threads > 0)
        printf("Warning: this build has no OpenMP support; --threads is ignored.\n");
#endif
    if(options.sequence_length>options.model.max_seq)
        die("sequence length cannot exceed model context length");
    const size_t training_tokens=(size_t)options.batch_size*(size_t)options.sequence_length;
    const size_t activation_floats=training_tokens*(
        (size_t)(9+6*options.model.n_layers)*(size_t)options.model.dim +
        (size_t)(3*options.model.n_layers+3)*(size_t)options.model.ffn_dim +
        (size_t)options.model.vocab_size) +
        (size_t)options.sequence_length;
    const size_t activation_bytes=activation_floats*sizeof(float);
    if(activation_bytes>192u*1024u*1024u)
        die("estimated activations/logits require %.1f MiB; the temporary arena safety limit is 192 MiB",
            (double)activation_bytes/(1024.0*1024.0));

    printf("ProjectMinima text-corpus training experiment\n");
    printf("Vocabulary target: %d\n", options.model.vocab_size);
    if (options.chat_mode)
        printf("Mode: interactive chat (checkpoint resume)\n\n");
    else if (diagnostics_only)
        printf("Mode: diagnostics only (no optimizer updates)\n\n");
    else if (options.tokenizer_only)
        printf("Mode: tokenizer rebuild only (no model training)\n\n");
    else
        printf("Steps this run: %d%s\n\n", steps_to_run,
            resume ? " (resume)" : "");

    if (!optim_check_adamw())
        die("AdamW numerical regression check failed");
    printf("AdamW bias-correction check: PASS\n");
    if (!optim_check_lr_schedule()) die("global-step LR schedule regression check failed");
    printf("Global-step LR schedule check: PASS (split continuity, warmup, cosine endpoint)\n");

    // ------------------------------------------------------------------
    // 1. Arenas
    // ------------------------------------------------------------------
    Arena arena;
    Arena temp;
    Arena corpus_arena;
    ArenaBudget memory_budget;
    const size_t memory_budget_bytes = options.memory_budget_gb > 0
        ? (size_t)options.memory_budget_gb * 1024u * 1024u * 1024u : 0;
    arena_budget_init(&memory_budget, memory_budget_bytes);
    memset(&corpus_arena, 0, sizeof(corpus_arena));

    if (!arena_init_with_budget(&arena, 8ull * 1024 * 1024 * 1024, &memory_budget))
        die("permanent arena failed");
    if (!arena_init_with_budget(&temp, 256ull * 1024 * 1024, &memory_budget))
        die("temp arena failed");
    if (!options.tokenizer_only && !options.chat_mode &&
        !arena_init_with_budget(&corpus_arena, 8ull * 1024 * 1024 * 1024, &memory_budget))
        die("corpus arena failed");

    printf("Engine memory budget: %s\n", options.memory_budget_gb > 0 ? "enabled" : "unlimited");
    if (options.memory_budget_gb > 0)
        printf("  Aggregate arena + tracked heap limit: %d GiB\n", options.memory_budget_gb);

    // ------------------------------------------------------------------
    // 2. Select corpus sources and sample the tokenizer directly from disk
    // ------------------------------------------------------------------
    const char* project_paths[] = {
        "../data/books.txt",
        "../data/chatlogs.txt",
        "../data/britannica.txt"
    };
    const char* root_paths[] = {
        "data/books.txt",
        "data/chatlogs.txt",
        "data/britannica.txt"
    };
    const char* const* corpus_paths = user_path_count > 0
        ? user_paths : project_paths;
    size_t corpus_path_count = user_path_count > 0
        ? (size_t)user_path_count : 3u;
    const bool default_paths = user_path_count == 0;

    uint8_t* tokenizer_sample = NULL;
    size_t tokenizer_sample_len = 0;
    size_t tokenizer_sample_segment_ends[CORPUS_MAX_SAMPLE_SEGMENTS];
    size_t tokenizer_sample_segment_count = 0;
    size_t tokenizer_sample_files = 0;
    size_t tokenizer_source_bytes = 0;

    if (!options.chat_mode && !resume && !options.use_existing_tokenizer) {
        CorpusLoadResult sample_result = corpus_sample_files_segmented(&temp,
            corpus_paths, corpus_path_count, options.tokenizer_sample_bytes,
            options.tokenizer_spread, &tokenizer_sample,
            &tokenizer_sample_len, tokenizer_sample_segment_ends,
            CORPUS_MAX_SAMPLE_SEGMENTS, &tokenizer_sample_segment_count,
            &tokenizer_sample_files,
            &tokenizer_source_bytes);

        if (sample_result == CORPUS_LOAD_EMPTY && default_paths) {
            corpus_paths = root_paths;
            sample_result = corpus_sample_files_segmented(&temp, corpus_paths,
                corpus_path_count, options.tokenizer_sample_bytes,
                options.tokenizer_spread, &tokenizer_sample,
                &tokenizer_sample_len, tokenizer_sample_segment_ends,
                CORPUS_MAX_SAMPLE_SEGMENTS, &tokenizer_sample_segment_count,
                &tokenizer_sample_files,
                &tokenizer_source_bytes);
        }
        if (sample_result == CORPUS_LOAD_ERROR)
            die("streaming tokenizer sampling failed (check corpus file access and available memory)");
        if (sample_result == CORPUS_LOAD_EMPTY && options.tokenizer_only)
            die("tokenizer rebuild requires actual corpus text; synthetic smoke text is not allowed");
        if (sample_result == CORPUS_LOAD_OK) {
            printf("Tokenizer sample sources: %zu files, %zu source bytes\n",
                tokenizer_sample_files, tokenizer_source_bytes);
            printf("Tokenizer training sample: %zu bytes in %zu independent pieces (source-balanced, %s)\n",
                tokenizer_sample_len,
                tokenizer_sample_segment_count,
                options.tokenizer_spread ? "spread" : "prefix");
        }
    }

    // Tokenizer-only mode deliberately does not load the complete corpus.
    // This keeps a 900+ MiB source set out of RAM while rebuilding minima.bpe.
    const size_t max_corpus_bytes = SIZE_MAX;
    Corpus loaded_corpus;
    bool have_corpus = false;
    if (!options.tokenizer_only && !options.chat_mode) {
        const CorpusLoadResult load_result = corpus_load_files(&loaded_corpus,
            &corpus_arena, corpus_paths, corpus_path_count, max_corpus_bytes);
        if (load_result == CORPUS_LOAD_ERROR)
            die("corpus loading failed (check file access, available memory, and supported corpus size)");
        have_corpus = load_result == CORPUS_LOAD_OK;
        if (!have_corpus && user_path_count > 0)
            die("the supplied corpus files contained no usable text");
    }

    uint8_t* corpus = NULL;
    size_t corpus_len = 0;
    if (have_corpus) {
        corpus = loaded_corpus.data;
        corpus_len = loaded_corpus.length;
        printf("\nCorpus: %zu bytes from %zu file(s)\n",
            corpus_len, loaded_corpus.files_loaded);
        printf("Estimated token-ID backing allocation: %.1f MiB (worst case)\n",
            (double)corpus_len * sizeof(uint16_t) / (1024.0 * 1024.0));
    }
    else if (!options.tokenizer_only) {
        const char* smoke_text =
            "The quick brown fox jumps over the lazy dog. "
            "Hello world from a tiny language model. "
            "Pure C is enough for small transformers. ";
        const size_t smoke_len = strlen(smoke_text);
        corpus_len = smoke_len * 80;
        corpus = (uint8_t*)arena_alloc(&arena, corpus_len + 1);
        if (!corpus) die("smoke corpus allocation failed");
        for (size_t i = 0; i < 80; i++)
            memcpy(corpus + i * smoke_len, smoke_text, smoke_len);
        corpus[corpus_len] = 0;
        if (options.chat_mode)
            printf("Chat mode: corpus loading skipped; using a tiny placeholder buffer for tokenizer setup.\n");
        else
            printf("No training text found; using synthetic smoke corpus.\n");
    }

    if (!options.tokenizer_only) {
        if (corpus_len < (size_t)options.batch_size * options.sequence_length + 1)
            die("corpus too small: need at least %d bytes",
                options.batch_size * options.sequence_length + 1);
        if (corpus_len > (size_t)INT_MAX)
            die("corpus exceeds tokenizer's current INT_MAX byte/token-capacity limit");
    }

    // ------------------------------------------------------------------
    // 3. Load or train tokenizer
    // ------------------------------------------------------------------
    static Tokenizer tok;
    if (resume || options.use_existing_tokenizer) {
        if (!tokenizer_load(&tok, &arena, "minima.bpe"))
            die(resume ? "resume requested, but minima.bpe could not be loaded" :
                "--use-existing-tokenizer requested, but minima.bpe could not be loaded");
        if (tok.vocab_size != options.model.vocab_size)
            die("saved tokenizer vocabulary does not match the configured model vocabulary; rebuild it or change the vocabulary setting");
        printf("Loaded %s tokenizer: minima.bpe (vocab = %d)\n",
            resume ? "checkpoint" : "prebuilt", tok.vocab_size);
    }
    else {
        if (!tokenizer_sample) {
            if (options.tokenizer_only)
                die("tokenizer rebuild requires a non-empty corpus sample");
            const size_t train_corpus_len = (corpus_len * 95u) / 100u;
            if (train_corpus_len < (size_t)options.batch_size * options.sequence_length + 1)
                die("training portion is too small");
            tokenizer_sample_len = train_corpus_len < options.tokenizer_sample_bytes
                ? train_corpus_len : options.tokenizer_sample_bytes;
            tokenizer_sample = (uint8_t*)arena_alloc(&arena, tokenizer_sample_len);
            if (!tokenizer_sample) die("tokenizer sample allocation failed");
            memcpy(tokenizer_sample, corpus, tokenizer_sample_len);
            tokenizer_sample_segment_ends[0] = tokenizer_sample_len;
            tokenizer_sample_segment_count = 1;
        }

        const size_t sample_len = tokenizer_sample_len;
        printf("Tokenizer trainer sample: %zu bytes\n", sample_len);
        printf("Estimated BPE workspace: approximately %.1f MiB (conservative)\n",
            ((double)sample_len * 230.0 + 3.0 * 1024.0 * 1024.0) /
                (1024.0 * 1024.0));
        if (!tokenizer_train_segments(&tok, &arena, tokenizer_sample,
            sample_len, tokenizer_sample_segment_ends,
            tokenizer_sample_segment_count, options.model.vocab_size))
            die("tokenizer_train failed");
        printf("Tokenizer preflight: valid tokenizer produced %d / %d requested vocabulary entries.\n",
            tok.vocab_size, options.model.vocab_size);
        if (tok.vocab_size != options.model.vocab_size)
            die("valid tokenizer did not reach the exact configured vocabulary target (%d / %d); this run treats --vocab-size as a hard requirement. Increase or diversify the tokenizer sample, or lower --vocab-size",
                tok.vocab_size, options.model.vocab_size);
        if (!tokenizer_save(&tok, "minima.bpe"))
            die("failed to save tokenizer to minima.bpe");
    }

    const uint64_t tokenizer_hash = tokenizer_fingerprint(&tok);
    if (!tokenizer_hash) die("failed to fingerprint tokenizer");
    printf("vocab size = %d\n", tok.vocab_size);
    int max_token_bytes=0;
    double total_token_bytes=0.0;
    for (int ti=0; ti<tok.vocab_size; ++ti) {
        if (tok.token_len[ti] > max_token_bytes) max_token_bytes=tok.token_len[ti];
        total_token_bytes += tok.token_len[ti];
    }
    printf("Tokenizer statistics: merges=%d, max_token_bytes=%d, mean_token_bytes=%.2f\n",
        tok.n_merges,max_token_bytes,total_token_bytes/(double)tok.vocab_size);

    if (options.tokenizer_only) {
        printf("Tokenizer rebuild completed: minima.bpe was replaced.\n");
        printf("The existing model checkpoint is now incompatible; start a fresh training run.\n");
        arena_free(&corpus_arena);
        arena_free(&temp);
        arena_free(&arena);
        return 0;
    }

    // ------------------------------------------------------------------
    // 4. Encode each source and create source-local held-out regions
    // ------------------------------------------------------------------
    const size_t source_count = have_corpus ? loaded_corpus.source_count : 1u;
    double source_weights[TRAIN_MAX_SOURCES];
    for (size_t i = 0; i < TRAIN_MAX_SOURCES; ++i) source_weights[i] = 1.0;
    if (options.source_weights) {
        size_t parsed_weight_count = 0;
        if (!parse_source_weights_csv(options.source_weights, source_weights,
                TRAIN_MAX_SOURCES, &parsed_weight_count))
            die("--source-weights must be a comma-separated list of 0.001..1000 multipliers");
        if (parsed_weight_count != source_count)
            die("--source-weights needs one multiplier for each non-empty corpus file (%zu expected, %zu supplied)",
                source_count, parsed_weight_count);
    }
    if (source_count == 0 || source_count > TRAIN_MAX_SOURCES ||
        source_count > SIZE_MAX / 2u ||
        corpus_len > SIZE_MAX - 2u * source_count)
        die("corpus source metadata or token mapping size is invalid");
    const size_t token_capacity = corpus_len + 2u * source_count;
    if (token_capacity > (size_t)INT_MAX ||
        token_capacity > SIZE_MAX / sizeof(uint16_t))
        die("token ID mapping size overflow");
    MappedBuffer token_ids_mapping;
    memset(&token_ids_mapping, 0, sizeof(token_ids_mapping));
    if (!mapped_buffer_create(&token_ids_mapping,
            token_capacity * sizeof(uint16_t)))
        die("could not create disk-backed token ID mapping");
    uint16_t* all_ids = (uint16_t*)token_ids_mapping.data;
    TrainSourceSpan train_sources[TRAIN_MAX_SOURCES];
    size_t train_source_count = 0;
    size_t token_cursor = 0;
    int train_tokens = 0;
    int validation_tokens = 0;
    const int minimum_source_text_tokens = options.sequence_length * 2;

    /* Tokenize each file independently so BPE merges cannot manufacture
       cross-file text. EOS markers terminate each source's train and held-out
       piece, and sampling never crosses either boundary. */
    for (size_t source_index = 0; source_index < source_count; ++source_index) {
        const uint8_t* source_text = corpus;
        size_t source_bytes = corpus_len;
        const char* source_name = "synthetic corpus";
        if (have_corpus) {
            const CorpusSource* source = &loaded_corpus.sources[source_index];
            source_text = corpus + source->byte_start;
            source_bytes = source->byte_length;
            source_name = source->path;
        }
        if (source_bytes == 0 || source_bytes > (size_t)INT_MAX ||
            token_cursor >= token_capacity)
            die("corpus source has invalid byte or token bounds");

        const size_t source_token_start = token_cursor;
        if (token_capacity - token_cursor < 2u)
            die("token mapping has no room for source boundary markers");
        const int out_capacity = (int)(token_capacity - token_cursor - 2u);
        const int encoded = tokenizer_encode(&tok, source_text, source_bytes,
            all_ids + token_cursor, out_capacity);
        if (encoded <= 0) die("corpus tokenization failed for %s", source_name);

        if (encoded < minimum_source_text_tokens) {
            printf("Skipping training source too short for separate train/validation windows: %s (%zu tokens)\n",
                source_name, (size_t)encoded);
            token_cursor = source_token_start;
            continue;
        }

        int validation_text_count = encoded / 20;
        if (validation_text_count < options.sequence_length)
            validation_text_count = options.sequence_length;
        if (validation_text_count > encoded - options.sequence_length)
            validation_text_count = encoded - options.sequence_length;
        const int source_train_text_count = encoded - validation_text_count;
        if (source_train_text_count < options.sequence_length ||
            validation_text_count < options.sequence_length) {
            token_cursor = source_token_start;
            continue;
        }

        const size_t validation_text_start =
            source_token_start + (size_t)source_train_text_count;
        memmove(all_ids + validation_text_start + 1u,
            all_ids + validation_text_start,
            (size_t)validation_text_count * sizeof(uint16_t));
        all_ids[validation_text_start] = TOK_EOS_ID;
        all_ids[validation_text_start + 1u + (size_t)validation_text_count] = TOK_EOS_ID;

        TrainSourceSpan* span = &train_sources[train_source_count++];
        span->name = source_name;
        span->train_start = (int)source_token_start;
        span->train_tokens = source_train_text_count + 1;
        span->validation_start = (int)validation_text_start + 1;
        span->validation_tokens = validation_text_count + 1;
        span->weight = source_weights[source_index];
        train_tokens += span->train_tokens;
        validation_tokens += span->validation_tokens;
        token_cursor += (size_t)encoded + 2u;
    }

    if (train_source_count == 0 || token_cursor == 0)
        die("no corpus source is long enough for the configured sequence length");
    const int n_tokens = (int)token_cursor;
    printf("total tokens = %d from %zu usable sources (disk-backed token array: %.1f MiB)\n\n",
        n_tokens, train_source_count,
        (double)token_capacity * sizeof(uint16_t) / (1024.0 * 1024.0));

    /* Raw source bytes are no longer needed after tokenization. Releasing this
       separate arena avoids retaining the complete text beside the token map. */
    arena_free(&corpus_arena);

    if (train_tokens < options.sequence_length + 1 ||
        validation_tokens < options.sequence_length + 1)
        die("usable corpus sources are too small for the configured sequence length");

    printf("Training tokens:   %d (source-wise held-out split)\n", train_tokens);
    printf("Validation tokens: %d (source-wise held-out split)\n\n", validation_tokens);

    // ------------------------------------------------------------------
    // 5. Model, optimizer, gradients, and optional checkpoint restore
    // ------------------------------------------------------------------
    Model model;
    options.model.vocab_size=tok.vocab_size;
    if (!model_init_config(&model, &arena, options.seed, &options.model))
        die("model_init failed");
    model_summary(&model);

    Optim opt;
    const float optimizer_beta1 = options.beta1;
    const float optimizer_beta2 = options.beta2;
    if (!optim_init_type(&opt, &model, &arena, optimizer_type,
        options.learning_rate, optimizer_beta1, optimizer_beta2, options.epsilon,
        options.weight_decay))
        die("optim_init failed");
    printf("Optimizer: %s (%.1f MiB optimizer state)\n",
        optimizer_type == OPTIMIZER_LION ? "Lion" : "AdamW",
        (double)model.weight_bytes *
            (optimizer_type == OPTIMIZER_LION ? 1.0 : 2.0) /
            (1024.0 * 1024.0));

    Rng rng;
    rng_seed(&rng, options.seed ^ 0x9e3779b97f4a7c15ULL);

    if (resume) {
        if (!checkpoint_load("minima.model", &model, &opt,
            &rng, tokenizer_hash))
            die("checkpoint load failed: model/tokenizer mismatch or invalid checkpoint");
        printf("Resumed checkpoint at optimizer step %lld\n",
            (long long)opt.step);
        if (options.total_steps_explicit && options.total_steps != opt.schedule_total_steps)
            die("--total-steps (%d) does not match checkpoint schedule horizon (%lld)",
                options.total_steps, (long long)opt.schedule_total_steps);
        printf("Restored LR schedule: %s, base_lr=%.7g, warmup=%lld, total_steps=%lld\n\n",
            opt.schedule_type==0?"cosine":"constant", opt.schedule_base_lr,
            (long long)opt.schedule_warmup_steps, (long long)opt.schedule_total_steps);
    }

    printf("Effective optimizer settings: beta1=%.7g beta2=%.7g epsilon=%.7g weight_decay=%.7g\n",
        opt.beta1, opt.beta2, opt.eps, opt.weight_decay);

    if (options.chat_mode) {
        KVCache chat_cache;
        if(!kvcache_init(&chat_cache,&arena,&model,model.max_seq))
            die("chat KV cache initialization failed");
        ChatOptions chat_options = {
            .seed = options.seed,
            .temperature = options.temperature,
            .top_p = options.top_p,
            .repetition_penalty = options.repetition_penalty,
            .min_p = options.min_p,
            .frequency_penalty = options.frequency_penalty,
            .presence_penalty = options.presence_penalty,
            .top_k = options.top_k,
            .max_new_tokens = options.max_new_tokens,
            .min_new_tokens = options.min_new_tokens,
            .no_repeat_ngram_size = options.no_repeat_ngram,
            .sampling_seed = options.sampling_seed,
            .stop_sequence = options.stop_sequence
        };
        int chat_result=chat_run_console(&model,&tok,&chat_cache,&chat_options);
        mapped_buffer_destroy(&token_ids_mapping);
        arena_free(&corpus_arena);arena_free(&temp);arena_free(&arena);return chat_result;
    }

    Gradients grads;
    if (!gradients_init(&grads, &arena, &model))
        die("gradient allocation failed");
    if (!gradients_check_layout(&grads, &model, &arena))
        die("gradient allocation layout does not match runtime model configuration");
    printf("Gradient allocation layout: PASS (runtime configuration, all 9 tensors)\n");
    printf("Engine budget at model setup: %.2f GiB arena + %.2f GiB tracked heap = %.2f GiB (peak %.2f GiB)\n",
        (double)memory_budget.committed / (1024.0*1024.0*1024.0),
        (double)memory_budget.external_committed / (1024.0*1024.0*1024.0),
        (double)arena_budget_total_committed(&memory_budget) / (1024.0*1024.0*1024.0),
        (double)memory_budget.peak / (1024.0*1024.0*1024.0));

    
    /* Lower-bound estimate for the live training allocations. */
    const double weight_mib = (double)model.weight_bytes / (1024.0 * 1024.0);
    const double gradient_mib = weight_mib;
    const double optimizer_mib = weight_mib *
        (optimizer_type == OPTIMIZER_LION ? 1.0 : 2.0);
    const double activation_mib = (double)activation_bytes / (1024.0 * 1024.0);
    printf("Estimated training working set: weights %.1f + gradients %.1f + "
        "optimizer %.1f + activations/logits %.1f = %.1f MiB\n",
        weight_mib, gradient_mib, optimizer_mib, activation_mib,
        weight_mib + gradient_mib + optimizer_mib + activation_mib);
    const double mapped_token_mib = (double)token_ids_mapping.size / (1024.0 * 1024.0);
    const double budgeted_now_mib = (double)arena_budget_total_committed(&memory_budget) / (1024.0 * 1024.0);
    const double budgeted_projected_mib = budgeted_now_mib + activation_mib;
    const double tracked_peak_mib = (double)memory_budget.peak / (1024.0 * 1024.0);
    const double estimated_process_peak_mib =
        (tracked_peak_mib > budgeted_projected_mib ? tracked_peak_mib : budgeted_projected_mib) +
        mapped_token_mib + 256.0;
    printf("Pre-training memory estimate: %.1f MiB budgeted peak + %.1f MiB disk-backed token map + 256 MiB runtime reserve = about %.1f MiB.\n",
        tracked_peak_mib > budgeted_projected_mib ? tracked_peak_mib : budgeted_projected_mib,
        mapped_token_mib, estimated_process_peak_mib);
    printf("  Token-map size is backing-file capacity, not an assertion that the entire map stays resident. Runtime reserve is a planning allowance, not a hard guarantee.\n");
    if (memory_budget.limit > 0 && budgeted_projected_mib >
        (double)memory_budget.limit / (1024.0 * 1024.0))
        printf("WARNING: projected arena + tracked-heap allocations exceed --memory-budget-gb; the run may stop cleanly at a later allocation.\n");
    // ------------------------------------------------------------------
    // 6. Numerical gradient check
    // ------------------------------------------------------------------
    uint16_t* check_tokens = arena_alloc(&arena,
        (size_t)options.batch_size * options.sequence_length * sizeof(uint16_t));
    uint16_t* check_targets = arena_alloc(&arena,
        (size_t)options.batch_size * options.sequence_length * sizeof(uint16_t));
    if (!check_tokens || !check_targets)
        die("gradient-check batch allocation failed");

    for (int b = 0; b < options.batch_size; b++) {
        const int start = b * options.sequence_length;
        for (int t = 0; t < options.sequence_length; t++) {
            check_tokens[b * options.sequence_length + t] = all_ids[start + t];
            check_targets[b * options.sequence_length + t] = all_ids[start + t + 1];
        }
    }

    if (!gradient_check(&model, &grads, &temp,
        check_tokens, check_targets, options.batch_size, options.sequence_length))
        die("numerical gradient check failed");

    // Establish one schedule horizon on a fresh run and persist it before updates.
    if (!resume) {
        const int total_steps = options.total_steps;
        if (total_steps < steps_to_run)
            die("--total-steps must be at least --steps for a fresh run");
        opt.schedule_base_lr = options.learning_rate;
        opt.schedule_total_steps = total_steps;
        opt.schedule_warmup_steps = options.warmup_steps < total_steps
            ? options.warmup_steps : (total_steps > 1 ? total_steps - 1 : 0);
        opt.schedule_type = strcmp(options.lr_schedule,"constant")==0 ? 1 : 0;
    }
    if (!diagnostics_only && !resume) {
        if (!checkpoint_archive_previous("minima.model", options.checkpoint_retention))
            printf("WARNING: could not rotate the previous checkpoint archive.\n");
        if (!checkpoint_save("minima.model", &model, &opt, &rng, tokenizer_hash))
            die("initial checkpoint save failed");
    }

    TrainRunOptions train_options = {
        .steps = steps_to_run,
        .log_interval = options.log_interval,
        .validation_batches = options.validation_batches,
        .validation_interval = options.validation_interval,
        .checkpoint_interval = options.checkpoint_interval,
        .checkpoint_retention = options.checkpoint_retention,
        .batch_size = options.batch_size,
        .sequence_length = options.sequence_length,
        .grad_clip_norm = options.grad_clip_norm,
        .seed = options.seed,
        .tokenizer_sample_bytes = options.tokenizer_sample_bytes,
        .tokenizer_spread = options.tokenizer_spread,
        .diagnostics_only = diagnostics_only,
        .corpus_mix = options.corpus_mix,
        .corpus_sources = train_sources,
        .corpus_source_count = train_source_count,
        .activation_bytes = activation_bytes,
        .metrics_path = options.metrics_path,
        .stop_file = options.stop_file,
        .pause_file = options.pause_file,
        .save_file = options.save_file
    };
    if (!write_run_metadata(options.metrics_path, &options, &opt,
            train_sources, train_source_count, steps_to_run, resume, tokenizer_hash))
        printf("WARNING: could not write run metadata sidecar for metrics comparison.\\n");
    train_run(&model, &opt, &grads, &temp, all_ids,
        train_tokens, validation_tokens, &rng, tokenizer_hash, train_options);
    const int64_t final_step = opt.step;

    // ------------------------------------------------------------------
    // 8. Tokenizer and checkpoint round-trip regression checks
    // ------------------------------------------------------------------
    Tokenizer* loaded_tok = (Tokenizer*)arena_alloc(&arena, sizeof(Tokenizer));
    if (!loaded_tok || !tokenizer_load(loaded_tok, &arena, "minima.bpe"))
        die("tokenizer round-trip load failed");

    const char* tokenizer_test_text = "The quick brown fox";
    uint16_t original_ids[64], loaded_ids[64];
    const int tokenizer_test_len = (int)strlen(tokenizer_test_text);
    const int original_count = tokenizer_encode(&tok,
        (const uint8_t*)tokenizer_test_text, (size_t)tokenizer_test_len,
        original_ids, 64);
    const int loaded_count = tokenizer_encode(loaded_tok,
        (const uint8_t*)tokenizer_test_text, (size_t)tokenizer_test_len,
        loaded_ids, 64);
    if (loaded_tok->vocab_size != tok.vocab_size ||
        loaded_tok->n_merges != tok.n_merges ||
        tokenizer_fingerprint(loaded_tok) != tokenizer_hash ||
        original_count != loaded_count ||
        memcmp(original_ids, loaded_ids,
            (size_t)original_count * sizeof(uint16_t)) != 0)
        die("tokenizer round-trip verification failed");

    printf("Tokenizer round-trip: PASS (vocabulary, merges, encoding)\n");

    /* An undersized output buffer must fail without returning a prefix. */
    uint16_t capacity_guard[8];
    for (int i = 0; i < 8; ++i) capacity_guard[i] = 0xBEEFu;
    const int capacity_result = tokenizer_encode(&tok,
        (const uint8_t*)"123456789", 9, capacity_guard, 8);
    if (capacity_result != -1)
        die("tokenizer capacity regression check failed: expected -1");
    for (int i = 0; i < 8; ++i)
        if (capacity_guard[i] != 0xBEEFu)
            die("tokenizer capacity regression check modified output");
    printf("Tokenizer insufficient-capacity check: PASS (no partial output)\n");

    const float saved_weight = model.tok_emb[0];
    const float saved_moment = opt.m_tok_emb[0];
    const float saved_variance = opt.v_tok_emb[0];
    const uint64_t saved_rng_state = rng.state;
    const float saved_schedule_lr = opt.schedule_base_lr;
    const int64_t saved_schedule_total = opt.schedule_total_steps;
    const int64_t saved_schedule_warmup = opt.schedule_warmup_steps;
    const int32_t saved_schedule_type = opt.schedule_type;
    model.tok_emb[0] += 1.0f;
    opt.m_tok_emb[0] += 1.0f;
    opt.v_tok_emb[0] += 1.0f;
    rng.state ^= 0x55aa55aa55aa55aaULL;

    if (!checkpoint_load("minima.model", &model, &opt,
        &rng, tokenizer_hash))
        die("checkpoint round-trip load failed");
    if (model.tok_emb[0] != saved_weight ||
        opt.m_tok_emb[0] != saved_moment ||
        opt.v_tok_emb[0] != saved_variance ||
        opt.step != final_step || rng.state != saved_rng_state ||
        opt.schedule_base_lr != saved_schedule_lr ||
        opt.schedule_total_steps != saved_schedule_total ||
        opt.schedule_warmup_steps != saved_schedule_warmup ||
        opt.schedule_type != saved_schedule_type)
        die("checkpoint round-trip verification failed");

    if (opt.schedule_base_lr <= 0.0f || opt.schedule_total_steps <= 0 ||
        opt.schedule_warmup_steps < 0 ||
        (opt.schedule_type != 0 && opt.schedule_type != 1))
        die("checkpoint LR schedule round-trip verification failed");
    printf("Checkpoint round-trip: PASS (weights, optimizer, step, RNG, LR schedule)\n");
    printf("Checkpoint compatibility: PASS (model architecture and tokenizer)\n");

    /* A truncated checkpoint must not alter any live destination state. */
    {
        FILE* source = fopen("minima.model", "rb");
        FILE* truncated = fopen("minima_truncated.model", "wb");
        uint8_t prefix[128];
        size_t bytes = source ? fread(prefix, 1, sizeof(prefix), source) : 0;
        if (source) fclose(source);
        if (!truncated || bytes != sizeof(prefix) ||
            fwrite(prefix, 1, bytes, truncated) != bytes) {
            if (truncated) fclose(truncated);
            remove("minima_truncated.model");
            die("checkpoint atomicity test setup failed");
        }
        if (fclose(truncated) != 0) {
            remove("minima_truncated.model");
            die("checkpoint atomicity test setup failed");
        }

        const float guard_weight = model.tok_emb[0];
        const float guard_moment = opt.m_tok_emb[0];
        const float guard_variance = opt.v_tok_emb[0];
        const float guard_lr = opt.lr;
        const int64_t guard_step = opt.step;
        const uint64_t guard_rng = rng.state;
        const float guard_schedule_lr = opt.schedule_base_lr;
        const int64_t guard_schedule_total = opt.schedule_total_steps;
        const int64_t guard_schedule_warmup = opt.schedule_warmup_steps;
        const int32_t guard_schedule_type = opt.schedule_type;
        model.tok_emb[0] = guard_weight + 3.0f;
        opt.m_tok_emb[0] = guard_moment + 4.0f;
        opt.v_tok_emb[0] = guard_variance + 5.0f;
        opt.lr = guard_lr + 0.125f;
        opt.step = guard_step + 7;
        rng.state ^= 0x1122334455667788ULL;

        const float expected_weight = model.tok_emb[0];
        const float expected_moment = opt.m_tok_emb[0];
        const float expected_variance = opt.v_tok_emb[0];
        const float expected_lr = opt.lr;
        const int64_t expected_step = opt.step;
        const uint64_t expected_rng = rng.state;
        const bool rejected = !checkpoint_load("minima_truncated.model",
            &model, &opt, &rng, tokenizer_hash);
        const bool unchanged = model.tok_emb[0] == expected_weight &&
            opt.m_tok_emb[0] == expected_moment &&
            opt.v_tok_emb[0] == expected_variance &&
            opt.lr == expected_lr && opt.step == expected_step &&
            opt.schedule_base_lr == guard_schedule_lr &&
            opt.schedule_total_steps == guard_schedule_total &&
            opt.schedule_warmup_steps == guard_schedule_warmup &&
            opt.schedule_type == guard_schedule_type && rng.state == expected_rng;

        model.tok_emb[0] = guard_weight;
        opt.m_tok_emb[0] = guard_moment;
        opt.v_tok_emb[0] = guard_variance;
        opt.lr = guard_lr;
        opt.step = guard_step;
        rng.state = guard_rng;
        remove("minima_truncated.model");
        if (!rejected || !unchanged)
            die("checkpoint failed-load atomicity regression check failed");
    }
    printf("Checkpoint failed-load atomicity: PASS (destination unchanged)\n");

    // ------------------------------------------------------------------
    // 9. KV-cache parity and deterministic multi-prompt generation
    // ------------------------------------------------------------------
    KVCache cache;
    if (!kvcache_init(&cache, &arena, &model, model.max_seq))
        die("KV cache initialization failed");

    if (!generation_check_kv_cache(&model, &cache, &temp,
        check_tokens, options.sequence_length, 1e-4f))
        die("KV-cache parity check failed");

    if (!diagnostics_run(&model, &tok, &arena, &temp, all_ids,
        train_tokens, validation_tokens, options.sequence_length))
        die("model-quality diagnostics failed");

    printf("\nSmoke test finished.\n");
    printf("Training, validation, persistence, and generation checks completed.\n");

    printf("Engine budget high-water mark: %.2f GiB (arena + tracked heap); rejected allocations: %zu\n",
        (double)memory_budget.peak / (1024.0*1024.0*1024.0), memory_budget.failed);
    mapped_buffer_destroy(&token_ids_mapping);
    arena_free(&corpus_arena);
    arena_free(&temp);
    arena_free(&arena);
    return 0;
}
