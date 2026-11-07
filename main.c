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

#define DEFAULT_BATCH_SIZE 4
#define DEFAULT_SEQUENCE_LENGTH 32
#define DEFAULT_STEPS 2000
#define DEFAULT_LR 3e-4f
#define DEFAULT_WARMUP_STEPS 100
#define DEFAULT_LOG_INTERVAL 100
#define DEFAULT_VALIDATION_BATCHES 16
#define DEFAULT_CHECKPOINT_INTERVAL 100
#define DEFAULT_TOKENIZER_SAMPLE_BYTES (1024u * 1024u)

typedef struct {
    float learning_rate;
    int total_steps;
    bool total_steps_explicit;
    int warmup_steps, log_interval, validation_batches, validation_interval, checkpoint_interval;
    int batch_size, sequence_length;
    uint64_t seed;
    size_t tokenizer_sample_bytes;
    int threads;
    const char* metrics_path;
    const char* stop_file;
    const char* save_file;
    float temperature, top_p, repetition_penalty;
    int top_k, max_new_tokens;
    bool chat_mode;
    ModelConfig model;
    const char* lr_schedule;
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
    printf("  --validation-batches N --validation-interval N --checkpoint-interval N --seed N\n");
    printf("  --tokenizer-sample-bytes N --vocab-size N --dim N --layers N\n");
    printf("  --heads N --ffn-dim N --max-seq N --batch-size N --sequence-length N\n");
    printf("  --lr-schedule cosine|constant --tokenizer-sampling spread|prefix\n");
    printf("  --temperature F --top-k N --top-p F --repetition-penalty F\n");
    printf("  --max-new-tokens N --chat (requires --resume) --metrics PATH\n");
    printf("  --retrain-tokenizer-only (rebuild minima.bpe from corpus and exit)\n");
    printf("  --use-existing-tokenizer (fresh model using a previously rebuilt minima.bpe)\n");
    printf("  --diagnostics-only --help\n");
}

static int parse_arguments(int argc, char** argv, bool* resume,
    bool* diagnostics_only, int* steps, const char** corpus_paths, TrainOptions* o) {
    int path_count = 0;
    *resume = false; *diagnostics_only = false; *steps = DEFAULT_STEPS;
    ModelConfig default_model=MODEL_CONFIG_DEFAULT;
    o->model=default_model;
    o->batch_size=DEFAULT_BATCH_SIZE; o->sequence_length=DEFAULT_SEQUENCE_LENGTH;
    o->learning_rate=DEFAULT_LR; o->warmup_steps=DEFAULT_WARMUP_STEPS;
    o->log_interval=DEFAULT_LOG_INTERVAL; o->validation_batches=DEFAULT_VALIDATION_BATCHES;
    o->validation_interval=DEFAULT_LOG_INTERVAL;
    o->checkpoint_interval=DEFAULT_CHECKPOINT_INTERVAL; o->seed=1234;
    o->tokenizer_sample_bytes=DEFAULT_TOKENIZER_SAMPLE_BYTES;
    o->threads=0;
    o->metrics_path="minima_metrics.csv"; o->stop_file=NULL; o->save_file=NULL;
    o->temperature=0.8f; o->top_k=40; o->top_p=0.9f;
    o->repetition_penalty=1.1f; o->max_new_tokens=32; o->chat_mode=false;
    o->lr_schedule="cosine"; o->tokenizer_spread=true;
    o->total_steps=0; o->total_steps_explicit=false;
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
            strcmp(argv[i], "--seed") == 0 ||
            strcmp(argv[i], "--tokenizer-sample-bytes") == 0 ||
            strcmp(argv[i], "--threads") == 0 ||
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
            if (strcmp(opt,"--seed")==0) { lo=0; hi=LONG_MAX; }
            if (strcmp(opt,"--tokenizer-sample-bytes")==0) { lo=4096; hi=4194304; }
            if (strcmp(opt,"--threads")==0) { lo=1; hi=256; }
            if (strcmp(opt,"--vocab-size")==0) { lo=260; hi=8192; }
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
            else o->tokenizer_sample_bytes=(size_t)v;
        }
        else if (strcmp(argv[i], "--top-k") == 0 || strcmp(argv[i], "--max-new-tokens") == 0) {
            const char* option=argv[i]; if(++i>=argc)die("%s requires a value",option);
            long v=parse_long_value(option,argv[i],1,
                strcmp(option,"--top-k")==0?256:512);
            if(strcmp(option,"--top-k")==0)o->top_k=(int)v;else o->max_new_tokens=(int)v;
        }
        else if (strcmp(argv[i], "--temperature") == 0 ||
            strcmp(argv[i], "--top-p") == 0 ||
            strcmp(argv[i], "--repetition-penalty") == 0) {
            const char* option=argv[i];if(++i>=argc)die("%s requires a value",option);
            if(strcmp(option,"--temperature")==0)o->temperature=parse_float_value(option,argv[i],0.05f,5.0f);
            else if(strcmp(option,"--top-p")==0)o->top_p=parse_float_value(option,argv[i],0.01f,1.0f);
            else o->repetition_penalty=parse_float_value(option,argv[i],1.0f,3.0f);
        }
        else if (strcmp(argv[i], "--lr-schedule") == 0) {
            if(++i>=argc)die("--lr-schedule requires cosine or constant");
            if(strcmp(argv[i],"cosine")!=0 && strcmp(argv[i],"constant")!=0)
                die("--lr-schedule must be cosine or constant");
            o->lr_schedule=argv[i];
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
        }
        else if (strcmp(argv[i], "--metrics") == 0 || strcmp(argv[i], "--stop-file") == 0 || strcmp(argv[i], "--save-file") == 0) {
            const char* option=argv[i];
            if (++i >= argc || !argv[i][0]) die("%s requires a path",option);
            if (strcmp(option,"--metrics")==0) o->metrics_path=argv[i];
            else if(strcmp(option,"--stop-file")==0)o->stop_file=argv[i];
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
        2u*(size_t)options.model.vocab_size) +
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

    if (!arena_init(&arena, 8ull * 1024 * 1024 * 1024))
        die("permanent arena failed");
    if (!arena_init(&temp, 256ull * 1024 * 1024))
        die("temp arena failed");

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
    size_t tokenizer_sample_files = 0;
    size_t tokenizer_source_bytes = 0;

    if (!options.chat_mode && !resume && !options.use_existing_tokenizer) {
        CorpusLoadResult sample_result = corpus_sample_files(&temp,
            corpus_paths, corpus_path_count, options.tokenizer_sample_bytes,
            options.tokenizer_spread, &tokenizer_sample,
            &tokenizer_sample_len, &tokenizer_sample_files,
            &tokenizer_source_bytes);

        if (sample_result == CORPUS_LOAD_EMPTY && default_paths) {
            corpus_paths = root_paths;
            sample_result = corpus_sample_files(&temp, corpus_paths,
                corpus_path_count, options.tokenizer_sample_bytes,
                options.tokenizer_spread, &tokenizer_sample,
                &tokenizer_sample_len, &tokenizer_sample_files,
                &tokenizer_source_bytes);
        }
        if (sample_result == CORPUS_LOAD_ERROR)
            die("streaming tokenizer sampling failed (check corpus file access and available memory)");
        if (sample_result == CORPUS_LOAD_EMPTY && options.tokenizer_only)
            die("tokenizer rebuild requires actual corpus text; synthetic smoke text is not allowed");
        if (sample_result == CORPUS_LOAD_OK) {
            printf("Tokenizer sample sources: %zu files, %zu source bytes\n",
                tokenizer_sample_files, tokenizer_source_bytes);
            printf("Tokenizer training sample: %zu bytes (source-balanced, %s)\n",
                tokenizer_sample_len,
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
            &arena, corpus_paths, corpus_path_count, max_corpus_bytes);
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
    Tokenizer tok;
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
        }

        const size_t sample_len = tokenizer_sample_len;
        printf("Tokenizer trainer sample: %zu bytes\n", sample_len);
        if (!tokenizer_train(&tok, &arena, tokenizer_sample,
            sample_len, options.model.vocab_size))
            die("tokenizer_train failed");
        if (tok.vocab_size != options.model.vocab_size)
            die("BPE stopped at vocabulary %d before the configured model vocabulary %d; the sample may not contain enough repeated pairs or eligible merges",
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
        arena_free(&temp);
        arena_free(&arena);
        return 0;
    }

    // ------------------------------------------------------------------
    // 4. Encode the complete corpus and create held-out validation region
    // ------------------------------------------------------------------
    uint16_t* all_ids = (uint16_t*)arena_alloc(&arena,
        corpus_len * sizeof(uint16_t));
    if (!all_ids) die("token id allocation failed");

    const int n_tokens = tokenizer_encode(&tok, corpus, corpus_len,
        all_ids, (int)corpus_len);
    if (n_tokens <= 0) die("corpus tokenization failed");
    printf("total tokens = %d\n\n", n_tokens);

    const int train_tokens = n_tokens - n_tokens / 20;
    const int validation_tokens = n_tokens - train_tokens;
    if (train_tokens < options.batch_size * options.sequence_length + 1 ||
        validation_tokens < options.batch_size * options.sequence_length + 1)
        die("corpus too small for a 95/5 train/validation split");

    printf("Training tokens:   %d (95%%)\n", train_tokens);
    printf("Validation tokens: %d (5%%)\n\n", validation_tokens);

    // ------------------------------------------------------------------
    // 5. Model, optimizer, gradients, and optional checkpoint restore
    // ------------------------------------------------------------------
    Model model;
    options.model.vocab_size=tok.vocab_size;
    if (!model_init_config(&model, &arena, options.seed, &options.model))
        die("model_init failed");
    model_summary(&model);

    Optim opt;
    if (!optim_init(&opt, &model, &arena, options.learning_rate, 0.9f, 0.95f, 1e-8f, 0.01f))
        die("optim_init failed");

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

    if (options.chat_mode) {
        KVCache chat_cache;
        if(!kvcache_init(&chat_cache,&arena,&model,model.max_seq))
            die("chat KV cache initialization failed");
        ChatOptions chat_options = {
            .seed = options.seed,
            .temperature = options.temperature,
            .top_p = options.top_p,
            .repetition_penalty = options.repetition_penalty,
            .top_k = options.top_k,
            .max_new_tokens = options.max_new_tokens
        };
        int chat_result=chat_run_console(&model,&tok,&chat_cache,&chat_options);
        arena_free(&temp);arena_free(&arena);return chat_result;
    }

    Gradients grads;
    if (!gradients_init(&grads, &arena, &model))
        die("gradient allocation failed");
    if (!gradients_check_layout(&grads, &model, &arena))
        die("gradient allocation layout does not match runtime model configuration");
    printf("Gradient allocation layout: PASS (runtime configuration, all 10 tensors)\n");

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
        const int total_steps = options.total_steps_explicit ? options.total_steps : steps_to_run;
        if (total_steps < steps_to_run)
            die("--total-steps must be at least --steps for a fresh run");
        opt.schedule_base_lr = options.learning_rate;
        opt.schedule_total_steps = total_steps;
        opt.schedule_warmup_steps = options.warmup_steps < total_steps
            ? options.warmup_steps : (total_steps > 1 ? total_steps - 1 : 0);
        opt.schedule_type = strcmp(options.lr_schedule,"constant")==0 ? 1 : 0;
    }
    if (!diagnostics_only && !resume && !checkpoint_save("minima.model", &model, &opt,
        &rng, tokenizer_hash))
        die("initial checkpoint save failed");

    TrainRunOptions train_options = {
        .steps = steps_to_run,
        .log_interval = options.log_interval,
        .validation_batches = options.validation_batches,
        .validation_interval = options.validation_interval,
        .checkpoint_interval = options.checkpoint_interval,
        .batch_size = options.batch_size,
        .sequence_length = options.sequence_length,
        .seed = options.seed,
        .tokenizer_sample_bytes = options.tokenizer_sample_bytes,
        .tokenizer_spread = options.tokenizer_spread,
        .diagnostics_only = diagnostics_only,
        .activation_bytes = activation_bytes,
        .metrics_path = options.metrics_path,
        .stop_file = options.stop_file,
        .save_file = options.save_file
    };
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

    arena_free(&temp);
    arena_free(&arena);
    return 0;
}
