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

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>
#include <stdint.h>
#include <math.h>
#include <errno.h>
#include <time.h>

#define BATCH 4
#define SEQ 32
#define DEFAULT_STEPS 2000
#define DEFAULT_LR 3e-4f
#define DEFAULT_WARMUP_STEPS 100
#define DEFAULT_LOG_INTERVAL 100
#define DEFAULT_VALIDATION_BATCHES 16
#define DEFAULT_CHECKPOINT_INTERVAL 100
#define DEFAULT_TOKENIZER_SAMPLE_BYTES (256u * 1024u)

typedef struct {
    float learning_rate;
    int warmup_steps, log_interval, validation_batches, checkpoint_interval;
    uint64_t seed;
    size_t tokenizer_sample_bytes;
    const char* metrics_path;
    const char* stop_file;
    float temperature, top_p, repetition_penalty;
    int top_k, max_new_tokens;
    bool chat_mode;
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
    printf("  --resume --steps N --lr FLOAT --warmup N --log-interval N\n");
    printf("  --validation-batches N --checkpoint-interval N --seed N\n");
    printf("  --tokenizer-sample-bytes N --metrics PATH --diagnostics-only\n");
}

static float evaluate_loss(Model* model, Arena* temp,
    const uint16_t* all_ids, int validation_start, int validation_tokens,
    int validation_batches) {

    if (!model || !temp || !all_ids ||
        validation_start < 0 || validation_tokens < BATCH * SEQ + 1 || validation_batches < 1)
        return NAN;

    arena_reset(temp);

    Activations act;
    if (!activations_init(&act, temp, BATCH, SEQ, model))
        return NAN;

    float* dlogits = arena_alloc(temp,
        (size_t)BATCH * SEQ * model->vocab_size * sizeof(float));
    uint16_t* batch_tokens = arena_alloc(temp,
        (size_t)BATCH * SEQ * sizeof(uint16_t));
    uint16_t* batch_targets = arena_alloc(temp,
        (size_t)BATCH * SEQ * sizeof(uint16_t));

    if (!dlogits || !batch_tokens || !batch_targets)
        return NAN;

    const size_t span = (size_t)validation_tokens - SEQ - 1;
    double loss_sum = 0.0;

    // Evaluate fixed, evenly spaced windows across the held-out region.
    for (int eval_batch = 0; eval_batch < validation_batches; eval_batch++) {
        for (int b = 0; b < BATCH; b++) {
            const size_t sample_index = (size_t)eval_batch * BATCH + b;
            const size_t sample_count = (size_t)validation_batches * BATCH;
            const size_t offset = (span * sample_index) / (sample_count - 1);
            const int start = validation_start + (int)offset;

            for (int t = 0; t < SEQ; t++) {
                batch_tokens[b * SEQ + t] = all_ids[start + t];
                batch_targets[b * SEQ + t] = all_ids[start + t + 1];
            }
        }

        float* logits = model_forward(model, &act,
            batch_tokens, BATCH, SEQ);
        if (!logits) return NAN;

        const float loss = loss_cross_entropy(logits, batch_targets,
            dlogits, BATCH, SEQ, model->vocab_size);
        if (!isfinite(loss)) return NAN;
        loss_sum += loss;
    }

    return (float)(loss_sum / validation_batches);
}

static int run_chat_console(Model* model, Tokenizer* tok, KVCache* cache,
    const TrainOptions* options) {
    uint16_t history[512], input_ids[512], prompt[512], output[512];
    int history_len=0;
    char line[8192], decoded[512* TOK_MAX_TOKEN_LEN+1];
    puts("CHAT_READY");
    puts("ProjectMinima chat. Type /quit to exit.");
    for(;;){
        if(!fgets(line,sizeof(line),stdin))break;
        size_t bytes=strlen(line);while(bytes && (line[bytes-1]=='\\n'||line[bytes-1]=='\\r'))line[--bytes]=0;
        if(strcmp(line,"/quit")==0||strcmp(line,"/exit")==0)break;
        if(!bytes){puts("CHAT_READY");continue;}
        int input_count=tokenizer_encode(tok,(const uint8_t*)line,bytes,input_ids,512);
        if(input_count<=0){puts("BOT: [input encoding failed]");puts("CHAT_READY");continue;}
        if(input_count>=model->max_seq){memmove(input_ids,input_ids+(input_count-(model->max_seq-1)),(size_t)(model->max_seq-1)*sizeof(uint16_t));input_count=model->max_seq-1;}
        int prompt_len=history_len+input_count;
        if(prompt_len>model->max_seq-1){
            int drop=prompt_len-(model->max_seq-1);
            if(drop<history_len){memmove(history,history+drop,(size_t)(history_len-drop)*sizeof(uint16_t));history_len-=drop;}
            else history_len=0;
            prompt_len=history_len+input_count;
        }
        memcpy(prompt,history,(size_t)history_len*sizeof(uint16_t));
        memcpy(prompt+history_len,input_ids,(size_t)input_count*sizeof(uint16_t));
        Rng rng; rng_seed(&rng,((uint64_t)clock()<<17)^(uint64_t)opt->seed^(uint64_t)history_len);
        int max_new=options->max_new_tokens;
        if(max_new>model->max_seq-prompt_len)max_new=model->max_seq-prompt_len;
        int total=generate_ex(model,tok,cache,prompt,prompt_len,output,512,max_new,
            options->temperature,options->top_k,options->top_p,options->repetition_penalty,&rng);
        if(total<=prompt_len){puts("BOT: [generation failed]");puts("CHAT_READY");continue;}
        int decoded_bytes=tokenizer_decode(tok,output+prompt_len,total-prompt_len,decoded,(int)sizeof(decoded));
        if(decoded_bytes<0)decoded[0]=0;
        printf("BOT: %s\\n",decoded);
        history_len=total;
        if(history_len>model->max_seq-1){int drop=history_len-(model->max_seq-1);memmove(history,output+drop,(size_t)(history_len-drop)*sizeof(uint16_t));history_len-=drop;}
        else memcpy(history,output,(size_t)history_len*sizeof(uint16_t));
        puts("CHAT_READY");
    }
    puts("CHAT_CLOSED");
    return 0;
}

static bool stop_file_exists(const char* path) {
    if (!path || !path[0]) return false;
    FILE* f=fopen(path,"rb");
    if (!f) return false;
    fclose(f);
    return true;
}

static int parse_arguments(int argc, char** argv, bool* resume,
    bool* diagnostics_only, int* steps, const char** corpus_paths, TrainOptions* o) {
    int path_count = 0;
    *resume = false; *diagnostics_only = false; *steps = DEFAULT_STEPS;
    o->learning_rate=DEFAULT_LR; o->warmup_steps=DEFAULT_WARMUP_STEPS;
    o->log_interval=DEFAULT_LOG_INTERVAL; o->validation_batches=DEFAULT_VALIDATION_BATCHES;
    o->checkpoint_interval=DEFAULT_CHECKPOINT_INTERVAL; o->seed=1234;
    o->tokenizer_sample_bytes=DEFAULT_TOKENIZER_SAMPLE_BYTES;
    o->metrics_path="minima_metrics.csv"; o->stop_file=NULL;
    o->temperature=0.8f; o->top_k=40; o->top_p=0.9f;
    o->repetition_penalty=1.1f; o->max_new_tokens=32; o->chat_mode=false;

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
        else if (strcmp(argv[i], "--chat") == 0) o->chat_mode=true;
        else if (strcmp(argv[i], "--steps") == 0 ||
            strcmp(argv[i], "--warmup") == 0 ||
            strcmp(argv[i], "--log-interval") == 0 ||
            strcmp(argv[i], "--validation-batches") == 0 ||
            strcmp(argv[i], "--checkpoint-interval") == 0 ||
            strcmp(argv[i], "--seed") == 0 ||
            strcmp(argv[i], "--tokenizer-sample-bytes") == 0) {
            const char* opt=argv[i];
            if (++i >= argc) die("%s requires a value", opt);
            long lo=1, hi=INT_MAX;
            if (strcmp(opt,"--seed")==0) { lo=0; hi=LONG_MAX; }
            if (strcmp(opt,"--tokenizer-sample-bytes")==0) { lo=4096; hi=67108864; }
            long v=parse_long_value(opt,argv[i],lo,hi);
            if (strcmp(opt,"--steps")==0) *steps=(int)v;
            else if (strcmp(opt,"--warmup")==0) o->warmup_steps=(int)v;
            else if (strcmp(opt,"--log-interval")==0) o->log_interval=(int)v;
            else if (strcmp(opt,"--validation-batches")==0) o->validation_batches=(int)v;
            else if (strcmp(opt,"--checkpoint-interval")==0) o->checkpoint_interval=(int)v;
            else if (strcmp(opt,"--seed")==0) o->seed=(uint64_t)v;
            else o->tokenizer_sample_bytes=(size_t)v;
        }
        else if (strcmp(argv[i], "--top-k") == 0 || strcmp(argv[i], "--max-new-tokens") == 0) {
            const char* option=argv[i]; if(++i>=argc)die("%s requires a value",option);
            long v=parse_long_value(option,argv[i],1,512);
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
        else if (strcmp(argv[i], "--lr") == 0) {
            if (++i >= argc) die("--lr requires a value");
            o->learning_rate=parse_float_value("--lr",argv[i],1e-7f,1.0f);
        }
        else if (strcmp(argv[i], "--metrics") == 0 || strcmp(argv[i], "--stop-file") == 0) {
            const char* option=argv[i];
            if (++i >= argc || !argv[i][0]) die("%s requires a path",option);
            if (strcmp(option,"--metrics")==0) o->metrics_path=argv[i];
            else o->stop_file=argv[i];
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
    if (*diagnostics_only) {
        if (!*resume)
            die("--diagnostics-only requires --resume");
        *steps = 0;
    }
    return path_count;
}

static void print_repetition_stats(const uint16_t* ids,
    int prompt_len, int total_len) {
    const int n = total_len - prompt_len;
    if (!ids || n <= 0) return;

    int unique = 0;
    int adjacent_repeats = 0;
    for (int i = 0; i < n; i++) {
        bool seen = false;
        for (int j = 0; j < i; j++) {
            if (ids[prompt_len + i] == ids[prompt_len + j]) {
                seen = true;
                break;
            }
        }
        if (!seen) unique++;
        if (i > 0 && ids[prompt_len + i] == ids[prompt_len + i - 1])
            adjacent_repeats++;
    }

    int frequent_ids[5] = { -1, -1, -1, -1, -1 };
    int frequent_counts[5] = { 0, 0, 0, 0, 0 };
    for (int i = 0; i < n; i++) {
        const uint16_t id = ids[prompt_len + i];
        bool counted = false;
        for (int j = 0; j < i; j++) {
            if (ids[prompt_len + j] == id) {
                counted = true;
                break;
            }
        }
        if (counted) continue;

        int count = 0;
        for (int j = i; j < n; j++)
            if (ids[prompt_len + j] == id) count++;

        for (int slot = 0; slot < 5; slot++) {
            if (count > frequent_counts[slot]) {
                for (int move = 4; move > slot; move--) {
                    frequent_counts[move] = frequent_counts[move - 1];
                    frequent_ids[move] = frequent_ids[move - 1];
                }
                frequent_counts[slot] = count;
                frequent_ids[slot] = id;
                break;
            }
        }
    }

    printf("      most frequent token IDs:");
    for (int i = 0; i < 5 && frequent_ids[i] >= 0; i++) {
        printf(" %d=%d(%.1f%%)", frequent_ids[i], frequent_counts[i],
            100.0 * (double)frequent_counts[i] / (double)n);
    }
    putchar('\n');

    int repeated_ngram_len = 0;
    int repeated_ngram_count = 1;
    for (int width = 2; width <= 4 && width <= n; width++) {
        for (int i = 0; i + width <= n; i++) {
            int count = 1;
            for (int j = i + 1; j + width <= n; j++) {
                bool equal = true;
                for (int k = 0; k < width; k++) {
                    if (ids[prompt_len + i + k] !=
                        ids[prompt_len + j + k]) {
                        equal = false;
                        break;
                    }
                }
                if (equal) count++;
            }
            if (count > repeated_ngram_count) {
                repeated_ngram_count = count;
                repeated_ngram_len = width;
            }
        }
    }

    printf("      repetition: unique=%d/%d, adjacent-identical=%d, ",
        unique, n, adjacent_repeats);
    if (repeated_ngram_len > 0)
        printf("most-repeated %d-token pattern occurs %d times\n",
            repeated_ngram_len, repeated_ngram_count);
    else
        printf("no repeated 2-4 token pattern\n");
}

static bool run_generation_benchmark(Model* model, Tokenizer* tok,
    KVCache* cache) {
    static const char* prompts[] = {
        "The quick brown fox",
        "Once upon a time",
        "In the beginning",
        "The meaning of life",
        "It was a dark"
    };

    printf("\nGeneration diagnostics\n");
    printf("  Greedy decoding is tested for repeatability; sampled modes use fixed seeds.\n");
    uint16_t prompt_ids[64];
    uint16_t greedy_ids[64];
    uint16_t verify_ids[64];
    uint16_t sample_ids[64];
    char output_text[64 * TOK_MAX_TOKEN_LEN + 1];

    for (size_t i = 0; i < sizeof(prompts) / sizeof(prompts[0]); i++) {
        const size_t prompt_bytes = strlen(prompts[i]);
        const int prompt_len = tokenizer_encode(tok,
            (const uint8_t*)prompts[i], prompt_bytes,
            prompt_ids, (int)(sizeof(prompt_ids) / sizeof(prompt_ids[0])));
        if (prompt_len <= 0) {
            printf("  [%zu] prompt encoding failed\n", i + 1);
            return false;
        }

        printf("\n  [%zu] Prompt: \"%s\"\n", i + 1, prompts[i]);
        if (!generation_inspect_next_tokens(model, tok, cache,
            prompt_ids, prompt_len, 10)) {
            printf("  next-token inspection failed\n");
            return false;
        }

        // Greedy run and exact deterministic repeat.
        Rng greedy_rng;
        rng_seed(&greedy_rng, 5678u + (uint64_t)i);
        const int greedy_len = generate(model, tok, cache,
            prompt_ids, prompt_len, greedy_ids, 64,
            32, 1.0f, 1, &greedy_rng);
        if (greedy_len <= prompt_len) return false;

        Rng verify_rng;
        rng_seed(&verify_rng, 5678u + (uint64_t)i);
        const int verify_len = generate(model, tok, cache,
            prompt_ids, prompt_len, verify_ids, 64,
            32, 1.0f, 1, &verify_rng);
        if (verify_len != greedy_len ||
            memcmp(greedy_ids, verify_ids,
                (size_t)greedy_len * sizeof(uint16_t)) != 0) {
            printf("  deterministic greedy check: FAIL\n");
            return false;
        }

        int bytes = tokenizer_decode(tok, greedy_ids,
            greedy_len, output_text, (int)sizeof(output_text));
        if (bytes <= 0) return false;
        printf("    greedy (T=1.0, k=1): %s\n", output_text);
        print_repetition_stats(greedy_ids, prompt_len, greedy_len);

        // Lower-temperature top-k sampling.
        Rng sample_rng;
        rng_seed(&sample_rng, 9100u + (uint64_t)i);
        int sample_len = generate(model, tok, cache,
            prompt_ids, prompt_len, sample_ids, 64,
            32, 0.8f, 40, &sample_rng);
        if (sample_len <= prompt_len) return false;
        bytes = tokenizer_decode(tok, sample_ids,
            sample_len, output_text, (int)sizeof(output_text));
        if (bytes <= 0) return false;
        printf("    sampled (T=0.8, k=40): %s\n", output_text);
        print_repetition_stats(sample_ids, prompt_len, sample_len);

        // Slightly warmer sampling shows whether diversity improves or
        // merely turns repetition into noisy token selection.
        rng_seed(&sample_rng, 12100u + (uint64_t)i);
        sample_len = generate(model, tok, cache,
            prompt_ids, prompt_len, sample_ids, 64,
            32, 1.0f, 40, &sample_rng);
        if (sample_len <= prompt_len) return false;
        bytes = tokenizer_decode(tok, sample_ids,
            sample_len, output_text, (int)sizeof(output_text));
        if (bytes <= 0) return false;
        printf("    sampled (T=1.0, k=40): %s\n", output_text);
        print_repetition_stats(sample_ids, prompt_len, sample_len);
    }

    printf("\nDeterministic greedy check: PASS (all prompts reproduced exactly)\n");
    return true;
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

    printf("ProjectMinima text-corpus training experiment\n");
    printf("Vocabulary target: %d\n", MODEL_VOCAB_SIZE);
    if (options.chat_mode)
        printf("Mode: interactive chat (checkpoint resume)\n\n");
    else if (diagnostics_only)
        printf("Mode: diagnostics only (no optimizer updates)\n\n");
    else
        printf("Steps this run: %d%s\n\n", steps_to_run,
            resume ? " (resume)" : "");

    if (!optim_check_adamw())
        die("AdamW numerical regression check failed");
    printf("AdamW bias-correction check: PASS\n");

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
    // 2. Load plain-text training data
    // ------------------------------------------------------------------
    const size_t max_corpus_bytes = SIZE_MAX;
    Corpus loaded_corpus;
    bool have_corpus = false;

    if (user_path_count > 0) {
        const CorpusLoadResult load_result = corpus_load_files(&loaded_corpus,
            &arena, user_paths, (size_t)user_path_count, max_corpus_bytes);
        if (load_result == CORPUS_LOAD_ERROR)
            die("corpus loading failed (check file access, available memory, and supported corpus size)");
        if (load_result == CORPUS_LOAD_EMPTY)
            die("the supplied corpus paths contained no usable text");
        have_corpus = true;
    }
    else {
        const char* project_paths[] = {
            "../data/books.txt",
            "../data/chatlogs.txt"
        };
        CorpusLoadResult load_result = corpus_load_files(&loaded_corpus,
            &arena, project_paths, 2, max_corpus_bytes);

        if (load_result == CORPUS_LOAD_ERROR)
            die("corpus loading failed (check file access, available memory, and supported corpus size)");

        if (load_result == CORPUS_LOAD_EMPTY) {
            const char* root_paths[] = {
                "data/books.txt",
                "data/chatlogs.txt"
            };
            load_result = corpus_load_files(&loaded_corpus, &arena,
                root_paths, 2, max_corpus_bytes);
            if (load_result == CORPUS_LOAD_ERROR)
                die("corpus loading failed (check file access, available memory, and supported corpus size)");
        }
        have_corpus = (load_result == CORPUS_LOAD_OK);
    }

    uint8_t* corpus = NULL;
    size_t corpus_len = 0;

    if (have_corpus) {
        corpus = loaded_corpus.data;
        corpus_len = loaded_corpus.length;
        printf("\nCorpus: %zu bytes from %zu file(s)\n",
            corpus_len, loaded_corpus.files_loaded);
    }
    else {
        if (user_path_count > 0)
            die("the supplied corpus files contained no usable text");

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
        printf("No training text found; using synthetic smoke corpus.\n");
    }

    if (corpus_len < (size_t)BATCH * SEQ + 1)
        die("corpus too small: need at least %d bytes", BATCH * SEQ + 1);
    if (corpus_len > (size_t)INT_MAX)
        die("corpus exceeds tokenizer's current INT_MAX byte/token-capacity limit");

    // ------------------------------------------------------------------
    // 3. Load or train tokenizer
    // ------------------------------------------------------------------
    Tokenizer tok;
    if (resume) {
        if (!tokenizer_load(&tok, &arena, "minima.bpe"))
            die("resume requested, but minima.bpe could not be loaded");
        if (tok.vocab_size != MODEL_VOCAB_SIZE)
            die("saved tokenizer vocabulary does not match MODEL_VOCAB_SIZE; start a fresh run");
        printf("Loaded tokenizer: minima.bpe (vocab = %d)\n", tok.vocab_size);
    }
    else {
        const size_t train_corpus_len = (corpus_len * 95u) / 100u;
        if (train_corpus_len < (size_t)BATCH * SEQ + 1)
            die("training portion is too small");

        const size_t tokenizer_sample_cap = options.tokenizer_sample_bytes;
        const size_t sample_len = train_corpus_len < tokenizer_sample_cap
            ? train_corpus_len : tokenizer_sample_cap;
        uint8_t* tokenizer_sample = (uint8_t*)arena_alloc(&arena, sample_len);
        if (!tokenizer_sample) die("tokenizer sample allocation failed");

        if (train_corpus_len <= tokenizer_sample_cap) {
            memcpy(tokenizer_sample, corpus, sample_len);
        }
        else {
            const size_t chunks = 64;
            const size_t chunk_len = sample_len / chunks;
            for (size_t i = 0; i < chunks; i++) {
                const size_t start =
                    ((train_corpus_len - chunk_len) * i) / (chunks - 1);
                memcpy(tokenizer_sample + i * chunk_len,
                    corpus + start, chunk_len);
            }
        }

        printf("Tokenizer training sample: %zu bytes\n", sample_len);
        if (!tokenizer_train(&tok, &arena, tokenizer_sample,
            sample_len, MODEL_VOCAB_SIZE))
            die("tokenizer_train failed");
        if (tok.vocab_size != MODEL_VOCAB_SIZE)
            die("BPE could not reach the configured vocabulary size");
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
    if (train_tokens < BATCH * SEQ + 1 ||
        validation_tokens < BATCH * SEQ + 1)
        die("corpus too small for a 95/5 train/validation split");

    printf("Training tokens:   %d (95%%)\n", train_tokens);
    printf("Validation tokens: %d (5%%)\n\n", validation_tokens);

    // ------------------------------------------------------------------
    // 5. Model, optimizer, gradients, and optional checkpoint restore
    // ------------------------------------------------------------------
    Model model;
    if (!model_init(&model, &arena, options.seed, tok.vocab_size))
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
        printf("Resumed checkpoint at optimizer step %lld\n\n",
            (long long)opt.step);
    }

    if (options.chat_mode) {
        KVCache chat_cache;
        if(!kvcache_init(&chat_cache,&arena,&model,model.max_seq))
            die("chat KV cache initialization failed");
        int chat_result=run_chat_console(&model,&tok,&chat_cache,&options);
        arena_free(&temp);arena_free(&arena);return chat_result;
    }

    Gradients grads;
    if (!gradients_init(&grads, &arena, &model))
        die("gradient allocation failed");

    // ------------------------------------------------------------------
    // 6. Numerical gradient check
    // ------------------------------------------------------------------
    uint16_t* check_tokens = arena_alloc(&arena,
        (size_t)BATCH * SEQ * sizeof(uint16_t));
    uint16_t* check_targets = arena_alloc(&arena,
        (size_t)BATCH * SEQ * sizeof(uint16_t));
    if (!check_tokens || !check_targets)
        die("gradient-check batch allocation failed");

    for (int b = 0; b < BATCH; b++) {
        const int start = b * SEQ;
        for (int t = 0; t < SEQ; t++) {
            check_tokens[b * SEQ + t] = all_ids[start + t];
            check_targets[b * SEQ + t] = all_ids[start + t + 1];
        }
    }

    if (!gradient_check(&model, &grads, &temp,
        check_tokens, check_targets, BATCH, SEQ))
        die("numerical gradient check failed");

    // Create a valid step-zero checkpoint before training starts. If the
    // process is interrupted early, --resume still has a matching baseline.
    if (!resume && !checkpoint_save("minima.model", &model, &opt,
        &rng, tokenizer_hash))
        die("initial checkpoint save failed");

    // ------------------------------------------------------------------
    // 7. Continue training; optimizer step and RNG state are global
    // ------------------------------------------------------------------
    Activations act;
    uint16_t* batch_tokens;
    uint16_t* batch_targets;
    float* dlogits;

    const int warmup_steps = options.warmup_steps < steps_to_run
        ? options.warmup_steps : (steps_to_run > 1 ? steps_to_run - 1 : 0);
    const int64_t initial_step = opt.step;

    printf(" step | train loss | valid loss | learning rate\n");
    printf("------+------------+------------+--------------\n");
    FILE* metrics=fopen(options.metrics_path,"a");
    if (!metrics) die("could not open metrics CSV: %s",options.metrics_path);
    if (fseek(metrics,0,SEEK_END)!=0) die("could not seek metrics CSV");
    if (ftell(metrics)==0) { fprintf(metrics,"step,train_loss,validation_loss,learning_rate,train_perplexity,validation_perplexity,elapsed_seconds\n"); fflush(metrics); }
    const clock_t training_started=clock();
    printf("Metrics CSV: %s\n",options.metrics_path);
    printf("Settings: lr=%.7g warmup=%d log=%d val_batches=%d checkpoint=%d seed=%llu tokenizer_sample=%zu\n",
        options.learning_rate,options.warmup_steps,options.log_interval,options.validation_batches,
        options.checkpoint_interval,(unsigned long long)options.seed,options.tokenizer_sample_bytes);

    double train_loss_sum = 0.0;
    int train_loss_count = 0;

    if (diagnostics_only) {
        printf("Diagnostics-only mode: optimizer updates skipped.\n");
    }
    else for (int local_step = 0; local_step < steps_to_run; local_step++) {
        arena_reset(&temp);
        if (!activations_init(&act, &temp, BATCH, SEQ, &model))
            die("activation allocation failed");

        dlogits = arena_alloc(&temp,
            (size_t)BATCH * SEQ * model.vocab_size * sizeof(float));
        batch_tokens = arena_alloc(&temp, BATCH * SEQ * sizeof(uint16_t));
        batch_targets = arena_alloc(&temp, BATCH * SEQ * sizeof(uint16_t));
        if (!dlogits || !batch_tokens || !batch_targets)
            die("training scratch allocation failed");

        for (int b = 0; b < BATCH; b++) {
            const int start = rng_int(&rng, 0, train_tokens - SEQ - 1);
            for (int t = 0; t < SEQ; t++) {
                batch_tokens[b * SEQ + t] = all_ids[start + t];
                batch_targets[b * SEQ + t] = all_ids[start + t + 1];
            }
        }

        float* logits = model_forward(&model, &act,
            batch_tokens, BATCH, SEQ);
        if (!logits) die("model_forward failed");

        const float loss = loss_cross_entropy(logits, batch_targets, dlogits,
            BATCH, SEQ, model.vocab_size);
        if (!isfinite(loss)) die("training loss became non-finite");

        gradients_zero(&grads, &model);
        model_backward(&model, &grads, &act, dlogits,
            batch_tokens, BATCH, SEQ);

        // Each invocation is an additional training segment with its own
        // warmup + cosine decay. AdamW's bias-correction step remains global.
        opt.lr = steps_to_run == 1 ? options.learning_rate :
            optim_lr_cosine(options.learning_rate, local_step + 1,
                warmup_steps, steps_to_run);
        optim_step(&opt, &model, &grads);
        if (opt.step % options.checkpoint_interval == 0) {
            if (!checkpoint_save("minima.model", &model, &opt, &rng, tokenizer_hash))
                die("periodic checkpoint save failed");
            printf("Checkpoint saved at optimizer step %lld\n", (long long)opt.step);
        }

        train_loss_sum += loss;
        train_loss_count++;

        if ((local_step + 1) % options.log_interval == 0 ||
            local_step == steps_to_run - 1 ||
            stop_file_exists(options.stop_file)) {
            const float validation_loss = evaluate_loss(&model, &temp,
                all_ids, train_tokens, validation_tokens, options.validation_batches);
            if (!isfinite(validation_loss))
                die("validation loss evaluation failed");

            const double average_train_loss =
                train_loss_sum / (double)train_loss_count;

            const double elapsed=(double)(clock()-training_started)/CLOCKS_PER_SEC;
            printf("%5lld | %10.4f | %10.4f | %.7f\n",(long long)opt.step,(float)average_train_loss,validation_loss,opt.lr);
            fprintf(metrics,"%lld,%.7f,%.7f,%.9g,%.7f,%.7f,%.3f\n",
                (long long)opt.step,(float)average_train_loss,validation_loss,opt.lr,
                expf((float)average_train_loss),expf(validation_loss),elapsed);
            fflush(metrics);
            train_loss_sum=0.0; train_loss_count=0;
            if (stop_file_exists(options.stop_file)) {
                printf("Cooperative stop requested; saving checkpoint and stopping after optimizer step %lld.\\n",
                    (long long)opt.step);
                break;
            }
        }
    }

    const int64_t final_step = opt.step;
    if (!diagnostics_only && (final_step == initial_step || final_step % options.checkpoint_interval != 0))
        if (!checkpoint_save("minima.model",&model,&opt,&rng,tokenizer_hash)) die("final checkpoint save failed");
    const double training_seconds=(double)(clock()-training_started)/CLOCKS_PER_SEC;
    fclose(metrics);
    printf("\nTraining segment complete: %lld -> %lld\n",
        (long long)initial_step, (long long)final_step);

    const float final_train_loss = evaluate_loss(&model, &temp,
        all_ids, 0, train_tokens, options.validation_batches);
    const float final_validation_loss = evaluate_loss(&model, &temp,
        all_ids, train_tokens, validation_tokens, options.validation_batches);
    if (!isfinite(final_train_loss) || !isfinite(final_validation_loss))
        die("final loss diagnostics are unavailable");

    printf("\nFinal evaluation loss and perplexity (fixed, evenly spaced windows)\n");
    printf("  train loss      : %.4f\n", final_train_loss);
    printf("  train perplexity: %.2f\n", expf(final_train_loss));
    printf("  valid loss      : %.4f\n", final_validation_loss);
    printf("  valid perplexity: %.2f\n", expf(final_validation_loss));
    printf("  training duration: %.2f seconds\n",training_seconds);
    printf("  total optimizer steps: %lld\n",(long long)opt.step);
    printf("  checkpoint status: saved (minima.model)\n");

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

    const float saved_weight = model.tok_emb[0];
    const float saved_moment = opt.m_tok_emb[0];
    const float saved_variance = opt.v_tok_emb[0];
    const uint64_t saved_rng_state = rng.state;
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
        opt.step != final_step || rng.state != saved_rng_state)
        die("checkpoint round-trip verification failed");

    printf("Checkpoint round-trip: PASS (weights, optimizer, step, RNG)\n");
    printf("Checkpoint compatibility: PASS (model architecture and tokenizer)\n");

    // ------------------------------------------------------------------
    // 9. KV-cache parity and deterministic multi-prompt generation
    // ------------------------------------------------------------------
    KVCache cache;
    if (!kvcache_init(&cache, &arena, &model, model.max_seq))
        die("KV cache initialization failed");

    if (!generation_check_kv_cache(&model, &cache, &temp,
        check_tokens, SEQ, 1e-4f))
        die("KV-cache parity check failed");

    if (!run_generation_benchmark(&model, &tok, &cache))
        die("deterministic generation benchmark failed");

    printf("\nSmoke test finished.\n");
    printf("Training, validation, persistence, and generation checks completed.\n");

    arena_free(&temp);
    arena_free(&arena);
    return 0;
}
