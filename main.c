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

#define BATCH     4
#define SEQ       32
#define DEFAULT_STEPS 2000
#define LR        3e-4f
#define WARMUP_STEPS 100
#define LOG_INTERVAL 100
#define VALIDATION_BATCHES 16

static uint64_t hash_bytes(uint64_t hash, const void* data, size_t length) {
    const uint8_t* bytes = (const uint8_t*)data;
    for (size_t i = 0; i < length; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static void print_usage(const char* exe) {
    printf("Usage: %s [--resume] [--steps N] [corpus.txt ...]\n", exe);
    printf("  --resume   Continue from minima.model and minima.bpe\n");
    printf("  --steps N  Run N additional optimizer steps (default: %d)\n",
        DEFAULT_STEPS);
    printf("  --help     Show this help\n");
}

static float evaluate_loss(Model* model, Arena* temp,
    const uint16_t* all_ids, int validation_start, int validation_tokens) {

    if (!model || !temp || !all_ids ||
        validation_start < 0 || validation_tokens < BATCH * SEQ + 1)
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
    for (int eval_batch = 0; eval_batch < VALIDATION_BATCHES; eval_batch++) {
        for (int b = 0; b < BATCH; b++) {
            const size_t sample_index = (size_t)eval_batch * BATCH + b;
            const size_t sample_count = (size_t)VALIDATION_BATCHES * BATCH;
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

    return (float)(loss_sum / VALIDATION_BATCHES);
}

static int parse_arguments(int argc, char** argv, bool* resume,
    int* steps, const char** corpus_paths) {
    int path_count = 0;
    *resume = false;
    *steps = DEFAULT_STEPS;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
        else if (strcmp(argv[i], "--resume") == 0) {
            *resume = true;
        }
        else if (strcmp(argv[i], "--steps") == 0) {
            if (i + 1 >= argc) die("--steps requires a positive integer");
            char* end = NULL;
            errno = 0;
            long value = strtol(argv[++i], &end, 10);
            if (errno != 0 || !end || *end != '\0' ||
                value <= 0 || value > INT_MAX)
                die("--steps must be an integer between 1 and INT_MAX");
            *steps = (int)value;
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
    return path_count;
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

    printf("\nDeterministic generation benchmark (greedy, top-k=1)\n");
    uint16_t prompt_ids[64];
    uint16_t output_ids[64];
    uint16_t verify_ids[64];
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

        kvcache_reset(cache);
        Rng generation_rng;
        rng_seed(&generation_rng, 5678u + (uint64_t)i);
        const int generated_len = generate(model, tok, cache,
            prompt_ids, prompt_len, output_ids, 64,
            32, 1.0f, 1, &generation_rng);
        if (generated_len <= prompt_len) {
            printf("  [%zu] generation failed\n", i + 1);
            return false;
        }

        // Repeat from the same cache state and RNG seed. Greedy top-k=1
        // must reproduce the exact same token sequence.
        kvcache_reset(cache);
        Rng verify_rng;
        rng_seed(&verify_rng, 5678u + (uint64_t)i);
        const int verify_len = generate(model, tok, cache,
            prompt_ids, prompt_len, verify_ids, 64,
            32, 1.0f, 1, &verify_rng);
        if (verify_len != generated_len ||
            memcmp(output_ids, verify_ids,
                (size_t)generated_len * sizeof(uint16_t)) != 0) {
            printf("  [%zu] deterministic generation: FAIL\n", i + 1);
            return false;
        }

        const int bytes = tokenizer_decode(tok, output_ids,
            generated_len, output_text, (int)sizeof(output_text));
        if (bytes <= 0) {
            printf("  [%zu] decoding failed\n", i + 1);
            return false;
        }
        printf("  [%zu] %s\n", i + 1, output_text);
    }

    printf("Determinism check: PASS (all prompts reproduced exactly)\n");
    return true;
}

int main(int argc, char** argv) {
    bool resume = false;
    int steps_to_run = DEFAULT_STEPS;
    const char* user_paths[32];
    const int user_path_count = parse_arguments(argc, argv,
        &resume, &steps_to_run, user_paths);

    printf("ProjectMinima text-corpus training experiment\n");
    printf("Vocabulary target: %d\n", MODEL_VOCAB_SIZE);
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

        const size_t tokenizer_sample_cap = 256u * 1024u;
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
    if (!model_init(&model, &arena, 42, tok.vocab_size))
        die("model_init failed");
    model_summary(&model);

    Optim opt;
    if (!optim_init(&opt, &model, &arena, LR, 0.9f, 0.95f, 1e-8f, 0.01f))
        die("optim_init failed");

    Rng rng;
    rng_seed(&rng, 1234);

    if (resume) {
        if (!checkpoint_load("minima.model", &model, &opt,
            &rng, tokenizer_hash))
            die("checkpoint load failed: model/tokenizer mismatch or invalid checkpoint");
        printf("Resumed checkpoint at optimizer step %lld\n\n",
            (long long)opt.step);
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

    const int warmup_steps = steps_to_run < WARMUP_STEPS
        ? (steps_to_run > 1 ? steps_to_run / 10 : 0) : WARMUP_STEPS;
    const int64_t initial_step = opt.step;

    printf(" step | train loss | valid loss | learning rate\n");
    printf("------+------------+------------+--------------\n");

    double train_loss_sum = 0.0;
    int train_loss_count = 0;

    for (int local_step = 0; local_step < steps_to_run; local_step++) {
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
        opt.lr = optim_lr_cosine(LR, local_step + 1,
            warmup_steps, steps_to_run);
        optim_step(&opt, &model, &grads);

        train_loss_sum += loss;
        train_loss_count++;

        if ((local_step + 1) % LOG_INTERVAL == 0 ||
            local_step == steps_to_run - 1) {
            const float validation_loss = evaluate_loss(&model, &temp,
                all_ids, train_tokens, validation_tokens);
            if (!isfinite(validation_loss))
                die("validation loss evaluation failed");

            const double average_train_loss =
                train_loss_sum / (double)train_loss_count;

            printf("%5lld | %10.4f | %10.4f | %.7f\n",
                (long long)opt.step, (float)average_train_loss,
                validation_loss, opt.lr);

            train_loss_sum = 0.0;
            train_loss_count = 0;

            // Save at every reporting interval so an interrupted run loses
            // at most LOG_INTERVAL - 1 optimizer steps.
            if (!checkpoint_save("minima.model", &model, &opt,
                &rng, tokenizer_hash))
                die("periodic checkpoint save failed");
        }
    }

    const int64_t final_step = opt.step;
    printf("\nTraining segment complete: %lld -> %lld\n",
        (long long)initial_step, (long long)final_step);

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
    printf("Checkpoint compatibility: PASS (tokenizer and corpus fingerprints)\n");

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
