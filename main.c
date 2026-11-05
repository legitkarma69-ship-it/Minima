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

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>

#define BATCH     4
#define SEQ       32
#define STEPS     2000
#define LR        3e-4f
#define WARMUP_STEPS 100
#define LOG_INTERVAL 100
#define VALIDATION_BATCHES 16

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

int main(int argc, char** argv) {
    printf("ProjectMinima text-corpus training smoke test\n\n");
    if (!optim_check_adamw())
        die("AdamW numerical regression check failed");
    printf("AdamW bias-correction check: PASS\n");

    // ------------------------------------------------------------------
    // 1. Arenas
    // ------------------------------------------------------------------
    Arena arena;          // permanent (weights, optim, tokenizer)
    Arena temp;           // per-step activations / gradients scratch

    if (!arena_init(&arena, 512ull * 1024 * 1024))  // 512 MB
        die("permanent arena failed");
    if (!arena_init(&temp, 256ull * 1024 * 1024))  // 256 MB
        die("temp arena failed");

    // ------------------------------------------------------------------
    // 2. Load plain-text training data
    // ------------------------------------------------------------------
    const size_t max_corpus_bytes = 5u * 1024u * 1024u;
    Corpus loaded_corpus;
    bool have_corpus = false;

    if (argc > 1) {
        if (argc - 1 > 32)
            die("too many corpus paths (maximum 32)");

        const char* user_paths[32];
        for (int i = 1; i < argc; i++)
            user_paths[i - 1] = argv[i];

        if (!corpus_load_files(&loaded_corpus, &arena,
            user_paths, (size_t)(argc - 1), max_corpus_bytes))
            die("could not load usable text from the supplied corpus paths");
        have_corpus = true;
    }
    else {
        const char* project_paths[] = {
            "../data/books.txt",
            "../data/chatlogs.txt"
        };
        have_corpus = corpus_load_files(&loaded_corpus, &arena,
            project_paths, 2, max_corpus_bytes);

        // Also support launching with the repository root as working directory.
        if (!have_corpus) {
            const char* root_paths[] = {
                "data/books.txt",
                "data/chatlogs.txt"
            };
            have_corpus = corpus_load_files(&loaded_corpus, &arena,
                root_paths, 2, max_corpus_bytes);
        }
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
        if (argc > 1)
            die("the supplied corpus files contained no usable text");

        // Empty placeholder files fall back to the deterministic smoke corpus.
        const char* smoke_text =
            "The quick brown fox jumps over the lazy dog. "
            "Hello world from a tiny language model. "
            "Pure C is enough for small transformers. ";
        size_t smoke_len = strlen(smoke_text);
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
        die("corpus exceeds tokenizer's integer token-capacity limit");

    // ------------------------------------------------------------------
    // 3. Train BPE on a bounded, evenly distributed corpus sample
    // ------------------------------------------------------------------
    // Hold out the final 5% of the corpus for validation. BPE learns only
    // from the training portion, not from the held-out validation text.
    const size_t train_corpus_len =
        (corpus_len * 95u) / 100u;
    if (train_corpus_len < (size_t)BATCH * SEQ + 1)
        die("training portion is too small");

    const size_t tokenizer_sample_cap = 256u * 1024u;
    size_t sample_len = train_corpus_len < tokenizer_sample_cap
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
            size_t start = ((train_corpus_len - chunk_len) * i) / (chunks - 1);
            memcpy(tokenizer_sample + i * chunk_len,
                corpus + start, chunk_len);
        }
    }

    printf("Tokenizer training sample: %zu bytes\n", sample_len);

    // ------------------------------------------------------------------
    // 4. Tokenizer and corpus encoding
    // ------------------------------------------------------------------
    Tokenizer tok;
    if (!tokenizer_train(&tok, &arena, tokenizer_sample,
        sample_len, MODEL_VOCAB_SIZE))
        die("tokenizer_train failed");
    printf("vocab size = %d\n", tok.vocab_size);

    uint16_t* all_ids = (uint16_t*)arena_alloc(&arena,
        corpus_len * sizeof(uint16_t));
    if (!all_ids) die("token id allocation failed");

    int n_tokens = tokenizer_encode(&tok, corpus, corpus_len,
        all_ids, (int)corpus_len);
    printf("total tokens = %d\n\n", n_tokens);

    const int train_tokens = (n_tokens * 95) / 100;
    const int validation_tokens = n_tokens - train_tokens;
    if (train_tokens < BATCH * SEQ + 1 ||
        validation_tokens < BATCH * SEQ + 1)
        die("corpus too small for a 95/5 train/validation split");

    printf("Training tokens:   %d (95%%)\n", train_tokens);
    printf("Validation tokens: %d (5%%)\n\n", validation_tokens);

    // ------------------------------------------------------------------
    // 5. Model + optimiser + gradients
    // ------------------------------------------------------------------
    // All translation units now use the same values from config.h.
    Model model;
    if (!model_init(&model, &arena, 42, tok.vocab_size))
        die("model_init failed");
    model_summary(&model);

    Optim opt;
    if (!optim_init(&opt, &model, &arena, LR, 0.9f, 0.95f, 1e-8f, 0.01f))
        die("optim_init failed");

    Gradients grads;
    if (!gradients_init(&grads, &arena, &model))
        die("gradients_init failed");

    // ------------------------------------------------------------------
    // 6. Numerical gradient check
    // ------------------------------------------------------------------
    // Use a fixed, deterministic batch so every finite-difference
    // evaluation sees exactly the same tokens and targets.
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

    // ------------------------------------------------------------------
    // 7. Training loop
    // ------------------------------------------------------------------
    Activations act;
    Rng rng;
    rng_seed(&rng, 1234);

    uint16_t* batch_tokens;
    uint16_t* batch_targets;
    float* dlogits;

    printf(" step | train loss | valid loss | learning rate\n");
    printf("------+------------+------------+--------------\n");

    double train_loss_sum = 0.0;
    int train_loss_count = 0;

    for (int step = 0; step < STEPS; step++) {
        // Activations and batch buffers are scratch allocations.
        arena_reset(&temp);
        if (!activations_init(&act, &temp, BATCH, SEQ, &model))
            die("activation allocation failed");

        dlogits = arena_alloc(&temp,
            (size_t)BATCH * SEQ * model.vocab_size * sizeof(float));
        batch_tokens = arena_alloc(&temp, BATCH * SEQ * sizeof(uint16_t));
        batch_targets = arena_alloc(&temp, BATCH * SEQ * sizeof(uint16_t));

        if (!dlogits || !batch_tokens || !batch_targets)
            die("training scratch allocation failed");

        // Sample random starting positions
        for (int b = 0; b < BATCH; b++) {
            int start = rng_int(&rng, 0, train_tokens - SEQ - 1);
            for (int t = 0; t < SEQ; t++) {
                batch_tokens[b * SEQ + t] = all_ids[start + t];
                batch_targets[b * SEQ + t] = all_ids[start + t + 1];
            }
        }

        // Forward
        float* logits = model_forward(&model, &act, batch_tokens, BATCH, SEQ);
        if (!logits) die("model_forward failed");

        // Loss + dlogits
        float loss = loss_cross_entropy(logits, batch_targets, dlogits,
            BATCH, SEQ, model.vocab_size);

        // Backward
        gradients_zero(&grads, &model);
        model_backward(&model, &grads, &act, dlogits, batch_tokens, BATCH, SEQ);

        // Linear warmup followed by cosine decay.
        opt.lr = optim_lr_cosine(LR, step + 1, WARMUP_STEPS, STEPS);

        // AdamW
        optim_step(&opt, &model, &grads);

        train_loss_sum += loss;
        train_loss_count++;

        if ((step + 1) % LOG_INTERVAL == 0 || step == STEPS - 1) {
            const float validation_loss = evaluate_loss(&model, &temp,
                all_ids, train_tokens, validation_tokens);
            if (!isfinite(validation_loss))
                die("validation loss evaluation failed");

            const double average_train_loss =
                train_loss_sum / (double)train_loss_count;

            printf("%5d | %10.4f | %10.4f | %.7f\n",
                step + 1, (float)average_train_loss,
                validation_loss, opt.lr);

            train_loss_sum = 0.0;
            train_loss_count = 0;
        }
    }

    // ------------------------------------------------------------------
    // 8. Save tokenizer, model weights, and AdamW state
    // ------------------------------------------------------------------
    // Store the base learning rate in the checkpoint. The schedule is
    // deterministic from the configured run length and step count.
    opt.lr = LR;

    if (!tokenizer_save(&tok, "minima.bpe"))
        die("failed to save tokenizer to minima.bpe");
    if (!checkpoint_save("minima.model", &model, &opt))
        die("failed to save model checkpoint to minima.model");

    // Verify that the tokenizer can be reloaded and produces identical IDs.
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
        original_count != loaded_count ||
        memcmp(original_ids, loaded_ids,
            (size_t)original_count * sizeof(uint16_t)) != 0)
        die("tokenizer round-trip verification failed");

    printf("Tokenizer round-trip: PASS (vocabulary, merges, encoding)\n");

    printf("\nSaved tokenizer: minima.bpe\n");
    printf("Saved model + optimizer: minima.model (step %d)\n", STEPS);

    // Exercise the loader by deliberately corrupting one saved value,
    // then confirm the checkpoint restores both weights and AdamW moments.
    float saved_weight = model.tok_emb[0];
    float saved_moment = opt.m_tok_emb[0];
    float saved_variance = opt.v_tok_emb[0];
    model.tok_emb[0] += 1.0f;
    opt.m_tok_emb[0] += 1.0f;
    opt.v_tok_emb[0] += 1.0f;

    if (!checkpoint_load("minima.model", &model, &opt))
        die("checkpoint round-trip load failed");
    if (model.tok_emb[0] != saved_weight ||
        opt.m_tok_emb[0] != saved_moment ||
        opt.v_tok_emb[0] != saved_variance ||
        opt.step != STEPS)
        die("checkpoint round-trip verification failed");

    printf("Checkpoint round-trip: PASS (weights, optimizer moments, step)\n");

    // ------------------------------------------------------------------
    // 9. Autoregressive generation smoke test
    // ------------------------------------------------------------------
    const char* prompt_text = "The quick ";
    const size_t prompt_bytes = strlen(prompt_text);
    uint16_t* prompt_ids = arena_alloc(&arena,
        prompt_bytes * sizeof(uint16_t));
    uint16_t generated_ids[32];
    char generated_text[32 * TOK_MAX_TOKEN_LEN + 1];

    if (!prompt_ids) die("generation prompt allocation failed");

    int prompt_len = tokenizer_encode(&tok,
        (const uint8_t*)prompt_text, prompt_bytes,
        prompt_ids, (int)prompt_bytes);
    if (prompt_len <= 0) die("generation prompt tokenization failed");

    KVCache cache;
    if (!kvcache_init(&cache, &arena, &model, model.max_seq))
        die("KV cache initialization failed");

    if (!generation_check_kv_cache(&model, &cache, &temp,
        check_tokens, SEQ, 1e-4f))
        die("KV-cache parity check failed");

    Rng generation_rng;
    rng_seed(&generation_rng, 5678);

    int generated_len = generate(&model, &tok, &cache,
        prompt_ids, prompt_len, generated_ids, 32,
        16, 0.8f, 40, &generation_rng);
    if (generated_len <= prompt_len)
        die("generation smoke test failed");

    int generated_bytes = tokenizer_decode(&tok, generated_ids,
        generated_len, generated_text, (int)sizeof(generated_text));
    if (generated_bytes <= 0)
        die("generation decode failed");

    printf("\nGeneration smoke test\n");
    printf("  prompt tokens : %d\n", prompt_len);
    printf("  generated     : %d new tokens\n", generated_len - prompt_len);
    printf("  text          : %s\n", generated_text);

    printf("\nSmoke test finished.\n");
    printf("If loss decreased significantly, the full pipeline is working.\n");

    arena_free(&temp);
    arena_free(&arena);
    return 0;
}
