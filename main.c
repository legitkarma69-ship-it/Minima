#include "arena.h"
#include "util.h"
#include "tokenizer.h"
#include "model.h"
#include "optim.h"
#include "forward.h"
#include "backward.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define BATCH     4
#define SEQ       32
#define STEPS     80
#define LR        3e-4f

int main(void) {
    printf("tinyllm smoke test - overfitting a short string\n\n");

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
    // 2. Tiny synthetic corpus
    // ------------------------------------------------------------------
    const char* text =
        "The quick brown fox jumps over the lazy dog. "
        "Hello world from a tiny language model. "
        "Pure C is enough for small transformers. ";
    size_t text_len = strlen(text);

    // Repeat it so we have enough tokens
    size_t corpus_len = text_len * 40;
    uint8_t* corpus = arena_alloc(&arena, corpus_len + 1);
    if (!corpus) die("corpus allocation failed");

    for (size_t i = 0; i < 40; i++)
        memcpy(corpus + i * text_len, text, text_len);
    corpus[corpus_len] = 0;

    // ------------------------------------------------------------------
    // 3. Tokenizer
    // ------------------------------------------------------------------
    Tokenizer tok;
    if (!tokenizer_train(&tok, &arena, corpus, corpus_len, MODEL_VOCAB_SIZE))
        die("tokenizer_train failed");
    printf("vocab size = %d\n", tok.vocab_size);

    // ------------------------------------------------------------------
    // 4. Tokenise the whole corpus once
    // ------------------------------------------------------------------
    uint16_t* all_ids = arena_alloc(&arena, corpus_len * sizeof(uint16_t));
    if (!all_ids) die("token id allocation failed");

    int n_tokens = tokenizer_encode(&tok, corpus, corpus_len, all_ids, (int)corpus_len);
    printf("total tokens = %d\n\n", n_tokens);

    if (n_tokens < BATCH * SEQ + 1)
        die("not enough tokens");

    // ------------------------------------------------------------------
    // 5. Model + optimiser + gradients
    // ------------------------------------------------------------------
    // All translation units now use the same values from config.h.
    Model model;
    if (!model_init(&model, &arena, 42))
        die("model_init failed");
    model_summary(&model);

    Optim opt;
    if (!optim_init(&opt, &model, &arena, LR, 0.9f, 0.95f, 1e-8f, 0.01f))
        die("optim_init failed");

    Gradients grads;
    if (!gradients_init(&grads, &arena, &model))
        die("gradients_init failed");

    // ------------------------------------------------------------------
    // 6. Activations (fixed batch / seq for the test)
    // ------------------------------------------------------------------
    Activations act;
    if (!activations_init(&act, &temp, BATCH, SEQ, &model))
        die("activations_init failed");

    // ------------------------------------------------------------------
    // 7. Training loop
    // ------------------------------------------------------------------
    Rng rng;
    rng_seed(&rng, 1234);

    uint16_t* batch_tokens;
    uint16_t* batch_targets;
    float* dlogits;

    printf("step | loss\n");
    printf("-----+--------\n");

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
            int start = rng_int(&rng, 0, n_tokens - SEQ - 1);
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

        // AdamW
        optim_step(&opt, &model, &grads);

        if (step % 5 == 0 || step == STEPS - 1)
            printf("%4d | %.4f\n", step, loss);
    }

    printf("\nSmoke test finished.\n");
    printf("If loss decreased significantly, the full pipeline is working.\n");

    arena_free(&temp);
    arena_free(&arena);
    return 0;
}
