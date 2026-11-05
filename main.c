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
    printf("tinyllm smoke test – overfitting a short string\n\n");

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
    for (size_t i = 0; i < 40; i++)
        memcpy(corpus + i * text_len, text, text_len);
    corpus[corpus_len] = 0;

    // ------------------------------------------------------------------
    // 3. Tokenizer (tiny vocab for the smoke test)
    // ------------------------------------------------------------------
    Tokenizer tok;
    if (!tokenizer_train(&tok, &arena, corpus, corpus_len, 512))
        die("tokenizer_train failed");
    printf("vocab size = %d\n", tok.vocab_size);

    // ------------------------------------------------------------------
    // 4. Tokenise the whole corpus once
    // ------------------------------------------------------------------
    uint16_t* all_ids = arena_alloc(&arena, corpus_len * sizeof(uint16_t));
    int n_tokens = tokenizer_encode(&tok, corpus, corpus_len, all_ids, (int)corpus_len);
    printf("total tokens = %d\n\n", n_tokens);

    if (n_tokens < BATCH * SEQ + 1)
        die("not enough tokens");

    // ------------------------------------------------------------------
    // 5. Model + optimiser + gradients
    // ------------------------------------------------------------------
    // Override config for the smoke test (much smaller)
    // (In a real run you keep the values from config.h)
#undef MODEL_DIM
#undef MODEL_N_LAYERS
#undef MODEL_N_HEADS
#undef MODEL_FFN_DIM
#undef MODEL_MAX_SEQ
#undef MODEL_VOCAB_SIZE
#define MODEL_VOCAB_SIZE  512
#define MODEL_DIM         64
#define MODEL_N_LAYERS    2
#define MODEL_N_HEADS     2
#define MODEL_HEAD_DIM    32
#define MODEL_FFN_DIM     128
#define MODEL_MAX_SEQ     64

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

    uint16_t* batch_tokens = arena_alloc(&temp, BATCH * SEQ * sizeof(uint16_t));
    uint16_t* batch_targets = arena_alloc(&temp, BATCH * SEQ * sizeof(uint16_t));
    float* dlogits = arena_alloc(&temp, (size_t)BATCH * SEQ * model.vocab_size * sizeof(float));

    printf("step | loss\n");
    printf("-----+--------\n");

    for (int step = 0; step < STEPS; step++) {
        arena_reset(&temp);   // free all per-step scratch (activations stay valid because we re-init below if needed)

        // Re-init activations after reset (simple approach)
        activations_init(&act, &temp, BATCH, SEQ, &model);
        dlogits = arena_alloc(&temp, (size_t)BATCH * SEQ * model.vocab_size * sizeof(float));
        batch_tokens = arena_alloc(&temp, BATCH * SEQ * sizeof(uint16_t));
        batch_targets = arena_alloc(&temp, BATCH * SEQ * sizeof(uint16_t));

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