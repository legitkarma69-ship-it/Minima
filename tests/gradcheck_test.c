#include "backward.h"
#include "gradcheck.h"
#include "model.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
    return 1; } } while (0)

static uint32_t next_value(uint32_t* state) {
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

static int run_case(uint64_t seed) {
    const ModelConfig config = { 2048, 64, 2, 2, 128, 64 };
    Arena model_arena, gradient_arena, temp_arena;
    CHECK(arena_init(&model_arena, 16u * 1024u * 1024u));
    CHECK(arena_init(&gradient_arena, 16u * 1024u * 1024u));
    CHECK(arena_init(&temp_arena, 64u * 1024u * 1024u));

    Model model;
    CHECK(model_init_config(&model, &model_arena, seed, &config));
    Gradients gradients;
    CHECK(gradients_init(&gradients, &gradient_arena, &model));

    enum { BATCH = 4, SEQUENCE = 32, COUNT = BATCH * SEQUENCE };
    uint16_t tokens[COUNT];
    uint16_t targets[COUNT];
    uint32_t state = (uint32_t)seed ^ 0x9e3779b9u;
    for (int i = 0; i < COUNT; i++) {
        tokens[i] = (uint16_t)(4u + next_value(&state) % 2044u);
        targets[i] = (uint16_t)(4u + next_value(&state) % 2044u);
    }

    printf("\nDefault model gradient-check regression, seed=%llu\n",
        (unsigned long long)seed);
    const bool passed = gradient_check(&model, &gradients, &temp_arena,
        tokens, targets, BATCH, SEQUENCE);

    /* Reuse the same activation workspace for two backward passes. With
       parameter gradients cleared between passes, every gradient must match
       bit-for-bit; stale dx/dx_tmp/dqkv state must never leak across calls. */
    bool reuse_passed = false;
    if (passed) {
        arena_reset(&temp_arena);
        Activations act;
        if (activations_init(&act, &temp_arena, BATCH, SEQUENCE, &model)) {
            const size_t logit_count =
                (size_t)BATCH * SEQUENCE * (size_t)model.vocab_size;
            float* dlogits = arena_alloc(&temp_arena,
                logit_count * sizeof(float));
            float* snapshot = (float*)malloc(model.weight_bytes);
            float* logits = model_forward(&model, &act, tokens, BATCH, SEQUENCE);
            if (dlogits && snapshot && logits &&
                isfinite(loss_cross_entropy(logits, targets, dlogits,
                    BATCH, SEQUENCE, model.vocab_size))) {
                gradients_zero(&gradients, &model);
                model_backward(&model, &gradients, &act, dlogits,
                    tokens, BATCH, SEQUENCE);
                memcpy(snapshot, gradients.tok_emb, model.weight_bytes);
                gradients_zero(&gradients, &model);
                model_backward(&model, &gradients, &act, dlogits,
                    tokens, BATCH, SEQUENCE);
                reuse_passed =
                    memcmp(snapshot, gradients.tok_emb, model.weight_bytes) == 0;
            }
            free(snapshot);
        }
    }
    printf("  backward workspace reuse %s\n", reuse_passed ? "PASS" : "FAIL");

    arena_free(&temp_arena);
    arena_free(&gradient_arena);
    arena_free(&model_arena);
    return passed && reuse_passed ? 0 : 1;
}

int main(void) {
    CHECK(run_case(1234) == 0);
    CHECK(run_case(5678) == 0);
    CHECK(run_case(98765) == 0);
    puts("Default-configuration numerical gradient checks passed.");
    return 0;
}
