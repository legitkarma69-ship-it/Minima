#include "backward.h"
#include "gradcheck.h"
#include "model.h"

#include <stdio.h>
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

    arena_free(&temp_arena);
    arena_free(&gradient_arena);
    arena_free(&model_arena);
    return passed ? 0 : 1;
}

int main(void) {
    CHECK(run_case(1234) == 0);
    CHECK(run_case(5678) == 0);
    CHECK(run_case(98765) == 0);
    puts("Default-configuration numerical gradient checks passed.");
    return 0;
}
