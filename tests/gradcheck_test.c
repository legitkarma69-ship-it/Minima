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

static void gradient_counts(const Model* model, size_t counts[10]) {
    const size_t dim = (size_t)model->dim;
    const size_t layers = (size_t)model->n_layers;
    const size_t ffn = (size_t)model->ffn_dim;
    counts[0] = (size_t)model->vocab_size * dim;
    counts[1] = (size_t)model->max_seq * dim;
    counts[2] = layers * 3 * dim * dim;
    counts[3] = layers * dim * dim;
    counts[4] = layers * ffn * dim;
    counts[5] = layers * ffn * dim;
    counts[6] = layers * dim * ffn;
    counts[7] = layers * dim;
    counts[8] = layers * dim;
    counts[9] = dim;
}

static bool snapshot_gradients(const Gradients* g, const Model* model,
    float* snapshot) {
    size_t counts[10];
    const float* tensors[] = {
        g->tok_emb, g->pos_emb, g->attn_qkv, g->attn_proj,
        g->ffn_up, g->ffn_gate, g->ffn_down,
        g->rms1_scale, g->rms2_scale, g->rms_final_scale
    };
    gradient_counts(model, counts);
    size_t offset = 0;
    for (size_t i = 0; i < 10; i++) {
        memcpy(snapshot + offset, tensors[i], counts[i] * sizeof(float));
        offset += counts[i];
    }
    return offset <= SIZE_MAX / sizeof(float) &&
        offset * sizeof(float) == model->weight_bytes;
}

static bool gradients_match_snapshot(const Gradients* g, const Model* model,
    const float* snapshot) {
    size_t counts[10];
    const float* tensors[] = {
        g->tok_emb, g->pos_emb, g->attn_qkv, g->attn_proj,
        g->ffn_up, g->ffn_gate, g->ffn_down,
        g->rms1_scale, g->rms2_scale, g->rms_final_scale
    };
    (void)gradient_tensors(g, model, counts);
    size_t offset = 0;
    for (size_t i = 0; i < 10; i++) {
        const size_t bytes = counts[i] * sizeof(float);
        if (memcmp(snapshot + offset, tensors[i], bytes) != 0)
            return false;
        offset += counts[i];
    }
    return offset <= SIZE_MAX / sizeof(float) &&
        offset * sizeof(float) == model->weight_bytes;
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

    /* Reuse the same forward/backward workspace twice. Clearing parameter
       gradients between calls must produce identical results, proving that
       scratch buffers do not retain stale values from the previous pass. */
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
                const bool saved = snapshot_gradients(
                    &gradients, &model, snapshot);
                gradients_zero(&gradients, &model);
                model_backward(&model, &gradients, &act, dlogits,
                    tokens, BATCH, SEQUENCE);
                reuse_passed = saved &&
                    gradients_match_snapshot(&gradients, &model, snapshot);
            }
            free(snapshot);
        }
    }
    printf("  backward workspace reuse %s\\n",
        reuse_passed ? "PASS" : "FAIL");

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
