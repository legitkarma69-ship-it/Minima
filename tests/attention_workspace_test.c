#include "backward.h"
#include "forward.h"
#include "model.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
    return 1; } } while (0)

static int run_sequence(Model* model, Gradients* gradients,
    Arena* temp_arena, int sequence) {
    enum { BATCH = 1 };
    const size_t count = (size_t)BATCH * (size_t)sequence;
    const size_t logit_count = count * (size_t)model->vocab_size;
    uint16_t* tokens = (uint16_t*)malloc(count * sizeof(uint16_t));
    float* dlogits = (float*)malloc(logit_count * sizeof(float));
    CHECK(tokens && dlogits);

    for (size_t i = 0; i < count; i++)
        tokens[i] = (uint16_t)(4u + (unsigned)(i % 2044u));
    for (size_t i = 0; i < logit_count; i++)
        dlogits[i] = (float)((int)(i % 7u) - 3) * 1.0e-6f;

    arena_reset(temp_arena);
    Activations act;
    CHECK(activations_init(&act, temp_arena, BATCH, sequence, model));
    CHECK(act.attention_scores != NULL);

    const uintptr_t arena_begin = (uintptr_t)temp_arena->base;
    const uintptr_t arena_end = arena_begin + temp_arena->used;
    const uintptr_t scratch_begin = (uintptr_t)act.attention_scores;
    CHECK(scratch_begin >= arena_begin && scratch_begin <= arena_end);
    CHECK((size_t)(arena_end - scratch_begin) >=
        (size_t)sequence * sizeof(float));

    float* logits = model_forward(model, &act, tokens, BATCH, sequence);
    CHECK(logits != NULL);
    for (size_t i = 0; i < logit_count; i++)
        CHECK(isfinite(logits[i]));

    gradients_zero(gradients, model);
    model_backward(model, gradients, &act, dlogits, tokens, BATCH, sequence);
    CHECK(isfinite(gradients->tok_emb[0]));
    CHECK(isfinite(gradients->attn_qkv[0]));
    CHECK(isfinite(gradients->rms_final_scale[0]));

    free(dlogits);
    free(tokens);
    printf("  sequence %d attention workspace PASS\n", sequence);
    return 0;
}

int main(void) {
    const ModelConfig config = { 2048, 64, 2, 2, 128, 512 };
    Arena model_arena, gradient_arena, temp_arena;
    CHECK(arena_init(&model_arena, 16u * 1024u * 1024u));
    CHECK(arena_init(&gradient_arena, 16u * 1024u * 1024u));
    CHECK(arena_init(&temp_arena, 64u * 1024u * 1024u));

    Model model;
    CHECK(model_init_config(&model, &model_arena, 24680, &config));
    Gradients gradients;
    CHECK(gradients_init(&gradients, &gradient_arena, &model));

    puts("Attention arena workspace regression");
    CHECK(run_sequence(&model, &gradients, &temp_arena, 1) == 0);
    CHECK(run_sequence(&model, &gradients, &temp_arena, 32) == 0);
    CHECK(run_sequence(&model, &gradients, &temp_arena, 512) == 0);

    arena_free(&temp_arena);
    arena_free(&gradient_arena);
    arena_free(&model_arena);
    puts("Attention arena workspace regression passed.");
    return 0;
}
