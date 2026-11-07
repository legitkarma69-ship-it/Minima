#include "backward.h"
#include "util.h"

#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
    return 1; } } while (0)

static bool all_zero(const float* values, size_t count) {
    for (size_t i = 0; i < count; i++)
        if (values[i] != 0.0f) return false;
    return true;
}

static int check_gradient_tensors(const Gradients* g, const Model* model) {
    const size_t dim = (size_t)model->dim;
    const size_t layers = (size_t)model->n_layers;
    const size_t ffn = (size_t)model->ffn_dim;
    const size_t counts[] = {
        (size_t)model->vocab_size * dim,
        (size_t)model->max_seq * dim,
        layers * 3 * dim * dim,
        layers * dim * dim,
        layers * ffn * dim,
        layers * ffn * dim,
        layers * dim * ffn,
        layers * dim,
        layers * dim,
        dim
    };
    const float* tensors[] = {
        g->tok_emb, g->pos_emb, g->attn_qkv, g->attn_proj,
        g->ffn_up, g->ffn_gate, g->ffn_down,
        g->rms1_scale, g->rms2_scale, g->rms_final_scale
    };
    for (size_t i = 0; i < sizeof(counts) / sizeof(counts[0]); i++)
        if (!all_zero(tensors[i], counts[i])) return 0;
    return 1;
}

int main(void) {
    Arena model_arena;
    Arena gradient_arena;
    CHECK(arena_init(&model_arena, 1024u * 1024u));
    CHECK(arena_init(&gradient_arena, 1024u * 1024u));

    const ModelConfig config = {
        260, 8, 1, 2, 16, 32
    };
    Model model;
    CHECK(model_init_config(&model, &model_arena, 1234, &config));

    Gradients gradients;
    memset(&gradients, 0xA5, sizeof(gradients));
    CHECK(gradients_init(&gradients, &gradient_arena, &model));
    CHECK(gradients_check_layout(&gradients, &model, &gradient_arena));
    CHECK(gradients.weight_bytes == model.weight_bytes);
    CHECK(check_gradient_tensors(&gradients, &model));

    /* A partial allocation failure must not consume arena space or leave
       a half-initialized gradient layout behind. */
    Arena small_arena;
    CHECK(arena_init(&small_arena, 10u * 1024u));
    Gradients failed;
    memset(&failed, 0xA5, sizeof(failed));
    const size_t used_before = small_arena.used;
    const size_t peak_before = small_arena.peak;
    CHECK(!gradients_init(&failed, &small_arena, &model));
    CHECK(small_arena.used == used_before);
    CHECK(small_arena.peak == peak_before);
    CHECK(failed.tok_emb == NULL && failed.weight_bytes == 0);

    arena_free(&small_arena);
    arena_free(&gradient_arena);
    arena_free(&model_arena);
    puts("Direct zero-allocation gradient tests passed.");
    return 0;
}
