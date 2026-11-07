#include "checkpoint.h"
#include "model.h"
#include "optim.h"
#include "util.h"

#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
    return 1; } } while (0)

int main(void) {
    const char* path = "checkpoint_hash_test.model";
    const ModelConfig config = { 260, 8, 1, 2, 16, 32 };
    Arena model_arena, optimizer_arena;
    CHECK(arena_init(&model_arena, 1024u * 1024u));
    CHECK(arena_init(&optimizer_arena, 1024u * 1024u));

    Model model;
    CHECK(model_init_config(&model, &model_arena, 1234, &config));
    Optim opt;
    CHECK(optim_init(&opt, &model, &optimizer_arena,
        0.001f, 0.9f, 0.999f, 1e-8f, 0.01f));
    opt.schedule_base_lr = opt.lr;
    opt.schedule_total_steps = 100;
    opt.schedule_warmup_steps = 10;
    opt.schedule_type = 0;

    Rng rng;
    rng_seed(&rng, 5678);

    /* Zero is a valid 64-bit tokenizer fingerprint, not an invalid state. */
    CHECK(checkpoint_save(path, &model, &opt, &rng, UINT64_C(0)));
    CHECK(checkpoint_load(path, &model, &opt, &rng, UINT64_C(0)));
    CHECK(!checkpoint_load(path, &model, &opt, &rng, UINT64_C(1)));

    remove(path);
    arena_free(&optimizer_arena);
    arena_free(&model_arena);
    puts("Zero tokenizer fingerprint checkpoint round-trip passed.");
    return 0;
}
