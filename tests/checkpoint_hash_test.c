#include "checkpoint.h"
#include "model.h"
#include "optim.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
    return 1; } } while (0)

static bool write_checkpoint_bytes(const char* path,
    const unsigned char* bytes, size_t length) {
    FILE* file = fopen(path, "wb");
    if (!file) return false;
    const bool ok = fwrite(bytes, 1, length, file) == length;
    return fclose(file) == 0 && ok;
}

static int first_byte_is(const char* path, unsigned char expected) {
    FILE* file = fopen(path, "rb");
    if (!file) return 0;
    const int value = fgetc(file);
    fclose(file);
    return value == expected;
}

static bool test_checkpoint_rotation(void) {
    const char* current = "checkpoint_rotation_test.model";
    const char* one = "checkpoint_rotation_test.model.prev.1";
    const char* two = "checkpoint_rotation_test.model.prev.2";
    const char* three = "checkpoint_rotation_test.model.prev.3";
    const char* stale = "checkpoint_rotation_test.model.prev.4";
    const char* temporary = "checkpoint_rotation_test.model.archive.tmp";
    remove(current); remove(one); remove(two); remove(three);
    remove(stale); remove(temporary);

    const unsigned char a[] = { 'A' }, b[] = { 'B' };
    const unsigned char c[] = { 'C' }, d[] = { 'D' };
    if (!write_checkpoint_bytes(current, a, sizeof(a)) ||
        !checkpoint_archive_previous(current, 3) ||
        !first_byte_is(one, 'A')) return false;
    if (!write_checkpoint_bytes(current, b, sizeof(b)) ||
        !checkpoint_archive_previous(current, 3) ||
        !first_byte_is(one, 'B') || !first_byte_is(two, 'A')) return false;
    if (!write_checkpoint_bytes(current, c, sizeof(c)) ||
        !checkpoint_archive_previous(current, 3) ||
        !first_byte_is(one, 'C') || !first_byte_is(two, 'B') ||
        !first_byte_is(three, 'A')) return false;
    if (!write_checkpoint_bytes(current, d, sizeof(d)) ||
        !checkpoint_archive_previous(current, 3) ||
        !first_byte_is(one, 'D') || !first_byte_is(two, 'C') ||
        !first_byte_is(three, 'B')) return false;

    /* Reducing retention removes stale slots; zero leaves archives untouched. */
    if (!checkpoint_archive_previous(current, 1) ||
        !first_byte_is(one, 'D') || first_byte_is(two, 'C')) return false;
    if (!write_checkpoint_bytes(current, a, sizeof(a)) ||
        !checkpoint_archive_previous(current, 0) ||
        !first_byte_is(one, 'D')) return false;

    remove(current); remove(one); remove(two); remove(three);
    remove(stale); remove(temporary);
    return true;
}

static bool expect_rejected_without_mutation(const char* path,
    const unsigned char* bytes, size_t length, uint64_t tokenizer_hash,
    Model* model, Optim* opt, Rng* rng) {
    if (!write_checkpoint_bytes(path, bytes, length)) return false;

    const float sentinel_weight = 123.25f;
    const float sentinel_lr = 0.0125f;
    const int64_t sentinel_step = 7;
    const uint64_t sentinel_rng = UINT64_C(0xabcdef0123456789);
    model->tok_emb[0] = sentinel_weight;
    opt->lr = sentinel_lr;
    opt->step = sentinel_step;
    rng->state = sentinel_rng;

    const bool rejected = !checkpoint_load(
        path, model, opt, rng, tokenizer_hash);
    return rejected && model->tok_emb[0] == sentinel_weight &&
        opt->lr == sentinel_lr && opt->step == sentinel_step &&
        rng->state == sentinel_rng;
}

int main(void) {
    const char* path = "checkpoint_hash_test.model";
    CHECK(test_checkpoint_rotation());
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

    /* Zero is a valid 64-bit tokenizer fingerprint, not an invalid state.
       Fresh-run optimizer hyperparameters must survive checkpoint restore. */
    opt.beta1 = 0.77f;
    opt.beta2 = 0.88f;
    opt.eps = 2e-7f;
    opt.weight_decay = 0.037f;
    CHECK(checkpoint_save(path, &model, &opt, &rng, UINT64_C(0)));
    opt.beta1 = 0.1f;
    opt.beta2 = 0.2f;
    opt.eps = 1e-4f;
    opt.weight_decay = 0.5f;
    CHECK(checkpoint_load(path, &model, &opt, &rng, UINT64_C(0)));
    CHECK(fabsf(opt.beta1 - 0.77f) < 1e-7f);
    CHECK(fabsf(opt.beta2 - 0.88f) < 1e-7f);
    CHECK(fabsf(opt.eps - 2e-7f) < 1e-12f);
    CHECK(fabsf(opt.weight_decay - 0.037f) < 1e-7f);
    CHECK(!checkpoint_load(path, &model, &opt, &rng, UINT64_C(1)));

    FILE* file = fopen(path, "rb");
    CHECK(file != NULL);
    CHECK(fseek(file, 0, SEEK_END) == 0);
    const long file_size = ftell(file);
    CHECK(file_size > 112);
    CHECK(fseek(file, 0, SEEK_SET) == 0);
    unsigned char* original = (unsigned char*)malloc((size_t)file_size);
    unsigned char* corrupted = (unsigned char*)malloc((size_t)file_size + 1);
    CHECK(original != NULL && corrupted != NULL);
    CHECK(fread(original, 1, (size_t)file_size, file) == (size_t)file_size);
    CHECK(fclose(file) == 0);

    /* Header offsets are part of checkpoint format version 3. */
    memcpy(corrupted, original, (size_t)file_size);
    corrupted[0] ^= 0x01; /* bad magic */
    CHECK(expect_rejected_without_mutation(path, corrupted,
        (size_t)file_size, UINT64_C(0), &model, &opt, &rng));

    memcpy(corrupted, original, (size_t)file_size);
    corrupted[8] ^= 0x01; /* unsupported version */
    CHECK(expect_rejected_without_mutation(path, corrupted,
        (size_t)file_size, UINT64_C(0), &model, &opt, &rng));

    memcpy(corrupted, original, (size_t)file_size);
    corrupted[12] ^= 0x01; /* incompatible vocabulary dimension */
    CHECK(expect_rejected_without_mutation(path, corrupted,
        (size_t)file_size, UINT64_C(0), &model, &opt, &rng));

    memcpy(corrupted, original, (size_t)file_size);
    corrupted[60] ^= 0x01; /* mismatched tokenizer fingerprint */
    CHECK(expect_rejected_without_mutation(path, corrupted,
        (size_t)file_size, UINT64_C(0), &model, &opt, &rng));

    memcpy(corrupted, original, (size_t)file_size);
    {
        const uint32_t quiet_nan = UINT32_C(0x7fc00000);
        memcpy(corrupted + 112, &quiet_nan, sizeof(quiet_nan));
    }
    CHECK(expect_rejected_without_mutation(path, corrupted,
        (size_t)file_size, UINT64_C(0), &model, &opt, &rng));

    CHECK(expect_rejected_without_mutation(path, original,
        (size_t)file_size - 1, UINT64_C(0), &model, &opt, &rng));

    memcpy(corrupted, original, (size_t)file_size);
    corrupted[file_size] = 0x5a; /* unexpected trailing data */
    CHECK(expect_rejected_without_mutation(path, corrupted,
        (size_t)file_size + 1, UINT64_C(0), &model, &opt, &rng));

    /* Restore and prove the valid zero-hash checkpoint still loads. */
    CHECK(write_checkpoint_bytes(path, original, (size_t)file_size));
    CHECK(checkpoint_load(path, &model, &opt, &rng, UINT64_C(0)));

    free(corrupted);
    free(original);
    remove(path);
    arena_free(&optimizer_arena);
    arena_free(&model_arena);
    puts("Checkpoint zero-hash and corruption validation tests passed.");
    return 0;
}
