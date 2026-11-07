#include "generate.h"

#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
    return 1; } } while (0)

int main(void) {
    static Tokenizer tokenizer;
    Arena model_arena, tokenizer_arena, cache_arena;
    CHECK(arena_init(&model_arena, 1024u * 1024u));
    CHECK(arena_init(&tokenizer_arena, 16u * 1024u * 1024u));
    CHECK(arena_init(&cache_arena, 1024u * 1024u));

    const ModelConfig config = { 260, 8, 1, 2, 16, 32 };
    Model model;
    CHECK(model_init_config(&model, &model_arena, 1234, &config));

    static const uint8_t corpus[] = "A short raw-byte corpus.";
    CHECK(tokenizer_train(&tokenizer, &tokenizer_arena,
        corpus, sizeof(corpus) - 1, 260));
    CHECK(tokenizer.vocab_size == 260);

    const uint8_t prompt[] = "Hi";
    uint16_t prompt_ids[sizeof(prompt) - 1];
    const int prompt_len = tokenizer_encode(&tokenizer, prompt,
        sizeof(prompt) - 1, prompt_ids, (int)(sizeof(prompt) - 1));
    CHECK(prompt_len == (int)(sizeof(prompt) - 1));
    for (int i = 0; i < prompt_len; i++)
        CHECK(prompt_ids[i] >= TOK_SPECIAL_COUNT);

    KVCache cache;
    CHECK(kvcache_init(&cache, &cache_arena, &model, 32));
    Rng rng;
    rng_seed(&rng, 5678);

    CHECK(generation_inspect_next_tokens(&model, &tokenizer, &cache,
        prompt_ids, prompt_len, 10));

    uint16_t output[32];
    const int output_count = generate_ex(&model, &tokenizer, &cache,
        prompt_ids, prompt_len, output, 32, 20,
        1.0f, 256, 1.0f, 1.0f, &rng);
    CHECK(output_count == prompt_len + 20);
    for (int i = prompt_len; i < output_count; i++) {
        CHECK(output[i] >= TOK_SPECIAL_COUNT);
        CHECK(output[i] < tokenizer.vocab_size);
    }

    arena_free(&cache_arena);
    arena_free(&tokenizer_arena);
    arena_free(&model_arena);
    puts("Reserved-token generation tests passed.");
    return 0;
}
