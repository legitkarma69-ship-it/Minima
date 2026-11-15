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

    /* A one-token stop sequence is removed from the returned output. */
    const uint16_t stop_id = output[prompt_len];
    GenerationOptions stop_options = {
        .temperature = 1.0f, .top_p = 1.0f, .repetition_penalty = 1.0f,
        .min_p = 0.0f, .frequency_penalty = 0.0f, .presence_penalty = 0.0f,
        .top_k = 256, .max_new_tokens = 20, .min_new_tokens = 0,
        .no_repeat_ngram_size = 0, .stop_sequence_ids = &stop_id,
        .stop_sequence_len = 1
    };
    rng_seed(&rng, 5678);
    const int stopped_count = generate_with_options(&model, &tokenizer, &cache,
        prompt_ids, prompt_len, output, 32, &stop_options, &rng);
    CHECK(stopped_count == prompt_len);

    /* Min-p=1 keeps only the highest-probability token for a fixed model. */
    GenerationOptions minp_options = stop_options;
    minp_options.stop_sequence_ids = NULL;
    minp_options.stop_sequence_len = 0;
    minp_options.min_p = 1.0f;
    minp_options.max_new_tokens = 1;
    rng_seed(&rng, 111);
    const int minp_count_a = generate_with_options(&model, &tokenizer, &cache,
        prompt_ids, prompt_len, output, 32, &minp_options, &rng);
    CHECK(minp_count_a == prompt_len + 1);
    const uint16_t minp_token = output[prompt_len];
    rng_seed(&rng, 999);
    const int minp_count_b = generate_with_options(&model, &tokenizer, &cache,
        prompt_ids, prompt_len, output, 32, &minp_options, &rng);
    CHECK(minp_count_b == prompt_len + 1);
    CHECK(output[prompt_len] == minp_token);

    /* No-repeat unigram filtering must avoid every token in the prompt. */
    GenerationOptions ngram_options = minp_options;
    ngram_options.min_p = 0.0f;
    ngram_options.no_repeat_ngram_size = 1;
    ngram_options.max_new_tokens = 8;
    rng_seed(&rng, 2468);
    const int ngram_count = generate_with_options(&model, &tokenizer, &cache,
        prompt_ids, prompt_len, output, 32, &ngram_options, &rng);
    CHECK(ngram_count == prompt_len + 8);
    for (int i = prompt_len; i < ngram_count; ++i) {
        for (int j = 0; j < prompt_len; ++j)
            CHECK(output[i] != prompt_ids[j]);
        for (int j = prompt_len; j < i; ++j)
            CHECK(output[i] != output[j]);
    }

    arena_free(&cache_arena);
    arena_free(&tokenizer_arena);
    arena_free(&model_arena);
    puts("Reserved-token generation tests passed.");
    return 0;
}
