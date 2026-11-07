#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
    return 1; } } while (0)

static Arena reference_arena;
static Arena incremental_arena;
static Tokenizer reference_tokenizer;
static Tokenizer incremental_tokenizer;
static uint16_t reference_ids[65536];
static uint16_t incremental_ids[65536];

static int compare_trainers(const uint8_t* sample, size_t sample_len,
    int target_vocab) {
    reference_arena.used = 0;
    reference_arena.peak = 0;
    incremental_arena.used = 0;
    incremental_arena.peak = 0;

    if (!tokenizer_train_reference(&reference_tokenizer, &reference_arena,
            sample, sample_len, target_vocab)) {
        fprintf(stderr, "Reference trainer failed (bytes=%zu, vocab=%d).\n",
            sample_len, target_vocab);
        return 0;
    }
    if (!tokenizer_train_incremental(&incremental_tokenizer,
            &incremental_arena, sample, sample_len, target_vocab)) {
        fprintf(stderr, "Incremental trainer failed (bytes=%zu, vocab=%d).\n",
            sample_len, target_vocab);
        return 0;
    }

    const uint64_t reference_fingerprint =
        tokenizer_fingerprint(&reference_tokenizer);
    const uint64_t incremental_fingerprint =
        tokenizer_fingerprint(&incremental_tokenizer);
    if (reference_fingerprint != incremental_fingerprint) {
        fprintf(stderr,
            "Trainer mismatch (bytes=%zu, vocab=%d): ref=%016llx inc=%016llx "
            "ref_merges=%d inc_merges=%d.\n",
            sample_len, target_vocab,
            (unsigned long long)reference_fingerprint,
            (unsigned long long)incremental_fingerprint,
            reference_tokenizer.n_merges, incremental_tokenizer.n_merges);
        const int common = reference_tokenizer.n_merges <
            incremental_tokenizer.n_merges ? reference_tokenizer.n_merges :
            incremental_tokenizer.n_merges;
        for (int i = 0; i < common; i++) {
            const Merge a = reference_tokenizer.merges[i];
            const Merge b = incremental_tokenizer.merges[i];
            if (a.left != b.left || a.right != b.right) {
                fprintf(stderr, "First merge mismatch at %d: (%u,%u) vs (%u,%u).\n",
                    i, (unsigned)a.left, (unsigned)a.right,
                    (unsigned)b.left, (unsigned)b.right);
                break;
            }
        }
        return 0;
    }

    const int reference_count = tokenizer_encode_reference(
        &reference_tokenizer, sample, sample_len, reference_ids,
        (int)(sizeof(reference_ids) / sizeof(reference_ids[0])));
    const int hybrid_reference_count = tokenizer_encode(&reference_tokenizer,
        sample, sample_len, incremental_ids,
        (int)(sizeof(incremental_ids) / sizeof(incremental_ids[0])));
    if (reference_count != hybrid_reference_count || reference_count < 0 ||
        memcmp(reference_ids, incremental_ids,
            (size_t)reference_count * sizeof(uint16_t)) != 0) {
        fprintf(stderr, "Hybrid encoder mismatch against reference (bytes=%zu).\n",
            sample_len);
        return 0;
    }

    const int incremental_count = tokenizer_encode(&incremental_tokenizer,
        sample, sample_len, incremental_ids,
        (int)(sizeof(incremental_ids) / sizeof(incremental_ids[0])));
    if (reference_count != incremental_count ||
        reference_count < 0 ||
        memcmp(reference_ids, incremental_ids,
            (size_t)reference_count * sizeof(uint16_t)) != 0) {
        fprintf(stderr, "Encoder mismatch after equivalent training (bytes=%zu).\n",
            sample_len);
        return 0;
    }
    return 1;
}

static uint32_t random_state = 0x9e3779b9u;

static uint32_t next_random(void) {
    random_state ^= random_state << 13;
    random_state ^= random_state >> 17;
    random_state ^= random_state << 5;
    return random_state;
}

int main(void) {
    CHECK(arena_init(&reference_arena, 64u * 1024u * 1024u));
    CHECK(arena_init(&incremental_arena, 64u * 1024u * 1024u));

    /* Exhaust every sequence over a three-byte alphabet through length seven.
       This naturally covers overlapping pairs and frequent tie situations. */
    uint8_t sample[65536];
    size_t exhaustive_cases = 0;
    for (size_t length = 1; length <= 7; length++) {
        size_t combinations = 1;
        for (size_t i = 0; i < length; i++) combinations *= 3;
        for (size_t value = 0; value < combinations; value++) {
            size_t encoded = value;
            for (size_t i = 0; i < length; i++) {
                sample[i] = (uint8_t)('a' + (encoded % 3));
                encoded /= 3;
            }
            CHECK(compare_trainers(sample, length, 276));
            exhaustive_cases++;
        }
    }
    printf("Incremental trainer: %zu exhaustive cases passed.\n",
        exhaustive_cases);

    /* A target vocabulary is a ceiling: unique adjacent pairs cannot merge. */
    {
        static const uint8_t all_unique[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
        CHECK(compare_trainers(all_unique, sizeof(all_unique), 300));
        CHECK(reference_tokenizer.vocab_size == 260);
        CHECK(reference_tokenizer.n_merges == 0);
        CHECK(incremental_tokenizer.vocab_size == 260);
        incremental_arena.used = 0;
        incremental_arena.peak = 0;
        static Tokenizer public_tokenizer;
        CHECK(tokenizer_train(&public_tokenizer, &incremental_arena,
            all_unique, sizeof(all_unique), 300));
        CHECK(public_tokenizer.vocab_size == 260);
        CHECK(public_tokenizer.n_merges == 0);
    }

    /* Explicit overlap and token-length boundary cases. */
    {
        static const char* patterns[] = {
            "AAA", "AAAAA", "AAAAAAAAAAAAAAAA",
            "ABABABA", "ABABAABABAABABA",
            "abcabcabcabcabcabc", "aabbaabbaabbaabb",
            "mississippi mississippi"
        };
        for (size_t i = 0; i < sizeof(patterns) / sizeof(patterns[0]); i++)
            CHECK(compare_trainers((const uint8_t*)patterns[i],
                strlen(patterns[i]), 300));
        memset(sample, 'A', sizeof(sample));
        CHECK(compare_trainers(sample, sizeof(sample), 300));
        CHECK(reference_tokenizer.vocab_size < 300);
        CHECK(incremental_tokenizer.vocab_size == reference_tokenizer.vocab_size);
        /* Exactly 64 KiB crosses the production encoder's heap-path threshold. */
        CHECK(compare_trainers(sample, 64u * 1024u, 300));
        CHECK(reference_tokenizer.vocab_size < 300);
    }

    /* Deterministic randomized inputs with varied alphabets and lengths. */
    for (int test = 0; test < 200; test++) {
        const size_t length = 1 + (next_random() % 512);
        const uint32_t alphabet = 2 + (next_random() % 31);
        for (size_t i = 0; i < length; i++)
            sample[i] = (uint8_t)(next_random() % alphabet);
        CHECK(compare_trainers(sample, length, 300 + (int)(next_random() % 100)));
    }

    puts("Incremental trainer equivalence tests passed.");
    arena_free(&reference_arena);
    arena_free(&incremental_arena);
    return 0;
}
