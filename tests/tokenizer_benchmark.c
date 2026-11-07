#include "tokenizer.h"
#include "util.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static size_t parse_size(const char* value, const char* name) {
    char* end = NULL;
    unsigned long long parsed = strtoull(value, &end, 10);
    if (!value[0] || !end || *end || parsed == 0 || parsed > SIZE_MAX) {
        fprintf(stderr, "Invalid %s: %s\n", name, value);
        exit(2);
    }
    return (size_t)parsed;
}


typedef bool (*TokenizerTrainer)(Tokenizer*, Arena*,
    const uint8_t*, size_t, int);

static int run_benchmark_case(const char* label, TokenizerTrainer trainer,
    Tokenizer* tokenizer, Arena* arena, const uint8_t* sample,
    size_t sample_len, size_t vocab_size, size_t encode_repeats,
    uint16_t* encoded) {
    arena->used = 0;
    arena->peak = 0;

    const double train_start = now_seconds();
    const bool trained = trainer(tokenizer, arena, sample, sample_len,
        (int)vocab_size);
    const double train_seconds = now_seconds() - train_start;
    if (!trained) {
        fprintf(stderr, "%s tokenizer training failed.\\n", label);
        return 1;
    }

    int encoded_count = -1;
    const double encode_start = now_seconds();
    for (size_t i = 0; i < encode_repeats; ++i) {
        encoded_count = tokenizer_encode(tokenizer, sample, sample_len,
            encoded, (int)sample_len);
        if (encoded_count < 0) {
            fprintf(stderr, "%s tokenizer encoding failed.\\n", label);
            return 1;
        }
    }
    const double encode_seconds = now_seconds() - encode_start;
    const double encode_avg = encode_seconds / (double)encode_repeats;

    printf("\\nTokenizer benchmark (%s)\\n", label);
    printf("Sample bytes: %zu\\n", sample_len);
    printf("Target vocabulary: %zu\\n", vocab_size);
    printf("Actual vocabulary: %d\\n", tokenizer->vocab_size);
    printf("Merge count: %d\\n", tokenizer->n_merges);
    printf("Tokenizer fingerprint: %016llx\\n",
        (unsigned long long)tokenizer_fingerprint(tokenizer));
    printf("Training seconds: %.6f\\n", train_seconds);
    printf("Training MiB/s: %.3f\\n",
        train_seconds > 0.0 ? ((double)sample_len / (1024.0 * 1024.0)) / train_seconds : 0.0);
    printf("Arena peak MiB: %.3f\\n", (double)arena->peak / (1024.0 * 1024.0));
    printf("Encoding repeats: %zu\\n", encode_repeats);
    printf("Average encode milliseconds: %.3f\\n", encode_avg * 1000.0);
    printf("Encoding MiB/s: %.3f\\n",
        encode_avg > 0.0 ? ((double)sample_len / (1024.0 * 1024.0)) / encode_avg : 0.0);
    printf("Encoded tokens: %d\\n", encoded_count);
    printf("Bytes per token: %.4f\\n",
        encoded_count > 0 ? (double)sample_len / (double)encoded_count : 0.0);
    return 0;
}


int main(int argc, char** argv) {
    if (argc < 2 || argc > 5) {
        fprintf(stderr,
            "Usage: %s corpus.txt [sample_bytes=65536] [vocab_size=1024] [encode_repeats=3]\n",
            argv[0]);
        return 2;
    }

    const char* corpus_path = argv[1];
    const size_t corpus_size = file_size(corpus_path);
    if (corpus_size == 0) {
        fprintf(stderr, "Corpus is missing or empty: %s\n", corpus_path);
        return 2;
    }

    size_t requested_bytes = argc > 2 ? parse_size(argv[2], "sample_bytes") : 65536;
    size_t sample_capacity = requested_bytes < corpus_size ? requested_bytes : corpus_size;
    size_t vocab_size = argc > 3 ? parse_size(argv[3], "vocab_size") : 1024;
    size_t encode_repeats = argc > 4 ? parse_size(argv[4], "encode_repeats") : 3;

    if (sample_capacity > INT32_MAX || vocab_size < 260 ||
        vocab_size > TOK_MAX_VOCAB || encode_repeats > 100000) {
        fprintf(stderr, "Benchmark arguments exceed supported limits.\n");
        return 2;
    }

    uint8_t* sample = (uint8_t*)malloc(sample_capacity);
    uint16_t* encoded = (uint16_t*)malloc(sample_capacity * sizeof(uint16_t));
    if (!sample || !encoded) {
        fprintf(stderr, "Unable to allocate benchmark buffers.\n");
        free(sample);
        free(encoded);
        return 1;
    }

    size_t sample_len = 0;
    if (!read_entire_file(corpus_path, sample, sample_capacity, &sample_len) ||
        sample_len == 0) {
        fprintf(stderr, "Unable to read corpus: %s\n", corpus_path);
        free(sample);
        free(encoded);
        return 1;
    }

    Arena arena;
    if (!arena_init(&arena, (size_t)512 * 1024 * 1024)) {
        fprintf(stderr, "Unable to reserve 512 MiB benchmark arena.\n");
        free(sample);
        free(encoded);
        return 1;
    }

    /* Tokenizer owns large fixed-size vocabulary/merge arrays. Keep it off the
       default Windows thread stack. The arena is reused between implementations. */
    static Tokenizer tokenizer;
    printf("Corpus: %s\\n", corpus_path);
    int result = run_benchmark_case("reference", tokenizer_train_reference,
        &tokenizer, &arena, sample, sample_len, vocab_size,
        encode_repeats, encoded);
    if (!result) {
        result = run_benchmark_case("incremental", tokenizer_train_incremental,
            &tokenizer, &arena, sample, sample_len, vocab_size,
            encode_repeats, encoded);
    }

    arena_free(&arena);
    if (result) {
        free(sample);
        free(encoded);
        return result;
    }

    free(sample);
    free(encoded);
    return 0;
}
