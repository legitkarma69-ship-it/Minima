#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    static const char phrase[] =
        "The quick brown fox jumps over the lazy dog. "
        "A deterministic tokenizer must preserve every boundary. ";
    const size_t phrase_len = sizeof(phrase) - 1;
    const size_t sample_len = phrase_len * 128u;
    uint8_t* sample = (uint8_t*)malloc(sample_len);
    if (!sample) return 1;
    for (size_t i = 0; i < 128u; i++)
        memcpy(sample + i * phrase_len, phrase, phrase_len);

    Arena arena;
    if (!arena_init(&arena, 64u * 1024u * 1024u)) {
        free(sample);
        return 1;
    }
    Tokenizer tok;
    if (!tokenizer_train(&tok, &arena, sample, sample_len, 270) ||
        tok.vocab_size != 270) {
        fprintf(stderr, "could not train test tokenizer to 270 entries\n");
        arena_free(&arena);
        free(sample);
        return 1;
    }
    free(sample);

    const size_t input_len = 4u * 1024u * 1024u + 4099u;
    uint8_t* input = (uint8_t*)malloc(input_len);
    uint16_t* chunked = (uint16_t*)malloc(input_len * sizeof(uint16_t));
    uint16_t* reference = (uint16_t*)malloc(input_len * sizeof(uint16_t));
    if (!input || !chunked || !reference) {
        free(input); free(chunked); free(reference);
        arena_free(&arena);
        return 1;
    }
    for (size_t i = 0; i < input_len; i++)
        input[i] = (uint8_t)phrase[i % phrase_len];

    const int chunked_count = tokenizer_encode(&tok, input, input_len,
        chunked, (int)input_len);
    const int reference_count = tokenizer_encode_reference(&tok,
        input, input_len, reference, (int)input_len);
    const int failed = chunked_count <= 0 ||
        chunked_count != reference_count ||
        memcmp(chunked, reference,
            (size_t)reference_count * sizeof(uint16_t)) != 0;

    free(input); free(chunked); free(reference);
    arena_free(&arena);
    if (failed) {
        fprintf(stderr, "chunked encoder did not match reference encoding\n");
        return 1;
    }
    printf("Chunked large-input tokenizer equivalence: PASS\n");
    return 0;
}
