#include "corpus.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int write_file(const char* path, const char* data, size_t length) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    const int ok = fwrite(data, 1, length, f) == length;
    return fclose(f) == 0 && ok;
}

static size_t count_byte(const uint8_t* data, size_t length, uint8_t value) {
    size_t count = 0;
    for (size_t i = 0; i < length; i++)
        if (data[i] == value) count++;
    return count;
}

int main(void) {
    const char* a = "corpus_sample_a.tmp";
    const char* b = "corpus_sample_b.tmp";
    const char* c = "corpus_sample_c.tmp";
    char text_a[80], text_b[20], text_c[200];
    memset(text_a, 'A', sizeof(text_a));
    memset(text_b, 'B', sizeof(text_b));
    memset(text_c, 'C', sizeof(text_c));
    if (!write_file(a, text_a, sizeof(text_a)) ||
        !write_file(b, text_b, sizeof(text_b)) ||
        !write_file(c, text_c, sizeof(text_c))) {
        remove(a); remove(b); remove(c);
        fprintf(stderr, "could not create sampler fixtures\n");
        return 1;
    }

    Arena arena;
    if (!arena_init(&arena, 1024u * 1024u)) {
        remove(a); remove(b); remove(c);
        return 1;
    }

    const char* paths[] = { a, b, c, "corpus_sample_missing.tmp" };
    uint8_t* sample = NULL;
    size_t sample_len = 0, sampled_files = 0, source_bytes = 0;
    const CorpusLoadResult result = corpus_sample_files(&arena, paths, 4,
        30, true, &sample, &sample_len, &sampled_files, &source_bytes);
    int failed = result != CORPUS_LOAD_OK || !sample ||
        sampled_files != 3 || source_bytes != 300 ||
        count_byte(sample, sample_len, 'A') != 10 ||
        count_byte(sample, sample_len, 'B') != 10 ||
        count_byte(sample, sample_len, 'C') != 10 ||
        sample_len != 34;

    if (!failed) {
        arena_reset(&arena);
        sample = NULL;
        sample_len = 0;
        const CorpusLoadResult prefix_result = corpus_sample_files(&arena,
            paths, 4, 30, false, &sample, &sample_len,
            &sampled_files, &source_bytes);
        failed = prefix_result != CORPUS_LOAD_OK || !sample ||
            sample_len != 34 || sample[0] != 'A' ||
            sample[12] != 'B' || sample[24] != 'C';
    }

    arena_free(&arena);
    remove(a); remove(b); remove(c);
    if (failed) {
        fprintf(stderr, "source-balanced corpus sampler regression failed\n");
        return 1;
    }
    printf("Source-balanced streaming corpus sampler: PASS\n");
    return 0;
}
