#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
    return 1; } } while (0)

static int read_file(const char* path, unsigned char** data, size_t* size) {
    FILE* f = fopen(path, "rb");
    if (!f) return 0;
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 0; }
    long n = ftell(f);
    if (n < 0 || fseek(f, 0, SEEK_SET) != 0) { fclose(f); return 0; }
    *data = (unsigned char*)malloc((size_t)n);
    if (!*data) { fclose(f); return 0; }
    *size = (size_t)n;
    int ok = fread(*data, 1, *size, f) == *size;
    fclose(f);
    return ok;
}

static int write_file(const char* path, const unsigned char* data, size_t size) {
    FILE* f = fopen(path, "wb");
    if (!f) return 0;
    int ok = fwrite(data, 1, size, f) == size;
    if (fclose(f) != 0) ok = 0;
    return ok;
}

static size_t token_data_offset(const unsigned char* data, size_t size,
    int wanted_id) {
    size_t p = 12;
    for (int i = 0; i <= wanted_id; i++) {
        if (p + 2 > size) return 0;
        unsigned len = (unsigned)data[p] | ((unsigned)data[p + 1] << 8);
        p += 2;
        if (p + len > size) return 0;
        if (i == wanted_id) return p;
        p += len;
    }
    return 0;
}

int main(void) {
    Arena train_arena, load_arena, bad_arena;
    CHECK(arena_init(&train_arena, 64u * 1024u * 1024u));
    CHECK(arena_init(&load_arena, 16u * 1024u * 1024u));
    CHECK(arena_init(&bad_arena, 16u * 1024u * 1024u));
    puts("Tokenizer test: arenas initialized."); fflush(stdout);

    /* Every possible byte must round-trip, including NUL and high-bit bytes. */
    uint8_t all_bytes[256];
    for (int i = 0; i < 256; i++) all_bytes[i] = (uint8_t)i;
    Tokenizer byte_tok;
    CHECK(tokenizer_train(&byte_tok, &train_arena, all_bytes,
        sizeof(all_bytes), 260));
    uint16_t byte_ids[256];
    char decoded[257];
    int n = tokenizer_encode(&byte_tok, all_bytes, sizeof(all_bytes),
        byte_ids, 256);
    CHECK(n == 256);
    CHECK(tokenizer_decode(&byte_tok, byte_ids, n, decoded,
        (int)sizeof(decoded)) == 256);
    CHECK(memcmp(decoded, all_bytes, 256) == 0);

    CHECK(tokenizer_decode(&byte_tok, NULL, 0, decoded, 1) == 0);
    CHECK(decoded[0] == 0);
    CHECK(tokenizer_decode(NULL, byte_ids, 1, decoded, 257) == -1);
    CHECK(tokenizer_decode(&byte_tok, byte_ids, -1, decoded, 257) == -1);
    CHECK(tokenizer_decode(&byte_tok, byte_ids, 1, decoded, -1) == -1);
    CHECK(tokenizer_decode(&byte_tok, NULL, 1, decoded, 257) == -1);
    CHECK(tokenizer_decode(&byte_tok, byte_ids, 1, NULL, 257) == -1);
    CHECK(tokenizer_decode(&byte_tok, byte_ids, 1, decoded, 0) == -1);
    {
        uint16_t invalid_id = 65535;
        decoded[0] = 'X';
        CHECK(tokenizer_decode(&byte_tok, &invalid_id, 1, decoded, 257) == -1);
        CHECK(decoded[0] == 'X');
        CHECK(tokenizer_decode(&byte_tok, byte_ids, 1, decoded, 1) == -1);
        CHECK(decoded[0] == 'X');
    }

    puts("Tokenizer test: byte round-trip passed."); fflush(stdout);

    /* Repeated pairs produce merge rules; persistence preserves exact behavior. */
    {
        static const uint8_t repeated[] =
            "abababababababababababababababababababababababababababababababab";
        Tokenizer trained, loaded;
        fprintf(stderr, "Tokenizer test: starting repeated-pair training (arena used=%zu, remaining=%zu).\\n", train_arena.used, arena_remaining(&train_arena));
        fflush(stderr);
        const bool repeated_ok = tokenizer_train(&trained, &train_arena, repeated,
            sizeof(repeated) - 1, 270);
        fprintf(stderr, "Tokenizer test: repeated-pair training returned %d (vocab=%d, merges=%d, arena used=%zu).\\n",
            repeated_ok ? 1 : 0, trained.vocab_size, trained.n_merges, train_arena.used);
        fflush(stderr);
        CHECK(repeated_ok);
        CHECK(trained.n_merges > 0);
        CHECK(tokenizer_save(&trained, "tokenizer_test.bpe"));
        CHECK(tokenizer_load(&loaded, &load_arena, "tokenizer_test.bpe"));
        CHECK(tokenizer_fingerprint(&trained) == tokenizer_fingerprint(&loaded));

        uint16_t before[128], after[128];
        char text_before[129], text_after[129];
        int before_n = tokenizer_encode(&trained, repeated,
            sizeof(repeated) - 1, before, 128);
        int after_n = tokenizer_encode(&loaded, repeated,
            sizeof(repeated) - 1, after, 128);
        CHECK(before_n == after_n);
        CHECK(memcmp(before, after, (size_t)before_n * sizeof(uint16_t)) == 0);
        CHECK(tokenizer_decode(&loaded, after, after_n, text_after,
            (int)sizeof(text_after)) == (int)sizeof(repeated) - 1);
        CHECK(memcmp(text_after, repeated, sizeof(repeated) - 1) == 0);
        CHECK(tokenizer_decode(&loaded, after, after_n, text_before,
            (int)sizeof(repeated) - 1) == -1);

        /* Corrupt a byte-token payload without changing its length. */
        {
            unsigned char* bytes = NULL; size_t size = 0;
            CHECK(read_file("tokenizer_test.bpe", &bytes, &size));
            size_t off = token_data_offset(bytes, size, TOK_BYTE_START + 'a');
            CHECK(off != 0);
            bytes[off] ^= 1;
            CHECK(write_file("tokenizer_bad_byte.bpe", bytes, size));
            free(bytes);
            size_t used = bad_arena.used;
            CHECK(!tokenizer_load(&loaded, &bad_arena, "tokenizer_bad_byte.bpe"));
            CHECK(bad_arena.used == used);
        }

        /* Corrupt a merge result while preserving its stored length. */
        {
            unsigned char* bytes = NULL; size_t size = 0;
            CHECK(read_file("tokenizer_test.bpe", &bytes, &size));
            size_t off = token_data_offset(bytes, size, 260);
            CHECK(off != 0);
            bytes[off] ^= 1;
            CHECK(write_file("tokenizer_bad_merge.bpe", bytes, size));
            free(bytes);
            size_t used = bad_arena.used;
            CHECK(!tokenizer_load(&loaded, &bad_arena, "tokenizer_bad_merge.bpe"));
            CHECK(bad_arena.used == used);
        }

        /* An out-of-range merge child must be rejected before array access. */
        {
            unsigned char* bytes = NULL; size_t size = 0;
            CHECK(read_file("tokenizer_test.bpe", &bytes, &size));
            size_t p = 12;
            for (int i = 0; i < trained.vocab_size; i++) {
                unsigned len = (unsigned)bytes[p] | ((unsigned)bytes[p + 1] << 8);
                p += 2 + len;
            }
            CHECK(p + sizeof(Merge) <= size);
            bytes[p] = 0xff; bytes[p + 1] = 0xff;
            CHECK(write_file("tokenizer_bad_id.bpe", bytes, size));
            free(bytes);
            size_t used = bad_arena.used;
            CHECK(!tokenizer_load(&loaded, &bad_arena, "tokenizer_bad_id.bpe"));
            CHECK(bad_arena.used == used);
        }

        /* Truncation and unexpected trailing bytes are both invalid. */
        {
            unsigned char* bytes = NULL; size_t size = 0;
            CHECK(read_file("tokenizer_test.bpe", &bytes, &size));
            CHECK(size > 1);
            CHECK(write_file("tokenizer_truncated.bpe", bytes, size - 1));
            bytes = (unsigned char*)realloc(bytes, size + 1);
            CHECK(bytes != NULL);
            bytes[size] = 0;
            CHECK(write_file("tokenizer_trailing.bpe", bytes, size + 1));
            free(bytes);
            size_t used = bad_arena.used;
            CHECK(!tokenizer_load(&loaded, &bad_arena, "tokenizer_truncated.bpe"));
            CHECK(bad_arena.used == used);
            CHECK(!tokenizer_load(&loaded, &bad_arena, "tokenizer_trailing.bpe"));
            CHECK(bad_arena.used == used);
        }
    }

    puts("Tokenizer test: persistence and corruption checks passed."); fflush(stdout);

    remove("tokenizer_test.bpe");
    remove("tokenizer_test.bpe.tmp");
    remove("tokenizer_bad_byte.bpe");
    remove("tokenizer_bad_merge.bpe");
    remove("tokenizer_bad_id.bpe");
    remove("tokenizer_truncated.bpe");
    remove("tokenizer_trailing.bpe");
    arena_free(&train_arena);
    arena_free(&load_arena);
    arena_free(&bad_arena);
    puts("Tokenizer tests passed.");
    return 0;
}
