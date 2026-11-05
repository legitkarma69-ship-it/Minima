#include "corpus.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define CORPUS_MAX_PATHS 32

static bool is_text_whitespace(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' ||
        c == '\r' || c == '\v' || c == '\f';
}

// Empty files and files containing only whitespace (including Windows CRLF
// line endings or a UTF-8 BOM) are not usable training sources.
static bool file_has_text(FILE* f, bool* has_text) {
    if (!f || !has_text) return false;

    *has_text = false;

    uint8_t prefix[3];
    size_t prefix_len = fread(prefix, 1, sizeof(prefix), f);
    if (ferror(f)) return false;

    size_t start = (prefix_len == 3 &&
        prefix[0] == 0xEF && prefix[1] == 0xBB && prefix[2] == 0xBF)
        ? 3 : 0;

    for (size_t i = start; i < prefix_len; i++) {
        if (!is_text_whitespace(prefix[i])) *has_text = true;
    }

    uint8_t buffer[4096];
    size_t got;
    while ((got = fread(buffer, 1, sizeof(buffer), f)) > 0) {
        for (size_t i = 0; i < got; i++) {
            if (!is_text_whitespace(buffer[i])) *has_text = true;
        }
    }

    return !ferror(f);
}

typedef struct {
    const char* path;
    size_t size;
} CorpusFile;

CorpusLoadResult corpus_load_files(Corpus* corpus, Arena* arena,
    const char* const* paths, size_t path_count, size_t max_bytes) {

    if (!corpus || !arena || !paths || path_count == 0 ||
        path_count > CORPUS_MAX_PATHS || max_bytes == 0)
        return CORPUS_LOAD_ERROR;

    memset(corpus, 0, sizeof(*corpus));

    CorpusFile files[CORPUS_MAX_PATHS];
    size_t file_count = 0;
    size_t total = 0;

    // First pass: determine which files exist and their sizes.
    for (size_t i = 0; i < path_count; i++) {
        if (!paths[i] || !paths[i][0]) continue;

        FILE* f = fopen(paths[i], "rb");
        if (!f) continue;

        if (fseek(f, 0, SEEK_END) != 0) {
            fclose(f);
            return CORPUS_LOAD_ERROR;
        }

        long end = ftell(f);
        if (end < 0 || fseek(f, 0, SEEK_SET) != 0) {
            fclose(f);
            return CORPUS_LOAD_ERROR;
        }
        bool has_text = false;
        if (end > 0 && !file_has_text(f, &has_text)) {
            fclose(f);
            return CORPUS_LOAD_ERROR;
        }
        fclose(f);

        if (end == 0 || !has_text) continue;
        size_t bytes = (size_t)end;
        if (bytes > max_bytes || total > max_bytes - bytes)
            return CORPUS_LOAD_ERROR;

        files[file_count].path = paths[i];
        files[file_count].size = bytes;
        file_count++;
        total += bytes;
    }

    if (file_count == 0) return CORPUS_LOAD_EMPTY;

    // Reserve two separator bytes between sources, plus a NUL terminator.
    if (file_count > 1) {
        size_t separators = (file_count - 1) * 2;
        if (separators > max_bytes || total > max_bytes - separators)
            return CORPUS_LOAD_ERROR;
        total += separators;
    }

    uint8_t* data = (uint8_t*)arena_alloc(arena, total + 1);
    if (!data) return CORPUS_LOAD_ERROR;

    size_t written = 0;
    size_t loaded = 0;

    // Second pass: read exact file lengths. UTF-8 BOMs are stripped.
    for (size_t i = 0; i < file_count; i++) {
        FILE* f = fopen(files[i].path, "rb");
        if (!f) return CORPUS_LOAD_ERROR;

        if (loaded > 0) {
            data[written++] = '\n';
            data[written++] = '\n';
        }

        size_t start = written;
        size_t got = fread(data + written, 1, files[i].size, f);
        int read_error = ferror(f);
        fclose(f);

        if (got != files[i].size || read_error) return CORPUS_LOAD_ERROR;

        if (got >= 3 && data[start] == 0xEF &&
            data[start + 1] == 0xBB && data[start + 2] == 0xBF) {
            memmove(data + start, data + start + 3, got - 3);
            got -= 3;
        }

        written += got;
        loaded++;

        printf("  loaded: %s (%zu bytes)\n", files[i].path, got);
    }

    if (written == 0) return CORPUS_LOAD_EMPTY;
    data[written] = 0;

    corpus->data = data;
    corpus->length = written;
    corpus->files_loaded = loaded;
    return CORPUS_LOAD_OK;
}
