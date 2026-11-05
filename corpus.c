#include "corpus.h"

#include <stdio.h>
#include <string.h>
#include <stdint.h>

#define CORPUS_MAX_PATHS 32

typedef struct {
    const char* path;
    size_t size;
} CorpusFile;

bool corpus_load_files(Corpus* corpus, Arena* arena,
    const char* const* paths, size_t path_count, size_t max_bytes) {

    if (!corpus || !arena || !paths || path_count == 0 ||
        path_count > CORPUS_MAX_PATHS || max_bytes == 0)
        return false;

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
            return false;
        }

        long end = ftell(f);
        if (end < 0 || fseek(f, 0, SEEK_SET) != 0) {
            fclose(f);
            return false;
        }
        fclose(f);

        if (end == 0) continue;
        size_t bytes = (size_t)end;
        if (bytes > max_bytes || total > max_bytes - bytes)
            return false;

        files[file_count].path = paths[i];
        files[file_count].size = bytes;
        file_count++;
        total += bytes;
    }

    if (file_count == 0) return false;

    // Reserve two separator bytes between sources, plus a NUL terminator.
    if (file_count > 1) {
        size_t separators = (file_count - 1) * 2;
        if (total > max_bytes - separators) return false;
        total += separators;
    }

    uint8_t* data = (uint8_t*)arena_alloc(arena, total + 1);
    if (!data) return false;

    size_t written = 0;
    size_t loaded = 0;

    // Second pass: read exact file lengths. UTF-8 BOMs are stripped.
    for (size_t i = 0; i < file_count; i++) {
        FILE* f = fopen(files[i].path, "rb");
        if (!f) return false;

        if (loaded > 0) {
            data[written++] = '\n';
            data[written++] = '\n';
        }

        size_t start = written;
        size_t got = fread(data + written, 1, files[i].size, f);
        int read_error = ferror(f);
        fclose(f);

        if (got != files[i].size || read_error) return false;

        if (got >= 3 && data[start] == 0xEF &&
            data[start + 1] == 0xBB && data[start + 2] == 0xBF) {
            memmove(data + start, data + start + 3, got - 3);
            got -= 3;
        }

        written += got;
        loaded++;

        printf("  loaded: %s (%zu bytes)\n", files[i].path, got);
    }

    if (written == 0) return false;
    data[written] = 0;

    corpus->data = data;
    corpus->length = written;
    corpus->files_loaded = loaded;
    return true;
}
