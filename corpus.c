#include "corpus.h"

#include <stdio.h>
#include <errno.h>
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


typedef struct {
    const char* path;
    size_t usable_bytes;
    size_t quota;
    size_t sampled;
} CorpusSampleFile;

/* Build a representative sample without materializing the complete corpus.
   The budget is shared evenly across sources; undersized sources return their
   unused share to the remaining sources instead of letting a giant file
   consume the entire tokenizer sample. */
CorpusLoadResult corpus_sample_files(Arena* arena,
    const char* const* paths, size_t path_count, size_t sample_cap,
    bool spread, uint8_t** sample_out, size_t* sample_len_out,
    size_t* files_sampled_out, size_t* source_bytes_out) {
    if (!arena || !paths || path_count == 0 ||
        path_count > CORPUS_MAX_PATHS || sample_cap == 0 ||
        !sample_out || !sample_len_out)
        return CORPUS_LOAD_ERROR;

    *sample_out = NULL;
    *sample_len_out = 0;
    if (files_sampled_out) *files_sampled_out = 0;
    if (source_bytes_out) *source_bytes_out = 0;

    CorpusSampleFile files[CORPUS_MAX_PATHS];
    memset(files, 0, sizeof(files));
    size_t file_count = 0;
    size_t source_total = 0;

    for (size_t i = 0; i < path_count; i++) {
        if (!paths[i] || !paths[i][0]) continue;
        FILE* f = fopen(paths[i], "rb");
        if (!f) {
            if (errno == ENOENT) continue;
            return CORPUS_LOAD_ERROR;
        }
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
        if (fseek(f, 0, SEEK_SET) != 0) {
            fclose(f);
            return CORPUS_LOAD_ERROR;
        }
        uint8_t prefix[3] = {0, 0, 0};
        const size_t prefix_len = fread(prefix, 1, sizeof(prefix), f);
        if (ferror(f)) {
            fclose(f);
            return CORPUS_LOAD_ERROR;
        }
        fclose(f);

        const size_t bom = prefix_len == 3 &&
            prefix[0] == 0xEF && prefix[1] == 0xBB && prefix[2] == 0xBF
            ? 3u : 0u;
        if (end == 0 || !has_text || (size_t)end <= bom) continue;
        const size_t usable = (size_t)end - bom;
        if (source_total > SIZE_MAX - usable) return CORPUS_LOAD_ERROR;
        source_total += usable;
        files[file_count].path = paths[i];
        files[file_count].usable_bytes = usable;
        file_count++;
    }

    if (file_count == 0) return CORPUS_LOAD_EMPTY;
    if (sample_cap > SIZE_MAX - 2 * (file_count - 1) - 1)
        return CORPUS_LOAD_ERROR;

    /* Equal source weighting is intentional: a reference encyclopedia should
       enrich the tokenizer, not erase the vocabulary patterns in conversations
       and fiction simply because it is much larger. Small sources saturate and
       their unused budget is redistributed on the next pass. */
    size_t remaining = sample_cap;
    bool active[CORPUS_MAX_PATHS];
    for (size_t i = 0; i < file_count; i++) active[i] = true;
    while (remaining > 0) {
        size_t active_count = 0;
        for (size_t i = 0; i < file_count; i++)
            if (active[i] && files[i].quota < files[i].usable_bytes)
                active_count++;
        if (active_count == 0) break;

        const size_t round_budget = remaining;
        const size_t base_share = round_budget / active_count;
        size_t extra = round_budget % active_count;
        bool allocated = false;
        for (size_t i = 0; i < file_count; i++) {
            if (!active[i] || files[i].quota >= files[i].usable_bytes)
                continue;
            size_t share = base_share + (extra ? 1u : 0u);
            if (extra) extra--;
            if (share == 0) continue;
            const size_t available =
                files[i].usable_bytes - files[i].quota;
            const size_t take = share < available ? share : available;
            files[i].quota += take;
            remaining -= take;
            allocated = allocated || take > 0;
            if (files[i].quota == files[i].usable_bytes)
                active[i] = false;
        }
        if (!allocated) break;
    }

    size_t actual_bytes = 0;
    for (size_t i = 0; i < file_count; i++) {
        if (actual_bytes > SIZE_MAX - files[i].quota)
            return CORPUS_LOAD_ERROR;
        actual_bytes += files[i].quota;
    }
    if (actual_bytes == 0) return CORPUS_LOAD_EMPTY;

    const size_t allocation_size = actual_bytes + 2 * (file_count - 1) + 1;
    uint8_t* sample = (uint8_t*)arena_alloc(arena, allocation_size);
    if (!sample) return CORPUS_LOAD_ERROR;

    size_t written = 0;
    size_t sampled_files = 0;
    for (size_t i = 0; i < file_count; i++) {
        const size_t quota = files[i].quota;
        if (quota == 0) continue;
        FILE* f = fopen(files[i].path, "rb");
        if (!f) return CORPUS_LOAD_ERROR;
        uint8_t prefix[3] = {0, 0, 0};
        const size_t prefix_len = fread(prefix, 1, sizeof(prefix), f);
        if (ferror(f)) {
            fclose(f);
            return CORPUS_LOAD_ERROR;
        }
        const size_t bom = prefix_len == 3 &&
            prefix[0] == 0xEF && prefix[1] == 0xBB && prefix[2] == 0xBF
            ? 3u : 0u;

        if (sampled_files > 0) {
            sample[written++] = '\n';
            sample[written++] = '\n';
        }

        const size_t chunks = spread && quota > 1
            ? (quota < 64u ? quota : 64u) : 1u;
        size_t remaining_file = quota;
        for (size_t chunk = 0; chunk < chunks; chunk++) {
            const size_t this_len = chunk + 1 == chunks
                ? remaining_file : quota / chunks;
            const size_t start = chunks == 1 ? 0 :
                ((files[i].usable_bytes - this_len) * chunk) / (chunks - 1);
            if (start > (size_t)LONG_MAX - bom ||
                fseek(f, (long)(bom + start), SEEK_SET) != 0) {
                fclose(f);
                return CORPUS_LOAD_ERROR;
            }
            const size_t got = fread(sample + written, 1, this_len, f);
            if (got != this_len || ferror(f)) {
                fclose(f);
                return CORPUS_LOAD_ERROR;
            }
            written += got;
            remaining_file -= got;
        }
        fclose(f);
        if (remaining_file != 0) return CORPUS_LOAD_ERROR;
        files[i].sampled = quota;
        sampled_files++;
        printf("  tokenizer source: %s (%zu / %zu bytes)\n",
            files[i].path, quota, files[i].usable_bytes);
    }

    sample[written] = 0;
    *sample_out = sample;
    *sample_len_out = written;
    if (files_sampled_out) *files_sampled_out = sampled_files;
    if (source_bytes_out) *source_bytes_out = source_total;
    return CORPUS_LOAD_OK;
}

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
        if (!f) {
            // Missing optional paths are skipped so default path fallback works.
            // Other failures (permissions, descriptor exhaustion, etc.) are real
            // I/O errors and must not be misreported as an empty corpus.
            if (errno == ENOENT) continue;
            return CORPUS_LOAD_ERROR;
        }

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
