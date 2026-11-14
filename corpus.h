#ifndef CORPUS_H
#define CORPUS_H

#include "arena.h"
#include <stddef.h>
#include <stdint.h>

#define CORPUS_MAX_PATHS 32u
#define CORPUS_MAX_SAMPLE_SEGMENTS \
    (CORPUS_MAX_PATHS * 64u + CORPUS_MAX_PATHS - 1u)

typedef struct {
    size_t byte_start;
    size_t byte_length;
    const char* path;
} CorpusSource;

typedef struct {
    uint8_t* data;
    size_t length;
    size_t files_loaded;
    size_t source_count;
    CorpusSource sources[CORPUS_MAX_PATHS];
} Corpus;

typedef enum {
    CORPUS_LOAD_ERROR = -1,
    CORPUS_LOAD_EMPTY = 0,
    CORPUS_LOAD_OK = 1
} CorpusLoadResult;

// Samples each source independently without loading the complete corpus. The byte budget is
// shared evenly across sources, with unused quota redistributed to larger sources.
CorpusLoadResult corpus_sample_files(Arena* arena,
    const char* const* paths, size_t path_count, size_t sample_cap,
    bool spread, uint8_t** sample_out, size_t* sample_len_out,
    size_t* files_sampled_out, size_t* source_bytes_out);

/* Like corpus_sample_files, but also reports the exclusive end offset of
   each contiguous sampled piece. Consumers can keep distant spread-samples
   from becoming artificial neighbors while preserving the returned bytes. */
CorpusLoadResult corpus_sample_files_segmented(Arena* arena,
    const char* const* paths, size_t path_count, size_t sample_cap,
    bool spread, uint8_t** sample_out, size_t* sample_len_out,
    size_t* segment_ends_out, size_t segment_capacity,
    size_t* segment_count_out,
    size_t* files_sampled_out, size_t* source_bytes_out);

// Loads files containing non-whitespace text and joins them with blank-line separators.
// Missing, empty, and whitespace-only files are skipped. Returns EMPTY when
// no usable text exists and ERROR on read/allocation/size-limit failures.
CorpusLoadResult corpus_load_files(Corpus* corpus, Arena* arena,
    const char* const* paths, size_t path_count, size_t max_bytes);

#endif
