#ifndef CORPUS_H
#define CORPUS_H

#include "arena.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint8_t* data;
    size_t length;
    size_t files_loaded;
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

// Loads files containing non-whitespace text and joins them with blank-line separators.
// Missing, empty, and whitespace-only files are skipped. Returns EMPTY when
// no usable text exists and ERROR on read/allocation/size-limit failures.
CorpusLoadResult corpus_load_files(Corpus* corpus, Arena* arena,
    const char* const* paths, size_t path_count, size_t max_bytes);

#endif
