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

// Loads files containing non-whitespace text and joins them with blank-line separators.
// Missing, empty, and whitespace-only files are skipped. Returns EMPTY when
// no usable text exists and ERROR on read/allocation/size-limit failures.
CorpusLoadResult corpus_load_files(Corpus* corpus, Arena* arena,
    const char* const* paths, size_t path_count, size_t max_bytes);

#endif
