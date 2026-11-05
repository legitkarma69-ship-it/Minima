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

// Loads non-empty files and joins them with blank-line separators.
// Missing and empty files are skipped. Returns false if no usable text
// was found or if a read/allocation/size-limit error occurred.
bool corpus_load_files(Corpus* corpus, Arena* arena,
    const char* const* paths, size_t path_count, size_t max_bytes);

#endif
