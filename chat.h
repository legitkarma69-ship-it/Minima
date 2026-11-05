#ifndef CHAT_H
#define CHAT_H

#include "generate.h"
#include "model.h"
#include "tokenizer.h"
#include "util.h"

#include <stdbool.h>
#include <stdint.h>

typedef struct {
    uint64_t seed;
    float temperature;
    float top_p;
    float repetition_penalty;
    int top_k;
    int max_new_tokens;
} ChatOptions;

/* Interactive console inference protocol used by the Win32 GUI. */
int chat_run_console(Model* model, Tokenizer* tok, KVCache* cache,
    const ChatOptions* options);

/* Deterministic generation and repetition diagnostics used by smoke tests. */
bool chat_run_generation_benchmark(Model* model, Tokenizer* tok,
    KVCache* cache);

#endif
