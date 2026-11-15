#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#include "arena.h"
#include "generate.h"
#include "model.h"
#include "tokenizer.h"
#include <stdbool.h>
#include <stdint.h>

/*
 * Read-only model-quality diagnostics.
 *
 * Reports tokenizer usage/distribution, token-frequency-bucket loss, and
 * deterministic generation comparisons. No model weights, optimizer state,
 * RNG state, or training behavior are modified.
 */
bool diagnostics_run(Model* model, const Tokenizer* tok,
    Arena* permanent_arena, Arena* temp_arena,
    const uint16_t* all_ids, int train_tokens, int validation_tokens,
    int sequence_length);

#endif
