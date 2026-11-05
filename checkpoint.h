#ifndef CHECKPOINT_H
#define CHECKPOINT_H

#include "model.h"
#include "optim.h"
#include "tokenizer.h"
#include "util.h"
#include <stdbool.h>

// Saves model weights, complete AdamW state, RNG state, and tokenizer
// fingerprint in one versioned file. The tokenizer is stored separately.
bool checkpoint_save(const char* path, const Model* model, const Optim* opt,
    const Rng* rng, uint64_t tokenizer_hash);

// Loads into already initialized objects. Model vocabulary and architecture
// must match the checkpoint, as must the tokenizer fingerprint.
// Optimizer hyperparameters, moments, global step, and RNG state are restored.
bool checkpoint_load(const char* path, Model* model, Optim* opt,
    Rng* rng, uint64_t tokenizer_hash);

#endif
