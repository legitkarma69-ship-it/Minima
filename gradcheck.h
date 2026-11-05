#ifndef GRADCHECK_H
#define GRADCHECK_H

#include "model.h"
#include "backward.h"
#include "arena.h"
#include <stdbool.h>
#include <stdint.h>

// Compare analytical gradients against central finite differences.
// Uses a small supplied batch and does not modify model weights.
bool gradient_check(Model* model, Gradients* grads, Arena* temp_arena,
    const uint16_t* tokens, const uint16_t* targets, int batch, int seq);

#endif
