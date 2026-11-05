#ifndef BACKWARD_H
#define BACKWARD_H

#include "model.h"
#include "forward.h"
#include "arena.h"
#include <stdint.h>

// Gradient container – identical layout to Model
typedef Model Gradients;

// Allocate gradient buffers (call once after model_init)
bool gradients_init(Gradients* g, Arena* arena, const Model* model);

// Zero all gradients (call at the start of every training step)
void gradients_zero(Gradients* g, const Model* model);

// Cross-entropy loss + gradient of logits
// logits: [batch*seq, vocab]
// targets: [batch*seq]   (token ids)
// Returns average loss over the batch.
// Writes dlogits into the provided buffer (same shape as logits).
float loss_cross_entropy(const float* logits, const uint16_t* targets,
    float* dlogits, int batch, int seq, int vocab);

// Full backward pass.
// Must be called after model_forward with the same activations.
// Fills the Gradients structure.
void model_backward(Model* model, Gradients* grads, Activations* act,
    const float* dlogits, const uint16_t* tokens,
    int batch, int seq);

#endif