#include "gradcheck.h"
#include "forward.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    const char* name;
    float* weights;
    const float* gradients;
    size_t count;
} GradientCheckTensor;

static float evaluate_loss(Model* model, Activations* act,
    const uint16_t* tokens, const uint16_t* targets,
    float* dlogits, int batch, int seq) {
    float* logits = model_forward(model, act, tokens, batch, seq);
    if (!logits) return NAN;
    return loss_cross_entropy(logits, targets, dlogits,
        batch, seq, model->vocab_size);
}

static bool check_tensor(Model* model, Activations* act,
    const uint16_t* tokens, const uint16_t* targets, float* dlogits,
    int batch, int seq, const GradientCheckTensor* tensor) {

    size_t best_index = 0;
    float best_magnitude = 0.0f;

    for (size_t i = 0; i < tensor->count; i++) {
        const float magnitude = fabsf(tensor->gradients[i]);
        if (magnitude > best_magnitude) {
            best_magnitude = magnitude;
            best_index = i;
        }
    }

    if (!isfinite(best_magnitude) || best_magnitude < 1.0e-8f) {
        printf("  %-16s SKIP (gradient too small)\n", tensor->name);
        return false;
    }

    const float original = tensor->weights[best_index];
    const float epsilon = 1.0e-2f;

    tensor->weights[best_index] = original + epsilon;
    const float loss_plus = evaluate_loss(model, act, tokens, targets,
        dlogits, batch, seq);

    tensor->weights[best_index] = original - epsilon;
    const float loss_minus = evaluate_loss(model, act, tokens, targets,
        dlogits, batch, seq);

    tensor->weights[best_index] = original;

    if (!isfinite(loss_plus) || !isfinite(loss_minus)) {
        printf("  %-16s FAIL (non-finite loss)\n", tensor->name);
        return false;
    }

    const float numerical = (loss_plus - loss_minus) / (2.0f * epsilon);
    const float analytical = tensor->gradients[best_index];
    const float denominator = fabsf(numerical) + fabsf(analytical) + 1.0e-6f;
    const float relative_error = fabsf(numerical - analytical) / denominator;

    const bool passed = isfinite(relative_error) && relative_error <= 0.08f;
    printf("  %-16s %s (analytic=% .6e numeric=% .6e rel.err=%.3e)\n",
        tensor->name, passed ? "PASS" : "FAIL",
        (double)analytical, (double)numerical, (double)relative_error);

    return passed;
}

bool gradient_check(Model* model, Gradients* grads, Arena* temp_arena,
    const uint16_t* tokens, const uint16_t* targets, int batch, int seq) {

    if (!model || !grads || !temp_arena || !tokens || !targets ||
        batch <= 0 || seq <= 0 || seq > model->max_seq || seq > 512)
        return false;

    arena_reset(temp_arena);

    Activations act;
    if (!activations_init(&act, temp_arena, batch, seq, model))
        return false;

    const size_t logit_count =
        (size_t)batch * seq * (size_t)model->vocab_size;
    float* dlogits = arena_alloc(temp_arena, logit_count * sizeof(float));
    if (!dlogits) return false;

    const float base_loss = evaluate_loss(model, &act, tokens, targets,
        dlogits, batch, seq);
    if (!isfinite(base_loss)) return false;

    gradients_zero(grads, model);
    model_backward(model, grads, &act, dlogits, tokens, batch, seq);

    const size_t layers = (size_t)model->n_layers;
    const size_t dim = (size_t)model->dim;
    const size_t vocab = (size_t)model->vocab_size;
    const size_t ffn = (size_t)model->ffn_dim;

    const GradientCheckTensor tensors[] = {
        { "tok_emb", model->tok_emb, grads->tok_emb, vocab * dim },
        { "pos_emb", model->pos_emb, grads->pos_emb,
            (size_t)model->max_seq * dim },
        { "attn_qkv", model->attn_qkv, grads->attn_qkv,
            layers * 3 * dim * dim },
        { "attn_proj", model->attn_proj, grads->attn_proj,
            layers * dim * dim },
        { "ffn_up", model->ffn_up, grads->ffn_up,
            layers * ffn * dim },
        { "ffn_gate", model->ffn_gate, grads->ffn_gate,
            layers * ffn * dim },
        { "ffn_down", model->ffn_down, grads->ffn_down,
            layers * dim * ffn },
        { "rms1_scale", model->rms1_scale, grads->rms1_scale,
            layers * dim },
        { "rms2_scale", model->rms2_scale, grads->rms2_scale,
            layers * dim },
        { "rms_final_scale", model->rms_final_scale,
            grads->rms_final_scale, dim }
    };

    const size_t tensor_count = sizeof(tensors) / sizeof(tensors[0]);
    size_t passed = 0;

    printf("\nProjectMinima numerical gradient check\n");
    printf("  baseline loss = %.6f, batch = %d, seq = %d\n",
        (double)base_loss, batch, seq);

    for (size_t i = 0; i < tensor_count; i++) {
        if (check_tensor(model, &act, tokens, targets, dlogits,
            batch, seq, &tensors[i]))
            passed++;
    }

    printf("Gradient checks passed: %zu / %zu\n\n", passed, tensor_count);
    return passed == tensor_count;
}
