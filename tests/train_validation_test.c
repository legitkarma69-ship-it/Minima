#include "train.h"

#include <stdio.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); \
    return 1; } } while (0)

#define CHECK_INVALID(m, o, g, a, ids, tr, va, r, opts) do { \
    const char* reason = NULL; \
    CHECK(!train_run_validate_inputs((m), (o), (g), (a), (ids), \
        (tr), (va), (r), (opts), &reason)); \
    CHECK(reason != NULL && reason[0] != 0); \
} while (0)

int main(void) {
    static float model_storage[4096];
    static float gradient_storage[4096];
    static float first_moment[4096];
    static float second_moment[4096];
    static uint8_t temp_storage[4096];
    static uint16_t token_ids[128];

    Model model;
    memset(&model, 0, sizeof(model));
    model.vocab_size = 260;
    model.dim = 8;
    model.n_layers = 1;
    model.n_heads = 2;
    model.head_dim = 4;
    model.ffn_dim = 16;
    model.max_seq = 32;
    model.tok_emb = model_storage;
    model.pos_emb = model_storage;
    model.attn_qkv = model_storage;
    model.attn_proj = model_storage;
    model.ffn_up = model_storage;
    model.ffn_gate = model_storage;
    model.ffn_down = model_storage;
    model.rms1_scale = model_storage;
    model.rms2_scale = model_storage;
    model.rms_final_scale = model_storage;
    const size_t expected_params =
        260u * 8u + 32u * 8u + 3u * 8u * 8u + 8u * 8u +
        2u * 16u * 8u + 8u * 16u + 2u * 8u + 8u;
    model.weight_bytes = expected_params * sizeof(float);

    Gradients grads;
    memset(&grads, 0, sizeof(grads));
    grads.vocab_size = model.vocab_size;
    grads.dim = model.dim;
    grads.n_layers = model.n_layers;
    grads.n_heads = model.n_heads;
    grads.head_dim = model.head_dim;
    grads.ffn_dim = model.ffn_dim;
    grads.max_seq = model.max_seq;
    grads.weight_bytes = model.weight_bytes;
    grads.tok_emb = gradient_storage;
    grads.pos_emb = gradient_storage;
    grads.attn_qkv = gradient_storage;
    grads.attn_proj = gradient_storage;
    grads.ffn_up = gradient_storage;
    grads.ffn_gate = gradient_storage;
    grads.ffn_down = gradient_storage;
    grads.rms1_scale = gradient_storage;
    grads.rms2_scale = gradient_storage;
    grads.rms_final_scale = gradient_storage;

    Optim opt;
    memset(&opt, 0, sizeof(opt));
    opt.lr = 0.001f;
    opt.beta1 = 0.9f;
    opt.beta2 = 0.999f;
    opt.eps = 1e-8f;
    opt.weight_decay = 0.01f;
    opt.step = 0;
    opt.schedule_base_lr = 0.001f;
    opt.schedule_total_steps = 100;
    opt.schedule_warmup_steps = 10;
    opt.schedule_type = 0;
    opt.n_params = expected_params;
    opt.m_tok_emb = opt.m_pos_emb = opt.m_attn_qkv = opt.m_attn_proj =
        opt.m_ffn_up = opt.m_ffn_gate = opt.m_ffn_down =
        opt.m_rms1_scale = opt.m_rms2_scale = opt.m_rms_final_scale =
        first_moment;
    opt.v_tok_emb = opt.v_pos_emb = opt.v_attn_qkv = opt.v_attn_proj =
        opt.v_ffn_up = opt.v_ffn_gate = opt.v_ffn_down =
        opt.v_rms1_scale = opt.v_rms2_scale = opt.v_rms_final_scale =
        second_moment;

    Arena temp;
    memset(&temp, 0, sizeof(temp));
    temp.base = temp_storage;
    temp.size = sizeof(temp_storage);

    Rng rng;
    rng_seed(&rng, 1234);
    TrainRunOptions options;
    memset(&options, 0, sizeof(options));
    options.steps = 10;
    options.log_interval = 1;
    options.validation_batches = 1;
    options.validation_interval = 1;
    options.checkpoint_interval = 1;
    options.batch_size = 1;
    options.sequence_length = 8;
    options.metrics_path = "train_validation_test.csv";

    const char* reason = NULL;
    CHECK(train_run_validate_inputs(&model, &opt, &grads, &temp,
        token_ids, 64, 32, &rng, options, &reason));
    CHECK(reason == NULL);

    TrainRunOptions bad = options;
    bad.log_interval = 0;
    CHECK_INVALID(&model, &opt, &grads, &temp, token_ids, 64, 32, &rng, bad);
    bad = options; bad.validation_interval = 0;
    CHECK_INVALID(&model, &opt, &grads, &temp, token_ids, 64, 32, &rng, bad);
    bad = options; bad.checkpoint_interval = 0;
    CHECK_INVALID(&model, &opt, &grads, &temp, token_ids, 64, 32, &rng, bad);
    bad = options; bad.validation_batches = 0;
    CHECK_INVALID(&model, &opt, &grads, &temp, token_ids, 64, 32, &rng, bad);
    bad = options; bad.batch_size = 0;
    CHECK_INVALID(&model, &opt, &grads, &temp, token_ids, 64, 32, &rng, bad);
    bad = options; bad.sequence_length = 33;
    CHECK_INVALID(&model, &opt, &grads, &temp, token_ids, 64, 32, &rng, bad);
    bad = options; bad.steps = -1;
    CHECK_INVALID(&model, &opt, &grads, &temp, token_ids, 64, 32, &rng, bad);
    bad = options; bad.metrics_path = NULL;
    CHECK_INVALID(&model, &opt, &grads, &temp, token_ids, 64, 32, &rng, bad);

    CHECK_INVALID(&model, &opt, &grads, &temp, token_ids, 8, 32, &rng, options);
    CHECK_INVALID(&model, &opt, &grads, &temp, token_ids, 64, 8, &rng, options);
    CHECK_INVALID(&model, &opt, &grads, &temp, NULL, 64, 32, &rng, options);
    CHECK_INVALID(&model, &opt, &grads, &temp, token_ids, 64, 32, NULL, options);

    temp.used = temp.size + 1;
    CHECK_INVALID(&model, &opt, &grads, &temp, token_ids, 64, 32, &rng, options);
    temp.used = 0;

    Optim bad_opt = opt;
    bad_opt.schedule_type = 2;
    CHECK_INVALID(&model, &bad_opt, &grads, &temp, token_ids, 64, 32, &rng, options);
    bad_opt = opt; bad_opt.schedule_total_steps = 0;
    CHECK_INVALID(&model, &bad_opt, &grads, &temp, token_ids, 64, 32, &rng, options);
    bad_opt = opt; bad_opt.step = 101;
    CHECK_INVALID(&model, &bad_opt, &grads, &temp, token_ids, 64, 32, &rng, options);
    bad_opt = opt; bad_opt.beta1 = 1.0f;
    CHECK_INVALID(&model, &bad_opt, &grads, &temp, token_ids, 64, 32, &rng, options);
    bad_opt = opt; bad_opt.v_ffn_down = NULL;
    CHECK_INVALID(&model, &bad_opt, &grads, &temp, token_ids, 64, 32, &rng, options);

    Gradients bad_grads = grads;
    bad_grads.dim++;
    CHECK_INVALID(&model, &opt, &bad_grads, &temp, token_ids, 64, 32, &rng, options);
    Model bad_model = model;
    bad_model.ffn_gate = NULL;
    CHECK_INVALID(&bad_model, &opt, &grads, &temp, token_ids, 64, 32, &rng, options);

    bad = options;
    opt.step = 95;
    CHECK_INVALID(&model, &opt, &grads, &temp, token_ids, 64, 32, &rng, bad);

    puts("train_run boundary validation tests passed.");
    return 0;
}
