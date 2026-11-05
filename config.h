#ifndef CONFIG_H
#define CONFIG_H

// ---------- Model hyper-parameters ----------
// Small defaults for the initial smoke test.
// Increase these only after the training and inference paths are validated.

#define MODEL_VOCAB_SIZE      2048
#define MODEL_DIM             64
#define MODEL_N_LAYERS        2
#define MODEL_N_HEADS         2
#define MODEL_HEAD_DIM        (MODEL_DIM / MODEL_N_HEADS)
#define MODEL_FFN_DIM         128
#define MODEL_MAX_SEQ         64

// Derived
#define MODEL_N_PARAMS_EST    ( \
    (size_t)MODEL_VOCAB_SIZE * MODEL_DIM +                  /* tok_emb */ \
    (size_t)MODEL_MAX_SEQ * MODEL_DIM +                     /* pos_emb */ \
    (size_t)MODEL_N_LAYERS * ( \
        3 * MODEL_DIM * MODEL_DIM +                         /* qkv */ \
        MODEL_DIM * MODEL_DIM +                             /* attn out */ \
        MODEL_DIM * MODEL_FFN_DIM +                         /* ffn up */ \
        MODEL_DIM * MODEL_FFN_DIM +                         /* ffn gate */ \
        MODEL_FFN_DIM * MODEL_DIM +                         /* ffn down */ \
        MODEL_DIM + MODEL_DIM                               /* two RMSNorm scales */ \
    ) + \
    MODEL_DIM                                               /* final norm */ \
)

#endif
