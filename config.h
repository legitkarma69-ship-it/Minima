#ifndef CONFIG_H
#define CONFIG_H

// ---------- Model hyper-parameters ----------
// Start conservative; you can raise these later.

#define MODEL_VOCAB_SIZE      8192
#define MODEL_DIM             384
#define MODEL_N_LAYERS        6
#define MODEL_N_HEADS         6
#define MODEL_HEAD_DIM        (MODEL_DIM / MODEL_N_HEADS)   // 64
#define MODEL_FFN_DIM         (MODEL_DIM * 8 / 3)           // ~1024 for SwiGLU-ish
#define MODEL_MAX_SEQ         256                           // training context

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