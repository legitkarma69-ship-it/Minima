#ifndef CONFIG_H
#define CONFIG_H

#include <stddef.h>

/* Conservative defaults. Runtime configurations are validated in model.c. */
#define MODEL_VOCAB_SIZE      2048
#define MODEL_DIM             64
#define MODEL_N_LAYERS        2
#define MODEL_N_HEADS         2
#define MODEL_FFN_DIM         128
#define MODEL_MAX_SEQ         64
#define MODEL_HEAD_DIM        (MODEL_DIM / MODEL_N_HEADS)

typedef struct {
    int vocab_size;
    int dim;
    int n_layers;
    int n_heads;
    int ffn_dim;
    int max_seq;
} ModelConfig;

#define MODEL_CONFIG_DEFAULT { \
    MODEL_VOCAB_SIZE, MODEL_DIM, MODEL_N_LAYERS, \
    MODEL_N_HEADS, MODEL_FFN_DIM, MODEL_MAX_SEQ \
}

#endif
