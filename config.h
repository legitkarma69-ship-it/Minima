#ifndef CONFIG_H
#define CONFIG_H

#include <stddef.h>

/* Supported model envelope. Keep runtime validation and regression tests aligned. */
#define MODEL_MIN_VOCAB_SIZE 260
#define MODEL_MAX_VOCAB_SIZE 8192
#define MODEL_MAX_DIM 512
#define MODEL_MAX_LAYERS 16
#define MODEL_MAX_HEADS 32
#define MODEL_MAX_FFN_DIM 1024
#define MODEL_MIN_SEQ 32
#define MODEL_MAX_SEQ 512

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
