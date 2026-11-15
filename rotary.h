#ifndef ROTARY_H
#define ROTARY_H

#include <math.h>
#include <stddef.h>

/* Apply RoPE to the Q and K portions of one or more fused GQA rows.
   `inverse` applies the transpose rotation used in backpropagation. */
static inline void rotary_apply_qkv(float* qkv, int batch, int seq,
    int n_heads, int n_kv_heads, int head_dim, int qkv_dim,
    int start_position, int inverse) {
    const int pairs = head_dim / 2;
    if (pairs <= 0) return;

    for (int t = 0; t < seq; t++) {
        for (int pair = 0; pair < pairs; pair++) {
            const float exponent = (2.0f * (float)pair) / (float)head_dim;
            const float angle = (float)(start_position + t) *
                powf(10000.0f, -exponent);
            const float c = cosf(angle);
            const float s = inverse ? -sinf(angle) : sinf(angle);
            for (int b = 0; b < batch; b++) {
                float* row = qkv + ((size_t)b * seq + t) * qkv_dim;
                for (int h = 0; h < n_heads; h++) {
                    float* x = row + h * head_dim + pair * 2;
                    const float a = x[0], z = x[1];
                    x[0] = a * c - z * s;
                    x[1] = a * s + z * c;
                }
                for (int h = 0; h < n_kv_heads; h++) {
                    float* x = row + n_heads * head_dim +
                        h * head_dim + pair * 2;
                    const float a = x[0], z = x[1];
                    x[0] = a * c - z * s;
                    x[1] = a * s + z * c;
                }
            }
        }
    }
}

#endif
