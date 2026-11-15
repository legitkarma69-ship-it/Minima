#ifndef UTIL_H
#define UTIL_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

// Simple xorshift64* RNG – fast, no state external to the seed
typedef struct {
    uint64_t state;
} Rng;

void rng_seed(Rng* r, uint64_t seed);
uint64_t rng_u64(Rng* r);
float    rng_f32(Rng* r);          // [0, 1)
int      rng_int(Rng* r, int lo, int hi); // inclusive

// File helpers
bool  file_exists(const char* path);
size_t file_size(const char* path);
bool  read_entire_file(const char* path, void* buf, size_t buf_size, size_t* out_read);

// Timing (Windows QueryPerformanceCounter)
double now_seconds(void);

// Fatal error – prints and exits
void die(const char* fmt, ...);

#endif