#ifndef ARENA_H
#define ARENA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    uint8_t* base;      // start of the block
    size_t   size;      // total bytes
    size_t   used;      // bytes currently allocated
    size_t   peak;      // high-water mark (for diagnostics)
} Arena;

// Create an arena of exactly `size` bytes.
// Returns false on failure (out of memory).
bool arena_init(Arena* a, size_t size);

// Destroy and free the underlying memory.
void arena_free(Arena* a);

// Allocate `n` bytes, 64-byte aligned (good for AVX).
// Returns NULL only if the arena is exhausted.
void* arena_alloc(Arena* a, size_t n);

// Allocate and zero.
void* arena_alloc_zero(Arena* a, size_t n);

// Reset used pointer to 0. Peak is left alone.
void arena_reset(Arena* a);

// How many bytes are still free.
size_t arena_remaining(const Arena* a);

#endif