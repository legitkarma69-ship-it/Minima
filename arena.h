#ifndef ARENA_H
#define ARENA_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct {
    size_t limit;      // maximum aggregate committed arena bytes; 0 = unlimited
    size_t committed;  // current committed bytes across attached arenas
    size_t peak;       // high-water mark across attached arenas
    size_t failed;     // number of allocations rejected by the budget
} ArenaBudget;

typedef struct {
    uint8_t* base;      // start of the block
    size_t   size;      // total bytes
    size_t   used;      // bytes currently allocated
    size_t   peak;      // high-water mark (for diagnostics)
    size_t   committed; // bytes backed by physical/commit memory
    size_t   page_size; // OS page size used for incremental commits
    ArenaBudget* budget; // optional shared committed-memory budget
} Arena;

// Create an arena of `size` bytes. On Windows, reserve address space and commit
// pages on demand as allocations grow. Other platforms allocate the full block.
// Returns false on failure (out of memory or address-space reservation failure).
bool arena_init(Arena* a, size_t size);

// Shared budget for committed pages across multiple arenas. A zero limit is unlimited.
void arena_budget_init(ArenaBudget* budget, size_t limit);
bool arena_init_with_budget(Arena* a, size_t size, ArenaBudget* budget);

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