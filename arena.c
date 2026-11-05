#include "arena.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

bool arena_init(Arena* a, size_t size) {
    if (!a || size == 0) return false;

#ifdef _WIN32
    // VirtualAlloc gives us large pages potential and exact control
    a->base = (uint8_t*)VirtualAlloc(NULL, size,
        MEM_COMMIT | MEM_RESERVE,
        PAGE_READWRITE);
#else
    a->base = (uint8_t*)malloc(size);
#endif

    if (!a->base) return false;

    a->size = size;
    a->used = 0;
    a->peak = 0;
    return true;
}

void arena_free(Arena* a) {
    if (!a || !a->base) return;

#ifdef _WIN32
    VirtualFree(a->base, 0, MEM_RELEASE);
#else
    free(a->base);
#endif

    a->base = NULL;
    a->size = a->used = a->peak = 0;
}

void* arena_alloc(Arena* a, size_t n) {
    if (!a || n == 0) return NULL;

    // 64-byte alignment for AVX2 / cache lines
    size_t align = 64;
    size_t offset = (a->used + (align - 1)) & ~(align - 1);

    if (offset + n > a->size) {
        // Exhausted – caller must handle
        return NULL;
    }

    void* ptr = a->base + offset;
    a->used = offset + n;
    if (a->used > a->peak) a->peak = a->used;
    return ptr;
}

void* arena_alloc_zero(Arena* a, size_t n) {
    void* p = arena_alloc(a, n);
    if (p) memset(p, 0, n);
    return p;
}

void arena_reset(Arena* a) {
    if (a) a->used = 0;
}

size_t arena_remaining(const Arena* a) {
    if (!a) return 0;
    return a->size - a->used;
}