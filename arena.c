#include "arena.h"
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

bool arena_init(Arena* a, size_t size) {
    if (!a || size == 0) return false;

    memset(a, 0, sizeof(*a));

#ifdef _WIN32
    SYSTEM_INFO info;
    GetSystemInfo(&info);
    if (info.dwPageSize == 0) return false;

    a->base = (uint8_t*)VirtualAlloc(NULL, size,
        MEM_RESERVE, PAGE_NOACCESS);
    if (!a->base) return false;

    a->page_size = (size_t)info.dwPageSize;
    a->committed = 0;
#else
    a->base = (uint8_t*)malloc(size);
    if (!a->base) return false;
    a->page_size = 4096;
    a->committed = size;
#endif

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

    memset(a, 0, sizeof(*a));
}

void* arena_alloc(Arena* a, size_t n) {
    if (!a || !a->base || n == 0) return NULL;

    const size_t align = 64;
    if (a->used > SIZE_MAX - (align - 1)) return NULL;
    const size_t offset = (a->used + (align - 1)) & ~(align - 1);
    if (offset > a->size || n > a->size - offset) return NULL;

    const size_t end = offset + n;

#ifdef _WIN32
    if (end > a->committed) {
        const size_t page = a->page_size;
        if (page == 0 || end > SIZE_MAX - (page - 1)) return NULL;
        const size_t commit_end = ((end + page - 1) / page) * page;
        if (commit_end > a->size) return NULL;

        const size_t commit_bytes = commit_end - a->committed;
        if (commit_bytes > 0) {
            void* committed = VirtualAlloc(a->base + a->committed,
                commit_bytes, MEM_COMMIT, PAGE_READWRITE);
            if (!committed) return NULL;
            a->committed = commit_end;
        }
    }
#endif

    void* ptr = a->base + offset;
    a->used = end;
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
    if (!a || a->used > a->size) return 0;
    return a->size - a->used;
}
