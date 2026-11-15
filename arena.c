#include "arena.h"
#include <string.h>
#include <stdint.h>
#include <stdio.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

void arena_budget_init(ArenaBudget* budget, size_t limit) {
    if (!budget) return;
    memset(budget, 0, sizeof(*budget));
    budget->limit = limit;
}

bool arena_init_with_budget(Arena* a, size_t size, ArenaBudget* budget) {
    if (!a || size == 0) return false;

    memset(a, 0, sizeof(*a));

    SYSTEM_INFO info;
    GetSystemInfo(&info);
    if (info.dwPageSize == 0) return false;

    a->base = (uint8_t*)VirtualAlloc(NULL, size,
        MEM_RESERVE, PAGE_NOACCESS);
    if (!a->base) return false;

    a->page_size = (size_t)info.dwPageSize;
    a->committed = 0;
    a->budget = budget;

    a->size = size;
    a->used = 0;
    a->peak = 0;
    return true;
}

bool arena_init(Arena* a, size_t size) {
    return arena_init_with_budget(a, size, NULL);
}

void arena_free(Arena* a) {
    if (!a || !a->base) return;

    VirtualFree(a->base, 0, MEM_RELEASE);
    if (a->budget) {
        if (a->budget->committed >= a->committed)
            a->budget->committed -= a->committed;
        else
            a->budget->committed = 0;
    }
    memset(a, 0, sizeof(*a));
}

void* arena_alloc(Arena* a, size_t n) {
    if (!a || !a->base || n == 0) return NULL;

    const size_t align = 64;
    if (a->used > SIZE_MAX - (align - 1)) return NULL;
    const size_t offset = (a->used + (align - 1)) & ~(align - 1);
    if (offset > a->size || n > a->size - offset) return NULL;

    const size_t end = offset + n;

    if (end > a->committed) {
        const size_t page = a->page_size;
        if (page == 0 || end > SIZE_MAX - (page - 1)) return NULL;
        const size_t commit_end = ((end + page - 1) / page) * page;
        if (commit_end > a->size) return NULL;

        const size_t commit_bytes = commit_end - a->committed;
        if (commit_bytes > 0) {
            if (a->budget && a->budget->limit > 0 &&
                (a->budget->committed > a->budget->limit ||
                 commit_bytes > a->budget->limit - a->budget->committed)) {
                ++a->budget->failed;
                fprintf(stderr,
                    "Memory budget reached: request needs another %.2f MiB of arena pages; %.2f MiB of %.2f MiB is already committed. Increase --memory-budget-gb or reduce the model/batch/corpus workload.\n",
                    (double)commit_bytes / (1024.0 * 1024.0),
                    (double)a->budget->committed / (1024.0 * 1024.0),
                    (double)a->budget->limit / (1024.0 * 1024.0));
                return NULL;
            }
            void* committed = VirtualAlloc(a->base + a->committed,
                commit_bytes, MEM_COMMIT, PAGE_READWRITE);
            if (!committed) {
                fprintf(stderr, "Arena page commitment failed (Windows error %lu).\n",
                    (unsigned long)GetLastError());
                return NULL;
            }
            a->committed = commit_end;
            if (a->budget) {
                a->budget->committed += commit_bytes;
                if (a->budget->committed > a->budget->peak)
                    a->budget->peak = a->budget->committed;
            }
        }
    }

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
