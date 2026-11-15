#include "arena.h"
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

typedef union {
    struct { uint64_t magic; ArenaBudget* budget; size_t size; } fields;
    uint64_t alignment[4]; /* 32-byte header preserves malloc's x64 alignment. */
} BudgetAllocationHeader;
#define BUDGET_ALLOCATION_MAGIC UINT64_C(0x4D494E494D414D45)

static void budget_lock(ArenaBudget* budget) {
    while (InterlockedCompareExchange((volatile LONG*)&budget->lock, 1, 0) != 0)
        Sleep(0);
}
static void budget_unlock(ArenaBudget* budget) {
    InterlockedExchange((volatile LONG*)&budget->lock, 0);
}
static bool budget_reserve(ArenaBudget* budget, size_t bytes, bool external) {
    if (!budget || bytes == 0) return true;
    budget_lock(budget);
    const size_t current = budget->committed + budget->external_committed;
    const bool overflow = current < budget->committed || bytes > SIZE_MAX - current;
    if (overflow || (budget->limit > 0 &&
        (current > budget->limit || bytes > budget->limit - current))) {
        budget->failed++;
        budget_unlock(budget);
        return false;
    }
    if (external) budget->external_committed += bytes;
    else budget->committed += bytes;
    const size_t total = budget->committed + budget->external_committed;
    if (total > budget->peak) budget->peak = total;
    budget_unlock(budget);
    return true;
}
static void budget_release(ArenaBudget* budget, size_t bytes, bool external) {
    if (!budget || bytes == 0) return;
    budget_lock(budget);
    size_t* current = external ? &budget->external_committed : &budget->committed;
    *current = *current >= bytes ? *current - bytes : 0;
    budget_unlock(budget);
}
void arena_budget_init(ArenaBudget* budget, size_t limit) {
    if (!budget) return;
    memset(budget, 0, sizeof(*budget));
    budget->limit = limit;
}
size_t arena_budget_total_committed(const ArenaBudget* budget) {
    if (!budget) return 0;
    if (budget->committed > SIZE_MAX - budget->external_committed) return SIZE_MAX;
    return budget->committed + budget->external_committed;
}
void* arena_budget_malloc(ArenaBudget* budget, size_t size) {
    if (size == 0) size = 1;
    if (size > SIZE_MAX - sizeof(BudgetAllocationHeader)) return NULL;
    if (!budget_reserve(budget, size, true)) {
        fprintf(stderr, "Engine memory budget reached: tracked heap request needs %.2f MiB. Increase --memory-budget-gb or reduce the model/tokenizer workload.\\n",
            (double)size / (1024.0 * 1024.0));
        return NULL;
    }
    BudgetAllocationHeader* header =
        (BudgetAllocationHeader*)malloc(sizeof(BudgetAllocationHeader) + size);
    if (!header) { budget_release(budget, size, true); return NULL; }
    header->fields.magic = BUDGET_ALLOCATION_MAGIC;
    header->fields.budget = budget;
    header->fields.size = size;
    return (void*)(header + 1);
}
void* arena_budget_calloc(ArenaBudget* budget, size_t count, size_t size) {
    if (count != 0 && size > SIZE_MAX / count) return NULL;
    const size_t bytes = count * size;
    void* allocation = arena_budget_malloc(budget, bytes);
    if (allocation) memset(allocation, 0, bytes);
    return allocation;
}
void arena_budget_free(void* allocation) {
    if (!allocation) return;
    BudgetAllocationHeader* header = ((BudgetAllocationHeader*)allocation) - 1;
    if (header->fields.magic != BUDGET_ALLOCATION_MAGIC) return;
    ArenaBudget* budget = header->fields.budget;
    const size_t size = header->fields.size;
    header->fields.magic = 0;
    budget_release(budget, size, true);
    free(header);
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
    budget_release(a->budget, a->committed, false);
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
            if (!budget_reserve(a->budget, commit_bytes, false)) {
                fprintf(stderr,
                    "Engine memory budget reached: arena request needs another %.2f MiB. Increase --memory-budget-gb or reduce the model/batch/corpus workload.\n",
                    (double)commit_bytes / (1024.0 * 1024.0));
                return NULL;
            }
            void* committed = VirtualAlloc(a->base + a->committed,
                commit_bytes, MEM_COMMIT, PAGE_READWRITE);
            if (!committed) {
                budget_release(a->budget, commit_bytes, false);
                fprintf(stderr, "Arena page commitment failed (Windows error %lu).\n",
                    (unsigned long)GetLastError());
                return NULL;
            }
            a->committed = commit_end;
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
