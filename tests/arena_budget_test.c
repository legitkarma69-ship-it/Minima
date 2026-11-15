#include "arena.h"
#include <stdio.h>

int main(void) {
    ArenaBudget budget;
    Arena first, second;
    arena_budget_init(&budget, 4096u);
    if (!arena_init_with_budget(&first, 65536u, &budget)) return 1;
    if (!arena_init_with_budget(&second, 65536u, &budget)) return 2;

    void* a = arena_alloc(&first, 1u);
    if (!a || budget.committed == 0 || budget.committed > budget.limit) return 3;
    const size_t committed_after_first = budget.committed;

    /* The next allocation crosses a page boundary and must be rejected. */
    void* b = arena_alloc(&second, 4096u);
    if (b != NULL || budget.failed != 1u ||
        budget.committed != committed_after_first) return 4;

    arena_free(&first);
    if (budget.committed != 0u) return 5;
    b = arena_alloc(&second, 4096u);
    if (!b || budget.committed == 0u || budget.committed > budget.limit) return 6;

    arena_free(&second);
    if (budget.committed != 0u || budget.peak < committed_after_first) return 7;

    /* Zero means unlimited, retaining the legacy allocation behavior. */
    arena_budget_init(&budget, 0u);
    if (!arena_init_with_budget(&first, 65536u, &budget)) return 8;
    a = arena_alloc(&first, 8192u);
    if (!a || budget.committed == 0u || budget.failed != 0u) return 9;
    arena_free(&first);

    /* Tracked heap allocations share the same ceiling as arena pages. */
    arena_budget_init(&budget, 8192u);
    if (!arena_init_with_budget(&first, 65536u, &budget)) return 10;
    a = arena_alloc(&first, 1u);
    if (!a || budget.committed == 0u) return 11;
    void* external = arena_budget_calloc(&budget, 1024u, sizeof(uint32_t));
    if (!external || budget.external_committed != 4096u ||
        arena_budget_total_committed(&budget) != 8192u) return 12;
    if (arena_budget_malloc(&budget, 1u) != NULL || budget.failed != 1u)
        return 13;
    arena_budget_free(external);
    if (budget.external_committed != 0u ||
        arena_budget_total_committed(&budget) != budget.committed)
        return 14;
    external = arena_budget_malloc(&budget, 4096u);
    if (!external || budget.external_committed != 4096u ||
        budget.peak != 8192u) return 15;
    arena_budget_free(external);
    arena_free(&first);
    if (arena_budget_total_committed(&budget) != 0u) return 16;

    /* Allocation-size overflow is rejected without corrupting the budget. */
    arena_budget_init(&budget, 0u);
    if (arena_budget_calloc(&budget, SIZE_MAX, 2u) != NULL) return 17;

    puts("Shared arena + tracked heap memory budget: PASS");
    return 0;
}
