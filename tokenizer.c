#include "tokenizer.h"
#include "util.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#ifdef _MSC_VER
#include <intrin.h>
#endif
#else
#include <unistd.h>
#endif

// ------------------------------------------------------------------
// Internal helpers
// ------------------------------------------------------------------

static void add_token(Tokenizer* tok, Arena* arena, const char* s, int len) {
    if (tok->vocab_size >= TOK_MAX_VOCAB) die("vocab overflow");
    char* copy = arena_alloc(arena, len + 1);
    if (!copy) die("arena OOM in tokenizer");
    memcpy(copy, s, len);
    copy[len] = 0;
    tok->token_str[tok->vocab_size] = copy;
    tok->token_len[tok->vocab_size] = len;
    tok->vocab_size++;
}

// Very small open-address hash for pair counting during training
typedef struct {
    uint32_t key;     // (left << 16) | right
    int32_t  count;
    int32_t  new_id;  // filled after merge decision
} PairSlot;


static size_t pair_hash(uint16_t left, uint16_t right, size_t table_size) {
    uint32_t k = ((uint32_t)left << 16) | right;
    k *= 0x9e3779b9u;
    return (size_t)k & (table_size - 1);
}

// ------------------------------------------------------------------
// Training
// ------------------------------------------------------------------

bool tokenizer_train_reference(Tokenizer* tok, Arena* arena,
    const uint8_t* sample, size_t sample_len,
    int target_vocab) {

    if (!tok || !arena || !sample || sample_len == 0 ||
        target_vocab < 260 || target_vocab > TOK_MAX_VOCAB)
        return false;

    memset(tok, 0, sizeof(*tok));

    // 1. Special tokens
    add_token(tok, arena, "<bos>", 5);
    add_token(tok, arena, "<eos>", 5);
    add_token(tok, arena, "<pad>", 5);
    add_token(tok, arena, "<unk>", 5);

    // 2. Raw bytes 0..255
    for (int b = 0; b < 256; b++) {
        char tmp[8];
        int len = snprintf(tmp, sizeof(tmp), "<0x%02x>", b);
        // We actually store the single byte for speed later
        // but keep a printable name for debugging.
        // For the real token we store the byte itself.
        char byte[2] = { (char)b, 0 };
        add_token(tok, arena, byte, 1);
        (void)len; // silence unused
    }

    // At this point vocab_size == 260 (4 specials + 256 bytes)

    // 3. Work buffer: we keep a list of token ids for the whole sample
    //    (this is the most memory-hungry part of training)
    size_t max_tokens = sample_len + 16;
    uint16_t* ids = arena_alloc(arena, max_tokens * sizeof(uint16_t));
    if (!ids) die("arena OOM for training ids");

    // Initial: every byte becomes its own token
    size_t n = 0;
    for (size_t i = 0; i < sample_len; i++) {
        ids[n++] = (uint16_t)(TOK_BYTE_START + sample[i]);
    }

    // 4. Pair counting table (temporary, lives in arena). Size it from
    // the maximum possible number of adjacent pairs in this sample, keeping
    // the load factor at or below 0.5 instead of relying on a fixed table.
    if (sample_len > SIZE_MAX / 2) return false;
    size_t pair_hash_size = (size_t)1 << 18;
    const size_t required_slots = sample_len * 2;
    while (pair_hash_size < required_slots) {
        if (pair_hash_size > SIZE_MAX / 2) return false;
        pair_hash_size *= 2;
    }
    if (pair_hash_size > SIZE_MAX / sizeof(PairSlot)) return false;
    PairSlot* table = arena_alloc_zero(arena,
        pair_hash_size * sizeof(PairSlot));
    if (!table) return false;

    // Track occupied hash slots so each merge only clears and scans entries
    // that were actually used. This avoids two full-table passes per merge.
    if (sample_len > SIZE_MAX / sizeof(size_t)) return false;
    size_t* occupied_slots = arena_alloc(arena,
        sample_len * sizeof(size_t));
    if (!occupied_slots) return false;

    int merges_needed = target_vocab - tok->vocab_size;
    size_t occupied_count = 0;

    for (int m = 0; m < merges_needed; m++) {
        for (size_t i = 0; i < occupied_count; i++) {
            const size_t slot = occupied_slots[i];
            table[slot].count = 0;
            table[slot].key = 0;
        }
        occupied_count = 0;

        // Count pairs
        for (size_t i = 0; i + 1 < n; i++) {
            uint16_t a = ids[i];
            uint16_t b = ids[i + 1];

            // Keep every learned token within TOK_MAX_TOKEN_LEN.
            int len_a = tok->token_len[a];
            int len_b = tok->token_len[b];
            if (len_a <= 0 || len_b <= 0 ||
                len_a > TOK_MAX_TOKEN_LEN - len_b)
                continue;

            size_t h = pair_hash(a, b, pair_hash_size);
            // Linear probing must either find this pair or an empty slot.
            // Probe the dynamically sized table until the pair or an empty slot is found.
            bool inserted = false;
            const uint32_t key = ((uint32_t)a << 16) | b;
            for (size_t probe = 0; probe < pair_hash_size; probe++) {
                size_t slot = (h + probe) & (pair_hash_size - 1);
                if (table[slot].count == 0) {
                    table[slot].key = key;
                    table[slot].count = 1;
                    occupied_slots[occupied_count++] = slot;
                    inserted = true;
                    break;
                }
                if (table[slot].key == key) {
                    if (table[slot].count == INT32_MAX)
                        return false;
                    table[slot].count++;
                    inserted = true;
                    break;
                }
            }
            if (!inserted)
                return false;
        }

        // Find most frequent pair
        int best_count = 0;
        size_t best_slot = SIZE_MAX;
        uint16_t best_left = 0, best_right = 0;
        for (size_t i = 0; i < occupied_count; i++) {
            const size_t slot = occupied_slots[i];
            const int count = table[slot].count;
            // Preserve the previous lowest-slot tie break, independent of
            // the order in which pairs first appeared in the sample.
            if (count > best_count ||
                (count == best_count && count > 0 && slot < best_slot)) {
                best_count = count;
                best_slot = slot;
                best_left = (uint16_t)(table[slot].key >> 16);
                best_right = (uint16_t)(table[slot].key & 0xffff);
            }
        }

        if (best_count < 2) break;   // nothing left to merge

        // Create new token
        int new_id = tok->vocab_size;
        // Build the string by concatenating the two children
        int len1 = tok->token_len[best_left];
        int len2 = tok->token_len[best_right];
        char* new_str = arena_alloc(arena, (size_t)len1 + (size_t)len2 + 1);
        if (!new_str) die("arena OOM for merged token");
        memcpy(new_str, tok->token_str[best_left], (size_t)len1);
        memcpy(new_str + len1, tok->token_str[best_right], (size_t)len2);
        new_str[len1 + len2] = 0;
        tok->token_str[new_id] = new_str;
        tok->token_len[new_id] = len1 + len2;
        tok->vocab_size++;

        // Record merge
        tok->merges[tok->n_merges].left = best_left;
        tok->merges[tok->n_merges].right = best_right;
        tok->n_merges++;

        // Replace all occurrences of the pair in the id stream
        size_t w = 0;
        for (size_t r = 0; r < n; ) {
            if (r + 1 < n && ids[r] == best_left && ids[r + 1] == best_right) {
                ids[w++] = (uint16_t)new_id;
                r += 2;
            }
            else {
                ids[w++] = ids[r++];
            }
        }
        n = w;

        if ((m + 1) % 500 == 0)
            printf("merge %d / %d   tokens left in sample: %zu\n",
                m + 1, merges_needed, n);
    }

    printf("BPE finished. vocab = %d  merges = %d\n",
        tok->vocab_size, tok->n_merges);
    return true;
}


/* Keep the production entry point on the proven reference path until the
   incremental trainer passes equivalence and benchmark acceptance gates. */
bool tokenizer_train(Tokenizer* tok, Arena* arena,
    const uint8_t* sample, size_t sample_len, int target_vocab) {
    return tokenizer_train_reference(tok, arena, sample, sample_len,
        target_vocab);
}


/* ------------------------------------------------------------------
   Incremental BPE trainer.

   The old recounting implementation remains available as the reference.
   Pair occurrence lists are ordered by original byte position. Every newly
   created pair contains the newest token ID, so occurrences for that pair
   are appended in left-to-right order during a merge sweep.
   ------------------------------------------------------------------ */

#define BPE_NONE UINT32_MAX

typedef struct {
    uint16_t id;
    uint16_t active;
    uint32_t prev;
    uint32_t next;
    uint32_t occ_prev;
    uint32_t occ_next;
    uint32_t pair_slot;
} BpeNode;

typedef struct {
    uint32_t key;
    uint32_t count;
    uint32_t head;
    uint32_t tail;
    uint32_t heap_pos_plus1;
} BpePairState;

typedef struct {
    uint32_t key;
    uint32_t count;
    uint32_t generation;
} BpeReferenceSlot;

static uint32_t bpe_ctz64(uint64_t value) {
#ifdef _MSC_VER
    unsigned long bit = 0;
    _BitScanForward64(&bit, value);
    return (uint32_t)bit;
#else
    return (uint32_t)__builtin_ctzll(value);
#endif
}

static bool bpe_pair_better(const BpePairState* states,
    uint32_t left_slot, uint32_t right_slot) {
    if (states[left_slot].count != states[right_slot].count)
        return states[left_slot].count > states[right_slot].count;
    return left_slot < right_slot;
}

static void bpe_heap_swap(BpePairState* states, uint32_t* heap,
    uint32_t a, uint32_t b) {
    const uint32_t tmp = heap[a];
    heap[a] = heap[b];
    heap[b] = tmp;
    states[heap[a]].heap_pos_plus1 = a + 1;
    states[heap[b]].heap_pos_plus1 = b + 1;
}

static void bpe_heap_up(BpePairState* states, uint32_t* heap,
    uint32_t index) {
    while (index > 0) {
        const uint32_t parent = (index - 1) / 2;
        if (!bpe_pair_better(states, heap[index], heap[parent]))
            break;
        bpe_heap_swap(states, heap, index, parent);
        index = parent;
    }
}

static void bpe_heap_down(BpePairState* states, uint32_t* heap,
    uint32_t heap_count, uint32_t index) {
    for (;;) {
        const uint32_t left = index * 2 + 1;
        const uint32_t right = left + 1;
        uint32_t best = index;
        if (left < heap_count &&
            bpe_pair_better(states, heap[left], heap[best]))
            best = left;
        if (right < heap_count &&
            bpe_pair_better(states, heap[right], heap[best]))
            best = right;
        if (best == index) break;
        bpe_heap_swap(states, heap, index, best);
        index = best;
    }
}

static void bpe_heap_remove(BpePairState* states, uint32_t* heap,
    uint32_t* heap_count, uint32_t slot) {
    const uint32_t position_plus1 = states[slot].heap_pos_plus1;
    if (!position_plus1) return;
    const uint32_t position = position_plus1 - 1;
    const uint32_t last_index = --(*heap_count);
    states[slot].heap_pos_plus1 = 0;
    if (position == last_index) return;
    heap[position] = heap[last_index];
    states[heap[position]].heap_pos_plus1 = position + 1;
    if (position > 0 &&
        bpe_pair_better(states, heap[position], heap[(position - 1) / 2]))
        bpe_heap_up(states, heap, position);
    else
        bpe_heap_down(states, heap, *heap_count, position);
}

static void bpe_heap_changed(BpePairState* states, uint32_t* heap,
    uint32_t* heap_count, uint32_t slot) {
    if (states[slot].count == 0) {
        bpe_heap_remove(states, heap, heap_count, slot);
        return;
    }
    if (!states[slot].heap_pos_plus1) {
        const uint32_t position = (*heap_count)++;
        heap[position] = slot;
        states[slot].heap_pos_plus1 = position + 1;
        bpe_heap_up(states, heap, position);
        return;
    }
    const uint32_t position = states[slot].heap_pos_plus1 - 1;
    if (position > 0 &&
        bpe_pair_better(states, heap[position], heap[(position - 1) / 2]))
        bpe_heap_up(states, heap, position);
    else
        bpe_heap_down(states, heap, *heap_count, position);
}

static uint32_t bpe_pair_state_find(BpePairState* states,
    size_t table_size, uint16_t left, uint16_t right, bool create) {
    const uint32_t key = ((uint32_t)left << 16) | right;
    const size_t start = pair_hash(left, right, table_size);
    for (size_t probe = 0; probe < table_size; probe++) {
        const size_t slot = (start + probe) & (table_size - 1);
        if (states[slot].key == key)
            return (uint32_t)slot;
        if (states[slot].key == 0) {
            if (!create) return BPE_NONE;
            states[slot].key = key;
            states[slot].head = BPE_NONE;
            states[slot].tail = BPE_NONE;
            return (uint32_t)slot;
        }
    }
    return BPE_NONE;
}

static bool bpe_pair_add(BpeNode* nodes, const Tokenizer* tok,
    BpePairState* states, size_t state_table_size,
    uint32_t* heap, uint32_t* heap_count, uint32_t left_index,
    uint32_t* first_slot_by_node, uint64_t* first_occurrence_bits) {
    const uint32_t right_index = nodes[left_index].next;
    if (right_index == BPE_NONE) return true;
    const uint16_t left_id = nodes[left_index].id;
    const uint16_t right_id = nodes[right_index].id;
    const int left_len = tok->token_len[left_id];
    const int right_len = tok->token_len[right_id];
    if (left_len <= 0 || right_len <= 0 ||
        left_len > TOK_MAX_TOKEN_LEN - right_len) {
        nodes[left_index].pair_slot = BPE_NONE;
        nodes[left_index].occ_prev = BPE_NONE;
        nodes[left_index].occ_next = BPE_NONE;
        return true;
    }

    const uint32_t slot = bpe_pair_state_find(states, state_table_size,
        left_id, right_id, true);
    if (slot == BPE_NONE || states[slot].count == UINT32_MAX)
        return false;

    const bool was_empty = states[slot].count == 0;
    nodes[left_index].pair_slot = slot;
    nodes[left_index].occ_prev = states[slot].tail;
    nodes[left_index].occ_next = BPE_NONE;
    if (states[slot].tail != BPE_NONE)
        nodes[states[slot].tail].occ_next = left_index;
    else
        states[slot].head = left_index;
    states[slot].tail = left_index;
    if (was_empty) {
        first_slot_by_node[left_index] = slot;
        first_occurrence_bits[left_index >> 6] |=
            (uint64_t)1 << (left_index & 63);
    }
    states[slot].count++;
    bpe_heap_changed(states, heap, heap_count, slot);
    return true;
}

static bool bpe_pair_remove(BpeNode* nodes, BpePairState* states,
    uint32_t* heap, uint32_t* heap_count, uint32_t left_index,
    uint32_t* first_slot_by_node, uint64_t* first_occurrence_bits) {
    const uint32_t slot = nodes[left_index].pair_slot;
    if (slot == BPE_NONE) return true;
    if (states[slot].count == 0) return false;

    const uint32_t before = nodes[left_index].occ_prev;
    const uint32_t after = nodes[left_index].occ_next;
    if (before != BPE_NONE) {
        nodes[before].occ_next = after;
    } else {
        states[slot].head = after;
        first_slot_by_node[left_index] = BPE_NONE;
        first_occurrence_bits[left_index >> 6] &=
            ~((uint64_t)1 << (left_index & 63));
        if (after != BPE_NONE) {
            first_slot_by_node[after] = slot;
            first_occurrence_bits[after >> 6] |=
                (uint64_t)1 << (after & 63);
        }
    }
    if (after != BPE_NONE)
        nodes[after].occ_prev = before;
    else
        states[slot].tail = before;

    nodes[left_index].pair_slot = BPE_NONE;
    nodes[left_index].occ_prev = BPE_NONE;
    nodes[left_index].occ_next = BPE_NONE;
    states[slot].count--;
    bpe_heap_changed(states, heap, heap_count, slot);
    return true;
}

/* Recreate the old hash table's insertion order without rescanning every
   active edge. The bitset contains exactly one bit per nonempty pair state,
   at the original position of that pair's first active occurrence. */
static bool bpe_reference_tie_winner(const BpePairState* states,
    const uint32_t* first_slot_by_node,
    const uint64_t* first_occurrence_bits, size_t bit_words,
    size_t reference_table_size, BpeReferenceSlot* table,
    uint32_t* generation_out, uint32_t* winner_key_out) {
    uint32_t generation = *generation_out + 1;
    if (generation == 0) {
        memset(table, 0, reference_table_size * sizeof(*table));
        generation = 1;
    }

    uint32_t best_count = 0;
    size_t best_slot = SIZE_MAX;
    uint32_t best_key = 0;
    for (size_t word_index = 0; word_index < bit_words; word_index++) {
        uint64_t bits = first_occurrence_bits[word_index];
        while (bits) {
            const uint32_t bit = bpe_ctz64(bits);
            const size_t node_index = word_index * 64 + bit;
            const uint32_t state_slot = first_slot_by_node[node_index];
            if (state_slot == BPE_NONE || states[state_slot].count == 0)
                return false;
            const uint32_t key = states[state_slot].key;
            const uint16_t left = (uint16_t)(key >> 16);
            const uint16_t right = (uint16_t)key;
            const size_t start = pair_hash(left, right,
                reference_table_size);
            bool inserted = false;
            size_t final_slot = SIZE_MAX;
            for (size_t probe = 0; probe < reference_table_size; probe++) {
                const size_t slot = (start + probe) &
                    (reference_table_size - 1);
                if (table[slot].generation != generation) {
                    table[slot].generation = generation;
                    table[slot].key = key;
                    table[slot].count = states[state_slot].count;
                    final_slot = slot;
                    inserted = true;
                    break;
                }
                if (table[slot].key == key)
                    return false; /* each pair state must appear only once */
            }
            if (!inserted) return false;

            const uint32_t count = states[state_slot].count;
            if (count > best_count ||
                (count == best_count && final_slot < best_slot)) {
                best_count = count;
                best_slot = final_slot;
                best_key = key;
            }
            bits &= bits - 1;
        }
    }

    *generation_out = generation;
    *winner_key_out = best_key;
    return best_count > 0;
}

bool tokenizer_train_incremental(Tokenizer* tok, Arena* arena,
    const uint8_t* sample, size_t sample_len, int target_vocab) {
    if (!tok || !arena || !sample || sample_len == 0 ||
        sample_len > INT32_MAX ||
        target_vocab < 260 || target_vocab > TOK_MAX_VOCAB ||
        sample_len > SIZE_MAX / sizeof(BpeNode) ||
        sample_len > SIZE_MAX / 2)
        return false;

    const size_t arena_used_before = arena->used;
    memset(tok, 0, sizeof(*tok));
    add_token(tok, arena, "<bos>", 5);
    add_token(tok, arena, "<eos>", 5);
    add_token(tok, arena, "<pad>", 5);
    add_token(tok, arena, "<unk>", 5);
    for (int b = 0; b < 256; b++) {
        const char byte[1] = { (char)b };
        add_token(tok, arena, byte, 1);
    }

    const uint32_t node_count = (uint32_t)sample_len;
    BpeNode* nodes = (BpeNode*)arena_alloc_zero(arena,
        sample_len * sizeof(BpeNode));
    if (!nodes) goto fail;

    /* Initial edges contribute at most n-1 pair keys. Every successful
       merge removes one active node and can add at most two new edges, so
       across the entire run no more than 3*(n-1) distinct keys can appear. */
    if (sample_len - 1 > (SIZE_MAX / 3)) goto fail;
    size_t max_pair_states = (sample_len - 1) * 3;
    if (max_pair_states == 0) max_pair_states = 1;
    if (max_pair_states > UINT32_MAX ||
        max_pair_states > SIZE_MAX / 2 ||
        max_pair_states > SIZE_MAX / sizeof(uint32_t))
        goto fail;

    size_t state_table_size = 8;
    /* Keep linear-probing occupancy at or below 75% at the proven upper
       bound, rather than reserving twice the maximum state count. */
    const size_t required_state_slots =
        max_pair_states + max_pair_states / 3 + 1;
    while (state_table_size < required_state_slots) {
        if (state_table_size > SIZE_MAX / 2) goto fail;
        state_table_size *= 2;
    }
    if (state_table_size > UINT32_MAX ||
        state_table_size > SIZE_MAX / sizeof(BpePairState))
        goto fail;

    BpePairState* states = (BpePairState*)arena_alloc_zero(arena,
        state_table_size * sizeof(BpePairState));
    uint32_t* heap = (uint32_t*)arena_alloc(arena,
        max_pair_states * sizeof(uint32_t));
    if (!states || !heap) goto fail;

    const size_t first_occurrence_words = (sample_len + 63) / 64;
    if (sample_len > SIZE_MAX / sizeof(uint32_t) ||
        first_occurrence_words > SIZE_MAX / sizeof(uint64_t))
        goto fail;
    uint32_t* first_slot_by_node = (uint32_t*)arena_alloc(arena,
        sample_len * sizeof(uint32_t));
    uint64_t* first_occurrence_bits = (uint64_t*)arena_alloc_zero(arena,
        first_occurrence_words * sizeof(uint64_t));
    if (!first_slot_by_node || !first_occurrence_bits) goto fail;
    for (uint32_t i = 0; i < node_count; i++)
        first_slot_by_node[i] = BPE_NONE;

    size_t reference_table_size = (size_t)1 << 18;
    const size_t required_reference_slots = sample_len * 2;
    while (reference_table_size < required_reference_slots) {
        if (reference_table_size > SIZE_MAX / 2) goto fail;
        reference_table_size *= 2;
    }
    if (reference_table_size > SIZE_MAX / sizeof(BpeReferenceSlot))
        goto fail;
    BpeReferenceSlot* reference_table =
        (BpeReferenceSlot*)arena_alloc_zero(arena,
            reference_table_size * sizeof(BpeReferenceSlot));
    if (!reference_table) goto fail;

    for (uint32_t i = 0; i < node_count; i++) {
        nodes[i].id = (uint16_t)(TOK_BYTE_START + sample[i]);
        nodes[i].active = 1;
        nodes[i].prev = i == 0 ? BPE_NONE : i - 1;
        nodes[i].next = i + 1 == node_count ? BPE_NONE : i + 1;
        nodes[i].occ_prev = BPE_NONE;
        nodes[i].occ_next = BPE_NONE;
        nodes[i].pair_slot = BPE_NONE;
    }

    uint32_t heap_count = 0;
    for (uint32_t i = 0; i + 1 < node_count; i++) {
        if (!bpe_pair_add(nodes, tok, states, state_table_size,
                heap, &heap_count, i, first_slot_by_node,
                first_occurrence_bits))
            goto fail;
    }

    uint32_t reference_generation = 0;
    size_t active_count = sample_len;

    const int merges_needed = target_vocab - tok->vocab_size;
    for (int merge_index = 0; merge_index < merges_needed; merge_index++) {
        if (heap_count == 0) break;
        uint32_t best_slot = heap[0];
        if (states[best_slot].count < 2) break;

        const bool tied_maximum =
            (heap_count > 1 &&
                states[heap[1]].count == states[best_slot].count) ||
            (heap_count > 2 &&
                states[heap[2]].count == states[best_slot].count);
        if (tied_maximum) {
            uint32_t winner_key = 0;
            if (!bpe_reference_tie_winner(states, first_slot_by_node,
                    first_occurrence_bits, first_occurrence_words,
                    reference_table_size, reference_table,
                    &reference_generation, &winner_key))
                goto fail;
            const uint16_t winner_left = (uint16_t)(winner_key >> 16);
            const uint16_t winner_right = (uint16_t)winner_key;
            best_slot = bpe_pair_state_find(states, state_table_size,
                winner_left, winner_right, false);
            if (best_slot == BPE_NONE || states[best_slot].count < 2)
                goto fail;
        }

        const uint16_t best_left = (uint16_t)(states[best_slot].key >> 16);
        const uint16_t best_right = (uint16_t)states[best_slot].key;
        const int left_len = tok->token_len[best_left];
        const int right_len = tok->token_len[best_right];
        if (left_len <= 0 || right_len <= 0 ||
            left_len > TOK_MAX_TOKEN_LEN - right_len)
            goto fail;

        const int new_id = tok->vocab_size;
        char* new_str = (char*)arena_alloc(arena,
            (size_t)left_len + (size_t)right_len + 1);
        if (!new_str) goto fail;
        memcpy(new_str, tok->token_str[best_left], (size_t)left_len);
        memcpy(new_str + left_len, tok->token_str[best_right],
            (size_t)right_len);
        new_str[left_len + right_len] = 0;
        tok->token_str[new_id] = new_str;
        tok->token_len[new_id] = left_len + right_len;
        tok->vocab_size++;
        tok->merges[tok->n_merges].left = best_left;
        tok->merges[tok->n_merges].right = best_right;
        tok->n_merges++;

        /* The selected pair's occurrence list is left-to-right. Removing
           the outgoing edge of the right node also invalidates an overlapping
           occurrence, exactly matching the reference's greedy sweep. */
        while (states[best_slot].head != BPE_NONE) {
            const uint32_t left_index = states[best_slot].head;
            const uint32_t right_index = nodes[left_index].next;
            if (right_index == BPE_NONE ||
                !nodes[left_index].active || !nodes[right_index].active ||
                nodes[left_index].id != best_left ||
                nodes[right_index].id != best_right ||
                nodes[left_index].pair_slot != best_slot)
                goto fail;

            const uint32_t prev_index = nodes[left_index].prev;
            const uint32_t next_index = nodes[right_index].next;
            if (prev_index != BPE_NONE &&
                !bpe_pair_remove(nodes, states, heap, &heap_count,
                    prev_index, first_slot_by_node, first_occurrence_bits))
                goto fail;
            if (!bpe_pair_remove(nodes, states, heap, &heap_count,
                    left_index, first_slot_by_node, first_occurrence_bits))
                goto fail;
            if (next_index != BPE_NONE &&
                !bpe_pair_remove(nodes, states, heap, &heap_count,
                    right_index, first_slot_by_node, first_occurrence_bits))
                goto fail;

            nodes[left_index].id = (uint16_t)new_id;
            nodes[left_index].next = next_index;
            if (next_index != BPE_NONE)
                nodes[next_index].prev = left_index;
            nodes[right_index].active = 0;
            active_count--;
            nodes[right_index].prev = BPE_NONE;
            nodes[right_index].next = BPE_NONE;

            if (prev_index != BPE_NONE &&
                !bpe_pair_add(nodes, tok, states, state_table_size,
                    heap, &heap_count, prev_index, first_slot_by_node,
                    first_occurrence_bits))
                goto fail;
            if (!bpe_pair_add(nodes, tok, states, state_table_size,
                    heap, &heap_count, left_index, first_slot_by_node,
                    first_occurrence_bits))
                goto fail;
        }

        if ((merge_index + 1) % 500 == 0)
            printf("merge %d / %d   tokens left in sample: %zu\n",
                merge_index + 1, merges_needed, active_count);
    }

    return true;

fail:
    arena->used = arena_used_before;
    memset(tok, 0, sizeof(*tok));
    return false;
}


// ------------------------------------------------------------------
// Encode
// ------------------------------------------------------------------

int tokenizer_encode(const Tokenizer* tok,
    const uint8_t* text, size_t text_len,
    uint16_t* out, int out_cap) {

    if (!tok || (!text && text_len) || (!out && text_len) ||
        out_cap < 0 || text_len > (size_t)INT_MAX)
        return -1;
    if (text_len == 0) return 0;
    /* BPE can only reduce the byte-token count. If the output cannot hold
       the byte-level worst case, fail before writing anything. */
    if (!out || (size_t)out_cap < text_len) return -1;

    // First pass: bytes → token ids. Capacity is now guaranteed.
    int n = 0;
    for (size_t i = 0; i < text_len; i++) {
        out[n++] = (uint16_t)(TOK_BYTE_START + text[i]);
    }

    // Apply merges in order (classic BPE)
    // Because the merge list is small we just rescanning is fine
    // for training batch sizes.
    for (int m = 0; m < tok->n_merges; m++) {
        uint16_t left = tok->merges[m].left;
        uint16_t right = tok->merges[m].right;
        uint16_t newid = (uint16_t)(260 + m);   // specials + bytes = 260

        int w = 0;
        for (int r = 0; r < n; ) {
            if (r + 1 < n && out[r] == left && out[r + 1] == right) {
                out[w++] = newid;
                r += 2;
            }
            else {
                out[w++] = out[r++];
            }
        }
        n = w;
    }

    return n;
}

// ------------------------------------------------------------------
// Decode
// ------------------------------------------------------------------

int tokenizer_decode(const Tokenizer* tok,
    const uint16_t* ids, int n_ids,
    char* out, int out_cap) {

    if (!tok || n_ids < 0 || out_cap <= 0 || !out ||
        (n_ids > 0 && !ids) ||
        tok->vocab_size < 260 || tok->vocab_size > TOK_MAX_VOCAB)
        return -1;

    size_t required = 0;
    for (int i = 0; i < n_ids; i++) {
        const uint16_t id = ids[i];
        if (id >= tok->vocab_size || !tok->token_str[id] ||
            tok->token_len[id] <= 0 ||
            tok->token_len[id] > TOK_MAX_TOKEN_LEN ||
            required > (size_t)(out_cap - 1) ||
            (size_t)tok->token_len[id] >
                (size_t)(out_cap - 1) - required)
            return -1;
        required += (size_t)tok->token_len[id];
    }

    size_t pos = 0;
    for (int i = 0; i < n_ids; i++) {
        const int len = tok->token_len[ids[i]];
        memcpy(out + pos, tok->token_str[ids[i]], (size_t)len);
        pos += (size_t)len;
    }
    out[pos] = 0;
    return (int)pos;
}

// ------------------------------------------------------------------
// Stable tokenizer fingerprint
// ------------------------------------------------------------------

static uint64_t tokenizer_hash_bytes(uint64_t hash,
    const void* data, size_t length) {
    const uint8_t* bytes = (const uint8_t*)data;
    for (size_t i = 0; i < length; i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t tokenizer_fingerprint(const Tokenizer* tok) {
    if (!tok || tok->vocab_size < 260 ||
        tok->vocab_size > TOK_MAX_VOCAB ||
        tok->n_merges < 0 || tok->n_merges > TOK_MAX_MERGES)
        return 0;

    uint64_t hash = 14695981039346656037ULL;
    const uint32_t vocab_size = (uint32_t)tok->vocab_size;
    const uint32_t n_merges = (uint32_t)tok->n_merges;
    hash = tokenizer_hash_bytes(hash, &vocab_size, sizeof(vocab_size));
    hash = tokenizer_hash_bytes(hash, &n_merges, sizeof(n_merges));

    for (int i = 0; i < tok->vocab_size; i++) {
        if (!tok->token_str[i] || tok->token_len[i] <= 0 ||
            tok->token_len[i] > TOK_MAX_TOKEN_LEN)
            return 0;
        const uint32_t len = (uint32_t)tok->token_len[i];
        hash = tokenizer_hash_bytes(hash, &len, sizeof(len));
        hash = tokenizer_hash_bytes(hash, tok->token_str[i], len);
    }

    for (int i = 0; i < tok->n_merges; i++) {
        hash = tokenizer_hash_bytes(hash, &tok->merges[i].left,
            sizeof(tok->merges[i].left));
        hash = tokenizer_hash_bytes(hash, &tok->merges[i].right,
            sizeof(tok->merges[i].right));
    }
    return hash ? hash : 1;
}

// ------------------------------------------------------------------
// Save / Load (simple binary format)
// ------------------------------------------------------------------

bool tokenizer_save(const Tokenizer* tok, const char* path) {
    if (!tok || !path || tok->vocab_size < 260 ||
        tok->vocab_size > TOK_MAX_VOCAB ||
        tok->n_merges < 0 || tok->n_merges > TOK_MAX_MERGES ||
        tok->n_merges > tok->vocab_size - 260)
        return false;

    const size_t path_len = strlen(path);
    if (path_len > SIZE_MAX - 5) return false;
    char* temp_path = (char*)malloc(path_len + 5);
    if (!temp_path) return false;
    memcpy(temp_path, path, path_len);
    memcpy(temp_path + path_len, ".tmp", 5);

    FILE* f = fopen(temp_path, "wb");
    if (!f) { free(temp_path); return false; }

    uint32_t magic = 0x42504531;
    bool ok = fwrite(&magic, sizeof(magic), 1, f) == 1 &&
        fwrite(&tok->vocab_size, sizeof(tok->vocab_size), 1, f) == 1 &&
        fwrite(&tok->n_merges, sizeof(tok->n_merges), 1, f) == 1;

    for (int i = 0; ok && i < tok->vocab_size; i++) {
        const int token_len = tok->token_len[i];
        if (!tok->token_str[i] || token_len <= 0 ||
            token_len > TOK_MAX_TOKEN_LEN || token_len > UINT16_MAX) {
            ok = false;
            break;
        }
        const uint16_t len = (uint16_t)token_len;
        ok = fwrite(&len, sizeof(len), 1, f) == 1 &&
            fwrite(tok->token_str[i], 1, len, f) == len;
    }

    if (ok && tok->n_merges > 0)
        ok = fwrite(tok->merges, sizeof(Merge),
            (size_t)tok->n_merges, f) == (size_t)tok->n_merges;
    if (ok && fflush(f) != 0) ok = false;
    if (ok) {
#ifdef _WIN32
        if (_commit(_fileno(f)) != 0) ok = false;
#else
        if (fsync(fileno(f)) != 0) ok = false;
#endif
    }
    if (fclose(f) != 0) ok = false;

    if (ok) {
#ifdef _WIN32
        ok = MoveFileExA(temp_path, path,
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
        ok = rename(temp_path, path) == 0;
#endif
    }
    if (!ok) remove(temp_path);
    free(temp_path);
    return ok;
}

bool tokenizer_load(Tokenizer* tok, Arena* arena, const char* path) {
    if (!tok || !arena || !path) return false;
    const size_t arena_used_before = arena->used;
    FILE* f = fopen(path, "rb");
    if (!f) return false;

    uint32_t magic = 0;
    int vocab_size = 0;
    int n_merges = 0;
    if (fread(&magic, sizeof(magic), 1, f) != 1 ||
        magic != 0x42504531 ||
        fread(&vocab_size, sizeof(vocab_size), 1, f) != 1 ||
        fread(&n_merges, sizeof(n_merges), 1, f) != 1 ||
        vocab_size < 260 || vocab_size > TOK_MAX_VOCAB ||
        n_merges < 0 || n_merges > TOK_MAX_MERGES ||
        n_merges > vocab_size - 260)
        goto fail;

    Tokenizer loaded;
    memset(&loaded, 0, sizeof(loaded));
    loaded.vocab_size = vocab_size;
    loaded.n_merges = n_merges;

    for (int i = 0; i < vocab_size; i++) {
        uint16_t len = 0;
        if (fread(&len, sizeof(len), 1, f) != 1 ||
            len == 0 || len > TOK_MAX_TOKEN_LEN)
            goto fail;
        char* s = (char*)arena_alloc(arena, (size_t)len + 1);
        if (!s || fread(s, 1, len, f) != len)
            goto fail;
        s[len] = 0;
        loaded.token_str[i] = s;
        loaded.token_len[i] = len;
    }

    if (n_merges > 0 &&
        fread(loaded.merges, sizeof(Merge), (size_t)n_merges, f) !=
            (size_t)n_merges)
        goto fail;

    {
        static const char* reserved[4] = { "<bos>", "<eos>", "<pad>", "<unk>" };
        for (int i = 0; i < 4; i++) {
            if (loaded.token_len[i] != 5 ||
                memcmp(loaded.token_str[i], reserved[i], 5) != 0)
                goto fail;
        }
        for (int b = 0; b < 256; b++) {
            const int id = TOK_BYTE_START + b;
            if (loaded.token_len[id] != 1 ||
                (uint8_t)loaded.token_str[id][0] != (uint8_t)b)
                goto fail;
        }
    }

    for (int i = 0; i < n_merges; i++) {
        const uint16_t new_id = (uint16_t)(260 + i);
        const uint16_t left = loaded.merges[i].left;
        const uint16_t right = loaded.merges[i].right;
        if (left >= new_id || right >= new_id)
            goto fail;
        const int left_len = loaded.token_len[left];
        const int right_len = loaded.token_len[right];
        if (left_len <= 0 || right_len <= 0 ||
            left_len > TOK_MAX_TOKEN_LEN - right_len ||
            loaded.token_len[new_id] != left_len + right_len ||
            memcmp(loaded.token_str[new_id], loaded.token_str[left],
                (size_t)left_len) != 0 ||
            memcmp(loaded.token_str[new_id] + left_len,
                loaded.token_str[right], (size_t)right_len) != 0)
            goto fail;
    }

    if (fgetc(f) != EOF || ferror(f))
        goto fail;
    if (fclose(f) != 0) {
        arena->used = arena_used_before;
        return false;
    }
    *tok = loaded;
    return true;

fail:
    fclose(f);
    arena->used = arena_used_before;
    return false;
}

