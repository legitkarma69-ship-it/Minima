# Hybrid BPE v2: implementation study and design

## Scope

Minima currently implements byte-level BPE over one contiguous sample. It starts with four reserved IDs and all 256 byte IDs, repeatedly recounts every adjacent token pair across the full sample, chooses the most frequent pair, and replaces its occurrences left-to-right without overlap. Encoding then replays every merge rule over the complete token sequence.

Hybrid BPE v2 should improve the trainer and encoder independently. The existing raw-byte behavior remains the compatibility baseline. Pretokenization is a separate, explicit model setting, not a silent change to how existing BPE1 files are interpreted.

## What the reference implementations actually do

### OpenAI tiktoken

Sources:
- https://github.com/openai/tiktoken/blob/main/tiktoken/_educational.py
- https://github.com/openai/tiktoken/blob/main/src/lib.rs

The educational implementation makes the algorithm easy to inspect: split text into regex-defined pieces, turn each piece into bytes, repeatedly select the adjacent pair with the lowest merge rank, and merge it. The production Rust implementation keeps that same rank-driven behavior but has two execution paths:

- For pieces shorter than 100 bytes, it uses a compact vector of byte offsets and pair ranks. It rescans the short vector after a merge because locality and low overhead beat a heap at this size.
- For longer pieces, it uses linked state (prev, end, next_end) and a binary heap ordered by merge rank and start position. A merge invalidates nearby candidates and inserts only the new neighboring candidates. Stale heap entries are discarded when popped.
- The implementation explicitly documents why the small path remains O(m*n): real pieces are usually short, and cache locality can outweigh the asymptotic advantage of a heap.
- Encoding first applies a regex pre-tokenizer. BPE cannot merge across those piece boundaries. Special tokens are handled separately from ordinary text.
- It avoids a global LRU cache in favor of exploiting the vocabulary as a natural hot set; its comments describe lock contention and multi-threading tradeoffs.

What Minima should borrow: the two-path encoder, rank/start deterministic priority, local neighbor updates, stale-candidate checks, and reusable scratch buffers. The regex pattern itself should not be copied blindly: it is language/text-policy behavior, not merely an optimization.

### Hugging Face Tokenizers

Sources:
- https://github.com/huggingface/tokenizers/blob/main/tokenizers/tk-train/src/trainers/bpe/mod.rs
- https://github.com/huggingface/tokenizers/blob/main/tokenizers/tk-train/src/trainers/bpe/word.rs
- https://github.com/huggingface/tokenizers/blob/main/tokenizers/tk-encode/src/models/bpe/model.rs

The current trainer aggregates pre-tokenized words and their frequencies. It maps symbols to integer IDs, represents each word as a sequence of linked symbols, and counts adjacent pairs across the word inventory. Each pair candidate tracks the set of words in which it occurs. When a pair is selected, only affected words are merged; each word reports the local pair-count deltas caused by that change. Updated candidates are pushed to an octonary heap. Queue entries are checked against the current pair count when popped, which makes stale entries cheap to tolerate.

The trainer also exposes useful policy controls: minimum pair frequency, alphabet limits and forced initial alphabet entries, continuing-subword prefixes, end-of-word suffixes, and maximum token length. The current encoder has evolved into separate multipass and hot/cold queue merge engines, a reusable per-thread scratch object, and an optional word cache.

What Minima should borrow: integer symbol IDs, explicit minimum frequency and maximum token length, incremental local pair-count deltas, candidate-position tracking, lazy heap validation, and scratch reuse. Per-word aggregation is attractive only when Minima has a deliberate pre-tokenizer and stores word frequencies. It must not be introduced as an accidental semantic change to the existing whole-byte-stream mode.

### Google SentencePiece

Sources:
- https://github.com/google/sentencepiece/blob/master/src/bpe_model_trainer.cc
- https://github.com/google/sentencepiece/blob/master/src/bpe_model.cc
- https://github.com/google/sentencepiece/blob/master/src/normalizer.cc

The BPE trainer creates reusable symbol records for characters and candidate pairs. A pair symbol stores its frequency and the positions where it occurs. It initializes bigram statistics in parallel across sentences, then uses a priority queue. After accepting a pair, it validates stored positions against the current symbols, merges only valid occurrences, subtracts frequencies for the two neighboring pairs that disappear, and adds frequencies for the two new neighboring pairs. Changed candidates are marked for recomputation and reinserted lazily. Sentence weights contribute to pair frequencies.

The inference implementation uses a linked symbol sequence and a priority queue. It seeds valid adjacent pairs, pops the best-scoring candidate, rejects stale entries, merges the pair, updates prev/next, and adds only the newly adjacent candidates. It explicitly excludes reserved/control symbols from merges. Its normalization layer has its own whitespace, Unicode, and character-map policy.

What Minima should borrow: position-based incremental updates, stale-position validation, linked-neighbor mutation, lazy priority queues, and explicit protection of reserved IDs. SentencePiece's Unicode-character model and normalization policy are not drop-in replacements for Minima's byte-level model.

## Proposed Hybrid BPE v2

### 1. Preserve two tokenizer modes

- raw_bytes remains the default compatibility mode. It has the current semantics: the entire sample is one byte stream, and merges may cross whitespace, punctuation, and line boundaries.
- Add an opt-in whitespace_pieces mode later. It splits at a documented byte-level whitespace boundary while preserving every input byte and every whitespace byte. Each piece is trained and encoded independently. This is intentionally simpler than tiktoken's Unicode regex and SentencePiece's normalization; it must be named honestly and tested on UTF-8 and CRLF data.
- Store the mode in a versioned tokenizer file format. Continue loading BPE1 as raw_bytes. Never infer a pre-tokenizer from a vocabulary.

A future Unicode-aware pre-tokenizer can be added as another explicit mode after a Unicode library and its compatibility policy are chosen.

### 2. Incremental training

Use the SentencePiece/Hugging Face idea, adapted to Minima's single raw byte sequence:

- Represent each original byte position as a node with token ID and prev/next links. Merged-away nodes become inactive; the active sequence remains linked without moving the whole token array.
- Maintain a hash table from (left_id, right_id) to pair state: frequency, occurrence-list head/tail, and heap position.
- Each active node represents at most one outgoing adjacent pair. Pair occurrence lists retain left-to-right order. A merge removes the three affected edges (prev->left, left->right, right->next) and adds the two replacement edges (prev->merged, merged->next).
- Maintain an indexed max-heap of pair states. Update only the pair states touched by each merge. Use the existing hash-slot tie-break initially so equal-frequency choices remain deterministic and comparable with the current trainer.
- Preserve the existing overlapping-frequency / non-overlapping replacement behavior. For example, pair counting sees both adjacent AA pairs in AAA, while the left-to-right merge consumes the first pair and invalidates the overlapping second occurrence.
- Preserve the maximum token byte length rule and stop threshold as explicit configuration, rather than burying them in the data structure.

This should reduce repeated whole-sample pair recounts to work proportional to initial edges plus the local edge changes caused by merges. The implementation must be validated against a slow reference trainer on small exhaustive inputs before replacing the current path.

### 3. Hybrid encoding

- Build a pair-to-merge-rank lookup once when a tokenizer is trained or loaded.
- Keep a compact multipass path for short inputs. Use 100 bytes as the initial experimental threshold because that is the threshold documented by tiktoken, then tune it with Minima's own benchmark.
- For long inputs, use linked token nodes and a min-heap ordered by (merge_rank, start_position). After merging, enqueue only the new left and right neighbor candidates.
- Validate each heap entry against the node's current adjacency/version before applying it. Reuse scratch allocations where the calling engine can retain them.
- Ensure the queue path produces exactly the same token IDs as the existing merge-list replay path for every test input.

### 4. Training controls and reserved IDs

Borrow the useful Hugging Face controls:
- min_pair_frequency (default 2 to preserve the current stopping rule);
- max_token_bytes (default the current 64-byte limit);
- an explicit pre-tokenizer mode;
- an explicit merge-count / target-vocabulary contract.

Keep BOS, EOS, PAD, and UNK out of ordinary byte-pair candidates. Define their runtime semantics separately; they are not ordinary text bytes.

### 5. Benchmark and acceptance criteria

The new tests/tokenizer_benchmark.c reports:
- training time and MiB/s;
- actual vocabulary and merge count;
- arena peak memory;
- average encode time and MiB/s;
- resulting tokens and bytes per token.

Run the same corpus prefix, sample length, target vocabulary, and repetition count against baseline and v2. Record small, medium, and large cases. Do not accept the heap trainer solely because its Big-O looks better: compare wall-clock time and peak memory. Likewise, tune the short/long encoder threshold from measured results.

Correctness gates:
- exact byte round-trip for all 256 values;
- exhaustive short-sequence comparison against the old reference trainer and encoder;
- deterministic results across repeated runs;
- overlapping pairs such as AAA, AAAAA, and repeated mixed patterns;
- empty and one-byte input;
- all maximum-token-length boundaries;
- no cross-piece merges in pretokenized mode;
- BPE1 load/save compatibility and BPE2 metadata validation;
- truncated/corrupt file rejection.

## Deliberate non-goals for the first v2 pass

- Do not replace BPE with SentencePiece Unigram.
- Do not copy SentencePiece normalization into byte-level Minima.
- Do not add a heavyweight regex dependency just to imitate tiktoken's exact pattern.
- Do not parallelize the merge loop before the incremental single-threaded algorithm is proven correct. Parallelize initial counting only if benchmarks show it is worth the complexity.
- Do not add an LRU word cache until repeated-piece workloads show encoding is cache-bound.

## Source snapshots reviewed

The design above is based on the source files linked in the three sections. Upstream implementations evolve; the exact revisions reviewed are recorded below.

### Reviewed source revisions

The source files were fetched at these Git object revisions during this review:

- tiktoken educational encoder: 317e77560c2b95c2c8a7b3d0d170b2af6de239c9
- tiktoken Rust core: ea54eac8c54fe1d9655d9183dbdd161d90a748c8
- Hugging Face BPE trainer: 97a05e64a9d86eeb346cd4a8d85d7d8eea0d84ca
- Hugging Face BPE word representation: 2ae19dbbd459cb6db3c59a418069686beaefdc40
- Hugging Face BPE inference model: 5fa02b766b39dd747a4ed15130d9fb4f63d44163
- SentencePiece BPE trainer: bb0e252c1efd34a039eb80a5b8e3d8711efc4b22
- SentencePiece BPE inference model: bedc1ff93242a6210a662c62f7cb03d2e037549c

