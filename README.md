# Minima

A lightweight, CPU-first language-model project written in C.

## Project roadmap

This is the central working backlog for Minima: planned features, known issues, technical debt, experiments, and longer-term research directions. It is intentionally broader than the immediate next release. Items are not release promises.

Keep this document aligned with the code. A feature being discussed or partially implemented is not the same as being complete and verified.

### Status key

- [ ] **Planned** — identified, not yet implemented.
- [~] **In progress / partial** — implementation exists, but work or verification remains.
- [x] **Done** — implemented and verified. Keep regression tests where appropriate.
- [?] **Investigate** — requires research, measurement, or a design decision.
- [>] **Deferred** — intentionally postponed; include the reason.

---

## Hardware target: Lenovo Ryzen 7 5700U / 16 GiB RAM

The first known development target is a Lenovo laptop with:

- AMD Ryzen 7 5700U (Lucienne, Zen 2): 8 physical CPU cores / 16 logical threads, 1.8 GHz base and up to 4.3 GHz boost; AMD lists a 15 W default TDP and configurable 10–25 W range.
- 16 GiB DDR4 installed as two 64-bit memory channels (reported as 2 x 64-bit). The exact operating memory speed should be confirmed in Windows; the processor supports DDR4-3200.
- Integrated AMD Radeon graphics using shared system memory.
- Windows 11 and Visual Studio 2022 / MSVC are the primary build environment.

This is a useful CPU-first target, not a reason to tune the whole engine around one benchmark. The 5700U is a low-power laptop processor: sustained clocks, cooling, power mode, and memory pressure matter more than its short boost clock. Keep the integrated GPU out of the critical path until a separate GPU backend is justified and measured.

### Initial operating envelope (provisional)

| Setting | Starting point | How to validate |
|---|---:|---|
| OpenMP workers | 8 | Compare 4, 8, 12, and 16 using identical Release builds and workloads |
| Engine memory budget | 8 GiB | Measure peak committed/private bytes while Windows and the GUI are running |
| Default model | Existing 5.7M-parameter configuration | Keep architecture fixed during the first benchmark |
| Training batch / sequence | Existing 4 x 128 | Benchmark alternatives only after a baseline is recorded |
| Power | Plugged in, Windows best-performance mode for benchmark runs | Record power mode and temperatures; repeat a normal-use run separately |
| GPU | CPU only | Revisit only with a measured, maintainable backend |

These are starting limits, not hard-coded truths. Do not automatically use all 16 logical threads: SMT threads share physical-core resources, and a 15 W laptop may become slower after it heats up. The CLI already accepts `--threads N`; the GUI does not yet expose that option. OpenMP is enabled in the Visual Studio project files. Current parallel regions cover selected matrix-multiplication work in the forward/backward paths, not every operation, so thread-count scaling must be measured rather than assumed.

### Memory implications

The engine currently reserves an 8 GiB permanent arena, an 8 GiB corpus arena for training, and a 256 MiB temporary arena. On Windows these arenas reserve virtual address space and commit pages as allocations are made; the reservation sizes are not the same as immediate physical RAM consumption. The corpus is nevertheless loaded into memory, and token IDs use 16-bit storage. A 906 MiB text corpus can therefore require roughly another 1.77 GiB for token IDs in a worst-case byte-per-token estimate, before tokenizer, model, optimizer, activations, and Windows overhead.

The current code also rejects corpus byte/token lengths above `INT_MAX`. Consequently, the roadmap's 1–5 GiB corpus goal is not yet achievable as a single ordinary training input: the upper end requires a streaming/chunked corpus and tokenization design, plus removal of the signed-int length bottleneck. Do not simply increase arena reservations or remove the guard. First measure the actual working set and design the data path.

### First hardware-tuning pass

1. Build **Release | x64** and run the same fixed training workload at 4, 8, 12, and 16 OpenMP threads. Do not use Debug numbers.
2. For each run, record commit, model/corpus fingerprints, batch, sequence, optimizer, worker count, elapsed time, milliseconds per optimizer step, tokens/second, peak process memory, and thermal/power conditions.
3. Repeat the best two settings for long enough to expose thermal throttling. Select the fastest sustainable setting, not the fastest first minute.
4. Add the selected worker count and a memory-budget setting to the GUI only after the benchmark demonstrates a useful default. Keep the CLI override.
5. Add stage profiling before attempting SIMD, blocked matrix multiplication, or GPU work. The current code parallelizes only selected kernels; profiling will show whether attention, optimizer, corpus/tokenizer work, or memory traffic is limiting the whole step.

The first goal is a reliable performance baseline on this laptop. Model width, layer count, vocabulary, and corpus size should not be changed during that baseline experiment.

## 1. Immediate priorities: correctness and CI

These tasks protect the reliability of the project. Avoid changing model layout or adding performance complexity while the test suite is failing.

| ID | Task | Status / notes |
|---|---|---|
| A1 | Re-verify Windows CI tests against the current RoPE + GQA layout; remove stale assumptions about learned position embeddings (`pos_emb`, `m_pos_emb`, `v_pos_emb`). | [~] Most changes are in place. Run and verify the full suite. |
| A2 | Keep `gradcheck_test.c` and `gradients_init_test.c` tensor counts aligned with the current 9-parameter layout. | [x] Updated in prior work; retain regression coverage. |
| A3 | Keep smoke-test expectations aligned with the current gradient-check count (9 / 9, not 10 / 10). | [x] Updated. |
| A4 | Verify resume LR-schedule metadata, including the explicit creation-time `--lr 0.0003` case and restoration of checkpoint metadata. | [x] Fixed; watch for regressions. |
| A5 | Keep the `--max-seq` / `--sequence-length` consistency guard and ensure smoke tests pass valid matching values. | [x] Fixed by supplying `--sequence-length 8` in the affected test. Do not weaken the guard. |
| A6 | Make exact-vocabulary preflight diagnostics distinguish an invalid tokenizer from a valid tokenizer that stopped before the requested vocabulary size. | [~] Policy is deliberate; finish diagnostic clarity. |
| A7 | Separate the model's supported configuration envelope from what is comfortable on the current machine. Include weights, gradients, AdamW state, activations, tokenizer, and corpus in startup estimates. | [~] Combined budget reporting needs refinement. |

**Recommended sequence:** verify A4 and A5, run the complete Windows CI suite, then finish A6 and A7.

## 2. Memory, training core, and performance

| ID | Task | Status / notes |
|---|---|---|
| B1 | Reuse the logits buffer in-place for cross-entropy/backpropagation after the max, sum, and loss calculations. | [x] Implemented in the backward contract and training path, with equivalence tests. The original estimate was about 58 MiB saved at batch 4 / sequence 128 / vocabulary 30k; re-measure for other configurations. |
| B2 | Replace source-only validation splitting with deterministic document-aware holdout. Keep each complete document on only one side and record the split in a manifest. | [ ] Not implemented. Current source-wise 95/5 split and boundary isolation are useful, but do not guarantee document-level separation within a source. |
| B3 | Add selective activation recomputation if memory remains a limiting factor after existing buffer reuse. | [>] Later; more invasive and should be justified by measurements. |
| B4 | Add structured stage profiling for attention, matrix multiplication, output projection, backward pass, optimizer, tokenizer, and validation. | [ ] Not present as a complete profiling harness. |
| B5 | Evaluate blocked and/or SIMD matrix multiplication while retaining the scalar implementation as the correctness reference. | [ ] Only after B4 identifies a meaningful bottleneck. |
| B6 | Measure optimizer cost and memory. Evaluate AdamW moment-loop costs and the existing Lion implementation; expose optimizer choice in the GUI when ready. | [~] Lion already exists in the engine; GUI exposure and comparative measurements remain. |

**Order:** confirm B1 on a local build, implement B2, add B4, then use actual profiles to decide whether B5 or B6 is worthwhile.

## 3. Tokenizer and data quality

| ID | Task | Status / notes |
|---|---|---|
| C1 | Keep model vocabulary limits and tokenizer capacity at 32,768, with a default target of 30,000. | [x] Implemented. |
| C2 | Use a 4 MiB spread sample for tokenizer rebuilds rather than sampling only the beginning of a corpus. | [x] Raised. |
| C3 | Build repeatable vocabulary-size experiments at 512 / 1k / 2k / 4k / 8k / 16k / 30k using the same corpus. Record bytes per token, BPB, train/validation loss, step time, and representative generation quality. | [ ] Framework needed. Raw loss values across different tokenizers are not directly comparable. |
| C4 | Expand frequency-aware construction, rare-token coverage, and unused-vocabulary diagnostics. | [~] Some diagnostics exist; coverage and reporting need work. |
| C5 | Grow the corpus toward 1–5 GiB using carefully selected FineWeb-Edu / Cosmopedia slices, modern Wikipedia, OpenStax, Standard Ebooks, conversational datasets, and curated chat logs. | [ ] Data pipeline and curation work. |
| C6 | Clean Gutenberg and OCR boilerplate; prefer clean public-domain editions such as Standard Ebooks where suitable. | [ ] Data hygiene task. |
| C7 | Add per-source or per-document weighting beyond the current proportional, square-root, and equal mix modes. | [ ] Desired; current mix modes are limited. |
| C8 | Make document-boundary detection explicit and format-aware. Avoid relying on blank-line heuristics where source markers or format-specific parsing are available. | [ ] Tied to B2. |

### Corpus planning notes

An initial mix to experiment with is approximately 40% narrative/fiction, 30% modern nonfiction and essays, 20% reference/textbooks, and 10% conversational data. These are hypotheses, not fixed targets; measure their effects.

For every dataset, preserve provenance: source name, license, origin URL, retrieval date, preprocessing steps, byte count, document count, and checksum. Keep raw source material separate from cleaned training output where practical.

The assembled Britannica text is approximately 906 MiB. It was merged from 195 source files and verified for byte preservation and UTF-8 decoding, but it has not yet received OCR/boilerplate cleanup. Do not treat it as training-ready merely because it is large.

## 4. Model architecture and scale

The current default configuration is approximately 5.74 million parameters:

- Vocabulary: 30,000
- Model dimension: 128
- Layers: 8
- Query heads: 4
- Feed-forward dimension: 512
- Maximum sequence length: 128

Confirm parameter counts from the actual configuration and model implementation when configurations change.

| ID | Task | Status / notes |
|---|---|---|
| D1 | Preserve weight tying, RMSNorm, SwiGLU, RoPE, and GQA as core architecture choices. | [x] Present in the current architecture. Keep tests and documentation aligned. |
| D2 | Expose advanced architecture controls such as KV-head count, RoPE base/scaling, initialization method/scale, and normalization epsilon. | [~] Mostly internal today; decide which controls are safe and useful to expose. |
| D3 | Add post-training int8 and int4 weight quantization for inference. Measure quality loss, RAM reduction, and CPU speed. | [ ] Not started; potentially high value for memory-constrained inference. |
| D4 | Preserve the pre-allocated, flat KV-cache buffers. | [x] Current design; avoid unnecessary allocation churn. |

Earlier model-size milestones remain useful experiment points: roughly 205k, 1M, and the current 5.7M-class model. Prefer a wider vocabulary with a capable narrow core over increasing vocabulary at the expense of an excessively narrow model, unless measurements show otherwise.

## 5. GUI, resource management, and user experience

### 5.1 Resource management and chat control — high priority

| ID | Feature | Status / notes |
|---|---|---|
| E1.1 | Clear Chat: reset the engine-side conversation state, not merely the GUI textbox. | [ ] |
| E1.2 | Separate Clear Display from Clear Saved Chat Log. | [ ] These must remain distinct operations. |
| E1.3 | Configurable RAM allocation limit in MB/GB, enforced as an internal budget with graceful failure. | [ ] |
| E1.4 | CPU utilization limit, preferably enforced with Windows Job Objects rather than relying only on thread count. | [ ] |
| E1.5 | Expose the existing CLI thread-count selector in the GUI. | [ ] |
| E1.6 | Live RAM, CPU, and peak-resource monitoring with configurable warning thresholds. | [ ] |
| E1.7 | Resource profiles: Eco, Balanced, Performance, Training, and Chat. | [ ] |
| E1.8 | Persistent status bar showing mode, RAM used/limit, CPU used/limit, thread count, and tokens/second. | [ ] |

### 5.2 Configuration surface

| ID | Feature | Status / notes |
|---|---|---|
| E2.1 | Expose optimizer selection (AdamW / Lion), weight decay, and advanced beta1/beta2/epsilon settings where applicable. | [ ] |
| E2.2 | Allow individual corpus-source weights, not only global mix modes. | [ ] |
| E2.3 | Add generation controls: min-p, frequency/presence penalties, no-repeat n-gram, sampling seed, stop sequences, and minimum new tokens. | [ ] |
| E2.4 | Configuration profiles with save, load, export, and import. | [ ] |
| E2.5 | Show a pre-training memory estimate covering weights, gradients, optimizer state, activations, tokenizer, and corpus; compare it with the configured resource budget. | [~] Related startup budget estimates exist but need a clearer combined presentation and budget integration. |

### 5.3 Training monitor

| ID | Feature | Status / notes |
|---|---|---|
| E3.1 | True pause/resume without terminating the engine process. | [ ] |
| E3.2 | Show ETA, tokens processed, tokens/second, current learning rate, and best-checkpoint status. | [ ] |
| E3.3 | Automatic checkpoint rotation with keep-best behavior. | [ ] |
| E3.4 | Compare training runs using recorded configurations and metrics. | [ ] |

### 5.4 Generation and chat

| ID | Feature | Status / notes |
|---|---|---|
| E4.1 | Generate multiple responses from one prompt and compare them side by side. | [ ] |
| E4.2 | Add token-probability inspection, generation history, and reusable prompt templates. | [ ] |
| E4.3 | Save, load, and export conversations. | [ ] |
| E4.4 | Implement proper EOS and stop-sequence handling. Reserved tokens are currently suppressed; generation should use explicit stopping semantics. | [ ] |

### 5.5 Diagnostics and organization

| ID | Feature | Status / notes |
|---|---|---|
| E5.1 | Reorganize GUI tabs into Training, Chat, Settings, Resources, and Diagnostics. | [ ] |
| E5.2 | Add corpus analysis for sizes, duplicates, encoding, document counts, and source contribution statistics. | [ ] |
| E5.3 | Add model/checkpoint integrity checks, fingerprints, and compatibility validation. | [ ] |
| E5.4 | Record structured benchmark results: commit, dimensions, vocabulary, corpus fingerprint, losses, peak RAM, milliseconds/step, tokens/second, and fixed-prompt generations. | [ ] |

## 6. Practical implementation roadmap

### Phase 0 — Safety net

1. Verify the LR-resume and sequence-length smoke-test cases (A4/A5).
2. Run the complete Windows CI suite and resolve any stale architecture assumptions.
3. Finish tokenizer preflight diagnostics (A6) and the supported-versus-comfortable resource estimate (A7).

Do not make model-layout changes while tests are red.

### Phase 1 — Measurement quality

1. Implement deterministic document-aware validation splits and manifests (B2/C8).
2. Build the vocabulary sweep harness with BPB and bytes-per-token reporting (C3).
3. Add stage-level timing instrumentation (B4).

### Phase 2 — Resource and chat UX

1. Implement engine-side Clear Chat and separate display/log clearing (E1.1/E1.2).
2. Add resource budgets, CPU limits, monitoring, profiles, thread selection, and a persistent status bar (E1.3–E1.8).
3. Integrate the memory estimator with the resource budget (E2.5).
4. Expose optimizer selection and relevant weight-decay controls (E2.1).

### Phase 3 — Training and generation ergonomics

1. Add training monitor improvements and pause/resume (E3).
2. Improve EOS and stop-sequence behavior and add generation controls (E2.3/E4).
3. Add configuration profiles and per-source corpus weights (E2.2/E2.4).

### Phase 4 — Scale and efficiency

1. Expand and clean the corpus toward 1–5 GiB (C5/C6).
2. Evaluate SIMD/blocked kernels and optimizer alternatives only when profiling supports the work (B5/B6).
3. Add inference quantization (D3).
4. Consider selective activation recomputation only if memory remains a demonstrated bottleneck (B3).

### Phase 5 — Research directions

Explore adaptive or hierarchical segmentation, SuperBPE-style vocabulary construction, and other tokenizer/model research only after the measurement and benchmark infrastructure is reliable.

---

## 7. Completed work: do not re-implement

This is a compact reminder of important work already present. Reopen the relevant code/tests if behavior is in doubt, but do not add a second implementation simply because an old note still exists.

- In-place logits/gradient buffer reuse for cross-entropy and backpropagation, with equivalence tests.
- Source-wise 95/5 validation holdout and existing boundary isolation. This is not the same as the still-planned document-aware split in B2.
- RoPE, GQA, SwiGLU, RMSNorm, and weight tying.
- Vocabulary capacity of 32,768, default target of 30,000, and 4 MiB spread tokenizer sampling.
- The approximately 5.7M-parameter default architecture.
- Gradient and initialization test updates for the current 9-parameter layout.
- Smoke-test updates for the 9/9 gradient-check count.
- LR-schedule resume fix using explicit creation-time learning-rate configuration.
- Sequence-length smoke-test correction while retaining the runtime guard.
- Existing diagnostics, including token frequency, generation entropy, and JS divergence.

---

## 8. Roadmap maintenance rules

- Keep this file as the project-level source of truth for planned work, known issues, and accepted research directions.
- Update an item's status when implementation begins, lands, or is verified. Do not mark something Done merely because code exists; run the relevant tests or benchmark.
- Add newly discussed ideas here before they disappear into chat history.
- Record important design decisions and the reason for them, especially when rejecting or deferring an idea.
- Keep detailed implementation notes in code, tests, or focused issue discussions; the README should describe the goal and current state, not every coding step.
- When an item is completed, retain it in the completed-work section if it is an important architectural or behavioral milestone. Otherwise remove it from the active backlog.
- Prefer measurements over assumptions for tokenizer, architecture, quantization, corpus composition, and performance decisions.
