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

## Product scope: text-only conversational reasoning

Minima is intentionally a **text-only** language model. This is a firm product boundary, not a temporary omission.

### In scope

- Natural text conversation and coherent multi-turn context.
- Following straightforward instructions and answering clearly.
- Basic logical reasoning: deduction, simple conditionals, classification, ordering and constraint problems, consistency checks, causal reasoning, and identifying when the available information is insufficient.
- Explaining a conclusion in a short, understandable chain of reasoning.
- General language understanding and useful everyday knowledge, with appropriate uncertainty when facts are unknown.

### Explicitly out of scope

- Audio input/output, speech recognition, music, image understanding/generation, video, and other multimodal capabilities.
- Teaching Minima to write, debug, or execute code.
- Advanced mathematics, symbolic algebra, long calculations, or specialist quantitative problem solving.
- Autonomous execution of arbitrary programs or shell commands.

Do not add multimodal encoders, media pipelines, code-generation training objectives, or complex-math subsystems. Do not spend model capacity or development time on them.

### Design consequences

- Keep the model and runtime text-only. Text documents and text-based retrieval may be considered later, but no media ingestion is planned.
- Prioritize high-quality conversational and reasoning examples over sheer corpus size.
- Evaluate reasoning behavior separately from next-token loss. A lower language-model loss does not prove that Minima can reason correctly.
- Teach the model to distinguish facts, assumptions, deductions, and uncertainty. It should not invent missing premises to force an answer.
- Prefer concise, transparent reasoning over elaborate chain-of-thought imitation. Training targets should show enough explanation to justify the answer without rewarding rambling.
- Treat tool calling, RAG, persistent memory, and few-shot retrieval as optional text-only runtime enhancements. They are not prerequisites for the core reasoning goal.
- Keep the C inference engine small and portable. Do not turn Minima into an agent framework.

**Primary success criterion:** given a short text problem and its stated premises, Minima should identify what follows, what does not follow, and what cannot be determined, then explain the distinction accurately.

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

## 0. First behavioral milestone: measure before teaching

Before changing model architecture or launching a large fine-tuning run, establish a small, held-out reasoning evaluation set. This is the first priority because current training loss and fluent-looking generation do not measure whether Minima can solve logic questions.

- [x] Create an initial text-only evaluation set covering deduction, conditionals, constraints/order, classification, causal reasoning, consistency, simple counting, ambiguity, and insufficient-information cases. See `data/evaluation/logic.jsonl` (24 cases).
- [x] Keep evaluation prompts and reference answers out of all training/tokenizer corpora. They remain held out.
- [x] Add a deterministic runner that sends only prompts to Minima and records model/tokenizer fingerprints. See `tools/evaluate_logic.py` and `data/evaluation/RUNNING.md`.
- [ ] Record and save the first baseline from the current checkpoint. The runner is ready, but no model outputs have been collected yet.
- [ ] Score conclusions and explanations separately with a simple, documented rubric. Do not rely only on exact string matching.
- [ ] Add a separate curated reasoning-training dataset only after the baseline exists. Keep its format and provenance documented.
- [ ] Compare every training change against the same held-out set and retain regressions as well as improvements.

The initial benchmark is a behavioral smoke test, not a claim of broad reasoning ability. Expand it gradually and include paraphrases so Minima cannot pass by memorizing question wording.

---

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
| E1.1 | Clear Chat: reset the active engine conversation and clear the visible and saved transcript. | [x] Implemented with the chat-tab Clear chat control and the engine `/clear` command. |
| E1.2 | Separate the Chat transcript from engine Logs, with independent Clear chat and Clear logs controls. | [x] Chat history and engine output now have separate tabs and clear actions. |
| E1.3 | Configurable RAM limit with graceful internal allocation budgeting. | [~] A hard Windows Job Object process-group committed-memory ceiling is exposed in GiB and persisted; it is not a physical working-set cap. Engine-aware soft budgeting and graceful allocation diagnostics remain. |
| E1.4 | CPU utilization limit, preferably enforced with Windows Job Objects rather than relying only on thread count. | [x] Hard CPU-rate cap (1-100% of total logical-processor capacity, 0=off) is applied to engine and evaluation process groups before execution. |
| E1.5 | Expose the existing CLI thread-count selector in the GUI. | [x] Implemented; 0 selects the runtime default. |
| E1.6 | Live RAM, CPU, and peak-resource monitoring with configurable warning thresholds. | [~] Live process-group CPU and committed-memory readings plus the directly launched process working set are displayed; peak tracking and warning thresholds remain. |
| E1.7 | Resource profiles: Eco, Balanced, Performance, Training, and Chat. | [ ] |
| E1.8 | Persistent status bar showing mode, RAM used/limit, CPU used/limit, thread count, and tokens/second. | [~] Resource readings and configurable limits are visible in the Monitor panel; a persistent cross-page status bar and tokens/second remain. |

### 5.2 Configuration surface

| ID | Feature | Status / notes |
|---|---|---|
| E2.1 | Expose optimizer selection (AdamW / Lion), weight decay, and advanced beta1/beta2/epsilon settings where applicable. | [~] Optimizer selection and weight decay are exposed; beta1/beta2/epsilon controls remain. AdamW now excludes RMSNorm scales from decoupled weight decay. |
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
| E4.5 | Run the held-out 24-question logic evaluation from the GUI, sequentially logging each isolated answer and preserving partial results. | [x] GUI launcher and streaming JSONL result output implemented; requires Python 3 via the Windows `py` launcher. |

### 5.5 Diagnostics and organization

| ID | Feature | Status / notes |
|---|---|---|
| E5.1 | Reorganize GUI tabs into Training, Chat, Settings, Resources, and Diagnostics. | [ ] |
| E5.2 | Add corpus analysis for sizes, duplicates, encoding, document counts, and source contribution statistics. | [ ] |
| E5.3 | Add model/checkpoint integrity checks, fingerprints, and compatibility validation. | [ ] |
| E5.4 | Record structured benchmark results: commit, dimensions, vocabulary, corpus fingerprint, losses, peak RAM, milliseconds/step, tokens/second, and fixed-prompt generations. | [ ] |

## 6. Conversation, memory, retrieval, tools, and safety architecture

This section records the intended direction for Minima's interaction/runtime architecture. These are design requirements and roadmap items, **not claims that the features already exist**. Keep the implementation incremental and preserve a small, dependable C inference core.

### 6.1 Current interaction baseline

The current chat path is deliberately simple:

- Training conversation data uses only `Human:` and `Assistant:` labels. See `data/README.md`.
- Interactive chat builds a plain-text prompt from the retained conversation plus the latest `Human:` turn and an `Assistant:` generation prefix.
- The console chat implementation currently keeps conversation history in a fixed-size in-process buffer. It removes the oldest complete user turn when the prompt no longer fits, and may shorten an oversized current user message.
- Chat prompt encoding is constrained by the model's configured context length. The current default context is 128 tokens; inference and KV-cache workspaces have implementation limits documented in `data/README.md` and enforced in code.
- The KV cache is temporary inference state. It is reset/reused for generation and is not persistent conversation memory.
- Generation currently suppresses the four reserved token IDs because the training pipeline does not yet teach their intended conversational semantics. EOS is not currently an active stopping marker.
- Saved conversations, long-term user memory, retrieval-augmented generation (RAG), structured tool calls, and a general safety/policy layer are not yet implemented as complete systems.

Keep the existing Human/Assistant format working until a versioned message format and migration path are implemented. Do not silently add a `System:` role to the current training corpus and assume the model understands it.

### 6.2 Canonical message protocol

**Planned:** replace ad hoc prompt-string construction with one canonical message representation shared by training-data preparation, inference, conversation persistence, and tool interaction.

The conceptual roles are:

- `system`: high-priority runtime behavior and policy instructions.
- `developer`: application-level instructions and task configuration.
- `user`: the human's input.
- `assistant`: model-generated content, including structured tool requests.
- `tool`: results returned by an explicitly registered tool.

A message should carry at least a role, content, and an explicit ending/type where needed. The protocol must distinguish an ordinary assistant message, an assistant turn-final message, a refusal, and a tool-call request. Tool replies must be associated with the call that produced them.

Requirements:

1. Define a versioned message structure and canonical serializer/parser. Use the same implementation wherever prompts are assembled or conversations are saved/loaded.
2. Introduce dedicated reserved token IDs for structural markers and train the model on their actual semantics. Do not rely on visible strings that merely look like special tokens.
3. Define unambiguous message boundaries and end-of-message, end-of-turn, and handoff-to-tool semantics. EOS/end-of-turn behavior must be tested, not inferred from text.
4. Preserve UTF-8 content exactly through serialization and round trips.
5. Version the format and provide compatibility handling for existing Human/Assistant chat logs and saved conversations.
6. Keep raw text continuation in Generation Lab separate from structured conversational inference.

The implementation is text-only. Multimodal payloads and media support are explicitly out of scope, not future extensions.

### 6.3 Context assembly and budgeting

**Planned:** add a context manager that assembles the model input in a predictable order and accounts for the actual tokenized size of every component.

Candidate context components:

1. System instructions.
2. Developer/application instructions.
3. Relevant persistent user preferences or memory.
4. A small set of selected few-shot examples, when useful.
5. Retrieved document/context passages, when relevant.
6. Recent conversation messages.
7. The current user message.
8. An assistant-generation prefix and a reserved output-token budget.

Requirements:

- Tokenize the assembled context before generation and enforce the model's real context limit.
- Reserve output capacity before accepting optional context. Never let retrieved text or examples consume the entire generation budget.
- Preserve message boundaries and instruction priority when trimming.
- Prefer removing low-relevance retrieved passages and examples first; then summarize or remove older conversation turns while preserving recent turns and the current request.
- Never silently truncate the middle of a message or corrupt UTF-8. If the current request alone is too large, report the limit clearly or apply an explicit, visible truncation policy.
- Record which context components were included and their token costs in diagnostics.
- Make context assembly deterministic where possible so prompt bugs can be reproduced.

A future context manager may summarize older history, retrieve relevant past turns, and use recency plus relevance. Summaries must be treated as lossy derived context, not as an exact replacement for the original conversation.

### 6.4 Memory model and persistence

Keep these concepts separate; they solve different problems:

| Memory layer | Purpose | Lifetime / storage |
|---|---|---|
| Active context | Messages and context currently supplied to the model | One generation / current context window |
| KV cache | Cached attention keys and values for incremental decoding | Temporary; reset between unrelated prompts |
| Conversation history | Exact user and assistant messages | Persistent, per conversation |
| Long-term user memory | Small, useful preferences or facts that can improve future interactions | Persistent, separately managed records |
| Knowledge documents | External reference material used through retrieval | Persistent document store and index |

**Planned requirements:**

- Persist conversations independently of model weights. SQLite is a candidate for a small local-first store; keep the storage interface replaceable.
- Store message role, content, order, timestamps, conversation ID, format version, and relevant metadata.
- Keep long-term memory distinct from raw chat logs. Do not automatically promote every conversation detail into permanent memory.
- Provide user-visible controls to inspect, edit, disable, and delete long-term memories and conversations.
- Make deletion meaningful: remove the stored record and its derived indexes/summaries, not merely hide it from the UI.
- Keep memory retrieval selective and token-budgeted. A memory item is context, not a new instruction with higher priority than system/developer policy.
- Do not bake private user memory or changing external knowledge into model weights as the default persistence mechanism.

The model's weights represent learned general patterns; they are not a reliable database of a particular user's current preferences or conversation history.

### 6.5 Function and tool calling

**Planned:** support controlled, structured tool use through an application-owned registry. Tool execution must remain outside the model's arbitrary text-generation process.

The intended cycle is:

1. The runtime selects the tools available for the current task and supplies concise descriptions plus argument schemas in context.
2. The model emits a structured tool-call request naming a registered tool and providing arguments.
3. The runtime parses the request and validates the tool name, argument types, required fields, bounds, and permissions.
4. The application executes only the registered implementation.
5. The result is appended as a `tool` message associated with the call.
6. The model resumes with the tool result in context and either makes another permitted call or responds to the user.

Requirements:

- Use an allowlisted registry. A model-generated name or argument must never cause arbitrary C functions, shell commands, scripts, or executable code to run.
- Validate every argument at the runtime boundary, even if the model was trained to produce valid calls.
- Apply per-tool permissions, timeouts, output-size limits, and resource limits.
- Treat tool output as untrusted data. It may contain malicious instructions or malformed content.
- Require explicit user confirmation for consequential or externally visible actions, such as deleting files, sending messages, making purchases, or changing system settings.
- Keep tool execution and platform-specific integrations behind interfaces so the core model remains portable.
- Log tool name, validated arguments, result status, and errors without unnecessarily recording secrets.
- Add deterministic tests for valid calls, malformed arguments, unknown tools, denied permissions, timeouts, and tool failures.

Start with a few harmless local tools (for example, calculator and read-only time/date utilities). File access, network access, process execution, and write operations require stricter permission design. Never treat prompt instructions as a security boundary.

### 6.6 Retrieval-augmented generation (RAG)

**Planned:** let Minima consult external documents at runtime instead of expecting all factual knowledge to fit in its weights.

Initial pipeline:

1. Ingest documents and preserve provenance (source, title, origin/path, license where applicable, retrieval date, and document metadata).
2. Extract text and normalize encoding/formatting without discarding meaningful structure.
3. Split content into bounded chunks, preferably respecting headings, paragraphs, and document boundaries. Store chunk-to-document references and page/section metadata where available.
4. Build a lexical index. BM25 is a sensible first implementation because it is useful without requiring a separate embedding model.
5. Retrieve candidate chunks for the current query, rank them, and select only those that fit the context budget.
6. Add selected passages as clearly delimited reference context, separate from system/developer instructions.
7. Preserve source references so the assistant can identify which documents support its answer and say when the available material does not support a claim.

Later experiments may add local embeddings, vector search, hybrid lexical/vector retrieval, and reranking. Measure retrieval quality before adding complexity.

Security and quality requirements:

- Retrieved documents are untrusted reference data, never instructions that override system/developer policy.
- Defend against prompt injection in documents and tool output. Delimit quoted/retrieved material and instruct the model/runtime to treat it as data.
- Avoid presenting unsupported guesses as sourced facts. Support abstention when retrieval is weak or conflicting.
- Deduplicate near-identical chunks and avoid flooding context with overlapping passages.
- Keep document storage and indexes separate from conversation memory, and provide a way to rebuild indexes when chunking or ranking changes.

### 6.7 Few-shot examples

**Planned:** maintain a small library of curated examples for tasks where a demonstration improves format-following or tool-use behavior.

- Select examples by task or capability, not by blindly appending a large static block to every prompt.
- Keep examples short, high quality, and consistent with the canonical message/tool protocol.
- Include examples for structured outputs, tool-call/response cycles, clarification, refusal, and citing retrieved sources as those capabilities are implemented.
- Count example tokens against the same context budget as all other context.
- Keep examples versioned and testable. Remove examples that teach obsolete formats or unsafe behavior.
- Distinguish runtime few-shot prompting from supervised fine-tuning: examples in context guide one request; SFT changes model parameters.

### 6.8 Safety and trust boundaries

**Planned:** use defense in depth. A prompt or keyword filter alone is not a security system.

Safety responsibilities should be split across layers:

- **Input boundary:** validate encoding, lengths, malformed protocol data, and resource limits. Apply policy checks where appropriate.
- **Instruction hierarchy:** preserve system/developer/user/tool roles and prevent lower-trust content from being reinterpreted as higher-priority instructions.
- **Untrusted context:** clearly mark retrieved documents, user-provided files, and tool results as data. Guard against prompt injection and instruction spoofing.
- **Tool boundary:** enforce allowlists, schemas, permissions, path restrictions, timeouts, output limits, and confirmation requirements in application code.
- **Output boundary:** validate structured outputs before use; safely render text and escape content where needed. A refusal marker or safety phrase generated by the model is not itself enforcement.
- **Resource boundary:** limit context size, generation length, memory, CPU time, and tool execution to protect the host application.
- **Auditability:** record useful diagnostic events while minimizing sensitive data collection.

Use policy tests and adversarial cases as regression tests. Keyword/rule-based filters may be a supplemental signal, but should not be the only safeguard and should not be mistaken for model understanding.

### 6.9 Training implications

The runtime architecture and model training format must evolve together.

- Keep general language pretraining separate from conversational supervised fine-tuning (SFT).
- Build SFT examples using the canonical message serialization, including the roles and structural tokens the runtime actually uses.
- During SFT, calculate loss on the intended assistant targets. Normally mask system, developer, user, and tool-result tokens from the assistant-response objective; define and document any deliberate exceptions.
- Include examples of complete multi-turn conversations, assistant turn endings, refusals, and tool handoffs/results once those behaviors exist in the runtime.
- Tool schemas and tool-use demonstrations must agree with the runtime registry and validation rules.
- Evaluate on held-out conversations and tasks, not only training loss. Track instruction following, response relevance, formatting validity, tool-call validity, retrieval grounding, refusal behavior, and context handling.
- Preference optimization (for example, DPO) is a later experiment after the SFT data and evaluation set are reliable. More advanced reward-based methods are research directions, not prerequisites for a useful local assistant.
- Do not assume RAG or persistent memory requires retraining the base model; these are primarily runtime capabilities.

### 6.10 Suggested implementation sequence

Keep the model/inference engine independent from the GUI, persistence, retrieval, and external integrations. The GUI, CLI, and any future local API should call the same runtime interfaces.

| Phase | Deliverable | Main work |
|---|---|---|
| I1 | Behavioral baseline | Held-out text-only logic evaluation, fixed baseline prompts/decoding, scoring rubric, saved outputs; no training on evaluation data |
| I2 | Reasoning-focused training data | Curated examples for basic deduction, conditionals, constraints, causal reasoning, uncertainty, and concise explanations; quality review and deduplication |
| I3 | Conversational training foundation | Versioned text message format, explicit turn endings, compatible chat serialization, assistant-target SFT and loss masking |
| I4 | Context and conversation reliability | Context budgeting, robust multi-turn history, save/load, clear/reset, separate conversation history from KV cache |
| I5 | Optional text retrieval and memory | Only if useful after core reasoning improves; local text documents, provenance, user-controlled memory, token-budgeted retrieval |
| I6 | Safety and reliability | Instruction hierarchy, untrusted text handling, input/output validation, safe runtime boundaries, refusal and uncertainty behavior |
| I7 | Behavioral refinement | Few-shot examples and preference tuning only when evaluation shows a clear need; continue held-out logic and conversation tests |
| I8 | Optional text-only extensions | Simple allowlisted tools or advanced text retrieval only when they directly support the intended assistant; no arbitrary code execution |

**Architecture rule:** keep the core C model responsible for tokenization, inference, and training. Put message orchestration, persistence, tool execution, retrieval, and policy enforcement in explicit runtime modules with narrow interfaces. Do not turn the inference engine into a monolith or let model output directly execute host operations.

---

## 7. Practical implementation roadmap

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

1. Extend chat/log UX with export and conversation management after the current separate tabs and clear controls (E1.1/E1.2).
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

## 8. Completed work: do not re-implement

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

## 9. Roadmap maintenance rules

- Keep this file as the project-level source of truth for planned work, known issues, and accepted research directions.
- Update an item's status when implementation begins, lands, or is verified. Do not mark something Done merely because code exists; run the relevant tests or benchmark.
- Add newly discussed ideas here before they disappear into chat history.
- Record important design decisions and the reason for them, especially when rejecting or deferring an idea.
- Keep detailed implementation notes in code, tests, or focused issue discussions; the README should describe the goal and current state, not every coding step.
- When an item is completed, retain it in the completed-work section if it is an important architectural or behavioral milestone. Otherwise remove it from the active backlog.
- Prefer measurements over assumptions for tokenizer, architecture, quantization, corpus composition, and performance decisions.
