# Minima

A lightweight, CPU-first language-model project written in C.

## Project roadmap

This section is the working backlog for possible features, known issues, experiments, and future improvements. Items are ideas or tasks, not promises or release commitments. Update their status as work progresses.

### Status key

- [ ] **Planned** — identified, not yet implemented.
- [x] **Done** — implemented and verified.
- [~] **In progress** — implementation or verification is underway.
- [?] **Investigate** — needs research, measurement, or a design decision.

---

## 1. Data pipeline and corpus expansion

- [ ] **Review and replace the 1911 Encyclopaedia Britannica.** Its archaic vocabulary, syntax, and historical gaps may be a poor fit for a modern general-purpose corpus. The currently assembled Britannica text is approximately 906 MiB and has not yet had OCR/boilerplate cleanup, so evaluate its quality before treating it as training-ready.
- [ ] **Integrate clean public datasets.** Evaluate WikiText-103, curated Wikipedia Vital Articles (Levels 3 and 4), Simple English Wikipedia, and OpenStax textbooks.
- [ ] **Add high-signal books.** Curate modern fiction and dialogue, clear expository essays, popular science, and first-person biographies.
- [ ] **Scale corpus volume.** Grow the plaintext corpus from its current tens-of-MiB baseline toward 1–5 GiB (roughly 250 million–1.2 billion tokens, depending on tokenizer and text mix).
- [ ] **Evaluate Hugging Face datasets.** Investigate filtered educational subsets such as FineWeb-Edu and Cosmopedia; document licensing, provenance, filtering, and download requirements.
- [ ] **Add conversational data.** Evaluate OpenAssistant OASst1/OASst2 to balance expository prose with conversational turn-taking.
- [ ] **Clean text sources.** Strip Gutenberg boilerplate, OCR noise, licensing headers, duplicate material, and indices where appropriate. Prefer clean editions such as Standard Ebooks for public-domain classics.
- [ ] **Track corpus composition.** Experiment with a target mix of 40% narrative/fiction, 30% modern nonfiction/essays, 20% encyclopedia/textbooks, and 10% chat logs. Treat these as initial hypotheses and measure the effect rather than assuming the ratios are optimal.
- [ ] **Add corpus provenance and manifests.** Record source, license, origin URL, preprocessing steps, byte count, document count, and checksum for every dataset.

## 2. Tokenizer and vocabulary design

- [ ] **Evaluate alternative tokenizer algorithms.** Compare the current tokenizer with Byte-Level BPE (BBPE) and Unigram approaches, especially for whitespace, punctuation, and mixed-language text.
- [ ] **Sweep vocabulary sizes.** Test sizes from 512 through 30,000 and identify the quality, model-size, memory, and speed trade-offs.
- [ ] **Evaluate frequency thresholds.** Test minimum token/document occurrence thresholds around 5–10 occurrences on large corpora; measure unknown/fallback behavior and vocabulary utilization.
- [ ] **Increase tokenizer sample size.** Raise the default sample from 1 MiB to 4 MiB and use spread sampling so the sample represents the whole corpus rather than only its beginning.
- [ ] **Build a tokenizer experiment framework.** Automate vocabulary sweeps and compare bits-per-byte (BPB), compression, token counts, training loss, throughput, and morphological robustness.
- [ ] **Test English and Spanish explicitly.** Include accents, punctuation, code-switching, and common morphology in tokenizer regression tests.

## 3. Model architecture and CPU efficiency

- [ ] **Evaluate RMSNorm.** Compare it with the current normalization implementation for numerical stability, training quality, and CPU cost.
- [ ] **Evaluate SwiGLU.** Compare it with the current feed-forward activation, accounting for its additional parameter and compute requirements.
- [x] **Rotary Position Embeddings (RoPE).** Present in the current architecture; preserve and keep its tests/documentation aligned with the implementation.
- [x] **Grouped-Query Attention (GQA).** Present in the current architecture; preserve and keep its tensor layout and tests aligned with the implementation.
- [ ] **Scale model capacity.** Benchmark progressively larger configurations, including approximately 1M and 5.74M parameters. One candidate configuration to investigate is dimension 128, 8 layers, 4 query heads, and FFN dimension 512. Confirm parameter counts from the actual configuration rather than relying on estimates.
- [ ] **Measure architecture trade-offs.** Compare quality, tokens/second, peak RAM, and model size on the same datasets and training budgets before adopting changes.

## 4. Training loop, memory, and performance

- [ ] **Stream large corpora.** Implement memory-mapped (mmap) or buffered data access so 1–5 GiB datasets do not need to be loaded entirely into RAM.
- [ ] **Pre-allocate flat memory buffers.** Reduce allocation overhead during training and inference by reusing contiguous buffers for the KV cache, weights, activations, and hidden states where practical.
- [ ] **Reuse the logits/gradient buffer.** Investigate in-place reuse in cross-entropy/backpropagation to reduce peak temporary memory (estimated savings in the original proposal: about 58.6 MiB for the cited default configuration; re-measure against current dimensions).
- [ ] **Add post-training quantization (PTQ).** Evaluate int8 and int4 weight formats, including quality loss, inference speed, and actual RAM reduction.
- [ ] **Add profiling instrumentation.** Measure forward pass, attention, output projection, backward pass, optimizer, tokenizer, and validation separately.
- [ ] **Create a repeatable benchmark suite.** Record commit, compiler/build settings, configuration, dataset manifest, train/validation loss, peak memory, tokens/second, and representative generation outputs.

## 5. Testing, CI, and diagnostics

- [ ] **Re-verify the full Windows CI suite.** Check for stale assumptions about learned position embeddings and the current RoPE/GQA parameter layout.
- [ ] **Update train_validation_test.c.** Align validation tests with the current architecture and data-splitting behavior.
- [ ] **Update gradcheck_test.c.** Verify numerical gradients against the current parameter and tensor layout.
- [ ] **Update gradients_init_test.c.** Verify initialization and gradient coverage for every trainable tensor in the current 9-tensor parameter layout.
- [ ] **Update MODEL_DIAGNOSTICS.md.** Make its description match the actual per-source split and boundary-isolation implementation.
- [ ] **Make validation document-aware.** Use deterministic document-level holdouts rather than relying only on per-source separation; ensure chunks from the same document cannot leak across train and validation.
- [ ] **Refine preflight diagnostics.** Distinguish an invalid tokenizer from a valid tokenizer that simply did not reach the requested vocabulary size.
- [ ] **Add regression coverage for corpus boundaries.** Verify that source/document boundaries, split boundaries, and sequence construction cannot leak text between training and validation.

## 6. GUI, resource management, and chat controls

### Resource management

- [ ] Add RAM allocation limits and configurable memory budgets.
- [ ] Add CPU utilization limits using Windows Job Objects.
- [ ] Add CPU thread selectors and process-priority controls.
- [ ] Add live RAM/CPU monitors and configurable warning thresholds.

### Chat and conversation management

- [ ] Add separate controls to clear the visible chat display and clear the active chat history.
- [ ] Add an explicit action to delete the persistent minima_chat.txt log, with confirmation.
- [ ] Support saving, loading, and exporting conversations.

### Training and generation controls

- [ ] Expose optimizer selection (AdamW or Lion) and weight decay.
- [ ] Expose advanced optimizer parameters (beta1, beta2, epsilon) where applicable to the selected optimizer.
- [ ] Expose individual corpus-source weighting ratios.
- [ ] Add generation controls for Min-p, frequency penalty, presence penalty, stop sequences, and a reproducible seed.
- [ ] Add token-probability inspection.
- [ ] Add configuration profiles with save, load, import, and export.
- [ ] Add training pause/resume without terminating the engine.
- [ ] Show training ETA, tokens per second, and best-checkpoint tracking.
- [ ] Add side-by-side response comparisons and reusable prompt templates.

---

## Roadmap maintenance

- Keep this list focused on project-level work. Small implementation details belong in code comments, tests, or focused issue discussions.
- Before marking an item **Done**, verify it in the code and, where applicable, with tests or benchmarks.
- When an idea is rejected or superseded, remove it or move it to a short **Deferred / rejected** note with the reason.
- Prefer measured results over assumptions, especially for tokenizer, architecture, quantization, and corpus-ratio decisions.
