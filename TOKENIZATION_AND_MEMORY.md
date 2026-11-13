# Tokenization and memory behavior

Minima separates tokenizer sampling from full-corpus training.

## Tokenizer rebuild

```text
Minima.exe --retrain-tokenizer-only --tokenizer-sample-bytes 1048576 ^
  --tokenizer-sampling spread data/books.txt data/chatlogs.txt data/britannica.txt
```

- The default tokenizer sample is 1 MiB.
- The sample budget is shared evenly across the selected source files. A small source can use all of its available text; its unused share is redistributed to the other sources.
- `spread` takes up to 64 evenly spaced chunks from each source. `prefix` samples the beginning of each source.
- Each sampled chunk and each inter-file separator is trained as an independent piece. The tokenizer never counts a byte pair across distant spread-sample chunks or across source-file boundaries.
- The sampler reads only the selected chunks. Tokenizer-only rebuilds do not load the full corpus into the permanent arena.
- Merge progress and elapsed time are printed every 50 merges.

## Full-corpus encoding and training

- Large inputs are encoded in 1 MiB windows with a 63-byte overlap on both sides. The encoder emits complete tokens at window boundaries, so chunked output is checked against the reference encoder in regression tests.
- The token-ID array is stored in a temporary disk-backed memory mapping. The operating system pages it in as training and diagnostics access it.
- After encoding, Minima releases the raw corpus arena. The mapped token IDs remain available for random training and validation windows.
- The temporary directory needs enough free space for the token-ID mapping. A corpus of about 950 MB can require up to about 1.9 GB for the worst-case `uint16_t` token array.
- Full-corpus training still has the existing `INT_MAX` token-count limit, so raw corpora must remain below roughly 2 GiB. Supporting multi-gigabyte training sets needs a separate `size_t` indexing/API migration; streaming tokenizer-only sampling does not remove that limit.

## CPU parallelism

OpenMP parallelizes independent large-input encoding windows, forward matrix rows, and backward gradient rows. The BPE merge sequence itself remains serial because each selected merge changes the candidate sequence for the next merge.

- By default, OpenMP uses the runtime's available worker-thread count.
- Set `OMP_NUM_THREADS` to control the default from the environment.
- Pass `--threads N` to the CLI to override it for one run.

For example:

```text
Minima.exe --threads 8 --steps 2000 data/books.txt data/chatlogs.txt data/britannica.txt
```

## Attention architecture

- Attention uses rotary position features (RoPE) on query and key vectors instead of a learned position-embedding table. This gives the model relative position information without a separate learned vector for every position; the configured `max_seq` still sets the runtime limit.
- Grouped-query attention derives fewer key/value heads from the configured query-head count. Each KV head serves an equal group of query heads. The default uses at most half as many KV heads, reducing attention projection weights and generation-cache memory; a one-head model keeps ordinary multi-head attention.
- These architecture changes alter checkpoint tensor shapes. Existing model checkpoints must be replaced by training a new model; tokenizer files remain usable.
- More aggressive options such as ternary-weight pretraining need specialized quantization-aware training and CPU kernels. This project keeps standard float32 weights and its existing AdamW trainer, which suit its portable CPU implementation.
