# Tokenization and memory behavior

Minima separates tokenizer sampling from full-corpus training.

## Tokenizer rebuild

\`\`\`text
Minima.exe --retrain-tokenizer-only --tokenizer-sample-bytes 1048576 ^
  --tokenizer-sampling spread data/books.txt data/chatlogs.txt data/britannica.txt
\`\`\`

- The default tokenizer sample is 1 MiB.
- The sample budget is shared evenly across the selected source files. A small source can use all of its available text; its unused share is redistributed to the other sources.
- \`spread\` takes up to 64 evenly spaced chunks from each source. \`prefix\` samples the beginning of each source.
- The sampler reads only the selected chunks. Tokenizer-only rebuilds do not load the full corpus into the permanent arena.
- Merge progress and elapsed time are printed every 50 merges.

## Full-corpus encoding and training

- Large inputs are encoded in 4 MiB windows with 63 bytes of lookahead. The encoder emits complete tokens at window boundaries, so chunked output is checked against the reference encoder in regression tests.
- The token-ID array is stored in a temporary disk-backed memory mapping. The operating system pages it in as training and diagnostics access it.
- After encoding, Minima releases the raw corpus arena. The mapped token IDs remain available for random training and validation windows.
- The temporary directory needs enough free space for the token-ID mapping. A corpus of about 950 MB can require up to about 1.9 GB for the worst-case \`uint16_t\` token array.

## CPU parallelism

OpenMP parallelizes independent forward matrix rows and backward gradient rows. The BPE merge sequence itself remains serial because each selected merge changes the candidate sequence for the next merge.

- By default, OpenMP uses the runtime's available worker-thread count.
- Set \`OMP_NUM_THREADS\` to control the default from the environment.
- Pass \`--threads N\` to the CLI to override it for one run.

For example:

\`\`\`text
Minima.exe --threads 8 --steps 2000 data/books.txt data/chatlogs.txt data/britannica.txt
\`\`\`
