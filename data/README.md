# Training text

Place UTF-8 plain-text material in these files:

- `books.txt`: books, essays, stories, documentation, and other long-form text.
- `chatlogs.txt`: conversation logs in plain text.

Keep the two sources separate so they can be loaded, sampled, and weighted independently later.

Use ordinary text. Separate documents with a blank line. For chat logs, a readable format such as:

```text
User: ...
Assistant: ...
```

is fine. UTF-8 files with or without a byte-order mark (BOM) are supported.

## Loading data

When launched from the Visual Studio project working directory, ProjectMinima checks:

- `../data/books.txt`
- `../data/chatlogs.txt`

When launched from the repository root, it checks the same files under `data/`.

You can also pass explicit paths as command-line arguments:

```text
Minima.exe ..\data\books.txt ..\data\chatlogs.txt
```

Empty or missing default files are skipped. If neither contains text, the program uses its small built-in synthetic corpus so the smoke tests remain runnable.

There is no arbitrary 5 MiB input cap. The loader attempts to read all supplied text, subject to available memory and the current implementation limits: the permanent arena reserves 8 GiB of virtual address space on Windows and commits pages only as needed, and the tokenizer uses integer byte/token counts with a maximum input length of INT_MAX bytes (approximately 2 GiB). The 8 GiB reservation does not mean 8 GiB of RAM is consumed at startup; Windows backs pages as the allocator uses them. Actual capacity is still limited by available commit charge/RAM, address space, and the additional memory needed for token IDs, tokenizer data, and model state. The Windows build should target x64; a 32-bit process cannot reserve an 8 GiB region. If allocation fails, loading or training stops with an error rather than silently truncating the corpus.

BPE learns its vocabulary from an evenly distributed 256 KiB sample taken only from the training portion, regardless of total corpus size. The tokenizer then encodes the entire corpus. The final 5% of tokens are held out for validation and are never sampled for gradient updates. Training uses the first 95% of tokens.

## Training output and checkpoints

The default experiment has no fixed plaintext-size cap and runs 2,000 optimizer
steps with batch size 4 and sequence length 32. The default BPE vocabulary target
is 2,048 entries (including the four reserved tokens and 256 byte tokens). It uses 100 warmup steps followed
by cosine learning-rate decay from a base rate of 3e-4. Training loss is averaged
and validation loss is evaluated every 100 steps over 16 fixed, evenly spaced
validation batches. This is still a small experimental model, not a claim that
2,000 steps produce a generally useful language model.

After training, the program writes two files in its working directory:

- `minima.bpe`: tokenizer vocabulary and ordered BPE merge rules.
- `minima.model`: model weights, AdamW first and second moments, optimizer
  hyperparameters, and optimizer step count.

The checkpoint format is version 2. It stores model weights, both AdamW
moment arrays, optimizer hyperparameters and global step, plus the training
RNG state. It also stores fingerprints of the exact tokenizer and loaded corpus.
Resume refuses to load if the model architecture, tokenizer, or corpus differs.
Keep `minima.bpe` paired with `minima.model`.

### Resuming training

The command-line syntax is:

```text
Minima.exe [--resume] [--steps N] [corpus.txt ...]
```

- A normal run starts a new model and tokenizer and runs 2,000 steps by default.
- `--steps N` changes the number of additional optimizer steps in that run.
- `--resume --steps 2000` loads the existing tokenizer, model, AdamW moments,
  global optimizer step, and training RNG state, then performs 2,000 more steps.
- Corpus paths can follow the options. If none are supplied, the normal default
  paths are used.

A step-zero checkpoint is written before fresh training begins, then updated
every 100 steps and at the end of each run. Checkpoint replacement is atomic,
so an interrupted write leaves the previous complete checkpoint intact. An
interrupted training run therefore loses at most 99 completed optimizer steps. The AdamW bias-correction counter and
random training-sample sequence continue from their saved states.

Each invocation is a separate training segment: its learning-rate schedule
starts with a fresh warmup and cosine decay, while the AdamW optimizer step
remains global. This is a deliberate warm restart, not an attempt to pretend
that a newly extended schedule was known during the previous run. Resume also
requires the same corpus contents and tokenizer. If you change the training
text or vocabulary configuration, start a fresh run instead. The previous
512-entry tokenizer/checkpoint from earlier experiments is incompatible with
the new 2,048-entry configuration; the first run must be a fresh run, without
`--resume`.

The smoke test includes a deterministic generation benchmark using five fixed
prompts and greedy top-k=1 decoding. Each prompt is generated twice from the
same initial state and the token sequences are compared exactly. This checks
repeatability; it does not measure generation quality by itself.

## Current implementation limits and checks

The training forward/backward attention kernels use a maximum sequence length of
512 because their per-query score workspaces are currently fixed at that size.
The inference path additionally requires dimension <= 512, FFN dimension <=
1024, and vocabulary size <= 8192. These limits are checked before those
buffers are used; increasing the model configuration beyond them requires
replacing the fixed inference workspaces with arena-backed buffers.

The smoke test now includes an AdamW bias-correction regression check, tokenizer
save/load encoding and fingerprint parity, checkpoint save/load verification
(including optimizer state and RNG restoration), numerical gradient checks,
KV-cache parity across every prefix position in the test sequence, and the
repeated deterministic multi-prompt generation benchmark. These checks are regression guards, not a substitute for evaluating
training on held-out text.
