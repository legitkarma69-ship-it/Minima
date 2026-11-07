# Training text

Place UTF-8 plain-text material in these files:

- `books.txt`: books, essays, stories, documentation, and other long-form text.
- `chatlogs.txt`: conversation logs in plain text.

Keep the sources separate. This makes it possible to introduce source-aware sampling and weighting without mixing literature and dialogue into one manually maintained file.

Use UTF-8 plain text. For conversational data, use exactly these speaker labels:

```text
Human: Hey there, bot.
Assistant: Hello! How can I help you today?

Human: What is a black hole?
Assistant: A black hole is a region of space where gravity is so strong that nothing, not even light, can escape from inside its event horizon.

Human: Does that mean it sucks in everything nearby?
Assistant: No. From far away, its gravity behaves like the gravity of any other object with the same mass.
```

Formatting rules for `chatlogs.txt`:

- Use `Human:` for user turns and `Assistant:` for assistant turns. Keep spelling and capitalization consistent.
- Put one speaker turn on each line. Keep the speaker label and message on the same line.
- Separate conversations with one blank line.
- Keep consecutive turns in the same conversation together, in their original order.
- Prefer complete exchanges. Do not leave an unanswered final Human turn or an incomplete Assistant response.
- Preserve natural wording, punctuation, and meaningful paragraph breaks. Avoid artificial spelling, excessive filler, and duplicated conversations.
- Do not mix book chapters, metadata, source notes, or preference-ranking records into this file.
- Do not add a `System:` role yet. The current conversational format supports Human and Assistant turns only.

The interactive chat engine uses the same serialized structure at inference time: it supplies `Human: <message>\nAssistant:` and carries the generated assistant response into the next turn. Generation Lab intentionally remains a raw text-continuation test and does not add conversational role tags.

UTF-8 files with or without a byte-order mark (BOM) are supported. LF and CRLF line endings are both accepted.

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

The checkpoint format is version 3. It stores model weights, both AdamW
moment arrays, optimizer hyperparameters and global step, plus the training
RNG state. It also stores a fingerprint of the exact tokenizer. Resume refuses to load
if the model architecture or tokenizer differs, but the corpus may be expanded
or updated between training segments.
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

The learning-rate schedule uses the absolute optimizer step and is stored in
the checkpoint. A resumed run continues the same warmup/cosine schedule rather
than restarting it. On a fresh run, use `--total-steps N` to choose the full
schedule horizon; `--steps N` controls how many updates this invocation
performs. Resume requires the same tokenizer, but it can use updated or
expanded training text. Old version-2 checkpoints are incompatible with the
version-3 format. The new corpus is encoded with the saved tokenizer, and the 95/5
train/validation split is recalculated from that corpus. If you change the
vocabulary configuration, start a fresh run instead. The previous
512-entry tokenizer/checkpoint from earlier experiments is incompatible with
the new 2,048-entry configuration; the first run must be a fresh run, without
`--resume`.

The smoke test includes a deterministic generation benchmark using five fixed
prompts and greedy top-k=1 decoding. Each prompt is generated twice from the
same initial state and the token sequences are compared exactly. This checks
repeatability; it does not measure generation quality by itself.

## Tokenizer maintenance

A normal fresh training run trains a tokenizer from the current corpus. Resume
loads the tokenizer paired with the checkpoint. The GUI's **Rebuild tokenizer**
button runs a tokenizer-only pass and backs up the previous tokenizer as
`minima.bpe.previous`. After it completes, use **Start training** with Resume
unchecked. The GUI reuses the prebuilt tokenizer if the corpus selection and
tokenizer settings have not changed.

The command-line equivalents are:

```text
Minima.exe --retrain-tokenizer-only --vocab-size 2048 --tokenizer-sample-bytes 262144 books.txt
Minima.exe --use-existing-tokenizer --steps 2000 --total-steps 10000 books.txt
```

The **Auto settings** GUI button estimates one token-pass from corpus file size,
using the current batch and sequence settings. It estimates roughly three
source bytes per token, so treat the suggested steps as a starting point and
check the exact token count reported by the engine. It adjusts steps, total
steps, warmup, logging/validation/checkpoint intervals, and tokenizer sample
size. It deliberately does not change model dimensions, batch size, or
sequence length because those affect memory use and should not be guessed from
corpus size alone.

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
