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

The loader currently accepts up to 5 MiB total input. BPE learns its vocabulary from an evenly distributed 256 KiB sample taken only from the training portion. The tokenizer then encodes the entire corpus. The final 5% of tokens are held out for validation and are never sampled for gradient updates. Training uses the first 95% of tokens.

## Training output and checkpoints

The default experiment accepts up to 5 MiB of plaintext and runs 2,000 optimizer
steps with batch size 4 and sequence length 32. It uses 100 warmup steps followed
by cosine learning-rate decay from a base rate of 3e-4. Training loss is averaged
and validation loss is evaluated every 100 steps over 16 fixed, evenly spaced
validation batches. This is still a small experimental model, not a claim that
2,000 steps produce a generally useful language model.

After training, the program writes two files in its working directory:

- `minima.bpe`: tokenizer vocabulary and ordered BPE merge rules.
- `minima.model`: model weights, AdamW first and second moments, optimizer
  hyperparameters, and optimizer step count.

The checkpoint format is versioned and checks the model architecture and
vocabulary size when loading. The save/load round-trip is also exercised by the
smoke test. The low-level load function is available, but a user-facing resume
command is not wired into the program yet. Keep the tokenizer file paired with
the model checkpoint; weights trained with a different tokenizer are not
compatible.

## Current implementation limits and checks

The training forward/backward attention kernels use a maximum sequence length of
512 because their per-query score workspaces are currently fixed at that size.
The inference path additionally requires dimension <= 512, FFN dimension <=
1024, and vocabulary size <= 8192. These limits are checked before those
buffers are used; increasing the model configuration beyond them requires
replacing the fixed inference workspaces with arena-backed buffers.

The smoke test now includes an AdamW bias-correction regression check, tokenizer
save/load encoding parity, checkpoint save/load verification, numerical
gradient checks, and KV-cache parity across every prefix position in the test
sequence. These checks are regression guards, not a substitute for evaluating
training on held-out text.
