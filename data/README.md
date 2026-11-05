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

The loader currently accepts up to 64 MiB total input. BPE learns its vocabulary from an evenly distributed sample capped at 256 KiB, then the tokenizer encodes the entire loaded corpus for training. The model itself is still the tiny smoke-test configuration; this loader is the first real-data path, not yet a production-scale training pipeline.

## Training output and checkpoints

The current smoke-training run uses 200 optimizer steps. This is a pipeline test,
not a claim that 200 steps are enough to train a useful language model on a book.

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
