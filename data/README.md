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
