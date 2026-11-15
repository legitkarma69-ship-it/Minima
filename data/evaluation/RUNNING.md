# Running the held-out reasoning evaluation

The benchmark runner is `tools/evaluate_logic.py`. It uses the existing interactive chat executable and the checkpoint already paired with `minima.bpe`.

## Requirements

- A Windows x64 Release build of `Minima.exe`.
- `minima.model` and its matching `minima.bpe` in the selected working directory.
- Python 3.9 or newer. The runner uses only the Python standard library.
- The held-out dataset at `data/evaluation/logic.jsonl`.

## Run from the repository root

```powershell
python tools/evaluate_logic.py
```

The default executable path is `x64/Release/Minima.exe`. If your executable or checkpoint files are elsewhere:

```powershell
python tools/evaluate_logic.py `
  --exe "path\to\Minima.exe" `
  --workdir "path\to\checkpoint-folder" `
  --output "logic_results.jsonl"
```

To change the response budget:

```powershell
python tools/evaluate_logic.py --max-new-tokens 64
```

## What the runner does

- Loads the checkpoint and tokenizer once.
- Sends every case as an independent `Human: ...` / `Assistant:` prompt through the chat engine's one-shot `/test` path. No earlier test response is included in a later prompt.
- Uses greedy decoding (`temperature=1`, `top-k=1`, `top-p=1`, repetition penalty `1`) so the same checkpoint and prompt produce repeatable token choices.
- Sends only the `prompt` field to Minima. `reference_answer` and `scoring_notes` are read by the runner for the result file and are never sent to the model.
- Writes one result per line to `logic_results.jsonl` by default. Each record includes the prompt, generated response, reference answer, scoring notes, decoding settings, and SHA-256 fingerprints of the model and tokenizer.

The runner does **not** automatically score answers. Review the generated responses and apply the rubric in this directory's `README.md`. A response that is fluent but logically invalid must not receive full credit.

## Important

- Run from the directory containing the checkpoint, or pass `--workdir`.
- Do not move `logic.jsonl` into a training corpus or tokenizer input.
- Keep result files out of `books.txt`, `chatlogs.txt`, and all training/tokenizer data.
- This first run is a baseline for the current checkpoint. Save the output unchanged before making training or architecture changes.
