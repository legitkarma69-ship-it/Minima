# Held-out reasoning evaluation

This directory contains evaluation material for Minima's intended text-only reasoning behavior.

## Rules

- These files are **evaluation-only**. Never include them in `books.txt`, `chatlogs.txt`, tokenizer training samples, pretraining, or supervised fine-tuning.
- When evaluating, send only each item's `prompt` to Minima. Do not expose `reference_answer` or `scoring_notes` to the model.
- Use deterministic decoding and record the checkpoint, tokenizer fingerprint, model configuration, prompt ID, generated answer, and decoding settings.
- Do not require exact wording. Score semantic correctness.
- Score the conclusion and reasoning separately. A correct guess with invalid reasoning should not receive full credit.
- Include uncertainty cases. Correctly saying that a conclusion cannot be determined is a positive behavior, not a failure to answer.

## Initial scoring rubric

Score each response from 0 to 2:

- **0:** Incorrect conclusion, contradiction, invented premise, or failure to recognize insufficient information.
- **1:** Correct conclusion but missing, weak, or partly incorrect explanation; or a mostly correct explanation with a minor omission.
- **2:** Correct conclusion supported by reasoning that follows from the stated premises, with uncertainty handled appropriately.

For each model run, report:
- Total score and maximum possible score.
- Accuracy by category.
- Number of unsupported assumptions.
- Number of cases where the model confidently answered despite insufficient information.
- Concision/readability notes.

The initial set is a small behavioral smoke test, not a comprehensive reasoning benchmark. Add paraphrases, harder examples, and adversarial distractors only after establishing a baseline. Keep a stable core set so changes can be compared over time.

## Files

- `logic.jsonl`: 24 short prompts covering deduction, conditionals, ordering, constraints, classification, causal reasoning, consistency, insufficient information, simple counting, and ambiguity.

The JSONL schema is one object per line:

- `id`: stable test identifier.
- `category`: behavior being evaluated.
- `difficulty`: approximate difficulty label.
- `prompt`: exact user-facing test prompt.
- `reference_answer`: expected conclusion and a concise rationale.
- `scoring_notes`: case-specific criteria for human or later automated-assisted scoring.
