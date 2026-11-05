# Model quality diagnostics

Minima runs a read-only quality-diagnostics pass after training and after
\`--diagnostics-only\` runs. The diagnostics do not update weights, optimizer
moments, checkpoint state, or the training RNG.

## Generated reports

- \`minima_token_frequency.csv\`: every tokenizer ID, ordered by training
  frequency, with its byte length, token category, and train/validation counts.
  Byte tokens are escaped as \`\\xNN\` so the CSV remains valid even for NUL and
  other non-printable byte values.
- \`minima_frequency_loss.csv\`: cross-entropy and perplexity for targets grouped
  by their training-frequency rank. The groups are the top 1%, next 9%, next
  40%, remaining observed tokens, and tokens absent from the training split.
- \`minima_generation_diagnostics.csv\`: fixed prompt/mode/seed combinations,
  generated text, token entropy, distinct-1/2/3, adjacent-repeat rate, and
  maximum repeated 2-4 token pattern count.

The console report also includes tokenizer compression and byte-fallback
statistics, whitespace-bearing token use, top-token mass, and Jensen-Shannon
divergence between the training and validation token distributions. For each
generation prompt it prints the top next-token predictions, their full
vocabulary probabilities, entropy, and cumulative top-1/top-5/top-10 mass.

## Methodology and limitations

Tokenizer counts cover the already-tokenized train and validation regions.
Frequency buckets are ranked using training counts only. Loss-by-frequency
uses 16 deterministic, evenly spaced windows from each split, with the
configured sequence length. It is a diagnostic sample, not a replacement for
the main validation-loss estimate.

Generation prompts and random seeds are fixed so runs can be compared. The
greedy path is repeated once per prompt to verify deterministic output.
Sampling modes vary temperature, top-k, top-p, and repetition penalty. The
metrics describe surface repetition and token diversity; they do not measure
semantic quality or factual correctness.

The current corpus split is still a contiguous 95/5 token-stream split after
the selected files have been concatenated. If a source document crosses that
boundary, related text can appear on both sides. The distribution divergence
and frequency reports help reveal some differences, but they do not fix that
data-splitting limitation.

## Interpretation

- High raw-byte fallback can mean the learned merges are not covering common
  text patterns effectively, but the cause may also be sample composition.
- High top-token mass can indicate a strongly skewed corpus, not automatically
  a tokenizer defect.
- A large train/validation distribution divergence can indicate a domain or
  source shift, or simply an unrepresentative split.
- Much higher validation loss for rare/unseen targets can point toward limited
  coverage or insufficient training signal.
- Higher generation diversity is not inherently better. Random nonsense is
  diverse too, which is a rather low bar.

Use these reports to decide what to change next. Do not treat any single
metric as a diagnosis by itself.
