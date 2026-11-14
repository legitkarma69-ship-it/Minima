# Training data and stability

Minima keeps the selected text files in one byte buffer, then tokenizes each
file independently. It inserts an end-of-text marker between training and
held-out portions, so training windows cannot accidentally cross from one
file into another or leak held-out text into training.

Each usable file receives its own held-out tail. The split reserves at least
one configured sequence for validation, while keeping at least one sequence
for training. A file that is too short for both is reported and skipped for
that run. Validation windows are spread across the held-out portions and use
the same source weights as training.

## Source mixture

The `--corpus-mix` option controls how training sequences are selected from
the usable files:

- `sqrt` (default): each file's probability is proportional to the square
  root of its available training windows. Large sources stay influential,
  while small sources receive more exposure than raw-size sampling gives them.
- `proportional`: probability follows available training windows, matching
  the old file-size-weighted behavior.
- `equal`: each file receives the same share, even when sizes differ greatly.

The Windows GUI exposes this choice under **Advanced → Data mix**.
Minima prints the effective source shares and estimated passes over each
source for the selected training segment.

There is no universally optimal mixture. Recent mixture-scaling work finds
that scarce target data can benefit from repetition, but the best repetition
depends on data size, model size, and compute budget. For Minima, `sqrt` is a
practical starting point; compare validation loss and generated text before
choosing a more aggressive setting. See [Scaling Laws for Mixture Pretraining
Under Data Constraints](https://arxiv.org/abs/2605.12715). Recent work also
frames mixture selection as controlled proxy experiments evaluated with
validation loss ([Data Mixing as Mixture Experiment](https://arxiv.org/abs/2608.23922));
that is a good next step once Minima's validation corpus is large enough to
compare short runs reliably.

## Gradient clipping

Training applies global L2 gradient clipping at `1.0` by default, before
AdamW or Lion updates. This bounds unusually large updates without changing
the direction of the full gradient. The step summary reports the mean and
peak unclipped norm and how many updates were clipped. Set `--grad-clip 0` or
the GUI's **Clip norm** field to `0` to disable it.

The default follows established autoregressive training practice, but the
best threshold can depend on Minima's model and corpus. Keep the printed clip
rate in view: clipping nearly every step can indicate that the threshold or
learning rate should be adjusted. Global norm clipping is also described in
the [GPT-3 training setup](https://arxiv.org/abs/2005.14165); its large-scale
results do not establish the best threshold for Minima, so the field remains
configurable.
