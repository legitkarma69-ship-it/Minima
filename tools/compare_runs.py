#!/usr/bin/env python3
"""Compare Minima metrics CSVs using their adjacent run-configuration sidecars."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def _number(value: Any, default: float = math.nan) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def load_run(metrics_path: str | Path) -> dict[str, Any]:
    path = Path(metrics_path)
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: metrics CSV has no data rows")

    valid_rows = []
    for row in rows:
        step = _number(row.get("step"))
        train_loss = _number(row.get("train_loss"))
        validation_loss = _number(row.get("validation_loss"))
        if math.isfinite(step) and math.isfinite(validation_loss):
            valid_rows.append((int(step), train_loss, validation_loss, row))
    if not valid_rows:
        raise ValueError(f"{path}: no rows contain valid step and validation_loss values")

    best = min(valid_rows, key=lambda item: item[2])
    final = valid_rows[-1]
    metadata_path = Path(str(path) + ".meta.json")
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as stream:
            loaded = json.load(stream)
        if isinstance(loaded, dict):
            metadata = loaded

    model = metadata.get("model") if isinstance(metadata.get("model"), dict) else {}
    training = metadata.get("training") if isinstance(metadata.get("training"), dict) else {}
    data = metadata.get("data") if isinstance(metadata.get("data"), dict) else {}
    sources = data.get("sources") if isinstance(data.get("sources"), list) else []

    final_step, final_train, final_validation, final_row = final
    initial_step = int(_number(training.get("initial_step"), valid_rows[0][0] - 1))
    steps_completed = max(0, final_step - initial_step)
    batch_size = int(_number(training.get("batch_size"), 0))
    sequence_length = int(_number(training.get("sequence_length"), 0))
    elapsed = _number(final_row.get("elapsed_seconds"))

    model_label = "unknown"
    if model:
        model_label = (
            f"v{model.get('vocab_size', '?')}/d{model.get('dim', '?')}/"
            f"L{model.get('layers', '?')}/H{model.get('heads', '?')}/"
            f"FF{model.get('ffn_dim', '?')}"
        )

    return {
        "run": path.parent.name or path.stem,
        "metrics_file": str(path),
        "metadata_file": str(metadata_path) if metadata else "",
        "optimizer": training.get("optimizer", "unknown"),
        "model": model_label,
        "batch_size": batch_size or "",
        "sequence_length": sequence_length or "",
        "corpus_mix": data.get("mix", "unknown"),
        "source_count": data.get("source_count", len(sources) or ""),
        "initial_step": initial_step,
        "final_step": final_step,
        "steps_completed": steps_completed,
        "estimated_tokens": steps_completed * batch_size * sequence_length,
        "best_validation_loss": best[2],
        "best_validation_step": best[0],
        "final_train_loss": final_train,
        "final_validation_loss": final_validation,
        "final_validation_perplexity": _number(final_row.get("validation_perplexity")),
        "elapsed_seconds": elapsed,
        "learning_rate": _number(final_row.get("learning_rate")),
        "seed": training.get("seed", ""),
        "weight_decay": training.get("weight_decay", ""),
        "beta1": training.get("beta1", ""),
        "beta2": training.get("beta2", ""),
        "epsilon": training.get("epsilon", ""),
    }


FIELDS = [
    "run", "optimizer", "model", "batch_size", "sequence_length",
    "corpus_mix", "source_count", "initial_step", "final_step",
    "steps_completed", "estimated_tokens", "best_validation_loss",
    "best_validation_step", "final_train_loss", "final_validation_loss",
    "final_validation_perplexity", "elapsed_seconds", "learning_rate",
    "seed", "weight_decay", "beta1", "beta2", "epsilon",
    "metrics_file", "metadata_file",
]


def compare_runs(metrics_files: list[str | Path], output: str | Path) -> list[dict[str, Any]]:
    if len(metrics_files) < 2:
        raise ValueError("provide at least two metrics CSV files to compare")
    summaries = [load_run(path) for path in metrics_files]
    summaries.sort(key=lambda row: row["best_validation_loss"])
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare Minima training runs by validation loss and recorded configuration."
    )
    parser.add_argument("metrics", nargs="+", help="metrics CSV files from separate runs")
    parser.add_argument("--output", default="minima_run_comparison.csv",
                        help="comparison CSV destination (default: minima_run_comparison.csv)")
    args = parser.parse_args()
    try:
        summaries = compare_runs(args.metrics, args.output)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(f"Comparison written to: {args.output}")
    print("Rank | Best val | Final val | Best step | Optimizer | Model | Run")
    for rank, row in enumerate(summaries, 1):
        print(
            f"{rank:>4} | {row['best_validation_loss']:>8.4f} | "
            f"{row['final_validation_loss']:>9.4f} | "
            f"{row['best_validation_step']:>9} | "
            f"{row['optimizer']:<9} | {row['model']:<32} | {row['run']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
