import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_runs import compare_runs, load_run


class CompareRunsTests(unittest.TestCase):
    def _make_run(self, root, name, best_loss, final_loss, optimizer, initial_step):
        directory = root / name
        directory.mkdir()
        metrics = directory / "minima_metrics.csv"
        with metrics.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow([
                "step", "train_loss", "validation_loss", "learning_rate",
                "train_perplexity", "validation_perplexity", "elapsed_seconds",
            ])
            writer.writerow([initial_step + 5, 1.8, best_loss, 0.0002, 6.0, 3.0, 2.0])
            writer.writerow([initial_step + 10, 1.5, final_loss, 0.0001, 4.5, 3.5, 4.0])
        metadata = {
            "schema_version": 1,
            "model": {
                "vocab_size": 2048, "dim": 128, "layers": 8,
                "heads": 4, "ffn_dim": 512, "max_seq": 128,
            },
            "training": {
                "optimizer": optimizer, "initial_step": initial_step,
                "lr_schedule": "cosine", "total_steps": 1000, "warmup_steps": 20,
                "grad_clip_norm": 1.0, "threads": 4, "memory_budget_gb": 8,
                "checkpoint_retention": 3, "log_interval": 50,
                "validation_batches": 4, "validation_interval": 100,
                "checkpoint_interval": 100, "tokenizer_sample_bytes": 65536,
                "tokenizer_spread": True,
                "batch_size": 4, "sequence_length": 128,
                "seed": 1234, "weight_decay": 0.01,
                "beta1": 0.9, "beta2": 0.95, "epsilon": 1e-8,
            },
            "data": {
                "mix": "sqrt", "source_count": 2,
                "sources": [{"name": "books.txt", "weight": 1.0}],
            },
        }
        Path(str(metrics) + ".meta.json").write_text(
            json.dumps(metadata), encoding="utf-8"
        )
        return metrics

    def test_compare_sorts_by_best_validation_loss_and_writes_config(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            weaker = self._make_run(root, "run-weaker", 1.4, 1.6, "adamw", 100)
            stronger = self._make_run(root, "run-stronger", 1.1, 1.2, "lion", 200)
            output = root / "comparison.csv"

            rows = compare_runs([weaker, stronger], output)

            self.assertEqual([row["run"] for row in rows],
                             ["run-stronger", "run-weaker"])
            self.assertEqual(rows[0]["optimizer"], "lion")
            self.assertEqual(rows[0]["model"], "v2048/d128/L8/H4/FF512")
            self.assertEqual(rows[0]["steps_completed"], 10)
            self.assertEqual(rows[0]["estimated_tokens"], 10 * 4 * 128)
            self.assertEqual(rows[0]["source_weights"], "books.txt=1.0")
            self.assertEqual(rows[0]["lr_schedule"], "cosine")
            self.assertEqual(rows[0]["memory_budget_gb"], 8)
            self.assertEqual(rows[0]["tokenizer_sample_bytes"], 65536)
            self.assertTrue(output.is_file())
            with output.open("r", newline="", encoding="utf-8") as stream:
                saved = list(csv.DictReader(stream))
            self.assertEqual(len(saved), 2)
            self.assertEqual(saved[0]["optimizer"], "lion")

    def test_missing_sidecar_still_compares_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.csv"
            second = root / "second.csv"
            for path, loss in ((first, 1.3), (second, 1.5)):
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.writer(stream)
                    writer.writerow(["step", "train_loss", "validation_loss",
                                     "learning_rate", "train_perplexity",
                                     "validation_perplexity", "elapsed_seconds"])
                    writer.writerow([10, 1.0, loss, 0.0001, 2.7, 3.7, 2.0])
            rows = compare_runs([first, second], root / "out.csv")
            self.assertEqual(rows[0]["best_validation_loss"], 1.3)
            self.assertEqual(rows[0]["optimizer"], "unknown")

    def test_requires_two_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                compare_runs([Path(temporary) / "one.csv"], Path(temporary) / "out.csv")


if __name__ == "__main__":
    unittest.main()
