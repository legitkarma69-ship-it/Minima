#!/usr/bin/env python3
"""Integration test for Minima's cooperative between-step pause/resume signal."""
from __future__ import annotations

import argparse
import queue
import subprocess
import threading
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    args = parser.parse_args()

    executable = args.exe.resolve()
    corpus = args.corpus.resolve()
    workdir = Path.cwd()
    pause_file = workdir / "minima_ci_pause.signal"
    stop_file = workdir / "minima_ci_pause.stop"
    metrics_file = workdir / "minima_ci_pause_metrics.csv"
    for path in (pause_file, stop_file, metrics_file):
        path.unlink(missing_ok=True)

    pause_file.write_text("pause\n", encoding="ascii")
    command = [
        str(executable), "--use-existing-tokenizer", "--steps", "2",
        "--total-steps", "3", "--warmup", "1", "--vocab-size", "300",
        "--tokenizer-sample-bytes", "8192", "--batch-size", "1",
        "--dim", "96", "--layers", "3", "--heads", "4", "--ffn-dim", "192",
        "--sequence-length", "8", "--max-seq", "80", "--validation-batches", "1",
        "--log-interval", "1", "--validation-interval", "1",
        "--checkpoint-interval", "1", "--metrics", str(metrics_file),
        "--pause-file", str(pause_file), "--stop-file", str(stop_file),
        "--source-weights", "1.0", str(corpus),
    ]

    process = subprocess.Popen(
        command, cwd=workdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    lines: queue.Queue[str | None] = queue.Queue()

    def collect_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line.rstrip("\r\n"))
        lines.put(None)

    reader = threading.Thread(target=collect_output, daemon=True)
    reader.start()
    observed_paused = False
    observed_resumed = False
    captured: list[str] = []
    try:
        while not observed_paused:
            try:
                line = lines.get(timeout=30)
            except queue.Empty as exc:
                raise RuntimeError("Minima did not enter the paused state within 30 seconds") from exc
            if line is None:
                raise RuntimeError(f"Minima exited before pausing (exit={process.poll()})")
            captured.append(line)
            print(line, flush=True)
            if line == "TRAIN_PAUSED":
                observed_paused = True

        pause_file.unlink(missing_ok=True)
        while not observed_resumed:
            try:
                line = lines.get(timeout=30)
            except queue.Empty as exc:
                raise RuntimeError("Minima did not resume within 30 seconds after removing the pause signal") from exc
            if line is None:
                raise RuntimeError(f"Minima exited before resuming (exit={process.poll()})")
            captured.append(line)
            print(line, flush=True)
            if line == "TRAIN_RESUMED":
                observed_resumed = True

        deadline = time.monotonic() + 120
        while True:
            try:
                line = lines.get(timeout=1)
            except queue.Empty:
                if process.poll() is not None:
                    break
                if time.monotonic() >= deadline:
                    raise RuntimeError("Minima did not finish the resumed training segment within 120 seconds")
                continue
            if line is None:
                break
            captured.append(line)
            print(line, flush=True)

        return_code = process.wait(timeout=120)
        if return_code != 0:
            raise RuntimeError(f"Minima exited with status {return_code}")
        if not observed_paused or not observed_resumed:
            raise RuntimeError("Pause/resume protocol was incomplete")
        print("Training pause/resume integration: PASS", flush=True)
        return 0
    except Exception:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    finally:
        pause_file.unlink(missing_ok=True)
        stop_file.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
