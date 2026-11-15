#!/usr/bin/env python3
"""Run Minima's held-out logic benchmark using isolated, greedy chat prompts.

Reference answers and scoring notes are read only by this Python process.
Only each prompt, wrapped in the normal Human:/Assistant: chat format, is sent
to Minima. The chat engine's /clear command keeps cases independent.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPOSITORY_ROOT / "data" / "evaluation" / "logic.jsonl"
DEFAULT_EXE = REPOSITORY_ROOT / "x64" / "Release" / "Minima.exe"

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            for field in ("id", "category", "difficulty", "prompt", "reference_answer", "scoring_notes"):
                if not isinstance(item.get(field), str) or not item[field]:
                    raise ValueError(f"{path}:{line_number}: missing or invalid {field!r}")
            if item["id"] in seen_ids:
                raise ValueError(f"{path}:{line_number}: duplicate case ID {item['id']!r}")
            if "\n" in item["prompt"] or "\r" in item["prompt"]:
                raise ValueError(f"{path}:{line_number}: prompts must be single-line")
            seen_ids.add(item["id"])
            cases.append(item)
    if not cases:
        raise ValueError(f"{path}: no evaluation cases found")
    return cases

def build_chat_input(cases: list[dict[str, Any]], max_new_tokens: int) -> str:
    """Build normal chat turns, clearing history before every independent case."""
    del max_new_tokens  # The executable's CLI controls the generation budget.
    chunks: list[str] = []
    for case in cases:
        chunks.append(f"/clear\n{case['prompt']}\n")
    chunks.append("/quit\n")
    return "".join(chunks)

def parse_chat_responses(stdout: str) -> list[str]:
    """Extract multiline BOT responses, ending at the chat-ready protocol line."""
    responses: list[str] = []
    current: list[str] | None = None
    for line in stdout.splitlines():
        if current is None:
            if line.startswith("BOT: "):
                current = [line[len("BOT: "):]]
        elif line == "CHAT_READY":
            responses.append("\n".join(current).strip())
            current = None
        else:
            current.append(line)
    if current is not None:
        raise ValueError("Minima output ended before CHAT_READY after a BOT response")
    return responses

def run(args: argparse.Namespace) -> int:
    executable = args.exe.resolve()
    dataset = args.dataset.resolve()
    if args.workdir is not None:
        workdir = args.workdir.resolve()
    else:
        candidates = (Path.cwd(), Path.cwd() / "Minima")
        workdir = next(
            (candidate.resolve() for candidate in candidates
             if (candidate / "minima.model").is_file()
             and (candidate / "minima.bpe").is_file()),
            Path.cwd().resolve(),
        )
    output = args.output.resolve()
    if not executable.is_file():
        raise FileNotFoundError(f"Minima executable not found: {executable}")
    model_path = workdir / "minima.model"
    tokenizer_path = workdir / "minima.bpe"
    if not model_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {model_path}")
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
    cases = load_cases(dataset)
    command = [
        str(executable), "--chat", "--resume",
        "--temperature", "1.0", "--top-k", "1", "--top-p", "1.0",
        "--repetition-penalty", "1.0", "--max-new-tokens", str(args.max_new_tokens),
    ]
    if args.memory_budget_gb > 0:
        command.extend(("--memory-budget-gb", str(args.memory_budget_gb)))
    model_options = (
        ("--vocab-size", args.vocab_size),
        ("--dim", args.dim),
        ("--layers", args.layers),
        ("--heads", args.heads),
        ("--ffn-dim", args.ffn_dim),
        ("--max-seq", args.max_seq),
    )
    for option, value in model_options:
        if value is not None:
            command.extend((option, str(value)))
    print(f"Loading checkpoint once for {len(cases)} isolated test prompts...", flush=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    decoding = {
        "temperature": 1.0, "top_k": 1, "top_p": 1.0,
        "repetition_penalty": 1.0, "max_new_tokens": args.max_new_tokens,
        "strategy": "greedy (top-k 1)",
    }
    model_hash = sha256_file(model_path)
    tokenizer_hash = sha256_file(tokenizer_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    failed = 0
    completed_count = 0
    with output.open("w", encoding="utf-8", newline="\\n") as stream:
        process = subprocess.Popen(
            command, cwd=workdir, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        try:
            assert process.stdin is not None and process.stdout is not None
            for index, case in enumerate(cases, 1):
                process.stdin.write(f"/clear\\n{case['prompt']}\\n")
                process.stdin.flush()
                current: list[str] | None = None
                response: str | None = None
                while True:
                    line = process.stdout.readline()
                    if line == "":
                        raise RuntimeError(
                            f"Minima ended while answering {case['id']}. "
                            f"Exit status: {process.poll()}"
                        )
                    clean = line.rstrip("\\r\\n")
                    if current is None:
                        if clean.startswith("BOT: "):
                            current = [clean[len("BOT: "):]]
                    elif clean == "CHAT_READY":
                        response = "\\n".join(current).strip()
                        break
                    else:
                        current.append(clean)

                if not response or response in ("[generation failed]", "[input encoding failed]"):
                    failed += 1
                record = {
                    "run_utc": timestamp,
                    "checkpoint_sha256": model_hash,
                    "tokenizer_sha256": tokenizer_hash,
                    "decoding": decoding,
                    "id": case["id"],
                    "category": case["category"],
                    "difficulty": case["difficulty"],
                    "prompt": case["prompt"],
                    "generated_answer": response,
                    "reference_answer": case["reference_answer"],
                    "scoring_notes": case["scoring_notes"],
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\\n")
                stream.flush()
                completed_count += 1
                print(f"\\n[{index}/{len(cases)}] {case['id']} ({case['category']})",
                      flush=True)
                print("Question: " + case["prompt"], flush=True)
                print("Answer: " + (response or "[empty response]") + "\\n", flush=True)

            process.stdin.write("/quit\\n")
            process.stdin.flush()
            process.stdin.close()
            return_code = process.wait(timeout=args.timeout)
            if return_code != 0:
                raise RuntimeError(f"Minima exited with status {return_code}")
        except Exception:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise

    print(f"Completed: {completed_count - failed}/{len(cases)} usable responses", flush=True)
    print("Decoding: greedy, top-k=1, max_new_tokens=" + str(args.max_new_tokens), flush=True)
    print("Checkpoint SHA-256: " + model_hash, flush=True)
    print("Tokenizer SHA-256:  " + tokenizer_hash, flush=True)
    print("Results: " + str(output), flush=True)
    if failed:
        print(f"Warning: {failed} case(s) did not produce a usable response.", flush=True)
        return 1
    print("No scores were assigned. Review and score the answers using the rubric.", flush=True)
    return 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Minima's held-out basic-logic evaluation.")
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE,
                        help="Path to Minima.exe (default: x64/Release/Minima.exe)")
    parser.add_argument("--workdir", type=Path, default=None,
                        help="Directory containing minima.model and minima.bpe (auto-detected in the current directory or ./Minima by default)")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                        help="Evaluation JSONL file (default: data/evaluation/logic.jsonl)")
    parser.add_argument("--output", type=Path, default=Path("logic_results.jsonl"),
                        help="Output JSONL path (default: logic_results.jsonl)")
    parser.add_argument("--memory-budget-gb", type=int, default=0,
                        help="Aggregate committed arena budget in GiB; 0 disables it")
    parser.add_argument("--max-new-tokens", type=int, default=64,
                        help="Maximum generated tokens per case (default: 64)")
    parser.add_argument("--vocab-size", type=int, default=None,
                        help="Checkpoint vocabulary size, if it differs from the engine default")
    parser.add_argument("--dim", type=int, default=None,
                        help="Checkpoint model dimension, if non-default")
    parser.add_argument("--layers", type=int, default=None,
                        help="Checkpoint transformer layer count, if non-default")
    parser.add_argument("--heads", type=int, default=None,
                        help="Checkpoint attention head count, if non-default")
    parser.add_argument("--ffn-dim", type=int, default=None,
                        help="Checkpoint feed-forward dimension, if non-default")
    parser.add_argument("--max-seq", type=int, default=None,
                        help="Checkpoint context length, if non-default")
    parser.add_argument("--timeout", type=int, default=900,
                        help="Process timeout in seconds (default: 900)")
    args = parser.parse_args()
    if not 0 <= args.memory_budget_gb <= 64:
        parser.error("--memory-budget-gb must be between 0 and 64")
    if not 1 <= args.max_new_tokens <= 512:
        parser.error("--max-new-tokens must be between 1 and 512")
    bounds = {
        "vocab_size": (260, 32768), "dim": (1, 512), "layers": (1, 16),
        "heads": (1, 32), "ffn_dim": (1, 1024), "max_seq": (32, 512),
    }
    for name, (minimum, maximum) in bounds.items():
        value = getattr(args, name)
        if value is not None and not minimum <= value <= maximum:
            parser.error(f"--{name.replace('_', '-')} must be between {minimum} and {maximum}")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    return args

if __name__ == "__main__":
    try:
        raise SystemExit(run(parse_args()))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
