#!/usr/bin/env python3
"""Parallel test runner.

Reads test examples from data/<task>/test/, creates a folder per test example
inside the current agent run's test/ directory, and launches a Claude Code
instance for each one in parallel.

Called by the `test` bash command from within a strategy agent session.
Expects environment variables: SCAFFOLD_ROOT, AGENT_RUN_DIR, AGENT_TASK, AGENT_RUN_ID
"""

import csv
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def get_env():
    """Read required environment variables set by agent.bashrc."""
    required = ["SCAFFOLD_ROOT", "AGENT_RUN_DIR", "AGENT_TASK", "AGENT_RUN_ID"]
    env = {}
    for key in required:
        val = os.environ.get(key)
        if not val:
            print(f"Error: {key} not set. Are you running inside an agent session?")
            sys.exit(1)
        env[key] = val
    return env


def collect_test_examples(data_test_dir: Path) -> list[dict]:
    """Collect test examples from JSON files."""
    examples = []
    for json_file in sorted(data_test_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
        example_id = json_file.stem
        npy_path = data_test_dir / f"{example_id}.npy"
        examples.append({
            "id": example_id,
            "json_path": json_file,
            "npy_path": npy_path if npy_path.exists() else None,
            "data": data,
        })
    return examples


def run_single_test(
    test_index: int,
    example: dict,
    run_dir: Path,
    strategy_dir: Path,
    trace_dir: Path,
    prompt_path: Path,
    task_description: str,
    bashrc_path: Path,
) -> dict:
    """Run a single test agent on one example. Returns result dict."""
    # Per-test folder sits directly under run_dir, sibling to strategy/
    test_folder = run_dir / f"test-{test_index:03d}"
    test_folder.mkdir(parents=True, exist_ok=True)

    # Copy test example into the folder, but strip the ground-truth label first
    # so the test agent cannot cheat by reading example.json.
    redacted = {k: v for k, v in example["data"].items() if k != "label"}
    with open(test_folder / "example.json", "w", encoding="utf-8") as f:
        json.dump(redacted, f, indent=2, ensure_ascii=False)
    if example["npy_path"]:
        shutil.copy2(example["npy_path"], test_folder / "example.npy")

    user_prompt = (
        f"You are classifying one test example (id={example['id']}).\n"
        f"The example is in `example.json` in your current directory "
        f"(ground-truth label has been removed).\n"
        f"Read STRATEGY.md (and any referenced files) from the strategy/ directory, "
        f"apply the strategy to this example, and write your answer to `answer.txt` — "
        f"exactly `yes` or `no`, no other text."
    )

    cmd = [
        "claude",
        "--print",
        "--dangerously-skip-permissions",
        "--system-prompt-file", str(prompt_path),
        "--add-dir", str(strategy_dir),
        "--allowed-tools", "Read,Write,Edit,Bash,Glob,Grep",
        user_prompt,
    ]

    env = os.environ.copy()
    env["BASH_ENV"] = str(bashrc_path)
    env["AGENT_TYPE"] = "test"

    trace_file = trace_dir / f"test-{test_index:03d}-trace.txt"

    result = subprocess.run(
        cmd,
        cwd=str(test_folder),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    trace_file.write_text(result.stdout)

    # Read answer
    answer_path = test_folder / "answer.txt"
    answer = None
    if answer_path.exists():
        answer = answer_path.read_text().strip().lower()

    return {
        "index": test_index,
        "example_id": example["id"],
        "answer": answer,
        "exit_code": result.returncode,
    }


def main():
    env = get_env()
    scaffold_root = Path(env["SCAFFOLD_ROOT"])
    run_dir = Path(env["AGENT_RUN_DIR"])
    task_name = env["AGENT_TASK"]
    run_id = env["AGENT_RUN_ID"]

    data_test_dir = scaffold_root / "data" / task_name / "test"
    strategy_dir = run_dir / "strategy"
    trace_dir = scaffold_root / "agent-traces" / task_name / f"run-{run_id}"
    prompt_path = scaffold_root / "prompts" / "test-agent.md"
    bashrc_path = run_dir / "agent.bashrc"

    if not data_test_dir.exists():
        print(f"Error: No test data at {data_test_dir}")
        sys.exit(1)

    if not prompt_path.exists():
        print(f"Error: Test prompt not found at {prompt_path}")
        sys.exit(1)

    # Load task description from run metadata
    run_meta_path = run_dir / "run.json"
    if run_meta_path.exists():
        with open(run_meta_path) as f:
            run_meta = json.load(f)
        task_description = run_meta.get("task_meta", {}).get("description", task_name)
    else:
        task_description = task_name

    examples = collect_test_examples(data_test_dir)
    if not examples:
        print("No test examples found.")
        return

    print(f"Running {len(examples)} test(s) in parallel...")

    trace_dir.mkdir(parents=True, exist_ok=True)
    results = []

    max_workers = min(len(examples), int(os.environ.get("AGENT_TEST_MAX_WORKERS", "10")))
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, example in enumerate(examples):
            future = executor.submit(
                run_single_test,
                i, example, run_dir, strategy_dir, trace_dir,
                prompt_path, task_description, bashrc_path,
            )
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                results.append(result)
                status = f"answer={result['answer']}" if result["answer"] else "NO ANSWER"
                print(f"  Test {idx} ({result['example_id']}): {status}")
            except Exception as e:
                print(f"  Test {idx}: ERROR - {e}")
                results.append({"index": idx, "example_id": "?", "answer": None, "exit_code": -1})

    # Sort by index and write results CSV at the run root (sibling of test-NNN/ folders)
    results.sort(key=lambda r: r["index"])
    results_path = run_dir / "results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "example_id", "answer", "exit_code"])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    answered = [r for r in results if r["answer"] in ("yes", "no")]
    yes_count = sum(1 for r in answered if r["answer"] == "yes")
    no_count = sum(1 for r in answered if r["answer"] == "no")
    missing = len(results) - len(answered)

    print(f"\nResults: {yes_count} yes, {no_count} no, {missing} missing/invalid")
    print(f"Details saved to {results_path}")


if __name__ == "__main__":
    main()
