#!/usr/bin/env python3
"""Parallel test runner.

Reads test examples from data/<task>/test/, creates a folder per test example
inside the current agent run's test/ directory, and launches one configured
agent CLI instance for each one in parallel.

Called by the `test` bash command from within a strategy agent session.
Expects environment variables: SCAFFOLD_ROOT, AGENT_RUN_DIR, AGENT_TASK, AGENT_RUN_ID
"""

import csv
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agent_backend import build_agent_launch_spec, get_agent_backend, supports_add_dirs


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
        logits_path = data_test_dir / f"{example_id}.logits.npz"
        examples.append({
            "id": example_id,
            "json_path": json_file,
            "npy_path": npy_path if npy_path.exists() else None,
            "logits_path": logits_path if logits_path.exists() else None,
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
    test_keep_fields: list[str] | None,
) -> dict:
    """Run a single test agent on one example. Returns result dict."""
    # Per-test folder sits directly under run_dir, sibling to strategy/
    test_folder = run_dir / f"test-{test_index:03d}"
    test_folder.mkdir(parents=True, exist_ok=True)

    # Resume support: short-circuit if this test already has a valid answer.
    existing_answer = test_folder / "answer.txt"
    if existing_answer.exists():
        prior = existing_answer.read_text().strip().lower()
        if prior in ("yes", "no"):
            return {
                "index": test_index,
                "example_id": example["id"],
                "answer": prior,
                "exit_code": 0,
            }

    # Filter the example before writing to test-NNN/example.json.
    # - Always drop `label` (ground truth).
    # - If test_keep_fields is set, keep ONLY those fields (whitelist).
    #   Otherwise keep everything except `label` (legacy behavior).
    src_data = example["data"]
    if test_keep_fields:
        redacted = {k: src_data[k] for k in test_keep_fields if k in src_data}
    else:
        redacted = {k: v for k, v in src_data.items() if k != "label"}
    with open(test_folder / "example.json", "w", encoding="utf-8") as f:
        json.dump(redacted, f, indent=2, ensure_ascii=False)
    if example["npy_path"]:
        shutil.copy2(example["npy_path"], test_folder / "example.npy")
    if example.get("logits_path"):
        shutil.copy2(example["logits_path"], test_folder / example["logits_path"].name)

    user_prompt = (
        f"You are classifying one test example (id={example['id']}).\n"
        f"The example is in `example.json` in your current directory "
        f"(ground-truth label has been removed).\n"
        f"Read STRATEGY.md (and any referenced files) from the strategy/ directory, "
        f"apply the strategy to this example, and write your answer to `answer.txt` — "
        f"exactly `yes` or `no`, no other text."
    )

    system_prompt = prompt_path.read_text(encoding="utf-8")

    env = os.environ.copy()
    env["BASH_ENV"] = str(bashrc_path)
    env["AGENT_TYPE"] = "test"
    env["AGENT_EXAMPLE_ID"] = example["id"]
    backend = get_agent_backend(env)

    run_cwd = test_folder
    if backend == "codex":
        run_cwd = run_dir
        user_prompt = (
            f"You are classifying one test example (id={example['id']}).\n"
            f"The example is in `test-{test_index:03d}/example.json` "
            f"(ground-truth label has been removed).\n"
            f"Read `strategy/STRATEGY.md` (and any referenced files) from the run root, "
            f"apply the strategy to this example, and write your answer to "
            f"`test-{test_index:03d}/answer.txt` — exactly `yes` or `no`, no other text."
        )

    launch = build_agent_launch_spec(
        backend=backend,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        add_dirs=[strategy_dir, trace_dir] if supports_add_dirs(backend) else None,
    )

    trace_base = trace_dir / f"test-{test_index:03d}-trace"

    try:
        result = subprocess.run(
            launch.cmd,
            cwd=str(run_cwd),
            env=env,
            input=launch.stdin_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            timeout=int(os.environ.get("AGENT_TEST_TIMEOUT_SEC", "600")),
        )
        stdout = result.stdout
        exit_code = result.returncode
    except subprocess.TimeoutExpired as e:
        stdout = (e.stdout or "") + f"\n\n[subprocess timed out after {e.timeout}s]"
        exit_code = -1

    from render_trace import write_trace_pair
    write_trace_pair(stdout, trace_base)

    # Read answer
    answer_path = test_folder / "answer.txt"
    answer = None
    if answer_path.exists():
        answer = answer_path.read_text().strip().lower()

    return {
        "index": test_index,
        "example_id": example["id"],
        "answer": answer,
        "exit_code": exit_code,
    }


def main():
    env = get_env()
    scaffold_root = Path(env["SCAFFOLD_ROOT"])
    run_dir = Path(env["AGENT_RUN_DIR"])
    task_name = env["AGENT_TASK"]
    run_id = env["AGENT_RUN_ID"]

    data_task = os.environ.get("AGENT_DATA_TASK", task_name)
    data_test_dir = scaffold_root / "data" / data_task / "test"
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

    # Load task description and the test_keep_fields whitelist from run metadata
    # (populated by scaffold.py from data/<task>/metadata.json).
    # In multi-partition mode AGENT_RUN_DIR is the partition dir, so look for
    # run.json at its parent.
    run_meta_path = run_dir / "run.json"
    if not run_meta_path.exists():
        run_meta_path = run_dir.parent / "run.json"
    task_description = task_name
    test_keep_fields: list[str] | None = None
    if run_meta_path.exists():
        with open(run_meta_path) as f:
            run_meta = json.load(f)
        task_meta = run_meta.get("task_meta", {})
        task_description = task_meta.get("description", task_name)
        tkf = task_meta.get("test_keep_fields")
        if isinstance(tkf, list) and tkf:
            test_keep_fields = tkf

    examples = collect_test_examples(data_test_dir)
    if not examples:
        print("No test examples found.")
        return

    # Round-robin partition slicing. Global sort order (by filename) is preserved;
    # partition k runs examples at indices [k, k+N, k+2N, ...].
    n_partitions = int(os.environ.get("AGENT_N_PARTITIONS", "1"))
    partition_idx = int(os.environ.get("AGENT_PARTITION_INDEX", "0"))
    if n_partitions > 1:
        examples = [ex for i, ex in enumerate(examples) if i % n_partitions == partition_idx]
        # Namespace trace files under a per-partition subdir
        trace_dir = trace_dir / f"partition-{partition_idx:03d}"
        print(f"Partition {partition_idx}/{n_partitions}: running {len(examples)} test(s) in parallel...")
    else:
        print(f"Running {len(examples)} test(s) in parallel...")

    trace_dir.mkdir(parents=True, exist_ok=True)
    results = []

    agent_backend = os.environ.get("AGENT_BACKEND", "").strip().lower()
    default_max_workers = "2" if agent_backend == "codex" else "10"
    max_workers = min(len(examples), int(os.environ.get("AGENT_TEST_MAX_WORKERS", default_max_workers)))
    # ThreadPoolExecutor (not ProcessPoolExecutor): work is I/O-bound (subprocess.run),
    # threads are simpler, avoid pickling, and don't leak grandchild pipe handles across
    # worker processes (which previously deadlocked shutdown on Windows).
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, example in enumerate(examples):
            future = executor.submit(
                run_single_test,
                i, example, run_dir, strategy_dir, trace_dir,
                prompt_path, task_description, bashrc_path,
                test_keep_fields,
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
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
