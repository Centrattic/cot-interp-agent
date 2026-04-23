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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow importing from src/ (for tools.sae_encode)
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))


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


def collect_test_examples(data_test_dir: Path, sae_source_dir: Path | None = None) -> list[dict]:
    """Collect test examples from JSON files.

    Sidecars (.npy activations, .sae.npz SAE caches) may live either next to
    the JSON in `data_test_dir` or in a separate `sae_source_dir` (the raw
    source split the scaffold task was derived from). We check both.
    """
    examples = []
    for json_file in sorted(data_test_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
        example_id = json_file.stem

        npy_path = None
        for d in filter(None, (data_test_dir, sae_source_dir)):
            candidate = d / f"{example_id}.npy"
            if candidate.exists():
                npy_path = candidate
                break
        sae_npz_path = None
        for d in filter(None, (data_test_dir, sae_source_dir)):
            candidate = d / f"{example_id}.sae.npz"
            if candidate.exists():
                sae_npz_path = candidate
                break

        examples.append({
            "id": example_id,
            "json_path": json_file,
            "npy_path": npy_path,
            "sae_npz_path": sae_npz_path,
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
                "ground_truth": example["data"].get("label", ""),
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
    if example.get("sae_npz_path"):
        shutil.copy2(example["sae_npz_path"], test_folder / "example.sae.npz")

    user_prompt = (
        f"You are classifying one test example (id={example['id']}).\n"
        f"The example is in `example.json` in your current directory "
        f"(ground-truth label has been removed).\n"
        f"Read STRATEGY.md (and any referenced files) from the strategy/ directory, "
        f"apply the strategy to this example, and write your answer to `answer.txt` — "
        f"exactly `yes` or `no`, no other text."
    )

    # --allowed-tools and --add-dir are variadic in the claude CLI, so they
    # greedily consume trailing positional args. Pass the user prompt via stdin.
    # --output-format stream-json emits full turn-by-turn events; we write both
    # the raw JSONL and a rendered .txt per test agent.
    system_prompt = prompt_path.read_text(encoding="utf-8")
    cmd = [
        "claude",
        "--print",
        "--dangerously-skip-permissions",
        "--output-format", "stream-json",
        "--verbose",  # required by --output-format stream-json
        "--system-prompt", system_prompt,
        "--add-dir", str(strategy_dir),
        "--allowed-tools", "Read,Write,Edit,Bash,Glob,Grep",
    ]

    env = os.environ.copy()
    env["BASH_ENV"] = str(bashrc_path)
    env["AGENT_TYPE"] = "test"
    env["AGENT_EXAMPLE_ID"] = example["id"]

    trace_base = trace_dir / f"test-{test_index:03d}-trace"

    try:
        result = subprocess.run(
            cmd,
            cwd=str(test_folder),
            env=env,
            input=user_prompt,
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
        "ground_truth": example["data"].get("label", ""),
        "exit_code": exit_code,
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

    # Load task description and the test_keep_fields whitelist from run metadata
    # (populated by scaffold.py from data/<task>/metadata.json).
    # In multi-partition mode AGENT_RUN_DIR is the partition dir, so look for
    # run.json at its parent.
    run_meta_path = run_dir / "run.json"
    if not run_meta_path.exists():
        run_meta_path = run_dir.parent / "run.json"
    task_description = task_name
    test_keep_fields: list[str] | None = None
    # SAE sidecars (.sae.npz) typically live in the raw source split rather
    # than next to the whitelisted JSONs in data/<task>/test/. Resolved from
    # metadata.source + test_split when available.
    sae_source_dir: Path | None = None
    if run_meta_path.exists():
        with open(run_meta_path) as f:
            run_meta = json.load(f)
        task_meta = run_meta.get("task_meta", {})
        task_description = task_meta.get("description", task_name)
        tkf = task_meta.get("test_keep_fields")
        if isinstance(tkf, list) and tkf:
            test_keep_fields = tkf
        src = task_meta.get("source")
        if src:
            cand = Path(src) / task_meta.get("test_split", "test")
            if cand.is_dir():
                sae_source_dir = cand

    examples = collect_test_examples(data_test_dir, sae_source_dir=sae_source_dir)
    if not examples:
        print("No test examples found.")
        return

    # Precompute SAE activations for test examples before launching agents
    try:
        from tools.sae_encode import precompute_dir
        precompute_dir(data_test_dir)
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: SAE precompute for test data failed: {e}")

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

    max_workers = min(len(examples), int(os.environ.get("AGENT_TEST_MAX_WORKERS", "10")))
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
                results.append({"index": idx, "example_id": "?", "answer": None, "ground_truth": "", "exit_code": -1})

    # Sort by index and write results CSV at the run root (sibling of test-NNN/ folders)
    results.sort(key=lambda r: r["index"])
    results_path = run_dir / "results.csv"
    with open(results_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "example_id", "answer", "ground_truth", "exit_code"])
        writer.writeheader()
        writer.writerows(results)

    # Summary
    answered = [r for r in results if r["answer"] in ("yes", "no")]
    yes_count = sum(1 for r in answered if r["answer"] == "yes")
    no_count = sum(1 for r in answered if r["answer"] == "no")
    missing = len(results) - len(answered)
    correct = sum(1 for r in answered if r["answer"] == r.get("ground_truth"))

    print(f"\nResults: {yes_count} yes, {no_count} no, {missing} missing/invalid")
    if answered:
        print(f"Accuracy: {correct}/{len(answered)} = {correct/len(answered):.1%}")
    print(f"Details saved to {results_path}")


if __name__ == "__main__":
    main()
