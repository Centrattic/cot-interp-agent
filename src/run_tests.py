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

from agent_backend import (
    build_agent_launch_spec,
    get_agent_backend,
    load_bash_exports,
    supports_add_dirs,
)

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

        def _find_sidecar(suffix: str) -> Path | None:
            for d in filter(None, (data_test_dir, sae_source_dir)):
                candidate = d / f"{example_id}{suffix}"
                if candidate.exists():
                    return candidate
            return None

        examples.append({
            "id": example_id,
            "json_path": json_file,
            "npy_path": _find_sidecar(".npy"),
            "sae_npz_path": _find_sidecar(".sae.npz"),
            "logits_path": _find_sidecar(".logits.npz"),
            "data": data,
        })
    return examples


def _load_answer_quick(p: Path):
    """Parse an answer.txt. Returns 1, 0, or None.

    Returns None if the file doesn't exist yet, or contains anything other
    than exactly ``yes`` / ``no`` (case-insensitive, whitespace stripped).
    Used to decide whether a test agent is done and can be shut down early.
    """
    if not p.exists():
        return None
    try:
        a = p.read_text(encoding="utf-8").strip().lower()
    except OSError:
        return None
    if a == "yes": return 1
    if a == "no":  return 0
    return None


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
    local_strategy_dir = test_folder / "strategy"

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
    if example.get("logits_path"):
        shutil.copy2(example["logits_path"], test_folder / example["logits_path"].name)

    # Materialize a local strategy/ view inside the test folder so the agent
    # never needs to traverse to sibling directories.
    if not local_strategy_dir.exists():
        try:
            local_strategy_dir.symlink_to(strategy_dir, target_is_directory=True)
        except OSError:
            shutil.copytree(strategy_dir, local_strategy_dir)

    user_prompt = (
        f"You are classifying one test example (id={example['id']}).\n"
        f"The example is in `example.json` in your current directory "
        f"(ground-truth label has been removed).\n"
        f"Read `strategy/STRATEGY.md` (and any referenced files) from the local "
        f"`strategy/` directory, "
        f"apply the strategy to this example, and write your answer to `answer.txt` — "
        f"exactly `yes` or `no`, no other text."
    )

    system_prompt = prompt_path.read_text(encoding="utf-8")
    project_settings = Path(os.environ["SCAFFOLD_ROOT"]) / ".claude" / "settings.json"
    env = load_bash_exports(bashrc_path, os.environ.copy())
    env["BASH_ENV"] = str(bashrc_path)
    env["AGENT_TYPE"] = "test"
    env["AGENT_EXAMPLE_ID"] = example["id"]
    backend = get_agent_backend(env)

    run_cwd = test_folder

    launch = build_agent_launch_spec(
        backend=backend,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        add_dirs=[strategy_dir, trace_dir] if supports_add_dirs(backend) else None,
        project_settings=project_settings if backend == "claude" else None,
    )

    trace_base = trace_dir / f"test-{test_index:03d}-trace"

    # Auto-shutdown: the test agent's only job is to write a valid yes/no to
    # answer.txt. Once that file has a clean answer we terminate the agent
    # subprocess immediately instead of waiting for it to exit on its own;
    # otherwise the agent commonly spends another 30-60s reflecting /
    # exploring after writing answer.txt, which is wasted wall time and
    # tokens.
    import threading as _threading
    import time as _time

    answer_path = test_folder / "answer.txt"
    timeout_sec = int(os.environ.get("AGENT_TEST_TIMEOUT_SEC", "600"))
    grace_after_answer = int(os.environ.get("AGENT_TEST_GRACE_SEC", "3"))

    p = subprocess.Popen(
        launch.cmd, cwd=str(run_cwd), env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8",
    )
    if launch.stdin_text is not None:
        p.stdin.write(launch.stdin_text)
    p.stdin.close()

    out_chunks: list[str] = []
    def _drain():
        assert p.stdout is not None
        for line in iter(p.stdout.readline, ""):
            out_chunks.append(line)
    drainer = _threading.Thread(target=_drain, daemon=True)
    drainer.start()

    start = _time.time()
    kill_reason = None
    while True:
        if p.poll() is not None:
            break
        if _time.time() - start > timeout_sec:
            p.terminate()
            kill_reason = "timeout"
            break
        if _load_answer_quick(answer_path) is not None:
            # Give the agent a short grace period to let final stream chunks flush
            _time.sleep(grace_after_answer)
            if p.poll() is None:
                p.terminate()
                kill_reason = "answer-written"
            break
        _time.sleep(1)
    try:
        p.wait(timeout=10)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
    drainer.join(timeout=2)
    exit_code = p.returncode if p.returncode is not None else -1
    stdout = "".join(out_chunks)
    if kill_reason == "timeout":
        stdout += f"\n\n[subprocess timed out after {timeout_sec}s]"

    from render_trace import write_trace_pair
    write_trace_pair(stdout, trace_base)

    # Read answer
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
    if missing:
        sys.exit(1)


if __name__ == "__main__":
    main()
