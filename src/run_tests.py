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
from prompt_builder import build_test_system_prompt

FROZEN_STRATEGY_DIRNAME = ".strategy-frozen"
PRIVATE_VALIDATION_DIRNAME = ".validation-private"

REVISER_UPDATES_START = "<!-- REVISER_UPDATES_START -->"
REVISER_UPDATES_END = "<!-- REVISER_UPDATES_END -->"
REVISER_REPL_START = "<!-- REVISER_REPLACEMENTS_START -->"
REVISER_REPL_END = "<!-- REVISER_REPLACEMENTS_END -->"


def _extract_between_markers(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0:
        return ""
    start += len(start_marker)
    end = text.find(end_marker, start)
    if end < 0:
        return ""
    return text[start:end].strip()


def _inject_reviser_frame(strategy_text: str) -> str:
    if all(
        m in strategy_text
        for m in (
            REVISER_UPDATES_START,
            REVISER_UPDATES_END,
            REVISER_REPL_START,
            REVISER_REPL_END,
        )
    ):
        return strategy_text

    frame = f"""

## Reviser Additions (append-only, newest-first)

These revisions are layered on top of the legacy text above and override only when they are explicit about that override.
\n- Place every new or changed rule in `Revision Blocks`.
\n- If you need a surgical text replacement, place entries in `Optional Replacements`.

### Revision Blocks
This run adds/extends rules for the next run only.
\n<!-- REVISER_UPDATES_START -->
<!-- REVISER_UPDATES_END -->

### Optional Replacements
Use only in this section. Keep each replacement minimal.
\n<!-- REVISER_REPLACEMENTS_START -->
<!-- REVISER_REPLACEMENTS_END -->
""".rstrip()
    return strategy_text.rstrip() + "\n\n" + frame


def _merge_reviser_strategy(base_text: str, revised_text: str) -> str:
    """Merge constrained reviser updates into the base strategy.

    Full-file rewrites are ignored unless they modify the allowed reviser frame.
    """
    base_text = base_text.strip()
    updates = _extract_between_markers(revised_text, REVISER_UPDATES_START, REVISER_UPDATES_END)
    replacements = _extract_between_markers(revised_text, REVISER_REPL_START, REVISER_REPL_END)

    if not updates and not replacements:
        # No constrained content captured; return base strategy unchanged so we
        # do not accidentally accept accidental wide-range rewrites.
        return base_text + "\n"

    updates_block = (
        "## Reviser Additions (append-only, newest-first)\n"
        "### Revision Blocks (newer rules override older rules)\n"
    )
    if updates:
        updates_block += f"{updates}\n\n"
    if replacements:
        updates_block += "### Optional Replacements (legacy-text scoped)\n" + f"{replacements}\n"

    if not updates_block.strip().endswith("\n"):
        updates_block += "\n"

    return base_text + "\n\n" + updates_block

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


def private_validation_dir(scaffold_root: Path, task_name: str, run_id: str) -> Path:
    partition_idx = os.environ.get("AGENT_PARTITION_INDEX")
    leaf = "single" if not partition_idx or os.environ.get("AGENT_N_PARTITIONS", "1") == "1" else f"partition-{int(partition_idx):03d}"
    return scaffold_root / PRIVATE_VALIDATION_DIRNAME / task_name / run_id / leaf


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


def freeze_strategy_dir(strategy_dir: Path, run_dir: Path) -> Path:
    """Snapshot strategy/ at the moment run-tests starts.

    Test agents read this frozen copy, not the live strategy directory. This
    makes the evaluated strategy exactly the one that existed when the
    strategy agent invoked run-tests, even if that parent agent later keeps
    thinking or attempts edits after seeing held-out results.
    """
    frozen = run_dir / FROZEN_STRATEGY_DIRNAME
    tmp = run_dir / f"{FROZEN_STRATEGY_DIRNAME}.tmp"
    if tmp.exists() or tmp.is_symlink():
        if tmp.is_dir() and not tmp.is_symlink():
            shutil.rmtree(tmp)
        else:
            tmp.unlink()
    if frozen.exists() or frozen.is_symlink():
        if frozen.is_dir() and not frozen.is_symlink():
            shutil.rmtree(frozen)
        else:
            frozen.unlink()
    ignore = shutil.ignore_patterns("__pycache__", ".strategy-frozen", ".strategy-frozen.tmp")
    shutil.copytree(strategy_dir, tmp, ignore=ignore)
    tmp.replace(frozen)
    return frozen


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
    folder_prefix: str = "test",
) -> dict:
    """Run a single test agent on one example. Returns result dict."""
    # Per-test folder sits directly under run_dir, sibling to strategy/
    test_folder = run_dir / f"{folder_prefix}-{test_index:03d}"
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

    system_prompt = build_test_system_prompt(prompt_path.parent, run_dir)
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

    trace_base = trace_dir / f"{folder_prefix}-{test_index:03d}-trace"

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
    raw_trace_path = trace_base.with_suffix(".jsonl")
    raw_trace_path.parent.mkdir(parents=True, exist_ok=True)
    def _drain():
        assert p.stdout is not None
        with raw_trace_path.open("w", encoding="utf-8") as raw_f:
            for line in iter(p.stdout.readline, ""):
                out_chunks.append(line)
                raw_f.write(line)
                raw_f.flush()
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


def answer_to_label(answer: str | None) -> int | None:
    if answer == "yes":
        return 1
    if answer == "no":
        return 0
    return None


def summarize_results(results: list[dict]) -> dict:
    counts = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "missing": 0}
    for r in results:
        pred = answer_to_label(r.get("answer"))
        try:
            gt = int(r.get("ground_truth"))
        except (TypeError, ValueError):
            gt = None
        if pred is None or gt not in (0, 1):
            counts["missing"] += 1
        elif pred == 1 and gt == 1:
            counts["tp"] += 1
        elif pred == 0 and gt == 0:
            counts["tn"] += 1
        elif pred == 1 and gt == 0:
            counts["fp"] += 1
        elif pred == 0 and gt == 1:
            counts["fn"] += 1
    n = counts["tp"] + counts["tn"] + counts["fp"] + counts["fn"]
    counts["n"] = n
    counts["accuracy"] = (counts["tp"] + counts["tn"]) / n if n else 0.0
    counts["tpr"] = counts["tp"] / (counts["tp"] + counts["fn"]) if (counts["tp"] + counts["fn"]) else 0.0
    counts["tnr"] = counts["tn"] / (counts["tn"] + counts["fp"]) if (counts["tn"] + counts["fp"]) else 0.0
    counts["gmean_squared"] = counts["tpr"] * counts["tnr"]
    return counts


def run_examples(
    *,
    examples: list[dict],
    run_dir: Path,
    strategy_dir: Path,
    trace_dir: Path,
    prompt_path: Path,
    task_description: str,
    bashrc_path: Path,
    test_keep_fields: list[str] | None,
    folder_prefix: str,
    results_name: str,
    label: str,
) -> tuple[list[dict], dict]:
    trace_dir.mkdir(parents=True, exist_ok=True)
    results = []
    if not examples:
        raise ValueError(f"no {label.lower()} examples to run")

    agent_backend = os.environ.get("AGENT_BACKEND", "").strip().lower()
    default_max_workers = "30" if agent_backend == "codex" else "10"
    max_workers = min(len(examples), int(os.environ.get("AGENT_TEST_MAX_WORKERS", default_max_workers)))
    results_by_index: dict[int, dict] = {}
    results_path = run_dir / results_name

    def _run_batch(indexed_examples: list[tuple[int, dict]], attempt_label: str) -> None:
        if not indexed_examples:
            return
        with ThreadPoolExecutor(max_workers=min(max_workers, len(indexed_examples))) as executor:
            futures = {}
            for i, example in indexed_examples:
                future = executor.submit(
                    run_single_test,
                    i, example, run_dir, strategy_dir, trace_dir,
                    prompt_path, task_description, bashrc_path,
                    test_keep_fields, folder_prefix,
                )
                futures[future] = i

            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    results_by_index[idx] = result
                    status = f"answer={result['answer']}" if result["answer"] else "NO ANSWER"
                    print(f"  {label} {idx} ({result['example_id']}){attempt_label}: {status}")
                except Exception as e:
                    print(f"  {label} {idx}{attempt_label}: ERROR - {e}")
                    results_by_index[idx] = {
                        "index": idx,
                        "example_id": examples[idx]["id"] if idx < len(examples) else "?",
                        "answer": None,
                        "ground_truth": "",
                        "exit_code": -1,
                    }

    def _write_outputs() -> tuple[list[dict], dict]:
        results = [results_by_index[i] for i in sorted(results_by_index)]
        with open(results_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "example_id", "answer", "ground_truth", "exit_code"])
            writer.writeheader()
            writer.writerows(results)

        summary = summarize_results(results)
        summary_path = run_dir / results_name.replace(".csv", "_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return results, summary

    _run_batch(list(enumerate(examples)), "")
    results, summary = _write_outputs()

    retry_attempts = int(os.environ.get("AGENT_TEST_RETRY_ATTEMPTS", "1"))
    retry_delay = float(os.environ.get("AGENT_TEST_RETRY_DELAY_SEC", "15"))
    for attempt in range(1, retry_attempts + 1):
        missing = [
            (r["index"], examples[r["index"]])
            for r in results
            if answer_to_label(r.get("answer")) is None
        ]
        if not missing:
            break
        print(f"\nRetrying {len(missing)} missing/invalid {label.lower()} prediction(s), attempt {attempt}/{retry_attempts}...")
        if retry_delay > 0:
            import time as _time
            _time.sleep(retry_delay)
        _run_batch(missing, f" retry {attempt}")
        results, summary = _write_outputs()

    print(
        f"\n{label} results: n={summary['n']} missing={summary['missing']} "
        f"acc={summary['accuracy']:.1%} TPR={summary['tpr']:.2f} "
        f"TNR={summary['tnr']:.2f} gmean²={summary['gmean_squared']:.3f}"
    )
    print(f"Details saved to {results_path}")
    return results, summary


def setup_mini_reviser_workspace(run_dir: Path) -> Path:
    """Build an isolated workspace exposing only the inputs the mini reviser
    is allowed to see: STRATEGY.md, the original few-shot JSONs (no SAE
    sidecars), each `val-*/example.json` (no `.sae.npz`, no `answer.txt`, no
    per-example strategy snapshot), and the two validation_results files.
    Recreated fresh on each call.
    """
    workspace = run_dir / ".reviser-mini"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)

    strategy_md = run_dir / "strategy" / "STRATEGY.md"
    shutil.copy2(strategy_md, workspace / "STRATEGY.md")

    few_shot_src = run_dir / "strategy" / "few-shot"
    if few_shot_src.is_dir():
        few_shot_dst = workspace / "few-shot"
        few_shot_dst.mkdir(parents=True, exist_ok=True)
        for json_file in sorted(few_shot_src.glob("*.json")):
            shutil.copy2(json_file, few_shot_dst / json_file.name)

    for val_dir in sorted(run_dir.glob("val-*")):
        if not val_dir.is_dir():
            continue
        example_json = val_dir / "example.json"
        if not example_json.exists():
            continue
        dst = workspace / val_dir.name
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(example_json, dst / "example.json")

    for fname in ("validation_results.csv", "validation_results_summary.json"):
        src = run_dir / fname
        if src.exists():
            shutil.copy2(src, workspace / fname)

    return workspace


def launch_reviser_agent(
    *,
    scaffold_root: Path,
    run_dir: Path,
    trace_dir: Path,
    bashrc_path: Path,
    validation_summary: dict,
    mini: bool = False,
) -> int:
    """Launch a reviser that edits live strategy/ using validation feedback.

    When ``mini`` is True, the reviser is sandboxed to a freshly-built
    `.reviser-mini/` workspace containing only STRATEGY.md, val-*/example.json,
    and the two validation_results files. The revised STRATEGY.md is copied
    back into `strategy/` after the agent exits.
    """
    strategy_dir = run_dir / "strategy"
    strategy_path = strategy_dir / "STRATEGY.md"
    prompt_filename = "reviser-agent-mini.md" if mini else "reviser-agent.md"
    reviser_prompt_path = scaffold_root / "prompts" / prompt_filename
    if reviser_prompt_path.exists():
        system_prompt = reviser_prompt_path.read_text(encoding="utf-8")
    else:
        system_prompt = (
            "You are a strategy revision agent. You may inspect the strategy "
            "workspace, validation examples, validation traces, and validation "
            "results. Revise strategy/STRATEGY.md to improve validation failure "
            "modes. Do not run final tests and do not inspect data/<task>/test."
        )

    base_strategy = ""
    if strategy_path.exists():
        base_strategy = strategy_path.read_text(encoding="utf-8")
        strategy_path.write_text(_inject_reviser_frame(base_strategy), encoding="utf-8")

    if mini:
        workspace = setup_mini_reviser_workspace(run_dir)
        cwd = workspace
        agent_add_dirs = [workspace]
        mini_strategy = cwd / "STRATEGY.md"
        if base_strategy:
            mini_strategy.write_text(_inject_reviser_frame(base_strategy), encoding="utf-8")
        user_prompt = (
            "A draft classification strategy has just been evaluated on a "
            "private validation split. Revise `STRATEGY.md` in this directory "
            "based on the validation examples and results.\n\n"
            f"Validation summary: {json.dumps(validation_summary, sort_keys=True)}\n\n"
            "Files available in this directory (and nothing else):\n"
            "- `STRATEGY.md` — edit this in place\n"
            "- `few-shot/*.json` — the original few-shot examples with labels\n"
            "- `val-*/example.json` — one per validation example; labels stripped\n"
            "- `validation_results.csv` — predicted answer, ground truth, exit code per example\n"
            "- `validation_results_summary.json` — aggregate metrics\n\n"
            "There are no validation traces or research tools available. "
            "Edit `STRATEGY.md` in place and then stop."
        )
    else:
        cwd = run_dir
        agent_add_dirs = [run_dir, trace_dir]
        user_prompt = (
            "A draft classification strategy has just been evaluated on a private "
            "validation split. Revise `strategy/STRATEGY.md` using only the "
            "few-shot examples, validation examples/traces, and validation results "
            "available in this run directory.\n\n"
            f"Validation summary: {json.dumps(validation_summary, sort_keys=True)}\n\n"
            "Inputs you may inspect:\n"
            "- `strategy/` including the current STRATEGY.md and few-shot examples\n"
            "- `val-*` directories; their example labels were stripped before agents saw them\n"
            "- `validation_results.csv` and `validation_results_summary.json`\n"
            f"- validation traces under `{trace_dir}`\n\n"
            "Do not launch `run-tests`. Do not inspect held-out final test data. "
            "Edit `strategy/STRATEGY.md` in place and then stop."
        )

    env = load_bash_exports(bashrc_path, os.environ.copy())
    env["BASH_ENV"] = str(bashrc_path)
    env["AGENT_TYPE"] = "strategy"
    backend = get_agent_backend(env)
    launch = build_agent_launch_spec(
        backend=backend,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        add_dirs=agent_add_dirs if supports_add_dirs(backend) else None,
        project_settings=scaffold_root / ".claude" / "settings.json" if backend == "claude" else None,
    )
    reviser_trace = trace_dir / "reviser-trace"
    timeout_sec = int(os.environ.get("AGENT_REVISER_TIMEOUT_SEC", "900"))

    label = "mini reviser" if mini else "validation reviser"
    print(f"Launching {label} agent...")
    try:
        p = subprocess.run(
            launch.cmd,
            cwd=str(cwd),
            env=env,
            input=launch.stdin_text,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            timeout=timeout_sec,
        )
        stdout = p.stdout
        return_code = p.returncode
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        stdout += f"\n\n[reviser timed out after {timeout_sec}s]"
        return_code = -1
    from render_trace import write_trace_pair
    write_trace_pair(stdout, reviser_trace)

    revised = (cwd / "STRATEGY.md") if mini else strategy_path
    if base_strategy:
        revised_text = (
            revised.read_text(encoding="utf-8")
            if revised.exists()
            else _inject_reviser_frame(base_strategy)
        )
        strategy_path.write_text(
            _merge_reviser_strategy(base_strategy, revised_text),
            encoding="utf-8",
        )

    print(f"{label.capitalize()} finished (exit code {return_code})")
    return return_code


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

    # Round-robin partition slicing. Global sort order (by filename) is preserved;
    # partition k runs examples at indices [k, k+N, k+2N, ...].
    n_partitions = int(os.environ.get("AGENT_N_PARTITIONS", "1"))
    partition_idx = int(os.environ.get("AGENT_PARTITION_INDEX", "0"))
    if n_partitions > 1:
        # Namespace trace files under a per-partition subdir
        trace_dir = trace_dir / f"partition-{partition_idx:03d}"

    if os.environ.get("AGENT_VALIDATE") == "1":
        val_dir = private_validation_dir(scaffold_root, task_name, run_id)
        if not val_dir.is_dir():
            print(f"Error: validation mode enabled but private validation dir is missing: {val_dir}")
            sys.exit(1)
        val_examples = collect_test_examples(val_dir)
        if not val_examples:
            print(f"Error: no validation examples found in {val_dir}")
            sys.exit(1)
        try:
            from tools.sae_encode import precompute_dir
            precompute_dir(val_dir)
        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: SAE precompute for validation data failed: {e}")

        draft_strategy = freeze_strategy_dir(strategy_dir, run_dir)
        val_trace_dir = trace_dir / "validation"
        print(f"Running {len(val_examples)} validation example(s) before final tests...")
        _, val_summary = run_examples(
            examples=val_examples,
            run_dir=run_dir,
            strategy_dir=draft_strategy,
            trace_dir=val_trace_dir,
            prompt_path=prompt_path,
            task_description=task_description,
            bashrc_path=bashrc_path,
            test_keep_fields=test_keep_fields,
            folder_prefix="val",
            results_name="validation_results.csv",
            label="Validation",
        )
        launch_reviser_agent(
            scaffold_root=scaffold_root,
            run_dir=run_dir,
            trace_dir=trace_dir,
            bashrc_path=bashrc_path,
            validation_summary=val_summary,
            mini=os.environ.get("AGENT_VALIDATE_MINI") == "1",
        )

    if not data_test_dir.exists():
        print(f"Error: No test data at {data_test_dir}")
        sys.exit(1)

    examples = collect_test_examples(data_test_dir, sae_source_dir=sae_source_dir)
    if not examples:
        print("No test examples found.")
        return

    # Precompute SAE activations for final test examples only after validation
    # and revision have completed, so validate mode does not touch held-out
    # test data before the reviser has finished.
    try:
        from tools.sae_encode import precompute_dir
        precompute_dir(data_test_dir)
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: SAE precompute for test data failed: {e}")

    if n_partitions > 1:
        examples = [ex for i, ex in enumerate(examples) if i % n_partitions == partition_idx]
        print(f"Partition {partition_idx}/{n_partitions}: running {len(examples)} test(s) in parallel...")
    else:
        print(f"Running {len(examples)} test(s) in parallel...")

    frozen_strategy_dir = freeze_strategy_dir(strategy_dir, run_dir)
    results, summary = run_examples(
        examples=examples,
        run_dir=run_dir,
        strategy_dir=frozen_strategy_dir,
        trace_dir=trace_dir,
        prompt_path=prompt_path,
        task_description=task_description,
        bashrc_path=bashrc_path,
        test_keep_fields=test_keep_fields,
        folder_prefix="test",
        results_name="results.csv",
        label="Test",
    )
    if summary["missing"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
