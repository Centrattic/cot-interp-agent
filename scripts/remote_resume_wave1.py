#!/usr/bin/env python3
"""Resume wave-1 partial runs on the remote box.

Wave-1 runs (started 18:54 UTC) burned through the active codex profile's
quota mid-way. After installing the rotating wrapper at /usr/local/bin/codex,
new codex calls auto-rotate across profiles. This script kicks off the
remaining strategy + test work for the wave-1 partial runs, scoped tightly so
we don't trample the still-live wave-2 scaffolds.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_backend import (  # noqa: E402
    build_agent_launch_spec,
    get_agent_backend,
    load_bash_exports,
    prepare_codex_home,
)
from prompt_builder import build_strategy_system_prompt  # noqa: E402

# Targeted gap fill: partial runs where we still need 3-per-task usable runs.
# atypical_cot_length already has 2/3 — need 1 more (the 8/10 partial below).
# followup_confidence is at 0/3 — claim 1 partial + 2 zero-strategy runs (the
# latter effectively fresh since all 10 partitions are stubs).
WAVE1_RUNS = [
    # atypical_answer_ood: 0/3 done, 3 partials to push over the line.
    "agent-runs/atypical_answer_ood/run-20260506-185418-925402",  # 9/10
    "agent-runs/atypical_answer_ood/run-20260506-185418-931951",  # 7/10
    "agent-runs/atypical_answer_ood/run-20260506-185418-940215",  # 4/10
    # atypical_cot_length: 2/3 done, finish this 8/10 partial.
    "agent-runs/atypical_cot_length/run-20260506-185418-930329",  # 8/10
    # followup_confidence: 0/3, claim 1 partial + 2 fresh-from-stubs.
    "agent-runs/followup_confidence/run-20260506-185418-930386",  # 7/10
    "agent-runs/followup_confidence/run-20260506-195639-040250",  # 0/10 (pre-wrapper failure)
    "agent-runs/followup_confidence/run-20260506-195649-038289",  # 0/10 (pre-wrapper failure)
]

BASE_ENV_OVERRIDES = {
    "CODEX_MODEL": "gpt-5.4",
    "CODEX_REASONING_EFFORT": "medium",
    "AGENT_TEST_MAX_WORKERS": "2",
    "AGENT_TEST_RETRY_ATTEMPTS": "8",
    "AGENT_TEST_RETRY_DELAY_SEC": "10",
    "TERM": "tmux",
    "PYTHON": "/usr/bin/python3",
}


def is_stub(part_dir: Path) -> bool:
    f = part_dir / "strategy" / "STRATEGY.md"
    return (not f.exists()) or f.stat().st_size <= 200


def has_data(part_dir: Path) -> bool:
    rcsv = part_dir / "results.csv"
    if not rcsv.exists():
        return False
    import csv as _csv
    try:
        rows = list(_csv.DictReader(open(rcsv)))
    except Exception:
        return False
    return any(r.get("answer", "") not in ("", None) for r in rows)


def _setup_env(part_dir: Path) -> dict[str, str]:
    bashrc = part_dir / "agent.bashrc"
    env = load_bash_exports(bashrc, os.environ.copy())
    if get_agent_backend(env) == "codex":
        target = (
            Path(env["CODEX_HOME"]).expanduser()
            if env.get("CODEX_HOME")
            else part_dir / ".codex-home"
        )
        env["CODEX_HOME"] = str(prepare_codex_home(target, os.environ))
    env["BASH_ENV"] = str(bashrc)
    env.update(BASE_ENV_OVERRIDES)
    return env


def run_strategy(part_dir: Path) -> tuple[str, int]:
    env = _setup_env(part_dir)
    env["AGENT_TYPE"] = "strategy"
    prompt_path = ROOT / "prompts" / "strategy-agent.md"
    user_prompt = (
        "Read README.md in the current directory for the task brief, available tools, "
        "and workspace layout. Develop a classification strategy, write it to STRATEGY.md, "
        "and run `run-tests` when ready."
    )
    run_meta = json.loads((part_dir.parent / "run.json").read_text(encoding="utf-8"))
    system_prompt = build_strategy_system_prompt(prompt_path.parent, run_meta.get("tools", []))
    backend = get_agent_backend(env)
    add_dirs = (
        [part_dir, ROOT / "agent-traces" / env["AGENT_TASK"] / f"run-{env['AGENT_RUN_ID']}"]
        if backend == "codex" else None
    )
    launch = build_agent_launch_spec(
        backend=backend, system_prompt=system_prompt,
        user_prompt=user_prompt, add_dirs=add_dirs,
    )
    strategy_dir = part_dir / "strategy"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{part_dir.parent.parent.name}/{part_dir.parent.name}/{part_dir.name}/STR"
    print(f"[{tag}] launching", flush=True)
    p = subprocess.run(
        launch.cmd, env=env, cwd=str(strategy_dir), input=launch.stdin_text,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return tag, p.returncode


def run_test_phase(part_dir: Path) -> tuple[str, int]:
    env = _setup_env(part_dir)
    env["AGENT_TYPE"] = "test"
    cmd = [sys.executable, str(ROOT / "src" / "run_tests.py")]
    tag = f"{part_dir.parent.parent.name}/{part_dir.parent.name}/{part_dir.name}/TEST"
    print(f"[{tag}] launching", flush=True)
    p = subprocess.run(
        cmd, env=env, cwd=str(part_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    return tag, p.returncode


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-parallel", type=int, default=8)
    args = ap.parse_args()

    jobs: list[tuple[str, Path]] = []  # ('strategy' or 'test', part_dir)
    for rel in WAVE1_RUNS:
        run_dir = ROOT / rel
        if not run_dir.exists():
            print(f"  MISSING: {rel}")
            continue
        for p in sorted(run_dir.iterdir()):
            if not (p.is_dir() and p.name.startswith("partition-")):
                continue
            if has_data(p):
                continue
            kind = "test" if not is_stub(p) else "strategy"
            jobs.append((kind, p))

    n_str = sum(1 for k, _ in jobs if k == "strategy")
    n_tst = sum(1 for k, _ in jobs if k == "test")
    print(f"\n=== queued: {len(jobs)} (strategy={n_str} test-phase={n_tst}) ===\n", flush=True)

    fails = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futs = {}
        for kind, part_dir in jobs:
            f = ex.submit(run_strategy if kind == "strategy" else run_test_phase, part_dir)
            futs[f] = (kind, part_dir)
        for f in as_completed(futs):
            kind, part_dir = futs[f]
            tag, rc = f.result()
            mark = "OK" if rc == 0 else "XX"
            print(f"[{tag}] {mark} exit={rc}", flush=True)
            if rc != 0:
                fails.append(tag)

    print(f"\n=== final ===")
    for rel in WAVE1_RUNS:
        run_dir = ROOT / rel
        if not run_dir.exists(): continue
        done = sum(1 for p in run_dir.iterdir()
                   if p.is_dir() and p.name.startswith("partition-") and has_data(p))
        marker = "USABLE" if done == 10 else f"done={done}/10"
        print(f"  {rel}: {marker}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
