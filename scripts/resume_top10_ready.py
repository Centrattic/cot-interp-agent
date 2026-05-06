#!/usr/bin/env python3
"""Run the test phase for a fixed list of READY partitions (top10-only OOD runs).

A "ready" partition has a real STRATEGY.md but no results.csv. This script
re-runs the test phase only — no strategy work, no stub partitions. Already-
answered tests inside each partition are skipped (run_tests.py resume support).
"""

from __future__ import annotations
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agent_backend import get_agent_backend, load_bash_exports, prepare_codex_home  # noqa: E402

READY_PARTS = [
    # Pass 2: partitions with results.csv but ALL rows missing 'answer'
    # (stuck mid-test, results.csv stub left behind). Test phase re-fills.
    "agent-runs/atypical_cot_length/run-20260505-150535-955374/partition-004",
    "agent-runs/followup_confidence/run-20260505-150533-462153/partition-001",
    "agent-runs/followup_confidence/run-20260505-150533-462153/partition-005",
    "agent-runs/reasoning_termination/run-20260505-150534-263556/partition-006",
]


def run_test_phase(part_dir: Path, max_workers: int) -> tuple[str, int, str]:
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
    env["AGENT_TYPE"] = "test"
    env["AGENT_TEST_MAX_WORKERS"] = str(max_workers)
    # Pin model so resumes don't drift if config.toml changes.
    env["CODEX_MODEL"] = "gpt-5.4"
    env["CODEX_REASONING_EFFORT"] = "medium"

    cmd = [sys.executable, str(ROOT / "src" / "run_tests.py")]
    tag = f"{part_dir.parent.parent.name}/{part_dir.parent.name}/{part_dir.name}"
    print(f"[{tag}] launching test phase (workers={max_workers})", flush=True)
    p = subprocess.run(
        cmd, env=env, cwd=str(part_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    tail = "\n".join(p.stdout.splitlines()[-5:])
    return tag, p.returncode, tail


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-parallel", type=int, default=4,
                    help="Max partitions in flight at once (default 4)")
    ap.add_argument("--test-workers", type=int, default=3,
                    help="Per-partition test concurrency (default 3)")
    args = ap.parse_args()

    todo = []
    for rel in READY_PARTS:
        p = ROOT / rel
        sm = p / "strategy" / "STRATEGY.md"
        if (not sm.exists()) or sm.stat().st_size <= 200:
            print(f"  SKIP (stub): {rel}")
            continue
        todo.append(p)
    print(f"\n=== {len(todo)} partitions to resume (test phase only) ===\n")

    fails = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futs = {ex.submit(run_test_phase, p, args.test_workers): p for p in todo}
        for f in as_completed(futs):
            tag, code, tail = f.result()
            print(f"[{tag}] DONE exit={code}\n{tail}\n", flush=True)
            if code != 0:
                fails.append(tag)

    print("\n=== Final state ===")
    for rel in READY_PARTS:
        p = ROOT / rel
        ok = (p / "results.csv").exists()
        print(f"  {'OK ' if ok else 'XX '}{rel}")

    if fails:
        print(f"\n{len(fails)} failures: {fails}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
