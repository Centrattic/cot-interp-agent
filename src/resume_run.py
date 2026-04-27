#!/usr/bin/env python3
"""Resume incomplete partitions of a multi-strategy run.

For each partition under agent-runs/<task>/<run_id>/partition-NNN/:
  - If STRATEGY.md is a stub (strategy agent never finished), re-launch the
    strategy agent. The strategy agent itself calls `test`, which spawns the
    per-partition test agents.
  - If STRATEGY.md exists but results.csv does not, run the test phase
    (src/run_tests.py). Already-answered tests are skipped (resume support
    in run_tests.run_single_test).

Concurrency: at most --max-parallel partitions are processed at once. Each
partition internally runs up to AGENT_TEST_MAX_WORKERS test agents.
"""

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agent_backend import (
    build_agent_launch_spec,
    get_agent_backend,
    load_bash_exports,
    prepare_codex_home,
)
from prompt_builder import build_strategy_system_prompt

ROOT = Path(__file__).resolve().parent.parent

def is_stub_strategy(part_dir: Path) -> bool:
    f = part_dir / "strategy" / "STRATEGY.md"
    return (not f.exists()) or f.stat().st_size <= 200


def has_results(part_dir: Path) -> bool:
    return (part_dir / "results.csv").exists()


def ensure_codex_home(part_dir: Path, env: dict[str, str]) -> None:
    if get_agent_backend(env) != "codex":
        return
    target = Path(env.get("CODEX_HOME", "")).expanduser() if env.get("CODEX_HOME") else part_dir / ".codex-home"
    env["CODEX_HOME"] = str(prepare_codex_home(target, os.environ))


def run_strategy_agent(part_dir: Path) -> int:
    bashrc = part_dir / "agent.bashrc"
    env = load_bash_exports(bashrc, os.environ.copy())
    ensure_codex_home(part_dir, env)
    env["BASH_ENV"] = str(bashrc)
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
    launch = build_agent_launch_spec(
        backend=backend,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        add_dirs=[part_dir, ROOT / "agent-traces" / env["AGENT_TASK"] / f"run-{env['AGENT_RUN_ID']}"] if backend == "codex" else None,
    )
    strategy_dir = part_dir / "strategy"
    tag = f"{part_dir.parent.name}/{part_dir.name}"
    print(f"[{tag}] STRATEGY agent launching (backend={backend})", flush=True)
    p = subprocess.run(
        launch.cmd, env=env, cwd=str(strategy_dir), input=launch.stdin_text,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    # Persist strategy trace
    run_id = env.get("AGENT_RUN_ID", "")
    task = env.get("AGENT_TASK", "")
    trace_dir = ROOT / "agent-traces" / task / f"run-{run_id}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_base = trace_dir / f"{part_dir.name}-strategy-resume-trace"
    sys.path.insert(0, str(ROOT / "src"))
    from render_trace import write_trace_pair
    write_trace_pair(p.stdout, trace_base)
    print(f"[{tag}] STRATEGY exit={p.returncode}", flush=True)
    return p.returncode


def run_test_phase(part_dir: Path, max_workers: int) -> int:
    bashrc = part_dir / "agent.bashrc"
    env = load_bash_exports(bashrc, os.environ.copy())
    ensure_codex_home(part_dir, env)
    env["BASH_ENV"] = str(bashrc)
    env["AGENT_TEST_MAX_WORKERS"] = str(max_workers)
    cmd = [sys.executable, str(ROOT / "src" / "run_tests.py")]
    tag = f"{part_dir.parent.name}/{part_dir.name}"
    print(f"[{tag}] TEST phase launching (workers={max_workers})", flush=True)
    p = subprocess.run(
        cmd, env=env, cwd=str(part_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    # Last 20 lines for visibility
    tail = "\n".join(p.stdout.splitlines()[-20:])
    print(f"[{tag}] TEST exit={p.returncode}\n{tail}", flush=True)
    return p.returncode


def collect_partitions(task: str) -> list[Path]:
    base = ROOT / "agent-runs" / task
    parts = []
    for run in sorted(base.iterdir()):
        if not run.is_dir():
            continue
        for p in sorted(run.iterdir()):
            if p.is_dir() and p.name.startswith("partition-"):
                parts.append(p)
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", help="Task name, e.g. stanford_hint")
    ap.add_argument("--max-parallel", type=int, default=3,
                    help="Max partitions processed concurrently (default 3).")
    ap.add_argument("--test-workers", type=int, default=5,
                    help="Per-partition AGENT_TEST_MAX_WORKERS (default 5).")
    ap.add_argument("--skip-stubs", action="store_true",
                    help="Do NOT re-launch strategy agents for stub partitions.")
    args = ap.parse_args()

    parts = collect_partitions(args.task)
    todo = [p for p in parts if not has_results(p)]
    print(f"Total partitions: {len(parts)}; incomplete: {len(todo)}")

    stubs = [p for p in todo if is_stub_strategy(p)]
    ready = [p for p in todo if not is_stub_strategy(p)]

    print(f"  STUB strategies (need re-launch): {len(stubs)}")
    for p in stubs:
        print(f"    {p.parent.name}/{p.name}")
    print(f"  Ready to test (just need test phase): {len(ready)}")
    for p in ready:
        print(f"    {p.parent.name}/{p.name}")

    # Phase 1: re-launch strategy agents for STUB partitions.
    # Each strategy agent will itself call `test` and spawn test agents internally.
    if stubs and not args.skip_stubs:
        print(f"\n=== Phase 1: re-launching {len(stubs)} strategy agents ===")
        with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
            futs = {ex.submit(run_strategy_agent, p): p for p in stubs}
            for f in as_completed(futs):
                p = futs[f]
                try:
                    f.result()
                except Exception as e:
                    print(f"[{p.parent.name}/{p.name}] STRATEGY ERROR: {e}", flush=True)

    # Phase 2: run test phase for any partition still without results.csv.
    # Re-audit because phase 1 may have produced results.csv via the agent's `test` call.
    leftover = [p for p in parts if not has_results(p) and not is_stub_strategy(p)]
    if leftover:
        print(f"\n=== Phase 2: running test phase for {len(leftover)} partition(s) ===")
        with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
            futs = {ex.submit(run_test_phase, p, args.test_workers): p for p in leftover}
            for f in as_completed(futs):
                p = futs[f]
                try:
                    f.result()
                except Exception as e:
                    print(f"[{p.parent.name}/{p.name}] TEST ERROR: {e}", flush=True)
    else:
        print("\nNo partitions need test phase.")

    # Final audit
    print("\n=== Final state ===")
    for p in parts:
        stub = is_stub_strategy(p)
        res = has_results(p)
        flag = "OK" if res else ("STUB" if stub else "INCOMPLETE")
        print(f"  {p.parent.name}/{p.name}: {flag}")


if __name__ == "__main__":
    main()
