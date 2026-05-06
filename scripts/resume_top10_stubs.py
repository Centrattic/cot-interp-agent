#!/usr/bin/env python3
"""Re-launch the strategy agent for stub partitions of CLAIMED top10-only runs.

A "stub" partition has no STRATEGY.md (or a tiny placeholder one). The strategy
agent itself calls `run-tests` when it finishes, so this single launch produces
both STRATEGY.md and per-partition results.csv. Already-answered tests inside
the partition (if any) are skipped via run_tests.py resume support.

We only operate on a hand-picked CLAIM list — runs we want to bring to
all-10-partitions-done — to avoid wasting work on stale partials we don't need.
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

# Claim list: partial top10 runs that are >50% complete — cheap to finish locally.
# (Sub-50% partials are restarted as fresh runs on the remote box instead.)
CLAIMED_RUNS = [
    "agent-runs/reasoning_termination/run-20260505-142824-296882",  # 6 done / 4 stubs
]


def is_stub(part_dir: Path) -> bool:
    f = part_dir / "strategy" / "STRATEGY.md"
    return (not f.exists()) or f.stat().st_size <= 200


def has_results_with_data(part_dir: Path) -> bool:
    rcsv = part_dir / "results.csv"
    if not rcsv.exists():
        return False
    import csv as _csv
    try:
        rows = list(_csv.DictReader(open(rcsv)))
    except Exception:
        return False
    return any(r.get("answer", "") not in ("", None) for r in rows)


def run_strategy_agent(part_dir: Path) -> tuple[str, int, str]:
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
    env["AGENT_TYPE"] = "strategy"
    # Keep test concurrency modest so we don't overwhelm the box.
    env.setdefault("AGENT_TEST_MAX_WORKERS", "2")
    # Pin model so resumes don't drift if config.toml changes.
    env["CODEX_MODEL"] = "gpt-5.4"
    env["CODEX_REASONING_EFFORT"] = "medium"

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
        if backend == "codex"
        else None
    )
    launch = build_agent_launch_spec(
        backend=backend,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        add_dirs=add_dirs,
    )
    strategy_dir = part_dir / "strategy"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{part_dir.parent.parent.name}/{part_dir.parent.name}/{part_dir.name}"
    print(f"[{tag}] STRATEGY launching (backend={backend})", flush=True)
    p = subprocess.run(
        launch.cmd, env=env, cwd=str(strategy_dir), input=launch.stdin_text,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    tail = "\n".join(p.stdout.splitlines()[-5:])
    return tag, p.returncode, tail


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-parallel", type=int, default=3,
                    help="Max stub partitions in flight (default 3 — be gentle on local box)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    todo: list[Path] = []
    for rel in CLAIMED_RUNS:
        run_dir = ROOT / rel
        if not run_dir.exists():
            print(f"  MISSING: {rel}")
            continue
        for p in sorted(run_dir.iterdir()):
            if not (p.is_dir() and p.name.startswith("partition-")):
                continue
            if has_results_with_data(p):
                continue
            if not is_stub(p):
                # Might be ready (rare after pass1+2) — skip; the ready driver handles those.
                continue
            todo.append(p)

    print(f"\n=== {len(todo)} stub partitions queued (max-parallel={args.max_parallel}) ===\n")
    if args.dry_run:
        for p in todo:
            print(f"  {p}")
        return 0

    fails = []
    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futs = {ex.submit(run_strategy_agent, p): p for p in todo}
        for f in as_completed(futs):
            tag, code, tail = f.result()
            ok_marker = "OK " if code == 0 else "XX "
            print(f"[{tag}] {ok_marker}exit={code}\n{tail}\n", flush=True)
            if code != 0:
                fails.append(tag)

    print("\n=== Final per-claim state ===")
    for rel in CLAIMED_RUNS:
        run_dir = ROOT / rel
        if not run_dir.exists():
            continue
        done = stub = ready = 0
        for p in sorted(run_dir.iterdir()):
            if not (p.is_dir() and p.name.startswith("partition-")):
                continue
            if has_results_with_data(p):
                done += 1
            elif is_stub(p):
                stub += 1
            else:
                ready += 1
        marker = "USABLE" if done == 10 else f"need {stub} more stubs / {ready} ready"
        print(f"  {rel}: done={done} ready={ready} stub={stub}  {marker}")

    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
