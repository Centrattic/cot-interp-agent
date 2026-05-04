#!/usr/bin/env python3
"""Salvage runs whose strategy phase completed but test phase walled.

Walks each task's combo runs (only those status=partial), and for any run that
has STRATEGY.md but few/no answer.txt files, re-invokes `bin/run-tests` per
partition (which honors the resume-by-existing-answer.txt path). After each
salvaged run, calls `score_run.py` to rebuild summary.txt.

The aggregator filter (miss < 50%) decides whether the salvaged run counts
toward the cell's TARGET.

Limits per call (controls burn):
  --max-runs-per-task=1    (default)
  --tasks <comma-list>     (default: all 7)
  --skip-combo=0|1         (default: 0 — salvage combo only)
  --parallel-partitions=3  (partitions retested concurrently)
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASKS = [
    'reasoning_termination','gemma_self_deletion','followup_confidence',
    'user_preference_sycophancy','stanford_hint','atypical_answer','atypical_cot_length',
]
NUDGE = {'20260502-120541-301368','20260502-125815-877735','20260502-130449-039488'}
COMBO_TOOLS = {'few-shot-diff','hedging-detector','hedging-detector-all',
               'repetition-mapper','repetition-mapper-all'}
TARGET = 3


def is_combo(tools: list[str]) -> bool:
    if not tools: return False
    return any(t in COMBO_TOOLS for t in tools)


def count_answers(part_dir: Path) -> tuple[int, int]:
    """Return (n_answered, n_test_dirs)."""
    n_ans = n_dirs = 0
    for d in part_dir.glob('test-*'):
        if not d.is_dir(): continue
        n_dirs += 1
        ans = d / 'answer.txt'
        if ans.exists() and ans.read_text().strip().lower() in ('yes','no'):
            n_ans += 1
    return n_ans, n_dirs


def find_salvage_targets(tasks: list[str], max_runs_per_task: int) -> list[tuple[str, Path, list[Path]]]:
    """Return [(task, run_dir, partitions_with_unfinished_tests), ...]."""
    # Read current grid state to skip already-filled cells.
    comp_path = ROOT / 'results' / 'comparison.json'
    if comp_path.exists():
        comp = json.loads(comp_path.read_text())
    else:
        comp = {}

    targets = []
    for task in tasks:
        # Skip if combo cell already at TARGET.
        runs_in_cell = comp.get(task,{}).get('combo',{}).get('runs',[])
        runs_in_cell = [r for r in runs_in_cell if r['run_id'] not in NUDGE]
        if len(runs_in_cell) >= TARGET:
            continue

        run_root = ROOT / 'agent-runs' / task
        if not run_root.exists(): continue

        picked = 0
        # Most recent first.
        for d in sorted(run_root.glob('run-*'), reverse=True):
            if picked >= max_runs_per_task: break
            rj = d / 'run.json'
            if not rj.exists(): continue
            try:
                meta = json.loads(rj.read_text())
            except Exception:
                continue
            if not is_combo(meta.get('tools', [])): continue
            if meta.get('status') != 'partial': continue
            # Skip if already counted in cell.
            if any(r['run_id'] == meta['run_id'] for r in runs_in_cell):
                continue

            parts_to_retest = []
            for p in sorted(d.glob('partition-*')):
                if not (p / 'strategy' / 'STRATEGY.md').exists(): continue
                n_ans, n_dirs = count_answers(p)
                # Salvage if at least one strategy exists and <50% answers.
                if n_dirs > 0 and n_ans < n_dirs / 2:
                    parts_to_retest.append(p)
                elif n_dirs == 0 and (p/'strategy').exists():
                    # Strategy completed but tests never even started — also salvage.
                    parts_to_retest.append(p)
            if parts_to_retest:
                targets.append((task, d, parts_to_retest))
                picked += 1
    return targets


def prime_codex_auth() -> None:
    """Issue a tiny codex call so the access token is fresh before parallel
    test workers spawn. Avoids OAuth refresh-token races."""
    subprocess.run(
        ['timeout', '20', 'codex', 'exec', '--skip-git-repo-check', 'reply ok'],
        capture_output=True, timeout=30,
    )


def refresh_partition_auth(partition_dir: Path) -> bool:
    """Copy the freshly-refreshed global ~/.codex/auth.json into the
    partition's isolated .codex-home so test workers spawn with valid tokens.

    Each partition has its own CODEX_HOME (set at original run time), and that
    auth.json may contain a used/burnt refresh token from a prior walled run.
    Returns True on success."""
    import shutil
    src = Path.home() / '.codex' / 'auth.json'
    dst = partition_dir / '.codex-home' / 'auth.json'
    if not src.exists():
        return False
    if not dst.parent.exists():
        return False
    try:
        shutil.copy2(src, dst)
        return True
    except Exception as e:
        print(f"  [auth-copy-fail] {e}")
        return False


def run_partition_tests(partition_dir: Path) -> int:
    """Prime auth, refresh partition-local auth, call run-tests. Returns exit code."""
    bashrc = partition_dir / 'agent.bashrc'
    if not bashrc.exists():
        print(f"  [skip] no agent.bashrc in {partition_dir}")
        return 1
    print(f"  [retest] {partition_dir.relative_to(ROOT)}")
    prime_codex_auth()
    if not refresh_partition_auth(partition_dir):
        print(f"  [auth-refresh-fail] skipping {partition_dir.name}")
        return 1
    cmd = (
        f'cd {partition_dir.as_posix()} && '
        f'set -a && source {bashrc.as_posix()} && set +a && '
        f'run-tests'
    )
    proc = subprocess.run(['bash', '-c', cmd], cwd=str(ROOT))
    return proc.returncode


def rescore_run(run_dir: Path) -> None:
    score_py = ROOT / 'src' / 'score_run.py'
    proc = subprocess.run(
        [sys.executable, str(score_py), str(run_dir)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(f"  [score-fail] {run_dir.name}: {proc.stderr[:200]}")
    else:
        print(f"  [scored]     {run_dir.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-runs-per-task', type=int, default=1)
    ap.add_argument('--tasks', default=','.join(TASKS))
    ap.add_argument('--max-partitions-per-call', type=int, default=10,
                    help='Cap partitions retested in one invocation to control burn.')
    ap.add_argument('--parallel-partitions', type=int, default=3,
                    help='How many partitions to retest concurrently.')
    args = ap.parse_args()
    tasks = [t.strip() for t in args.tasks.split(',') if t.strip()]

    # Refresh aggregate so we know cell counts.
    subprocess.run([sys.executable, str(ROOT/'src'/'aggregate_results.py')],
                   capture_output=True)

    targets = find_salvage_targets(tasks, args.max_runs_per_task)
    total_parts = sum(len(p) for _, _, p in targets)
    print(f"salvage: {len(targets)} runs, {total_parts} partitions to retest "
          f"(cap={args.max_partitions_per_call})")
    if not targets:
        return 0

    # Prime once at start; each partition's run_partition_tests will refresh
    # its local CODEX_HOME from the global ~/.codex/auth.json.
    prime_codex_auth()

    parts_done = 0
    for task, run_dir, parts in targets:
        # Cap partitions for this run within remaining budget.
        remaining = args.max_partitions_per_call - parts_done
        if remaining <= 0:
            print(f"  [cap reached] stopping at {parts_done} partitions")
            break
        parts_for_this_run = parts[:remaining]
        print(f"\n[{task}] {run_dir.name}: retesting {len(parts_for_this_run)} of {len(parts)} partitions "
              f"(parallel={args.parallel_partitions})")
        with ThreadPoolExecutor(max_workers=args.parallel_partitions) as ex:
            futures = {ex.submit(run_partition_tests, p): p for p in parts_for_this_run}
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    rc = fut.result()
                    print(f"  [done rc={rc}] {p.name}")
                except Exception as e:
                    print(f"  [error] {p.name}: {e}")
        parts_done += len(parts_for_this_run)
        rescore_run(run_dir)

    # Refresh aggregate so the wrapper sees salvaged cells.
    subprocess.run([sys.executable, str(ROOT/'src'/'aggregate_results.py')],
                   capture_output=True)
    print(f"\nsalvage done: {parts_done} partitions retested")
    return 0


if __name__ == '__main__':
    sys.exit(main())
