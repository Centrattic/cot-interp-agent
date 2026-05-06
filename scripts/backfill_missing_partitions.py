#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import scaffold  # type: ignore


def iter_missing_partitions(run_dir: Path):
    for part_dir in sorted(run_dir.glob("partition-*")):
        if (part_dir / "results.csv").exists():
            continue
        yield part_dir


def relaunch_partition(part_dir: Path, tools: list[str]) -> tuple[str, int]:
    trace_root = (
        ROOT
        / "agent-traces"
        / part_dir.parent.parent.name
        / part_dir.parent.name
    )
    trace_root.mkdir(parents=True, exist_ok=True)
    trace_base = trace_root / f"{part_dir.name}-strategy-retry-trace"
    code = scaffold._launch_strategy_agent(  # noqa: SLF001
        strategy_dir=part_dir / "strategy",
        trace_base=trace_base,
        bashrc_path=part_dir / "agent.bashrc",
        tools=tools,
        label=f"{part_dir.parent.parent.name}/{part_dir.parent.name}/{part_dir.name}",
    )
    return part_dir.as_posix(), code


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dirs", nargs="+", help="Run directories to backfill")
    ap.add_argument("--max-parallel", type=int, default=2)
    args = ap.parse_args()

    jobs: list[tuple[Path, list[str]]] = []
    for run_dir_str in args.run_dirs:
        run_dir = Path(run_dir_str).resolve()
        run_meta = scaffold.json.loads((run_dir / "run.json").read_text())
        tools = run_meta.get("tools", ["sample"])
        for part_dir in iter_missing_partitions(run_dir):
            jobs.append((part_dir, tools))

    print(f"Backfilling {len(jobs)} partition(s)")
    with ThreadPoolExecutor(max_workers=args.max_parallel) as ex:
        futs = {
            ex.submit(relaunch_partition, part_dir, tools): part_dir
            for part_dir, tools in jobs
        }
        for fut in as_completed(futs):
            part_dir = futs[fut]
            try:
                path, code = fut.result()
                print(f"{path}: exit_code={code}")
            except Exception as e:
                print(f"{part_dir}: FAILED {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
