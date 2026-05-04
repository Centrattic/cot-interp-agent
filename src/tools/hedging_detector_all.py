"""`hedging-detector-all` tool — run hedging-detector on every few-shot example.

Usage:
    hedging-detector-all

Strategy-agent only. Iterates over all few-shot examples, dispatches hedging
analyses in parallel (up to 8 concurrent Codex helper calls), and writes:
  - per-example: `hedging_<example_id>.json` (same format as `hedging-detector`;
    cache hits honored).
  - summary: `hedging_all_summary.json` — per-example overall scores grouped
    by label, plus mean/stdev by group.

Rate limit
- Once per strategy run. A lock file at `.subagent_locks/hedging_detector_all.lock`
  is created on success and blocks re-invocation.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from _common import fail, get_env, list_few_shot_ids, load_example
from _subagent import (
    DEFAULT_PARALLELISM,
    assert_lock_free,
    assert_strategy_only,
    run_parallel,
    write_lock,
)
from hedging_detector import analyse_example


LOCK_NAME = "hedging_detector_all"


def main(argv: list[str]) -> int:
    if argv:
        fail("usage: hedging-detector-all  (takes no arguments)")

    env = get_env()
    assert_strategy_only(env, "hedging-detector-all")
    assert_lock_free(LOCK_NAME, "hedging-detector-all")

    example_ids = list_few_shot_ids(env)
    if not example_ids:
        fail("no few-shot examples found.")

    cwd = Path.cwd()

    def job(example_id: str) -> dict:
        example = load_example(env, example_id)
        return analyse_example(
            example_id,
            example,
            cache_path=cwd / f"hedging_{example_id}.json",
        )

    print(
        f"running hedging-detector on {len(example_ids)} few-shot examples "
        f"(max_parallel={DEFAULT_PARALLELISM})..."
    )
    outcomes = run_parallel(job, example_ids, max_workers=DEFAULT_PARALLELISM)

    per_example: list[dict] = []
    errors: list[dict] = []
    for ex_id, result, err in outcomes:
        if err is not None:
            errors.append({"example_id": ex_id, "error": err})
            continue
        label = load_example(env, ex_id).get("label")
        per_example.append({
            "example_id": ex_id,
            "label": label,
            "overall": result["overall"],
            "peak_idx": result["peak_idx"],
            "n_sentences": result["n_sentences"],
            "cached": result["cached"],
        })

    by_label: dict[str, list[float]] = {}
    for row in per_example:
        by_label.setdefault(str(row["label"]), []).append(row["overall"])

    label_stats = {}
    for lbl, vals in by_label.items():
        label_stats[lbl] = {
            "n": len(vals),
            "mean": statistics.fmean(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }

    summary = {
        "tool": "hedging-detector-all",
        "n_examples": len(example_ids),
        "n_ok": len(per_example),
        "n_errors": len(errors),
        "per_example": sorted(per_example, key=lambda r: r["example_id"]),
        "label_stats": label_stats,
        "errors": errors,
    }
    summary_path = cwd / "hedging_all_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    if errors:
        print(f"status: partial  ({len(errors)} errors, see {summary_path.name})")
    else:
        write_lock(LOCK_NAME)
        print(f"status: success  (lock written)")

    # Surface the label-stratified signal *prominently*: the gap between mean
    # overall hedging by label is the actionable signal callers care about.
    # Earlier observation: agents glossed over the label_stats below and
    # walked away without using the signal even when the gap was large.
    if len(label_stats) == 2:
        labels = sorted(label_stats.keys())
        m0 = label_stats[labels[0]]["mean"]
        m1 = label_stats[labels[1]]["mean"]
        gap = m1 - m0  # signed: positive means label[1] hedges more
        direction = "MORE" if gap > 0 else "LESS"
        which_more = labels[1] if gap > 0 else labels[0]
        which_less = labels[0] if gap > 0 else labels[1]
        print(
            f"signal: label={which_more} hedges {direction} than label={which_less}; "
            f"|gap| = {abs(gap):.3f} in mean overall hedging "
            f"(label={labels[0]}: {m0:.3f}, label={labels[1]}: {m1:.3f})"
        )

    for lbl, s in sorted(label_stats.items()):
        print(
            f"label={lbl}: n={s['n']}  mean={s['mean']:.3f}  "
            f"stdev={s['stdev']:.3f}  min={s['min']:.3f}  max={s['max']:.3f}"
        )
    print(f"summary: {summary_path.name}")
    return 0 if not errors else 8


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
