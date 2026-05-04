"""`repetition-mapper-all` tool — run repetition-mapper on every few-shot example.

Usage:
    repetition-mapper-all

Strategy-agent only. Iterates over all few-shot examples, dispatches
repetition analyses in parallel (up to 8 concurrent Codex helper calls), and writes:
  - per-example: `repetition_<example_id>.json` (same format as
    `repetition-mapper`; cache hits honored).
  - summary: `repetition_all_summary.json` — per-example cluster counts and
    longest-chain spans grouped by label, plus mean/stdev by group.

Rate limit
- Once per strategy run. A lock file at
  `.subagent_locks/repetition_mapper_all.lock` is created on success and
  blocks re-invocation.
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
from repetition_mapper import analyse_example


LOCK_NAME = "repetition_mapper_all"


def _stats(vals: list[float]) -> dict:
    return {
        "n": len(vals),
        "mean": statistics.fmean(vals),
        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def main(argv: list[str]) -> int:
    if argv:
        fail("usage: repetition-mapper-all  (takes no arguments)")

    env = get_env()
    assert_strategy_only(env, "repetition-mapper-all")
    assert_lock_free(LOCK_NAME, "repetition-mapper-all")

    example_ids = list_few_shot_ids(env)
    if not example_ids:
        fail("no few-shot examples found.")

    cwd = Path.cwd()

    def job(example_id: str) -> dict:
        example = load_example(env, example_id)
        return analyse_example(
            example_id,
            example,
            cache_path=cwd / f"repetition_{example_id}.json",
        )

    print(
        f"running repetition-mapper on {len(example_ids)} few-shot examples "
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
            "cluster_count": result["cluster_count"],
            "longest_chain_span": result["longest_chain"].get("span", 0),
            "n_sentences": result["n_sentences"],
            "cached": result["cached"],
        })

    clusters_by_label: dict[str, list[float]] = {}
    span_by_label: dict[str, list[float]] = {}
    for row in per_example:
        lbl = str(row["label"])
        clusters_by_label.setdefault(lbl, []).append(float(row["cluster_count"]))
        span_by_label.setdefault(lbl, []).append(float(row["longest_chain_span"]))

    label_stats = {
        lbl: {
            "cluster_count": _stats(clusters_by_label.get(lbl, [])),
            "longest_chain_span": _stats(span_by_label.get(lbl, [])),
        }
        for lbl in clusters_by_label
    }

    summary = {
        "tool": "repetition-mapper-all",
        "n_examples": len(example_ids),
        "n_ok": len(per_example),
        "n_errors": len(errors),
        "per_example": sorted(per_example, key=lambda r: r["example_id"]),
        "label_stats": label_stats,
        "errors": errors,
    }
    summary_path = cwd / "repetition_all_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    if errors:
        print(f"status: partial  ({len(errors)} errors, see {summary_path.name})")
    else:
        write_lock(LOCK_NAME)
        print(f"status: success  (lock written)")

    # Surface the label-stratified signal prominently — agents tend to
    # gloss over per-label numeric stats and walk away without using them.
    if len(label_stats) == 2:
        labels = sorted(label_stats.keys())
        cc0 = label_stats[labels[0]]["cluster_count"]["mean"]
        cc1 = label_stats[labels[1]]["cluster_count"]["mean"]
        sp0 = label_stats[labels[0]]["longest_chain_span"]["mean"]
        sp1 = label_stats[labels[1]]["longest_chain_span"]["mean"]
        cc_gap = cc1 - cc0
        sp_gap = sp1 - sp0
        cc_dir = "MORE" if cc_gap > 0 else "FEWER"
        cc_more = labels[1] if cc_gap > 0 else labels[0]
        cc_less = labels[0] if cc_gap > 0 else labels[1]
        sp_dir = "WIDER" if sp_gap > 0 else "TIGHTER"
        sp_more = labels[1] if sp_gap > 0 else labels[0]
        sp_less = labels[0] if sp_gap > 0 else labels[1]
        print(
            f"signal: label={cc_more} has {cc_dir} clusters than label={cc_less}; "
            f"|gap| = {abs(cc_gap):.2f} clusters "
            f"(label={labels[0]}: {cc0:.2f}, label={labels[1]}: {cc1:.2f})"
        )
        print(
            f"signal: label={sp_more} has {sp_dir} longest-chain spans than label={sp_less}; "
            f"|gap| = {abs(sp_gap):.2f} sentences "
            f"(label={labels[0]}: {sp0:.2f}, label={labels[1]}: {sp1:.2f})"
        )

    for lbl, s in sorted(label_stats.items()):
        cc = s["cluster_count"]
        sp = s["longest_chain_span"]
        print(
            f"label={lbl}: n={cc['n']}  "
            f"clusters mean={cc['mean']:.2f} stdev={cc['stdev']:.2f}  "
            f"longest_span mean={sp['mean']:.2f} stdev={sp['stdev']:.2f}"
        )
    print(f"summary: {summary_path.name}")
    return 0 if not errors else 8


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
