"""Score a multi-partition agent run.

Walks `run-<ts>/partition-NNN/test-NNN/answer.txt`, recovers the global
test-example identity via the round-robin partitioning used at launch
(global_idx = local_idx * n_partitions + partition_idx), compares against
ground-truth labels in `data/<task>/test/*.json`, and emits:

  - `run-<ts>/results.csv`   — per-example: partition, local_idx, global_idx,
                               example_id, label, answer, pred, correct
  - `run-<ts>/summary.txt`   — per-partition confusion + metrics, plus
                               aggregate (all partitions) and
                               mean ± stdev of g-mean² across partitions.

Intended to be called by scaffold.py after all partition strategy agents
finish, but can also be run standalone:

    python score_run.py <run_dir>
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path


SCAFFOLD_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = SCAFFOLD_ROOT / "data"


def _metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    n = tp + tn + fp + fn
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    acc = (tp + tn) / n if n else 0.0
    return {
        "n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "acc": acc, "tpr": tpr, "tnr": tnr, "gmean2": tpr * tnr,
    }


def _load_answer(p: Path) -> int | None:
    """Return 0/1 if answer.txt is a clean yes/no, else None."""
    if not p.exists():
        return None
    a = p.read_text(encoding="utf-8").strip().lower()
    if a == "yes": return 1
    if a == "no":  return 0
    return None


def score_partitioned_run(run_dir: Path, task_meta: dict | None = None) -> dict:
    """Compute per-partition and aggregate metrics for a multi-partition run."""
    run_meta = json.load(open(run_dir / "run.json", encoding="utf-8"))
    task_name = run_meta["task"]
    n_partitions = int(run_meta.get("n_strategies", 1))
    test_dir = DATA_DIR / task_name / "test"
    test_files = sorted(test_dir.glob("*.json"))
    total_tests = len(test_files)

    # Build global ground-truth list
    ground_truth = []
    for f in test_files:
        d = json.load(open(f, encoding="utf-8"))
        ground_truth.append((f.stem, int(d["label"])))

    rows: list[dict] = []
    per_partition_counts: dict[int, dict] = {}

    for part_dir in sorted(run_dir.glob("partition-*")):
        if not part_dir.is_dir():
            continue
        part_idx = int(part_dir.name.split("-")[1])
        tp = tn = fp = fn = miss = 0
        for tf in sorted(part_dir.glob("test-*")):
            if not tf.is_dir():
                continue
            local_idx = int(tf.name.split("-")[1])
            global_idx = local_idx * n_partitions + part_idx
            if global_idx >= total_tests:
                continue
            example_id, gt = ground_truth[global_idx]
            pred = _load_answer(tf / "answer.txt")
            correct = (pred is not None) and (pred == gt)
            if pred is None:
                miss += 1
            elif gt == 1 and pred == 1: tp += 1
            elif gt == 0 and pred == 0: tn += 1
            elif gt == 0 and pred == 1: fp += 1
            elif gt == 1 and pred == 0: fn += 1
            rows.append({
                "partition": part_idx,
                "local_idx": local_idx,
                "global_idx": global_idx,
                "example_id": example_id,
                "label": gt,
                "answer": {0: "no", 1: "yes", None: ""}[pred],
                "pred": "" if pred is None else pred,
                "correct": "" if pred is None else int(correct),
            })
        per_partition_counts[part_idx] = {
            "tp": tp, "tn": tn, "fp": fp, "fn": fn, "miss": miss
        }

    # Write per-example results
    results_path = run_dir / "results.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["partition", "local_idx", "global_idx",
                           "example_id", "label", "answer", "pred", "correct"]
        )
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda r: r["global_idx"]))

    # Aggregate across all partitions
    agg_tp = sum(c["tp"] for c in per_partition_counts.values())
    agg_tn = sum(c["tn"] for c in per_partition_counts.values())
    agg_fp = sum(c["fp"] for c in per_partition_counts.values())
    agg_fn = sum(c["fn"] for c in per_partition_counts.values())
    agg_miss = sum(c["miss"] for c in per_partition_counts.values())
    agg = _metrics(agg_tp, agg_tn, agg_fp, agg_fn)

    # Per-partition metrics list (for mean/stdev)
    per_part_metrics: list[dict] = []
    for k in sorted(per_partition_counts):
        c = per_partition_counts[k]
        m = _metrics(c["tp"], c["tn"], c["fp"], c["fn"])
        m["partition"] = k
        m["miss"] = c["miss"]
        per_part_metrics.append(m)

    gm2s = [m["gmean2"] for m in per_part_metrics if m["n"] > 0]

    lines = []
    lines.append(f"# Run summary — {task_name}  (run {run_meta['run_id']})")
    lines.append(f"n_partitions = {n_partitions}  total_test_examples = {total_tests}")
    lines.append(f"tools = {run_meta.get('tools', [])}")
    lines.append("")
    lines.append("## Per-partition")
    lines.append(f"{'part':>4}  {'n':>3}  {'miss':>4}  {'acc':>6}  {'TPR':>5}  {'TNR':>5}  {'gmean²':>7}  TP/TN/FP/FN")
    lines.append("-" * 72)
    for m in per_part_metrics:
        lines.append(
            f"{m['partition']:>4}  {m['n']:>3}  {m['miss']:>4}  {m['acc']*100:>5.1f}%  "
            f"{m['tpr']:>5.2f}  {m['tnr']:>5.2f}  {m['gmean2']:>7.3f}  "
            f"{m['tp']}/{m['tn']}/{m['fp']}/{m['fn']}"
        )
    lines.append("")
    lines.append("## Aggregate (all partitions combined)")
    lines.append(
        f"n={agg['n']}  miss={agg_miss}  acc={agg['acc']*100:.1f}%  "
        f"TPR={agg['tpr']:.2f}  TNR={agg['tnr']:.2f}  gmean²={agg['gmean2']:.3f}  "
        f"TP/TN/FP/FN={agg['tp']}/{agg['tn']}/{agg['fp']}/{agg['fn']}"
    )
    lines.append("")
    if len(gm2s) >= 2:
        lines.append("## Variance across partitions (g-mean²)")
        lines.append(
            f"mean={statistics.mean(gm2s):.3f}  stdev={statistics.stdev(gm2s):.3f}  "
            f"min={min(gm2s):.3f}  max={max(gm2s):.3f}  n_partitions_scored={len(gm2s)}"
        )
    lines.append("")

    summary_text = "\n".join(lines)
    (run_dir / "summary.txt").write_text(summary_text, encoding="utf-8")

    print(summary_text)
    print(f"\nWrote {results_path}")
    print(f"Wrote {run_dir / 'summary.txt'}")

    return {
        "aggregate": agg,
        "per_partition": per_part_metrics,
        "agg_miss": agg_miss,
    }


def main():
    if len(sys.argv) != 2:
        print("usage: python score_run.py <run_dir>")
        sys.exit(2)
    run_dir = Path(sys.argv[1]).resolve()
    score_partitioned_run(run_dir)


if __name__ == "__main__":
    main()
