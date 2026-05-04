"""Walk agent-runs and produce a per-(task, config) gmean² comparison.

Filters
-------
- agent_backend == "codex"
- not validate / validate_mini
- run_id >= 20260430-123000 (when gpt-5.5 launches started)
- tools list matches one of CELL_TOOLS

Re-runs ``score_run`` on each matching run so the on-disk results reflect the
actual answer.txt files (the original aggregator can run early and miss
late-flushed answers).

Outputs
-------
- results/comparison.json   per-cell run lists + means
- results/comparison.png    grouped bar chart
- results/comparison.txt    plain-text table
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from score_run import score_run

TASKS = [
    ("reasoning_termination", "task 1"),
    ("gemma_self_deletion", "task 2"),
    ("followup_confidence", "task 3"),
    ("user_preference_sycophancy", "task 4"),
    ("stanford_hint", "task 5"),
    ("atypical_answer", "task 6"),
    ("atypical_cot_length", "task 7"),
]

CELL_TOOLS = {
    "bare": [],
    "word-stats": ["word-stats"],
    "sae": ["sae"],
    "combo": [
        "few-shot-diff",
        "hedging-detector",
        "hedging-detector-all",
        "repetition-mapper",
        "repetition-mapper-all",
    ],
}
CELLS = list(CELL_TOOLS.keys())

# Per-cell minimum run-id (= timestamp). A run only counts toward a cell if it
# was launched after the relevant code/prompt fix that defines the cell.
#   bare       : 20:12 today — default-to-no clause added to data/<task1,task2>/metadata.json
#   word-stats : 20:12 today — get_cot_prefix learned to handle messages-format data
#                              (also picks up the metadata clause)
#   sae        : 21:42 today — v2 subcommands (discriminate, validate, top-features stats)
#   combo      : same as latest of the above (no additional code changes for combo yet)
CELL_CUTOFFS = {
    "bare": "20260430-201200",
    "word-stats": "20260430-201200",
    "sae": "20260430-214200",
    "combo": "20260430-201200",
}

# Older runs lack the codex_runtime metadata field; the only confirmed
# non-gpt-5.5 runs from today (manually launched on spark) are these:
SPARK_RUN_IDS = {
    "20260430-153924-287795",
    "20260430-161314-826184",
}

# Runs from a transient config that was reverted (sae v2 with the
# feature-list-size nudge in TOOL_DESCRIPTIONS, which is no longer live):
EXCLUDE_RUN_IDS = {
    "20260430-215202-531460",  # sae v2 + nudge (nudge has since been removed)
}

OUT_DIR = ROOT / "results"


def classify(tools: list[str]) -> str | None:
    s = sorted(tools or [])
    for cell, cell_tools in CELL_TOOLS.items():
        if s == sorted(cell_tools):
            return cell
    return None


def collect() -> dict:
    """Return {task: {cell: [{run_id, gmean2, n, miss}, ...]}}."""
    out: dict = {}
    for task, _ in TASKS:
        task_dir = ROOT / "agent-runs" / task
        if not task_dir.is_dir():
            continue
        for run_dir in sorted(task_dir.glob("run-*")):
            meta_path = run_dir / "run.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if meta.get("validate") or meta.get("validate_mini"):
                continue
            if meta.get("agent_backend") != "codex":
                continue
            rid = meta.get("run_id", "")
            if rid in SPARK_RUN_IDS or rid in EXCLUDE_RUN_IDS:
                continue
            # Honor codex_runtime when present (newer runs)
            cr = meta.get("codex_runtime") or {}
            cm = (cr.get("codex_model") or "").strip()
            if cm and "gpt-5.5" not in cm.lower():
                continue
            # Canonical config is 20 few-shot per class (40 total). Drop
            # one-off experiments at other sizes from the comparison table.
            if int(meta.get("few_shot_per_class", 20)) != 20:
                continue
            # Don't trust run.json status — the original aggregator can run
            # before all answer.txt files have flushed. The re-aggregated
            # agg_miss below is the source of truth.
            cell = classify(meta.get("tools"))
            if cell is None:
                continue
            # Per-cell cutoff: only runs launched after the relevant fix
            if rid < CELL_CUTOFFS.get(cell, "20260430-000000"):
                continue
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    info = score_run(run_dir)
                agg = info["aggregate"]
                gm2 = float(agg["gmean2"])
                tpr = float(agg["tpr"])
                tnr = float(agg["tnr"])
                acc = float(agg["acc"])
                miss = int(info["agg_miss"])
                n = int(agg["n"])
                total_tests = int(info["total_tests"])
            except Exception:
                continue
            # Drop runs where most of the test set is missing — these are
            # auth/quota wall failures, not representative of the cell.
            if total_tests > 0 and miss > total_tests * 0.5:
                continue
            out.setdefault(task, {}).setdefault(cell, []).append({
                "run_id": rid,
                "gmean2": gm2,
                "tpr": tpr,
                "tnr": tnr,
                "acc": acc,
                "n": n,
                "miss": miss,
            })
    return out


def summarize(results: dict) -> dict:
    """Return {task: {cell: {n_runs, mean_gmean2, runs}}} for serialization."""
    summary = {}
    for task, _ in TASKS:
        per_task = {}
        for cell in CELLS:
            runs = results.get(task, {}).get(cell, [])
            if not runs:
                per_task[cell] = {"n_runs": 0, "mean_gmean2": None}
                continue
            gms = [r["gmean2"] for r in runs]
            tprs = [r["tpr"] for r in runs]
            tnrs = [r["tnr"] for r in runs]
            per_task[cell] = {
                "n_runs": len(runs),
                "mean_gmean2": sum(gms) / len(gms),
                "mean_tpr": sum(tprs) / len(tprs),
                "mean_tnr": sum(tnrs) / len(tnrs),
                "runs": runs,
            }
        summary[task] = per_task
    return summary


def make_text_table(summary: dict) -> str:
    lines = []
    header = f"{'task':30s}  " + "  ".join(f"{c:>14s}" for c in CELLS)
    lines.append(header)
    lines.append("-" * len(header))
    for task, label in TASKS:
        row = f"{label} {task:25s}"
        for cell in CELLS:
            data = summary[task].get(cell, {})
            n = data.get("n_runs", 0)
            m = data.get("mean_gmean2")
            if cell == "sae" and task == "gemma_self_deletion":
                cell_str = "  N/A (no SAE)"
            elif n == 0:
                cell_str = "       —"
            else:
                cell_str = f"  {m:.3f} (n={n})"
            row += f"  {cell_str:>14s}"
        lines.append(row)
    return "\n".join(lines)


def make_chart(summary: dict, png_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping chart")
        return

    import numpy as np
    n_tasks = len(TASKS)
    n_cells = len(CELLS)
    bar_w = 0.18
    x = np.arange(n_tasks)
    fig, ax = plt.subplots(figsize=(13, 5.5))

    colors = {"bare": "#4C72B0", "word-stats": "#55A868",
              "sae": "#C44E52", "combo": "#8172B2"}

    for i, cell in enumerate(CELLS):
        means = []
        ns = []
        for task, _ in TASKS:
            d = summary[task].get(cell, {})
            ns.append(d.get("n_runs", 0))
            if cell == "sae" and task == "gemma_self_deletion":
                means.append(np.nan)
            elif d.get("n_runs", 0) == 0:
                means.append(0.0)
            else:
                means.append(d["mean_gmean2"])
        offset = (i - (n_cells - 1) / 2) * bar_w
        bars = ax.bar(x + offset, means, bar_w, label=cell, color=colors.get(cell, None))
        for bar, m, n in zip(bars, means, ns):
            if cell == "sae" and bar.get_x() + bar.get_width() / 2 in [(x_ + offset) for x_, t in zip(x, TASKS) if t[0] == "gemma_self_deletion"]:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.005, "N/A", ha="center", va="bottom", fontsize=7, color="gray")
                continue
            if n == 0:
                ax.text(bar.get_x() + bar.get_width() / 2, 0.005, "—", ha="center", va="bottom", fontsize=8, color="gray")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, m + 0.005,
                        f"{m:.2f}\n(n={n})", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"task {i+1}\n{t}" for i, (t, _) in enumerate(TASKS)],
                       rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("mean gmean²")
    ax.set_title("Cross-task tool comparison (gpt-5.5 medium · 10 partitions · 40 few-shot)")
    ax.legend(loc="upper right", ncol=4, fontsize=9)
    ax.set_ylim(0, max(0.7, *(max((d.get("mean_gmean2", 0) or 0)
                                  for d in summary[t].values()) for t, _ in TASKS)) + 0.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, dpi=130)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = collect()
    summary = summarize(results)
    with open(OUT_DIR / "comparison.json", "w") as f:
        json.dump(summary, f, indent=2)
    table = make_text_table(summary)
    (OUT_DIR / "comparison.txt").write_text(table + "\n", encoding="utf-8")
    print(table)
    print()
    make_chart(summary, OUT_DIR / "comparison.png")
    print(f"Wrote {OUT_DIR / 'comparison.json'}")
    print(f"Wrote {OUT_DIR / 'comparison.txt'}")
    print(f"Wrote {OUT_DIR / 'comparison.png'}")


if __name__ == "__main__":
    main()
