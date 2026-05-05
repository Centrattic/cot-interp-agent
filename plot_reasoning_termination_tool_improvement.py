"""Plot reasoning_termination_ood top_10_logits improvement vs no-tool baseline."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parent
RUNS_DIR = REPO / "agent-runs" / "reasoning_termination_ood"
OUT_DIR = REPO / "plots"
OUT_CSV = OUT_DIR / "reasoning_termination_tool_improvement.csv"
OUT_PNG = OUT_DIR / "reasoning_termination_tool_improvement.png"

BG = "#FAFAF8"
GRID = "#E4E2DD"
TEXT = "#2D2D2D"
TEXT_SEC = "#8A8A8A"
CHANCE = "#C8C4BC"
OLD_TOOL = "#B8D8BA"
NEW_TOOL = "#2E7D32"
NO_TOOL = "#A89F90"
RUN_SPECS = [
    {
        "label": "Old\ntop_10_logits",
        "tool": "top_10_logits",
        "run_id": "run-20260424-162445-737040",
        "note": "earlier top_10_logits run",
        "color": OLD_TOOL,
    },
    {
        "label": "Updated\ntop_10_logits",
        "tool": "top_10_logits",
        "run_id": "run-20260427-172353-231472",
        "note": "current top_10_logits run",
        "color": NEW_TOOL,
    },
    {
        "label": "Current\nno tool",
        "tool": "no_tool",
        "run_id": "run-20260424-182810-528027",
        "note": "10-partition no-tool reference",
        "color": NO_TOOL,
    },
]


def parse_summary_metrics(summary_path: Path) -> dict[str, float]:
    text = summary_path.read_text(encoding="utf-8")
    match = re.search(
        r"## Aggregate \(all partitions combined\)\n"
        r"n=(?P<n>\d+)\s+miss=(?P<miss>\d+)\s+acc=(?P<acc>[0-9.]+)%\s+"
        r"TPR=(?P<tpr>[0-9.]+)\s+TNR=(?P<tnr>[0-9.]+)\s+gmean²=(?P<gmean2>[0-9.]+)",
        text,
    )
    if not match:
        raise ValueError(f"Could not parse aggregate metrics from {summary_path}")
    return {
        "n": int(match.group("n")),
        "miss": int(match.group("miss")),
        "acc": float(match.group("acc")) / 100.0,
        "tpr": float(match.group("tpr")),
        "tnr": float(match.group("tnr")),
        "gmean2": float(match.group("gmean2")),
    }


def collect_rows() -> list[dict]:
    rows = []
    for spec in RUN_SPECS:
        run_dir = RUNS_DIR / spec["run_id"]
        run_meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        metrics = parse_summary_metrics(run_dir / "summary.txt")
        rows.append(
            {
                "label": spec["label"],
                "tool": spec["tool"],
                "run_id": spec["run_id"],
                "created": run_meta["created"],
                "note": spec["note"],
                "color": spec["color"],
                **metrics,
            }
        )
    return rows


def write_csv(rows: list[dict]) -> None:
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "label",
            "tool",
            "run_id",
            "created",
            "note",
            "n",
            "miss",
            "acc",
            "tpr",
            "tnr",
            "gmean2",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in fieldnames} for row in rows)


def main() -> None:
    rows = collect_rows()
    write_csv(rows)

    x = np.arange(len(rows))
    vals = [row["gmean2"] for row in rows]
    colors = [row["color"] for row in rows]

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars = ax.bar(x, vals, width=0.60, color=colors, edgecolor="none", zorder=3)

    for bar, row in zip(bars, rows):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            row["gmean2"] + 0.010,
            f"{row['gmean2']:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT_SEC,
        )

    ax.axhline(0.25, color=CHANCE, linewidth=1.2, linestyle="--", zorder=1)
    ax.annotate(
        "chance",
        xy=(-0.45, 0.25),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=7,
        color=TEXT_SEC,
        ha="left",
        va="bottom",
        style="italic",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([row["label"] for row in rows], fontsize=10, color=TEXT)
    ax.set_ylabel("OOD g-mean²", fontsize=11, color=TEXT)
    ax.set_title(
        "Reasoning termination OOD: top_10_logits improvement",
        fontsize=13,
        color=TEXT,
        pad=10,
    )

    ax.set_ylim(0, 0.34)
    ax.set_xlim(-0.52, x[-1] + 0.56)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=TEXT_SEC, length=3)
    ax.tick_params(axis="x", colors=TEXT, length=0)

    fig.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    main()
