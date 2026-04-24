"""Plot atypical_answer_ood tool-run g-mean² comparison."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parent
OUT_DIR = REPO / "agent-runs" / "atypical_answer_ood"
OUT_CSV = OUT_DIR / "tool_gmean2_comparison.csv"
OUT_PNG = OUT_DIR / "tool_gmean2_comparison.png"

BG = "#FAFAF8"
GRID = "#E4E2DD"
TEXT = "#2D2D2D"
TEXT_SEC = "#8A8A8A"
BASELINE_C = "#C8C4BC"
CMAP = plt.get_cmap("viridis")


ROWS = [
    {
        "tool": "no_tool",
        "display": "no_tool",
        "gmean2": 0.200,
        "used_label": "used 0/10",
        "strict_recommend_label": "recommend 0/10",
        "used_count": 0,
        "run_id": "run-20260424-164336-413508",
    },
    {
        "tool": "ask",
        "display": "ask",
        "gmean2": 0.192,
        "used_label": "used 9/10",
        "strict_recommend_label": "recommend 1/10",
        "used_count": 9,
        "run_id": "run-20260424-173248-802301",
    },
    {
        "tool": "force",
        "display": "force",
        "gmean2": 0.224,
        "used_label": "used 0/10",
        "strict_recommend_label": "recommend 0/10",
        "used_count": 0,
        "run_id": "run-20260424-173248-803643",
    },
    {
        "tool": "top_10_logits",
        "display": "top10_logits",
        "gmean2": 0.223,
        "used_label": "used 0/10",
        "strict_recommend_label": "recommend 0/10",
        "used_count": 0,
        "run_id": "run-20260424-173249-038272",
    },
    {
        "tool": "top10_entropy",
        "display": "top10_entropy",
        "gmean2": 0.198,
        "used_label": "used 0/10",
        "strict_recommend_label": "recommend 0/10",
        "used_count": 0,
        "run_id": "run-20260424-173249-219624",
    },
]


def write_csv(rows: list[dict]) -> None:
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "tool",
            "display",
            "gmean2",
            "used_label",
            "strict_recommend_label",
            "run_id",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({k: row.get(k, "") for k in fieldnames} for row in rows)


def main() -> None:
    write_csv(ROWS)

    x = np.arange(len(ROWS)) * 0.82
    vals = [row["gmean2"] for row in ROWS]
    colors = [CMAP(0.18 + 0.72 * (row["used_count"] / 10.0)) for row in ROWS]

    fig, ax = plt.subplots(figsize=(8.3, 5.0))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    bars = ax.bar(x, vals, width=0.58, color=colors, edgecolor="none", zorder=3)

    for bar, row in zip(bars, ROWS):
        val = row["gmean2"]
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.010,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color=TEXT_SEC,
        )

    xticklabels = [
        "\n".join([row["display"], row["used_label"], row["strict_recommend_label"]])
        for row in ROWS
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, fontsize=8.5, color=TEXT)
    ax.set_ylabel("OOD g-mean²", fontsize=11, color=TEXT)
    ax.set_title("Atypical answer OOD: tool runs vs no-tool", fontsize=13, color=TEXT, pad=10)

    ax.axhline(0.25, color=BASELINE_C, linewidth=1.2, linestyle="--", zorder=1)
    ax.annotate(
        "chance",
        xy=(x[0] - 0.5, 0.25),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=7,
        color=TEXT_SEC,
        ha="left",
        va="bottom",
        style="italic",
    )

    ax.set_ylim(0, 0.28)
    ax.set_xlim(x[0] - 0.52, x[-1] + 0.52)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=TEXT_SEC, length=3)
    ax.tick_params(axis="x", colors=TEXT, length=0)

    ax.text(
        0.99,
        0.98,
        "bar color = partitions using tool",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=TEXT_SEC,
    )
    ax.text(
        0.99,
        0.93,
        "recommend = explicit instruction to test agent",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.5,
        color=TEXT_SEC,
    )

    fig.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    main()
