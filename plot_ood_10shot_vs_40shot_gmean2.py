"""Plot OOD g-mean² for base agent with 10-shot vs 40-shot few-shot context."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np


BLUE_DARK = "#4056CA"
ORANGE_DARK = "#E87E24"
BG = "#FAFAF8"
GRID = "#E4E2DD"
TEXT = "#2D2D2D"
TEXT_SEC = "#8A8A8A"
BASELINE_C = "#C8C4BC"

CORNER_R_PT = 5

REPO = Path(__file__).resolve().parent
PLOTS_DIR = REPO / "plots"
OUT_CSV = PLOTS_DIR / "ood_10shot_vs_40shot_gmean2.csv"
OUT_PNG = PLOTS_DIR / "ood_10shot_vs_40shot_gmean2.png"

TASKS = [
    (
        "Reasoning\ntermination",
        REPO / "agent-runs" / "reasoning_termination_ood" / "run-20260423-175349" / "summary.txt",
        REPO / "agent-runs" / "reasoning_termination_ood" / "run-20260424-182810-528027" / "summary.txt",
    ),
    (
        "Self\ndeletion",
        REPO / "agent-runs" / "gemma_self_deletion_ood" / "run-20260423-175349" / "summary.txt",
        REPO / "agent-runs" / "gemma_self_deletion_ood" / "run-20260424-182810-524393" / "summary.txt",
    ),
    (
        "Follow-up\nresponse",
        REPO / "agent-runs" / "followup_confidence_ood" / "run-20260423-175931" / "summary.txt",
        REPO / "agent-runs" / "followup_confidence_ood" / "run-20260424-182810-754498" / "summary.txt",
    ),
    (
        "Stanford\nhint",
        REPO / "agent-runs" / "stanford_hint_ood" / "run-20260423-180101" / "summary.txt",
        REPO / "agent-runs" / "stanford_hint_ood" / "run-20260424-182810-754576" / "summary.txt",
    ),
    (
        "Atypical\nanswer",
        REPO / "agent-runs" / "atypical_answer_ood" / "run-20260423-180546" / "summary.txt",
        REPO / "agent-runs" / "atypical_answer_ood" / "run-20260424-182810-783801" / "summary.txt",
    ),
    (
        "Atypical\nCoT length",
        REPO / "agent-runs" / "atypical_cot_length_ood" / "run-20260423-180933" / "summary.txt",
        REPO / "agent-runs" / "atypical_cot_length_ood" / "run-20260424-182810-785546" / "summary.txt",
    ),
]

GMEAN_RE = re.compile(r"gmean²=([0-9.]+)")


def parse_gmean2(summary_path: Path) -> float:
    text = summary_path.read_text(encoding="utf-8")
    m = GMEAN_RE.search(text)
    if not m:
        raise RuntimeError(f"gmean² not found in {summary_path}")
    return float(m.group(1))


def draw_rounded_bar(ax, cx, bottom, height, color, width, r_x, r_y, *, zorder=3):
    if not np.isfinite(height) or height <= 0:
        return
    x0, y0, h = cx - width / 2, bottom, height
    rx = min(r_x, width / 2)
    ry = min(r_y, h / 2)
    verts = [
        (x0, y0),
        (x0, y0 + h - ry),
        (x0, y0 + h),
        (x0 + rx, y0 + h),
        (x0 + width - rx, y0 + h),
        (x0 + width, y0 + h),
        (x0 + width, y0 + h - ry),
        (x0 + width, y0),
        (x0, y0),
    ]
    codes = [
        mpath.Path.MOVETO,
        mpath.Path.LINETO,
        mpath.Path.CURVE3,
        mpath.Path.CURVE3,
        mpath.Path.LINETO,
        mpath.Path.CURVE3,
        mpath.Path.CURVE3,
        mpath.Path.LINETO,
        mpath.Path.CLOSEPOLY,
    ]
    ax.add_patch(
        mpatches.PathPatch(
            mpath.Path(verts, codes),
            facecolor=color,
            edgecolor="none",
            zorder=zorder,
        )
    )


def write_csv(rows: list[dict[str, object]]) -> None:
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "shot_setting", "gmean2", "summary_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)

    rows = []
    for label, ten_path, forty_path in TASKS:
        rows.append({
            "label": label.replace("\n", " "),
            "shot_setting": "10-shot",
            "gmean2": parse_gmean2(ten_path),
            "summary_path": str(ten_path),
        })
        rows.append({
            "label": label.replace("\n", " "),
            "shot_setting": "40-shot",
            "gmean2": parse_gmean2(forty_path),
            "summary_path": str(forty_path),
        })

    write_csv(rows)

    labels = [label for label, _, _ in TASKS]
    vals10 = [parse_gmean2(ten_path) for _, ten_path, _ in TASKS]
    vals40 = [parse_gmean2(forty_path) for _, _, forty_path in TASKS]

    n = len(TASKS)
    fig, ax = plt.subplots(figsize=(n * 1.45 + 1.6, 5.2))
    x = np.arange(n)
    bar_w = 0.33
    gap = 0.04
    offsets = np.array([-1, 1]) * (bar_w + gap) / 2

    ymax = max(max(vals10), max(vals40))
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    ax.set_ylim(0, max(0.9, ymax * 1.15))
    fig.canvas.draw()

    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((1, 1))
    sx, sy = p1[0] - p0[0], p1[1] - p0[1]
    r_px = CORNER_R_PT * fig.dpi / 72.0
    r_x, r_y = r_px / sx, r_px / sy

    for i, (v10, v40) in enumerate(zip(vals10, vals40)):
        draw_rounded_bar(ax, x[i] + offsets[0], 0, v10, BLUE_DARK, bar_w, r_x, r_y)
        draw_rounded_bar(ax, x[i] + offsets[1], 0, v40, ORANGE_DARK, bar_w, r_x, r_y)
        ax.text(x[i] + offsets[0], v10 + 0.012, f"{v10:.2f}", ha="center", va="bottom", fontsize=7.5, color=TEXT_SEC)
        ax.text(x[i] + offsets[1], v40 + 0.012, f"{v40:.2f}", ha="center", va="bottom", fontsize=7.5, color=TEXT_SEC)

    ax.axhline(y=0.25, color=BASELINE_C, linewidth=1.2, linestyle="--", zorder=1)
    ax.annotate(
        "chance",
        xy=(x[0] - 0.48, 0.25),
        xytext=(0, 4),
        textcoords="offset points",
        fontsize=7,
        color=TEXT_SEC,
        ha="left",
        va="bottom",
        style="italic",
    )

    handles = [
        mpatches.Patch(facecolor=BLUE_DARK, edgecolor="none", label="Agent (OOD, 10-shot)"),
        mpatches.Patch(facecolor=ORANGE_DARK, edgecolor="none", label="Agent (OOD, 40-shot)"),
    ]
    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=9,
        frameon=True,
        facecolor=BG,
        edgecolor=GRID,
        framealpha=1,
        handlelength=1.2,
        handleheight=0.9,
    )

    ax.set_facecolor(BG)
    fig.set_facecolor(BG)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, color=TEXT)
    y_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{v:g}" for v in y_ticks], fontsize=9.5, color=TEXT_SEC)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_ylabel("OOD g-mean²", fontsize=11, color=TEXT, labelpad=8)
    ax.set_title("OOD base agent: 10-shot vs. 40-shot", fontsize=13, color=TEXT, pad=10)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=TEXT_SEC, length=3)
    ax.tick_params(axis="x", colors=TEXT, length=3)

    fig.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"Saved {OUT_PNG}")
    print(f"Saved {OUT_CSV}")
    plt.close()


if __name__ == "__main__":
    main()
