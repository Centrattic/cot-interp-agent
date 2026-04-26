"""Compare reasoning_termination_ood base run vs 40-shot run."""
from __future__ import annotations

import csv
import json
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
OUT_PNG = PLOTS_DIR / "reasoning_termination_ood_40shot_vs_base.png"
OUT_CSV = PLOTS_DIR / "reasoning_termination_ood_40shot_vs_base.csv"

TASK = "reasoning_termination_ood"
RUN_BASE = "run-20260424-162647-288149"
RUN_40SHOT = "run-20260424-154451-520105"


def score_partitioned_run(run_dir: Path) -> dict:
    rows = list(csv.DictReader((run_dir / "results.csv").open()))
    y_true, y_pred = [], []
    for row in rows:
        try:
            y_true.append(int(row["label"]))
            y_pred.append(int(row["pred"]))
        except (KeyError, TypeError, ValueError):
            continue
    yt = np.array(y_true)
    yp = np.array(y_pred)
    pos = yt == 1
    neg = yt == 0
    tpr = float((yp[pos] == 1).mean()) if pos.any() else 0.0
    tnr = float((yp[neg] == 0).mean()) if neg.any() else 0.0
    acc = float((yp == yt).mean()) if len(yt) else 0.0
    return {
        "n": int(len(yt)),
        "tpr": tpr,
        "tnr": tnr,
        "gmean2": tpr * tnr,
        "acc": acc,
    }


def score_single_run(run_dir: Path, task: str) -> dict:
    rows = list(csv.DictReader((run_dir / "results.csv").open()))
    data_dir = REPO / "data" / task / "test"
    labels = {p.stem: json.loads(p.read_text(encoding="utf-8"))["label"] for p in sorted(data_dir.glob("*.json"))}
    y_true, y_pred = [], []
    for row in rows:
        ex = row["example_id"]
        ans = row["answer"].strip().lower()
        if ans == "yes":
            pred = 1
        elif ans == "no":
            pred = 0
        else:
            continue
        y_true.append(int(labels[ex]))
        y_pred.append(pred)
    yt = np.array(y_true)
    yp = np.array(y_pred)
    pos = yt == 1
    neg = yt == 0
    tpr = float((yp[pos] == 1).mean()) if pos.any() else 0.0
    tnr = float((yp[neg] == 0).mean()) if neg.any() else 0.0
    acc = float((yp == yt).mean()) if len(yt) else 0.0
    return {
        "n": int(len(yt)),
        "tpr": tpr,
        "tnr": tnr,
        "gmean2": tpr * tnr,
        "acc": acc,
    }


def draw_rounded_bar(ax, cx, bottom, height, color, width, r_x, r_y, zorder=3):
    if height <= 0:
        return
    x0, y0, h = cx - width / 2, bottom, height
    rx = min(r_x, width / 2)
    ry = min(r_y, h / 2)
    verts = [
        (x0, y0), (x0, y0 + h - ry),
        (x0, y0 + h), (x0 + rx, y0 + h),
        (x0 + width - rx, y0 + h), (x0 + width, y0 + h),
        (x0 + width, y0 + h - ry), (x0 + width, y0),
        (x0, y0),
    ]
    codes = [
        mpath.Path.MOVETO, mpath.Path.LINETO,
        mpath.Path.CURVE3, mpath.Path.CURVE3,
        mpath.Path.LINETO,
        mpath.Path.CURVE3, mpath.Path.CURVE3,
        mpath.Path.LINETO, mpath.Path.CLOSEPOLY,
    ]
    ax.add_patch(mpatches.PathPatch(
        mpath.Path(verts, codes), facecolor=color, edgecolor="none", zorder=zorder))


def write_csv(base: dict, forty: dict) -> None:
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "run", "n", "gmean2", "tpr", "tnr", "acc"])
        writer.writeheader()
        writer.writerow({"variant": "base", "run": RUN_BASE, **base})
        writer.writerow({"variant": "40_shot", "run": RUN_40SHOT, **forty})


def main():
    PLOTS_DIR.mkdir(exist_ok=True)

    runs_root = REPO / "agent-runs" / TASK
    base = score_partitioned_run(runs_root / RUN_BASE)
    forty = score_single_run(runs_root / RUN_40SHOT, TASK)
    write_csv(base, forty)

    print(
        f"Base   ({RUN_BASE}): n={base['n']}  "
        f"TPR={base['tpr']:.3f}  TNR={base['tnr']:.3f}  gmean²={base['gmean2']:.3f}  acc={base['acc']:.3f}"
    )
    print(
        f"40shot ({RUN_40SHOT}): n={forty['n']}  "
        f"TPR={forty['tpr']:.3f}  TNR={forty['tnr']:.3f}  gmean²={forty['gmean2']:.3f}  acc={forty['acc']:.3f}"
    )

    groups = ["g-mean²", "TPR", "TNR", "Accuracy"]
    base_vals = [base["gmean2"], base["tpr"], base["tnr"], base["acc"]]
    forty_vals = [forty["gmean2"], forty["tpr"], forty["tnr"], forty["acc"]]

    n = len(groups)
    fig, ax = plt.subplots(figsize=(n * 1.55 + 1.5, 4.8))

    x = np.arange(n) * 1.0
    bar_w = 0.34
    gap = 0.04
    offsets = np.array([-1, 1]) * (bar_w + gap) / 2

    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylim(0, 1.0)
    fig.canvas.draw()

    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((1, 1))
    sx, sy = p1[0] - p0[0], p1[1] - p0[1]
    r_px = CORNER_R_PT * fig.dpi / 72.0
    r_x, r_y = r_px / sx, r_px / sy

    for i, (a, b) in enumerate(zip(base_vals, forty_vals)):
        cx_a = x[i] + offsets[0]
        cx_b = x[i] + offsets[1]
        draw_rounded_bar(ax, cx_a, 0, a, BLUE_DARK, bar_w, r_x, r_y)
        draw_rounded_bar(ax, cx_b, 0, b, ORANGE_DARK, bar_w, r_x, r_y)
        ax.text(cx_a, a + 0.015, f"{a:.2f}", ha="center", va="bottom", fontsize=9, color=TEXT_SEC)
        ax.text(cx_b, b + 0.015, f"{b:.2f}", ha="center", va="bottom", fontsize=9, color=TEXT_SEC)

    ax.axhline(y=0.5, color=BASELINE_C, linewidth=1.0, linestyle=":", zorder=1, alpha=0.6)
    ax.annotate(
        "binary chance (acc/TPR/TNR=0.5)",
        xy=(x[-1] + 0.55, 0.5),
        xytext=(-4, 4),
        textcoords="offset points",
        fontsize=7,
        color=TEXT_SEC,
        ha="right",
        va="bottom",
        style="italic",
    )
    ax.axhline(y=0.25, color=BASELINE_C, linewidth=1.0, linestyle="--", zorder=1)
    ax.annotate(
        "g-mean² chance (0.25)",
        xy=(x[0] - 0.55, 0.25),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=7,
        color=TEXT_SEC,
        ha="left",
        va="bottom",
        style="italic",
    )

    h1 = mpatches.Patch(facecolor=BLUE_DARK, edgecolor="none", label=f"Base no-tool  ({RUN_BASE}, n={base['n']})")
    h2 = mpatches.Patch(facecolor=ORANGE_DARK, edgecolor="none", label=f"40-shot no-tool  ({RUN_40SHOT}, n={forty['n']})")
    ax.legend(
        handles=[h1, h2],
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
    ax.set_xticklabels(groups, fontsize=10, color=TEXT)
    y_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{v:g}" for v in y_ticks], fontsize=9.5, color=TEXT_SEC)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_ylabel("Score", fontsize=11, color=TEXT, labelpad=8)
    ax.set_title("Reasoning termination OOD — base vs 40-shot no-tool", fontsize=13, color=TEXT, pad=10)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=TEXT_SEC, length=3)
    ax.tick_params(axis="x", colors=TEXT, length=3)

    fig.tight_layout()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved {OUT_PNG}")
    print(f"Saved {OUT_CSV}")
    plt.close()


if __name__ == "__main__":
    main()
