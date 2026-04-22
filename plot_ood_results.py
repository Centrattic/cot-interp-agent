"""Single plot showing OOD agent results across all OOD task variants.

Renders g-mean², TPR, TNR, and accuracy as grouped bars, one group per
OOD task. Tasks with empty results.csv are shown as 0-height bars and
labelled "(pending)" in the x-axis tick.

Picks the latest run-* directory per *_ood task (regardless of completeness).
"""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath


BLUE_DARK    = "#4056CA"
ORANGE_DARK  = "#E87E24"
GREEN_DARK   = "#2E8B57"
PURPLE_DARK  = "#8E5BB7"
BG           = "#FAFAF8"
GRID         = "#E4E2DD"
TEXT         = "#2D2D2D"
TEXT_SEC     = "#8A8A8A"
BASELINE_C   = "#C8C4BC"

CORNER_R_PT = 4

REPO = Path(__file__).resolve().parent
PLOTS_DIR = REPO / "plots"

OOD_TASKS = [
    ("reasoning_termination_ood", "Reasoning\ntermination"),
    ("gemma_self_deletion_ood",   "Self\ndeletion"),
    ("followup_confidence_ood",   "Follow-up\nconfidence"),
    ("stanford_hint_ood",         "Stanford\nhint"),
    ("atypical_answer_ood",       "Atypical\nanswer"),
    ("atypical_cot_length_ood",   "Atypical\nCoT length"),
]

METRICS = [
    ("gmean2", "g-mean\u00b2", BLUE_DARK),
    ("tpr",    "TPR",          ORANGE_DARK),
    ("tnr",    "TNR",          GREEN_DARK),
    ("acc",    "Accuracy",     PURPLE_DARK),
]


def latest_run(task: str) -> Path | None:
    base = REPO / "agent-runs" / task
    if not base.exists():
        return None
    runs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("run-")])
    return runs[-1] if runs else None


def score_run(run: Path) -> dict:
    """Score from results.csv. Returns 0s + n=0 if no data rows."""
    f = run / "results.csv"
    y_true, y_pred = [], []
    if f.exists():
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    label = int(r["label"]); pred = int(r["pred"])
                except (ValueError, KeyError, TypeError):
                    continue
                y_true.append(label); y_pred.append(pred)
    yt = np.array(y_true); yp = np.array(y_pred)
    pos = yt == 1; neg = yt == 0
    tpr = float((yp[pos] == 1).mean()) if pos.any() else 0.0
    tnr = float((yp[neg] == 0).mean()) if neg.any() else 0.0
    acc = float((yp == yt).mean()) if len(yt) else 0.0
    n_total_test = 0
    test_dir = REPO / "data" / run.parent.name / "test"
    if test_dir.exists():
        n_total_test = sum(1 for _ in test_dir.glob("*.json"))
    return {
        "n": int(len(yt)),
        "n_total": n_total_test,
        "tpr": tpr, "tnr": tnr, "acc": acc, "gmean2": tpr * tnr,
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


def main():
    PLOTS_DIR.mkdir(exist_ok=True)

    rows = []
    print(f"{'task':30s} {'run':25s}  n/total   TPR    TNR    gmean2  acc")
    for task, label in OOD_TASKS:
        run = latest_run(task)
        if run is None:
            print(f"{task:30s} (no run found)")
            rows.append((task, label, None, {"n": 0, "n_total": 0,
                                              "tpr": 0, "tnr": 0, "acc": 0, "gmean2": 0}))
            continue
        s = score_run(run)
        print(f"{task:30s} {run.name:25s}  {s['n']:3d}/{s['n_total']:<3d}  "
              f"{s['tpr']:.3f}  {s['tnr']:.3f}  {s['gmean2']:.3f}  {s['acc']:.3f}")
        rows.append((task, label, run, s))

    N = len(rows); M = len(METRICS)
    bar_pitch_in = 1.55
    fig, ax = plt.subplots(figsize=(N * bar_pitch_in + 1.6, 5.4))

    x = np.arange(N) * 1.0
    bar_w = 0.84 / M
    gap = 0.025
    centers = (np.arange(M) - (M - 1) / 2) * (bar_w + gap)

    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylim(0, 1.0)
    fig.canvas.draw()

    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((1, 1))
    sx, sy = p1[0] - p0[0], p1[1] - p0[1]
    r_px = CORNER_R_PT * fig.dpi / 72.0
    r_x, r_y = r_px / sx, r_px / sy

    for i, (task, label, run, s) in enumerate(rows):
        for j, (k, _, color) in enumerate(METRICS):
            cx = x[i] + centers[j]
            v = s[k]
            draw_rounded_bar(ax, cx, 0, v, color, bar_w, r_x, r_y)
            if s["n"] > 0:
                ax.text(cx, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7, color=TEXT_SEC, rotation=0)
        if s["n"] == 0:
            ax.text(x[i], 0.04, "pending", ha="center", va="bottom",
                    fontsize=8, color=TEXT_SEC, style="italic")

    # Chance lines
    ax.axhline(y=0.5, color=BASELINE_C, linewidth=1.0, linestyle=":", zorder=1, alpha=0.6)
    ax.annotate("binary chance (acc/TPR/TNR=0.5)", xy=(x[-1] + 0.55, 0.5),
                xytext=(-4, 4), textcoords="offset points",
                fontsize=7, color=TEXT_SEC, ha="right", va="bottom", style="italic")
    ax.axhline(y=0.25, color=BASELINE_C, linewidth=1.0, linestyle="--", zorder=1)
    ax.annotate("g-mean\u00b2 chance (0.25)", xy=(x[0] - 0.55, 0.25),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color=TEXT_SEC, ha="left", va="bottom", style="italic")

    handles = [mpatches.Patch(facecolor=color, edgecolor="none", label=name)
               for _, name, color in METRICS]
    ax.legend(handles=handles, loc="upper right", ncol=4, fontsize=9,
              frameon=True, facecolor=BG, edgecolor=GRID, framealpha=1,
              handlelength=1.2, handleheight=0.9)

    ax.set_facecolor(BG); fig.set_facecolor(BG)
    ax.set_xticks(x)
    xtl = []
    for task, label, run, s in rows:
        suffix = f"\nn={s['n']}/{s['n_total']}" if run is not None else "\n(no run)"
        xtl.append(label + suffix)
    ax.set_xticklabels(xtl, fontsize=9, color=TEXT)
    y_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{v:g}" for v in y_ticks], fontsize=9.5, color=TEXT_SEC)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_ylabel("Score", fontsize=11, color=TEXT, labelpad=8)
    ax.set_title("OOD agent results — base scaffold (no tools), 10 partitions per task",
                 fontsize=13, color=TEXT, pad=10)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=TEXT_SEC, length=3)
    ax.tick_params(axis="x", colors=TEXT, length=3)

    fig.tight_layout()
    out = PLOTS_DIR / "ood_results_all.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved {out}")
    plt.close()


if __name__ == "__main__":
    main()
