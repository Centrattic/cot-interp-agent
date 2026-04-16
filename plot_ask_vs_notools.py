"""Compare reasoning_termination agent: no tools vs ask tool (most recent completed)."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath

BLUE_DARK    = "#4056CA"
ORANGE_DARK  = "#E87E24"
BG           = "#FAFAF8"
GRID         = "#E4E2DD"
TEXT         = "#2D2D2D"
TEXT_SEC     = "#8A8A8A"
BASELINE_C   = "#C8C4BC"

CORNER_R_PT = 5

REPO = Path(__file__).resolve().parent
PLOTS_DIR = REPO / "plots"
TASK = "reasoning_termination"

# Most recent COMPLETED runs (run-20260416-163729 is still running, 13/58)
RUN_NOTOOLS = "run-20260416-160520"  # tools=[]
RUN_ASK     = "run-20260416-155609"  # tools=["ask"]


def score_run(run_dir: Path, data_dir: Path) -> dict:
    data_files = sorted(data_dir.glob("*.json"))
    labels = [json.loads(p.read_text(encoding="utf-8"))["label"] for p in data_files]

    test_dirs = sorted([d for d in run_dir.iterdir()
                        if d.is_dir() and d.name.startswith("test-")])
    y_true, y_pred = [], []
    for td in test_dirs:
        idx = int(td.name.split("-")[-1])
        if idx >= len(labels):
            continue
        ans_path = td / "answer.txt"
        if not ans_path.exists():
            continue
        ans = ans_path.read_text(encoding="utf-8").strip().lower()
        if ans.startswith("yes") or ans == "1":
            pred = 1
        elif ans.startswith("no") or ans == "0":
            pred = 0
        else:
            continue
        y_true.append(labels[idx])
        y_pred.append(pred)

    y_true = np.array(y_true); y_pred = np.array(y_pred)
    pos = y_true == 1; neg = y_true == 0
    tpr = (y_pred[pos] == 1).mean() if pos.any() else 0.0
    tnr = (y_pred[neg] == 0).mean() if neg.any() else 0.0
    acc = (y_pred == y_true).mean() if len(y_true) else 0.0
    return {
        "n":      int(len(y_true)),
        "tpr":    float(tpr),
        "tnr":    float(tnr),
        "gmean2": float(tpr * tnr),
        "acc":    float(acc),
        "pos":    int(pos.sum()),
        "neg":    int(neg.sum()),
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
        mpath.Path(verts, codes), facecolor=color,
        edgecolor="none", zorder=zorder))


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    data_dir = REPO / "data" / TASK / "test"

    runs_root = REPO / "agent-runs" / TASK
    notools = score_run(runs_root / RUN_NOTOOLS, data_dir)
    ask     = score_run(runs_root / RUN_ASK,     data_dir)

    print(f"No tools ({RUN_NOTOOLS}): n={notools['n']}  pos/neg={notools['pos']}/{notools['neg']}  "
          f"TPR={notools['tpr']:.3f}  TNR={notools['tnr']:.3f}  "
          f"gmean²={notools['gmean2']:.3f}  acc={notools['acc']:.3f}")
    print(f"Ask     ({RUN_ASK}):     n={ask['n']}  pos/neg={ask['pos']}/{ask['neg']}  "
          f"TPR={ask['tpr']:.3f}  TNR={ask['tnr']:.3f}  "
          f"gmean²={ask['gmean2']:.3f}  acc={ask['acc']:.3f}")

    # ── Figure ────────────────────────────────────────────────────
    groups = ["g-mean²", "TPR", "TNR", "Accuracy"]
    notools_vals = [notools["gmean2"], notools["tpr"], notools["tnr"], notools["acc"]]
    ask_vals     = [ask["gmean2"],     ask["tpr"],     ask["tnr"],     ask["acc"]]

    N = len(groups)
    fig, ax = plt.subplots(figsize=(N * 1.55 + 1.5, 4.8))

    x = np.arange(N) * 1.0
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

    for i, (a, b) in enumerate(zip(notools_vals, ask_vals)):
        cx_a = x[i] + offsets[0]
        cx_b = x[i] + offsets[1]
        draw_rounded_bar(ax, cx_a, 0, a, BLUE_DARK,   bar_w, r_x, r_y)
        draw_rounded_bar(ax, cx_b, 0, b, ORANGE_DARK, bar_w, r_x, r_y)
        ax.text(cx_a, a + 0.015, f"{a:.2f}", ha="center", va="bottom",
                fontsize=9, color=TEXT_SEC)
        ax.text(cx_b, b + 0.015, f"{b:.2f}", ha="center", va="bottom",
                fontsize=9, color=TEXT_SEC)

    # Chance line for g-mean² (0.25) — only annotate that bar's baseline
    ax.axhline(y=0.5, color=BASELINE_C, linewidth=1.0, linestyle=":", zorder=1, alpha=0.6)
    ax.annotate("binary chance (acc/TPR/TNR=0.5)", xy=(x[-1] + 0.55, 0.5),
                xytext=(-4, 4), textcoords="offset points",
                fontsize=7, color=TEXT_SEC, ha="right", va="bottom", style="italic")
    ax.axhline(y=0.25, color=BASELINE_C, linewidth=1.0, linestyle="--", zorder=1)
    ax.annotate("g-mean² chance (0.25)", xy=(x[0] - 0.55, 0.25),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color=TEXT_SEC, ha="left", va="bottom", style="italic")

    # Legend
    h1 = mpatches.Patch(facecolor=BLUE_DARK,   edgecolor="none",
                        label=f"No tools  ({RUN_NOTOOLS}, n={notools['n']})")
    h2 = mpatches.Patch(facecolor=ORANGE_DARK, edgecolor="none",
                        label=f"Ask tool  ({RUN_ASK}, n={ask['n']})")
    ax.legend(handles=[h1, h2], loc="upper right", fontsize=9,
              frameon=True, facecolor=BG, edgecolor=GRID, framealpha=1,
              handlelength=1.2, handleheight=0.9)

    ax.set_facecolor(BG); fig.set_facecolor(BG)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10, color=TEXT)
    y_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{v:g}" for v in y_ticks], fontsize=9.5, color=TEXT_SEC)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_ylabel("Score", fontsize=11, color=TEXT, labelpad=8)
    ax.set_title("Reasoning termination — agent with no tools vs. ask tool",
                 fontsize=13, color=TEXT, pad=10)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=TEXT_SEC, length=3)
    ax.tick_params(axis="x", colors=TEXT, length=3)

    fig.tight_layout()
    out = PLOTS_DIR / "reasoning_termination_ask_vs_notools.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved {out}")
    plt.close()


if __name__ == "__main__":
    main()
