"""Per-dataset bar plot: no-tools agent vs each tool variant (ask, force).

For every in-distribution dataset that has multiple tool runs, emits one plot
to plots/<task>_tools_vs_notools.png comparing g-mean², TPR, TNR, accuracy
across {no tools, ask, force}.

Run selection: for each (task, tools) tuple, picks the latest run-* directory
with n_strategies=10 and a non-empty results.csv.
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
BG           = "#FAFAF8"
GRID         = "#E4E2DD"
TEXT         = "#2D2D2D"
TEXT_SEC     = "#8A8A8A"
BASELINE_C   = "#C8C4BC"

CORNER_R_PT = 5

REPO = Path(__file__).resolve().parent
PLOTS_DIR = REPO / "plots"

# Visual ordering / colors per tool variant
TOOL_ORDER = [(), ("ask",), ("force",)]
TOOL_LABEL = {(): "No tools", ("ask",): "ask", ("force",): "force"}
TOOL_COLOR = {(): BLUE_DARK, ("ask",): ORANGE_DARK, ("force",): GREEN_DARK}


def list_runs(task: str) -> list[Path]:
    base = REPO / "agent-runs" / task
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("run-")])


def run_meta(run: Path) -> dict | None:
    meta = run / "run.json"
    if not meta.exists():
        return None
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except Exception:
        return None


def results_rows(run: Path) -> list[dict]:
    f = run / "results.csv"
    if not f.exists():
        return []
    with open(f, newline="") as fh:
        return list(csv.DictReader(fh))


def pick_run(task: str, tools_key: tuple[str, ...]) -> Path | None:
    """Latest run-* with matching tools and non-empty results.csv."""
    candidates = []
    for r in list_runs(task):
        m = run_meta(r)
        if m is None:
            continue
        if tuple(sorted(m.get("tools", []))) != tuple(sorted(tools_key)):
            continue
        if int(m.get("n_strategies", 1)) != 10:
            continue
        if not results_rows(r):
            continue
        candidates.append(r)
    return candidates[-1] if candidates else None


def score(rows: list[dict]) -> dict:
    y_true, y_pred = [], []
    for r in rows:
        try:
            label = int(r["label"])
            pred = int(r["pred"])
        except (ValueError, KeyError, TypeError):
            continue
        y_true.append(label); y_pred.append(pred)
    yt = np.array(y_true); yp = np.array(y_pred)
    pos = yt == 1; neg = yt == 0
    tpr = float((yp[pos] == 1).mean()) if pos.any() else 0.0
    tnr = float((yp[neg] == 0).mean()) if neg.any() else 0.0
    acc = float((yp == yt).mean()) if len(yt) else 0.0
    return {"n": int(len(yt)), "tpr": tpr, "tnr": tnr, "gmean2": tpr * tnr, "acc": acc}


def datasets_with_multiple_tools() -> list[str]:
    """Return tasks where >1 distinct tool variant has a usable run."""
    runs_root = REPO / "agent-runs"
    tasks = []
    for task_dir in sorted(runs_root.iterdir()):
        if not task_dir.is_dir() or task_dir.name.endswith("_ood"):
            continue
        variants = set()
        for r in list_runs(task_dir.name):
            m = run_meta(r)
            if m and int(m.get("n_strategies", 1)) == 10 and results_rows(r):
                variants.add(tuple(sorted(m.get("tools", []))))
        if len(variants) >= 2:
            tasks.append(task_dir.name)
    return tasks


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


def plot_one(task: str) -> Path | None:
    selected: list[tuple[tuple[str, ...], Path, dict]] = []
    for tk in TOOL_ORDER:
        run = pick_run(task, tk)
        if run is None:
            continue
        s = score(results_rows(run))
        selected.append((tk, run, s))

    if len(selected) < 2:
        print(f"[{task}] Skipping — only {len(selected)} variant(s) usable.")
        return None

    print(f"[{task}]")
    for tk, run, s in selected:
        print(f"  {TOOL_LABEL[tk]:9s}  {run.name}  n={s['n']:3d}  "
              f"TPR={s['tpr']:.3f}  TNR={s['tnr']:.3f}  "
              f"gmean²={s['gmean2']:.3f}  acc={s['acc']:.3f}")

    metrics = ["g-mean²", "TPR", "TNR", "Accuracy"]
    keys    = ["gmean2",  "tpr", "tnr", "acc"]
    N = len(metrics); M = len(selected)

    fig, ax = plt.subplots(figsize=(N * (0.5 * M + 1.0) + 1.0, 4.8))

    x = np.arange(N) * 1.0
    bar_w = 0.84 / M
    gap = 0.04
    centers = (np.arange(M) - (M - 1) / 2) * (bar_w + gap)

    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.set_ylim(0, 1.0)
    fig.canvas.draw()

    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((1, 1))
    sx, sy = p1[0] - p0[0], p1[1] - p0[1]
    r_px = CORNER_R_PT * fig.dpi / 72.0
    r_x, r_y = r_px / sx, r_px / sy

    for j, (tk, run, s) in enumerate(selected):
        for i, k in enumerate(keys):
            cx = x[i] + centers[j]
            v = s[k]
            draw_rounded_bar(ax, cx, 0, v, TOOL_COLOR[tk], bar_w, r_x, r_y)
            ax.text(cx, v + 0.015, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=8, color=TEXT_SEC)

    # Chance baselines
    ax.axhline(y=0.5, color=BASELINE_C, linewidth=1.0, linestyle=":", zorder=1, alpha=0.6)
    ax.annotate("binary chance (acc/TPR/TNR=0.5)", xy=(x[-1] + 0.55, 0.5),
                xytext=(-4, 4), textcoords="offset points",
                fontsize=7, color=TEXT_SEC, ha="right", va="bottom", style="italic")
    ax.axhline(y=0.25, color=BASELINE_C, linewidth=1.0, linestyle="--", zorder=1)
    ax.annotate("g-mean² chance (0.25)", xy=(x[0] - 0.55, 0.25),
                xytext=(4, 4), textcoords="offset points",
                fontsize=7, color=TEXT_SEC, ha="left", va="bottom", style="italic")

    handles = [
        mpatches.Patch(facecolor=TOOL_COLOR[tk], edgecolor="none",
                       label=f"{TOOL_LABEL[tk]}  ({run.name}, n={s['n']})")
        for tk, run, s in selected
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8.5,
              frameon=True, facecolor=BG, edgecolor=GRID, framealpha=1,
              handlelength=1.2, handleheight=0.9)

    ax.set_facecolor(BG); fig.set_facecolor(BG)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10, color=TEXT)
    y_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{v:g}" for v in y_ticks], fontsize=9.5, color=TEXT_SEC)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_ylabel("Score", fontsize=11, color=TEXT, labelpad=8)
    ax.set_title(f"{task} — agent variants (no tools vs ask vs force)",
                 fontsize=13, color=TEXT, pad=10)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=TEXT_SEC, length=3)
    ax.tick_params(axis="x", colors=TEXT, length=3)

    fig.tight_layout()
    out = PLOTS_DIR / f"{task}_tools_vs_notools.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"  Saved {out}")
    plt.close()
    return out


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    tasks = datasets_with_multiple_tools()
    if not tasks:
        print("No in-distribution tasks have ≥2 tool variants with results.")
        return
    print(f"Plotting {len(tasks)} task(s): {', '.join(tasks)}\n")
    for t in tasks:
        plot_one(t)


if __name__ == "__main__":
    main()
