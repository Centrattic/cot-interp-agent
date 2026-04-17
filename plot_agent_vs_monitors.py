"""Plot comparison of best monitor vs agent vs best overall method (ID) per task."""
from __future__ import annotations
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath

# Palette (mirrors the provided plot style)
BLUE_LIGHT   = "#BFC8F5"
BLUE_DARK    = "#4056CA"
ORANGE_LIGHT = "#FBDBB8"
ORANGE_DARK  = "#E87E24"
GREEN_DARK   = "#2E8B57"
BG           = "#FAFAF8"
GRID         = "#E4E2DD"
TEXT         = "#2D2D2D"
TEXT_SEC     = "#8A8A8A"
BASELINE_C   = "#C8C4BC"

CORNER_R_PT = 5

REPO = Path(__file__).resolve().parent
RESULTS_CSV = REPO.parent / "cot-proxy-tasks" / "results_summary.csv"
PLOTS_DIR = REPO / "plots"

# Map CSV task id → agent-run task dir
CSV_TO_RUN_DIR = {
    "1_reasoning_termination":  "reasoning_termination",
    "2_self_deletion":          "gemma_self_deletion",
    "3_follow_up_response":     "followup_confidence",
    "4_user_preference":        "user_preference_sycophancy",
    "5_stanford_hint":          "stanford_hint",
    "6_atypical_answer":        "atypical_answer",
    "7_atypical_cot_length":    "atypical_cot_length",
}

# Pin specific runs (instead of "latest") when a particular variant is canonical
PINNED_RUNS = {
    # Base agent (no tools) with val-split few-shot
    "reasoning_termination": "run-20260416-170803",
    # Clean rerun of the base (no tools) agent
    "atypical_answer":       "run-20260416-165206",
    # Clean rerun of the base (no tools) agent
    "atypical_cot_length":   "run-20260416-165207",
}

# Short labels used on x-axis
TASK_LABELS = {
    "1_reasoning_termination":  "Reasoning\ntermination",
    "2_self_deletion":          "Self\ndeletion",
    "3_follow_up_response":     "Follow-up\nresponse",
    "4_user_preference":        "User\npreference",
    "5_stanford_hint":          "Stanford\nhint",
    "6_atypical_answer":        "Atypical\nanswer",
    "7_atypical_cot_length":    "Atypical\nCoT length",
}

LLM_METHODS = {
    "Few-shot LLMs", "Zero-shot LLMs",
    "Few-shot LLMs (binary)", "Few-shot LLMs (conf)",
    "Zero-shot LLMs (binary)", "Zero-shot LLMs (conf)",
}


def load_results() -> dict:
    """Return nested dict: task -> {method -> {metric -> (id, ood)}}."""
    rows = {}
    with open(RESULTS_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            task = r["task"]
            method = r["method"]
            metric = r["metric"]
            id_val = float(r["id"]) if r["id"] else float("nan")
            ood_val = float(r["ood"]) if r["ood"] else float("nan")
            rows.setdefault(task, {}).setdefault(method, {})[metric] = (id_val, ood_val)
    return rows


def best_id_gmean2(methods: dict, llm_only: bool) -> tuple[str, float]:
    """Find method with best ID gmean² score (among LLM-only or all)."""
    best_name, best_val = None, -1.0
    for m, metrics in methods.items():
        if "gmean2" not in metrics:
            continue
        if llm_only and m not in LLM_METHODS:
            continue
        if not llm_only and m in LLM_METHODS:
            continue
        id_val = metrics["gmean2"][0]
        if not np.isnan(id_val) and id_val > best_val:
            best_val = id_val
            best_name = m
    return best_name, best_val


def best_overall_id(methods: dict) -> tuple[str, float]:
    """Best method across all of them (ID gmean²)."""
    best_name, best_val = None, -1.0
    for m, metrics in methods.items():
        if "gmean2" not in metrics:
            continue
        id_val = metrics["gmean2"][0]
        if not np.isnan(id_val) and id_val > best_val:
            best_val = id_val
            best_name = m
    return best_name, best_val


def pick_latest_run(task_dir: Path) -> Path:
    runs = sorted([p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("run-")])
    return runs[-1]


def compute_agent_gmean2(task_run_dir: str, data_dir: Path) -> tuple[float, int, int, str]:
    """Compute g-mean² = TPR × TNR for agent outputs vs ground-truth labels.

    Matches test-NNN dirs to the N-th sorted example in data/<task>/test/
    (that's the order run_tests.py dispatches them in).

    Returns (gmean2, n_tests_scored, n_tests_total_in_run, run_name).
    """
    run_root = REPO / "agent-runs" / task_run_dir
    pinned = PINNED_RUNS.get(task_run_dir)
    if pinned:
        run = run_root / pinned
    else:
        run = pick_latest_run(run_root)

    # Ordered list of labels, matching dispatch order
    data_files = sorted(data_dir.glob("*.json"))
    labels = [json.loads(p.read_text(encoding="utf-8"))["label"] for p in data_files]

    test_dirs = sorted([d for d in run.iterdir() if d.is_dir() and d.name.startswith("test-")])

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

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    pos = y_true == 1
    neg = y_true == 0
    tpr = (y_pred[pos] == 1).mean() if pos.any() else 0.0
    tnr = (y_pred[neg] == 0).mean() if neg.any() else 0.0
    gmean2 = tpr * tnr
    return float(gmean2), int(len(y_true)), int(len(test_dirs)), run.name


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
        edgecolor="none", zorder=zorder, transform=ax.transData))


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    results = load_results()

    tasks = list(CSV_TO_RUN_DIR.keys())
    monitor_vals = []
    agent_vals = []
    best_vals = []
    monitor_names = []
    best_names = []

    print(f"{'task':30s} {'best_monitor':35s}  {'agent':>6s}  {'best_all':35s}  (n_scored/n_total)")
    for task in tasks:
        methods = results[task]
        mon_name, mon_val = best_id_gmean2(methods, llm_only=True)
        best_name, best_val = best_overall_id(methods)
        run_dir = CSV_TO_RUN_DIR[task]
        data_dir = REPO / "data" / run_dir / "test"
        ag_val, n_scored, n_total, run_name = compute_agent_gmean2(run_dir, data_dir)

        monitor_vals.append(mon_val)
        agent_vals.append(ag_val)
        best_vals.append(best_val)
        monitor_names.append(mon_name)
        best_names.append(best_name)

        print(f"{task:30s} {mon_name:35s}  {mon_val:.3f} "
              f" {ag_val:.3f}  {best_name:35s}  ({n_scored}/{n_total}, {run_name})")

    # ── Figure ──────────────────────────────────────────────────────
    N = len(tasks)
    bar_pitch_in = 1.4
    fig, ax = plt.subplots(figsize=(N * bar_pitch_in + 1.6, 5.2))

    group_spacing = 1.0
    x = np.arange(N) * group_spacing
    bar_w = 0.24
    gap = 0.03
    offsets = np.array([-1, 0, 1]) * (bar_w + gap)

    ylim = (0, max(0.9, max(monitor_vals + agent_vals + best_vals) * 1.15))
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    ax.set_ylim(*ylim)
    fig.canvas.draw()

    # Corner radii in data coords
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((1, 1))
    sx, sy = p1[0] - p0[0], p1[1] - p0[1]
    dpi = fig.dpi
    r_px = CORNER_R_PT * dpi / 72.0
    r_x, r_y = r_px / sx, r_px / sy

    for i in range(N):
        cx_mon   = x[i] + offsets[0]
        cx_agent = x[i] + offsets[1]
        cx_best  = x[i] + offsets[2]

        draw_rounded_bar(ax, cx_mon,   0, monitor_vals[i], ORANGE_DARK, bar_w, r_x, r_y)
        draw_rounded_bar(ax, cx_agent, 0, agent_vals[i],   GREEN_DARK,  bar_w, r_x, r_y)
        draw_rounded_bar(ax, cx_best,  0, best_vals[i],    BLUE_DARK,   bar_w, r_x, r_y)

        # Value labels above each bar
        for cx, v in [(cx_mon, monitor_vals[i]),
                      (cx_agent, agent_vals[i]),
                      (cx_best, best_vals[i])]:
            ax.text(cx, v + 0.012, f"{v:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color=TEXT_SEC)

    # Chance baseline for binary gmean²
    ax.axhline(y=0.25, color=BASELINE_C, linewidth=1.2, linestyle="--", zorder=1)
    ax.annotate("chance", xy=(x[0] - 0.48, 0.25), xytext=(0, 4),
                textcoords="offset points", fontsize=7, color=TEXT_SEC,
                ha="left", va="bottom", style="italic")

    # Legend
    h1 = mpatches.Patch(facecolor=ORANGE_DARK, edgecolor="none", label="Best LLM monitor (ID)")
    h2 = mpatches.Patch(facecolor=GREEN_DARK,  edgecolor="none", label="Agent (ID)")
    h3 = mpatches.Patch(facecolor=BLUE_DARK,   edgecolor="none", label="Best overall method (ID)")
    ax.legend(handles=[h1, h2, h3], loc="upper right", fontsize=9,
              frameon=True, facecolor=BG, edgecolor=GRID, framealpha=1,
              handlelength=1.2, handleheight=0.9)

    # Axes cosmetics
    ax.set_facecolor(BG)
    fig.set_facecolor(BG)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=9.5, color=TEXT)

    y_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{v:g}" for v in y_ticks], fontsize=9.5, color=TEXT_SEC)

    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_ylabel("Test g-mean\u00b2 (in-distribution)", fontsize=11, color=TEXT, labelpad=8)
    ax.set_title("Agent vs. best LLM monitor vs. best overall method (ID)",
                 fontsize=13, color=TEXT, pad=10)

    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", colors=TEXT_SEC, length=3)
    ax.tick_params(axis="x", colors=TEXT, length=3)

    fig.tight_layout()
    out = PLOTS_DIR / "agent_vs_monitors_id.png"
    plt.savefig(out, dpi=300, bbox_inches="tight", facecolor=BG)
    print(f"\nSaved {out}")
    plt.close()


if __name__ == "__main__":
    main()
