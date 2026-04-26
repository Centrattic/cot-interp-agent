"""Plot OOD g-mean² for high-effort agent vs best monitor vs best non-monitor baseline."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np


BLUE_DARK = "#4056CA"
ORANGE_DARK = "#E87E24"
GREEN_DARK = "#2E8B57"
BG = "#FAFAF8"
GRID = "#E4E2DD"
TEXT = "#2D2D2D"
TEXT_SEC = "#8A8A8A"
BASELINE_C = "#C8C4BC"

CORNER_R_PT = 5

REPO = Path(__file__).resolve().parent
RESULTS_CSV = REPO.parent / "cot-proxy-tasks" / "results_summary.csv"
PLOTS_DIR = REPO / "plots"
OUT_CSV = PLOTS_DIR / "ood_agent_vs_monitor_nonmonitor_high.csv"
OUT_PNG = PLOTS_DIR / "ood_agent_vs_monitor_nonmonitor_high.png"

TASKS = [
    ("1_reasoning_termination", "reasoning_termination_ood", "Reasoning\ntermination"),
    ("2_self_deletion", "gemma_self_deletion_ood", "Self\ndeletion"),
    ("3_follow_up_response", "followup_confidence_ood", "Follow-up\nresponse"),
    ("5_stanford_hint", "stanford_hint_ood", "Stanford\nhint"),
    ("6_atypical_answer", "atypical_answer_ood", "Atypical\nanswer"),
    ("7_atypical_cot_length", "atypical_cot_length_ood", "Atypical\nCoT length"),
]

PINNED_HIGH_RUNS = {
    "reasoning_termination_ood": "run-20260424-162647-288149",
    "gemma_self_deletion_ood": "run-20260424-162647-292785",
    "followup_confidence_ood": "run-20260424-163302-968137",
    "stanford_hint_ood": "run-20260424-163546-537946",
    "atypical_answer_ood": "run-20260424-164336-413508",
    "atypical_cot_length_ood": "run-20260424-164555-541763",
}

LLM_METHODS = {
    "Few-shot LLMs",
    "Zero-shot LLMs",
    "Few-shot LLMs (binary)",
    "Few-shot LLMs (conf)",
    "Zero-shot LLMs (binary)",
    "Zero-shot LLMs (conf)",
}


def load_results() -> dict[str, dict[str, dict[str, tuple[float, float]]]]:
    rows: dict[str, dict[str, dict[str, tuple[float, float]]]] = {}
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


def best_ood_gmean2(
    methods: dict[str, dict[str, tuple[float, float]]], *, llm_only: bool
) -> tuple[str, float]:
    best_name, best_val = None, -1.0
    for method, metrics in methods.items():
        if "gmean2" not in metrics:
            continue
        is_llm = method in LLM_METHODS
        if llm_only != is_llm:
            continue
        ood_val = metrics["gmean2"][1]
        if not np.isnan(ood_val) and ood_val > best_val:
            best_name, best_val = method, ood_val
    if best_name is None:
        return "(none)", float("nan")
    return best_name, best_val


def pinned_run(task: str) -> Path:
    return REPO / "agent-runs" / task / PINNED_HIGH_RUNS[task]


def score_agent_run(run: Path) -> tuple[float, int, int]:
    results_csv = run / "results.csv"
    y_true, y_pred = [], []
    with open(results_csv, newline="") as f:
        for row in csv.DictReader(f):
            try:
                label = int(row["label"])
                pred = int(row["pred"])
            except (TypeError, ValueError, KeyError):
                continue
            y_true.append(label)
            y_pred.append(pred)

    yt = np.array(y_true)
    yp = np.array(y_pred)
    pos = yt == 1
    neg = yt == 0
    tpr = float((yp[pos] == 1).mean()) if pos.any() else 0.0
    tnr = float((yp[neg] == 0).mean()) if neg.any() else 0.0
    total = sum(1 for _ in (REPO / "data" / run.parent.name / "test").glob("*.json"))
    return tpr * tnr, int(len(yt)), total


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
    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "csv_task",
                "run_task",
                "label",
                "run",
                "agent_gmean2",
                "agent_n_scored",
                "agent_n_total",
                "monitor_name",
                "monitor_gmean2",
                "nonmonitor_name",
                "nonmonitor_gmean2",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    results = load_results()

    rows = []
    print(f"{'task':30s} {'agent_high':>10s}  {'best monitor (OOD)':28s}  {'best non-monitor (OOD)':28s}")
    for csv_task, run_task, label in TASKS:
        methods = results[csv_task]
        monitor_name, monitor_val = best_ood_gmean2(methods, llm_only=True)
        nonmonitor_name, nonmonitor_val = best_ood_gmean2(methods, llm_only=False)
        run = pinned_run(run_task)
        agent_val, n_scored, n_total = score_agent_run(run)
        row = {
            "csv_task": csv_task,
            "run_task": run_task,
            "label": label,
            "run": run.name,
            "agent_gmean2": agent_val,
            "agent_n_scored": n_scored,
            "agent_n_total": n_total,
            "monitor_name": monitor_name,
            "monitor_gmean2": monitor_val,
            "nonmonitor_name": nonmonitor_name,
            "nonmonitor_gmean2": nonmonitor_val,
        }
        rows.append(row)
        print(
            f"{csv_task:30s} {agent_val:10.3f}  "
            f"{monitor_name[:28]:28s} {monitor_val:6.3f}  "
            f"{nonmonitor_name[:28]:28s} {nonmonitor_val:6.3f}"
        )

    write_csv(rows)

    n = len(rows)
    fig, ax = plt.subplots(figsize=(n * 1.45 + 1.6, 5.2))
    x = np.arange(n)
    bar_w = 0.24
    gap = 0.03
    offsets = np.array([-1, 0, 1]) * (bar_w + gap)

    ymax = max(
        max(row["agent_gmean2"], row["monitor_gmean2"], row["nonmonitor_gmean2"]) for row in rows
    )
    ax.set_xlim(x[0] - 0.55, x[-1] + 0.55)
    ax.set_ylim(0, max(0.9, ymax * 1.15))
    fig.canvas.draw()

    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((1, 1))
    sx, sy = p1[0] - p0[0], p1[1] - p0[1]
    r_px = CORNER_R_PT * fig.dpi / 72.0
    r_x, r_y = r_px / sx, r_px / sy

    for i, row in enumerate(rows):
        specs = [
            (x[i] + offsets[0], row["monitor_gmean2"], ORANGE_DARK),
            (x[i] + offsets[1], row["agent_gmean2"], GREEN_DARK),
            (x[i] + offsets[2], row["nonmonitor_gmean2"], BLUE_DARK),
        ]
        for cx, val, color in specs:
            draw_rounded_bar(ax, cx, 0, val, color, bar_w, r_x, r_y)
            ax.text(cx, val + 0.012, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5, color=TEXT_SEC)

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
        mpatches.Patch(facecolor=ORANGE_DARK, edgecolor="none", label="Best monitor (OOD)"),
        mpatches.Patch(facecolor=GREEN_DARK, edgecolor="none", label="Agent (OOD, GPT-5.4 high)"),
        mpatches.Patch(facecolor=BLUE_DARK, edgecolor="none", label="Best non-monitor (OOD)"),
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
    ax.set_xticklabels(
        [f"{row['label']}\nn={row['agent_n_scored']}/{row['agent_n_total']}" for row in rows],
        fontsize=9.5,
        color=TEXT,
    )
    y_ticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{v:g}" for v in y_ticks], fontsize=9.5, color=TEXT_SEC)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
    ax.set_ylabel("OOD g-mean\u00b2", fontsize=11, color=TEXT, labelpad=8)
    ax.set_title("OOD agent vs. best monitor vs. best non-monitor (GPT-5.4 high)", fontsize=13, color=TEXT, pad=10)

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
