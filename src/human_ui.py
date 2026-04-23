#!/usr/bin/env python3
"""Local web UI for running human baselines on scaffold tasks."""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, urlencode, urlparse

from ingest_cot_proxy import load_with_normalized_label, sample_test
from run_tests import collect_test_examples
from scaffold import ROOT, RUNS_DIR, TOOL_DESCRIPTIONS, _setup_partition, load_task_metadata


TOOL_SCRIPTS = {
    "ask": ROOT / "src" / "tools" / "ask.py",
    "top_10_logits": ROOT / "src" / "tools" / "top_10_logits.py",
    "top10_entropy": ROOT / "src" / "tools" / "top10_entropy.py",
    "force": ROOT / "src" / "tools" / "force.py",
}

RUN_MODE = "human-ui"
SESSION_FILE = "human_session.json"
SUBMISSIONS_DIRNAME = "submissions"
TOOL_LOG_FILE = "tool_invocations.jsonl"

OOD_SPLIT_OVERRIDES = {
    "1": {"few_shot_split": "ood_val", "test_split": "test"},
    "2": {"few_shot_split": "ood_val", "test_split": "test"},
    "3": {"few_shot_split": "val", "test_split": "ood_test"},
    "5": {"few_shot_split": "val", "test_split": "ood_test"},
    "6": {"few_shot_split": "val", "test_split": "ood_test"},
    "7": {"few_shot_split": "val", "test_split": "ood_test"},
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def now_ts() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S")


def read_run_meta(run_dir: Path) -> dict[str, Any]:
    return load_json(run_dir / "run.json")


def write_run_meta(run_dir: Path, meta: dict[str, Any]) -> None:
    write_json(run_dir / "run.json", meta)


def read_session(run_dir: Path) -> dict[str, Any]:
    path = run_dir / SESSION_FILE
    if path.exists():
        return load_json(path)
    return {
        "phase": "strategy",
        "created": datetime.now(UTC).isoformat(),
        "test_started": False,
        "last_submission": None,
        "submissions": [],
    }


def write_session(run_dir: Path, session: dict[str, Any]) -> None:
    write_json(run_dir / SESSION_FILE, session)


def apply_ood_overrides(task_meta: dict[str, Any], enabled: bool) -> dict[str, Any]:
    meta = dict(task_meta)
    meta["ood"] = bool(enabled)
    if not enabled:
        return meta
    dataset_id = str(meta.get("dataset_id", ""))
    override = OOD_SPLIT_OVERRIDES.get(dataset_id)
    if override:
        meta.update(override)
    return meta


def create_human_run(
    task_name: str,
    tools: list[str],
    description: str | None = None,
    *,
    ood: bool,
) -> Path:
    task_meta = load_task_metadata(task_name)
    if description:
        task_meta["description"] = description
    task_meta = apply_ood_overrides(task_meta, ood)

    run_id = now_ts()
    run_dir = RUNS_DIR / task_name / f"run-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "task": task_name,
        "run_id": run_id,
        "created": datetime.now(UTC).isoformat(),
        "finished": None,
        "status": "interactive",
        "mode": RUN_MODE,
        "agent_backend": "human",
        "ood": bool(task_meta.get("ood")),
        "tools": tools,
        "n_strategies": 1,
        "task_meta": task_meta,
    }
    write_run_meta(run_dir, run_meta)

    _setup_partition(
        run_dir=run_dir,
        task_name=task_name,
        task_meta=task_meta,
        tools=tools,
        partition_idx=0,
        n_partitions=1,
        strategy_dir=run_dir / "strategy",
        bashrc_path=run_dir / "agent.bashrc",
        agent_backend="human",
        seed=None,
        few_shot_per_class=int(task_meta.get("few_shot_per_class", 5)),
        from_source=bool(task_meta.get("ood")),
    )

    strategy_path = run_dir / "strategy" / "STRATEGY.md"
    strategy_path.write_text(
        "# Human Strategy Notes\n\n"
        "<!-- Draft your strategy here before starting test mode. -->\n",
        encoding="utf-8",
    )

    write_session(run_dir, read_session(run_dir))
    return run_dir


def load_examples_csv(strategy_dir: Path) -> list[dict[str, str]]:
    path = strategy_dir / "Examples.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def materialize_test_examples(run_dir: Path) -> int:
    meta = read_run_meta(run_dir)
    task_name = meta["task"]
    task_meta = meta["task_meta"]
    test_keep_fields = task_meta.get("test_keep_fields")
    examples = load_test_examples(task_name, task_meta)

    for idx, example in enumerate(examples):
        test_dir = run_dir / f"test-{idx:03d}"
        test_dir.mkdir(parents=True, exist_ok=True)

        src_data = example["data"]
        if test_keep_fields:
            redacted = {k: src_data[k] for k in test_keep_fields if k in src_data}
        else:
            redacted = {k: v for k, v in src_data.items() if k != "label"}

        with open(test_dir / "example.json", "w", encoding="utf-8") as f:
            json.dump(redacted, f, indent=2, ensure_ascii=False)

        if example["npy_path"]:
            shutil.copy2(example["npy_path"], test_dir / "example.npy")

        write_json(
            test_dir / "test_meta.json",
            {
                "index": idx,
                "example_id": example["id"],
                "task": task_name,
            },
        )

    session = read_session(run_dir)
    session["phase"] = "test"
    session["test_started"] = True
    write_session(run_dir, session)
    return len(examples)


def load_test_examples(task_name: str, task_meta: dict[str, Any]) -> list[dict[str, Any]]:
    if task_meta.get("ood"):
        source_dir = Path(task_meta["source"]) / task_meta.get("test_split", "test")
        label_map = task_meta.get("label_map", {})
        items = sample_test(
            source_dir,
            task_meta.get("test_n"),
            int(task_meta.get("seed", 0)),
            label_map,
        )
        examples = []
        for src_path, data in items:
            examples.append(
                {
                    "id": src_path.stem,
                    "json_path": src_path,
                    "npy_path": src_path.with_suffix(".npy") if src_path.with_suffix(".npy").exists() else None,
                    "data": data,
                }
            )
        return examples

    data_test_dir = ROOT / "data" / task_name / "test"
    return collect_test_examples(data_test_dir)


def test_entries(run_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for test_dir in sorted(run_dir.glob("test-*")):
        if not test_dir.is_dir():
            continue
        meta_path = test_dir / "test_meta.json"
        if not meta_path.exists():
            continue
        info = load_json(meta_path)
        answer_path = test_dir / "answer.txt"
        answer = answer_path.read_text(encoding="utf-8").strip().lower() if answer_path.exists() else ""
        entries.append(
            {
                "index": int(info["index"]),
                "example_id": info["example_id"],
                "dir": test_dir,
                "answer": answer if answer in ("yes", "no") else "",
            }
        )
    entries.sort(key=lambda item: item["index"])
    return entries


def current_answer(test_dir: Path) -> str:
    answer_path = test_dir / "answer.txt"
    if not answer_path.exists():
        return ""
    answer = answer_path.read_text(encoding="utf-8").strip().lower()
    return answer if answer in ("yes", "no") else ""


def set_answer(test_dir: Path, answer: str) -> None:
    cleaned = answer.strip().lower()
    if cleaned not in ("yes", "no", ""):
        raise ValueError("answer must be yes, no, or empty")
    answer_path = test_dir / "answer.txt"
    if cleaned:
        answer_path.write_text(cleaned + "\n", encoding="utf-8")
    elif answer_path.exists():
        answer_path.unlink()


def score_human_run(run_dir: Path) -> dict[str, Any]:
    meta = read_run_meta(run_dir)
    task_name = meta["task"]
    ground_truth = []
    for example in load_test_examples(task_name, meta["task_meta"]):
        ground_truth.append((example["id"], int(example["data"]["label"])))

    rows = []
    tp = tn = fp = fn = miss = 0
    for idx, (example_id, label) in enumerate(ground_truth):
        answer = current_answer(run_dir / f"test-{idx:03d}")
        pred = 1 if answer == "yes" else 0 if answer == "no" else None
        correct = pred == label if pred is not None else None
        if pred is None:
            miss += 1
        elif pred == 1 and label == 1:
            tp += 1
        elif pred == 0 and label == 0:
            tn += 1
        elif pred == 1 and label == 0:
            fp += 1
        else:
            fn += 1
        rows.append(
            {
                "index": idx,
                "example_id": example_id,
                "label": label,
                "answer": answer,
                "pred": "" if pred is None else pred,
                "correct": "" if correct is None else int(correct),
            }
        )

    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    gmean2 = tpr * tnr

    results_path = run_dir / "results.csv"
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["index", "example_id", "label", "answer", "pred", "correct"],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = "\n".join(
        [
            f"# Human baseline summary — {task_name}  (run {meta['run_id']})",
            f"n={n}  miss={miss}  total_test_examples={len(ground_truth)}",
            f"acc={acc*100:.1f}%  TPR={tpr:.2f}  TNR={tnr:.2f}  gmean²={gmean2:.3f}",
            f"TP/TN/FP/FN={tp}/{tn}/{fp}/{fn}",
            "",
        ]
    )
    summary_path = run_dir / "summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    submission_id = f"submission-{now_ts()}"
    submission_dir = run_dir / SUBMISSIONS_DIRNAME / submission_id
    submission_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(results_path, submission_dir / "results.csv")
    shutil.copy2(summary_path, submission_dir / "summary.txt")
    shutil.copy2(run_dir / "strategy" / "STRATEGY.md", submission_dir / "STRATEGY.md")

    session = read_session(run_dir)
    session["phase"] = "submitted"
    session["last_submission"] = submission_id
    session.setdefault("submissions", []).append(submission_id)
    write_session(run_dir, session)

    meta["status"] = "submitted"
    meta["finished"] = datetime.now(UTC).isoformat()
    write_run_meta(run_dir, meta)

    return {
        "submission_id": submission_id,
        "answered": n,
        "miss": miss,
        "total": len(ground_truth),
        "acc": acc,
    }


def append_tool_log(run_dir: Path, record: dict[str, Any]) -> None:
    with open(run_dir / TOOL_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_tool_logs(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / TOOL_LOG_FILE
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def run_tool(
    run_dir: Path,
    *,
    scope: str,
    example_id: str,
    tool_name: str,
    token_position: str = "",
    question: str = "",
    forced_text: str = "",
) -> dict[str, Any]:
    if tool_name not in TOOL_SCRIPTS:
        raise ValueError(f"unknown tool {tool_name!r}")

    enabled = set(read_run_meta(run_dir).get("tools", []))
    if tool_name not in enabled:
        raise ValueError(f"tool {tool_name!r} is not enabled for this run")

    if scope == "strategy":
        cwd = run_dir / "strategy"
        agent_type = "strategy"
        extra_env = {}
    elif scope.startswith("test-"):
        cwd = run_dir / scope
        agent_type = "test"
        extra_env = {"AGENT_EXAMPLE_ID": example_id}
    else:
        raise ValueError(f"invalid scope {scope!r}")

    if tool_name == "ask":
        argv = [example_id, question]
    elif tool_name in ("top_10_logits", "top10_entropy"):
        argv = [example_id, token_position]
    elif tool_name == "force":
        argv = [example_id, token_position, forced_text]
    else:
        raise ValueError(f"tool {tool_name!r} not implemented")

    before = {path.name for path in cwd.iterdir()}
    env = os.environ.copy()
    env.update(
        {
            "SCAFFOLD_ROOT": str(ROOT),
            "AGENT_TASK": read_run_meta(run_dir)["task"],
            "AGENT_TYPE": agent_type,
            **extra_env,
        }
    )
    cmd = [sys.executable, str(TOOL_SCRIPTS[tool_name]), *argv]
    result = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    after = {path.name for path in cwd.iterdir()}
    created = sorted(after - before)

    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "scope": scope,
        "example_id": example_id,
        "tool": tool_name,
        "argv": argv,
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "files_created": created,
    }
    append_tool_log(run_dir, record)
    return record


def escape_json(payload: Any) -> str:
    return html.escape(json.dumps(payload, indent=2, ensure_ascii=False))


def build_query(path: str, **params: Any) -> str:
    encoded = {k: str(v) for k, v in params.items() if v is not None and v != ""}
    return path if not encoded else f"{path}?{urlencode(encoded)}"


def relative_artifact_links(run_dir: Path, cwd: Path, files: list[str]) -> str:
    if not files:
        return "<span class='muted'>none</span>"
    links = []
    for name in files:
        rel = (cwd / name).relative_to(run_dir)
        links.append(
            f"<a href='/artifact?path={quote(str(rel))}'>{html.escape(name)}</a>"
        )
    return ", ".join(links)


def render_tool_history(run_dir: Path, scope: str, example_id: str) -> str:
    logs = [
        log for log in reversed(read_tool_logs(run_dir))
        if log["scope"] == scope and log["example_id"] == example_id
    ][:10]
    if not logs:
        return "<p class='muted'>No tool calls yet.</p>"

    blocks = []
    for log in logs:
        cwd = run_dir / log["scope"] if log["scope"].startswith("test-") else run_dir / "strategy"
        blocks.append(
            "\n".join(
                [
                    "<div class='tool-log'>",
                    f"<div><strong>{html.escape(log['tool'])}</strong> "
                    f"<span class='muted'>{html.escape(log['timestamp'])}</span></div>",
                    f"<div>exit code: {log['exit_code']}</div>",
                    f"<div>files: {relative_artifact_links(run_dir, cwd, log.get('files_created', []))}</div>",
                    f"<pre>{html.escape(log['stdout'])}</pre>",
                    "</div>",
                ]
            )
        )
    return "\n".join(blocks)


def render_submission_history(run_dir: Path) -> str:
    subdir = run_dir / SUBMISSIONS_DIRNAME
    if not subdir.exists():
        return "<p class='muted'>No submissions yet.</p>"

    items = []
    for path in sorted(subdir.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        items.append(
            "<li>"
            f"{html.escape(path.name)} "
            f"(<a href='/artifact?path={quote(str(path.relative_to(run_dir) / 'summary.txt'))}'>summary</a>, "
            f"<a href='/artifact?path={quote(str(path.relative_to(run_dir) / 'results.csv'))}'>results</a>)"
            "</li>"
        )
    return "<ul>" + "".join(items) + "</ul>" if items else "<p class='muted'>No submissions yet.</p>"


def page_css() -> str:
    return """
body { font-family: "IBM Plex Sans", "Avenir Next", ui-sans-serif, system-ui, sans-serif; margin: 0; background:
linear-gradient(180deg, #efe4d2 0%, #f7f3eb 28%, #f2efe9 100%); color: #1f1d1a; }
a { color: #0c5a6b; text-decoration: none; }
a:hover { text-decoration: underline; }
.layout { display: grid; grid-template-columns: 300px minmax(0, 1fr); min-height: 100vh; }
.sidebar { background:
radial-gradient(circle at top left, rgba(246, 215, 158, 0.18), transparent 36%),
linear-gradient(180deg, #172126 0%, #1f2a2e 100%); color: #f8f3ea; padding: 20px; overflow: auto; border-right: 1px solid rgba(255,255,255,0.08); }
.sidebar a { color: #f6d79e; display: block; padding: 6px 8px; border-radius: 8px; }
.sidebar a:hover { background: rgba(255,255,255,0.08); text-decoration: none; }
.content { padding: 20px; overflow: hidden; }
.shell { display: grid; grid-template-rows: auto auto minmax(0, 1fr); gap: 16px; height: calc(100vh - 40px); }
.panel { background: rgba(255,255,255,0.84); backdrop-filter: blur(8px); border: 1px solid #d9d1c2; border-radius: 16px; padding: 16px; margin-bottom: 0; box-shadow: 0 14px 34px rgba(31, 29, 26, 0.08); }
.top-grid { display: grid; grid-template-columns: 1.15fr 1fr; gap: 16px; min-height: 0; }
.bottom-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; min-height: 0; }
.muted { color: #6d665b; }
.badge { display: inline-block; padding: 4px 10px; border-radius: 999px; background: #e9dfce; color: #5f4520; font-size: 12px; margin-right: 6px; }
.ok { color: #0e6b39; }
.warn { color: #8a4a00; }
.danger { color: #8f1d2c; }
textarea, input[type=text], input[type=number], select { width: 100%; box-sizing: border-box; padding: 10px; border-radius: 8px; border: 1px solid #c9c1b5; background: #fffdfa; }
textarea { min-height: 180px; font-family: ui-monospace, SFMono-Regular, monospace; }
button { background: linear-gradient(180deg, #0e7085 0%, #0c5a6b 100%); color: white; border: 0; border-radius: 10px; padding: 10px 14px; cursor: pointer; box-shadow: inset 0 1px 0 rgba(255,255,255,0.2); }
button.secondary { background: #7c6f5f; }
button.warn { background: linear-gradient(180deg, #b76b16 0%, #9b5c12 100%); }
pre { white-space: pre-wrap; word-break: break-word; background: #f8f5ef; padding: 12px; border-radius: 12px; border: 1px solid #e3dccf; overflow: auto; }
code { font-family: ui-monospace, SFMono-Regular, monospace; }
.tool-log { border-top: 1px solid #e7e0d4; padding-top: 12px; margin-top: 12px; }
.nav-group { margin-bottom: 20px; }
.example-active { font-weight: 700; background: rgba(255,255,255,0.12); }
.flash { background: #fff1c7; border: 1px solid #ebd37b; border-radius: 10px; padding: 12px; margin-bottom: 16px; }
.scroll { min-height: 0; overflow: auto; }
.example-pre { max-height: 100%; min-height: 320px; }
.sidebar details { background: rgba(255,255,255,0.05); border-radius: 12px; padding: 8px 10px; margin-bottom: 14px; }
.sidebar summary { cursor: pointer; color: #f8f3ea; font-weight: 700; }
.link-list { margin-top: 10px; max-height: 260px; overflow: auto; }
.header-panel { background:
linear-gradient(135deg, rgba(255,255,255,0.86), rgba(252,247,240,0.92)),
radial-gradient(circle at top right, rgba(12, 90, 107, 0.08), transparent 40%); }
.panel h3, .panel h2, .panel h1 { margin-top: 0; }
.strategy-panel { display: grid; grid-template-rows: auto 1fr auto; min-height: 0; }
.strategy-panel textarea { min-height: 0; height: 100%; }
.tool-panel, .history-panel, .example-panel { min-height: 0; display: grid; grid-template-rows: auto 1fr; }
.compact-form { display: grid; gap: 10px; }
.toolbar { margin-top: 12px; display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
@media (max-width: 1100px) {
  .layout { grid-template-columns: 1fr; }
  .sidebar { max-height: 40vh; }
  .shell { height: auto; }
  .top-grid, .bottom-grid { grid-template-columns: 1fr; }
  .example-pre { min-height: 240px; max-height: 50vh; }
}
"""


def render_main_page(run_dir: Path, query: dict[str, list[str]]) -> str:
    run_meta = read_run_meta(run_dir)
    session = read_session(run_dir)
    strategy_dir = run_dir / "strategy"
    few_shot = load_examples_csv(strategy_dir)
    tests = test_entries(run_dir)

    scope = query.get("scope", ["strategy"])[0]
    example_id = query.get("example_id", [""])[0]
    test_idx = query.get("test_idx", ["0"])[0]
    flash = query.get("msg", [""])[0]

    if scope == "strategy":
        active = next((row for row in few_shot if row["id"] == example_id), few_shot[0] if few_shot else None)
        active_example = load_json(strategy_dir / active["path"]) if active else {}
        active_scope = "strategy"
        active_example_id = active["id"] if active else ""
        active_title = f"Few-shot: {active_example_id}"
        answer_box = ""
    else:
        idx = int(test_idx) if test_idx.isdigit() else 0
        active = next((row for row in tests if row["index"] == idx), tests[0] if tests else None)
        active_scope = f"test-{active['index']:03d}" if active else "test-000"
        active_example_id = active["example_id"] if active else ""
        active_example = load_json(active["dir"] / "example.json") if active else {}
        active_title = f"Test {active['index']:03d}: {active_example_id}" if active else "No test example"
        selected = active["answer"] if active else ""
        answer_box = f"""
        <div class="panel">
          <h3>Classification</h3>
          <form method="post" action="/classify">
            <input type="hidden" name="scope" value="{html.escape(active_scope)}">
            <input type="hidden" name="test_idx" value="{active['index'] if active else 0}">
            <label><input type="radio" name="answer" value="yes" {"checked" if selected == "yes" else ""}> yes</label>
            <label><input type="radio" name="answer" value="no" {"checked" if selected == "no" else ""}> no</label>
            <label><input type="radio" name="answer" value="" {"checked" if selected == "" else ""}> clear</label>
            <div style="margin-top: 12px;"><button type="submit">Save Answer</button></div>
          </form>
        </div>
        """

    enabled_tools = run_meta.get("tools", [])
    strategy_text = (strategy_dir / "STRATEGY.md").read_text(encoding="utf-8")
    answered = sum(1 for item in tests if item["answer"])

    sidebar_few_shot = "".join(
        f'<a class="{ "example-active" if scope == "strategy" and row["id"] == active_example_id else "" }" '
        f'href="{html.escape(build_query("/", scope="strategy", example_id=row["id"]))}">{html.escape(row["id"])} '
        f'<span class="muted">(label {html.escape(row["label"])})</span></a>'
        for row in few_shot
    ) or "<div class='muted'>No few-shot examples.</div>"

    sidebar_tests = "".join(
        f'<a class="{ "example-active" if scope != "strategy" and item["example_id"] == active_example_id else "" }" '
        f'href="{html.escape(build_query("/", scope="test", test_idx=item["index"]))}">{item["index"]:03d} · {html.escape(item["example_id"])} '
        f'<span class="{ "ok" if item["answer"] else "muted" }">[{html.escape(item["answer"] or "pending")}]</span></a>'
        for item in tests
    ) or "<div class='muted'>Start test mode to materialize held-out examples.</div>"

    tool_options = "".join(
        f"<option value='{html.escape(name)}'>{html.escape(name)}</option>" for name in enabled_tools
    ) or "<option value=''>No tools enabled</option>"

    tool_docs = "".join(
        f"<details><summary><code>{html.escape(name)}</code></summary><pre>{html.escape(TOOL_DESCRIPTIONS.get(name, ''))}</pre></details>"
        for name in enabled_tools
    ) or "<p class='muted'>No custom tools enabled for this run.</p>"

    start_test_button = (
        "<button class='warn' type='submit' form='start-test-form'>Start Test</button>"
        if not session.get("test_started")
        else "<span class='badge'>test mode active</span>"
    )

    submit_button = (
        "<button type='submit' form='submit-form'>Submit Run</button>"
        if session.get("test_started")
        else "<span class='muted'>Submit is available after test mode starts.</span>"
    )

    flash_box = f"<div class='flash'>{html.escape(flash)}</div>" if flash else ""

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Human Baseline UI</title>
  <style>{page_css()}</style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h2>{html.escape(run_meta['task'])}</h2>
      <div class="muted">run-{html.escape(run_meta['run_id'])}</div>
      <div style="margin: 10px 0 18px 0;">
        <span class="badge">{html.escape(session.get('phase', 'strategy'))}</span>
        <span class="badge">ood={str(bool(run_meta.get('ood', False))).lower()}</span>
      </div>
      <details open>
        <summary>Few-shot ({len(few_shot)})</summary>
        <div class="link-list">{sidebar_few_shot}</div>
      </details>
      <details open>
        <summary>Test ({answered}/{len(tests)} answered)</summary>
        <div class="link-list">{sidebar_tests}</div>
      </details>
      <div class="nav-group">
        <strong>Artifacts</strong>
        <a href="/artifact?path=run.json">run.json</a>
        <a href="/artifact?path=human_session.json">human_session.json</a>
        <a href="/artifact?path=strategy/README.md">strategy/README.md</a>
        <a href="/artifact?path=strategy/STRATEGY.md">strategy/STRATEGY.md</a>
        <a href="/artifact?path=results.csv">results.csv</a>
        <a href="/artifact?path=summary.txt">summary.txt</a>
      </div>
    </aside>
    <main class="content">
      <div class="shell">
      {flash_box}
      <div class="panel header-panel">
        <h1>{html.escape(run_meta['task_meta']['name'])}</h1>
        <p>{html.escape(run_meta['task_meta']['description'])}</p>
        <div>
          <span class="badge">tools: {html.escape(", ".join(enabled_tools) or "none")}</span>
          <span class="badge">answered: {answered}/{len(tests)}</span>
          <span class="badge">few-shot split: {html.escape(run_meta['task_meta'].get('few_shot_split', ''))}</span>
          <span class="badge">test split: {html.escape(run_meta['task_meta'].get('test_split', ''))}</span>
        </div>
      </div>

      <div class="top-grid">
        <div class="panel strategy-panel">
          <h3>Strategy</h3>
          <form id="start-test-form" method="post" action="/start-test"></form>
          <form id="submit-form" method="post" action="/submit"></form>
          <form method="post" action="/save-strategy" class="scroll">
            <textarea name="strategy_text">{html.escape(strategy_text)}</textarea>
            <div class="toolbar">
              <button type="submit">Save STRATEGY.md</button>
              {start_test_button}
              {submit_button}
            </div>
          </form>
        </div>
        <div class="panel example-panel">
          <h3>Current Example</h3>
          <div class="muted">{html.escape(active_title)}</div>
          <pre class="example-pre scroll">{escape_json(active_example)}</pre>
        </div>
      </div>

      {answer_box}

      <div class="bottom-grid">
        <div class="panel tool-panel">
          <h3>Run Tool</h3>
          <form method="post" action="/run-tool" class="compact-form scroll">
            <input type="hidden" name="scope" value="{html.escape(active_scope)}">
            <input type="hidden" name="example_id" value="{html.escape(active_example_id)}">
            <input type="hidden" name="test_idx" value="{html.escape(test_idx)}">
            <label>Tool</label>
            <select name="tool_name">{tool_options}</select>
            <label style="margin-top: 10px; display: block;">Token Position</label>
            <input type="number" name="token_position" value="0">
            <label style="margin-top: 10px; display: block;">Question (for <code>ask</code>)</label>
            <input type="text" name="question" placeholder="yes or no only">
            <label style="margin-top: 10px; display: block;">Forced Text (for <code>force</code>)</label>
            <input type="text" name="forced_text" placeholder="</think>">
            <div style="margin-top: 12px;"><button type="submit">Run Tool</button></div>
            <div style="margin-top: 16px;">{tool_docs}</div>
          </form>
        </div>
        <div class="panel history-panel">
          <h3>Tool History</h3>
          <div class="scroll">{render_tool_history(run_dir, active_scope, active_example_id)}</div>
        </div>
      </div>

      <div class="bottom-grid">
        <div class="panel">
          <h3>Submission History</h3>
          {render_submission_history(run_dir)}
        </div>
        <div class="panel">
          <h3>Run Structure</h3>
          <p><code>{html.escape(str(run_dir))}</code></p>
          <p class="muted">Few-shot examples live in <code>strategy/few-shot/</code>. Tool outputs are saved in the current strategy/test folder. Each submit snapshots results under <code>submissions/</code>.</p>
        </div>
      </div>
      </div>
    </main>
  </div>
</body>
</html>"""


class HumanUIHandler(BaseHTTPRequestHandler):
    run_dir: Path

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        if parsed.path == "/":
            body = render_main_page(self.run_dir, query).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if parsed.path == "/artifact":
            self.serve_artifact(query)
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        form = {k: v[0] for k, v in parse_qs(body).items()}

        try:
            if parsed.path == "/save-strategy":
                (self.run_dir / "strategy" / "STRATEGY.md").write_text(
                    form.get("strategy_text", ""),
                    encoding="utf-8",
                )
                self.redirect("/", "Saved STRATEGY.md")
                return
            if parsed.path == "/start-test":
                count = materialize_test_examples(self.run_dir)
                self.redirect("/", f"Materialized {count} test examples")
                return
            if parsed.path == "/classify":
                test_idx = int(form.get("test_idx", "0"))
                test_dir = self.run_dir / f"test-{test_idx:03d}"
                set_answer(test_dir, form.get("answer", ""))
                self.redirect(f"/?scope=test&test_idx={test_idx}", f"Saved answer for test {test_idx:03d}")
                return
            if parsed.path == "/run-tool":
                scope = form.get("scope", "strategy")
                record = run_tool(
                    self.run_dir,
                    scope=scope,
                    example_id=form.get("example_id", ""),
                    tool_name=form.get("tool_name", ""),
                    token_position=form.get("token_position", ""),
                    question=form.get("question", ""),
                    forced_text=form.get("forced_text", ""),
                )
                location = "/"
                if scope == "strategy":
                    location += f"?scope=strategy&example_id={quote(form.get('example_id', ''))}"
                else:
                    location += f"?scope=test&test_idx={form.get('test_idx', '0')}"
                self.redirect(location, f"Ran {record['tool']} (exit {record['exit_code']})")
                return
            if parsed.path == "/submit":
                result = score_human_run(self.run_dir)
                self.redirect(
                    "/",
                    f"Saved {result['submission_id']} ({result['answered']}/{result['total']} answered, miss={result['miss']})",
                )
                return
            self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.redirect("/", f"Error: {exc}")

    def serve_artifact(self, query: dict[str, list[str]]) -> None:
        raw = query.get("path", [""])[0]
        if not raw:
            self.send_error(HTTPStatus.BAD_REQUEST, "missing path")
            return
        target = (self.run_dir / raw).resolve()
        try:
            target.relative_to(self.run_dir.resolve())
        except ValueError:
            self.send_error(HTTPStatus.FORBIDDEN)
            return
        if not target.exists() or not target.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        text = target.read_text(encoding="utf-8", errors="replace")
        body = (
            "<!doctype html><html><head><meta charset='utf-8'>"
            f"<title>{html.escape(raw)}</title><style>{page_css()}</style></head><body>"
            f"<div class='content'><div class='panel'><h2>{html.escape(raw)}</h2>"
            f"<p><a href='/'>Back</a></p><pre>{html.escape(text)}</pre></div></div></body></html>"
        ).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def redirect(self, location: str, msg: str) -> None:
        parsed = urlparse(location)
        query = parse_qs(parsed.query)
        query["msg"] = [msg]
        new_query = urlencode([(k, v) for k, vals in query.items() for v in vals])
        path = parsed.path or "/"
        if new_query:
            path += f"?{new_query}"
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", path)
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(f"[human-ui] {fmt % args}\n")


def make_handler(run_dir: Path):
    class _Handler(HumanUIHandler):
        pass

    _Handler.run_dir = run_dir
    return _Handler


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Human baseline web UI")
    parser.add_argument("--task", help="Task name under data/")
    parser.add_argument("--run-dir", help="Open an existing human run directory")
    parser.add_argument(
        "--tools",
        default="ask,top_10_logits,top10_entropy,force",
        help="Comma-separated enabled tools for a new run",
    )
    parser.add_argument("--description", help="Override task description for a new run")
    parser.add_argument("--ood", action=argparse.BooleanOptionalAction, default=True,
                        help="Use OOD few-shot/test split rules (default: on)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    args = parser.parse_args(argv)

    if not args.run_dir and not args.task:
        parser.error("either --task or --run-dir is required")
    return args


def resolve_run(args: argparse.Namespace) -> Path:
    if args.run_dir:
        return Path(args.run_dir).resolve()
    tools = [name.strip() for name in args.tools.split(",") if name.strip()]
    return create_human_run(args.task, tools, description=args.description, ood=args.ood)


def serve(run_dir: Path, host: str, port: int) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(run_dir))
    print(f"Human baseline UI: http://{host}:{port}/")
    print(f"Run directory: {run_dir}")
    server.serve_forever()


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    run_dir = resolve_run(args)
    serve(run_dir, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
