#!/usr/bin/env python3
"""Launch 13 fresh top10-only runs on the remote box to fill the 3-per-task quota.

Designed to be rsynced to the remote and executed there. Runs go in parallel
up to MAX_PARALLEL; each run launches the scaffold's strategy + test pipeline.

Each launch sets:
  TERM=tmux                       (per remote convention)
  CODEX_MODEL=gpt-5.4 medium
  AGENT_TEST_MAX_WORKERS=2        (matches prior batch settings)
  Retry knobs identical to scripts/top10_full_runs_queue_20260506.py
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = Path("/tmp/remote_top10_fresh_launch.log")
MAX_PARALLEL = 6  # 128-core box, but each scaffold spawns 10 partitions internally

BASE_ENV = {
    "TERM": "tmux",
    # Pin the Python interpreter scaffold.sh picks up. Without this, scaffold.sh
    # finds uv-managed cpython-3.11 first and that env is missing numpy/etc.
    "PYTHON": "/usr/bin/python3",
    "AGENT_STRATEGY_PARALLEL": "1",
    "AGENT_TEST_MAX_WORKERS": "2",
    "AGENT_TEST_RETRY_ATTEMPTS": "8",
    "AGENT_TEST_RETRY_DELAY_SEC": "10",
    "AGENT_PARTITION_RESUME_ATTEMPTS": "8",
    "CODEX_MODEL": "gpt-5.4",
    "CODEX_REASONING_EFFORT": "medium",
}

# (task_name, count) — fresh runs needed per task to reach 3 usable per task,
# *after* a parallel local resume of reasoning_termination/run-20260505-142824
# (which finishes 4 stubs and bumps reasoning_termination's usable count by 1).
RUN_PLAN: list[tuple[str, int]] = [
    ("atypical_answer_ood", 3),
    ("atypical_cot_length", 2),
    ("followup_confidence", 3),
    ("gemma_self_deletion_clean", 1),
    ("reasoning_termination", 1),
    ("stanford_hint_clean", 2),
    ("user_preference_sycophancy", 1),
]


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def make_jobs() -> list[tuple[str, list[str]]]:
    jobs: list[tuple[str, list[str]]] = []
    for task, count in RUN_PLAN:
        for i in range(count):
            tag = f"{task}#{i+1}"
            cmd = [
                "bash",
                "-lc",
                f"./scaffold.sh run {task} --agent-backend codex --tools top_10_logits --n-strategies 10",
            ]
            jobs.append((tag, cmd))
    return jobs


def main() -> int:
    LOG.write_text("")  # truncate
    jobs = make_jobs()
    running: list[tuple[str, subprocess.Popen[str]]] = []
    idx = 0

    log(f"queue-start: total={len(jobs)} parallel={MAX_PARALLEL}")
    while idx < len(jobs) or running:
        while idx < len(jobs) and len(running) < MAX_PARALLEL:
            tag, cmd = jobs[idx]
            idx += 1
            env = os.environ.copy()
            env.update(BASE_ENV)
            stdout_path = LOG.parent / f"remote_top10_fresh_{tag.replace('#', '_')}.out"
            fout = stdout_path.open("w", encoding="utf-8")
            p = subprocess.Popen(
                cmd, cwd=str(ROOT), env=env,
                stdout=fout, stderr=subprocess.STDOUT, text=True,
            )
            running.append((tag, p))
            log(f"start: {tag} pid={p.pid} → {stdout_path}")

        # Reap finished
        still: list[tuple[str, subprocess.Popen[str]]] = []
        for tag, p in running:
            rc = p.poll()
            if rc is None:
                still.append((tag, p))
            else:
                log(f"done: {tag} pid={p.pid} exit={rc}")
        running = still
        if running and idx == len(jobs):
            time.sleep(15)
        elif running:
            time.sleep(5)
    log("queue-end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
