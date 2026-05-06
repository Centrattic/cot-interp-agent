#!/usr/bin/env python3
"""Launch 10 fresh sample-only runs on the remote box.

Mirrors remote_top10_fresh_launch.py but with --tools sample, gpt-5.4 medium.
Codex calls go through /usr/local/bin/codex (auto-rotating wrapper) to handle
weekly usage-limit rotation across the 6 profiles in /root/.codex-profiles/.
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = Path("/tmp/remote_sample_fresh_launch.log")
# Conservative — there's already a top10 batch (MAX_PARALLEL=6) running on the box.
MAX_PARALLEL = 4

BASE_ENV = {
    "TERM": "tmux",
    "PYTHON": "/usr/bin/python3",
    "AGENT_STRATEGY_PARALLEL": "1",
    "AGENT_TEST_MAX_WORKERS": "2",
    "AGENT_TEST_RETRY_ATTEMPTS": "8",
    "AGENT_TEST_RETRY_DELAY_SEC": "10",
    "AGENT_PARTITION_RESUME_ATTEMPTS": "8",
    "CODEX_MODEL": "gpt-5.4",
    "CODEX_REASONING_EFFORT": "medium",
}

# 10 fresh runs to push each task closer to 3-of-3 sample-only quota.
# Skips reasoning_termination (covered once partial fills finish) and
# user_preference_sycophancy (already 2 done + 1 partial → handled separately).
RUN_PLAN: list[tuple[str, int]] = [
    ("stanford_hint_clean", 2),
    ("gemma_self_deletion_clean", 2),
    ("atypical_answer", 2),
    ("atypical_cot_length", 2),
    ("followup_confidence", 2),
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
                f"./scaffold.sh run {task} --agent-backend codex --tools sample --n-strategies 10",
            ]
            jobs.append((tag, cmd))
    return jobs


def main() -> int:
    LOG.write_text("")
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
            stdout_path = LOG.parent / f"remote_sample_fresh_{tag.replace('#', '_')}.out"
            fout = stdout_path.open("w", encoding="utf-8")
            p = subprocess.Popen(
                cmd, cwd=str(ROOT), env=env,
                stdout=fout, stderr=subprocess.STDOUT, text=True,
            )
            running.append((tag, p))
            log(f"start: {tag} pid={p.pid} → {stdout_path}")

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
