#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LOG = Path("/tmp/top10_fill_remaining_20260506.log")
MAX_PARALLEL = 2

JOBS = [
    "atypical_answer",
    "atypical_answer",
    "atypical_cot_length",
    "atypical_cot_length",
    "followup_confidence",
    "followup_confidence",
    "gemma_self_deletion_clean",
    "gemma_self_deletion_clean",
    "reasoning_termination",
    "stanford_hint_clean",
]


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {msg}\n")


def launch(task: str) -> subprocess.Popen[str]:
    env = os.environ.copy()
    env.update(
        {
            "AGENT_STRATEGY_PARALLEL": "1",
            "AGENT_TEST_MAX_WORKERS": "2",
            "AGENT_TEST_RETRY_ATTEMPTS": "8",
            "AGENT_TEST_RETRY_DELAY_SEC": "10",
            "AGENT_PARTITION_RESUME_ATTEMPTS": "8",
            "CODEX_MODEL": "gpt-5.5",
            "CODEX_REASONING_EFFORT": "medium",
        }
    )
    cmd = [
        "bash",
        "-lc",
        f"source .venv/bin/activate && ./scaffold.sh run {task} --agent-backend codex --tools top_10_logits --n-strategies 10",
    ]
    return subprocess.Popen(
        cmd,
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        text=True,
    )


def main() -> int:
    running: list[tuple[str, subprocess.Popen[str]]] = []
    idx = 0
    log("fill-start")
    while idx < len(JOBS) or running:
        while idx < len(JOBS) and len(running) < MAX_PARALLEL:
            task = JOBS[idx]
            log(f"launching {task}")
            running.append((task, launch(task)))
            idx += 1

        next_running: list[tuple[str, subprocess.Popen[str]]] = []
        for task, proc in running:
            rc = proc.poll()
            if rc is None:
                next_running.append((task, proc))
            else:
                log(f"finished {task} rc={rc}")
        running = next_running
        time.sleep(15)

    log("fill-done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
