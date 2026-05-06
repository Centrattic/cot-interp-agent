#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
LOG = Path("/tmp/top10_full_runs_queue_20260506b.log")
MAX_PARALLEL = 2


BASE_ENV = {
    "AGENT_STRATEGY_PARALLEL": "1",
    "AGENT_TEST_MAX_WORKERS": "2",
    "AGENT_TEST_RETRY_ATTEMPTS": "8",
    "AGENT_TEST_RETRY_DELAY_SEC": "10",
    "AGENT_PARTITION_RESUME_ATTEMPTS": "8",
    "CODEX_MODEL": "gpt-5.5",
    "CODEX_REASONING_EFFORT": "medium",
}


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d_%H:%M:%S")
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"[{stamp}] {msg}\n")


def make_jobs() -> list[tuple[str, list[str], dict[str, str] | None]]:
    jobs: list[tuple[str, list[str], dict[str, str] | None]] = []

    jobs.append((
        "salvage:reasoning_termination:run-20260505-180425-372935",
        [
            "bash",
            "-lc",
            "source agent-runs/reasoning_termination/run-20260505-180425-372935/agent.bashrc && "
            "python src/run_tests.py && "
            "python src/score_run.py agent-runs/reasoning_termination/run-20260505-180425-372935",
        ],
        BASE_ENV,
    ))

    for task, count in [
        ("atypical_answer", 2),
        ("atypical_cot_length", 2),
        ("followup_confidence", 2),
        ("gemma_self_deletion_clean", 2),
        ("reasoning_termination", 2),
        ("stanford_hint_clean", 1),
    ]:
        for _ in range(count):
            jobs.append((
                f"run:{task}",
                [
                    "bash",
                    "-lc",
                    f"source .venv/bin/activate && "
                    f"CODEX_MODEL=gpt-5.5 CODEX_REASONING_EFFORT=medium "
                    f"AGENT_STRATEGY_PARALLEL=1 AGENT_TEST_MAX_WORKERS=2 "
                    f"AGENT_TEST_RETRY_ATTEMPTS=8 AGENT_TEST_RETRY_DELAY_SEC=10 "
                    f"AGENT_PARTITION_RESUME_ATTEMPTS=8 "
                    f"./scaffold.sh run {task} --agent-backend codex --tools top_10_logits --n-strategies 10",
                ],
                None,
            ))
    return jobs


def main() -> int:
    jobs = make_jobs()
    running: list[tuple[str, subprocess.Popen[str]]] = []
    idx = 0

    log("queue-start")
    while idx < len(jobs) or running:
        while idx < len(jobs) and len(running) < MAX_PARALLEL:
            label, cmd, env_extra = jobs[idx]
            env = os.environ.copy()
            if env_extra:
                env.update(env_extra)
            log(f"launching {label}")
            proc = subprocess.Popen(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                text=True,
            )
            running.append((label, proc))
            idx += 1

        next_running: list[tuple[str, subprocess.Popen[str]]] = []
        for label, proc in running:
            rc = proc.poll()
            if rc is None:
                next_running.append((label, proc))
            else:
                log(f"finished {label} rc={rc}")
        running = next_running
        time.sleep(15)

    log("queue-done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
