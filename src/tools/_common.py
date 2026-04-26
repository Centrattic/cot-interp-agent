"""Shared helpers for custom agent tools (ask / force / top_10_logits / top10_entropy).

Responsibilities:
- Resolve the correct example directory based on AGENT_TYPE
  (strategy -> few-shot/, test -> test/)
- Load an example by ID
- Parse integers with a friendly error message
"""

from __future__ import annotations

import json
import os
import sys
from csv import DictWriter
from pathlib import Path


VALID_AGENT_TYPES = ("strategy", "test")


def fail(msg: str, code: int = 2) -> None:
    """Print an error to stderr and exit."""
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def get_env() -> dict:
    """Pull required env vars set by the scaffold launcher."""
    required = ["SCAFFOLD_ROOT", "AGENT_TASK", "AGENT_TYPE", "AGENT_RUN_DIR"]
    out = {}
    for key in required:
        val = os.environ.get(key)
        if not val:
            fail(
                f"{key} not set. This tool must be invoked inside an agent session."
            )
        out[key] = val
    if out["AGENT_TYPE"] not in VALID_AGENT_TYPES:
        fail(
            f"AGENT_TYPE={out['AGENT_TYPE']!r} invalid; expected one of {VALID_AGENT_TYPES}"
        )
    return out


def example_dir(env: dict) -> Path:
    """Return the example directory this agent may query.

    Prefer the agent workspace copy when present:
    - strategy: ``<cwd>/few-shot/`` or ``$AGENT_RUN_DIR/strategy/few-shot/``
      (populated fresh by scaffold.py for every partition in multi-partition mode).
    - test: current test folder if it contains ``example.json``.

    Falls back to ``data/<task>/{few-shot,test}`` for legacy runs. Test agents
    are still restricted to their own ``AGENT_EXAMPLE_ID`` by downstream scope
    checks in individual tools.
    """
    cwd = Path.cwd()
    if env["AGENT_TYPE"] == "strategy":
        local = cwd / "few-shot"
        if local.exists():
            return local
        run_dir = os.environ.get("AGENT_RUN_DIR", "").strip()
        if run_dir:
            run_local = Path(run_dir) / "strategy" / "few-shot"
            if run_local.exists():
                return run_local
    else:
        if (cwd / "example.json").exists():
            return cwd


    scaffold_root = Path(env["SCAFFOLD_ROOT"])
    task = env["AGENT_TASK"]
    if env["AGENT_TYPE"] == "strategy":
        path = Path(env["AGENT_RUN_DIR"]) / "strategy" / "few-shot"
    else:
        path = scaffold_root / "data" / task / "test"
    if not path.exists():
        fail(f"example directory not found: {path}")
    return path


def list_few_shot_ids(env: dict) -> list[str]:
    """Return sorted example-ids (filename stems) from the strategy agent's
    current few-shot directory. Strategy-only helper — errors the same way
    ``example_dir`` does if the dir is missing."""
    if env["AGENT_TYPE"] != "strategy":
        fail("list_few_shot_ids() is only valid for strategy agents.")
    base = example_dir(env)
    return sorted(p.stem for p in base.glob("*.json"))


def load_example(env: dict, example_id: str) -> dict:
    """Load the JSON record for an example ID from the agent's allowed scope."""
    base = example_dir(env)
    if env["AGENT_TYPE"] == "test" and (base / "example.json").exists():
        with open(base / "example.json") as f:
            return json.load(f)
    json_path = base / f"{example_id}.json"
    if not json_path.exists():
        fail(
            f"example {example_id!r} not found in {base}. "
            f"(This agent only has access to {env['AGENT_TYPE']} examples.)"
        )
    with open(json_path) as f:
        return json.load(f)


def parse_int(s: str, name: str) -> int:
    try:
        return int(s)
    except ValueError:
        fail(f"{name} must be an integer, got {s!r}")
        raise  # unreachable, keeps type checkers happy


def next_numbered_output_path(prefix: str, suffix: str = ".csv", cwd: Path | None = None) -> Path:
    base = cwd or Path.cwd()
    n = 1
    while True:
        path = base / f"{prefix}_{n}{suffix}"
        if not path.exists():
            return path
        n += 1


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
