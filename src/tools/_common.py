"""Shared helpers for custom agent tools (force / logit / entropy / ask).

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
from pathlib import Path


VALID_AGENT_TYPES = ("strategy", "test")


def fail(msg: str, code: int = 2) -> None:
    """Print an error to stderr and exit."""
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def get_env() -> dict:
    """Pull required env vars set by the scaffold launcher."""
    required = ["SCAFFOLD_ROOT", "AGENT_TASK", "AGENT_TYPE"]
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
    """Return the data directory this agent may query, based on AGENT_TYPE."""
    scaffold_root = Path(env["SCAFFOLD_ROOT"])
    task = env["AGENT_TASK"]
    subdir = "few-shot" if env["AGENT_TYPE"] == "strategy" else "test"
    path = scaffold_root / "data" / task / subdir
    if not path.exists():
        fail(f"example directory not found: {path}")
    return path


def load_example(env: dict, example_id: str) -> dict:
    """Load the JSON record for an example ID from the agent's allowed scope."""
    base = example_dir(env)
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
