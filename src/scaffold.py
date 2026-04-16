#!/usr/bin/env python3
"""Agent scaffold orchestrator.

Manages agent runs: initializes directories, launches strategy agents,
and coordinates test evaluation.

Usage:
    python scaffold.py init
    python scaffold.py run <task_name> [--description <desc>]
    python scaffold.py status [<run_id>]
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "agent-runs"
TRACES_DIR = ROOT / "agent-traces"
PROMPTS_DIR = ROOT / "prompts"
BIN_DIR = ROOT / "bin"


def init():
    """Create top-level directories and validate data/ exists."""
    for d in [RUNS_DIR, TRACES_DIR, BIN_DIR, ROOT / "src" / "tools"]:
        d.mkdir(parents=True, exist_ok=True)

    if not DATA_DIR.exists():
        print(f"Warning: {DATA_DIR} does not exist. Create it and add task folders.")
    else:
        tasks = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())
        print(f"Found {len(tasks)} task(s): {', '.join(tasks) if tasks else '(none)'}")

    print("Scaffold initialized.")


def load_task_metadata(task_name: str) -> dict:
    """Load task metadata from data/<task>/metadata.json if it exists."""
    meta_path = DATA_DIR / task_name / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return {"name": task_name, "description": f"Classification task: {task_name}"}


def populate_few_shot(task_name: str, strategy_dir: Path) -> list[dict]:
    """Copy raw few-shot JSONs into strategy/few-shot/ and write an index CSV.

    Returns a list of {id, label} entries.
    """
    src_dir = DATA_DIR / task_name / "few-shot"
    dst_dir = strategy_dir / "few-shot"
    dst_dir.mkdir(parents=True, exist_ok=True)

    index = []
    if not src_dir.exists():
        print(f"Warning: {src_dir} does not exist")
        (strategy_dir / "Examples.csv").write_text("id,label,path\n")
        return index

    for json_file in sorted(src_dir.glob("*.json")):
        shutil.copy2(json_file, dst_dir / json_file.name)
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        index.append({
            "id": json_file.stem,
            "label": data.get("label", ""),
            "path": f"few-shot/{json_file.name}",
        })
        # Copy companion .npy if present
        npy = src_dir / f"{json_file.stem}.npy"
        if npy.exists():
            shutil.copy2(npy, dst_dir / npy.name)

    with open(strategy_dir / "Examples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "path"])
        writer.writeheader()
        writer.writerows(index)

    print(f"Copied {len(index)} few-shot examples into {dst_dir}")
    return index


TOOL_DESCRIPTIONS = {
    # Populate as custom tools are added. Key = tool name, value = one-line description.
}


def render_tools_section(tools: list[str]) -> str:
    if not tools:
        return (
            "This run has **no custom research tools enabled**. "
            "You have standard file I/O (Read, Write, Edit, Bash, Glob, Grep) only."
        )
    lines = ["The following custom research tools are available on your PATH:\n"]
    for name in tools:
        desc = TOOL_DESCRIPTIONS.get(name, "(no description)")
        lines.append(f"- `{name}` — {desc}")
    return "\n".join(lines)


def generate_readme(task_meta: dict, tools: list[str], examples_index: list[dict], run_dir: Path):
    """Generate README.md for the strategy directory, tailored to task + tool set."""
    label_counts = {}
    for e in examples_index:
        label_counts[e["label"]] = label_counts.get(e["label"], 0) + 1
    label_summary = ", ".join(f"label={k}: {v}" for k, v in sorted(label_counts.items()))

    readme = f"""# Task: {task_meta['name']}

## Task Description
{task_meta['description']}

## Labels
- `label = 1` (positive) → answer **yes**
- `label = 0` (negative) → answer **no**

## Workspace Contents
- `README.md` — this file
- `STRATEGY.md` — write your classification strategy here (test agents will read it)
- `Examples.csv` — index of few-shot examples (id, label, path)
- `few-shot/` — raw JSON files for the {len(examples_index)} few-shot examples ({label_summary})

## Research Tools
{render_tools_section(tools)}

## The `test` command
Running `test` evaluates your current strategy against all held-out test examples in parallel. Each test example is given to an independent test agent that sees only the contents of this `strategy/` directory plus its own single test example. Call `test` when STRATEGY.md is ready.

## Instructions
1. Study the few-shot JSONs in `few-shot/` to understand what distinguishes positive from negative examples.
2. Write a clear, concrete classification strategy in `STRATEGY.md`. Test agents follow it literally.
3. Optionally create supporting CSVs or notes in this directory; reference them from STRATEGY.md.
4. Run `test` when ready.
"""
    (run_dir / "strategy" / "README.md").write_text(readme, encoding="utf-8")


def create_run(task_name: str, description: str | None = None, tools: list[str] | None = None):
    """Create a new agent run for a task and launch the strategy agent."""
    task_dir = DATA_DIR / task_name
    if not task_dir.exists():
        print(f"Error: Task '{task_name}' not found in {DATA_DIR}")
        sys.exit(1)

    tools = list(tools or [])

    # Create run directory. Per-test folders are created later by run_tests.py as
    # siblings of strategy/ (test-000/, test-001/, ...).
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / task_name / f"run-{run_id}"
    strategy_dir = run_dir / "strategy"
    trace_dir = TRACES_DIR / task_name / f"run-{run_id}"

    for d in [strategy_dir, trace_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load task metadata
    task_meta = load_task_metadata(task_name)
    if description:
        task_meta["description"] = description

    # Populate few-shot workspace (copies raw JSONs into strategy/few-shot/)
    examples_index = populate_few_shot(task_name, strategy_dir)

    # Generate task-and-toolset-specific README
    generate_readme(task_meta, tools, examples_index, run_dir)
    (strategy_dir / "STRATEGY.md").write_text(
        "# Strategy\n\n<!-- Write your classification strategy here -->\n"
    )

    # Save run metadata
    run_meta = {
        "task": task_name,
        "run_id": run_id,
        "created": datetime.now().isoformat(),
        "status": "running",
        "tools": tools,
        "task_meta": task_meta,
    }
    with open(run_dir / "run.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"Created run: {run_dir}")
    print(f"Traces: {trace_dir}")

    # Build the agent.bashrc with run-specific env vars
    bashrc_path = run_dir / "agent.bashrc"
    bashrc_content = f"""# Auto-generated bash environment for agent run
export SCAFFOLD_ROOT="{ROOT.as_posix()}"
export AGENT_RUN_DIR="{run_dir.as_posix()}"
export AGENT_TASK="{task_name}"
export AGENT_RUN_ID="{run_id}"
export PATH="{BIN_DIR.as_posix()}:$PATH"
"""
    bashrc_path.write_text(bashrc_content)

    # Launch Claude Code non-interactively
    strategy_prompt_path = PROMPTS_DIR / "strategy-agent.md"
    if not strategy_prompt_path.exists():
        print(f"Error: Strategy prompt not found at {strategy_prompt_path}")
        sys.exit(1)

    user_prompt = (
        "Read README.md in the current directory for the task brief, available tools, "
        "and workspace layout. Develop a classification strategy, write it to STRATEGY.md, "
        "and run `test` when ready."
    )

    # Do NOT pass --add-dir to the raw task dir: the strategy agent must not
    # read data/<task>/test/ (those JSONs still contain the ground-truth label).
    # Few-shot examples are already copied into strategy/few-shot/.
    cmd = [
        "claude",
        "--print",
        "--dangerously-skip-permissions",
        "--system-prompt-file", str(strategy_prompt_path),
        "--allowed-tools", "Read,Write,Edit,Bash,Glob,Grep",
        user_prompt,
    ]

    env = os.environ.copy()
    env["BASH_ENV"] = str(bashrc_path)
    env["AGENT_TYPE"] = "strategy"

    print(f"Launching strategy agent for {task_name}...")
    trace_file = trace_dir / "strategy-trace.txt"

    with open(trace_file, "w") as trace_out:
        result = subprocess.run(
            cmd,
            cwd=str(strategy_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        trace_out.write(result.stdout)

    # Update run status
    run_meta["status"] = "completed" if result.returncode == 0 else "failed"
    run_meta["finished"] = datetime.now().isoformat()
    with open(run_dir / "run.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"Strategy agent finished (exit code {result.returncode})")
    print(f"Trace saved to {trace_file}")

    return run_dir


def show_status(run_id: str | None = None):
    """Show status of agent runs."""
    if not RUNS_DIR.exists():
        print("No runs directory. Run 'init' first.")
        return

    runs = []
    for task_dir in sorted(RUNS_DIR.iterdir()):
        if not task_dir.is_dir():
            continue
        for rd in sorted(task_dir.iterdir()):
            if not rd.is_dir():
                continue
            meta_path = rd / "run.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                if run_id and meta["run_id"] != run_id:
                    continue
                runs.append(meta)

    if not runs:
        print("No runs found." + (f" (filter: {run_id})" if run_id else ""))
        return

    for r in runs:
        rd = RUNS_DIR / r["task"] / f"run-{r['run_id']}"
        test_folders = [d for d in rd.iterdir() if d.is_dir() and d.name.startswith("test-")]
        test_count = len(test_folders)
        answer_count = sum(1 for d in test_folders if (d / "answer.txt").exists())

        print(f"  {r['task']}/run-{r['run_id']}  status={r['status']}  "
              f"tests={answer_count}/{test_count}  created={r['created']}")


def main():
    parser = argparse.ArgumentParser(description="Agent scaffold orchestrator")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Initialize scaffold directories")

    run_parser = sub.add_parser("run", help="Launch a strategy agent on a task")
    run_parser.add_argument("task_name", help="Name of task folder in data/")
    run_parser.add_argument("--description", help="Override task description")
    run_parser.add_argument(
        "--tools",
        default="",
        help="Comma-separated list of custom research tools to enable (default: empty)",
    )

    status_parser = sub.add_parser("status", help="Show run status")
    status_parser.add_argument("run_id", nargs="?", help="Filter by run ID")

    args = parser.parse_args()

    if args.command == "init":
        init()
    elif args.command == "run":
        tools = [t.strip() for t in args.tools.split(",") if t.strip()]
        create_run(args.task_name, args.description, tools)
    elif args.command == "status":
        show_status(args.run_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
