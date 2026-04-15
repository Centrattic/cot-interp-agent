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


def build_examples_csv(task_name: str, output_path: Path):
    """Read few-shot JSON examples and write Examples.csv."""
    few_shot_dir = DATA_DIR / task_name / "few-shot"
    if not few_shot_dir.exists():
        print(f"Warning: {few_shot_dir} does not exist, creating empty Examples.csv")
        output_path.write_text("id,content,label\n")
        return

    examples = []
    for json_file in sorted(few_shot_dir.glob("*.json")):
        with open(json_file) as f:
            data = json.load(f)
        example_id = json_file.stem
        # Flatten: extract common fields, keep rest as JSON string
        examples.append({
            "id": example_id,
            "content": json.dumps(data.get("content", data), ensure_ascii=False),
            "label": data.get("label", ""),
            "has_activations": (few_shot_dir / f"{example_id}.npy").exists(),
        })

    if not examples:
        output_path.write_text("id,content,label,has_activations\n")
        return

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "content", "label", "has_activations"])
        writer.writeheader()
        writer.writerows(examples)

    print(f"Wrote {len(examples)} examples to Examples.csv")


def generate_readme(task_meta: dict, run_dir: Path):
    """Generate README.md for the strategy directory."""
    readme = f"""# {task_meta['name']}

## Task Description
{task_meta['description']}

## Workspace Contents
- **STRATEGY.md** — Write your classification strategy here
- **Examples.csv** — Few-shot examples with labels
- Any additional CSV files you create will be available to test agents

## Available Commands
- `test` — Evaluate your strategy against test examples
"""
    (run_dir / "strategy" / "README.md").write_text(readme)


def create_run(task_name: str, description: str | None = None):
    """Create a new agent run for a task and launch the strategy agent."""
    task_dir = DATA_DIR / task_name
    if not task_dir.exists():
        print(f"Error: Task '{task_name}' not found in {DATA_DIR}")
        sys.exit(1)

    # Create run directory
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / task_name / f"run-{run_id}"
    strategy_dir = run_dir / "strategy"
    test_dir = run_dir / "test"
    trace_dir = TRACES_DIR / task_name / f"run-{run_id}"

    for d in [strategy_dir, test_dir, trace_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Load task metadata
    task_meta = load_task_metadata(task_name)
    if description:
        task_meta["description"] = description

    # Generate strategy workspace files
    generate_readme(task_meta, run_dir)
    (strategy_dir / "STRATEGY.md").write_text(
        "# Strategy\n\n<!-- Write your classification strategy here -->\n"
    )
    build_examples_csv(task_name, strategy_dir / "Examples.csv")

    # Save run metadata
    run_meta = {
        "task": task_name,
        "run_id": run_id,
        "created": datetime.now().isoformat(),
        "status": "running",
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

    task_desc = task_meta["description"]
    user_prompt = (
        f"You are working on task: {task_name}\n\n"
        f"Task description: {task_desc}\n\n"
        f"Your workspace is the current directory (strategy/).\n"
        f"Study the few-shot examples, develop a classification strategy, "
        f"write it to STRATEGY.md, and run `test` when ready."
    )

    cmd = [
        "claude",
        "--print",
        "--dangerously-skip-permissions",
        "--system-prompt-file", str(strategy_prompt_path),
        "--add-dir", str(task_dir),
        "--allowed-tools", "Read,Write,Edit,Bash,Glob,Grep",
        user_prompt,
    ]

    env = os.environ.copy()
    env["BASH_ENV"] = str(bashrc_path)

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
        test_dir = RUNS_DIR / r["task"] / f"run-{r['run_id']}" / "test"
        test_count = sum(1 for d in test_dir.iterdir() if d.is_dir()) if test_dir.exists() else 0
        answer_count = sum(
            1 for d in test_dir.iterdir()
            if d.is_dir() and (d / "answer.txt").exists()
        ) if test_dir.exists() else 0

        print(f"  {r['task']}/run-{r['run_id']}  status={r['status']}  "
              f"tests={answer_count}/{test_count}  created={r['created']}")


def main():
    parser = argparse.ArgumentParser(description="Agent scaffold orchestrator")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("init", help="Initialize scaffold directories")

    run_parser = sub.add_parser("run", help="Launch a strategy agent on a task")
    run_parser.add_argument("task_name", help="Name of task folder in data/")
    run_parser.add_argument("--description", help="Override task description")

    status_parser = sub.add_parser("status", help="Show run status")
    status_parser.add_argument("run_id", nargs="?", help="Filter by run ID")

    args = parser.parse_args()

    if args.command == "init":
        init()
    elif args.command == "run":
        create_run(args.task_name, args.description)
    elif args.command == "status":
        show_status(args.run_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
