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

# Allow importing from src/ (for tools.sae_encode)
sys.path.insert(0, str(ROOT / "src"))
DATA_DIR = ROOT / "data"
RUNS_DIR = ROOT / "agent-runs"
TRACES_DIR = ROOT / "agent-traces"
PROMPTS_DIR = ROOT / "prompts"
BIN_DIR = ROOT / "bin"
ENV_FILE = ROOT / ".env"


def load_dotenv(path: Path) -> None:
    """Populate os.environ from a KEY=VALUE .env file (does not overwrite
    values already set in the shell). Used so scaffolded agents inherit
    secrets like OPENROUTER_API_KEY without the user having to export them."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


load_dotenv(ENV_FILE)


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


def _apply_whitelist(data: dict, test_keep_fields: list[str] | None) -> dict:
    """Keep only whitelisted fields plus `label` (few-shot retains label for GT)."""
    if not test_keep_fields:
        return data
    filtered = {k: data[k] for k in test_keep_fields if k in data}
    if "label" in data:
        filtered["label"] = data["label"]
    return filtered


def populate_few_shot(
    task_name: str,
    strategy_dir: Path,
    test_keep_fields: list[str] | None = None,
) -> list[dict]:
    """Legacy single-run mode: copy data/<task>/few-shot/ (already-sampled at ingest
    time) into strategy/few-shot/ with the whitelist applied.

    Returns a list of {id, label, path} entries (used for Examples.csv).
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
        with open(json_file, encoding="utf-8") as f:
            data = json.load(f)
        filtered = _apply_whitelist(data, test_keep_fields)
        with open(dst_dir / json_file.name, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)
        index.append({
            "id": json_file.stem,
            "label": data.get("label", ""),
            "path": f"few-shot/{json_file.name}",
        })
        npy = src_dir / f"{json_file.stem}.npy"
        if npy.exists():
            shutil.copy2(npy, dst_dir / npy.name)

    _write_examples_csv(strategy_dir, index)

    wl = f" (whitelisted to {len(test_keep_fields)} fields + label)" if test_keep_fields else ""
    print(f"Wrote {len(index)} few-shot examples into {dst_dir}{wl}")
    return index


def populate_few_shot_from_source(
    task_meta: dict,
    strategy_dir: Path,
    seed: int,
    per_class: int,
    test_keep_fields: list[str] | None,
) -> list[dict]:
    """Multi-partition mode: sample a fresh balanced few-shot set directly from the
    cot-proxy-tasks source split (per metadata.few_shot_split). Different `seed`
    values → different sampled sets per partition.

    Requires metadata.json to have `source`, `few_shot_split`, and `label_map`.
    """
    source = Path(task_meta["source"])
    split = task_meta.get("few_shot_split", "train")
    src_dir = source / split
    if not src_dir.is_dir():
        raise RuntimeError(
            f"Source few-shot split not found: {src_dir}. "
            "Multi-partition requires metadata.json with 'source' and 'few_shot_split'."
        )

    # Parse label_map. JSON keys are always strings; normalize back to int where applicable.
    raw_label_map = task_meta.get("label_map", {})
    def _key(k):
        try: return int(k)
        except (ValueError, TypeError): return k
    label_map = {_key(k): int(v) for k, v in raw_label_map.items()}

    # Import sampler from ingest_cot_proxy (co-located in src/)
    from ingest_cot_proxy import sample_balanced
    picked = sample_balanced(src_dir, per_class, seed, label_map)

    dst_dir = strategy_dir / "few-shot"
    dst_dir.mkdir(parents=True, exist_ok=True)

    index = []
    for src_path, data in picked:
        filtered = _apply_whitelist(data, test_keep_fields)
        with open(dst_dir / src_path.name, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)
        index.append({
            "id": src_path.stem,
            "label": data["label"],
            "path": f"few-shot/{src_path.name}",
        })

    _write_examples_csv(strategy_dir, index)
    wl = f" (whitelisted to {len(test_keep_fields)} fields + label)" if test_keep_fields else ""
    print(f"Sampled {len(index)} few-shot from {split}/ (seed={seed}){wl}")
    return index


def _write_examples_csv(strategy_dir: Path, index: list[dict]) -> None:
    with open(strategy_dir / "Examples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "path"])
        writer.writeheader()
        writer.writerows(index)


TOOL_DESCRIPTIONS = {
    # Key = tool name. Value = multi-line markdown blurb rendered into strategy/README.md
    # under "## Research Tools". Both the strategy agent and the test agent read that
    # README, so each blurb should cover both scopes (what each agent may query).
    "ask": """### `ask <example_id> "<question>"`

Ask a short follow-up question about an example via an oracle model
(OpenRouter / Qwen3-32B, pinned to DeepInfra for reproducibility).

**Limits**
- Question must tokenize to **≤20 Qwen tokens** (else the call fails and no file is written).
- The reply is truncated to the **first 5 tokens** (reasoning / thinking excluded). Because the reply is so short, **phrase the question to demand a very concise answer** (e.g. "answer in 3 words", "yes or no only", "one word"). Open-ended phrasings get cut mid-sentence.
- No logit access.

**Scope**
- **Strategy agent:** may query any few-shot `<example_id>` (filename stem in `Examples.csv`).
- **Test agent:** may only query its own assigned example — `AGENT_EXAMPLE_ID` names it; other IDs are rejected.

**Output contract.** Prints a `status:` line to stdout:
- On success, also prints `response:` and `details:` — the latter names a new file `ask_<n>.json` in your current directory with the full question, truncated response, raw response, model, and token counts.
- On failure (token limit exceeded, wrong example id, etc.), prints the reason and writes **no** file.
""".strip(),
    "top_10_logits": """### `top_10_logits <example_id> <token_position>`

Print the top-10 tokens and their logit values at `token_position` within
the example's chain-of-thought, read from a precomputed sidecar
(`<example_id>.logits.npz` next to the example's JSON). Output is 10 lines
of `<token_repr>\\t<logit>`, logits descending.

**Positions are CoT-relative.** `token_position = 0` is the first token of
`cot_prefix` (what you see in example.json). Valid range is
`[0, len(cot_prefix_tokens))`.

**Scope**
- **Strategy agent:** any few-shot `<example_id>`.
- **Test agent:** only its assigned `AGENT_EXAMPLE_ID`.

**Limits**
- Position must be inside the CoT range (fails with a clear error otherwise).
- If the sidecar file is missing, the tool fails and tells you to run
  `src/precompute_logits.py` for this task.
""".strip(),
    "top10_entropy": """### `top10_entropy <example_id> <token_position>`

Entropy (nats) of the softmaxed top-10 token distribution at `token_position`,
read from the same precomputed sidecar as `top_10_logits`. A proxy for how
confident the model is about what comes next at that position. Low entropy =
the model is peaked on one/few tokens; high entropy = the top-10 are spread
out.

Positions are **CoT-relative** (same convention as `top_10_logits`).

**Scope**
- **Strategy agent:** any few-shot `<example_id>`.
- **Test agent:** only its assigned `AGENT_EXAMPLE_ID`.

**Note**
- Entropy is computed over only the top-10 logits (softmax restricted to
  those 10 values), not over the full vocabulary. It is therefore an
  underestimate of true entropy but preserves the relative ordering
  (more peaked vs more spread) which is what matters for this signal.
""".strip(),
    "force": """### `force <example_id> <token_position> <tokens_to_force...>`

Splice **up to 10 tokens** into the example's chain-of-thought at a given
CoT-relative position, ask the model what it would emit next, and print the
next token plus the top-10 logprobs at that next-token slot.

The effective prompt is
`<chat prefix> + <cot_prefix[:token_position] tokens> + <tokens_to_force>`,
and the tool returns the greedy next token together with the top-10
distribution at that slot — one forward pass through Tinker's SamplingClient.

**Positions are CoT-relative.** `0` means "prepend the forced tokens at the
start of the CoT"; `len(cot_prefix_tokens)` means "append them at the end
of the CoT (before any continuation)".

**Scope**
- **Strategy agent:** any few-shot `<example_id>`.
- **Test agent:** only its assigned `AGENT_EXAMPLE_ID`.

**Limits**
- `tokens_to_force` must tokenize to ≤10 tokens with the base model's tokenizer.
- Requires TINKER_BASE_MODEL (and Tinker auth) to be set in the environment
  of the scaffold launcher.

**Output**
```
next_token: '<token>'
top_10:
  '<tok>'\\t<logprob>
  ...
```
""".strip(),
}


def render_tools_section(tools: list[str]) -> str:
    if not tools:
        return (
            "This run has **no custom research tools enabled**. "
            "You have standard file I/O (Read, Write, Edit, Bash, Glob, Grep) only."
        )
    lines = [
        "The following custom research tools are available on your PATH. "
        "(The system prompt does not enumerate tools — this section is authoritative.)\n"
    ]
    for name in tools:
        desc = TOOL_DESCRIPTIONS.get(name, f"`{name}` — (no description available)")
        lines.append(desc)
        lines.append("")  # blank line between tool blurbs
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


def _write_bashrc(
    bashrc_path: Path,
    run_dir: Path,
    task_name: str,
    run_id: str,
    extra_exports: dict[str, str] | None = None,
) -> None:
    """Write the per-agent bash environment file sourced via BASH_ENV."""
    lines = [
        "# Auto-generated bash environment for agent run",
        f'export SCAFFOLD_ROOT="{ROOT.as_posix()}"',
        f'export AGENT_RUN_DIR="{run_dir.as_posix()}"',
        f'export AGENT_TASK="{task_name}"',
        f'export AGENT_RUN_ID="{run_id}"',
        f'export PATH="{BIN_DIR.as_posix()}:$PATH"',
        f'export PYTHON="{Path(sys.executable).as_posix()}"',
    ]
    for k, v in (extra_exports or {}).items():
        lines.append(f'export {k}="{v}"')
    bashrc_path.write_text("\n".join(lines) + "\n")


def _launch_strategy_agent(
    strategy_dir: Path,
    trace_base: Path,
    bashrc_path: Path,
    label: str = "",
) -> int:
    """Run one strategy-agent claude subprocess in strategy_dir. Writes
    trace_base.jsonl / .txt. Returns exit code.
    """
    strategy_prompt_path = PROMPTS_DIR / "strategy-agent.md"
    if not strategy_prompt_path.exists():
        print(f"Error: Strategy prompt not found at {strategy_prompt_path}")
        sys.exit(1)

    user_prompt = (
        "Read README.md in the current directory for the task brief, available tools, "
        "and workspace layout. Develop a classification strategy, write it to STRATEGY.md, "
        "and run `test` when ready."
    )
    system_prompt = strategy_prompt_path.read_text(encoding="utf-8")
    cmd = [
        "claude", "--print", "--dangerously-skip-permissions",
        "--output-format", "stream-json", "--verbose",
        "--system-prompt", system_prompt,
        "--allowed-tools", "Read,Write,Edit,Bash,Glob,Grep",
    ]
    env = os.environ.copy()
    env["BASH_ENV"] = str(bashrc_path)
    env["AGENT_TYPE"] = "strategy"

    tag = f" [{label}]" if label else ""
    print(f"Launching strategy agent{tag}...")
    result = subprocess.run(
        cmd,
        cwd=str(strategy_dir),
        env=env,
        input=user_prompt,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    )
    from render_trace import write_trace_pair
    write_trace_pair(result.stdout, trace_base)
    print(f"Strategy agent{tag} finished (exit code {result.returncode})")
    return result.returncode


def create_run(
    task_name: str,
    description: str | None = None,
    tools: list[str] | None = None,
    n_strategies: int = 1,
    strategy_seed_base: int = 0,
    few_shot_per_class: int | None = None,
):
    """Create a new agent run.

    If n_strategies == 1: legacy layout (run-<ts>/strategy/, run-<ts>/test-NNN/).
    If n_strategies > 1: partition layout (run-<ts>/partition-NNN/{strategy/, test-NNN/}).
      Each partition gets a freshly-sampled few-shot set (seed = strategy_seed_base + k)
      drawn from the cot-proxy-tasks source split, and runs the k-th round-robin
      slice of the test set.
    """
    task_dir = DATA_DIR / task_name
    if not task_dir.exists():
        print(f"Error: Task '{task_name}' not found in {DATA_DIR}")
        sys.exit(1)

    tools = list(tools or [])
    task_meta = load_task_metadata(task_name)
    if description:
        task_meta["description"] = description
    fspc = few_shot_per_class if few_shot_per_class is not None else \
        int(task_meta.get("few_shot_per_class", 5))

    # Precompute SAE activations for few-shot and test examples (if .npy files exist).
    # Cached as .sae.npz alongside .npy; idempotent across runs.
    try:
        from tools.sae_encode import precompute_task
        precompute_task(ROOT, task_name)
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: SAE precompute failed: {e}")

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = RUNS_DIR / task_name / f"run-{run_id}"
    trace_dir = TRACES_DIR / task_name / f"run-{run_id}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "task": task_name,
        "run_id": run_id,
        "created": datetime.now().isoformat(),
        "status": "running",
        "tools": tools,
        "n_strategies": n_strategies,
        "strategy_seed_base": strategy_seed_base,
        "task_meta": task_meta,
    }
    with open(run_dir / "run.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    print(f"Created run: {run_dir}  (n_strategies={n_strategies})")
    print(f"Traces: {trace_dir}")

    if n_strategies == 1:
        _setup_partition(
            run_dir=run_dir, task_name=task_name, task_meta=task_meta, tools=tools,
            partition_idx=0, n_partitions=1,
            strategy_dir=run_dir / "strategy",
            bashrc_path=run_dir / "agent.bashrc",
            seed=None, few_shot_per_class=fspc, from_source=False,
        )
        trace_base = trace_dir / "strategy-trace"
        code = _launch_strategy_agent(
            strategy_dir=run_dir / "strategy",
            trace_base=trace_base,
            bashrc_path=run_dir / "agent.bashrc",
        )
        run_meta["status"] = "completed" if code == 0 else "failed"
    else:
        # Multi-partition: set up N partition dirs, launch their strategy agents in parallel.
        max_parallel = int(os.environ.get("AGENT_STRATEGY_PARALLEL", "10"))
        partition_launch_jobs = []
        for k in range(n_strategies):
            part_dir = run_dir / f"partition-{k:03d}"
            part_bashrc = part_dir / "agent.bashrc"
            _setup_partition(
                run_dir=run_dir, task_name=task_name, task_meta=task_meta, tools=tools,
                partition_idx=k, n_partitions=n_strategies,
                strategy_dir=part_dir / "strategy",
                bashrc_path=part_bashrc,
                seed=strategy_seed_base + k, few_shot_per_class=fspc, from_source=True,
            )
            partition_launch_jobs.append((
                part_dir / "strategy",
                trace_dir / f"partition-{k:03d}-strategy-trace",
                part_bashrc,
                f"partition-{k:03d}",
            ))

        # Launch strategy agents concurrently
        from concurrent.futures import ThreadPoolExecutor, as_completed
        exit_codes = {}
        with ThreadPoolExecutor(max_workers=min(max_parallel, n_strategies)) as ex:
            futures = {
                ex.submit(_launch_strategy_agent, sd, tb, br, lbl): lbl
                for (sd, tb, br, lbl) in partition_launch_jobs
            }
            for fut in as_completed(futures):
                lbl = futures[fut]
                try:
                    exit_codes[lbl] = fut.result()
                except Exception as e:
                    print(f"{lbl}: FAILED with {e}")
                    exit_codes[lbl] = -1

        all_ok = all(c == 0 for c in exit_codes.values())
        run_meta["status"] = "completed" if all_ok else "partial"
        run_meta["partition_exit_codes"] = exit_codes

        # Aggregate scoring (writes results.csv + summary.txt at run_dir)
        try:
            from score_run import score_partitioned_run
            score_partitioned_run(run_dir, task_meta)
        except Exception as e:
            print(f"Note: aggregate scoring skipped/failed: {e}")

    run_meta["finished"] = datetime.now().isoformat()
    with open(run_dir / "run.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    return run_dir


def _setup_partition(
    *,
    run_dir: Path,
    task_name: str,
    task_meta: dict,
    tools: list[str],
    partition_idx: int,
    n_partitions: int,
    strategy_dir: Path,
    bashrc_path: Path,
    seed: int | None,
    few_shot_per_class: int,
    from_source: bool,
) -> list[dict]:
    """Create one partition's strategy workspace + bashrc."""
    strategy_dir.mkdir(parents=True, exist_ok=True)

    if from_source:
        examples_index = populate_few_shot_from_source(
            task_meta=task_meta,
            strategy_dir=strategy_dir,
            seed=seed if seed is not None else 0,
            per_class=few_shot_per_class,
            test_keep_fields=task_meta.get("test_keep_fields"),
        )
    else:
        examples_index = populate_few_shot(
            task_name, strategy_dir, test_keep_fields=task_meta.get("test_keep_fields"),
        )

    generate_readme(task_meta, tools, examples_index, strategy_dir.parent)
    (strategy_dir / "STRATEGY.md").write_text(
        "# Strategy\n\n<!-- Write your classification strategy here -->\n"
    )

    # Partition run.json (for inspection/debug)
    part_meta = {
        "task": task_name,
        "partition_idx": partition_idx,
        "n_partitions": n_partitions,
        "seed": seed,
        "few_shot_per_class": few_shot_per_class,
    }
    with open(strategy_dir.parent / "partition.json", "w") as f:
        json.dump(part_meta, f, indent=2)

    # agent.bashrc — per-partition; AGENT_RUN_DIR points at the partition dir
    _write_bashrc(
        bashrc_path=bashrc_path,
        run_dir=strategy_dir.parent,
        task_name=task_name,
        run_id=run_dir.name.replace("run-", ""),
        extra_exports={
            "AGENT_PARTITION_INDEX": str(partition_idx),
            "AGENT_N_PARTITIONS": str(n_partitions),
        },
    )
    return examples_index


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
    run_parser.add_argument(
        "--n-strategies", type=int, default=1,
        help="Run N independent strategy agents on disjoint test partitions "
             "(cross-val-style). N=1 uses the legacy single-strategy layout.",
    )
    run_parser.add_argument(
        "--strategy-seed-base", type=int, default=0,
        help="Partition k's few-shot is sampled with seed = strategy_seed_base + k "
             "(only used when --n-strategies > 1).",
    )
    run_parser.add_argument(
        "--few-shot-per-class", type=int, default=None,
        help="Few-shot size per class per strategy. Default: inherit from metadata.",
    )

    status_parser = sub.add_parser("status", help="Show run status")
    status_parser.add_argument("run_id", nargs="?", help="Filter by run ID")

    args = parser.parse_args()

    if args.command == "init":
        init()
    elif args.command == "run":
        tools = [t.strip() for t in args.tools.split(",") if t.strip()]
        create_run(
            args.task_name, args.description, tools,
            n_strategies=args.n_strategies,
            strategy_seed_base=args.strategy_seed_base,
            few_shot_per_class=args.few_shot_per_class,
        )
    elif args.command == "status":
        show_status(args.run_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
