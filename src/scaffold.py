#!/usr/bin/env python3
"""Agent scaffold orchestrator.

Manages agent runs: initializes directories, launches strategy agents,
and coordinates test evaluation.

Usage:
    python scaffold.py init
    python scaffold.py run <task_name> [--description <desc>]
    python scaffold.py human-ui --task <task_name>
    python scaffold.py status [<run_id>]
"""

import argparse
import csv
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from agent_backend import (
    VALID_AGENT_BACKENDS,
    build_agent_launch_spec,
    get_agent_backend,
    load_bash_exports,
    prepare_codex_home,
    supports_add_dirs,
)


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


def make_run_id() -> str:
    """Return a timestamped run id with sub-second precision to avoid collisions."""
    return datetime.now().strftime("%Y%m%d-%H%M%S-%f")


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


def _copy_optional_sidecars(src_json: Path, dst_dir: Path) -> None:
    for suffix in (".npy", ".logits.npz", ".sae.npz"):
        src = src_json.with_suffix(suffix)
        if src.exists():
            shutil.copy2(src, dst_dir / src.name)


def populate_few_shot(
    task_name: str,
    strategy_dir: Path,
    *,
    seed: int,
    per_class: int,
    label_map: dict,
    test_keep_fields: list[str] | None = None,
) -> list[dict]:
    """Sample a balanced few-shot subset from the cached data/<task>/few-shot/ pool.

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

    from ingest_cot_proxy import sample_balanced

    picked = sample_balanced(src_dir, per_class, seed, label_map)
    for json_file, data in picked:
        filtered = _apply_whitelist(data, test_keep_fields)
        with open(dst_dir / json_file.name, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)
        index.append({
            "id": json_file.stem,
            "label": data.get("label", ""),
            "path": f"few-shot/{json_file.name}",
        })
        _copy_optional_sidecars(json_file, dst_dir)

    _write_examples_csv(strategy_dir, index)

    wl = f" (whitelisted to {len(test_keep_fields)} fields + label)" if test_keep_fields else ""
    print(f"Sampled {len(index)} cached few-shot examples from {src_dir} (seed={seed}){wl}")
    return index


def populate_few_shot_from_source(
    task_meta: dict,
    strategy_dir: Path,
    seed: int,
    per_class: int,
    test_keep_fields: list[str] | None,
) -> list[dict]:
    """Backward-compatible wrapper: sample from the cached data pool, not raw source."""
    raw_label_map = task_meta.get("label_map", {})
    def _key(k):
        try:
            return int(k)
        except (ValueError, TypeError): return k
    label_map = {_key(k): int(v) for k, v in raw_label_map.items()}
    data_task = task_meta.get("data_task", task_meta.get("name"))
    return populate_few_shot(
        str(data_task),
        strategy_dir,
        seed=seed,
        per_class=per_class,
        label_map=label_map,
        test_keep_fields=test_keep_fields,
    )


def _write_examples_csv(strategy_dir: Path, index: list[dict]) -> None:
    with open(strategy_dir / "Examples.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "label", "path"])
        writer.writeheader()
        writer.writerows(index)


def _load_tool_readme_description(tool_name: str) -> str | None:
    """Load a tool's README blurb from its implementation, if it exposes one."""
    tool_path = ROOT / "src" / "tools" / f"{tool_name}.py"
    if not tool_path.exists():
        return None

    tools_dir = str(tool_path.parent)
    added_path = False
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
        added_path = True

    try:
        spec = importlib.util.spec_from_file_location(f"tool_{tool_name}", tool_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        factory = getattr(module, "get_readme_description", None)
        if callable(factory):
            return factory()
        return None
    finally:
        if added_path:
            try:
                sys.path.remove(tools_dir)
            except ValueError:
                pass


TOOL_DESCRIPTIONS = {
    # Key = tool name. Value = multi-line markdown blurb rendered into strategy/README.md
    # under "## Research Tools". Both the strategy agent and the test agent read that
    # README, so each blurb should cover both scopes (what each agent may query).
    "top_10_logits": """### `top_10_logits <example_id> <token_position>`

Print the top-10 tokens and their logprob values at `token_position` within
the example's chain-of-thought, read from a precomputed sidecar
(`<example_id>.logits.npz` next to the example's JSON).

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

**Output**
- Writes `top_10_logits_<n>.csv` in the current directory with columns
  `example_id,token_position,rank,token,logprob`.
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
- Entropy is computed over only the top-10 logprobs (softmax restricted to
  those 10 values), not over the full vocabulary. It is therefore an
  underestimate of true entropy but preserves the relative ordering
  (more peaked vs more spread) which is what matters for this signal.

**Output**
- Writes `top10_entropy_<n>.csv` in the current directory with columns
  `example_id,token_position,top10_entropy`.
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
- Prints `next_token:` and `details:`.
- Writes `force_<n>.csv` in the current directory with columns
  `example_id,token_position,forced_text,next_token,next_token_logprob,top_10_logprobs_json`.
""".strip(),
    "hedging-detector": """### `hedging-detector <example_id>`

Score each sentence of the example's CoT prefix for **epistemic hedging**
via a fresh **claude-opus-4-7** subagent (extended thinking on). Scores are
in `[0, 1]`: `0.0` = committed / assertive, `1.0` = heavily hedged /
self-doubting. A 5-point centered moving average gives a hedging trajectory
across the CoT.

**Scope**
- **Strategy agent:** any few-shot `<example_id>`.
- **Test agent:** only its assigned `AGENT_EXAMPLE_ID`.

**Rate limit**
- **Once per sample.** A second call with the same `<example_id>` returns
  the cached `hedging_<example_id>.json` (printed as `status: cache-hit`).

**Output contract.** Prints a `status:` line plus summary stats and writes
`hedging_<example_id>.json` in the current directory. The JSON contains
per-sentence scores + rationales, the smoothed trajectory, overall mean,
and the index of the most-hedged sentence.
""".strip(),
    "hedging-detector-all": """### `hedging-detector-all`

Run `hedging-detector` **in parallel** over every few-shot example (≤8
concurrent claude-opus-4-7 calls). Writes per-example `hedging_<id>.json`
files (cache hits honored) plus a `hedging_all_summary.json` with per-label
mean/stdev/min/max of the overall hedging score.

**Scope**
- **Strategy agent only.** Invoking from a test agent fails.

**Rate limit**
- **Once per strategy run.** On success a lock is written to
  `.subagent_locks/hedging_detector_all.lock`; subsequent invocations refuse.
""".strip(),
    "repetition-mapper": """### `repetition-mapper <example_id>`

Cluster sentences that **restate or re-derive the same claim** in the
example's CoT, via a fresh **claude-opus-4-7** subagent (extended thinking
on). Returns clusters (each with a label, the sentence indices it spans,
and a restatement count) plus the "longest chain" — the cluster whose
restatements are spread furthest apart in the CoT. Singleton clusters are
excluded.

**Scope**
- **Strategy agent:** any few-shot `<example_id>`.
- **Test agent:** only its assigned `AGENT_EXAMPLE_ID`.

**Rate limit**
- **Once per sample.** A second call with the same `<example_id>` returns
  the cached `repetition_<example_id>.json`.

**Output contract.** Prints a `status:` line plus summary stats and writes
`repetition_<example_id>.json` in the current directory.
""".strip(),
    "repetition-mapper-all": """### `repetition-mapper-all`

Run `repetition-mapper` **in parallel** over every few-shot example (≤8
concurrent claude-opus-4-7 calls). Writes per-example `repetition_<id>.json`
files (cache hits honored) plus a `repetition_all_summary.json` with
per-label stats on cluster count and longest-chain span.

**Scope**
- **Strategy agent only.**

**Rate limit**
- **Once per strategy run.** Lock: `.subagent_locks/repetition_mapper_all.lock`.
""".strip(),
    "sae": """### `sae` — SAE feature inspection

Three subcommands for exploring a labelled **BatchTopK SAE (width 65,536,
k=80, trainer_2)** trained on Qwen3-32B `resid_post_layer_32`. Features
have natural-language labels (19,970 of 65,536 are labelled; the rest
print as `(unlabeled)`).

#### `sae search <query> [--n N]`
Full-text search over feature labels. `<query>` is any keywords (e.g.
`hedging uncertain`, `let me think`, `heterocyclic`). Returns up to `N`
features whose labels contain the most query words, ranked by overlap.
Writes `sae_search_<query_slug>.csv` (columns: `feature_id, score, label`)
in the current directory and prints the top matches to stdout.
**Search results are stable across paraphrases** — running the same
concept with different synonyms ("conclusion final answer" →
"concluding summary complete" → "wrap up finalize") rarely surfaces
new features. Use the strongest concept word once, look at the top
10–20 results, then move on. Better yield from `top-features` on a
specific few-shot example with the property you care about.

#### `sae top-features <example_id> [--n N]`
Top `N` SAE features active on `<example_id>`, sorted by max activation
across the CoT. Writes `sae_top_features_<example_id>.csv` (columns:
`feature_id, max_activation, peak_token_pos, label`) and prints the table
to stdout. `peak_token_pos` is **CoT-relative** — the same convention
other tools use, so you can cross-reference.

#### `sae feature <feature_id>`
Shows how a single feature activates across **every few-shot example in
your workspace**. Writes `sae_feature_<feature_id>.csv` (columns:
`example_id, label, max_activation, peak_token_pos`) and prints the table.
Use this to check whether a feature discriminates label=yes vs label=no.

**Scope**
- **Strategy agent:** may call all three subcommands on any few-shot `<example_id>`.
- **Test agent:** `top-features` only on its own `AGENT_EXAMPLE_ID`; `search` and `feature` unrestricted but rarely useful.

**Output format.** All three write a CSV in the current directory and
print a human-readable summary. CSVs overwrite on repeat (filename is
keyed by query / fid / example id, not auto-incremented).
""".strip(),
    "few-shot-diff": """### `few-shot-diff`

Ask a fresh **claude-opus-4-7** subagent (extended thinking on) to identify
**ranked distinguishing features** between two blind-labelled groups of
your few-shot examples. The few-shot set is split by ground-truth label into
"Group 1" and "Group 2"; the subagent is given neither the task description
nor the label meanings, and is told to characterise the groups empirically.
The goal is to surface signals you might have missed or overweighted because
you had the label in front of you.

**Scope**
- **Strategy agent only.**

**Rate limit**
- **Once per strategy run.** Lock: `.subagent_locks/few_shot_diff.lock`.

**Output contract.** Writes `few_shot_diff.json` containing a ranked list of
features (description, confidence in `[0, 1]`, per-group example-id evidence)
plus a one-paragraph summary, and prints the group → label mapping so you
can interpret the result.
""".strip(),
    "word-stats": """### `word-stats` — text statistics over the few-shot

Four subcommands for surfacing surface-level lexical signals
(complementary to the SAE tool, which surfaces learned features).
Tokenisation is lowercase, whitespace-split (`\\b\\w+\\b`); n-grams cover
unigrams through trigrams. Statistical scoring uses **Monroe et al. 2008
log-odds with informative Dirichlet prior** built from the combined
yes+no few-shot corpus — robust to stopwords and rare-term flukes.

#### `word-stats count <sample_id> <w1> [<w2> …]`
Count one or more terms in a specific example. Whole-word match for
single tokens, substring (non-overlapping, left-to-right) for multi-word
phrases. Useful for sanity-checking specific candidate signals.
Writes `word_stats_count_<sample_id>.csv`.
- Strategy agent: any few-shot `<sample_id>`.
- Test agent: only its assigned `AGENT_EXAMPLE_ID`.

#### `word-stats tf-idf`
Find the n-grams that most distinguish yes-labelled from no-labelled
examples in the few-shot. Returns top 20 yes-distinctive and top 20
no-distinctive terms, each with the log-odds z-score, raw counts in
each class, and the **number of distinct examples** in which the term
appears (so you can tell signal from a single-example fluke). Writes
`word_stats_tfidf_yes.csv` and `word_stats_tfidf_no.csv`.
- **Strategy agent only.**

#### `word-stats compare <sample_id>`
For a specific example, show which n-grams *in that example* are most
distinctive (positive or negative) relative to the few-shot's opposite
class. Top 20 by absolute z-score, signed. Useful at test time when you
want to know which terms in the test example actually carry signal,
rather than guessing which to count. Writes
`word_stats_compare_<sample_id>.csv`.
- Strategy agent: any few-shot `<sample_id>`.
- Test agent: only its assigned `AGENT_EXAMPLE_ID`.

#### `word-stats rank <concept> [--words "w1,w2,…"]`
Rank all few-shot examples by how often a concept appears. By default,
expands `<concept>` into 15-30 keywords/phrases via a fresh **claude-sonnet-4-6**
call (cached locally per concept), then matches each keyword against
each example. Returns: the keyword list at the top of the output (so
you can audit it), then a ranked table with per-example total hit count
+ per-keyword breakdown + true label. Pass `--words "w1,w2,…"` to skip
the LLM expansion and use an exact comma-separated list.
Writes `word_stats_rank_<concept_slug>.csv`.
- **Strategy agent only.**

**Tuning knobs (env vars)**
- `WORD_STATS_ALPHA0`: total prior strength for the Dirichlet prior
  (default 100; smaller = data dominates more, larger = stronger
  smoothing of rare terms).
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
        desc = _load_tool_readme_description(name)
        if desc is None:
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

## The `run-tests` command
Running `run-tests` evaluates your current strategy against all held-out test examples in parallel. Each test example is given to an independent test agent that sees only the contents of this `strategy/` directory plus its own single test example. Call `run-tests` when STRATEGY.md is ready.

## Instructions
1. Study the few-shot JSONs in `few-shot/` to understand what distinguishes positive from negative examples.
2. Write a clear, concrete classification strategy in `STRATEGY.md`. Test agents follow it literally.
3. Optionally create supporting CSVs or notes in this directory; reference them from STRATEGY.md.
4. Run `run-tests` when ready.
"""
    (run_dir / "strategy" / "README.md").write_text(readme, encoding="utf-8")


def _write_bashrc(
    bashrc_path: Path,
    run_dir: Path,
    task_name: str,
    run_id: str,
    agent_backend: str,
    extra_exports: dict[str, str] | None = None,
) -> None:
    """Write the per-agent bash environment file sourced via BASH_ENV."""
    lines = [
        "# Auto-generated bash environment for agent run",
        f'export SCAFFOLD_ROOT="{ROOT.as_posix()}"',
        f'export AGENT_RUN_DIR="{run_dir.as_posix()}"',
        f'export AGENT_TASK="{task_name}"',
        f'export AGENT_RUN_ID="{run_id}"',
        f'export AGENT_BACKEND="{agent_backend}"',
        f'export PATH="{BIN_DIR.as_posix()}:$PATH"',
        f'export PYTHON="{Path(sys.executable).as_posix()}"',
    ]
    for k, v in (extra_exports or {}).items():
        lines.append(f'export {k}="{v}"')
    bashrc_path.write_text("\n".join(lines) + "\n")


def _parse_bashrc_exports(bashrc_path: Path, base_env: dict) -> dict:
    """Return the ``export KEY="VAL"`` vars from ``bashrc_path`` as a dict.

    Needed because Claude Code's Bash tool runs in zsh on macOS, which does
    not honor ``BASH_ENV`` — so the per-partition ``agent.bashrc`` we write
    never reaches the strategy/test agent's shell. By parsing those exports
    in Python and merging them into the subprocess env, we make the agent
    see ``AGENT_N_PARTITIONS``, ``AGENT_PARTITION_INDEX``, ``PATH=bin/:…``
    etc. regardless of which shell Claude Code picks.

    ``$VAR`` references are expanded against ``base_env`` plus the
    already-parsed exports, which matches what bash would do on source.
    """
    import re as _re
    exports: dict[str, str] = {}
    text = bashrc_path.read_text(encoding="utf-8")
    export_re = _re.compile(r'^\s*export\s+(\w+)=(.*?)\s*$')
    varref_re = _re.compile(r'\$(\w+)|\$\{(\w+)\}')
    for line in text.splitlines():
        m = export_re.match(line)
        if not m:
            continue
        key, raw = m.group(1), m.group(2)
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ('"', "'"):
            raw = raw[1:-1]
        def _sub(match):
            var = match.group(1) or match.group(2)
            return exports.get(var, base_env.get(var, ""))
        resolved = varref_re.sub(_sub, raw)
        exports[key] = resolved
    return exports


def _launch_strategy_agent(
    strategy_dir: Path,
    trace_base: Path,
    bashrc_path: Path,
    label: str = "",
) -> int:
    """Run one strategy-agent subprocess in strategy_dir. Writes
    trace_base.jsonl / .txt. Returns exit code.

    Safety nets on top of the claude subprocess:
      - Wall-clock timeout (``AGENT_STRATEGY_TIMEOUT_SEC``, default 2700s / 45
        min) — kills the agent if it hangs (e.g., stuck in a bad Bash call).
      - Fallback invocation of ``run-tests``: if the agent exited having
        written a non-empty STRATEGY.md but never created any ``test-*/``
        folders, run the test phase here so the partition is still scored.
    """
    strategy_prompt_path = PROMPTS_DIR / "strategy-agent.md"
    if not strategy_prompt_path.exists():
        print(f"Error: Strategy prompt not found at {strategy_prompt_path}")
        sys.exit(1)

    user_prompt = (
        "Read README.md in the current directory for the task brief, available tools, "
        "and workspace layout. Develop a classification strategy, write it to STRATEGY.md, "
        "test each enabled research tool at least once and look for creative relevant uses, "
        "and run `run-tests` when ready."
    )
    system_prompt = strategy_prompt_path.read_text(encoding="utf-8")
    env = load_bash_exports(bashrc_path, os.environ.copy())
    env["BASH_ENV"] = str(bashrc_path)
    env["AGENT_TYPE"] = "strategy"
    backend = get_agent_backend(env)
    launch = build_agent_launch_spec(
        backend=backend,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        add_dirs=[strategy_dir.parent, trace_base.parent] if supports_add_dirs(backend) else None,
        project_settings=project_settings if backend == "claude" else None,
    )

    tag = f" [{label}]" if label else ""
    print(f"Launching strategy agent{tag} with backend={backend}...")
    timeout_sec = int(os.environ.get("AGENT_STRATEGY_TIMEOUT_SEC", "2700"))
    posttest_grace = int(os.environ.get("AGENT_STRATEGY_POSTTEST_GRACE_SEC", "30"))

    # Auto-shutdown: poll for completion of run-tests (signaled by
    # results.csv being written into the partition root) and SIGTERM the
    # strategy agent ~30s after it lands. Otherwise agents commonly spend
    # several minutes after `run-tests` returns reading test-NNN/answer.txt,
    # results.csv, src/run_tests.py, etc. — wasted tokens and wall time.
    import threading as _threading
    import time as _time

    strategy_md = strategy_dir / "STRATEGY.md"
    results_csv = strategy_dir.parent / "results.csv"

    p = subprocess.Popen(
        launch.cmd, cwd=str(strategy_dir), env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8",
    )
    if launch.stdin_text is not None:
        p.stdin.write(launch.stdin_text)
    p.stdin.close()

    out_chunks: list[str] = []
    def _drain():
        assert p.stdout is not None
        for line in iter(p.stdout.readline, ""):
            out_chunks.append(line)
    drainer = _threading.Thread(target=_drain, daemon=True)
    drainer.start()

    start = _time.time()
    results_seen_at: float | None = None
    kill_reason: str | None = None
    while True:
        if p.poll() is not None:
            break
        if _time.time() - start > timeout_sec:
            p.terminate()
            kill_reason = "timeout"
            print(f"Strategy agent{tag}: TIMED OUT after {timeout_sec}s")
            break
        if (
            strategy_md.exists()
            and strategy_md.stat().st_size > 100
            and results_csv.exists()
        ):
            if results_seen_at is None:
                results_seen_at = _time.time()
            elif _time.time() - results_seen_at >= posttest_grace:
                p.terminate()
                kill_reason = "post-test-grace-elapsed"
                break
        _time.sleep(2)
    try:
        p.wait(timeout=15)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
    drainer.join(timeout=2)
    exit_code = p.returncode if p.returncode is not None else -1
    stdout = "".join(out_chunks)
    if kill_reason == "timeout":
        stdout += f"\n\n[strategy agent timed out after {timeout_sec}s]"
    elif kill_reason == "post-test-grace-elapsed":
        stdout += f"\n\n[strategy agent shut down {posttest_grace}s after results.csv landed]"

    from render_trace import write_trace_pair
    write_trace_pair(stdout, trace_base)

    # Fallback: agent wrote STRATEGY.md but never called `test`. Invoke
    # run-tests ourselves so the partition is still evaluated.
    strategy_md = strategy_dir / "STRATEGY.md"
    has_content = strategy_md.exists() and strategy_md.stat().st_size > 100
    part_dir = strategy_dir.parent
    already_tested = any(part_dir.glob("test-*"))
    if has_content and not already_tested:
        print(f"Strategy agent{tag}: STRATEGY.md written but `test` not called; "
              f"running fallback run-tests...")
        fallback_timeout = int(os.environ.get("AGENT_FALLBACK_TESTS_TIMEOUT_SEC", "1800"))
        fallback_env = os.environ.copy()
        fallback_env.update(_parse_bashrc_exports(bashrc_path, fallback_env))
        try:
            subprocess.run(
                [str(BIN_DIR / "run-tests")],
                cwd=str(strategy_dir),
                env=fallback_env,
                timeout=fallback_timeout,
            )
        except subprocess.TimeoutExpired:
            print(f"Strategy agent{tag}: fallback run-tests TIMED OUT after {fallback_timeout}s")
        except Exception as e:
            print(f"Strategy agent{tag}: fallback run-tests FAILED: {e}")

    print(f"Strategy agent{tag} finished (exit code {exit_code})")
    return exit_code


def create_run(
    task_name: str,
    description: str | None = None,
    tools: list[str] | None = None,
    n_strategies: int = 1,
    strategy_seed_base: int = 0,
    few_shot_per_class: int | None = None,
    agent_backend: str = "claude",
    codex_reasoning_effort: str | None = None,
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
        int(task_meta.get("strategy_few_shot_per_class", task_meta.get("few_shot_per_class", 5)))

    # Precompute SAE activations for few-shot and test examples (if .npy files exist).
    # Cached as .sae.npz alongside .npy; idempotent across runs.
    try:
        from tools.sae_encode import precompute_task
        precompute_task(ROOT, task_name)
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: SAE precompute failed: {e}")

    run_id = make_run_id()
    run_dir = RUNS_DIR / task_name / f"run-{run_id}"
    trace_dir = TRACES_DIR / task_name / f"run-{run_id}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "task": task_name,
        "run_id": run_id,
        "created": datetime.now().isoformat(),
        "status": "running",
        "agent_backend": agent_backend,
        "tools": tools,
        "n_strategies": n_strategies,
        "strategy_seed_base": strategy_seed_base,
        "few_shot_per_class": fspc,
        "codex_reasoning_effort": codex_reasoning_effort,
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
            agent_backend=agent_backend,
            codex_reasoning_effort=codex_reasoning_effort,
            seed=None, few_shot_per_class=fspc, from_source=False,
        )
        trace_base = trace_dir / "strategy-trace"
        code = _launch_strategy_agent(
            strategy_dir=run_dir / "strategy",
            trace_base=trace_base,
            bashrc_path=run_dir / "agent.bashrc",
        )
        run_meta["strategy_exit_code"] = code
        try:
            from score_run import score_run
            score_info = score_run(run_dir, task_meta)
            run_meta["agg_miss"] = score_info.get("agg_miss")
            run_meta["total_tests"] = score_info.get("total_tests")
        except Exception as e:
            print(f"Note: scoring skipped/failed: {e}")
            score_info = None
        if code == 0 and score_info and score_info.get("agg_miss", 1) == 0:
            run_meta["status"] = "completed"
        elif score_info and (
            score_info["aggregate"]["n"] > 0 or score_info.get("agg_miss", 0) > 0
        ):
            run_meta["status"] = "partial"
        else:
            run_meta["status"] = "failed"
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
                agent_backend=agent_backend,
                codex_reasoning_effort=codex_reasoning_effort,
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
            from score_run import score_run
            score_info = score_run(run_dir, task_meta)
            run_meta["agg_miss"] = score_info.get("agg_miss")
            run_meta["total_tests"] = score_info.get("total_tests")
            if score_info.get("agg_miss", 0) > 0:
                run_meta["status"] = "partial"
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
    agent_backend: str,
    codex_reasoning_effort: str | None,
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
        raw_label_map = task_meta.get("label_map", {})
        def _key(k):
            try:
                return int(k)
            except (ValueError, TypeError):
                return k
        label_map = {_key(k): int(v) for k, v in raw_label_map.items()}
        examples_index = populate_few_shot(
            str(task_meta.get("data_task", task_name)),
            strategy_dir,
            seed=seed if seed is not None else 0,
            per_class=few_shot_per_class,
            label_map=label_map,
            test_keep_fields=task_meta.get("test_keep_fields"),
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

    extra_exports = {
        "AGENT_PARTITION_INDEX": str(partition_idx),
        "AGENT_N_PARTITIONS": str(n_partitions),
        "AGENT_DATA_TASK": str(task_meta.get("data_task", task_name)),
    }
    if agent_backend == "codex":
        codex_home = prepare_codex_home(strategy_dir.parent / ".codex-home", os.environ)
        extra_exports["CODEX_HOME"] = codex_home.as_posix()
        if codex_reasoning_effort:
            extra_exports["CODEX_REASONING_EFFORT"] = codex_reasoning_effort

    # agent.bashrc — per-partition; AGENT_RUN_DIR points at the partition dir
    _write_bashrc(
        bashrc_path=bashrc_path,
        run_dir=strategy_dir.parent,
        task_name=task_name,
        run_id=run_dir.name.replace("run-", ""),
        agent_backend=agent_backend,
        extra_exports=extra_exports,
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
    run_parser.add_argument(
        "--agent-backend",
        choices=VALID_AGENT_BACKENDS,
        default="claude",
        help="Agent CLI backend to launch (default: claude).",
    )
    run_parser.add_argument(
        "--codex-reasoning-effort",
        choices=("low", "medium", "high", "xhigh"),
        help="Optional Codex reasoning effort override. Only applies when --agent-backend=codex.",
    )

    status_parser = sub.add_parser("status", help="Show run status")
    status_parser.add_argument("run_id", nargs="?", help="Filter by run ID")

    human_parser = sub.add_parser("human-ui", help="Launch the local human baseline UI")
    human_parser.add_argument("--task", help="Name of task folder in data/")
    human_parser.add_argument("--run-dir", help="Open an existing human run directory")
    human_parser.add_argument(
        "--tools",
        default="ask,top_10_logits,top10_entropy,force",
        help="Comma-separated list of tools to enable for a new run",
    )
    human_parser.add_argument("--description", help="Override task description for a new run")
    human_parser.add_argument(
        "--ood",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use OOD few-shot/test split rules (default: on)",
    )
    human_parser.add_argument("--host", default="127.0.0.1", help="Bind host for the UI")
    human_parser.add_argument("--port", type=int, default=8000, help="Bind port for the UI")

    args = parser.parse_args()

    if args.command == "init":
        init()
    elif args.command == "run":
        tools = [t.strip() for t in args.tools.split(",") if t.strip()]
        os.environ["AGENT_BACKEND"] = args.agent_backend
        create_run(
            args.task_name, args.description, tools,
            n_strategies=args.n_strategies,
            strategy_seed_base=args.strategy_seed_base,
            few_shot_per_class=args.few_shot_per_class,
            agent_backend=args.agent_backend,
            codex_reasoning_effort=args.codex_reasoning_effort,
        )
    elif args.command == "status":
        show_status(args.run_id)
    elif args.command == "human-ui":
        cmd = [sys.executable, str(ROOT / "src" / "human_ui.py")]
        if args.task:
            cmd.extend(["--task", args.task])
        if args.run_dir:
            cmd.extend(["--run-dir", args.run_dir])
        if args.tools is not None:
            cmd.extend(["--tools", args.tools])
        if args.description:
            cmd.extend(["--description", args.description])
        cmd.append("--ood" if args.ood else "--no-ood")
        cmd.extend(["--host", args.host, "--port", str(args.port)])
        raise SystemExit(subprocess.call(cmd, cwd=str(ROOT)))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
