from __future__ import annotations

import json
from pathlib import Path


def _load_base_prompt(prompts_dir: Path, prompt_name: str) -> str:
    prompt_path = prompts_dir / prompt_name
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt not found: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8").rstrip()


def _strategy_command_section(tools: list[str]) -> str:
    lines = [
        "## Available Commands",
        "",
        "- `run-tests` — Evaluate your strategy against held-out test examples. Call this exactly once, when your strategy is finalized.",
        "- Run `run-tests` synchronously in the foreground. Do not background it, do not pass `run_in_background: true`, and do not monitor it with `tail -f` / `tail --follow`.",
    ]
    for tool_name in tools:
        lines.append(
            f"- `{tool_name}` — Enabled for this run. Read `README.md` for authoritative usage, limits, and scope."
        )
    lines.append("- Any other research tools listed in `README.md` are also available exactly as documented there.")
    lines.append("")
    lines.append(
        "When invoking a research tool, `<example_id>` refers to the filename stem from `Examples.csv` "
        "(e.g. `ex_001`). You (the strategy agent) may run tools against any few-shot example."
    )
    return "\n".join(lines)


def build_strategy_system_prompt(prompts_dir: Path, tools: list[str]) -> str:
    base = _load_base_prompt(prompts_dir, "strategy-agent.md")
    return f"{base}\n\n{_strategy_command_section(tools)}\n"


def _load_tools_for_run(run_dir: Path) -> list[str]:
    run_meta_path = run_dir / "run.json"
    if not run_meta_path.exists():
        run_meta_path = run_dir.parent / "run.json"
    if not run_meta_path.exists():
        return []
    try:
        run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    tools = run_meta.get("tools", [])
    return [tool for tool in tools if isinstance(tool, str) and tool]


def build_test_system_prompt(prompts_dir: Path, run_dir: Path) -> str:
    base = _load_base_prompt(prompts_dir, "test-agent.md")
    tools = _load_tools_for_run(run_dir)
    if not tools:
        return base + "\n"
    lines = [
        "## Enabled Research Tools",
        "",
        "The following research tools are enabled for this run. Read `strategy/README.md` for authoritative usage, limits, and scope:",
    ]
    lines.extend(f"- `{tool_name}`" for tool_name in tools)
    return f"{base}\n\n" + "\n".join(lines) + "\n"
