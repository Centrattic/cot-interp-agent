"""Backend-specific agent launch helpers."""

from __future__ import annotations

import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path


VALID_AGENT_BACKENDS = ("claude", "codex")
CODEX_HOME_SEED_FILES = ("auth.json", "config.toml", "version.json", "installation_id")


@dataclass(frozen=True)
class AgentLaunchSpec:
    cmd: list[str]
    stdin_text: str | None


def get_agent_backend(env: dict[str, str] | None = None) -> str:
    env = env or os.environ
    backend = env.get("AGENT_BACKEND", "claude").strip().lower()
    if backend not in VALID_AGENT_BACKENDS:
        raise ValueError(
            f"AGENT_BACKEND={backend!r} invalid; expected one of {VALID_AGENT_BACKENDS}"
        )
    return backend


def supports_add_dirs(backend: str) -> bool:
    return backend in ("claude", "codex")


def resolve_codex_home(env: dict[str, str] | None = None) -> Path:
    env = env or os.environ
    raw = (env.get("CODEX_HOME") or "").strip()
    if raw:
        return Path(raw).expanduser()
    home = (env.get("HOME") or "").strip()
    if home:
        return Path(home).expanduser() / ".codex"
    return Path.home() / ".codex"


def prepare_codex_home(target_dir: Path, env: dict[str, str] | None = None) -> Path:
    """Seed a run-local CODEX_HOME so nested codex exec calls can authenticate
    even when the parent sandbox cannot read ~/.codex directly.
    """
    source_dir = resolve_codex_home(env)
    target_dir.mkdir(parents=True, exist_ok=True)
    if not source_dir.exists():
        return target_dir
    for name in CODEX_HOME_SEED_FILES:
        src = source_dir / name
        if src.exists():
            shutil.copy2(src, target_dir / name)
    return target_dir


def _build_claude_command(system_prompt: str, add_dirs: list[Path] | None) -> list[str]:
    cmd = [
        os.environ.get("CLAUDE_BIN", "claude"),
        "--print",
        "--dangerously-skip-permissions",
        "--output-format",
        "stream-json",
        "--verbose",
        "--system-prompt",
        system_prompt,
        "--allowed-tools",
        "Read,Write,Edit,Bash,Glob,Grep",
    ]
    for extra_dir in add_dirs or []:
        cmd.extend(["--add-dir", str(extra_dir)])
    return cmd


def _build_codex_command(prompt_text: str, add_dirs: list[Path] | None) -> AgentLaunchSpec:
    override = os.environ.get("CODEX_EXEC_CMD", "").strip()
    if override:
        cmd = shlex.split(override)
        stdin_text = prompt_text
    else:
        cmd = [
            os.environ.get("CODEX_BIN", "codex"),
            "exec",
            "--json",
            "--skip-git-repo-check",
            "--ephemeral",
            "--dangerously-bypass-approvals-and-sandbox",
            "-",
        ]
        for extra_dir in add_dirs or []:
            cmd.extend(["--add-dir", str(extra_dir)])
        model = os.environ.get("CODEX_MODEL", "").strip()
        if model:
            cmd.extend(["-m", model])
        reasoning_effort = os.environ.get("CODEX_REASONING_EFFORT", "").strip()
        if reasoning_effort:
            cmd.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
        stdin_text = prompt_text
    return AgentLaunchSpec(cmd=cmd, stdin_text=stdin_text)


def build_agent_launch_spec(
    *,
    backend: str,
    system_prompt: str,
    user_prompt: str,
    add_dirs: list[Path] | None = None,
) -> AgentLaunchSpec:
    if backend == "claude":
        return AgentLaunchSpec(
            cmd=_build_claude_command(system_prompt, add_dirs),
            stdin_text=user_prompt,
        )
    if backend == "codex":
        combined = (
            f"{system_prompt}\n\n"
            "Follow the system instructions above exactly.\n\n"
            f"{user_prompt}"
        )
        return _build_codex_command(combined, add_dirs)
    raise ValueError(f"unsupported backend: {backend}")
