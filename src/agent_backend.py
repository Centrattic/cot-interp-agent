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

def load_bash_exports(path: Path, base_env: dict[str, str] | None = None) -> dict[str, str]:
    """Parse simple `export KEY="VALUE"` lines from an agent bashrc into env."""
    env = dict(base_env or os.environ)
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        kv = line[len("export "):]
        if "=" not in kv:
            continue
        key, _, value = kv.partition("=")
        env[key.strip()] = value.strip().strip('"').strip("'")
    if "PATH" in env and "$PATH" in env["PATH"]:
        env["PATH"] = env["PATH"].replace("$PATH", os.environ.get("PATH", ""))
    return env

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
    # Claude Code doesn't walk up the cwd tree to find .claude/settings.json,
    # so when the agent runs from a deep run/test subdirectory the project's
    # peek-block hooks would be missed. Pass --settings explicitly.
    if project_settings is not None:
        cmd.extend(["--settings", str(project_settings)])
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
    project_settings: Path | None = None,
) -> AgentLaunchSpec:
    if backend == "claude":
        return AgentLaunchSpec(
            cmd=_build_claude_command(system_prompt, add_dirs, project_settings),
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
