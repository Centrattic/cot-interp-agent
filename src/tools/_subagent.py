"""Shared infra for Codex-backed helper-agent tools.

All helper-agent tools in this repo (hedging-detector, repetition-mapper,
few-shot-diff, plus their _all variants) share the same mechanics:
  - A nested `codex exec` call, authenticated through CODEX_HOME.
  - Structured output via Codex's `--output-schema`.
  - Once-per-sample or once-per-run semantics enforced by cache files + locks.

This module centralises the Codex call pattern, content parsing helpers
(split_content, sentencize), and lock/parallel helpers.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from _common import fail


class SubagentError(Exception):
    """Raised by call_structured() on any unrecoverable API / response issue.

    Distinct from fail() (which calls sys.exit) so the exception propagates
    cleanly through ThreadPoolExecutor in the _all batch tools. Top-level
    main() entry points should catch SubagentError and re-raise via fail().
    """


MODEL = os.environ.get("CODEX_SUBAGENT_MODEL", "gpt-5.5")
SONNET_MODEL = MODEL
DEFAULT_THINKING_EFFORT = os.environ.get("CODEX_SUBAGENT_REASONING_EFFORT", "high")
DEFAULT_MAX_TOKENS = 32000
DEFAULT_PARALLELISM = 8
CONTENT_DELIMITER = "Chain-of-thought prefix:\n"
CODEX_HOME_SEED_FILES = ("auth.json", "config.toml", "version.json", "installation_id")


def _closed_schema(obj: Any) -> Any:
    if isinstance(obj, dict):
        out = {k: _closed_schema(v) for k, v in obj.items()}
        if out.get("type") == "object":
            out.setdefault("additionalProperties", False)
        return out
    if isinstance(obj, list):
        return [_closed_schema(v) for v in obj]
    return obj


def _schema_for_output(tool_input_schema: dict) -> dict:
    """Wrap a tool input schema as a final-response JSON schema.

    Codex `--output-schema` constrains the assistant's final answer, not a
    tool call. The callers already define the exact object shape they need as
    `tool_input_schema`, so we reuse it directly and set a title for easier
    debugging.
    """
    schema = _closed_schema(dict(tool_input_schema))
    schema.setdefault("title", "structured_tool_result")
    schema.setdefault("additionalProperties", False)
    return schema


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as e:
        raise SubagentError(f"Codex returned non-JSON output: {e}: {text[:500]!r}") from e
    if not isinstance(obj, dict):
        raise SubagentError(f"Codex returned {type(obj).__name__}, expected object.")
    return obj


def _extract_result_from_jsonl(stdout: str) -> str | None:
    final: str | None = None
    for line in stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result" and isinstance(ev.get("result"), str):
            final = ev["result"]
    return final


def _codex_work_dir() -> Path:
    if os.environ.get("CODEX_HOME") or os.environ.get("AGENT_RUN_DIR"):
        base = Path.cwd() / ".codex-subagents"
    else:
        base = Path(tempfile.gettempdir()) / "cot-interp-agent-codex-subagents"
    base.mkdir(exist_ok=True)
    return base


def _ensure_codex_home(work_dir: Path, env: dict[str, str]) -> None:
    if env.get("CODEX_HOME"):
        Path(env["CODEX_HOME"]).mkdir(parents=True, exist_ok=True)
        return
    target = work_dir / "home"
    target.mkdir(parents=True, exist_ok=True)
    source = Path.home() / ".codex"
    if source.exists():
        for name in CODEX_HOME_SEED_FILES:
            src = source / name
            dst = target / name
            if src.exists() and not dst.exists():
                shutil.copy2(src, dst)
    env["CODEX_HOME"] = str(target)


def _call_codex_structured(
    *,
    system: str,
    user: str,
    tool_name: str,
    tool_description: str,
    tool_input_schema: dict,
    model: str = MODEL,
    thinking_effort: str = DEFAULT_THINKING_EFFORT,
) -> dict:
    work_dir = _codex_work_dir()
    stamp = f"{tool_name}_{os.getpid()}_{time.time_ns()}"
    schema_path = work_dir / f"{stamp}.schema.json"
    output_path = work_dir / f"{stamp}.out.json"
    schema_path.write_text(
        json.dumps(_schema_for_output(tool_input_schema), indent=2),
        encoding="utf-8",
    )

    prompt = (
        f"{system}\n\n"
        "You are being used as a structured helper for a research tool.\n"
        f"Logical tool name: {tool_name}\n"
        f"Logical tool description: {tool_description}\n\n"
        "Return exactly one JSON object matching the provided output schema. "
        "Do not include markdown fences or commentary outside the JSON object.\n\n"
        f"{user}"
    )
    cmd = [
        os.environ.get("CODEX_BIN", "codex"),
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--ephemeral",
        "--sandbox",
        "read-only",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "-m",
        model,
        "-c",
        f'model_reasoning_effort="{thinking_effort}"',
        "-",
    ]
    env = os.environ.copy()
    _ensure_codex_home(work_dir, env)
    try:
        proc = subprocess.run(
            cmd,
            input=prompt,
            cwd=str(work_dir),
            env=env,
            text=True,
            encoding="utf-8",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=int(os.environ.get("CODEX_SUBAGENT_TIMEOUT_SEC", "900")),
        )
    except subprocess.TimeoutExpired as e:
        raise SubagentError(
            f"Codex helper timed out after {e.timeout}s for {tool_name}."
        ) from e

    if proc.returncode != 0:
        raise SubagentError(
            f"Codex helper failed for {tool_name} (exit {proc.returncode}): "
            f"{proc.stdout[-2000:]}"
        )

    if output_path.exists():
        return _parse_json_object(output_path.read_text(encoding="utf-8"))
    final = _extract_result_from_jsonl(proc.stdout)
    if final is None:
        raise SubagentError(
            f"Codex helper produced no final result for {tool_name}: "
            f"{proc.stdout[-2000:]}"
        )
    return _parse_json_object(final)


def call_structured(
    *,
    system: str,
    user: str,
    tool_name: str,
    tool_description: str,
    tool_input_schema: dict,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    thinking_effort: str = DEFAULT_THINKING_EFFORT,
) -> dict:
    """Run a Codex helper with reasoning enabled and return structured JSON."""
    del max_tokens  # Codex CLI does not expose a direct max-token flag here.
    return _call_codex_structured(
        system=system,
        user=user,
        tool_name=tool_name,
        tool_description=tool_description,
        tool_input_schema=tool_input_schema,
        model=MODEL,
        thinking_effort=thinking_effort,
    )


def call_simple(
    *,
    system: str,
    user: str,
    tool_name: str,
    tool_description: str,
    tool_input_schema: dict,
    model: str = MODEL,
    max_tokens: int = 8192,
) -> dict:
    """Run a Codex helper for smaller structured tasks.

    This still uses reasoning mode; the name is kept for call-site
    compatibility with the older implementation.
    """
    del max_tokens
    return _call_codex_structured(
        system=system,
        user=user,
        tool_name=tool_name,
        tool_description=tool_description,
        tool_input_schema=tool_input_schema,
        model=model or MODEL,
        thinking_effort=DEFAULT_THINKING_EFFORT,
    )


def split_content(content: str) -> tuple[str, str]:
    """Split an example's `content` into (question_block, cot_prefix).

    Format produced by ingestion: ``"Question: ...\\n\\nChain-of-thought prefix:\\n<CoT>"``.
    If the delimiter is absent, returns (content, "").
    """
    idx = content.find(CONTENT_DELIMITER)
    if idx == -1:
        return content, ""
    question_block = content[:idx].strip()
    cot = content[idx + len(CONTENT_DELIMITER):]
    return question_block, cot


def get_cot_prefix(example: dict) -> str:
    """Return just the CoT prefix from an example record.

    Handles both data layouts the scaffold may hand an agent:
      - Ingested (``ingest_cot_proxy.py`` pipeline): single ``content`` field
        with the CoT embedded after ``"Chain-of-thought prefix:\\n"``.
      - Raw (``cot-proxy-tasks`` format as copied into the data dir): a
        top-level ``cot_prefix`` field.
      - Task-specific proxy records whose reasoning text is stored under
        fields such as ``cot_text``, ``thinking``, ``chain_of_thought``, or
        ``cot_content``.

    Prefer explicit CoT/reasoning fields when present so text-analysis tools
    do not silently operate on an empty string for task-specific JSON schemas.
    """
    for key in (
        "cot_prefix",
        "cot_text",
        "thinking",
        "chain_of_thought",
        "cot_content",
        "reasoning",
        "rationale",
    ):
        value = example.get(key)
        if isinstance(value, str) and value.strip():
            return value
        # Some pipelines store reasoning as a list of structured chunks, e.g.
        # [{"type": "reasoning.text", "text": "...", "format": "unknown", "index": 0}].
        # Concatenate the .text fields in declared order.
        if isinstance(value, list) and value:
            chunks: list[str] = []
            for chunk in value:
                if isinstance(chunk, dict):
                    text = chunk.get("text") or chunk.get("content")
                    if isinstance(text, str) and text.strip():
                        chunks.append(text)
                elif isinstance(chunk, str) and chunk.strip():
                    chunks.append(chunk)
            if chunks:
                return "\n".join(chunks)

    messages = example.get("messages")
    if isinstance(messages, list) and messages:
        parts: list[str] = []
        for m in messages:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", "")).strip() or "msg"
            content = m.get("content")
            if isinstance(content, list):
                content = "".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            content = str(content or "").strip()
            if content:
                parts.append(f"{role}: {content}")
        if parts:
            return "\n\n".join(parts)

    content = example.get("content") or ""
    _question, cot = split_content(content)
    if cot.strip():
        return cot
    return content if isinstance(content, str) else ""


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def sentencize(text: str) -> list[str]:
    """Split text into sentences via regex.

    Splits on sentence-terminal punctuation followed by whitespace and an
    uppercase letter or digit. Decimals like ``0.5`` are preserved because the
    lookahead requires a capital/digit *after* the space.
    """
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_SPLIT.split(text)
    return [p.strip() for p in parts if p.strip()]


def moving_avg(xs: list[float], window: int = 5) -> list[float]:
    """Centered moving average over a list of floats."""
    if not xs:
        return []
    half = window // 2
    out = []
    for i in range(len(xs)):
        lo = max(0, i - half)
        hi = min(len(xs), i + half + 1)
        chunk = xs[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def lock_dir() -> Path:
    d = Path.cwd() / ".subagent_locks"
    d.mkdir(exist_ok=True)
    return d


def assert_lock_free(lock_name: str, friendly: str) -> None:
    """Refuse to run if the named lock file exists in the current directory."""
    lock = lock_dir() / f"{lock_name}.lock"
    if lock.exists():
        fail(
            f"{friendly} has already been run in this strategy run "
            f"(lock file: {lock}). Each once-per-run tool may be invoked only once.",
            code=6,
        )


def write_lock(lock_name: str) -> None:
    (lock_dir() / f"{lock_name}.lock").write_text("")


def run_parallel(
    fn: Callable[[Any], Any],
    items: list[Any],
    *,
    max_workers: int = DEFAULT_PARALLELISM,
) -> list[tuple[Any, Any, str | None]]:
    """Run ``fn(item)`` for each item concurrently.

    Returns a list of (item, result, error_str) in completion order. If fn
    raised, result is None and error_str holds the exception text.
    """
    results: list[tuple[Any, Any, str | None]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for fut in as_completed(futs):
            it = futs[fut]
            try:
                results.append((it, fut.result(), None))
            except Exception as e:
                results.append((it, None, f"{type(e).__name__}: {e}"))
    return results


def assert_strategy_only(env: dict, friendly: str) -> None:
    """Refuse to run when invoked from a test agent context."""
    if env["AGENT_TYPE"] != "strategy":
        fail(
            f"{friendly} is available only to the strategy agent "
            f"(AGENT_TYPE={env['AGENT_TYPE']!r}).",
            code=7,
        )


def check_example_scope(env: dict, example_id: str) -> None:
    """Test agents may only query their assigned example; strategy agents are unrestricted."""
    if env["AGENT_TYPE"] != "test":
        return
    assigned = os.environ.get("AGENT_EXAMPLE_ID")
    if assigned and assigned != example_id:
        fail(
            f"test agent can only query its own example "
            f"(AGENT_EXAMPLE_ID={assigned!r}), not {example_id!r}",
            code=7,
        )
