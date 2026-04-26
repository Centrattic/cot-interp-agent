"""Render agent stdout into a readable text log.

Claude's stream-json format emits one JSON object per line representing a
discrete event:
  - system init (session id, model, tools)
  - user messages (original prompt + tool-result replies)
  - assistant messages (text + tool_use blocks)
  - result (final summary with cost / turns / exit reason)

When stdout is not valid JSONL (for example a different backend), this module
still preserves it line-by-line in the rendered `.txt` trace.
"""
from __future__ import annotations

import json
from pathlib import Path


def _text_of(content) -> str:
    """Content blocks are either a string or a list of typed blocks. Flatten to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if not isinstance(block, dict):
                parts.append(str(block))
                continue
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                parts.append(
                    f"[tool_use name={block.get('name')!r} id={block.get('id')!r}]\n"
                    f"  input: {json.dumps(block.get('input', {}), ensure_ascii=False)[:2000]}"
                )
            elif btype == "tool_result":
                tc = block.get("content")
                if isinstance(tc, list):
                    tc = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in tc)
                parts.append(
                    f"[tool_result for={block.get('tool_use_id')!r}"
                    + (" is_error=True" if block.get("is_error") else "")
                    + f"]\n{str(tc)[:4000]}"
                )
            elif btype == "thinking":
                parts.append(f"[thinking]\n{block.get('thinking','')}")
            else:
                parts.append(f"[{btype}] {json.dumps(block, ensure_ascii=False)[:500]}")
        return "\n".join(parts)
    return str(content)


def render_jsonl_trace(jsonl_text: str) -> str:
    """Return a human-readable text rendering of a stream-json trace."""
    out: list[str] = []
    for i, line in enumerate(jsonl_text.splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            out.append(f"# [line {i}] non-json: {line[:200]}")
            continue
        etype = ev.get("type")
        if etype == "system":
            sub = ev.get("subtype", "system")
            out.append(f"=== system/{sub} ===")
            # Keep the init details compact.
            keys = ("session_id", "model", "cwd", "tools", "permissionMode")
            for k in keys:
                if k in ev:
                    v = ev[k]
                    if isinstance(v, list):
                        v = ", ".join(v)
                    out.append(f"  {k}: {v}")
        elif etype == "user":
            msg = ev.get("message", {})
            out.append("=== user ===")
            out.append(_text_of(msg.get("content", "")))
        elif etype == "assistant":
            msg = ev.get("message", {})
            out.append("=== assistant ===")
            out.append(_text_of(msg.get("content", "")))
            usage = msg.get("usage")
            if usage:
                out.append(
                    "  [usage] in={} out={} cache_read={} cache_create={}".format(
                        usage.get("input_tokens"),
                        usage.get("output_tokens"),
                        usage.get("cache_read_input_tokens"),
                        usage.get("cache_creation_input_tokens"),
                    )
                )
        elif etype == "result":
            out.append("=== result ===")
            for k in ("subtype", "is_error", "duration_ms", "num_turns",
                     "total_cost_usd", "session_id"):
                if k in ev:
                    out.append(f"  {k}: {ev[k]}")
            if "result" in ev:
                out.append(f"  final_text: {str(ev['result'])[:2000]}")
        else:
            out.append(f"=== {etype or 'unknown'} ===")
            out.append(json.dumps(ev, ensure_ascii=False)[:1000])
        out.append("")
    return "\n".join(out)


def write_trace_pair(jsonl_text: str, base_path: Path) -> None:
    """Given raw JSONL stdout, write `<base>.jsonl` and a rendered `<base>.txt`."""
    base_path.with_suffix(".jsonl").write_text(jsonl_text, encoding="utf-8")
    base_path.with_suffix(".txt").write_text(render_jsonl_trace(jsonl_text), encoding="utf-8")
