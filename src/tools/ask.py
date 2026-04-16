"""`ask` tool — ask a short follow-up question about an example via OpenRouter.

Usage:
    ask <example_id> "<question>"

Limits (enforced with the Qwen tokenizer):
    - Question must be ≤20 tokens
    - Response is truncated to the first 5 tokens (thinking/reasoning excluded)

Behavior:
    - Strategy agent can query any few-shot example.
    - Test agent can only query its own assigned example (AGENT_EXAMPLE_ID).
    - On success, writes ask_<n>.json into the current directory (the
      strategy/ or test/<n>/ folder) and prints a status summary.
    - On failure, prints a status line with the reason; no file written.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

from _common import fail, get_env, load_example


MAX_QUESTION_TOKENS = 20
MAX_RESPONSE_TOKENS = 5

DEFAULT_TOKENIZER = "Qwen/Qwen3-32B"
DEFAULT_MODEL = "qwen/qwen3-32b"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def get_tokenizer():
    """Load the Qwen tokenizer once per process."""
    from tokenizers import Tokenizer  # lazy import

    name = os.environ.get("QWEN_TOKENIZER", DEFAULT_TOKENIZER)
    return Tokenizer.from_pretrained(name)


def next_output_path(cwd: Path) -> Path:
    n = 1
    while True:
        p = cwd / f"ask_{n}.json"
        if not p.exists():
            return p
        n += 1


def check_test_agent_scope(env: dict, example_id: str) -> None:
    """Test agents can only ask about the example assigned to them."""
    if env["AGENT_TYPE"] != "test":
        return
    assigned = os.environ.get("AGENT_EXAMPLE_ID")
    if assigned and assigned != example_id:
        fail(
            f"test agent can only query its own example "
            f"(AGENT_EXAMPLE_ID={assigned!r}), not {example_id!r}"
        )


def call_openrouter(example_content: str, question: str, model: str) -> dict:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        fail("OPENROUTER_API_KEY not set in environment")

    system = (
        "You will be shown an example and a short follow-up question about it. "
        "Answer the question directly and concisely."
    )
    user = f"Example:\n{example_content}\n\nQuestion: {question}"

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # Route exclusively through DeepInfra so behavior is reproducible
        # across runs (other providers handle thinking / tokenization slightly
        # differently).
        "provider": {"only": ["deepinfra"], "allow_fallbacks": False},
        # Ask OpenRouter to disable + hide reasoning. `enabled: false` isn't
        # honored by every provider for Qwen3, so we also give max_tokens
        # enough headroom that any reasoning can't crowd out the answer.
        "reasoning": {"enabled": False, "exclude": True},
        "max_tokens": 1024,
    }
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:500]
        fail(f"OpenRouter HTTP {e.code}: {detail}", code=4)
    except urllib.error.URLError as e:
        fail(f"OpenRouter request failed: {e}", code=4)


def extract_content(resp: dict) -> str:
    """Pull the assistant's text reply, ignoring any reasoning field."""
    try:
        msg = resp["choices"][0]["message"]
    except (KeyError, IndexError):
        fail(f"unexpected OpenRouter response shape: {json.dumps(resp)[:500]}")
    return (msg.get("content") or "").strip()


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        fail('usage: ask <example_id> "<question>"')

    example_id = argv[0]
    question = argv[1] if len(argv) == 2 else " ".join(argv[1:])
    question = question.strip()
    if not question:
        fail("question must be non-empty")

    env = get_env()
    check_test_agent_scope(env, example_id)
    example = load_example(env, example_id)

    tok = get_tokenizer()
    q_ids = tok.encode(question).ids
    q_count = len(q_ids)

    if q_count > MAX_QUESTION_TOKENS:
        print(
            f"status: failed -- question is {q_count} tokens "
            f"(limit {MAX_QUESTION_TOKENS}); no details file written"
        )
        return 1

    model = os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
    # Always strip the ground-truth label before handing the example to the
    # oracle — it would otherwise trivially answer label-related questions.
    payload = example.get("content")
    if payload is None:
        payload = {k: v for k, v in example.items() if k != "label"}
    example_content = (
        payload if isinstance(payload, str)
        else json.dumps(payload, ensure_ascii=False)
    )
    resp = call_openrouter(example_content, question, model)
    raw = extract_content(resp)

    resp_ids = tok.encode(raw).ids
    truncated_ids = resp_ids[:MAX_RESPONSE_TOKENS]
    truncated = tok.decode(truncated_ids)

    cwd = Path.cwd()
    details_path = next_output_path(cwd)
    details = {
        "status": "success",
        "example_id": example_id,
        "question": question,
        "question_tokens": q_count,
        "model": model,
        "response": truncated,
        "response_tokens": len(truncated_ids),
        "response_raw": raw,
    }
    details_path.write_text(json.dumps(details, indent=2, ensure_ascii=False))

    print(f"status: success")
    print(f"response: {truncated}")
    print(f"details: {details_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
