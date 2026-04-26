"""`ask` tool — ask a short follow-up question about an example via OpenRouter.

Usage:
    ask <example_id> "<question>" [--times N]

Limits (enforced with the Qwen tokenizer):
    - Question must be ≤30 tokens
    - Response is truncated to the first 20 tokens (thinking/reasoning excluded)

Behavior:
    - Strategy agent can query any few-shot example.
    - Test agent can only query its own assigned example (AGENT_EXAMPLE_ID).
    - By default, asks the same question 5 times and reports every answer.
    - On success, writes ask_<n>.csv into the current directory (the
      strategy/ or test/<n>/ folder) and prints a status summary.
    - On failure, prints a status line with the reason; no file written.
"""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from argparse import ArgumentParser

from _common import fail, get_env, load_example, next_numbered_output_path, write_csv


MAX_QUESTION_TOKENS = 30
MAX_RESPONSE_TOKENS = 20
DEFAULT_NUM_SAMPLES = 5

DEFAULT_TOKENIZER = "Qwen/Qwen3-32B"
DEFAULT_MODEL = "qwen/qwen3-32b"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def get_readme_description() -> str:
    """Return the agent-facing README blurb for this tool."""
    return f"""### `ask <example_id> "<question>" [--times N]`

Ask a follow-up question about an example via an oracle model
(OpenRouter / Qwen3-32B, pinned to DeepInfra for reproducibility).

**Limits**
- Question must tokenize to **≤{MAX_QUESTION_TOKENS} Qwen tokens** (else the call fails and no file is written).
- The reply is truncated to the **first {MAX_RESPONSE_TOKENS} tokens** (reasoning / thinking excluded). Phrase the question so a concise answer still fits inside that budget.
- The same question is asked **{DEFAULT_NUM_SAMPLES} times by default**. Override with `--times N`.
- No logit access.

**Scope**
- **Strategy agent:** may query any few-shot `<example_id>` (filename stem in `Examples.csv`).
- **Test agent:** may only query its own assigned example — `AGENT_EXAMPLE_ID` names it; other IDs are rejected.

**Output contract.** Prints a `status:` line to stdout:
- On success, also prints `responses:` and `details:` — the latter names a new file `ask_<n>.csv` in your current directory with columns `status,example_id,question,question_tokens,model,num_samples,sample_index,response,response_tokens,response_raw`.
- On failure (token limit exceeded, wrong example id, etc.), prints the reason and writes **no** file.
""".strip()


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="ask",
        description="Ask a short follow-up question about an example via OpenRouter.",
    )
    parser.add_argument("example_id")
    parser.add_argument("question")
    parser.add_argument(
        "--times",
        type=int,
        default=DEFAULT_NUM_SAMPLES,
        help=f"Number of times to resample the same question (default: {DEFAULT_NUM_SAMPLES}).",
    )
    return parser


def get_tokenizer():
    """Load the Qwen tokenizer once per process."""
    from tokenizers import Tokenizer  # lazy import

    name = os.environ.get("QWEN_TOKENIZER", DEFAULT_TOKENIZER)
    return Tokenizer.from_pretrained(name)


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
    parser = build_parser()
    args = parser.parse_args(argv)

    example_id = args.example_id
    question = args.question
    question = question.strip()
    if not question:
        fail("question must be non-empty")
    if args.times < 1:
        fail(f"--times must be >= 1, got {args.times}")

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
    details_path = next_numbered_output_path("ask")
    rows = []
    truncated_responses = []
    for sample_index in range(1, args.times + 1):
        resp = call_openrouter(example_content, question, model)
        raw = extract_content(resp)

        resp_ids = tok.encode(raw).ids
        truncated_ids = resp_ids[:MAX_RESPONSE_TOKENS]
        truncated = tok.decode(truncated_ids)
        truncated_responses.append(truncated)

        rows.append({
            "status": "success",
            "example_id": example_id,
            "question": question,
            "question_tokens": q_count,
            "model": model,
            "num_samples": args.times,
            "sample_index": sample_index,
            "response": truncated,
            "response_tokens": len(truncated_ids),
            "response_raw": raw,
        })
    write_csv(details_path, list(rows[0].keys()), rows)

    print(f"status: success")
    print("responses:")
    for sample_index, truncated in enumerate(truncated_responses, start=1):
        print(f"{sample_index}: {truncated}")
    print(f"details: {details_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
