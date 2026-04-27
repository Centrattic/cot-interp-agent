"""`ask` tool — ask a short follow-up question about an example via OpenRouter.

Usage:
    ask <example_id> "<question>" [--times N] [--ans LABEL [LABEL ...]]

Limits (enforced with the Qwen tokenizer):
    - Question must be ≤50 tokens
    - Response is truncated to the first 100 tokens (thinking/reasoning excluded)

Behavior:
    - Strategy agent can query any few-shot example.
    - Test agent can only query its own assigned example (AGENT_EXAMPLE_ID).
    - By default, asks until it collects 5 valid answers and reports every attempt.
    - If `--ans` is omitted, every response counts as valid.
    - On success, writes ask_<n>.csv into the current directory (the
      strategy/ or test/<n>/ folder) and prints a status summary.
    - On failure, prints a status line with the reason; no file written.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import urllib.error
import urllib.request
from argparse import ArgumentParser
from collections import Counter

from _common import fail, get_env, load_example, next_numbered_output_path, write_csv


MAX_QUESTION_TOKENS = 50
MAX_RESPONSE_TOKENS = 100
DEFAULT_NUM_SAMPLES = 5
MAX_ATTEMPTS_MULTIPLIER = 3

DEFAULT_TOKENIZER = "Qwen/Qwen3-32B"
DEFAULT_MODEL = "qwen/qwen3-32b"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def get_readme_description() -> str:
    """Return the agent-facing README blurb for this tool."""
    return f"""### `ask <example_id> "<question>" [--times N] [--ans LABEL [LABEL ...]]`

Ask a follow-up question about an example via an oracle model
(OpenRouter / Qwen3-32B, pinned to DeepInfra for reproducibility).

**Limits**
- Question must tokenize to **≤{MAX_QUESTION_TOKENS} Qwen tokens** (else the call fails and no file is written).
- The reply is truncated to the **first {MAX_RESPONSE_TOKENS} tokens** (reasoning / thinking excluded). OpenRouter is also capped at {MAX_RESPONSE_TOKENS} output tokens.
- The same question is asked **{DEFAULT_NUM_SAMPLES} times by default**. Override with `--times N`.
- If useful, you may specify a desired answer format directly inside the question itself, as long as the full question still fits inside the token budget.
- If you want the tool to aggregate explicit labels, pass them with `--ans`, e.g. `--ans yes no` or `--ans A B`.
- No logit access.

**Scope**
- **Strategy agent:** may query any few-shot `<example_id>` (filename stem in `Examples.csv`).
- **Test agent:** may only query its own assigned example — `AGENT_EXAMPLE_ID` names it; other IDs are rejected.

**Output contract.** Prints a `status:` line to stdout:
- On success, also prints `summary:`, `responses:`, and `details:`. With `--ans`, the summary reports parsed-label majority, valid-sample count, invalid-attempt count, and label entropy. Without `--ans`, it reports unique-answer count and response entropy across collected responses. `details:` names a new file `ask_<n>.csv` in your current directory with columns including parsed labels and validity flags.
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
    parser.add_argument(
        "--ans",
        nargs="+",
        help="Optional explicit labels to aggregate, e.g. --ans yes no or --ans A B.",
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
        "max_tokens": MAX_RESPONSE_TOKENS,
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


def normalize_label(text: str) -> str:
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def normalize_response(text: str) -> str:
    """Canonicalize a short response for agreement statistics."""
    normalized = text.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = re.sub(r"[^\w\s]", "", normalized)
    return normalized.strip()


def extract_explicit_label(text: str, allowed_labels: list[str]) -> str | None:
    """Return an allowed label if it appears within the first 10 whitespace tokens."""
    first_tokens = text.strip().split()[:10]
    allowed = {normalize_label(label): label for label in allowed_labels}
    for token in first_tokens:
        cleaned = re.sub(r"[^\w]", "", token).lower()
        if cleaned in allowed:
            return allowed[cleaned]
    return None


def summarize_responses(
    truncated_responses: list[str],
    normalized_responses: list[str],
    parsed_labels: list[str] | None = None,
    invalid_attempts: int = 0,
) -> dict[str, float | int | str | None]:
    if parsed_labels is not None:
        total = len(parsed_labels)
        counts = Counter(parsed_labels)
        probs = [count / total for count in counts.values()]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        majority_label = None
        majority_label_count = 0
        majority_label_rate = 0.0
        if counts:
            majority_label, majority_label_count = counts.most_common(1)[0]
            majority_label_rate = majority_label_count / total
        return {
            "majority_label": majority_label,
            "majority_label_count": majority_label_count,
            "majority_label_rate": majority_label_rate,
            "valid_samples": total,
            "invalid_attempts": invalid_attempts,
            "unique_labels": len(counts),
            "label_entropy": entropy,
        }

    total = len(normalized_responses)
    counts = Counter(normalized_responses)
    probs = [count / total for count in counts.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return {
        "unique_responses": len(counts),
        "response_entropy": entropy,
    }


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
    answer_labels = args.ans or []
    if answer_labels:
        normalized_labels = [normalize_label(label) for label in answer_labels]
        if any(not label.strip() for label in answer_labels):
            fail("--ans labels must be non-empty")
        if len(set(normalized_labels)) != len(normalized_labels):
            fail(f"--ans labels must be distinct, got {answer_labels!r}")

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
    normalized_responses = []
    parsed_labels = []
    attempts = 0
    invalid_attempts = 0
    max_attempts = args.times * (MAX_ATTEMPTS_MULTIPLIER if answer_labels else 1)
    while len(truncated_responses) < args.times and attempts < max_attempts:
        attempts += 1
        resp = call_openrouter(example_content, question, model)
        raw = extract_content(resp)

        resp_ids = tok.encode(raw).ids
        truncated_ids = resp_ids[:MAX_RESPONSE_TOKENS]
        truncated = tok.decode(truncated_ids)
        normalized = normalize_response(truncated)
        parsed_label = extract_explicit_label(truncated, answer_labels) if answer_labels else None
        is_valid = parsed_label is not None if answer_labels else True
        if is_valid:
            truncated_responses.append(truncated)
            normalized_responses.append(normalized)
            if parsed_label is not None:
                parsed_labels.append(parsed_label)
        else:
            invalid_attempts += 1

        rows.append({
            "status": "success" if is_valid else "invalid",
            "example_id": example_id,
            "question": question,
            "question_tokens": q_count,
            "model": model,
            "num_samples": args.times,
            "attempt_index": attempts,
            "sample_index": len(truncated_responses) if is_valid else "",
            "is_valid": str(is_valid).lower(),
            "parsed_label": parsed_label or "",
            "response": truncated,
            "response_normalized": normalized,
            "response_tokens": len(truncated_ids),
            "response_raw": raw,
        })
    if answer_labels and len(truncated_responses) < args.times:
        print(
            "status: failed -- "
            f"collected {len(truncated_responses)}/{args.times} valid responses "
            f"after {attempts} attempts; no details file written"
        )
        return 1
    write_csv(details_path, list(rows[0].keys()), rows)
    summary = summarize_responses(
        truncated_responses,
        normalized_responses,
        parsed_labels=parsed_labels if answer_labels else None,
        invalid_attempts=invalid_attempts,
    )

    print(f"status: success")
    if answer_labels:
        print(
            "summary: "
            f"majority_label={summary['majority_label']!r}; "
            f"majority_label_count={summary['majority_label_count']}/{args.times}; "
            f"majority_label_rate={summary['majority_label_rate']:.2f}; "
            f"valid_samples={summary['valid_samples']}/{args.times}; "
            f"invalid_attempts={summary['invalid_attempts']}; "
            f"unique_labels={summary['unique_labels']}; "
            f"label_entropy={summary['label_entropy']:.2f}"
        )
    else:
        print(
            "summary: "
            f"unique_responses={summary['unique_responses']}; "
            f"response_entropy={summary['response_entropy']:.2f}"
        )
    print("responses:")
    for sample_index, truncated in enumerate(truncated_responses, start=1):
        print(f"{sample_index}: {truncated}")
    print(f"details: {details_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
