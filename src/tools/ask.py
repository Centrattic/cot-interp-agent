"""`ask` tool — ask the model a short follow-up question about an example.

Usage:
    ask <example_id> "<question>"

Limits:
    - `question` must tokenize to at most 10 tokens
    - Only the first 5 tokens of the response are returned
    - No logit access — this is a plain text echo
"""

from __future__ import annotations

import sys

from _backend import BackendNotConfigured, ask_model, tokenize_count
from _common import fail, get_env, load_example


MAX_QUESTION_TOKENS = 10
MAX_RESPONSE_TOKENS = 5


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        fail('usage: ask <example_id> "<question>"')

    example_id = argv[0]
    # Accept either a single quoted arg or a shell-split question.
    question = argv[1] if len(argv) == 2 else " ".join(argv[1:])
    if not question.strip():
        fail("question must be non-empty")

    n_tokens = tokenize_count(question)
    if n_tokens > MAX_QUESTION_TOKENS:
        fail(
            f"question has {n_tokens} tokens, exceeds limit of "
            f"{MAX_QUESTION_TOKENS}"
        )

    env = get_env()
    example = load_example(env, example_id)

    try:
        response = ask_model(example, question, MAX_RESPONSE_TOKENS)
    except BackendNotConfigured as e:
        fail(str(e), code=3)
        return 3

    print(response)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
