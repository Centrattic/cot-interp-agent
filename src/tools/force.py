"""`force` tool — splice forced tokens into an example and read the next token.

Usage:
    force <example_id> <token_position> <tokens_to_force...>

Limits:
    - `tokens_to_force` must tokenize to at most 10 tokens
    - Always returns exactly 1 token (the model's prediction after the splice)
"""

from __future__ import annotations

import sys

from _backend import BackendNotConfigured, force_tokens, tokenize_count
from _common import fail, get_env, load_example, parse_int


MAX_FORCE_TOKENS = 10


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        fail("usage: force <example_id> <token_position> <tokens_to_force...>")

    example_id = argv[0]
    position = parse_int(argv[1], "token_position")
    forced = " ".join(argv[2:])

    n_tokens = tokenize_count(forced)
    if n_tokens == 0:
        fail("tokens_to_force must be non-empty")
    if n_tokens > MAX_FORCE_TOKENS:
        fail(
            f"tokens_to_force has {n_tokens} tokens, exceeds limit of "
            f"{MAX_FORCE_TOKENS}"
        )

    env = get_env()
    example = load_example(env, example_id)

    try:
        next_token = force_tokens(example, position, forced)
    except BackendNotConfigured as e:
        fail(str(e), code=3)
        return 3

    print(next_token)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
