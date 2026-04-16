"""`entropy` tool — entropy of the model distribution restricted to a set of tokens.

Usage:
    entropy <example_id> <token_position> <tokens_to_get_entropy_over...>

Tokens may be given as separate args or as one whitespace-separated string.
"""

from __future__ import annotations

import sys

from _backend import BackendNotConfigured, get_entropy
from _common import fail, get_env, load_example, parse_int


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        fail(
            "usage: entropy <example_id> <token_position> "
            "<tokens_to_get_entropy_over...>"
        )

    example_id = argv[0]
    position = parse_int(argv[1], "token_position")

    # Accept either a single quoted whitespace-separated string or multiple args.
    raw = argv[2:]
    if len(raw) == 1:
        tokens = raw[0].split()
    else:
        tokens = list(raw)

    if len(tokens) < 2:
        fail("need at least 2 tokens to compute entropy")

    env = get_env()
    example = load_example(env, example_id)

    try:
        value = get_entropy(example, position, tokens)
    except BackendNotConfigured as e:
        fail(str(e), code=3)
        return 3

    print(value)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
