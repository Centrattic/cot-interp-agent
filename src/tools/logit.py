"""`logit` tool — read the logit assigned to a specific token at a position.

Usage:
    logit <example_id> <token_position> <token_to_extract>
"""

from __future__ import annotations

import sys

from _backend import BackendNotConfigured, get_logit
from _common import fail, get_env, load_example, parse_int


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        fail("usage: logit <example_id> <token_position> <token_to_extract>")

    example_id = argv[0]
    position = parse_int(argv[1], "token_position")
    token = argv[2]
    if not token:
        fail("token_to_extract must be non-empty")

    env = get_env()
    example = load_example(env, example_id)

    try:
        value = get_logit(example, position, token)
    except BackendNotConfigured as e:
        fail(str(e), code=3)
        return 3

    print(value)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
