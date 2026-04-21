"""`top_10_logits` tool — show the top-10 tokens and their logits at a position.

Reads precomputed logits written by `src/precompute_logits.py` (a sidecar
`<example_id>.logits.npz` next to the example JSON).

Usage:
    top_10_logits <example_id> <token_position>
"""

from __future__ import annotations

import os
import sys

from _backend import BackendNotConfigured, get_top_10_logits
from _common import fail, get_env, load_example, parse_int


def _check_test_agent_scope(env: dict, example_id: str) -> None:
    if env["AGENT_TYPE"] != "test":
        return
    assigned = os.environ.get("AGENT_EXAMPLE_ID")
    if assigned and assigned != example_id:
        fail(
            f"test agent can only query its own example "
            f"(AGENT_EXAMPLE_ID={assigned!r}), not {example_id!r}"
        )


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        fail("usage: top_10_logits <example_id> <token_position>")

    example_id = argv[0]
    position = parse_int(argv[1], "token_position")

    env = get_env()
    _check_test_agent_scope(env, example_id)
    example = load_example(env, example_id)

    try:
        pairs = get_top_10_logits(env, example_id, example, position)
    except BackendNotConfigured as e:
        fail(str(e), code=3)
        return 3

    for token, logit in pairs:
        print(f"{token!r}\t{logit:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
