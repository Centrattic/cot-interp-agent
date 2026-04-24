"""`force` tool — splice up to 10 tokens into the CoT at a given position,
read back the next token the model would emit and the top-10 logprobs over
that next-token slot.

Usage:
    force <example_id> <token_position> <tokens_to_force...>

`token_position` is a **CoT-relative** index: position 0 == the first token
of `cot_prefix`. The spliced prompt is
    <chat prefix> + <cot_prefix[:position] tokens> + <forced tokens>
and the model is asked to emit the next token from there.

`tokens_to_force` may be given as separate words or one quoted string; it
is tokenized with the model's tokenizer and must come out to at most 10
tokens.

Output:
    next_token: '<token>'
    top_10:
      '<tok>'\\t<logprob>
      ...

Backed by Tinker's SamplingClient — the invocation needs a TINKER_API_KEY
in the environment.
"""

from __future__ import annotations

import json
import os
import sys

from _backend import BackendNotConfigured, force_and_next_top10
from _common import (
    fail,
    get_env,
    load_example,
    next_numbered_output_path,
    parse_int,
    write_csv,
)


MAX_FORCE_TOKENS = 10


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
    if len(argv) < 3:
        fail("usage: force <example_id> <token_position> <tokens_to_force...>")

    example_id = argv[0]
    position = parse_int(argv[1], "token_position")
    forced = argv[2] if len(argv) == 3 else " ".join(argv[2:])
    if not forced:
        fail("tokens_to_force must be non-empty")

    env = get_env()
    _check_test_agent_scope(env, example_id)
    example = load_example(env, example_id)

    try:
        next_token, top10 = force_and_next_top10(
            env=env,
            example_id=example_id,
            example=example,
            cot_position=position,
            forced_text=forced,
            max_forced_tokens=MAX_FORCE_TOKENS,
        )
    except BackendNotConfigured as e:
        fail(str(e), code=3)
        return 3

    out_path = next_numbered_output_path("force")
    top10_json = json.dumps(
        [{"rank": rank, "token": token, "logprob": lp} for rank, (token, lp) in enumerate(top10, start=1)],
        ensure_ascii=False,
    )
    row = {
        "example_id": example_id,
        "token_position": position,
        "forced_text": forced,
        "next_token": next_token,
        "next_token_logprob": float(top10[0][1]),
        "top_10_logprobs_json": top10_json,
    }
    write_csv(out_path, list(row.keys()), [row])
    print(f"next_token: {next_token!r}")
    print(f"details: {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
