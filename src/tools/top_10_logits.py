"""`top_10_logits` tool — inspect top-token distributions at one or more CoT positions.

Reads precomputed logits written by `src/precompute_logits.py` (a sidecar
`<example_id>.logits.npz` next to the example JSON).
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from argparse import ArgumentParser
from collections import Counter
from pathlib import Path

from _backend import BackendNotConfigured, get_top_10_logits
from _common import (
    example_dir,
    fail,
    get_env,
    load_example,
    next_numbered_output_path,
    write_csv,
)


DEFAULT_TOKENIZER = "Qwen/Qwen3-32B"


def _check_test_agent_scope(env: dict, example_id: str) -> None:
    if env["AGENT_TYPE"] != "test":
        return
    assigned = os.environ.get("AGENT_EXAMPLE_ID")
    if assigned and assigned != example_id:
        fail(
            f"test agent can only query its own example "
            f"(AGENT_EXAMPLE_ID={assigned!r}), not {example_id!r}"
        )


def get_readme_description() -> str:
    return """### `top_10_logits <example_id> <token_position>`

Inspect top-token distributions from precomputed CoT logprobs.

**Basic mode**
- `top_10_logits <example_id> <token_position>` shows the top-10 tokens/logprobs at one exact CoT-relative position.

**Aggregate modes**
- `top_10_logits <example_id> --last-k K` averages the token distribution across the last `K` CoT tokens and reports the top-10 aggregate tokens.
- `top_10_logits <example_id> --around-text "..."` finds the **last** case-insensitive match of the text in the visible CoT and aggregates across the matched token span.

**Diff mode**
- Add `--diff` to compare all visible positive vs negative examples using the selected anchor (`<token_position>`, `--last-k`, or `--around-text`).
- Diff mode is mainly useful for the strategy agent, because it depends on labeled few-shot examples.

**Output**
- Single-example mode writes `top_10_logits_<n>.csv`.
- Diff mode writes `top_10_logits_diff_<n>.csv`.
- Aggregate outputs include `avg_probability`, `logprob` (the log of that averaged probability), and `support_count`.
""".strip()


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="top_10_logits",
        description="Inspect top-token distributions from precomputed CoT logprobs.",
    )
    parser.add_argument("example_id", nargs="?")
    parser.add_argument("token_position", nargs="?")
    parser.add_argument("--last-k", type=int, dest="last_k")
    parser.add_argument("--around-text")
    parser.add_argument("--diff", action="store_true")
    return parser


def get_tokenizer():
    from tokenizers import Tokenizer  # lazy import

    name = os.environ.get("QWEN_TOKENIZER", DEFAULT_TOKENIZER)
    return Tokenizer.from_pretrained(name)


def extract_cot_text(example: dict) -> str:
    for key in ("cot_content", "cot_prefix", "chain_of_thought", "thinking"):
        value = example.get(key)
        if isinstance(value, str) and value:
            return value
    fail("could not locate visible CoT text in example JSON")
    raise AssertionError("unreachable")


def parse_explicit_position(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        fail(f"token_position must be an integer, got {raw!r}")
        raise AssertionError("unreachable")


def resolve_positions(
    *,
    example: dict,
    tokenizer,
    explicit_position: int | None,
    last_k: int | None,
    around_text: str | None,
) -> tuple[list[int], str]:
    selectors = [explicit_position is not None, last_k is not None, around_text is not None]
    if sum(selectors) != 1:
        fail(
            "select exactly one anchor: either <token_position>, --last-k K, "
            'or --around-text "..."'
        )

    cot_text = extract_cot_text(example)
    cot_ids = tokenizer.encode(cot_text).ids
    n = len(cot_ids)
    if n == 0:
        fail("example CoT tokenization is empty")

    if explicit_position is not None:
        if explicit_position < 0 or explicit_position >= n:
            fail(f"token_position {explicit_position} out of range (have {n} CoT token positions)")
        return [explicit_position], f"position={explicit_position}"

    if last_k is not None:
        if last_k < 1:
            fail(f"--last-k must be >= 1, got {last_k}")
        start = max(0, n - last_k)
        positions = list(range(start, n))
        return positions, f"last_k={last_k}"

    assert around_text is not None
    haystack = cot_text.lower()
    needle = around_text.lower()
    idx = haystack.rfind(needle)
    if idx < 0:
        fail(f'--around-text match not found in visible CoT: {around_text!r}')
    prefix_ids = tokenizer.encode(cot_text[:idx]).ids
    span_ids = tokenizer.encode(cot_text[idx: idx + len(around_text)]).ids
    if not span_ids:
        fail(f'--around-text matched empty token span: {around_text!r}')
    positions = list(range(len(prefix_ids), min(len(prefix_ids) + len(span_ids), n)))
    return positions, f"around_text={around_text!r}"


def aggregate_pairs_across_positions(
    per_position_pairs: list[list[tuple[str, float]]],
) -> list[dict]:
    total_positions = len(per_position_pairs)
    avg_probs: dict[str, float] = {}
    support: Counter[str] = Counter()
    for pairs in per_position_pairs:
        for token, logprob in pairs:
            prob = math.exp(logprob)
            avg_probs[token] = avg_probs.get(token, 0.0) + (prob / total_positions)
            support[token] += 1

    rows = []
    for rank, (token, avg_prob) in enumerate(
        sorted(avg_probs.items(), key=lambda kv: kv[1], reverse=True)[:10],
        start=1,
    ):
        rows.append({
            "rank": rank,
            "token": token,
            "avg_probability": avg_prob,
            "logprob": math.log(avg_prob) if avg_prob > 0 else float("-inf"),
            "support_count": support[token],
        })
    return rows


def load_visible_examples_with_labels(env: dict) -> list[tuple[str, dict]]:
    base = example_dir(env)
    out = []
    for path in sorted(base.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if "label" in data:
            out.append((path.stem, data))
    return out


def compute_example_distribution(
    env: dict,
    tokenizer,
    example_id: str,
    example: dict,
    explicit_position: int | None,
    last_k: int | None,
    around_text: str | None,
) -> tuple[list[dict], list[int], str]:
    positions, selector_desc = resolve_positions(
        example=example,
        tokenizer=tokenizer,
        explicit_position=explicit_position,
        last_k=last_k,
        around_text=around_text,
    )
    per_position = [get_top_10_logits(env, example_id, example, pos) for pos in positions]
    rows = aggregate_pairs_across_positions(per_position)
    return rows, positions, selector_desc


def run_diff(
    env: dict,
    tokenizer,
    explicit_position: int | None,
    last_k: int | None,
    around_text: str | None,
) -> int:
    if env["AGENT_TYPE"] != "strategy":
        fail("--diff is only supported for strategy agents with labeled few-shot examples")

    labeled = load_visible_examples_with_labels(env)
    if not labeled:
        fail("no labeled visible examples available for --diff")

    grouped: dict[int, list[dict[str, float]]] = {0: [], 1: []}
    selector_desc = None
    skipped = 0
    for example_id, example in labeled:
        label = example.get("label")
        if label not in (0, 1):
            continue
        try:
            rows, _, selector_desc = compute_example_distribution(
                env, tokenizer, example_id, example, explicit_position, last_k, around_text
            )
        except Exception:
            skipped += 1
            continue
        grouped[int(label)].append(rows)

    if not grouped[0] or not grouped[1]:
        fail(
            f"--diff needs at least one usable example from each class; "
            f"usable_neg={len(grouped[0])} usable_pos={len(grouped[1])} skipped={skipped}"
        )

    def mean_prob(rows_per_example: list[list[dict[str, float]]]) -> dict[str, float]:
        token_to_total: dict[str, float] = {}
        n = len(rows_per_example)
        for rows in rows_per_example:
            local = {row["token"]: row["avg_probability"] for row in rows}
            for token, prob in local.items():
                token_to_total[token] = token_to_total.get(token, 0.0) + prob / n
        return token_to_total

    neg_probs = mean_prob(grouped[0])
    pos_probs = mean_prob(grouped[1])
    all_tokens = set(neg_probs) | set(pos_probs)

    out_rows = []
    for rank, token in enumerate(
        sorted(all_tokens, key=lambda t: abs(pos_probs.get(t, 0.0) - neg_probs.get(t, 0.0)), reverse=True)[:50],
        start=1,
    ):
        pos_prob = pos_probs.get(token, 0.0)
        neg_prob = neg_probs.get(token, 0.0)
        out_rows.append({
            "rank": rank,
            "token": token,
            "positive_avg_probability": pos_prob,
            "negative_avg_probability": neg_prob,
            "delta_probability": pos_prob - neg_prob,
            "positive_logprob": math.log(pos_prob) if pos_prob > 0 else float("-inf"),
            "negative_logprob": math.log(neg_prob) if neg_prob > 0 else float("-inf"),
            "n_positive_examples": len(grouped[1]),
            "n_negative_examples": len(grouped[0]),
            "selector": selector_desc or "",
        })

    out_path = next_numbered_output_path("top_10_logits_diff")
    write_csv(out_path, list(out_rows[0].keys()), out_rows)
    print(
        f"diff: selector={selector_desc}; n_positive={len(grouped[1])}; "
        f"n_negative={len(grouped[0])}; skipped={skipped}; rows={len(out_rows)}"
    )
    print(f"details: {out_path.name}")
    return 0


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    env = get_env()
    tokenizer = get_tokenizer()
    explicit_position = parse_explicit_position(args.token_position)

    if args.diff:
        return run_diff(env, tokenizer, explicit_position, args.last_k, args.around_text)

    if not args.example_id:
        fail("example_id is required unless you are running a different subcommand")

    example_id = args.example_id
    _check_test_agent_scope(env, example_id)
    example = load_example(env, example_id)

    try:
        rows, positions, selector_desc = compute_example_distribution(
            env, tokenizer, example_id, example, explicit_position, args.last_k, args.around_text
        )
    except BackendNotConfigured as e:
        fail(str(e), code=3)
        return 3

    out_path = next_numbered_output_path("top_10_logits")
    rows = [
        {
            "example_id": example_id,
            "selector": selector_desc,
            "positions": ",".join(str(p) for p in positions),
            **row,
        }
        for row in rows
    ]
    write_csv(out_path, list(rows[0].keys()), rows)
    print(f"selector: {selector_desc}")
    print(f"positions: {positions[0]}..{positions[-1]} ({len(positions)} token(s))")
    print(f"rows: {len(rows)}")
    print(f"details: {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
