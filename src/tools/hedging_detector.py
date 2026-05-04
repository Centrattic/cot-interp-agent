"""`hedging-detector` tool — score sentences of a CoT prefix for epistemic hedging.

Usage:
    hedging-detector <example_id>

Backed by a nested Codex helper with reasoning enabled. The helper reads the
sentences of the CoT prefix and returns, for each sentence, a score in [0, 1] where 0 = a
fully committed / assertive statement and 1 = heavily hedged / uncertain /
self-doubting. A five-point centered moving average produces a hedging
trajectory across the CoT.

Scope
- Strategy agent: any few-shot `<example_id>`.
- Test agent: only its assigned `AGENT_EXAMPLE_ID`.

Rate limit
- Once per sample. On a second invocation with the same `<example_id>` the
  existing `hedging_<example_id>.json` is returned as a cache hit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from _common import fail, get_env, load_example
from _subagent import (
    SubagentError,
    call_structured,
    check_example_scope,
    get_cot_prefix,
    moving_avg,
    sentencize,
)


SYSTEM_PROMPT = (
    "You are an expert at analyzing chains-of-thought from a math reasoning "
    "model. You score each sentence for epistemic hedging: the degree to "
    "which the sentence expresses uncertainty, self-doubt, conditionality, "
    "or reversal, as opposed to making a committed assertion.\n\n"
    "A score of 0.0 means the sentence is a direct, assertive statement "
    "(e.g., a definition, a computed value, a confident conclusion). "
    "A score of 1.0 means the sentence is heavily hedged (e.g., 'Wait, "
    "maybe I got that wrong', 'I'm not sure if this is right', "
    "'Let me reconsider'). Scores in between capture partial hedging "
    "(conditional framings, soft qualifiers like 'probably', 'I think', "
    "'roughly', 'approximately', exploratory 'let me try...')."
)


TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "sentence_scores": {
            "type": "array",
            "description": (
                "One entry per sentence, in order. `idx` is the 0-based "
                "sentence index matching the numbered list in the prompt."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "idx": {"type": "integer"},
                    "score": {
                        "type": "number",
                        "description": "Hedging score in [0, 1].",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One short phrase justifying the score.",
                    },
                },
                "required": ["idx", "score", "rationale"],
            },
        },
    },
    "required": ["sentence_scores"],
}


def _build_user_prompt(sentences: list[str]) -> str:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    return (
        f"Here are the {len(sentences)} sentences of the chain-of-thought, in "
        f"order:\n\n{numbered}\n\n"
        f"Score each sentence for epistemic hedging. Respond by calling the "
        f"submit_hedging_scores tool with one entry per sentence. Scores must "
        f"be in [0, 1]; rationales must be short (≤10 words)."
    )


def analyse_example(
    example_id: str,
    example: dict,
    *,
    cache_path: Path,
) -> dict:
    """Run the hedging analysis for one example, writing the result JSON.

    If ``cache_path`` already exists it is returned unchanged and annotated
    with ``cached: True``.
    """
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
        data["cached"] = True
        return data

    cot = get_cot_prefix(example)
    if not cot:
        fail(
            f"example {example_id!r}: no CoT prefix found (neither 'cot_prefix' "
            f"field nor 'Chain-of-thought prefix:' delimiter in 'content')."
        )

    sentences = sentencize(cot)
    if not sentences:
        fail(f"example {example_id!r}: sentencizer returned no sentences.")

    raw = call_structured(
        system=SYSTEM_PROMPT,
        user=_build_user_prompt(sentences),
        tool_name="submit_hedging_scores",
        tool_description="Submit the per-sentence hedging scores and rationales.",
        tool_input_schema=TOOL_SCHEMA,
    )

    by_idx = {int(e["idx"]): e for e in raw.get("sentence_scores", [])}
    scores: list[float] = []
    rationales: list[str] = []
    for i in range(len(sentences)):
        e = by_idx.get(i)
        if e is None:
            scores.append(0.0)
            rationales.append("(missing from model response)")
        else:
            s = float(e["score"])
            scores.append(max(0.0, min(1.0, s)))
            rationales.append(e.get("rationale", ""))

    trajectory = moving_avg(scores, window=5)
    overall = sum(scores) / len(scores) if scores else 0.0
    peak_idx = max(range(len(scores)), key=lambda i: scores[i]) if scores else -1

    result = {
        "example_id": example_id,
        "n_sentences": len(sentences),
        "sentences": sentences,
        "scores": scores,
        "rationales": rationales,
        "trajectory": trajectory,
        "overall": overall,
        "peak_idx": peak_idx,
        "cached": False,
    }
    cache_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        fail("usage: hedging-detector <example_id>")

    example_id = argv[0]
    env = get_env()
    check_example_scope(env, example_id)
    example = load_example(env, example_id)

    cache_path = Path.cwd() / f"hedging_{example_id}.json"
    try:
        result = analyse_example(example_id, example, cache_path=cache_path)
    except SubagentError as e:
        fail(str(e), code=5)

    status = "cache-hit" if result["cached"] else "success"
    print(f"status: {status}")
    print(f"example_id: {example_id}")
    print(f"n_sentences: {result['n_sentences']}")
    print(f"overall: {result['overall']:.3f}")
    print(f"peak_idx: {result['peak_idx']}")
    print(f"details: {cache_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
