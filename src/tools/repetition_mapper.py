"""`repetition-mapper` tool — cluster semantically repeated conclusions in a CoT.

Usage:
    repetition-mapper <example_id>

Backed by a nested Codex helper with reasoning enabled. The helper reads the
sentences of the CoT prefix and groups those that restate or re-derive the
same conclusion / intermediate result. For each cluster, it reports the sentence indices that
belong to it, a short label, and a restatement count. The longest chain of
restatements of a single conclusion is surfaced separately.

Scope
- Strategy agent: any few-shot `<example_id>`.
- Test agent: only its assigned `AGENT_EXAMPLE_ID`.

Rate limit
- Once per sample. On a second invocation with the same `<example_id>` the
  existing `repetition_<example_id>.json` is returned as a cache hit.
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
    sentencize,
)


SYSTEM_PROMPT = (
    "You are an expert at analyzing chains-of-thought from a math reasoning "
    "model. Your job is to identify repetition: clusters of sentences that "
    "restate, re-derive, or re-affirm the same conclusion or intermediate "
    "result.\n\n"
    "Two sentences belong to the same cluster when they express the same "
    "claim (even if phrased differently), recompute the same quantity, or "
    "re-assert the same conclusion. Do not cluster sentences that merely "
    "share topic; require that the underlying *claim* or *computed value* "
    "be the same. A cluster of size 1 (a unique claim) should NOT be "
    "reported — only report clusters with ≥2 sentences."
)


TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "clusters": {
            "type": "array",
            "description": (
                "List of clusters, each grouping ≥2 sentence indices that "
                "restate the same claim. Exclude singletons. Order clusters "
                "by `restatement_count` descending."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Short phrase naming the shared claim (≤10 words).",
                    },
                    "sentence_indices": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "0-based indices of sentences in this cluster.",
                    },
                    "restatement_count": {
                        "type": "integer",
                        "description": "Number of sentences in the cluster.",
                    },
                },
                "required": ["label", "sentence_indices", "restatement_count"],
            },
        },
        "longest_chain": {
            "type": "object",
            "description": (
                "The single cluster whose sentence indices span the widest "
                "range in the CoT (i.e., is restated furthest apart). If "
                "there are no clusters, return {indices: [], span: 0}."
            ),
            "properties": {
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "span": {
                    "type": "integer",
                    "description": "max(indices) - min(indices); 0 if indices empty.",
                },
                "label": {"type": "string"},
            },
            "required": ["indices", "span", "label"],
        },
    },
    "required": ["clusters", "longest_chain"],
}


def _build_user_prompt(sentences: list[str]) -> str:
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    return (
        f"Here are the {len(sentences)} sentences of the chain-of-thought, in "
        f"order:\n\n{numbered}\n\n"
        f"Cluster sentences that restate or re-derive the same claim. Respond "
        f"by calling the submit_repetition_clusters tool with the resulting "
        f"clusters and the longest-range chain."
    )


def analyse_example(
    example_id: str,
    example: dict,
    *,
    cache_path: Path,
) -> dict:
    """Run the repetition analysis for one example, writing the result JSON."""
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
        tool_name="submit_repetition_clusters",
        tool_description="Submit the repetition clusters and longest chain.",
        tool_input_schema=TOOL_SCHEMA,
    )

    clusters = raw.get("clusters", [])
    longest = raw.get("longest_chain", {"indices": [], "span": 0})

    cluster_count = len(clusters)

    result = {
        "example_id": example_id,
        "n_sentences": len(sentences),
        "sentences": sentences,
        "clusters": clusters,
        "cluster_count": cluster_count,
        "longest_chain": longest,
        "cached": False,
    }
    cache_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main(argv: list[str]) -> int:
    if len(argv) != 1:
        fail("usage: repetition-mapper <example_id>")

    example_id = argv[0]
    env = get_env()
    check_example_scope(env, example_id)
    example = load_example(env, example_id)

    cache_path = Path.cwd() / f"repetition_{example_id}.json"
    try:
        result = analyse_example(example_id, example, cache_path=cache_path)
    except SubagentError as e:
        fail(str(e), code=5)

    status = "cache-hit" if result["cached"] else "success"
    print(f"status: {status}")
    print(f"example_id: {example_id}")
    print(f"n_sentences: {result['n_sentences']}")
    print(f"cluster_count: {result['cluster_count']}")
    print(f"longest_chain_span: {result['longest_chain'].get('span', 0)}")
    print(f"details: {cache_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
