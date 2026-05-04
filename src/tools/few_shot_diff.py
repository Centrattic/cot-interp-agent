"""`few-shot-diff` tool — ask a fresh Codex helper to blind-diff two groups.

Usage:
    few-shot-diff

Strategy-agent only. Loads every few-shot example, splits them by ground-truth
label into two groups (presented as "Group 1" and "Group 2" without labels),
and asks a fresh Codex helper with reasoning enabled to identify ranked
distinguishing features. The goal is to surface signals the strategy agent
might otherwise miss or overweight because it has the label in front of it.

Output
- `few_shot_diff.json` in the strategy directory, containing ranked features
  (description, confidence, per-group evidence example ids) plus a summary.

Rate limit
- Once per strategy run. Lock:
  `.subagent_locks/few_shot_diff.lock` is created on success.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from _common import fail, get_env, list_few_shot_ids, load_example
from _subagent import (
    SubagentError,
    assert_lock_free,
    assert_strategy_only,
    call_structured,
    get_cot_prefix,
    write_lock,
)


LOCK_NAME = "few_shot_diff"


@lru_cache(maxsize=1)
def _load_task_metadata(scaffold_root: str, task: str) -> dict:
    p = Path(scaffold_root) / "data" / task / "metadata.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _load_prompt_text(env: dict, question_id: str) -> str | None:
    """Look up the original prompt for an example's `question_id`.

    Prompts live at `data/<task>/prompts/<few_shot_split>/<question_id>.json`
    with key `prompt_text` (that's the layout cot-proxy-tasks ships). Returns
    None if the file isn't present.
    """
    meta = _load_task_metadata(env["SCAFFOLD_ROOT"], env["AGENT_TASK"])
    split = meta.get("few_shot_split", "few-shot")
    p = (
        Path(env["SCAFFOLD_ROOT"]) / "data" / env["AGENT_TASK"]
        / "prompts" / split / f"{question_id}.json"
    )
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d.get("prompt_text")


def _build_content(env: dict, example: dict) -> str:
    """Construct the text block to show the subagent for one example.

    Preference order:
      1. Pre-assembled `content` field (ingest-pipeline-normalised format).
      2. Extracted CoT/reasoning/report text plus prompt text looked up by
         `question_id` — wrapped in the same "Question:/Chain-of-thought
         prefix:" shape as (1).
      3. Bare extracted CoT/reasoning/report text if no prompt file is
         available.
    """
    content = example.get("content")
    if content:
        return content

    cot = get_cot_prefix(example)
    qid = example.get("question_id")
    if qid:
        prompt_text = _load_prompt_text(env, qid)
        if prompt_text:
            return f"Question: {prompt_text}\n\nChain-of-thought prefix:\n{cot}"
    return cot


SYSTEM_PROMPT = (
    "You are a careful reader. You will see two groups of texts, labelled "
    "'Group 1' and 'Group 2'. Each text is an unlabelled excerpt from a "
    "reasoning transcript. Your job is to identify features that distinguish "
    "Group 1 from Group 2, **and flag which features may not transfer** if "
    "the caller applies them to texts from a different domain (e.g., a "
    "different subject area or writing style).\n\n"
    "Ground rules:\n"
    "- You do NOT know what Group 1 or Group 2 mean. Do not speculate about "
    "  their semantics. Characterise them empirically.\n"
    "- Report features in ranked order of how reliably they separate the "
    "  groups **within the given data**. Use `confidence` in [0, 1].\n"
    "- For each feature, also report `domain_dependence` in [0, 1]: how much "
    "  the feature depends on the specific domain / content / writing style "
    "  of these particular texts, rather than reflecting a task-general "
    "  signal. `0` = the feature is a general linguistic / structural / "
    "  reasoning-move signal likely to appear in any domain; `1` = the "
    "  feature is a surface pattern tied to this data's domain (e.g., "
    "  subject-specific vocabulary, formatting conventions common in one "
    "  discipline, presence of a markup style used by these particular "
    "  writers but not a general indicator of the underlying distinction). "
    "  If the caller's downstream task involves a different domain, any "
    "  high-`domain_dependence` feature should be **down-weighted or "
    "  discarded** even if its `confidence` is high.\n"
    "- For each feature, cite concrete evidence: the example ids (given in "
    "  square brackets before each text) where the feature appears.\n"
    "- Prefer features that are observable in the text (surface phrases, "
    "  structural patterns, linguistic markers, reasoning moves) over vague "
    "  descriptions."
)


TOOL_SCHEMA = {
    "type": "object",
    "properties": {
        "features": {
            "type": "array",
            "description": (
                "Ranked list of distinguishing features, most reliable first."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": (
                            "Concrete description of the feature. What "
                            "observable pattern separates the groups?"
                        ),
                    },
                    "confidence": {
                        "type": "number",
                        "description": "How reliably this feature separates the groups, in [0, 1].",
                    },
                    "evidence_group1": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Example ids from Group 1 that exemplify this feature.",
                    },
                    "evidence_group2": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Example ids from Group 2 that exemplify this feature.",
                    },
                    "domain_dependence": {
                        "type": "number",
                        "description": (
                            "In [0, 1]: how much this feature is tied to the "
                            "domain of the given data (0 = task-general, "
                            "1 = surface pattern specific to this domain). "
                            "Callers doing domain transfer should down-weight "
                            "high-domain_dependence features."
                        ),
                    },
                    "domain_dependence_reason": {
                        "type": "string",
                        "description": (
                            "One short phrase (≤20 words) explaining the "
                            "`domain_dependence` score."
                        ),
                    },
                },
                "required": [
                    "description",
                    "confidence",
                    "evidence_group1",
                    "evidence_group2",
                    "domain_dependence",
                    "domain_dependence_reason",
                ],
            },
        },
        "summary": {
            "type": "string",
            "description": "One-paragraph summary characterising how the groups differ.",
        },
    },
    "required": ["features", "summary"],
}


def _format_group(group_name: str, entries: list[tuple[str, str]]) -> str:
    parts = [f"=== {group_name} ==="]
    for ex_id, content in entries:
        parts.append(f"\n[{ex_id}]\n{content}")
    return "\n".join(parts)


def _group_label(label) -> str:
    """Normalise a label to a stable group key.

    Accepts either the canonical string ("yes"/"no") or integer (1/0) the
    pipeline uses interchangeably. Returns a string key.
    """
    if isinstance(label, str):
        return label
    return str(label)


def main(argv: list[str]) -> int:
    if argv:
        fail("usage: few-shot-diff  (takes no arguments)")

    env = get_env()
    assert_strategy_only(env, "few-shot-diff")
    assert_lock_free(LOCK_NAME, "few-shot-diff")

    example_ids = list_few_shot_ids(env)
    if not example_ids:
        fail("no few-shot examples found.")

    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for ex_id in example_ids:
        example = load_example(env, ex_id)
        label = _group_label(example.get("label"))
        content = _build_content(env, example)
        grouped[label].append((ex_id, content))

    if len(grouped) != 2:
        fail(
            f"expected exactly 2 label groups in few-shot, got {len(grouped)}: "
            f"{sorted(grouped)}"
        )

    # Sort so the mapping from label → Group 1/2 is deterministic.
    labels_sorted = sorted(grouped.keys())
    group1_label, group2_label = labels_sorted[0], labels_sorted[1]
    group1 = grouped[group1_label]
    group2 = grouped[group2_label]

    # Pass the caller's task description (from metadata.json) so the subagent
    # can reason about what "domain dependence" means for *this* caller. The
    # subagent still must not use the label information — we only share the
    # task framing, not the group→label mapping.
    meta = _load_task_metadata(env["SCAFFOLD_ROOT"], env["AGENT_TASK"])
    task_description = meta.get("description", "(no task description provided)")

    user = (
        "## Caller context\n"
        f"{task_description}\n\n"
        "The two groups below come from the caller's few-shot examples. Note "
        "that if the caller's task description mentions a domain shift "
        "between the few-shot and the held-out test set, features tied to "
        "the few-shot's specific domain should receive a high "
        "`domain_dependence` score.\n\n"
        + _format_group("Group 1", group1)
        + "\n\n"
        + _format_group("Group 2", group2)
        + "\n\nIdentify features that distinguish Group 1 from Group 2. "
        "Respond by calling the submit_diff tool with ranked features, "
        "confidences, and domain-dependence ratings, plus a summary. Do not "
        "speculate about which group corresponds to which label."
    )

    try:
        raw = call_structured(
            system=SYSTEM_PROMPT,
            user=user,
            tool_name="submit_diff",
            tool_description="Submit the ranked distinguishing features and summary.",
            tool_input_schema=TOOL_SCHEMA,
        )
    except SubagentError as e:
        fail(str(e), code=5)

    result = {
        "tool": "few-shot-diff",
        "group_mapping": {
            "Group 1": {"label": group1_label, "n": len(group1)},
            "Group 2": {"label": group2_label, "n": len(group2)},
        },
        "features": raw.get("features", []),
        "summary": raw.get("summary", ""),
    }

    out_path = Path.cwd() / "few_shot_diff.json"
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    write_lock(LOCK_NAME)

    print(f"status: success  (lock written)")
    print(
        f"Group 1 = label {group1_label!r} (n={len(group1)})   "
        f"Group 2 = label {group2_label!r} (n={len(group2)})"
    )
    print(f"n_features: {len(result['features'])}")
    print(f"details: {out_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
