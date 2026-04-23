"""Shared helpers for reconstructing a task's model input and splitting it
into (prefix_token_ids, cot_token_ids).

Every per-example "position" passed to a tool is a **CoT-relative** index:
position 0 means "the first token of `cot_prefix` (or whatever equivalent
the task uses), the thing the agent actually sees in example.json".

Used by:
    - src/precompute_logits.py (to know which token range to persist)
    - src/tools/_backend.py force_and_next_top10 (to splice forced tokens
      into the CoT at a CoT-relative position while still sending the full
      chat-template prefix to the model)
"""

from __future__ import annotations

import json
from pathlib import Path


DEFAULT_SYSTEM_MSG = "You are a helpful assistant."


def _load_prompt_text(source_root: Path, split: str, question_id: str) -> str:
    """Read the original user prompt from
    cot-proxy-tasks/datasets/<id>/prompts/<split>/<question_id>.json,
    falling back to other splits if the id ended up under a different one."""
    prompts_root = source_root.parent / "prompts"
    candidates = [split] + [
        s for s in ("train", "val", "test", "ood_train", "ood_test") if s != split
    ]
    for s in candidates:
        path = prompts_root / s / f"{question_id}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["prompt_text"]
    raise FileNotFoundError(
        f"prompt_text for question_id={question_id!r} not found in any split under {prompts_root}"
    )


def _reasoning_termination_parts(
    example: dict, tokenizer, source_root: Path, split: str
) -> tuple[str, str]:
    """Return (prefix_str, cot_str) for a reasoning_termination example.

    Mirrors cot-proxy-tasks/src/utils/chat_template.build_thinking_prompt:
    prefix = chat_template(system, user=prompt_text) + '<think>\\n'
    cot    = example['cot_prefix']
    """
    prompt_text = _load_prompt_text(source_root, split, example["question_id"])
    chat = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": DEFAULT_SYSTEM_MSG},
            {"role": "user", "content": prompt_text},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    return chat + "<think>\n", example["cot_prefix"]


TASK_PARTS_BUILDERS = {
    "reasoning_termination": _reasoning_termination_parts,
    "termination": _reasoning_termination_parts,
}


def build_prompt_parts(
    task: str, example: dict, tokenizer, source_root: Path, split: str
) -> tuple[list[int], list[int]]:
    """Return (prefix_token_ids, cot_token_ids).

    `prefix_token_ids + cot_token_ids` matches (close enough) the token
    stream the model saw when it produced the labelled rollout. CoT-relative
    positions index into `cot_token_ids` directly.
    """
    if task not in TASK_PARTS_BUILDERS:
        raise ValueError(
            f"no prompt-parts builder registered for task {task!r}. "
            f"Registered: {sorted(TASK_PARTS_BUILDERS)}"
        )
    prefix_str, cot_str = TASK_PARTS_BUILDERS[task](example, tokenizer, source_root, split)
    prefix_ids = tokenizer.encode(prefix_str, add_special_tokens=False)
    cot_ids = tokenizer.encode(cot_str, add_special_tokens=False)
    return prefix_ids, cot_ids


def load_task_meta(scaffold_root: Path, task: str) -> tuple[dict, Path, dict[str, str]]:
    """Return (metadata_dict, source_root, {'few-shot': split, 'test': split}).

    `source_root` is the qwen-3-32b / gemma-3-27b model dir under cot-proxy-tasks.
    """
    task_dir = scaffold_root / "data" / task
    meta = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
    source_root = Path(meta["source"])
    split_of = {
        "few-shot": meta.get("few_shot_split", "train"),
        "test": meta.get("test_split", "test"),
    }
    return meta, source_root, split_of
