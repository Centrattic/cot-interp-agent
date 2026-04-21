#!/usr/bin/env python3
"""Precompute top-10 logprobs at every CoT-token position for each example in a task.

For every example JSON in `data/<task>/{few-shot,test}/`, reconstructs the
full model input the rollout was generated from (chat template + any task
prefill, e.g. `<think>\\n<cot_prefix>` for reasoning_termination), runs a
single vLLM `/v1/completions` call with `echo=True, logprobs=10,
max_tokens=1`, and stores the top-10 **for CoT positions only** in a
sidecar `<example_id>.logits.npz` next to the JSON.

Positions are **CoT-relative**: position 0 == first token of cot_prefix
(matching what `force`, `top_10_logits`, and `top10_entropy` expose to
the agent).

Note on terminology: vLLM returns log-probabilities, not raw logits. The
stored field is named `top_logits` to match the `top_10_logits` tool's
CLI naming, but the numeric values are logprobs. Entropy computation in
`top10_entropy` is invariant to this distinction.

Usage:
    python src/precompute_logits.py \\
        --task reasoning_termination \\
        --vllm-url http://localhost:8000/v1 \\
        --model Qwen/Qwen3-32B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests
from transformers import AutoTokenizer


SCAFFOLD_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = SCAFFOLD_ROOT / "data"
TOP_K = 10

# Reuse the shared task-input builder from tools/ so force / precompute stay in sync.
sys.path.insert(0, str(SCAFFOLD_ROOT / "src" / "tools"))
from _task_io import build_prompt_parts, load_task_meta  # noqa: E402


# ---------------------------------------------------------------------------
# vLLM client
# ---------------------------------------------------------------------------

def call_vllm_on_ids(
    vllm_url: str, model: str, token_ids: list[int], top_k: int = TOP_K,
    timeout: int = 600,
) -> list[dict | None]:
    """POST to vLLM /v1/completions with echo+logprobs, prompting by token ids.

    Returns `top_logprobs`: list of length len(token_ids); entry i is either
    None (for position 0) or a dict {token_str: logprob, …} with up to
    `top_k` entries for the distribution that produced prompt token i.
    """
    resp = requests.post(
        f"{vllm_url.rstrip('/')}/completions",
        json={
            "model": model,
            "prompt": token_ids,   # vLLM accepts a list of token ids
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": top_k,
            "echo": True,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["logprobs"]["top_logprobs"]


def to_arrays(
    top_logprobs: list[dict | None], top_k: int = TOP_K
) -> tuple[np.ndarray, np.ndarray]:
    """Collate per-position top-logprob dicts into dense (N, top_k) arrays,
    sorted descending by logprob within each row."""
    n = len(top_logprobs)
    out_tokens = np.full((n, top_k), "", dtype=object)
    out_logits = np.full((n, top_k), -np.inf, dtype=np.float32)
    for i, row in enumerate(top_logprobs):
        if not row:
            continue
        items = sorted(row.items(), key=lambda kv: -kv[1])[:top_k]
        for j, (tok, score) in enumerate(items):
            out_tokens[i, j] = tok
            out_logits[i, j] = float(score)
    return out_tokens, out_logits


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def process_one(
    json_path: Path,
    *,
    split: str,
    tokenizer,
    source_root: Path,
    vllm_url: str,
    model: str,
    task: str,
) -> Path:
    example = json.loads(json_path.read_text(encoding="utf-8"))
    prefix_ids, cot_ids = build_prompt_parts(task, example, tokenizer, source_root, split)
    full_ids = prefix_ids + cot_ids

    top_logprobs = call_vllm_on_ids(vllm_url, model, full_ids)

    # Slice to CoT positions only. vLLM's echo with logprobs returns one
    # entry per PROMPT position (len == len(full_ids)); position i's dict is
    # the distribution that produced token i given tokens 0..i-1.
    # CoT-relative position p corresponds to absolute position len(prefix_ids) + p.
    cot_slice = top_logprobs[len(prefix_ids):]
    top_tokens, top_logits = to_arrays(cot_slice)

    out_path = json_path.parent / f"{json_path.stem}.logits.npz"
    np.savez_compressed(
        out_path,
        top_tokens=top_tokens,
        top_logits=top_logits,
        cot_token_ids=np.array(cot_ids, dtype=np.int32),
        prefix_len=np.int32(len(prefix_ids)),
    )
    return out_path


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--task", required=True, help="Task name in data/")
    p.add_argument("--vllm-url", default="http://localhost:8000/v1",
                   help="vLLM OpenAI-compatible server base URL (default: %(default)s)")
    p.add_argument("--model", required=True,
                   help="Model name as registered with the vLLM server (e.g. Qwen/Qwen3-32B)")
    p.add_argument("--tokenizer", default=None,
                   help="HF tokenizer name for local chat-template building. Defaults to --model.")
    p.add_argument("--split", default="both", choices=["few-shot", "test", "both"])
    p.add_argument("--force", action="store_true", help="Overwrite existing .logits.npz files.")
    p.add_argument("--limit", type=int, default=None, help="Process at most N examples per split.")
    args = p.parse_args()

    task_dir = DATA_DIR / args.task
    if not task_dir.exists():
        p.error(f"{task_dir} not found; ingest the task first")

    _, source_root, split_of = load_task_meta(SCAFFOLD_ROOT, args.task)

    tokenizer_name = args.tokenizer or args.model
    print(f"Loading tokenizer: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    splits = ["few-shot", "test"] if args.split == "both" else [args.split]
    for split in splits:
        split_dir = task_dir / split
        if not split_dir.exists():
            print(f"[{split}] {split_dir} missing; skipping")
            continue
        files = sorted(split_dir.glob("*.json"))
        if args.limit is not None:
            files = files[: args.limit]
        print(f"[{split}] {len(files)} examples (source split = {split_of[split]!r})")
        for i, jp in enumerate(files, 1):
            out_path = jp.parent / f"{jp.stem}.logits.npz"
            if out_path.exists() and not args.force:
                print(f"  ({i}/{len(files)}) skip {jp.stem} (exists)")
                continue
            try:
                process_one(
                    jp,
                    split=split_of[split],
                    tokenizer=tokenizer,
                    source_root=source_root,
                    vllm_url=args.vllm_url,
                    model=args.model,
                    task=args.task,
                )
                print(f"  ({i}/{len(files)}) wrote {out_path.name}")
            except Exception as e:
                print(f"  ({i}/{len(files)}) ERROR {jp.stem}: {e}")


if __name__ == "__main__":
    main()
