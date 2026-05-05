#!/usr/bin/env python3
"""Precompute top-10 CoT logprobs with a local Transformers model.

This is a fallback for GPU boxes where vLLM does not come ready cleanly. It
writes the same sidecar format as ``precompute_logits.py``:
``<example_id>.logits.npz`` with ``top_tokens``, ``top_logits``,
``cot_token_ids``, and ``prefix_len``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SCAFFOLD_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = SCAFFOLD_ROOT / "data"
TOP_K = 10

sys.path.insert(0, str(SCAFFOLD_ROOT / "src" / "tools"))
from _task_io import build_prompt_parts, load_task_meta  # noqa: E402


def _topk_for_ids(model, tokenizer, token_ids: list[int], top_k: int) -> list[dict | None]:
    if len(token_ids) < 2:
        return [None] * len(token_ids)

    device = next(model.parameters()).device
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=False)
        logits = out.logits[0, :-1, :].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        vals, ids = torch.topk(logprobs, k=top_k, dim=-1)

    rows: list[dict | None] = [None]
    vals_cpu = vals.cpu()
    ids_cpu = ids.cpu()
    for row_ids, row_vals in zip(ids_cpu.tolist(), vals_cpu.tolist()):
        rows.append(
            {
                tokenizer.decode([tok_id], skip_special_tokens=False): float(score)
                for tok_id, score in zip(row_ids, row_vals)
            }
        )
    return rows


def _to_arrays(
    top_logprobs: list[dict | None], top_k: int = TOP_K
) -> tuple[np.ndarray, np.ndarray]:
    out_tokens = np.full((len(top_logprobs), top_k), "", dtype=object)
    out_logits = np.full((len(top_logprobs), top_k), -np.inf, dtype=np.float32)
    for i, row in enumerate(top_logprobs):
        if not row:
            continue
        for j, (tok, score) in enumerate(row.items()):
            out_tokens[i, j] = tok
            out_logits[i, j] = float(score)
    return out_tokens, out_logits


def process_one(
    json_path: Path,
    *,
    split: str,
    tokenizer,
    model,
    source_root: Path,
    task: str,
) -> Path:
    example = json.loads(json_path.read_text(encoding="utf-8"))
    prefix_ids, cot_ids = build_prompt_parts(task, example, tokenizer, source_root, split)
    full_ids = prefix_ids + cot_ids

    top_logprobs = _topk_for_ids(model, tokenizer, full_ids, TOP_K)
    top_tokens, top_logits = _to_arrays(top_logprobs[len(prefix_ids) :])

    out_path = json_path.parent / f"{json_path.stem}.logits.npz"
    np.savez_compressed(
        out_path,
        top_tokens=top_tokens,
        top_logits=top_logits,
        cot_token_ids=np.array(cot_ids, dtype=np.int32),
        prefix_len=np.int32(len(prefix_ids)),
    )
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, nargs="+")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--split", default="both", choices=["few-shot", "test", "both"])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()

    tokenizer_name = args.tokenizer or args.model
    print(f"Loading tokenizer: {tokenizer_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    print(f"Loading model: {args.model}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()

    splits = ["few-shot", "test"] if args.split == "both" else [args.split]
    for task in args.task:
        task_dir = DATA_DIR / task
        if not task_dir.exists():
            parser.error(f"{task_dir} not found")
        _, source_root, split_of = load_task_meta(SCAFFOLD_ROOT, task)
        print(f"[task] {task}", flush=True)
        for split in splits:
            split_dir = task_dir / split
            files = sorted(split_dir.glob("*.json"))
            if args.limit is not None:
                files = files[: args.limit]
            print(f"[{split}] {len(files)} examples (source split = {split_of[split]!r})", flush=True)
            for i, json_path in enumerate(files, 1):
                out_path = json_path.parent / f"{json_path.stem}.logits.npz"
                if out_path.exists() and not args.force:
                    print(f"  ({i}/{len(files)}) skip {json_path.stem} (exists)", flush=True)
                    continue
                try:
                    process_one(
                        json_path,
                        split=split_of[split],
                        tokenizer=tokenizer,
                        model=model,
                        source_root=source_root,
                        task=task,
                    )
                    print(f"  ({i}/{len(files)}) wrote {out_path.name}", flush=True)
                except Exception as exc:
                    print(f"  ({i}/{len(files)}) ERROR {json_path.stem}: {exc}", flush=True)
                finally:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
