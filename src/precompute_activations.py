#!/usr/bin/env python3
"""Extract layer-32 residual-stream activations for cot-interp-agent tasks and
encode them through the Qwen3-32B SAE (trainer_2).

For every example JSON under each task's source pool
(`datasets/<id>/qwen-3-32b/<split>/<example>.json`), this:
  1. Reconstructs the exact model input the rollout was sampled from, by
     reusing `_task_io.build_prompt_parts` (chat template + `<think>\\n` +
     example's CoT field).
  2. Runs a single Qwen3-32B forward pass and grabs the residual stream at
     layer 32 (post-residual) at the CoT-token positions only.
  3. Encodes those activations through the JumpReLU SAE in
     tools/sae_encode.py and writes a `<example_stem>.sae.npz` sidecar next
     to the source JSON.

The intermediate `.npy` is not persisted by default (pass `--keep-npy` for
debugging). The `.sae.npz` sidecars are what the agent-facing `sae` tool
consumes (via runtime sampling into per-run few-shot dirs).

Usage (on a GPU host with Qwen3-32B):
    # all six Qwen tasks, both few-shot and test pools
    python src/precompute_activations.py --tasks all

    # one task, smoke test
    python src/precompute_activations.py --tasks followup_confidence --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCAFFOLD_ROOT = Path(__file__).resolve().parent.parent
SAE_HIDDEN_LAYER = 32  # resid_post_layer_32 → hidden_states[33] in HF
D_MODEL = 5120
DEFAULT_MODEL = "Qwen/Qwen3-32B"

# All six Qwen3-32B tasks (gemma_self_deletion uses a different model and SAE).
ALL_QWEN_TASKS = (
    "reasoning_termination",
    "followup_confidence",
    "user_preference_sycophancy",
    "stanford_hint",
    "atypical_answer",
    "atypical_cot_length",
)

# Pull in the shared prompt builder and the SAE encoder.
sys.path.insert(0, str(SCAFFOLD_ROOT / "src" / "tools"))
from _task_io import build_prompt_parts  # noqa: E402
import sae_encode  # noqa: E402


def _load_task_meta(task: str) -> tuple[Path, dict[str, str]]:
    """Return (source_root, {alias: split_dir_name}) for a task.

    `source_root` is the qwen-3-32b dir under datasets/<id>/.
    """
    meta_path = SCAFFOLD_ROOT / "data" / task / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source_root = Path(meta["source"])
    splits = {
        "few-shot": meta.get("few_shot_split", "few-shot"),
        "test": meta.get("test_split", "test"),
    }
    return source_root, splits


def process_one(
    json_path: Path,
    *,
    task: str,
    split: str,
    tokenizer,
    model,
    sae_weights: dict,
    source_root: Path,
    keep_npy: bool,
    force: bool,
    device: str,
) -> tuple[str, int]:
    """Encode one example. Returns (status, n_active_features)."""
    import torch

    out_npz = json_path.parent / f"{json_path.stem}.sae.npz"
    if out_npz.exists() and not force:
        return ("skip", -1)

    example = json.loads(json_path.read_text(encoding="utf-8"))
    try:
        prefix_ids, cot_ids = build_prompt_parts(
            task, example, tokenizer, source_root, split,
        )
    except NotImplementedError as e:
        return (f"unsupported:{e}", -1)

    if not cot_ids:
        return ("empty_cot", 0)

    full_ids = prefix_ids + cot_ids
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)

    # hidden_states has len == num_layers + 1; index 0 is the embedding, index
    # i+1 is the output of layer i (0-indexed). resid_post_layer_32 → index 33.
    hs = out.hidden_states[SAE_HIDDEN_LAYER + 1][0]  # (seq_len, d_model)
    assert hs.shape == (len(full_ids), D_MODEL), (
        f"unexpected hidden state shape {tuple(hs.shape)} "
        f"(expected ({len(full_ids)}, {D_MODEL}))"
    )

    # CoT-relative slice.
    cot_hidden = hs[len(prefix_ids):].to(torch.float16).cpu().numpy()
    assert cot_hidden.shape == (len(cot_ids), D_MODEL)

    if keep_npy:
        np.save(str(json_path.parent / f"{json_path.stem}.npy"), cot_hidden)

    result = sae_encode.encode_example(cot_hidden, sae_weights)
    np.savez_compressed(str(out_npz), **result)

    return ("wrote", int(result["active_feature_ids"].size))


def run_split(
    *,
    task: str,
    source_root: Path,
    split_alias: str,
    split_dir_name: str,
    tokenizer,
    model,
    sae_weights: dict,
    keep_npy: bool,
    force: bool,
    limit: int | None,
    device: str,
) -> tuple[int, int, int, list[str]]:
    """Process one (task, split) pair. Returns (seen, wrote, skipped, errors)."""
    split_dir = source_root / split_dir_name
    if not split_dir.is_dir():
        print(f"  [{task}/{split_alias}] split dir not found: {split_dir}", flush=True)
        return (0, 0, 0, [])

    files = sorted(split_dir.glob("*.json"))
    if limit is not None:
        files = files[:limit]

    print(f"\n[{task}/{split_alias}] {len(files)} examples in {split_dir}", flush=True)
    seen = wrote = skipped = 0
    errors: list[str] = []
    n_active_samples: list[int] = []
    t0 = time.time()

    for i, jp in enumerate(files, 1):
        seen += 1
        try:
            status, n_active = process_one(
                jp,
                task=task,
                split=split_dir_name,
                tokenizer=tokenizer,
                model=model,
                sae_weights=sae_weights,
                source_root=source_root,
                keep_npy=keep_npy,
                force=force,
                device=device,
            )
        except Exception as e:
            errors.append(f"{jp.name}: {e}")
            print(f"  ({i}/{len(files)}) ERROR {jp.stem}: {e}", flush=True)
            continue

        if status == "skip":
            skipped += 1
            continue
        if status.startswith("unsupported"):
            errors.append(f"{jp.name}: {status}")
            print(f"  ({i}/{len(files)}) skip-unsupported {jp.stem}: {status[12:]}", flush=True)
            continue
        if status == "empty_cot":
            errors.append(f"{jp.name}: empty CoT")
            print(f"  ({i}/{len(files)}) skip-empty {jp.stem}", flush=True)
            continue

        wrote += 1
        n_active_samples.append(n_active)
        if i % 50 == 0 or i == len(files):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            print(
                f"  ({i}/{len(files)}) {jp.stem}.sae.npz n_active={n_active} "
                f"[{rate:.2f} ex/s, {elapsed:.0f}s elapsed]",
                flush=True,
            )

    summary = (
        f"min={min(n_active_samples) if n_active_samples else 'n/a'} "
        f"median={int(np.median(n_active_samples)) if n_active_samples else 'n/a'} "
        f"max={max(n_active_samples) if n_active_samples else 'n/a'}"
    )
    print(
        f"[{task}/{split_alias}] done in {time.time() - t0:.1f}s — "
        f"seen={seen} wrote={wrote} skipped={skipped} errors={len(errors)} "
        f"n_active: {summary}",
        flush=True,
    )
    return seen, wrote, skipped, errors


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--tasks", default="all",
        help="Comma-separated task names, or 'all' for all six Qwen tasks "
             "(default: %(default)s). Available: " + ",".join(ALL_QWEN_TASKS),
    )
    p.add_argument(
        "--splits", default="few-shot,test",
        help="Comma-separated split aliases (default: %(default)s).",
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id")
    p.add_argument(
        "--tokenizer", default=None,
        help="HF tokenizer id (defaults to --model)",
    )
    p.add_argument(
        "--keep-npy", action="store_true",
        help="Also persist the raw .npy residual stream alongside .sae.npz",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Overwrite existing .sae.npz sidecars",
    )
    p.add_argument(
        "--limit", type=int, default=None,
        help="Process at most N examples per split (for smoke-testing)",
    )
    p.add_argument(
        "--cpu-smoke", action="store_true",
        help="Skip model load and just exercise build_prompt_parts on the first "
             "example of each split (CPU-only debugging).",
    )
    args = p.parse_args()

    if args.tasks.strip().lower() == "all":
        tasks = list(ALL_QWEN_TASKS)
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        unknown = [t for t in tasks if t not in ALL_QWEN_TASKS]
        if unknown:
            p.error(f"unknown task(s): {unknown}; known: {ALL_QWEN_TASKS}")

    split_aliases = [s.strip() for s in args.splits.split(",") if s.strip()]

    # Validate task metadata + split dirs up front so we fail fast.
    task_specs = []
    for task in tasks:
        source_root, splits_map = _load_task_meta(task)
        for alias in split_aliases:
            if alias not in splits_map:
                p.error(f"unknown split alias {alias!r} for task {task!r}; "
                        f"known: {list(splits_map)}")
            d = source_root / splits_map[alias]
            if not d.is_dir():
                p.error(f"[{task}/{alias}] split dir not found: {d}")
        task_specs.append((task, source_root, splits_map))

    print(f"[plan] {len(tasks)} tasks × {len(split_aliases)} splits", flush=True)
    for task, source_root, splits_map in task_specs:
        for alias in split_aliases:
            d = source_root / splits_map[alias]
            n = len(list(d.glob("*.json")))
            print(f"  {task}/{alias}: {n} files in {d}", flush=True)

    if args.cpu_smoke:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer or args.model, trust_remote_code=True,
        )
        print("\n[cpu-smoke] tokenizing one example per (task,split):", flush=True)
        for task, source_root, splits_map in task_specs:
            for alias in split_aliases:
                d = source_root / splits_map[alias]
                files = sorted(d.glob("*.json"))
                if not files:
                    continue
                jp = files[0]
                example = json.loads(jp.read_text(encoding="utf-8"))
                try:
                    prefix_ids, cot_ids = build_prompt_parts(
                        task, example, tokenizer, source_root, splits_map[alias],
                    )
                    print(
                        f"  {task}/{alias} {jp.stem}: "
                        f"prefix={len(prefix_ids)}tok cot={len(cot_ids)}tok",
                        flush=True,
                    )
                except Exception as e:
                    print(f"  {task}/{alias} {jp.stem}: ERROR {e}", flush=True)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        p.error("CUDA not available; this extractor requires a GPU")

    device = "cuda"

    tokenizer_name = args.tokenizer or args.model
    print(f"[load] tokenizer: {tokenizer_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    print(f"[load] SAE weights (trainer_2)", flush=True)
    weights_dir = sae_encode.get_sae_weights_dir()
    sae_weights = sae_encode.load_sae_weights(weights_dir)

    print(f"[load] model: {args.model} (bf16, device_map=auto)", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[load] model ready in {time.time() - t0:.1f}s", flush=True)

    grand_seen = grand_wrote = grand_skipped = 0
    grand_errors: list[str] = []
    overall_t0 = time.time()
    for task, source_root, splits_map in task_specs:
        for alias in split_aliases:
            seen, wrote, skipped, errors = run_split(
                task=task,
                source_root=source_root,
                split_alias=alias,
                split_dir_name=splits_map[alias],
                tokenizer=tokenizer,
                model=model,
                sae_weights=sae_weights,
                keep_npy=args.keep_npy,
                force=args.force,
                limit=args.limit,
                device=device,
            )
            grand_seen += seen
            grand_wrote += wrote
            grand_skipped += skipped
            grand_errors.extend(f"[{task}/{alias}] " + e for e in errors)

    print(
        f"\n[grand] elapsed={time.time() - overall_t0:.1f}s "
        f"seen={grand_seen} wrote={grand_wrote} skipped={grand_skipped} "
        f"errors={len(grand_errors)}",
        flush=True,
    )
    if grand_errors:
        print("[grand] first 20 errors:", flush=True)
        for line in grand_errors[:20]:
            print(f"  {line}", flush=True)


if __name__ == "__main__":
    main()
