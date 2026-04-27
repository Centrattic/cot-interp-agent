#!/usr/bin/env python3
"""Extract layer-32 residual-stream activations for the termination task and
encode them through the Qwen3-32B SAE (trainer_2).

For every example JSON in `data/termination/qwen-3-32b/<split>/`, this:
  1. Reconstructs the exact model input the rollout was sampled from, by
     reusing `_task_io.build_prompt_parts` (chat template + `<think>\\n` +
     cot_prefix).
  2. Runs a single Qwen3-32B forward pass and grabs the residual stream at
     layer 32 (post-residual) at the CoT-token positions only.
  3. Encodes those activations through the JumpReLU SAE in tools/sae_encode.py
     and writes a `<example_stem>.sae.npz` sidecar next to the JSON.

The intermediate `.npy` is not persisted by default (pass `--keep-npy` for
debugging). The `.sae.npz` sidecars are what the agent-facing `sae` tool
consumes.

Usage (on a GPU host with Qwen3-32B):
    python src/precompute_activations.py --splits test,ood_val
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCAFFOLD_ROOT = Path(__file__).resolve().parent.parent
TASK_NAME = "termination"
SOURCE_ROOT = SCAFFOLD_ROOT / "data" / "termination" / "qwen-3-32b"
SAE_HIDDEN_LAYER = 32  # resid_post_layer_32 → hidden_states[33] in HF
D_MODEL = 5120
DEFAULT_MODEL = "Qwen/Qwen3-32B"

# Pull in the shared prompt builder and the SAE encoder.
sys.path.insert(0, str(SCAFFOLD_ROOT / "src" / "tools"))
from _task_io import build_prompt_parts  # noqa: E402
import sae_encode  # noqa: E402


def process_one(
    json_path: Path,
    *,
    split: str,
    tokenizer,
    model,
    sae_weights: dict,
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
    prefix_ids, cot_ids = build_prompt_parts(
        TASK_NAME, example, tokenizer, SOURCE_ROOT, split,
    )
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


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--splits", default="test,ood_val",
        help="Comma-separated raw split names under data/termination/qwen-3-32b/ "
             "(default: %(default)s)",
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
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        p.error("CUDA not available; this extractor requires a GPU")

    device = "cuda"
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    # Validate split dirs up front.
    for split in splits:
        d = SOURCE_ROOT / split
        if not d.is_dir():
            p.error(f"split dir not found: {d}")

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

    # Device the *input* should go on. With device_map="auto" over a single GPU
    # the whole model lives on cuda:0; input_ids on the same device is fine.
    input_device = device

    total_seen = 0
    total_wrote = 0
    n_active_samples: list[int] = []
    for split in splits:
        split_dir = SOURCE_ROOT / split
        files = sorted(split_dir.glob("*.json"))
        if args.limit is not None:
            files = files[: args.limit]
        print(f"\n[{split}] {len(files)} examples", flush=True)

        t_split = time.time()
        for i, jp in enumerate(files, 1):
            total_seen += 1
            try:
                status, n_active = process_one(
                    jp,
                    split=split,
                    tokenizer=tokenizer,
                    model=model,
                    sae_weights=sae_weights,
                    keep_npy=args.keep_npy,
                    force=args.force,
                    device=input_device,
                )
            except Exception as e:
                print(f"  ({i}/{len(files)}) ERROR {jp.stem}: {e}", flush=True)
                continue

            if status == "skip":
                print(f"  ({i}/{len(files)}) skip {jp.stem} (exists)", flush=True)
                continue

            total_wrote += 1
            n_active_samples.append(n_active)
            print(
                f"  ({i}/{len(files)}) wrote {jp.stem}.sae.npz  "
                f"n_active={n_active}",
                flush=True,
            )

        print(f"[{split}] done in {time.time() - t_split:.1f}s", flush=True)

    print(
        f"\n[summary] seen={total_seen} wrote={total_wrote} "
        f"n_active stats: "
        f"min={min(n_active_samples) if n_active_samples else 'n/a'} "
        f"median={int(np.median(n_active_samples)) if n_active_samples else 'n/a'} "
        f"max={max(n_active_samples) if n_active_samples else 'n/a'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
