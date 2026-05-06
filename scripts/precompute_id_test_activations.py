#!/usr/bin/env python3
"""Extract Qwen3-32B layer-32 SAE activations for the ID-test splits.

Sibling of src/precompute_activations.py with two key differences:
  - examples come from data/<task>/id_test/*.json (not datasets/<id>/<model>/<split>/)
  - the prompt lookup root is datasets copy/<id>/<model>/, so prompts are read
    from datasets copy/<id>/prompts/test/<qid>.json

Sidecars (.sae.npz) are written next to the id_test JSONs.

gemma_self_deletion_clean is intentionally skipped — it uses a different
model and SAE.

Usage (on a GPU host with Qwen3-32B installed):
    python3 scripts/precompute_id_test_activations.py --tasks all
    python3 scripts/precompute_id_test_activations.py --tasks reasoning_termination --limit 3
    python3 scripts/precompute_id_test_activations.py --cpu-smoke   # CPU dry run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
ID_SOURCE_ROOT = REPO_ROOT / "datasets copy"

sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "tools"))

# Reuse the per-example extractor + SAE encoder from the OOD pipeline.
import precompute_activations as pa  # noqa: E402
import sae_encode  # noqa: E402
from _task_io import build_prompt_parts  # noqa: E402

DEFAULT_MODEL = pa.DEFAULT_MODEL


# Each entry: (task_dir_under_data, dataset_id, model_subdir, split_for_prompts)
# split_for_prompts="test" because the ID source's prompts dir is prompts/test/.
ID_TASK_SPECS = [
    ("reasoning_termination",      "1", "qwen-3-32b", "test"),
    ("followup_confidence",        "3", "qwen-3-32b", "test"),
    ("user_preference_sycophancy", "4", "qwen-3-32b", "test"),
    ("stanford_hint_clean",        "5", "qwen-3-32b", "test"),
    ("atypical_answer",            "6", "qwen-3-32b", "test"),
    ("atypical_cot_length",        "7", "qwen-3-32b", "test"),
    # gemma_self_deletion_clean intentionally omitted (different model/SAE).
]


def _id_test_dir(task: str) -> Path:
    return DATA / task / "id_test"


def _id_source_root(dataset_id: str, model: str) -> Path:
    return ID_SOURCE_ROOT / dataset_id / model


def run_id_split(
    *,
    task: str,
    source_root: Path,
    split: str,
    examples_dir: Path,
    tokenizer,
    model,
    sae_weights: dict,
    keep_npy: bool,
    force: bool,
    limit: int | None,
    device: str,
) -> tuple[int, int, int, list[str]]:
    files = sorted(examples_dir.glob("*.json"))
    files = [f for f in files if not f.name.startswith("_")]
    if limit is not None:
        files = files[:limit]

    print(f"\n[{task}/id_test] {len(files)} examples in {examples_dir}", flush=True)
    seen = wrote = skipped = 0
    errors: list[str] = []
    n_active_samples: list[int] = []
    t0 = time.time()

    for i, jp in enumerate(files, 1):
        seen += 1
        try:
            status, n_active = pa.process_one(
                jp,
                task=task,
                split=split,
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
        if i % 25 == 0 or i == len(files):
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
        f"[{task}/id_test] done in {time.time()-t0:.1f}s — "
        f"seen={seen} wrote={wrote} skipped={skipped} errors={len(errors)} "
        f"n_active: {summary}",
        flush=True,
    )
    return seen, wrote, skipped, errors


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    known = [t[0] for t in ID_TASK_SPECS]
    p.add_argument(
        "--tasks", default="all",
        help="Comma-separated task names (default: all). Available: " + ",".join(known),
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--keep-npy", action="store_true",
                   help="Also persist raw .npy activations alongside .sae.npz")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing .sae.npz sidecars")
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N examples per task (smoke-test)")
    p.add_argument("--cpu-smoke", action="store_true",
                   help="Skip model load; just prompt-build the first example of "
                        "each selected task (catches missing-prompt issues).")
    args = p.parse_args()

    if args.tasks.strip().lower() == "all":
        selected = list(ID_TASK_SPECS)
    else:
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        unknown = wanted - set(known)
        if unknown:
            p.error(f"unknown task(s): {sorted(unknown)}; known: {known}")
        selected = [s for s in ID_TASK_SPECS if s[0] in wanted]

    # Validate id_test dirs up front. Prompts dir may legitimately be absent
    # for tasks 4 (prompt inlined in example) and 5_clean (prompt parts
    # inlined); build_prompt_parts will only read the file when needed.
    plan = []
    for task, did, model_dir, split in selected:
        ex_dir = _id_test_dir(task)
        src_root = _id_source_root(did, model_dir)
        if not ex_dir.is_dir():
            p.error(f"[{task}] id_test dir missing: {ex_dir}")
        plan.append((task, src_root, split, ex_dir))

    print(f"[plan] {len(plan)} tasks", flush=True)
    for task, src_root, split, ex_dir in plan:
        n = len([f for f in ex_dir.glob("*.json") if not f.name.startswith("_")])
        prompts_dir = src_root.parent / "prompts" / split
        prompts_note = f"prompts via {prompts_dir}" if prompts_dir.is_dir() else "(prompts inlined)"
        print(f"  {task}: {n} files — {prompts_note}", flush=True)

    if args.cpu_smoke:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer or args.model, trust_remote_code=True,
        )
        print("\n[cpu-smoke] building prompts on one example per task:", flush=True)
        any_err = False
        for task, src_root, split, ex_dir in plan:
            files = sorted(f for f in ex_dir.glob("*.json") if not f.name.startswith("_"))
            if not files:
                continue
            jp = files[0]
            example = json.loads(jp.read_text(encoding="utf-8"))
            try:
                prefix_ids, cot_ids = build_prompt_parts(
                    task, example, tokenizer, src_root, split,
                )
                print(f"  {task} {jp.stem}: prefix={len(prefix_ids)}tok cot={len(cot_ids)}tok",
                      flush=True)
            except Exception as e:
                any_err = True
                print(f"  {task} {jp.stem}: ERROR {e}", flush=True)
        return 1 if any_err else 0

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        p.error("CUDA not available; this extractor requires a GPU")
    device = "cuda"

    tokenizer_name = args.tokenizer or args.model
    print(f"[load] tokenizer: {tokenizer_name}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    print("[load] SAE weights (trainer_2)", flush=True)
    sae_weights = sae_encode.load_sae_weights(sae_encode.get_sae_weights_dir())

    print(f"[load] model: {args.model} (bf16, device_map=auto)", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    model.eval()
    print(f"[load] model ready in {time.time()-t0:.1f}s", flush=True)

    grand_seen = grand_wrote = grand_skipped = 0
    grand_errors: list[str] = []
    overall_t0 = time.time()
    for task, src_root, split, ex_dir in plan:
        seen, wrote, skipped, errors = run_id_split(
            task=task,
            source_root=src_root,
            split=split,
            examples_dir=ex_dir,
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
        grand_errors.extend(f"[{task}] " + e for e in errors)

    print(
        f"\n[grand] elapsed={time.time()-overall_t0:.1f}s "
        f"seen={grand_seen} wrote={grand_wrote} skipped={grand_skipped} "
        f"errors={len(grand_errors)}",
        flush=True,
    )
    if grand_errors:
        print("[grand] first 20 errors:", flush=True)
        for line in grand_errors[:20]:
            print(f"  {line}", flush=True)
    return 0 if not grand_errors else 2


if __name__ == "__main__":
    sys.exit(main())
