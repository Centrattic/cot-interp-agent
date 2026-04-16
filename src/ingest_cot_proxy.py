#!/usr/bin/env python3
"""Ingest a cot-proxy-tasks dataset into the scaffold's data/ layout.

Source layout (per dataset_id in ../cot-proxy-tasks/datasets/):
    datasets/<id>/<model>/{train,test,val,ood_train,ood_test}/*.json

Scaffold layout this script writes:
    data/<task_name>/
        metadata.json
        few-shot/   (balanced sample from the train split)
        test/       (copy of the test split)

Usage:
    python ingest_cot_proxy.py \
        --source ../cot-proxy-tasks/datasets/2/gemma-3-27b \
        --task gemma_self_deletion \
        --description "Predict whether gemma-3-27b would self-delete reliably given one more turn." \
        --few-shot-per-class 5 \
        --seed 0

Test split is copied wholesale (all files). Few-shot is sampled balanced across
label={0,1} with a fixed seed for reproducibility.
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path


SCAFFOLD_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = SCAFFOLD_ROOT / "data"


def sample_balanced(train_dir: Path, per_class: int, seed: int) -> list[Path]:
    """Return `per_class` label=0 files and `per_class` label=1 files, deterministic given seed."""
    by_label: dict[int, list[Path]] = {0: [], 1: []}
    for f in sorted(train_dir.glob("*.json")):
        with open(f, encoding="utf-8") as fh:
            data = json.load(fh)
        lbl = data.get("label")
        if lbl in by_label:
            by_label[lbl].append(f)

    rng = random.Random(seed)
    picked: list[Path] = []
    for lbl in (0, 1):
        pool = by_label[lbl]
        if len(pool) < per_class:
            print(f"Error: only {len(pool)} label={lbl} files available, need {per_class}")
            sys.exit(1)
        picked.extend(rng.sample(pool, per_class))
    return picked


def copy_split(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(src_dir.glob("*.json")):
        shutil.copy2(f, dst_dir / f.name)
        # Also copy companion .npy if present
        npy = src_dir.parent / src_dir.name / f"{f.stem}.npy"
        if npy.exists():
            shutil.copy2(npy, dst_dir / npy.name)
        n += 1
    return n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", required=True, help="Path to <dataset_id>/<model>/ dir")
    p.add_argument("--task", required=True, help="Task name (folder in data/)")
    p.add_argument("--description", required=True, help="Task description (label semantics)")
    p.add_argument("--few-shot-per-class", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-split", default="test", help="Which subdir under source to use as test")
    p.add_argument("--train-split", default="train", help="Which subdir under source to sample few-shot from")
    args = p.parse_args()

    source = Path(args.source).resolve()
    train_dir = source / args.train_split
    test_dir = source / args.test_split

    if not train_dir.is_dir() or not test_dir.is_dir():
        print(f"Error: missing {train_dir} or {test_dir}")
        sys.exit(1)

    task_dir = DATA_DIR / args.task
    few_shot_dst = task_dir / "few-shot"
    test_dst = task_dir / "test"

    if few_shot_dst.exists() or test_dst.exists():
        print(f"Note: {task_dir} already exists. Clearing few-shot/ and test/ to re-ingest.")
        if few_shot_dst.exists():
            shutil.rmtree(few_shot_dst)
        if test_dst.exists():
            shutil.rmtree(test_dst)

    # Sample balanced few-shot and copy
    few_shot_dst.mkdir(parents=True, exist_ok=True)
    picked = sample_balanced(train_dir, args.few_shot_per_class, args.seed)
    for src in picked:
        shutil.copy2(src, few_shot_dst / src.name)
        npy = train_dir / f"{src.stem}.npy"
        if npy.exists():
            shutil.copy2(npy, few_shot_dst / npy.name)
    print(f"Wrote {len(picked)} few-shot examples "
          f"({args.few_shot_per_class} per class) to {few_shot_dst}")

    # Copy full test split
    test_count = copy_split(test_dir, test_dst)
    print(f"Copied {test_count} test examples to {test_dst}")

    # Write metadata.json
    meta = {
        "name": args.task,
        "description": args.description,
        "source": str(source),
        "train_split": args.train_split,
        "test_split": args.test_split,
        "few_shot_per_class": args.few_shot_per_class,
        "seed": args.seed,
    }
    with open(task_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {task_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
