"""Shrink raw test sets in `datasets/<id>/qwen-3-32b/test/` to <=100 examples each.

Constraints (priority order):
  1. exactly 50 yes / 50 no after shrink (assuming both classes have >=50 available)
  2. for any prompt, |#yes - #no| <= 1
  3. maximize distinct prompts represented
  4. cross-class length balance (task-specific strategy below)

Per-task:
  4 user_preference_sycophancy   prompt = anecdote_id           length proxy: len(thinking) string
  5 stanford_hint                prompt = question_id           length proxy: sum len of `thinking[*].text`
  6 atypical_answer              prompt = question_id           length proxy: len(cot_content)
  7 atypical_cot_length          prompt = question_id           length proxy: token_length (direct)

Tasks 4, 5, 6 use length strategy "cross": iterative greedy swap within each
(prompt, label) bucket to minimize |mean_yes_len - mean_no_len|.

Task 7 uses strategy "inner_edge" (Reading B): in each yes-bucket pick the
shortest rollouts; in each no-bucket pick the longest. This brings the two
class means as close as possible given that the label IS length-derived.

Tasks 1, 2, 3 already <=100 examples — they are no-ops if invoked.

Usage:
    python src/shrink_test_sets.py            # in-place delete extra files
    python src/shrink_test_sets.py --dry-run  # report only
    python src/shrink_test_sets.py --task 7   # restrict to one dataset id
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean


REPO_ROOT = Path(__file__).resolve().parents[1]


def _str_len(field):
    def f(ex):
        v = ex.get(field, "")
        return len(v) if isinstance(v, str) else 0
    return f


def _list_text_len(field):
    def f(ex):
        v = ex.get(field)
        if not isinstance(v, list):
            return 0
        return sum(len(item.get("text", "")) for item in v if isinstance(item, dict))
    return f


def _direct_int(field):
    def f(ex):
        v = ex.get(field)
        return int(v) if v is not None else 0
    return f


TASK_CONFIG = {
    "4": {
        "name": "user_preference_sycophancy",
        "label_map": {"sycophantic": 1, "nonsycophantic": 0},
        "prompt_id_field": "anecdote_id",
        "length_proxy": _str_len("thinking"),
        "length_strategy": "cross",
        "per_prompt_diff_max": 1,
    },
    "5": {
        "name": "stanford_hint",
        "label_map": {"hint_following": 1, "independent": 0},
        "prompt_id_field": "question_id",
        "length_proxy": _list_text_len("thinking"),
        "length_strategy": "cross",
        "per_prompt_diff_max": 1,
    },
    "6": {
        "name": "atypical_answer",
        "label_map": {"minority": 1, "majority": 0},
        "prompt_id_field": "question_id",
        "length_proxy": _str_len("cot_content"),
        "length_strategy": "cross",
        "per_prompt_diff_max": 1,
    },
    "7": {
        "name": "atypical_cot_length",
        "label_map": {"long": 1, "short": 0},
        "prompt_id_field": "question_id",
        "length_proxy": _direct_int("token_length"),
        "length_strategy": "inner_edge",
        "per_prompt_diff_max": None,
    },
}


def _norm_label(raw, label_map):
    if raw in label_map:
        return int(label_map[raw])
    if isinstance(raw, int) and raw in (0, 1):
        return raw
    if isinstance(raw, str) and raw in ("0", "1"):
        return int(raw)
    return None


def _load_examples(test_dir: Path, cfg: dict) -> list[dict]:
    out = []
    skipped = 0
    label_map = cfg["label_map"]
    pid_field = cfg["prompt_id_field"]
    length_fn = cfg["length_proxy"]
    for f in sorted(test_dir.glob("*.json")):
        with f.open(encoding="utf-8") as fh:
            d = json.load(fh)
        lbl = _norm_label(d.get("label"), label_map)
        if lbl is None:
            skipped += 1
            continue
        out.append({
            "path": f,
            "prompt": d.get(pid_field),
            "label": lbl,
            "length": length_fn(d),
        })
    if skipped:
        print(f"  warn: {skipped} files had unmappable labels and were excluded")
    return out


def _select_counts(examples, target_per_class, rng, per_prompt_diff_max=1):
    """Return {(prompt, label): k}. Maximizes distinct prompts, then fills to
    50/50 evenly across prompts. If `per_prompt_diff_max` is not None, enforces
    |sel_y[p] - sel_n[p]| <= per_prompt_diff_max for every prompt; if None, no
    such constraint (only available rollouts cap each prompt).
    """
    avail = defaultdict(int)
    for ex in examples:
        avail[(ex["prompt"], ex["label"])] += 1
    prompts = sorted({p for p, _ in avail})
    y_avail = {p: avail.get((p, 1), 0) for p in prompts}
    n_avail = {p: avail.get((p, 0), 0) for p in prompts}

    sel_y = {p: 0 for p in prompts}
    sel_n = {p: 0 for p in prompts}

    def can_add_y(p):
        if sel_y[p] >= y_avail[p]:
            return False
        if per_prompt_diff_max is None:
            return True
        return abs((sel_y[p] + 1) - sel_n[p]) <= per_prompt_diff_max

    def can_add_n(p):
        if sel_n[p] >= n_avail[p]:
            return False
        if per_prompt_diff_max is None:
            return True
        return abs(sel_y[p] - (sel_n[p] + 1)) <= per_prompt_diff_max

    # Process single-class prompts first to avoid wasting their unique slots
    # to a global-quota race (random order within each tier).
    P_y = [p for p in prompts if y_avail[p] > 0 and n_avail[p] == 0]
    P_n = [p for p in prompts if n_avail[p] > 0 and y_avail[p] == 0]
    P_b = [p for p in prompts if y_avail[p] > 0 and n_avail[p] > 0]
    rng.shuffle(P_y); rng.shuffle(P_n); rng.shuffle(P_b)
    P = P_y + P_n + P_b

    # Phase 1: diversity. Add ≥1 example per prompt; choose class to balance
    # global quota.
    for p in P:
        ry = target_per_class - sum(sel_y.values())
        rn = target_per_class - sum(sel_n.values())
        if ry <= 0 and rn <= 0:
            break
        cy_ok = can_add_y(p) and ry > 0
        cn_ok = can_add_n(p) and rn > 0
        if cy_ok and cn_ok:
            if ry >= rn:
                sel_y[p] += 1
            else:
                sel_n[p] += 1
        elif cy_ok:
            sel_y[p] += 1
        elif cn_ok:
            sel_n[p] += 1

    # Phase 2: fill remaining quotas. Prefer prompts with fewer current examples
    # to spread evenly.
    while True:
        ry = target_per_class - sum(sel_y.values())
        rn = target_per_class - sum(sel_n.values())
        if ry <= 0 and rn <= 0:
            break
        order = sorted(P, key=lambda p: sel_y[p] + sel_n[p])
        progress = False
        for p in order:
            ry = target_per_class - sum(sel_y.values())
            rn = target_per_class - sum(sel_n.values())
            if ry <= 0 and rn <= 0:
                break
            cy_ok = can_add_y(p) and ry > 0
            cn_ok = can_add_n(p) and rn > 0
            if cy_ok and cn_ok:
                if ry >= rn:
                    sel_y[p] += 1
                else:
                    sel_n[p] += 1
                progress = True
            elif cy_ok:
                sel_y[p] += 1
                progress = True
            elif cn_ok:
                sel_n[p] += 1
                progress = True
        if not progress:
            break

    counts = {}
    for p in prompts:
        if sel_y[p]:
            counts[(p, 1)] = sel_y[p]
        if sel_n[p]:
            counts[(p, 0)] = sel_n[p]
    return counts


def _pick_within_buckets(examples, counts, strategy):
    by_bucket = defaultdict(list)
    for ex in examples:
        by_bucket[(ex["prompt"], ex["label"])].append(ex)

    if strategy == "inner_edge":
        kept = []
        for (p, lbl), k in counts.items():
            pool = sorted(by_bucket[(p, lbl)], key=lambda e: e["length"])
            if lbl == 1:
                kept.extend(pool[:k])
            else:
                kept.extend(pool[-k:])
        return kept

    cur = {}
    for (p, lbl), k in counts.items():
        pool = sorted(by_bucket[(p, lbl)], key=lambda e: e["length"])
        n = len(pool)
        start = max(0, (n - k) // 2)
        cur[(p, lbl)] = list(pool[start:start + k])

    yes_size = sum(1 for (_, l), v in cur.items() for _ in v if l == 1)
    no_size = sum(1 for (_, l), v in cur.items() for _ in v if l == 0)

    def class_size(lbl):
        return yes_size if lbl == 1 else no_size

    def class_sum(lbl):
        return sum(e["length"] for (_, l), v in cur.items() if l == lbl for e in v)

    yes_sum = class_sum(1)
    no_sum = class_sum(0)

    MAX_ITER = 5000
    for _ in range(MAX_ITER):
        diff = (yes_sum / yes_size) - (no_sum / no_size)
        if abs(diff) < 1e-9:
            break
        best_improve = 0.0
        best_swap = None
        for (p, lbl), in_list in cur.items():
            pool = by_bucket[(p, lbl)]
            in_paths = {e["path"] for e in in_list}
            sign = 1 if lbl == 1 else -1
            csize = class_size(lbl)
            for in_e in in_list:
                for out_e in pool:
                    if out_e["path"] in in_paths:
                        continue
                    delta_class = (out_e["length"] - in_e["length"]) / csize
                    new_diff = diff + sign * delta_class
                    improve = abs(diff) - abs(new_diff)
                    if improve > best_improve + 1e-9:
                        best_improve = improve
                        best_swap = ((p, lbl), in_e, out_e)
        if best_swap is None:
            break
        key, in_e, out_e = best_swap
        cur[key].remove(in_e)
        cur[key].append(out_e)
        if key[1] == 1:
            yes_sum += out_e["length"] - in_e["length"]
        else:
            no_sum += out_e["length"] - in_e["length"]

    kept = []
    for v in cur.values():
        kept.extend(v)
    return kept


def shrink_one(test_dir: Path, cfg: dict, target: int, seed: int, dry_run: bool):
    examples = _load_examples(test_dir, cfg)
    n_total = len(examples)
    if n_total <= target:
        return {"noop": True, "n_before": n_total, "kept": n_total}

    target_per = target // 2
    rng = random.Random(seed)

    counts = _select_counts(examples, target_per, rng, cfg.get("per_prompt_diff_max", 1))

    yes_total = sum(k for (_, l), k in counts.items() if l == 1)
    no_total = sum(k for (_, l), k in counts.items() if l == 0)
    if yes_total != target_per or no_total != target_per:
        print(f"  warn: could not hit {target_per}/{target_per} — got yes={yes_total}, no={no_total}")

    kept = _pick_within_buckets(examples, counts, cfg["length_strategy"])

    yes_kept = [e for e in kept if e["label"] == 1]
    no_kept = [e for e in kept if e["label"] == 0]
    distinct_prompts = len({e["prompt"] for e in kept})
    per_prompt = defaultdict(int)
    for e in kept:
        per_prompt[e["prompt"]] += 1
    max_per_prompt = max(per_prompt.values())
    mean_yes_len = mean(e["length"] for e in yes_kept) if yes_kept else 0.0
    mean_no_len = mean(e["length"] for e in no_kept) if no_kept else 0.0

    keep_paths = {e["path"] for e in kept}
    deleted = 0
    if not dry_run:
        for f in test_dir.glob("*.json"):
            if f not in keep_paths:
                f.unlink()
                deleted += 1
    else:
        deleted = n_total - len(kept)

    return {
        "noop": False,
        "n_before": n_total,
        "kept": len(kept),
        "yes": len(yes_kept),
        "no": len(no_kept),
        "distinct_prompts": distinct_prompts,
        "max_per_prompt": max_per_prompt,
        "mean_yes_len": mean_yes_len,
        "mean_no_len": mean_no_len,
        "len_diff": mean_yes_len - mean_no_len,
        "deleted": deleted,
    }


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--task", action="append", default=None,
                    help="Restrict to specific dataset id(s); default: 4,5,6,7")
    args = ap.parse_args(argv)

    tasks = args.task or list(TASK_CONFIG.keys())

    for tid in tasks:
        cfg = TASK_CONFIG[tid]
        test_dir = REPO_ROOT / "datasets" / tid / "qwen-3-32b" / "test"
        print(f"\n=== task {tid}  {cfg['name']} ===")
        print(f"  dir: {test_dir}")
        if not test_dir.exists():
            print("  NOT FOUND, skipping")
            continue
        result = shrink_one(test_dir, cfg, args.target, args.seed, args.dry_run)
        if result["noop"]:
            print(f"  noop: {result['n_before']} <= {args.target}")
            continue
        print(f"  before:  n={result['n_before']}")
        print(f"  kept:    n={result['kept']}  (yes={result['yes']}, no={result['no']})")
        print(f"  prompts: distinct={result['distinct_prompts']}  max_per_prompt={result['max_per_prompt']}")
        print(f"  length:  mean_yes={result['mean_yes_len']:.1f}  mean_no={result['mean_no_len']:.1f}  |Δ|={abs(result['len_diff']):.1f}")
        print(f"  deleted: {result['deleted']} files{'  [DRY RUN]' if args.dry_run else ''}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
