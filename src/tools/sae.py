#!/usr/bin/env python3
"""SAE feature inspection tool for the interpretability agent.

Usage:
    sae search <query> [--n N]
    sae feature <feature_id>
    sae top-features <example_id> [--n N] [--last-k K] [--no-few-shot-stats]
    sae diff-features [--n N] [--last-k K]
    sae discriminate [--n N] [--last-k K] [--min-active K] [--cv-folds N]
    sae validate --positive FID,FID --negative FID,FID [--threshold T] [--last-k K]
    sae precompute
"""

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# Ensure src/ is on the path for sibling imports (tools.sae_encode)
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def get_paths():
    """Resolve standard paths from environment variables.

    `run_few_shot_dir` is where the strategy agent's sampled few-shot JSONs
    (and their `.sae.npz` sidecars, copied by scaffold.populate_few_shot*)
    actually live. That's the canonical location at run time; the static
    `data/<task>/few-shot/` path is kept for backwards-compat with tasks that
    pre-populate few-shot at ingest time.
    """
    scaffold_root = os.environ.get("SCAFFOLD_ROOT")
    task_name = os.environ.get("AGENT_TASK")
    if not scaffold_root or not task_name:
        print("Error: SCAFFOLD_ROOT and AGENT_TASK must be set.", file=sys.stderr)
        sys.exit(1)
    scaffold_root = Path(scaffold_root)
    run_dir_env = os.environ.get("AGENT_RUN_DIR")
    run_few_shot_dir = (Path(run_dir_env) / "strategy" / "few-shot") if run_dir_env else None
    return {
        "scaffold_root": scaffold_root,
        "task_name": task_name,
        "data_dir": scaffold_root / "data" / task_name,
        "few_shot_dir": scaffold_root / "data" / task_name / "few-shot",
        "test_dir": scaffold_root / "data" / task_name / "test",
        "run_few_shot_dir": run_few_shot_dir,
        "sae_dir": scaffold_root / "src" / "tools" / "qwen_sae",
    }


def _active_few_shot_dir(paths: dict) -> Path | None:
    """Pick the few-shot directory that actually exists at runtime.

    Prefers the per-run sampled set (AGENT_RUN_DIR/strategy/few-shot), falls
    back to the legacy static data/<task>/few-shot.
    """
    for d in (paths.get("run_few_shot_dir"), paths.get("few_shot_dir")):
        if d is not None and d.exists():
            return d
    return None


def _check_test_agent_scope(example_id: str) -> None:
    """Test agents may only look up their own assigned example."""
    if os.environ.get("AGENT_TYPE") != "test":
        return
    assigned = os.environ.get("AGENT_EXAMPLE_ID")
    if assigned and assigned != example_id:
        print(
            f"Error: test agent can only query its own example "
            f"(AGENT_EXAMPLE_ID={assigned!r}), not {example_id!r}",
            file=sys.stderr,
        )
        sys.exit(1)


def load_labels(sae_dir: Path) -> dict[str, str]:
    """Load feature_id -> label mapping."""
    labels_path = sae_dir / "feature_labels.json"
    with open(labels_path) as f:
        return json.load(f)


def load_label_frequencies(sae_dir: Path) -> dict[str, float]:
    """Load feature_id -> activation_freq from CSV."""
    csv_path = sae_dir / "feature_labels.csv"
    freqs = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            freqs[str(row["feature_id"])] = float(row["activation_freq"])
    return freqs


# ── search ──────────────────────────────────────────────────────────────────

def score_label(query_words: list[str], label: str) -> float:
    """Score a label against query words by case-insensitive token overlap."""
    label_words = set(re.findall(r"[a-z0-9]+", label.lower()))
    hits = sum(1 for w in query_words if w in label_words)
    if hits == 0:
        return 0.0
    return hits / len(query_words)


def cmd_search(args):
    paths = get_paths()
    labels = load_labels(paths["sae_dir"])
    query = " ".join(args.query)
    query_words = [w.lower() for w in re.split(r"\s+", query) if len(w) >= 2]

    if not query_words:
        print("Error: query must contain at least one word (2+ chars).", file=sys.stderr)
        sys.exit(1)

    # Score all labeled features
    scored = []
    for fid, label in labels.items():
        s = score_label(query_words, label)
        if s > 0:
            scored.append((fid, label, s))

    scored.sort(key=lambda x: -x[2])
    top_n = scored[: args.n]

    if not top_n:
        print(f"No features matched query: '{query}'")
        return

    # Write CSV
    sanitized = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")[:40]
    csv_path = Path(f"sae_search_{sanitized}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_id", "score", "label"])
        for fid, label, s in top_n:
            writer.writerow([fid, f"{s:.2f}", label])

    # Print summary
    print(f"Search: '{query}' — {len(scored)} matches, showing top {len(top_n)}")
    print(f"Results saved to {csv_path}")
    print()
    for fid, label, s in top_n:
        truncated = label[:80] + "..." if len(label) > 80 else label
        print(f"  [{fid:>5}] ({s:.2f}) {truncated}")


# ── feature ─────────────────────────────────────────────────────────────────

def load_sae_npz(npz_path: Path) -> dict:
    """Load a precomputed .sae.npz file."""
    import numpy as np
    data = np.load(str(npz_path))
    out = {
        "active_feature_ids": data["active_feature_ids"],
        "max_per_feature": data["max_per_feature"],
        "argmax_per_feature": data["argmax_per_feature"],
    }
    if "seq_len" in data.files:
        seq = data["seq_len"]
        out["seq_len"] = int(seq[0] if getattr(seq, "ndim", 0) else seq)
    elif len(out["argmax_per_feature"]):
        # Legacy sidecars did not store sequence length. This fallback makes
        # --last-k usable, but it can only approximate the true CoT end.
        out["seq_len"] = int(out["argmax_per_feature"].max()) + 1
        out["seq_len_is_fallback"] = True
    else:
        out["seq_len"] = 0
        out["seq_len_is_fallback"] = True
    return out


def find_feature_in_npz(npz_data: dict, feature_id: int) -> tuple[float, int] | None:
    """Look up a feature in precomputed data. Returns (max_val, argmax_pos) or None."""
    import numpy as np
    ids = npz_data["active_feature_ids"]
    idx = np.searchsorted(ids, feature_id)
    if idx < len(ids) and ids[idx] == feature_id:
        return float(npz_data["max_per_feature"][idx]), int(npz_data["argmax_per_feature"][idx])
    return None


def ensure_cached(npy_path: Path) -> Path | None:
    """Ensure .sae.npz exists for a .npy file, lazy-computing if needed."""
    npz_path = npy_path.with_suffix(".sae.npz")
    if npz_path.exists():
        return npz_path
    # Lazy precompute
    try:
        from tools.sae_encode import precompute_single_locked
        return precompute_single_locked(npy_path)
    except Exception as e:
        print(f"  Warning: could not encode {npy_path.name}: {e}", file=sys.stderr)
        return None


def cmd_feature(args):
    paths = get_paths()
    feature_id = int(args.feature_id)
    labels = load_labels(paths["sae_dir"])
    label = labels.get(str(feature_id), "(unlabeled)")

    few_shot_dir = _active_few_shot_dir(paths)
    if few_shot_dir is None:
        print(
            "Error: few-shot directory not found. Tried "
            f"{paths.get('run_few_shot_dir')} and {paths.get('few_shot_dir')}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Collect results across few-shot examples
    rows = []
    json_files = sorted(few_shot_dir.glob("*.json"))
    for json_file in json_files:
        example_id = json_file.stem
        sae_npz = few_shot_dir / f"{example_id}.sae.npz"
        npy_path = few_shot_dir / f"{example_id}.npy"

        if sae_npz.exists():
            npz_path = sae_npz
        elif npy_path.exists():
            npz_path = ensure_cached(npy_path)
        else:
            rows.append({"example_id": example_id, "label": _get_example_label(json_file),
                         "max_activation": "N/A", "peak_token_pos": "N/A"})
            continue

        if npz_path is None:
            rows.append({"example_id": example_id, "label": _get_example_label(json_file),
                         "max_activation": "error", "peak_token_pos": "error"})
            continue

        npz_data = load_sae_npz(npz_path)
        result = find_feature_in_npz(npz_data, feature_id)
        ex_label = _get_example_label(json_file)

        if result:
            max_val, argmax_pos = result
            rows.append({"example_id": example_id, "label": ex_label,
                         "max_activation": f"{max_val:.4f}", "peak_token_pos": str(argmax_pos)})
        else:
            rows.append({"example_id": example_id, "label": ex_label,
                         "max_activation": "0", "peak_token_pos": "-"})

    # Write CSV
    csv_path = Path(f"sae_feature_{feature_id}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["example_id", "label", "max_activation", "peak_token_pos"])
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    print(f"Feature {feature_id}: {label}")
    print(f"Activation across {len(rows)} few-shot examples → {csv_path}")
    print()
    print(f"  {'example_id':<50} {'label':>5} {'max_act':>10} {'peak_tok':>8}")
    print(f"  {'─' * 50} {'─' * 5} {'─' * 10} {'─' * 8}")
    for r in rows:
        eid = r["example_id"][:50]
        print(f"  {eid:<50} {r['label']:>5} {r['max_activation']:>10} {r['peak_token_pos']:>8}")


def _get_example_label(json_path: Path) -> str:
    """Read the ground-truth label from an example JSON."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        return data.get("label", "?")
    except Exception:
        return "?"


def _last_k_start(npz_data: dict, last_k: int | None) -> int | None:
    if last_k is None:
        return None
    if last_k <= 0:
        raise ValueError("--last-k must be positive")
    return max(0, int(npz_data.get("seq_len", 0)) - last_k)


def _sidecar_for_example(few_shot_dir: Path, example_id: str) -> Path | None:
    sae_npz = few_shot_dir / f"{example_id}.sae.npz"
    if sae_npz.exists():
        return sae_npz
    npy_path = few_shot_dir / f"{example_id}.npy"
    if npy_path.exists():
        return ensure_cached(npy_path)
    return None


# ── few-shot pool gathering (shared by discriminate / validate / top-features stats) ──

def _gather_few_shot_activations(few_shot_dir: Path, last_k: int | None):
    """Build a per-example dict of feature → max activation across the few-shot pool.

    Returns
    -------
    examples : list[dict]
        One entry per few-shot example with a usable label and SAE sidecar.
        Each is ``{"id": str, "label": int, "fids": np.ndarray, "acts": np.ndarray}``
        where ``fids[i]`` and ``acts[i]`` are the active feature IDs and their
        max activations after the optional ``--last-k`` window filter.
    fallback_windows : int
        Count of legacy sidecars whose ``seq_len`` had to be inferred.
    skipped : int
        Examples skipped (missing label or missing sidecar).
    """
    import numpy as np
    examples = []
    fallback_windows = 0
    skipped = 0
    for json_file in sorted(few_shot_dir.glob("*.json")):
        raw_label = _get_example_label(json_file)
        try:
            ex_label = int(raw_label)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if ex_label not in (0, 1):
            skipped += 1
            continue
        npz_path = _sidecar_for_example(few_shot_dir, json_file.stem)
        if npz_path is None:
            skipped += 1
            continue
        npz_data = load_sae_npz(npz_path)
        if npz_data.get("seq_len_is_fallback") and last_k is not None:
            fallback_windows += 1
        start_pos = _last_k_start(npz_data, last_k)
        active_ids = npz_data["active_feature_ids"]
        max_vals = npz_data["max_per_feature"]
        argmax_pos = npz_data["argmax_per_feature"]
        if start_pos is not None:
            keep = argmax_pos >= start_pos
            active_ids = active_ids[keep]
            max_vals = max_vals[keep]
        examples.append({
            "id": json_file.stem,
            "label": ex_label,
            "fids": np.asarray(active_ids, dtype=np.int64),
            "acts": np.asarray(max_vals, dtype=np.float32),
        })
    return examples, fallback_windows, skipped


def _per_feature_class_stats(examples, restrict_fids=None):
    """Compute per-feature yes/no statistics from gathered few-shot examples.

    Treats absent features as activation 0 (so means and standard deviations
    are taken over **all** examples in each class, not just those where the
    feature was active).

    Returns ``{fid: {...}}`` with keys::

        n_yes_active, n_no_active     # examples in each class where fid was active
        mean_yes, mean_no             # mean of max activation across all examples
        sd_yes, sd_no                 # population stddev over all examples
        cohens_d                      # (mean_yes - mean_no) / pooled_sd
        rate_yes, rate_no             # active count / total examples in each class
    """
    import numpy as np
    n_yes = sum(1 for e in examples if e["label"] == 1)
    n_no = sum(1 for e in examples if e["label"] == 0)
    if n_yes == 0 or n_no == 0:
        return {}, n_yes, n_no

    seen: set[int] = set()
    if restrict_fids is None:
        for e in examples:
            seen.update(int(f) for f in e["fids"].tolist())
    else:
        seen.update(int(f) for f in restrict_fids)

    # Build acts_yes[fid] and acts_no[fid] as zero-padded arrays per class.
    acts_yes_idx = [i for i, e in enumerate(examples) if e["label"] == 1]
    acts_no_idx = [i for i, e in enumerate(examples) if e["label"] == 0]
    fid_to_acts = {fid: {"yes": np.zeros(n_yes, dtype=np.float32),
                          "no": np.zeros(n_no, dtype=np.float32)}
                   for fid in seen}
    for slot_yes, ex_idx in enumerate(acts_yes_idx):
        e = examples[ex_idx]
        for fid, act in zip(e["fids"].tolist(), e["acts"].tolist()):
            if fid in fid_to_acts:
                fid_to_acts[fid]["yes"][slot_yes] = act
    for slot_no, ex_idx in enumerate(acts_no_idx):
        e = examples[ex_idx]
        for fid, act in zip(e["fids"].tolist(), e["acts"].tolist()):
            if fid in fid_to_acts:
                fid_to_acts[fid]["no"][slot_no] = act

    stats = {}
    for fid in seen:
        ya = fid_to_acts[fid]["yes"]
        na = fid_to_acts[fid]["no"]
        mean_yes = float(ya.mean())
        mean_no = float(na.mean())
        sd_yes = float(ya.std())
        sd_no = float(na.std())
        pooled_var = (sd_yes ** 2 + sd_no ** 2) / 2
        if pooled_var > 0:
            d = (mean_yes - mean_no) / (pooled_var ** 0.5)
        else:
            d = 0.0
        stats[fid] = {
            "n_yes_active": int((ya > 0).sum()),
            "n_no_active": int((na > 0).sum()),
            "mean_yes": mean_yes,
            "mean_no": mean_no,
            "sd_yes": sd_yes,
            "sd_no": sd_no,
            "cohens_d": d,
            "rate_yes": float((ya > 0).mean()),
            "rate_no": float((na > 0).mean()),
        }
    return stats, n_yes, n_no


# ── top-features ────────────────────────────────────────────────────────────

def resolve_example_npz(example_id: str, paths: dict) -> Path | None:
    """Find the .sae.npz for an example, checking multiple locations.

    Order of preference:
      1. The run's strategy/few-shot/ (sampled at run time by scaffold.py).
      2. Agent's CWD (test agents get `example.sae.npz` copied in).
      3. Legacy static data/<task>/{few-shot,test}/ layout.
    """
    cwd = Path.cwd()
    candidates = []
    if paths.get("run_few_shot_dir") is not None:
        candidates.append(paths["run_few_shot_dir"] / f"{example_id}.sae.npz")
    candidates.extend([
        cwd / f"{example_id}.sae.npz",
        cwd / "example.sae.npz",
        paths["few_shot_dir"] / f"{example_id}.sae.npz",
        paths["test_dir"] / f"{example_id}.sae.npz",
    ])

    for p in candidates:
        if p.exists():
            return p

    # Try lazy precompute from .npy (rare — only if an activation file
    # exists without its encoded sidecar, e.g. during dev).
    npy_candidates = []
    if paths.get("run_few_shot_dir") is not None:
        npy_candidates.append(paths["run_few_shot_dir"] / f"{example_id}.npy")
    npy_candidates.extend([
        cwd / f"{example_id}.npy",
        cwd / "example.npy",
        paths["few_shot_dir"] / f"{example_id}.npy",
        paths["test_dir"] / f"{example_id}.npy",
    ])
    for npy_path in npy_candidates:
        if npy_path.exists():
            result = ensure_cached(npy_path)
            if result:
                return result

    return None


def cmd_top_features(args):
    paths = get_paths()
    example_id = args.example_id
    n = args.n
    _check_test_agent_scope(example_id)
    labels = load_labels(paths["sae_dir"])

    npz_path = resolve_example_npz(example_id, paths)
    if npz_path is None:
        print(f"Error: no cached SAE activations found for '{example_id}'.", file=sys.stderr)
        print("Checked: few-shot/, test/, and current directory.", file=sys.stderr)
        print("Ensure .npy activation files exist and run `sae precompute` if needed.", file=sys.stderr)
        sys.exit(1)

    import numpy as np
    npz_data = load_sae_npz(npz_path)
    active_ids = npz_data["active_feature_ids"]
    max_vals = npz_data["max_per_feature"]
    argmax_pos = npz_data["argmax_per_feature"]
    start_pos = _last_k_start(npz_data, args.last_k)

    if start_pos is not None:
        keep = argmax_pos >= start_pos
        active_ids = active_ids[keep]
        max_vals = max_vals[keep]
        argmax_pos = argmax_pos[keep]

    # Sort by max activation descending
    order = np.argsort(-max_vals)
    top_indices = order[:n]

    # Optional: enrich each top feature with class statistics from the few-shot pool
    fs_stats: dict[int, dict] = {}
    fs_totals = (0, 0)
    fs_warning = None
    if not args.no_few_shot_stats:
        few_shot_dir = _active_few_shot_dir(paths)
        if few_shot_dir is not None:
            top_fids = [int(active_ids[i]) for i in top_indices]
            fs_examples, _fb, _skipped = _gather_few_shot_activations(few_shot_dir, args.last_k)
            fs_stats, n_yes, n_no = _per_feature_class_stats(fs_examples, restrict_fids=top_fids)
            fs_totals = (n_yes, n_no)
            if not fs_examples or n_yes == 0 or n_no == 0:
                fs_warning = "few-shot pool unusable for stats (missing labels or sidecars)"
        else:
            fs_warning = "few-shot pool not found; per-feature class stats unavailable"

    rows = []
    for idx in top_indices:
        fid = int(active_ids[idx])
        label = labels.get(str(fid), "(unlabeled)")
        row = {
            "feature_id": fid,
            "max_activation": f"{max_vals[idx]:.4f}",
            "peak_token_pos": int(argmax_pos[idx]),
            "label": label,
        }
        st = fs_stats.get(fid)
        if st is not None:
            row.update({
                "fewshot_yes_active": f"{st['n_yes_active']}/{fs_totals[0]}",
                "fewshot_no_active": f"{st['n_no_active']}/{fs_totals[1]}",
                "fewshot_cohens_d": f"{st['cohens_d']:+.2f}",
            })
        elif not args.no_few_shot_stats and fs_warning is None:
            row.update({"fewshot_yes_active": "", "fewshot_no_active": "", "fewshot_cohens_d": ""})
        rows.append(row)

    # Write CSV
    suffix = f"_last{args.last_k}" if args.last_k is not None else ""
    csv_path = Path(f"sae_top_features_{example_id[:60]}{suffix}.csv")
    fieldnames = ["feature_id", "max_activation", "peak_token_pos", "label"]
    if fs_stats:
        fieldnames += ["fewshot_yes_active", "fewshot_no_active", "fewshot_cohens_d"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    if args.last_k is None:
        print(f"Top {len(rows)} SAE features for '{example_id}' → {csv_path}")
    else:
        print(
            f"Top {len(rows)} SAE features for '{example_id}' with peak in "
            f"last {args.last_k} CoT tokens → {csv_path}"
        )
        print(f"  (window: peak_token_pos >= {start_pos}, seq_len={npz_data.get('seq_len')})")
        if npz_data.get("seq_len_is_fallback"):
            print("  Warning: legacy sidecar lacks seq_len; window end is approximate.")
    print(f"  (source: {npz_path})")
    print()
    if fs_warning:
        print(f"  Note: {fs_warning}")
    if fs_stats:
        n_yes, n_no = fs_totals
        print(f"  Few-shot pool stats: yes={n_yes}, no={n_no} examples (Cohen's d on max-activation, treating absent as 0)")
        print()
        header = (
            f"  {'fid':>6} {'max_act':>10} {'peak_tok':>8}  "
            f"{'y_act':>5} {'n_act':>5} {'d':>6}  label"
        )
        print(header)
        print(f"  {'─' * 6} {'─' * 10} {'─' * 8}  {'─' * 5} {'─' * 5} {'─' * 6}  {'─' * 50}")
        for r in rows:
            truncated = r["label"][:60] + "..." if len(r["label"]) > 60 else r["label"]
            y = r.get("fewshot_yes_active", "")
            n = r.get("fewshot_no_active", "")
            d = r.get("fewshot_cohens_d", "")
            print(
                f"  {r['feature_id']:>6} {r['max_activation']:>10} {r['peak_token_pos']:>8}  "
                f"{y:>5} {n:>5} {d:>6}  {truncated}"
            )
    else:
        print()
        print(f"  {'fid':>6} {'max_act':>10} {'peak_tok':>8}  label")
        print(f"  {'─' * 6} {'─' * 10} {'─' * 8}  {'─' * 50}")
        for r in rows:
            truncated = r["label"][:60] + "..." if len(r["label"]) > 60 else r["label"]
            print(f"  {r['feature_id']:>6} {r['max_activation']:>10} {r['peak_token_pos']:>8}  {truncated}")


# ── diff-features ───────────────────────────────────────────────────────────

def cmd_diff_features(args):
    paths = get_paths()
    labels = load_labels(paths["sae_dir"])
    few_shot_dir = _active_few_shot_dir(paths)
    if few_shot_dir is None:
        print(
            "Error: few-shot directory not found. Tried "
            f"{paths.get('run_few_shot_dir')} and {paths.get('few_shot_dir')}.",
            file=sys.stderr,
        )
        sys.exit(1)

    stats: dict[int, dict] = {}
    totals = {0: 0, 1: 0}
    used = skipped = 0
    fallback_windows = 0

    for json_file in sorted(few_shot_dir.glob("*.json")):
        raw_label = _get_example_label(json_file)
        try:
            ex_label = int(raw_label)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if ex_label not in (0, 1):
            skipped += 1
            continue

        npz_path = _sidecar_for_example(few_shot_dir, json_file.stem)
        if npz_path is None:
            skipped += 1
            continue

        npz_data = load_sae_npz(npz_path)
        if npz_data.get("seq_len_is_fallback") and args.last_k is not None:
            fallback_windows += 1
        start_pos = _last_k_start(npz_data, args.last_k)
        active_ids = npz_data["active_feature_ids"]
        max_vals = npz_data["max_per_feature"]
        argmax_pos = npz_data["argmax_per_feature"]
        if start_pos is not None:
            keep = argmax_pos >= start_pos
            active_ids = active_ids[keep]
            max_vals = max_vals[keep]
            argmax_pos = argmax_pos[keep]

        totals[ex_label] += 1
        used += 1
        for fid, val, pos in zip(active_ids, max_vals, argmax_pos):
            fid_i = int(fid)
            rec = stats.setdefault(
                fid_i,
                {
                    "active": {0: 0, 1: 0},
                    "sum": {0: 0.0, 1: 0.0},
                    "max": {0: 0.0, 1: 0.0},
                    "peak_sum": {0: 0, 1: 0},
                },
            )
            val_f = float(val)
            rec["active"][ex_label] += 1
            rec["sum"][ex_label] += val_f
            rec["max"][ex_label] = max(rec["max"][ex_label], val_f)
            rec["peak_sum"][ex_label] += int(pos)

    if totals[0] == 0 or totals[1] == 0:
        print(
            f"Error: need at least one label=0 and label=1 example with SAE sidecars; "
            f"got totals={totals}, skipped={skipped}",
            file=sys.stderr,
        )
        sys.exit(1)

    rows = []
    for fid, rec in stats.items():
        active0 = rec["active"][0]
        active1 = rec["active"][1]
        if active0 + active1 < args.min_active:
            continue
        rate0 = active0 / totals[0]
        rate1 = active1 / totals[1]
        mean0 = rec["sum"][0] / totals[0]
        mean1 = rec["sum"][1] / totals[1]
        peak0 = rec["peak_sum"][0] / active0 if active0 else ""
        peak1 = rec["peak_sum"][1] / active1 if active1 else ""
        score = abs(rate1 - rate0) + (abs(mean1 - mean0) / max(mean1, mean0, 1.0))
        rows.append({
            "feature_id": fid,
            "score": score,
            "label1_active_rate": rate1,
            "label0_active_rate": rate0,
            "label1_mean_max": mean1,
            "label0_mean_max": mean0,
            "label1_active_count": active1,
            "label0_active_count": active0,
            "label1_peak_mean": peak1,
            "label0_peak_mean": peak0,
            "feature_label": labels.get(str(fid), "(unlabeled)"),
        })

    rows.sort(
        key=lambda r: (
            -r["score"],
            -abs(r["label1_active_rate"] - r["label0_active_rate"]),
            -abs(r["label1_mean_max"] - r["label0_mean_max"]),
        )
    )
    rows = rows[: args.n]

    suffix = f"_last{args.last_k}" if args.last_k is not None else ""
    csv_path = Path(f"sae_diff_features{suffix}.csv")
    fieldnames = [
        "feature_id",
        "score",
        "label1_active_rate",
        "label0_active_rate",
        "label1_mean_max",
        "label0_mean_max",
        "label1_active_count",
        "label0_active_count",
        "label1_peak_mean",
        "label0_peak_mean",
        "feature_label",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = dict(r)
            for k in ("score", "label1_active_rate", "label0_active_rate", "label1_mean_max", "label0_mean_max"):
                out[k] = f"{out[k]:.4f}"
            for k in ("label1_peak_mean", "label0_peak_mean"):
                if out[k] != "":
                    out[k] = f"{out[k]:.1f}"
            writer.writerow(out)

    window = f" with peak in last {args.last_k} CoT tokens" if args.last_k is not None else ""
    print(f"Top {len(rows)} label-differential SAE features{window} → {csv_path}")
    print(f"  examples used: label=1 n={totals[1]}, label=0 n={totals[0]}, skipped={skipped}")
    if fallback_windows:
        print(f"  Warning: {fallback_windows} legacy sidecar(s) lacked seq_len; last-k windows are approximate.")
    print()
    print(f"  {'fid':>6} {'score':>7} {'act1':>5} {'act0':>5} {'mean1':>8} {'mean0':>8}  label")
    print(f"  {'-' * 6} {'-' * 7} {'-' * 5} {'-' * 5} {'-' * 8} {'-' * 8}  {'-' * 50}")
    for r in rows[: min(len(rows), 30)]:
        label = r["feature_label"]
        truncated = label[:60] + "..." if len(label) > 60 else label
        print(
            f"  {r['feature_id']:>6} {r['score']:>7.3f} "
            f"{r['label1_active_rate']:>5.2f} {r['label0_active_rate']:>5.2f} "
            f"{r['label1_mean_max']:>8.2f} {r['label0_mean_max']:>8.2f}  {truncated}"
        )


# ── discriminate ────────────────────────────────────────────────────────────

def cmd_discriminate(args):
    """Rank few-shot SAE features by Cohen's d effect size between yes and no.

    Treats unobserved features as activation 0, so means and standard deviations
    are taken across **all** few-shot examples in each class. Optional
    leave-one-out CV stability column reports the fraction of LOO splits where
    the feature appears in the unfiltered top-N by ``|d|``.
    """
    paths = get_paths()
    labels = load_labels(paths["sae_dir"])
    few_shot_dir = _active_few_shot_dir(paths)
    if few_shot_dir is None:
        print(
            "Error: few-shot directory not found. Tried "
            f"{paths.get('run_few_shot_dir')} and {paths.get('few_shot_dir')}.",
            file=sys.stderr,
        )
        sys.exit(1)

    examples, fallback_windows, skipped = _gather_few_shot_activations(few_shot_dir, args.last_k)
    stats, n_yes, n_no = _per_feature_class_stats(examples)
    if n_yes == 0 or n_no == 0:
        print(
            f"Error: need at least one label=0 and label=1 example with SAE sidecars; "
            f"got n_yes={n_yes}, n_no={n_no}, skipped={skipped}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Apply min-active filter, rank by |d|
    eligible = [
        fid for fid, s in stats.items()
        if (s["n_yes_active"] + s["n_no_active"]) >= args.min_active
    ]
    eligible.sort(key=lambda fid: -abs(stats[fid]["cohens_d"]))
    ranked = eligible[: args.n]

    # Optional leave-one-out CV stability: for each example, recompute top-N and
    # see if each ranked feature reappears. Stability = fraction of LOO splits
    # where the feature is in top-N by |d|.
    cv_stability: dict[int, float] | None = None
    if args.cv_folds is not None and args.cv_folds > 0:
        n_total = len(examples)
        if args.cv_folds > n_total:
            print(f"  Warning: cv_folds={args.cv_folds} > n_examples={n_total}; using LOO instead.")
            folds = n_total
        else:
            folds = args.cv_folds
        # Use leave-one-out only when folds == n_total; otherwise round-robin
        cv_appearances = {fid: 0 for fid in ranked}
        for fold in range(folds):
            held_out = [i for i in range(n_total) if i % folds == fold] if folds < n_total else [fold]
            kept = [examples[i] for i in range(n_total) if i not in held_out]
            kept_stats, k_yes, k_no = _per_feature_class_stats(kept)
            if k_yes == 0 or k_no == 0:
                continue
            kept_eligible = [
                fid for fid, s in kept_stats.items()
                if (s["n_yes_active"] + s["n_no_active"]) >= args.min_active
            ]
            kept_eligible.sort(key=lambda fid: -abs(kept_stats[fid]["cohens_d"]))
            top_set = set(kept_eligible[: args.n])
            for fid in ranked:
                if fid in top_set:
                    cv_appearances[fid] += 1
        cv_stability = {fid: cv_appearances[fid] / folds for fid in ranked}

    # Write CSV
    suffix = f"_last{args.last_k}" if args.last_k is not None else ""
    csv_path = Path(f"sae_discriminate{suffix}.csv")
    fieldnames = [
        "feature_id", "cohens_d", "rate_yes", "rate_no",
        "mean_yes", "mean_no", "n_yes_active", "n_no_active", "feature_label",
    ]
    if cv_stability is not None:
        fieldnames.insert(2, "cv_stability")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fid in ranked:
            s = stats[fid]
            row = {
                "feature_id": fid,
                "cohens_d": f"{s['cohens_d']:+.4f}",
                "rate_yes": f"{s['rate_yes']:.3f}",
                "rate_no": f"{s['rate_no']:.3f}",
                "mean_yes": f"{s['mean_yes']:.3f}",
                "mean_no": f"{s['mean_no']:.3f}",
                "n_yes_active": s["n_yes_active"],
                "n_no_active": s["n_no_active"],
                "feature_label": labels.get(str(fid), "(unlabeled)"),
            }
            if cv_stability is not None:
                row["cv_stability"] = f"{cv_stability[fid]:.2f}"
            writer.writerow(row)

    # Print summary
    window = f" (last-{args.last_k}-token window)" if args.last_k is not None else ""
    print(f"Top {len(ranked)} discriminating SAE features{window} → {csv_path}")
    print(f"  examples: yes={n_yes}, no={n_no}; skipped={skipped}; min_active={args.min_active}")
    if fallback_windows:
        print(f"  Warning: {fallback_windows} legacy sidecar(s) lacked seq_len; window is approximate.")
    print()
    cv_col = f"  {'cv':>5}" if cv_stability is not None else ""
    print(f"  {'fid':>6} {'d':>7}{cv_col} {'r_yes':>5} {'r_no':>5} {'m_yes':>7} {'m_no':>7}  label")
    print(f"  {'─' * 6} {'─' * 7}{('  ─────' if cv_stability else '')} {'─' * 5} {'─' * 5} {'─' * 7} {'─' * 7}  {'─' * 50}")
    for fid in ranked:
        s = stats[fid]
        lbl = labels.get(str(fid), "(unlabeled)")
        truncated = lbl[:60] + "..." if len(lbl) > 60 else lbl
        cv_cell = f"  {cv_stability[fid]:>5.2f}" if cv_stability is not None else ""
        print(
            f"  {fid:>6} {s['cohens_d']:>+7.3f}{cv_cell} "
            f"{s['rate_yes']:>5.2f} {s['rate_no']:>5.2f} "
            f"{s['mean_yes']:>7.2f} {s['mean_no']:>7.2f}  {truncated}"
        )


# ── validate ────────────────────────────────────────────────────────────────

def _parse_fid_list(s: str | None) -> list[int]:
    if not s:
        return []
    out: list[int] = []
    for part in s.replace(" ", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            print(f"Error: invalid feature ID '{part}'", file=sys.stderr)
            sys.exit(1)
    return out


def cmd_validate(args):
    """Evaluate a candidate decision rule (positive/negative feature lists +
    activation threshold) on the few-shot pool via leave-one-out.

    Decision per example::
        pos_hits = #(positive features with max_activation > threshold)
        neg_hits = #(negative features with max_activation > threshold)
        if pos_hits > neg_hits          → predict yes
        elif neg_hits > pos_hits        → predict no
        else (tie, including 0–0)       → predict ``--tie-default`` (default 'no')

    Reports overall accuracy, TP/TN/FP/FN, and per-fold (LOO) breakdown.
    """
    paths = get_paths()
    few_shot_dir = _active_few_shot_dir(paths)
    if few_shot_dir is None:
        print(
            "Error: few-shot directory not found. Tried "
            f"{paths.get('run_few_shot_dir')} and {paths.get('few_shot_dir')}.",
            file=sys.stderr,
        )
        sys.exit(1)

    pos_fids = set(_parse_fid_list(args.positive))
    neg_fids = set(_parse_fid_list(args.negative))
    if not pos_fids and not neg_fids:
        print("Error: at least one of --positive or --negative must be non-empty.", file=sys.stderr)
        sys.exit(1)

    examples, fallback_windows, skipped = _gather_few_shot_activations(few_shot_dir, args.last_k)
    if not examples:
        print(f"Error: no usable few-shot examples found (skipped={skipped}).", file=sys.stderr)
        sys.exit(1)

    threshold = args.threshold
    tie_default_label = 1 if args.tie_default == "yes" else 0

    def predict(ex) -> int:
        fids = ex["fids"].tolist()
        acts = ex["acts"].tolist()
        active_above = {int(f) for f, a in zip(fids, acts) if float(a) > threshold}
        pos_hits = len(active_above & pos_fids)
        neg_hits = len(active_above & neg_fids)
        if pos_hits > neg_hits:
            return 1
        if neg_hits > pos_hits:
            return 0
        return tie_default_label

    # LOO: prediction for each example is independent of training set under this
    # fixed rule (no learning), so loop is just the per-example evaluation.
    tp = tn = fp = fn = 0
    rows = []
    for ex in examples:
        pred = predict(ex)
        gt = ex["label"]
        if pred == 1 and gt == 1: tp += 1
        elif pred == 0 and gt == 0: tn += 1
        elif pred == 1 and gt == 0: fp += 1
        elif pred == 0 and gt == 1: fn += 1
        rows.append({
            "example_id": ex["id"],
            "ground_truth": gt,
            "prediction": pred,
            "correct": int(pred == gt),
        })

    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else 0.0
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    gmean2 = tpr * tnr

    suffix = f"_last{args.last_k}" if args.last_k is not None else ""
    csv_path = Path(f"sae_validate{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["example_id", "ground_truth", "prediction", "correct"])
        writer.writeheader()
        writer.writerows(rows)

    window = f" (last-{args.last_k}-token window)" if args.last_k is not None else ""
    print(f"Validation on few-shot pool{window} → {csv_path}")
    print(f"  rule: pos_features={sorted(pos_fids)}  neg_features={sorted(neg_fids)}")
    print(f"        threshold={threshold}  tie_default={args.tie_default}")
    if fallback_windows:
        print(f"  Warning: {fallback_windows} legacy sidecar(s) lacked seq_len; window is approximate.")
    print()
    print(f"  n={n}  acc={acc*100:.1f}%  TPR={tpr:.2f}  TNR={tnr:.2f}  gmean²={gmean2:.3f}")
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    if skipped:
        print(f"  (skipped {skipped} few-shot examples lacking labels or sidecars)")


# ── precompute ──────────────────────────────────────────────────────────────

def cmd_precompute(args):
    paths = get_paths()
    from tools.sae_encode import precompute_task
    precompute_task(paths["scaffold_root"], paths["task_name"])


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SAE feature inspection tool")
    sub = parser.add_subparsers(dest="command")

    # search
    sp = sub.add_parser("search", help="Search feature labels by keyword")
    sp.add_argument("query", nargs="+", help="Search query (keywords)")
    sp.add_argument("--n", type=int, default=20, help="Number of results (default: 20)")

    # feature
    sp = sub.add_parser("feature", help="Show feature activation across few-shot examples")
    sp.add_argument("feature_id", help="SAE feature ID (integer)")

    # top-features
    sp = sub.add_parser("top-features", help="Show top activating features for an example")
    sp.add_argument("example_id", help="Example ID (filename stem)")
    sp.add_argument("--n", type=int, default=20, help="Number of top features (default: 20)")
    sp.add_argument(
        "--last-k",
        type=int,
        default=None,
        help="Only show features whose peak activation is in the final K CoT tokens",
    )
    sp.add_argument(
        "--no-few-shot-stats",
        action="store_true",
        help="Skip the few-shot pool yes/no/Cohen's-d enrichment (faster, smaller output)",
    )

    # diff-features
    sp = sub.add_parser(
        "diff-features",
        help="Rank few-shot SAE features that differ between label=1 and label=0",
    )
    sp.add_argument("--n", type=int, default=30, help="Number of features to show (default: 30)")
    sp.add_argument(
        "--min-active",
        type=int,
        default=2,
        help="Minimum total active examples required for a feature (default: 2)",
    )
    sp.add_argument(
        "--last-k",
        type=int,
        default=None,
        help="Only compare features whose peak activation is in each example's final K CoT tokens",
    )

    # discriminate
    sp = sub.add_parser(
        "discriminate",
        help="Rank few-shot SAE features by Cohen's d effect size between yes and no",
    )
    sp.add_argument("--n", type=int, default=30, help="Number of features to show (default: 30)")
    sp.add_argument(
        "--min-active",
        type=int,
        default=2,
        help="Minimum total active examples (yes+no) required for a feature (default: 2)",
    )
    sp.add_argument(
        "--last-k",
        type=int,
        default=None,
        help="Restrict to features whose peak activation is in each example's final K CoT tokens",
    )
    sp.add_argument(
        "--cv-folds",
        type=int,
        default=None,
        help="If set, also compute LOO/k-fold stability (fraction of splits where the feature stays in top-N).",
    )

    # validate
    sp = sub.add_parser(
        "validate",
        help="Score a candidate decision rule (positive/negative feature lists + threshold) on the few-shot pool",
    )
    sp.add_argument(
        "--positive",
        type=str,
        default="",
        help="Comma-separated feature IDs that vote yes when active above threshold",
    )
    sp.add_argument(
        "--negative",
        type=str,
        default="",
        help="Comma-separated feature IDs that vote no when active above threshold",
    )
    sp.add_argument(
        "--threshold",
        type=float,
        default=4.0,
        help="Activation threshold (max_per_feature) for counting a feature as firing (default: 4.0)",
    )
    sp.add_argument(
        "--tie-default",
        choices=("yes", "no"),
        default="no",
        help="Prediction when pos_hits == neg_hits, including 0-0 (default: 'no')",
    )
    sp.add_argument(
        "--last-k",
        type=int,
        default=None,
        help="Only count features whose peak activation is in the final K CoT tokens",
    )

    # precompute
    sub.add_parser("precompute", help="Precompute SAE activations for all .npy files")

    args = parser.parse_args()

    if args.command == "search":
        cmd_search(args)
    elif args.command == "feature":
        cmd_feature(args)
    elif args.command == "top-features":
        cmd_top_features(args)
    elif args.command == "diff-features":
        cmd_diff_features(args)
    elif args.command == "discriminate":
        cmd_discriminate(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "precompute":
        cmd_precompute(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
