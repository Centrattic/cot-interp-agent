#!/usr/bin/env python3
"""SAE feature inspection tool for the interpretability agent.

Usage:
    sae search <query> [--n N]
    sae feature <feature_id>
    sae top-features <example_id> [--n N]
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
    """Score a label against query words by case-insensitive word overlap."""
    label_lower = label.lower()
    hits = sum(1 for w in query_words if w in label_lower)
    if hits == 0:
        return 0.0
    # Bonus for matching more query words, normalized by query length
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
    return {
        "active_feature_ids": data["active_feature_ids"],
        "max_per_feature": data["max_per_feature"],
        "argmax_per_feature": data["argmax_per_feature"],
    }


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

    # Sort by max activation descending
    order = np.argsort(-max_vals)
    top_indices = order[:n]

    rows = []
    for idx in top_indices:
        fid = int(active_ids[idx])
        label = labels.get(str(fid), "(unlabeled)")
        rows.append({
            "feature_id": fid,
            "max_activation": f"{max_vals[idx]:.4f}",
            "peak_token_pos": int(argmax_pos[idx]),
            "label": label,
        })

    # Write CSV
    csv_path = Path(f"sae_top_features_{example_id[:60]}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["feature_id", "max_activation", "peak_token_pos", "label"])
        writer.writeheader()
        writer.writerows(rows)

    # Print summary
    print(f"Top {len(rows)} SAE features for '{example_id}' → {csv_path}")
    print(f"  (source: {npz_path})")
    print()
    print(f"  {'fid':>6} {'max_act':>10} {'peak_tok':>8}  label")
    print(f"  {'─' * 6} {'─' * 10} {'─' * 8}  {'─' * 50}")
    for r in rows:
        truncated = r["label"][:60] + "..." if len(r["label"]) > 60 else r["label"]
        print(f"  {r['feature_id']:>6} {r['max_activation']:>10} {r['peak_token_pos']:>8}  {truncated}")


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

    # precompute
    sub.add_parser("precompute", help="Precompute SAE activations for all .npy files")

    args = parser.parse_args()

    if args.command == "search":
        cmd_search(args)
    elif args.command == "feature":
        cmd_feature(args)
    elif args.command == "top-features":
        cmd_top_features(args)
    elif args.command == "precompute":
        cmd_precompute(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
