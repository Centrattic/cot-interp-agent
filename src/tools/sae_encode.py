#!/usr/bin/env python3
"""SAE encoding: download weights, apply JumpReLU, cache sparse activations."""

import fcntl
import json
import os
import sys
from pathlib import Path

import numpy as np


SAE_REPO_ID = "adamkarvonen/qwen3-32b-saes"
SAE_SUBDIR = "saes_Qwen_Qwen3-32B_batch_top_k/resid_post_layer_32/trainer_2"
CACHE_DIR = Path(os.environ.get("COT_CACHE_DIR") or (Path.home() / ".cache" / "cot-interp-agent"))

# Expected dimensions from metadata
D_MODEL = 5120
D_SAE = 65536


def get_sae_weights_dir() -> Path:
    """Return the local cache directory for SAE weights, downloading if needed."""
    weights_dir = CACHE_DIR / "sae-weights" / SAE_SUBDIR
    marker = weights_dir / ".download_complete"

    if marker.exists():
        return weights_dir

    weights_dir.mkdir(parents=True, exist_ok=True)

    # Use a lock to prevent concurrent downloads
    lock_path = weights_dir / ".download.lock"
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        # Re-check after acquiring lock
        if marker.exists():
            return weights_dir

        print(f"Downloading SAE weights from {SAE_REPO_ID}/{SAE_SUBDIR}...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=SAE_REPO_ID,
                allow_patterns=[f"{SAE_SUBDIR}/*"],
                local_dir=str(CACHE_DIR / "sae-weights"),
            )
            marker.touch()
            print("SAE weights downloaded successfully.")
        except Exception as e:
            print(f"Error downloading SAE weights: {e}", file=sys.stderr)
            raise

    return weights_dir


def load_sae_weights(weights_dir: Path) -> dict:
    """Load BatchTopK SAE encoder weights from the `ae.pt` checkpoint.

    The HF repo stores weights as a torch state_dict with keys
    {encoder.weight, encoder.bias, decoder.weight, b_dec, threshold, k}.
    We normalize to the `W_enc / b_enc / b_dec / threshold` schema the rest
    of the pipeline expects (W_enc in (d_model, d_sae) orientation).
    """
    import torch  # local import so non-GPU callers don't pay import cost

    pt_path = weights_dir / "ae.pt"
    if not pt_path.exists():
        pts = list(weights_dir.glob("*.pt"))
        if not pts:
            raise FileNotFoundError(
                f"No ae.pt (or .pt) in {weights_dir}. "
                f"Contents: {[f.name for f in weights_dir.iterdir()]}"
            )
        pt_path = pts[0]

    sd = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()

    missing = [k for k in ("encoder.weight", "encoder.bias", "b_dec", "threshold") if k not in sd]
    if missing:
        raise KeyError(
            f"missing keys in {pt_path}: {missing}; available: {list(sd.keys())}"
        )

    # encoder.weight is (d_sae, d_model); the rest of the pipeline wants
    # W_enc in (d_model, d_sae) so `x @ W_enc` shapes match.
    W_enc = sd["encoder.weight"].T.contiguous().cpu().numpy().astype(np.float32)
    b_enc = sd["encoder.bias"].cpu().numpy().astype(np.float32)
    b_dec = sd["b_dec"].cpu().numpy().astype(np.float32)
    threshold = float(sd["threshold"].cpu().item())  # scalar JumpReLU threshold

    cfg_path = weights_dir / "config.json"
    cfg = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)

    print(
        f"  W_enc: {W_enc.shape}, b_enc: {b_enc.shape}, b_dec: {b_dec.shape}, "
        f"threshold (scalar): {threshold:.4f}",
        file=sys.stderr,
    )

    return {
        "W_enc": W_enc,
        "b_enc": b_enc,
        "b_dec": b_dec,
        "threshold": threshold,
        "cfg": cfg,
    }


def encode_example(activations: np.ndarray, weights: dict) -> dict:
    """Apply JumpReLU SAE encoding to raw residual stream activations.

    Args:
        activations: shape (seq_len, d_model) raw residual stream
        weights: dict from load_sae_weights()

    Returns:
        dict with keys: active_feature_ids, max_per_feature, argmax_per_feature
    """
    W_enc = weights["W_enc"]
    b_enc = weights["b_enc"]
    b_dec = weights["b_dec"]
    threshold = weights["threshold"]

    x = activations.astype(np.float32)

    # Subtract decoder bias if present (standard SAELens preprocessing)
    if b_dec is not None:
        x = x - b_dec

    # Encode: pre_acts = x @ W_enc + b_enc
    pre_acts = x @ W_enc + b_enc  # (seq_len, d_sae)

    # JumpReLU: features = pre_acts * (pre_acts > threshold)
    if threshold is not None:
        features = pre_acts * (pre_acts > threshold)
    else:
        # Fallback to standard ReLU if no threshold
        features = np.maximum(pre_acts, 0)

    # Find features that fire at least once
    max_per_token = features.max(axis=0)  # (d_sae,)
    active_mask = max_per_token > 0
    active_ids = np.where(active_mask)[0].astype(np.int32)

    if len(active_ids) == 0:
        return {
            "active_feature_ids": np.array([], dtype=np.int32),
            "max_per_feature": np.array([], dtype=np.float32),
            "argmax_per_feature": np.array([], dtype=np.int32),
        }

    # For active features: max value and argmax token position
    active_features = features[:, active_ids]  # (seq_len, num_active)
    max_vals = active_features.max(axis=0).astype(np.float32)
    argmax_pos = active_features.argmax(axis=0).astype(np.int32)

    return {
        "active_feature_ids": active_ids,
        "max_per_feature": max_vals,
        "argmax_per_feature": argmax_pos,
    }


def precompute_single(npy_path: Path, weights: dict) -> Path | None:
    """Encode a single .npy file and save the .sae.npz cache.

    Returns the output path, or None if skipped.
    """
    out_path = npy_path.with_suffix(".sae.npz")
    if out_path.exists():
        return out_path

    activations = np.load(str(npy_path))
    if activations.ndim != 2 or activations.shape[1] != D_MODEL:
        print(
            f"  Warning: {npy_path.name} has shape {activations.shape}, "
            f"expected (seq_len, {D_MODEL}). Skipping.",
            file=sys.stderr,
        )
        return None

    result = encode_example(activations, weights)
    np.savez_compressed(str(out_path), **result)
    return out_path


def precompute_dir(dir_path: Path, weights: dict | None = None) -> int:
    """Batch-encode all .npy files in a directory.

    Args:
        dir_path: directory containing .npy files
        weights: pre-loaded SAE weights (loaded if None)

    Returns:
        number of files processed
    """
    npy_files = sorted(dir_path.glob("*.npy"))
    if not npy_files:
        return 0

    # Skip if all already cached
    uncached = [f for f in npy_files if not f.with_suffix(".sae.npz").exists()]
    if not uncached:
        print(f"  All {len(npy_files)} files already cached in {dir_path.name}/")
        return 0

    if weights is None:
        weights_dir = get_sae_weights_dir()
        weights = load_sae_weights(weights_dir)

    count = 0
    for npy_path in uncached:
        result = precompute_single(npy_path, weights)
        if result:
            count += 1
            print(f"  Encoded {npy_path.name} -> {result.name}")

    print(f"  Precomputed {count} file(s) in {dir_path.name}/")
    return count


def precompute_task(scaffold_root: Path, task_name: str):
    """Precompute SAE activations for all .npy files in a task's data dirs."""
    data_dir = scaffold_root / "data" / task_name
    few_shot_dir = data_dir / "few-shot"
    test_dir = data_dir / "test"

    has_npy = False
    for d in [few_shot_dir, test_dir]:
        if d.exists() and list(d.glob("*.npy")):
            has_npy = True
            break

    if not has_npy:
        print(f"No .npy activation files found for task '{task_name}'. Skipping SAE precompute.")
        return

    print(f"Precomputing SAE features for task '{task_name}'...")
    weights_dir = get_sae_weights_dir()
    weights = load_sae_weights(weights_dir)

    for d in [few_shot_dir, test_dir]:
        if d.exists():
            precompute_dir(d, weights)

    print("SAE precompute complete.")


def precompute_single_locked(npy_path: Path) -> Path | None:
    """Lazy precompute for a single file, with file locking for concurrency safety."""
    out_path = npy_path.with_suffix(".sae.npz")
    if out_path.exists():
        return out_path

    lock_path = npy_path.with_suffix(".sae.lock")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        # Re-check after acquiring lock
        if out_path.exists():
            return out_path

        weights_dir = get_sae_weights_dir()
        weights = load_sae_weights(weights_dir)
        return precompute_single(npy_path, weights)
