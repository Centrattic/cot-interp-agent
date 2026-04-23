#!/usr/bin/env bash
# Set up a GPU venv, run src/precompute_activations.py for termination,
# and exit. Mirrors bin/extract_on_gpu.sh but uses transformers directly
# (no vLLM — vLLM's OpenAI API doesn't expose hidden states).
#
# Usage:
#   bin/extract_activations_on_gpu.sh [splits]
# Example:
#   bin/extract_activations_on_gpu.sh test,ood_val
#
# Requires: CUDA-capable GPU with ≥64 GB VRAM (H100/H200/A100-80GB).
# Disk requirement: ~100 GB (model weights dominate).

set -euo pipefail

SPLITS="${1:-test,ood_val}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# 0. Redirect caches onto /workspace. Root FS on many pods is <30 GB,
#    which can't hold Qwen3-32B (~65 GB) or the HF download staging.
export HF_HOME="${HF_HOME:-/workspace/hf-cache}"
export COT_CACHE_DIR="${COT_CACHE_DIR:-/workspace/cot-interp-cache}"
mkdir -p "$HF_HOME" "$COT_CACHE_DIR"

# 1. Dedicated venv so we don't clobber anything the pod ships with.
VENV="${EXTRACT_VENV:-/workspace/venv-extract}"
if [ ! -x "$VENV/bin/python3" ]; then
    echo "[setup] creating venv at $VENV"
    mkdir -p "$(dirname "$VENV")"
    /usr/bin/python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q -U pip wheel
fi
PY="$VENV/bin/python3"

if ! "$PY" -c "import torch, transformers, safetensors, huggingface_hub" 2>/dev/null; then
    echo "[setup] installing deps into $VENV (one-time, several minutes)..."
    # torch installed first so transformers resolves against the already-installed wheel.
    "$VENV/bin/pip" install -q torch
    "$VENV/bin/pip" install -q transformers accelerate safetensors huggingface_hub numpy
fi

echo "[run] extracting activations for splits: $SPLITS"
exec "$PY" src/precompute_activations.py --splits "$SPLITS"
