#!/usr/bin/env bash
# Agent scaffold CLI entry point.
# Usage:
#   ./scaffold.sh init              — Initialize directories
#   ./scaffold.sh run <task_name>   — Launch strategy agent on a task
#   ./scaffold.sh status [run_id]   — Show run status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve a usable Python interpreter.
# Preference: explicit PYTHON env var > uv-managed install > python3/python on PATH.
resolve_python() {
    if [[ -n "${PYTHON:-}" ]] && [[ -x "$PYTHON" || -f "$PYTHON" ]]; then
        echo "$PYTHON"; return
    fi
    # Project-local venv (preferred — has third-party deps like `tokenizers`)
    for candidate in "$SCRIPT_DIR/.venv/Scripts/python.exe" "$SCRIPT_DIR/.venv/bin/python"; do
        if [[ -f "$candidate" ]]; then echo "$candidate"; return; fi
    done
    # uv-managed installs (Windows)
    for candidate in "$APPDATA/uv/python"/*/python.exe "$HOME/.local/share/uv/python"/*/python.exe; do
        if [[ -f "$candidate" ]]; then echo "$candidate"; return; fi
    done
    # uv fallback
    if command -v uv >/dev/null 2>&1; then
        local uv_python
        uv_python="$(uv python find 2>/dev/null || true)"
        if [[ -n "$uv_python" && -f "$uv_python" ]]; then echo "$uv_python"; return; fi
    fi
    # System Python (but avoid Microsoft Store stub on Windows)
    for cmd in python3 python; do
        local p
        p="$(command -v "$cmd" 2>/dev/null || true)"
        if [[ -n "$p" && "$p" != *"Microsoft/WindowsApps"* ]]; then
            echo "$p"; return
        fi
    done
}

PYTHON="$(resolve_python)"

if [[ -z "$PYTHON" ]]; then
    echo "Error: Python not found. Install Python 3.10+ or set PYTHON env var."
    exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/src/scaffold.py" "$@"
