#!/usr/bin/env bash
# Agent scaffold CLI entry point.
# Usage:
#   ./scaffold.sh init              — Initialize directories
#   ./scaffold.sh run <task_name>   — Launch strategy agent on a task
#   ./scaffold.sh status [run_id]   — Show run status

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Use python3 if available, fall back to python
PYTHON="${PYTHON:-$(command -v python3 2>/dev/null || command -v python 2>/dev/null)}"

if [[ -z "$PYTHON" ]]; then
    echo "Error: Python not found. Install Python 3.10+ or set PYTHON env var."
    exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/src/scaffold.py" "$@"
