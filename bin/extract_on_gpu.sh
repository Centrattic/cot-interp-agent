#!/usr/bin/env bash
# Start vLLM locally on this GPU host, run precompute_logits for a task, then shut down.
#
# Usage:
#   bin/extract_on_gpu.sh <task> [model]
# Example:
#   bin/extract_on_gpu.sh reasoning_termination Qwen/Qwen3-32B

set -euo pipefail

TASK="${1:?usage: $0 <task> [model]}"
MODEL="${2:-Qwen/Qwen3-32B}"
PORT="${VLLM_PORT:-8000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# 1. Deps (idempotent) — dedicated venv so vLLM pulls a matched torch+CUDA.
# The system's /venv/main has torch cu13, but vLLM wheels are cu12; mixing breaks.
VENV="${VLLM_VENV:-/workspace/venv-vllm}"
if [ ! -x "$VENV/bin/python3" ]; then
    echo "[setup] creating venv at $VENV"
    /usr/bin/python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q -U pip wheel
fi
PY="$VENV/bin/python3"
if ! "$PY" -c "import vllm" 2>/dev/null; then
    echo "[setup] installing vllm + deps into $VENV (one-time, several minutes)..."
    "$VENV/bin/pip" install -q vllm transformers requests numpy
fi

# 2. Launch vLLM server in background.
LOG="$REPO_ROOT/vllm.log"
echo "[serve] launching vLLM on port $PORT (log: $LOG)"
"$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --max-logprobs 10 \
    --trust-remote-code \
    > "$LOG" 2>&1 &
VLLM_PID=$!
trap 'echo "[cleanup] stopping vLLM (pid $VLLM_PID)"; kill $VLLM_PID 2>/dev/null || true; wait $VLLM_PID 2>/dev/null || true' EXIT

# 3. Wait for readiness (up to 30 min for cold model download + load).
echo "[wait] polling http://localhost:$PORT/v1/models (tail $LOG for progress)"
ready=0
for i in $(seq 1 180); do
    if curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        ready=1
        echo "[wait] vLLM ready after ${i}0s"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[error] vLLM exited early; see $LOG" >&2
        exit 1
    fi
    sleep 10
done
if [ "$ready" != "1" ]; then
    echo "[error] vLLM not ready after 30 min; see $LOG" >&2
    exit 1
fi

# 4. Run extraction.
"$PY" src/precompute_logits.py \
    --task "$TASK" \
    --model "$MODEL" \
    --vllm-url "http://localhost:$PORT/v1"

echo "[done] extraction complete; vLLM will be shut down by trap"
