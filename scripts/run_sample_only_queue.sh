#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MAX_PARALLEL="${MAX_PARALLEL:-2}"
AGENT="${AGENT:-codex}"
N_STRATEGIES="${N_STRATEGIES:-10}"
TOOLS="${TOOLS:-sample}"

TASKS=(
  "atypical_answer"
  "atypical_cot_length"
  "followup_confidence"
  "gemma_self_deletion_clean"
  "reasoning_termination"
  "stanford_hint_clean"
)

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

active_jobs() {
  jobs -pr | wc -l | tr -d ' '
}

launch_task() {
  local task="$1"
  log "starting task=${task}"
  python src/scaffold.py run "$task" \
    --agent "$AGENT" \
    --tools "$TOOLS" \
    --n-strategies "$N_STRATEGIES" &
}

for task in "${TASKS[@]}"; do
  while [[ "$(active_jobs)" -ge "$MAX_PARALLEL" ]]; do
    wait -n
  done
  launch_task "$task"
done

while [[ "$(active_jobs)" -gt 0 ]]; do
  wait -n
done

log "all queued tasks completed"
