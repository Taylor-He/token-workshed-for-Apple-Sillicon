#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/run_openclaw_hermes_bridge.sh "<MODEL_ID_OR_PATH>" [run|dry-run] [openclaw|hermes|all]
#
# Example:
#   ./scripts/run_openclaw_hermes_bridge.sh "ibm-granite/granite-4.0-1b" dry-run all
#   ./scripts/run_openclaw_hermes_bridge.sh "Qwen/Qwen3.5-3B-Instruct" run all

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

MODEL="${1:-}"
MODE="${2:-dry-run}"
AGENT="${3:-all}"

if [[ -z "$MODEL" ]]; then
  echo "Error: model is required."
  echo "Usage: ./scripts/run_openclaw_hermes_bridge.sh \"<MODEL_ID_OR_PATH>\" [run|dry-run] [openclaw|hermes|all]"
  exit 1
fi

if [[ "$MODE" != "run" && "$MODE" != "dry-run" ]]; then
  echo "Error: mode must be run or dry-run."
  exit 1
fi

if [[ "$AGENT" != "openclaw" && "$AGENT" != "hermes" && "$AGENT" != "all" ]]; then
  echo "Error: agent must be openclaw, hermes, or all."
  exit 1
fi

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PY="$ROOT_DIR/.venv/bin/python"
elif command -v python3.10 >/dev/null 2>&1; then
  PY="python3.10"
elif command -v python3 >/dev/null 2>&1; then
  PY="python3"
else
  echo "Error: no usable Python found (.venv/bin/python / python3.10 / python3)."
  exit 1
fi

COMMON_ARGS=(
  scripts/openclaw_hermes_vllm_bridge.py
  "$MODEL"
  --agent "$AGENT"
  --host 127.0.0.1
  --port 8000
  --report-file "$ROOT_DIR/.openclaw/bridge_report.json"
)

if [[ "$MODE" == "run" ]]; then
  exec "$PY" "${COMMON_ARGS[@]}" --run
else
  exec "$PY" "${COMMON_ARGS[@]}" --dry-run
fi

