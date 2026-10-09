#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
MODEL="${1:-}"
SERVER_URL="${2:-http://127.0.0.1:8000}"
UI_URL="${3:-http://127.0.0.1:7862}"
RUNTIME="${4:-openclaw}"

if [[ -z "${MODEL}" ]]; then
  echo "Usage: scripts/run_timeout_matrix.sh \"<MODEL_ID>\" [SERVER_URL] [UI_URL] [openclaw|hermes]"
  echo "Example: scripts/run_timeout_matrix.sh \"ibm-granite/granite-4.0-1b\""
  exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Error: python executable not found: ${PYTHON_BIN}"
  exit 1
fi

if ! "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import requests
print("ok")
PY
then
  echo "Error: '${PYTHON_BIN}' missing dependency 'requests'."
  echo "Install with: ${PYTHON_BIN} -m pip install requests"
  exit 1
fi

REPORT_DIR="${ROOT_DIR}/.openclaw/diagnostics"
mkdir -p "${REPORT_DIR}"
STAMP="$(date +%Y%m%d_%H%M%S)"
REPORT_FILE="${REPORT_DIR}/timeout_matrix_${STAMP}.json"

echo "Running timeout matrix..."
echo "  model:      ${MODEL}"
echo "  server_url: ${SERVER_URL}"
echo "  ui_url:     ${UI_URL}"
echo "  runtime:    ${RUNTIME}"
echo "  report:     ${REPORT_FILE}"

PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u "${ROOT_DIR}/scripts/diagnose_timeout_matrix.py" \
  --model "${MODEL}" \
  --server-url "${SERVER_URL}" \
  --ui-url "${UI_URL}" \
  --runtime "${RUNTIME}" \
  --report-file "${REPORT_FILE}"

echo "Done."
