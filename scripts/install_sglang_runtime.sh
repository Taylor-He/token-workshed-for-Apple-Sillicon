#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${TOKEN_WORKSHED_NEXTCORE_PYTHON:-$ROOT_DIR/.venv/bin/python}"
SGLANG_ROOT="${TOKEN_WORKSHED_NEXTCORE_SGLANG_ROOT:-$HOME/Library/Application Support/token-workshed-nextcore/sglang-v0.5.22}"
SGLANG_COMMIT="baa9befa381511cd4d56f83980951e5a45fffc6d"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python environment not found: $PYTHON_BIN" >&2
  echo "Create a Python 3.12 environment or set TOKEN_WORKSHED_NEXTCORE_PYTHON." >&2
  exit 1
fi
"$PYTHON_BIN" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit("NextCore SGLang runtime requires Python 3.12.")
PY

if [[ ! -d "$SGLANG_ROOT/.git" ]]; then
  mkdir -p "$(dirname "$SGLANG_ROOT")"
  git clone --depth 1 --branch v0.5.22 https://github.com/sgl-project/sglang.git "$SGLANG_ROOT"
fi
if [[ "$(git -C "$SGLANG_ROOT" rev-parse HEAD)" != "$SGLANG_COMMIT" ]]; then
  echo "SGLang source does not match the pinned commit: $SGLANG_ROOT" >&2
  exit 1
fi

cp "$SGLANG_ROOT/python/pyproject_other.toml" "$SGLANG_ROOT/python/pyproject.toml"
if command -v uv >/dev/null 2>&1; then
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.22 \
    SGLANG_BUILD_RUST_EXTS=none \
    uv pip install --python "$PYTHON_BIN" -e "$SGLANG_ROOT/python[srt_mps]"
else
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.22 \
    SGLANG_BUILD_RUST_EXTS=none \
    "$PYTHON_BIN" -m pip install --editable "$SGLANG_ROOT/python[srt_mps]"
fi

"$PYTHON_BIN" - <<'PY'
from importlib.metadata import version

installed = version("sglang")
if installed != "0.5.22":
    raise SystemExit(f"Expected SGLang 0.5.22, found {installed}.")
print(f"Installed SGLang {installed} from the pinned v0.5.22 source checkout.")
PY
