#!/usr/bin/env bash
set -euo pipefail

# Omni requires a different Torch version from the SRT MLX runtime.
OMNI_ROOT="${TOKEN_WORKSHED_NEXTCORE_OMNI_ROOT:-$HOME/Library/Application Support/token-workshed-nextcore/omni}"
OMNI_SOURCE="$OMNI_ROOT/sglang-omni-v0.2.0"
SRT_SOURCE="$OMNI_ROOT/sglang-v0.5.22"
OMNI_COMMIT="e8c635ecb9c68bcf2fbc32ceb3c40f85f3590657"
SRT_COMMIT="baa9befa381511cd4d56f83980951e5a45fffc6d"
BREW_BIN="/opt/homebrew/bin/brew"
if [[ ! -x "$BREW_BIN" ]] || ! "$BREW_BIN" list --formula ffmpeg@7 >/dev/null 2>&1 || ! command -v uv >/dev/null 2>&1; then
  echo "The experimental Omni Apple installer requires Homebrew ffmpeg@7 and uv on PATH. Install those prerequisites before running this script." >&2
  exit 1
fi
mkdir -p "$OMNI_ROOT"
if [[ ! -d "$OMNI_SOURCE/.git" ]]; then
  git clone --depth 1 --branch v0.2.0 https://github.com/sgl-project/sglang-omni.git "$OMNI_SOURCE"
fi
if [[ ! -d "$SRT_SOURCE/.git" ]]; then
  git clone --depth 1 --branch v0.5.22 https://github.com/sgl-project/sglang.git "$SRT_SOURCE"
fi
[[ "$(git -C "$OMNI_SOURCE" rev-parse HEAD)" == "$OMNI_COMMIT" ]] || { echo "Omni commit mismatch" >&2; exit 1; }
[[ "$(git -C "$SRT_SOURCE" rev-parse HEAD)" == "$SRT_COMMIT" ]] || { echo "SGLang commit mismatch" >&2; exit 1; }
SGLANG_OMNI_VENV="$OMNI_ROOT/.venv" SGLANG_OMNI_CACHE="$OMNI_ROOT/cache" \
SGLANG_SOURCE_DIR="$SRT_SOURCE" SGLANG_VERSION=v0.5.22 NONINTERACTIVE=1 \
  "$OMNI_SOURCE/install.sh" --non-interactive
