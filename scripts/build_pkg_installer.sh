#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-0.2.0}"
RUNTIME_VERSION="${2:-0.4.0}"
PYTHON_BOOTSTRAP_VERSION="${3:-3.12.8}"
PYTHON_BOOTSTRAP_SHA256="${4:-}"

APP_NAME="token-workshed.app"
APP_PATH="$ROOT_DIR/dist/$APP_NAME"
OUT_PKG="$ROOT_DIR/dist/token-workshed-${VERSION}-installer.pkg"
PYTHON_PKG_URL="https://www.python.org/ftp/python/${PYTHON_BOOTSTRAP_VERSION}/python-${PYTHON_BOOTSTRAP_VERSION}-macos11.pkg"

if [[ ! -d "$APP_PATH" ]]; then
  echo "Error: missing app bundle: $APP_PATH"
  echo "Build the .app first, then run this script."
  exit 1
fi

NATIVE_UI_APP_BIN="$APP_PATH/Contents/MacOS/token-workshed-native-ui"
if [[ ! -x "$NATIVE_UI_APP_BIN" ]]; then
  echo "Error: missing native UI binary in app bundle: $NATIVE_UI_APP_BIN"
  echo "Build native-ui with cargo --release, then rebuild the .app with token-workshed.spec."
  exit 1
fi

BUILD_DIR="$ROOT_DIR/build/pkg"
COMP_DIR="$BUILD_DIR/components"
ROOTS_DIR="$BUILD_DIR/pkgroots"
SCRIPTS_DIR="$BUILD_DIR/scripts"
DIST_FILE="$BUILD_DIR/distribution.xml"
SOURCE_STAGE_DIR="$ROOTS_DIR/source/Library/Application Support/token-workshed/token-workshed-${VERSION}"

rm -rf "$BUILD_DIR"
mkdir -p "$COMP_DIR" "$ROOTS_DIR" "$SCRIPTS_DIR/runtime"

cat >"$SCRIPTS_DIR/runtime/postinstall" <<POSTINSTALL
#!/bin/bash
set -euo pipefail

APP_SUPPORT_DIR="/Library/Application Support/token-workshed"
RUNTIME_DIR="\$APP_SUPPORT_DIR/runtime"
VENV_DIR="\$RUNTIME_DIR/.venv"
RUNTIME_VERSION="$RUNTIME_VERSION"
PYTHON_BOOTSTRAP_VERSION="$PYTHON_BOOTSTRAP_VERSION"
PYTHON_BOOTSTRAP_SHA256="$PYTHON_BOOTSTRAP_SHA256"
PYTHON_PKG_URL="$PYTHON_PKG_URL"
LOG_FILE="/var/log/token-workshed-install.log"

mkdir -p "\$APP_SUPPORT_DIR"
touch "\$LOG_FILE"
exec >>"\$LOG_FILE" 2>&1

echo "=============================="
echo "token-workshed runtime postinstall started at \$(date)"
echo "=============================="

if [[ "\$(uname -m)" != "arm64" ]]; then
  echo "ERROR: arm64 is required for vllm-mlx runtime."
  exit 1
fi

python_is_compatible() {
  local candidate="\$1"
  [[ -x "\$candidate" ]] || return 1
  "\$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
}

find_python_bin() {
  local candidate
  for candidate in \
    /opt/homebrew/bin/python3 \
    /usr/local/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
    /Library/Frameworks/Python.framework/Versions/3.11/bin/python3 \
    /usr/bin/python3
  do
    if python_is_compatible "\$candidate"; then
      echo "\$candidate"
      return 0
    fi
  done
  return 1
}

install_python_bootstrap() {
  local pkg_file="/private/tmp/token-workshed-python-\${PYTHON_BOOTSTRAP_VERSION}.pkg"
  echo "Python >=3.10 not found. Installing Python \${PYTHON_BOOTSTRAP_VERSION}..."
  /usr/bin/curl -fL "\$PYTHON_PKG_URL" -o "\$pkg_file"

  if ! /usr/sbin/pkgutil --check-signature "\$pkg_file" | /usr/bin/grep -qi "Python Software Foundation"; then
    echo "ERROR: Python bootstrap package signature check failed."
    /usr/sbin/pkgutil --check-signature "\$pkg_file" || true
    /bin/rm -f "\$pkg_file"
    exit 1
  fi

  if [[ -n "\$PYTHON_BOOTSTRAP_SHA256" ]]; then
    local actual_sha
    local actual_sha_lc
    local expected_sha_lc
    actual_sha="\$(/usr/bin/shasum -a 256 "\$pkg_file" | /usr/bin/awk '{print \$1}')"
    actual_sha_lc="\$(printf "%s" "\$actual_sha" | /usr/bin/tr '[:upper:]' '[:lower:]')"
    expected_sha_lc="\$(printf "%s" "\$PYTHON_BOOTSTRAP_SHA256" | /usr/bin/tr '[:upper:]' '[:lower:]')"
    if [[ "\$actual_sha_lc" != "\$expected_sha_lc" ]]; then
      echo "ERROR: Python bootstrap SHA256 mismatch."
      echo "Expected: \$expected_sha_lc"
      echo "Actual:   \$actual_sha_lc"
      /bin/rm -f "\$pkg_file"
      exit 1
    fi
  fi

  /usr/sbin/installer -pkg "\$pkg_file" -target /
  /bin/rm -f "\$pkg_file"
}

PYTHON_BIN="\$(find_python_bin || true)"
if [[ -z "\$PYTHON_BIN" ]]; then
  install_python_bootstrap
  PYTHON_BIN="\$(find_python_bin || true)"
fi

if [[ -z "\$PYTHON_BIN" ]]; then
  echo "ERROR: Python >=3.10 is required but not available after bootstrap install."
  exit 1
fi

echo "Using Python: \$PYTHON_BIN"
"\$PYTHON_BIN" -V || true
"\$PYTHON_BIN" -m ensurepip --upgrade || true

mkdir -p "\$RUNTIME_DIR"

# If an old/incompatible venv already exists (e.g. python3.9), recreate it.
if [[ -x "\$VENV_DIR/bin/python3" ]]; then
  if ! "\$VENV_DIR/bin/python3" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
  then
    echo "Removing incompatible existing venv at \$VENV_DIR"
    rm -rf "\$VENV_DIR"
  fi
fi

if [[ ! -x "\$VENV_DIR/bin/python3" ]]; then
  echo "Creating venv at \$VENV_DIR"
  "\$PYTHON_BIN" -m venv "\$VENV_DIR"
fi

echo "Upgrading pip/setuptools/wheel..."
"\$VENV_DIR/bin/python3" -m pip install --upgrade --retries 6 --timeout 120 pip setuptools wheel

echo "Installing vllm-mlx==\${RUNTIME_VERSION} ..."
"\$VENV_DIR/bin/python3" -m pip install --upgrade --retries 6 --timeout 120 "vllm-mlx==\${RUNTIME_VERSION}"

mkdir -p /usr/local/bin
ln -sf "\$VENV_DIR/bin/vllm-mlx" /usr/local/bin/token-workshed

cat >"\$APP_SUPPORT_DIR/runtime-info.txt" <<INFO
vllm-mlx=\${RUNTIME_VERSION}
venv=\$VENV_DIR
python=\$PYTHON_BIN
INFO

echo "Runtime installation completed at \$(date)"
POSTINSTALL
chmod +x "$SCRIPTS_DIR/runtime/postinstall"

pkgbuild \
  --nopayload \
  --scripts "$SCRIPTS_DIR/runtime" \
  --identifier "com.taylorhe.tokenworkshed.runtime" \
  --version "$VERSION" \
  --install-location "/" \
  "$COMP_DIR/token-workshed-runtime.pkg"

mkdir -p "$ROOTS_DIR/ui/Applications"
cp -R "$APP_PATH" "$ROOTS_DIR/ui/Applications/"

pkgbuild \
  --root "$ROOTS_DIR/ui" \
  --identifier "com.taylorhe.tokenworkshed.ui" \
  --version "$VERSION" \
  --install-location "/" \
  "$COMP_DIR/token-workshed-ui.pkg"

if ! command -v rsync >/dev/null 2>&1; then
  echo "Error: rsync is required to stage full source bundle payload."
  exit 1
fi

mkdir -p "$SOURCE_STAGE_DIR"

rsync -a \
  --exclude "/.git/" \
  --exclude "/.venv/" \
  --exclude "/venv/" \
  --exclude "/__pycache__/" \
  --exclude "/*.py[co]" \
  --exclude "/.pytest_cache/" \
  --exclude "/.ruff_cache/" \
  --exclude "/*.egg-info/" \
  --exclude "/.token-workshed-state/" \
  --exclude "/.openclaw/" \
  --exclude "/.codex/" \
  --exclude "/.aws/" \
  --exclude "/.ssh/" \
  --exclude "/.gnupg/" \
  --exclude "/node_modules/" \
  --exclude "/design-qa.md" \
  --exclude "/docs/testing/2026-10-08-native-ui-smoke-test.md" \
  --exclude "/docs/quantization/2026-09-30-workshed-smoke-test.md" \
  --exclude "**/.git/" \
  --exclude "**/.venv/" \
  --exclude "**/venv/" \
  --exclude "**/__pycache__/" \
  --exclude "**/*.py[co]" \
  --exclude "**/.pytest_cache/" \
  --exclude "**/.ruff_cache/" \
  --exclude "**/*.egg-info/" \
  --exclude "**/.token-workshed-state/" \
  --exclude "**/.openclaw/" \
  --exclude "**/.codex/" \
  --exclude "**/.aws/" \
  --exclude "**/.ssh/" \
  --exclude "**/.gnupg/" \
  --exclude "**/node_modules/" \
  --include "**/.env.example" \
  --exclude "**/.env" \
  --exclude "**/.env.*" \
  --exclude "**/*.log" \
  --exclude "**/.DS_Store" \
  --exclude "/build/" \
  --exclude "/dist/" \
  --exclude "/native-ui/target/" \
  --exclude "/native-ui/vendor/libcosmic/" \
  --exclude "/native-ui/vendor/cosmic-protocols/" \
  --exclude "/native-ui/baseline-screenshots/" \
  --exclude "/integrations/cypherclaw/.artifacts/" \
  --exclude "/gsm8k_qwen3_0.6b_results.json" \
  --exclude "/vlm_benchmark_results.json" \
  --exclude "/AGENTS.md" \
  --exclude "/BOOTSTRAP.md" \
  --exclude "/HEARTBEAT.md" \
  --exclude "/IDENTITY.md" \
  --exclude "/MEMORY.md" \
  --exclude "/SOUL.md" \
  --exclude "/TOOLS.md" \
  --exclude "/USER.md" \
  --exclude "/memory/" \
  "$ROOT_DIR/" \
  "$SOURCE_STAGE_DIR/"

# Fail closed if a future integration adds runtime state outside the filters
# above. Source installers must never contain credentials, browser sessions,
# local databases, virtual environments, or generated bytecode.
SENSITIVE_PAYLOAD_PATH="$(find "$SOURCE_STAGE_DIR" \
  \( -type d \( \
    -name ".git" -o \
    -name ".venv" -o \
    -name "venv" -o \
    -name "__pycache__" -o \
    -name ".token-workshed-state" -o \
    -name ".openclaw" \
    -o -name ".codex" \
    -o -name ".aws" \
    -o -name ".ssh" \
    -o -name ".gnupg" -o \
    -name "node_modules" \
  \) -o -type f \( \
    -name ".env" -o \
    -name ".env.*" ! -name ".env.example" -o \
    -name "*.pyc" -o \
    -name "*.pyo" -o \
    -name "*.log" -o \
    -name "id_rsa" -o \
    -name "id_ed25519" -o \
    -name "*.pem" -o \
    -name "*.p12" \
  \) \) -print -quit)"
if [[ -n "$SENSITIVE_PAYLOAD_PATH" ]]; then
  echo "Error: sensitive or generated payload escaped source staging filters:"
  echo "  $SENSITIVE_PAYLOAD_PATH"
  exit 1
fi

pkgbuild \
  --root "$ROOTS_DIR/source" \
  --identifier "com.taylorhe.tokenworkshed.source" \
  --version "$VERSION" \
  --install-location "/" \
  "$COMP_DIR/token-workshed-source.pkg"

cat >"$DIST_FILE" <<DISTXML
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
  <title>token-workshed Installer</title>
  <domains enable_anywhere="false" enable_currentUserHome="false" enable_localSystem="true"/>
  <options customize="never" require-scripts="true" hostArchitectures="arm64"/>
  <choices-outline>
    <line choice="runtime"/>
    <line choice="ui"/>
    <line choice="source"/>
  </choices-outline>
  <choice id="runtime" title="token-workshed Runtime" description="Install token-workshed runtime (vllm-mlx backend) into a dedicated virtual environment.">
    <pkg-ref id="com.taylorhe.tokenworkshed.runtime"/>
  </choice>
  <choice id="ui" title="token-workshed UI" description="Install the token-workshed desktop app into /Applications.">
    <pkg-ref id="com.taylorhe.tokenworkshed.ui"/>
  </choice>
  <choice id="source" title="token-workshed Source Bundle" description="Install token-workshed $VERSION source/integration bundle into /Library/Application Support/token-workshed/token-workshed-$VERSION.">
    <pkg-ref id="com.taylorhe.tokenworkshed.source"/>
  </choice>
  <pkg-ref id="com.taylorhe.tokenworkshed.runtime" version="$VERSION" onConclusion="none">token-workshed-runtime.pkg</pkg-ref>
  <pkg-ref id="com.taylorhe.tokenworkshed.ui" version="$VERSION" onConclusion="none">token-workshed-ui.pkg</pkg-ref>
  <pkg-ref id="com.taylorhe.tokenworkshed.source" version="$VERSION" onConclusion="none">token-workshed-source.pkg</pkg-ref>
</installer-gui-script>
DISTXML

productbuild \
  --distribution "$DIST_FILE" \
  --package-path "$COMP_DIR" \
  "$OUT_PKG"

echo "Built installer: $OUT_PKG"
