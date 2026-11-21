#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION="${1:-NextCore}"
RUNTIME_VERSION="0.5.22"
SGLANG_COMMIT="baa9befa381511cd4d56f83980951e5a45fffc6d"
PYTHON_BOOTSTRAP_VERSION="${3:-3.12.8}"
PYTHON_BOOTSTRAP_SHA256="${4:-}"

APP_NAME="nextcore.app"
APP_PATH="$ROOT_DIR/dist/$APP_NAME"
OUT_PKG="$ROOT_DIR/dist/token-workshed NextCore.pkg"
UPDATE_OUT_PKG="$ROOT_DIR/dist/token-workshed NextCore Update.pkg"
PYTHON_PKG_URL="https://www.python.org/ftp/python/${PYTHON_BOOTSTRAP_VERSION}/python-${PYTHON_BOOTSTRAP_VERSION}-macos11.pkg"

if [[ "$VERSION" != "NextCore" && ! "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Invalid installer version: $VERSION (expected NextCore or MAJOR.MINOR.PATCH)." >&2
  exit 2
fi

if [[ ! -d "$APP_PATH" ]]; then
  echo "Error: missing app bundle: $APP_PATH"
  echo "Build the .app first, then run this script."
  exit 1
fi

NATIVE_UI_APP_BIN="$APP_PATH/Contents/Helpers/NextCore UI.app/Contents/MacOS/token-workshed-native-ui"
if [[ ! -x "$NATIVE_UI_APP_BIN" ]]; then
  echo "Error: missing native UI binary in app bundle: $NATIVE_UI_APP_BIN"
  echo "Build native-ui with cargo --release, then rebuild the .app with token-workshed.spec."
  exit 1
fi

APP_BUNDLE_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$APP_PATH/Contents/Info.plist")"
APP_BUNDLE_BUILD_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleVersion' "$APP_PATH/Contents/Info.plist")"
if [[ "$APP_BUNDLE_VERSION" != "$VERSION" || "$APP_BUNDLE_BUILD_VERSION" != "$VERSION" ]]; then
  echo "Error: installer version $VERSION does not match app bundle version $APP_BUNDLE_VERSION/$APP_BUNDLE_BUILD_VERSION."
  echo "Rebuild the app with TOKEN_WORKSHED_VERSION=$VERSION before packaging."
  exit 1
fi

BUILD_DIR="$ROOT_DIR/build/pkg"
COMP_DIR="$BUILD_DIR/components"
ROOTS_DIR="$BUILD_DIR/pkgroots"
SCRIPTS_DIR="$BUILD_DIR/scripts"
DIST_FILE="$BUILD_DIR/distribution.xml"
UPDATE_DIST_FILE="$BUILD_DIR/update-distribution.xml"
SOURCE_COMPONENTS_PLIST="$BUILD_DIR/source-components.plist"
SOURCE_STAGE_DIR="$ROOTS_DIR/source/Library/Application Support/token-workshed-nextcore/source"

rm -rf "$BUILD_DIR"
mkdir -p "$COMP_DIR" "$ROOTS_DIR" "$SCRIPTS_DIR/runtime" "$SCRIPTS_DIR/ui" "$SCRIPTS_DIR/source"

cat >"$SCRIPTS_DIR/runtime/postinstall" <<POSTINSTALL
#!/bin/bash
set -euo pipefail

APP_SUPPORT_DIR="/Library/Application Support/token-workshed-nextcore"
RUNTIME_DIR="\$APP_SUPPORT_DIR/runtime"
VENV_DIR="\$RUNTIME_DIR/.venv"
RUNTIME_VERSION="$RUNTIME_VERSION"
SGLANG_COMMIT="$SGLANG_COMMIT"
APP_VERSION="$VERSION"
PYTHON_BOOTSTRAP_VERSION="$PYTHON_BOOTSTRAP_VERSION"
PYTHON_BOOTSTRAP_SHA256="$PYTHON_BOOTSTRAP_SHA256"
PYTHON_PKG_URL="$PYTHON_PKG_URL"
LOG_FILE="/var/log/token-workshed-nextcore-install.log"

mkdir -p "\$APP_SUPPORT_DIR"
touch "\$LOG_FILE"
exec >>"\$LOG_FILE" 2>&1

echo "=============================="
echo "token-workshed runtime postinstall started at \$(date)"
echo "=============================="

if [[ "\$(uname -m)" != "arm64" ]]; then
  echo "ERROR: arm64 is required for SGLang runtime."
  exit 1
fi

python_is_compatible() {
  local candidate="\$1"
  [[ -x "\$candidate" ]] || return 1
  "\$candidate" - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)
PY
}

find_python_bin() {
  local candidate
  for candidate in \
    /opt/homebrew/bin/python3.12 \
    /usr/local/bin/python3.12 \
    /Library/Frameworks/Python.framework/Versions/3.12/bin/python3
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
  echo "Python 3.12 not found. Installing Python \${PYTHON_BOOTSTRAP_VERSION}..."
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
  echo "ERROR: Python 3.12 is required but not available after bootstrap install."
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
raise SystemExit(0 if sys.version_info[:2] == (3, 12) else 1)
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

SGLANG_SOURCE="\$RUNTIME_DIR/sglang-\$SGLANG_COMMIT"
if [[ ! -d "\$SGLANG_SOURCE/.git" ]]; then
  /usr/bin/git clone --depth 1 --branch v0.5.22 https://github.com/sgl-project/sglang.git "\$SGLANG_SOURCE"
fi
if [[ "\$(/usr/bin/git -C "\$SGLANG_SOURCE" rev-parse HEAD)" != "\$SGLANG_COMMIT" ]]; then
  echo "ERROR: SGLang checkout does not match the pinned commit."
  exit 1
fi
/bin/cp "\$SGLANG_SOURCE/python/pyproject_other.toml" "\$SGLANG_SOURCE/python/pyproject.toml"
SGLANG_BUILD_RUST_EXTS="none" SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG="\$RUNTIME_VERSION" \\
  "\$VENV_DIR/bin/python3" -m pip install --retries 6 --timeout 120 "\$SGLANG_SOURCE/python[srt_mps]"
"\$VENV_DIR/bin/python3" - <<'PY'
from importlib.metadata import version

installed = version("sglang")
if installed != "$RUNTIME_VERSION":
    raise SystemExit(f"Expected SGLang Runtime $RUNTIME_VERSION, found {installed}.")
PY
"\$VENV_DIR/bin/python3" -m pip install --retries 6 --timeout 120 "\$APP_SUPPORT_DIR/source[ide]"
mkdir -p /usr/local/bin
ln -sf "\$VENV_DIR/bin/token-workshed-nextcore" /usr/local/bin/token-workshed-nextcore
cat >"\$APP_SUPPORT_DIR/runtime-info.txt" <<INFO
sglang=\$RUNTIME_VERSION
sglang_commit=\$SGLANG_COMMIT
venv=\$VENV_DIR
python=\$PYTHON_BIN
INFO

echo "Runtime installation completed at \$(date)"
POSTINSTALL
chmod +x "$SCRIPTS_DIR/runtime/postinstall"

cat >"$SCRIPTS_DIR/ui/postinstall" <<'UI_POSTINSTALL'
#!/bin/bash
set -euo pipefail

OLD_APP="/Applications/token-workshed NextCore.app"
NEW_APP="/Applications/nextcore.app"
BACKUP_ROOT="/Library/Application Support/token-workshed-nextcore/backups"

if [[ -d "$OLD_APP" && -d "$NEW_APP" ]]; then
  if /usr/bin/pgrep -f '/Applications/token-workshed NextCore\.app/Contents/MacOS/nextcore' >/dev/null 2>&1; then
    echo "The previous NextCore app is running; keeping its bundle until it quits."
    exit 0
  fi

  mkdir -p "$BACKUP_ROOT"
  backup="$BACKUP_ROOT/token-workshed NextCore.app.pre-nextcore-bundle-path.$(/bin/date +%Y%m%d-%H%M%S).zip"
  if [[ -e "$backup" ]]; then
    backup="$BACKUP_ROOT/token-workshed NextCore.app.pre-nextcore-bundle-path.$(/bin/date +%s).zip"
  fi
  /usr/bin/ditto -c -k --sequesterRsrc --keepParent "$OLD_APP" "$backup"
  /usr/bin/unzip -tq "$backup" >/dev/null
  /bin/rm -rf "$OLD_APP"
fi
UI_POSTINSTALL
chmod +x "$SCRIPTS_DIR/ui/postinstall"

pkgbuild \
  --nopayload \
  --scripts "$SCRIPTS_DIR/runtime" \
  --identifier "com.taylorhe.tokenworkshed.nextcore.runtime" \
  --version "$VERSION" \
  --install-location "/" \
  "$COMP_DIR/token-workshed-runtime.pkg"

mkdir -p "$ROOTS_DIR/ui/Applications"
cp -R "$APP_PATH" "$ROOTS_DIR/ui/Applications/"

pkgbuild \
  --root "$ROOTS_DIR/ui" \
  --scripts "$SCRIPTS_DIR/ui" \
  --identifier "com.taylorhe.tokenworkshed.nextcore.ui" \
  --version "$VERSION" \
  --install-location "/" \
  "$COMP_DIR/token-workshed-ui.pkg"

if ! command -v rsync >/dev/null 2>&1; then
  echo "Error: rsync is required to stage full source bundle payload."
  exit 1
fi

mkdir -p "$SOURCE_STAGE_DIR"

rsync -a \
  --exclude "/.git" \
  --exclude "/.venv/" \
  --exclude "/venv/" \
  --exclude "/__pycache__/" \
  --exclude "/*.py[co]" \
  --exclude "/.pytest_cache/" \
  --exclude "/tests/" \
  --exclude "/examples/" \
  --exclude "**/tests/" \
  --exclude "**/examples/" \
  --exclude "/.ruff_cache/" \
  --exclude "/*.egg-info/" \
  --exclude "/.token-workshed-state/" \
  --exclude "/RELEASE_NOTES_v0.1.5.md" \
  --exclude "/RELEASE_NOTES_v0.1.5_SHORT.md" \
  --exclude "/RELEASE_NOTES_v0.2.0.md" \
  --exclude "/RELEASE_NOTES_NEXTCORE.md" \
  --include "/docs/" \
  --include "/docs/nextcore/" \
  --include "/docs/nextcore/***" \
  --exclude "/docs/**" \
  --exclude "/.openclaw/" \
  --exclude "/.codex/" \
  --exclude "/.aws/" \
  --exclude "/.ssh/" \
  --exclude "/.gnupg/" \
  --exclude "/node_modules/" \
  --exclude "/design-qa.md" \
  --exclude "/docs/testing/2026-10-08-native-ui-smoke-test.md" \
  --exclude "/docs/quantization/2026-09-30-workshed-smoke-test.md" \
  --exclude "**/.git" \
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
  --exclude "/native-ui/vendor/" \
  --exclude "/native-ui/baseline-screenshots/" \
  --exclude "/integrations/cypherclaw/.artifacts/" \
  --exclude "/integrations/cypherclaw/" \
  --exclude "/integrations/hermes/" \
  --exclude "/integrations/openclaw" \
  --exclude "/integrations/node-runtime/" \
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

cat >"$SOURCE_COMPONENTS_PLIST" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><array/></plist>
PLIST

cat >"$SCRIPTS_DIR/source/postinstall" <<'SOURCE_POSTINSTALL'
#!/bin/bash
set -euo pipefail

APP_SUPPORT_DIR="/Library/Application Support/token-workshed-nextcore"
SOURCE_DIR="$APP_SUPPORT_DIR/source"

# Remove generated copies and code left by older Agent Profile releases.
# These are source/build artifacts only; user settings and model caches stay intact.
/bin/rm -rf "$SOURCE_DIR/build/lib"
/bin/rm -rf "$SOURCE_DIR/token_workshed_nextcore/__pycache__"
/bin/rm -rf "$SOURCE_DIR/vllm_mlx"
/bin/rm -rf "$SOURCE_DIR/vllm_mlx.egg-info"
/bin/rm -f "$SOURCE_DIR/token_workshed_nextcore/sglang_compatibility.py"
/bin/rm -f "$SOURCE_DIR/token_workshed_nextcore/agent_profiles.py"
/bin/rm -f "$SOURCE_DIR/tests/test_agent_profiles.py"
/bin/rm -f "$SOURCE_DIR/tests/test_desktop_ui_agent_profiles.py"

RUNTIME_PYTHON="$APP_SUPPORT_DIR/runtime/.venv/bin/python3"
if [[ -x "$RUNTIME_PYTHON" ]]; then
  "$RUNTIME_PYTHON" -m pip install \
    --disable-pip-version-check \
    --no-deps \
    --no-build-isolation \
    --force-reinstall \
    "$SOURCE_DIR"
  SITE_PACKAGES=$("$RUNTIME_PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
  /bin/rm -f "$SITE_PACKAGES/token_workshed_nextcore/sglang_compatibility.py"
fi
SOURCE_POSTINSTALL
chmod +x "$SCRIPTS_DIR/source/postinstall"

pkgbuild \
  --root "$ROOTS_DIR/source" \
  --scripts "$SCRIPTS_DIR/source" \
  --component-plist "$SOURCE_COMPONENTS_PLIST" \
  --identifier "com.taylorhe.tokenworkshed.nextcore.source" \
  --version "$VERSION" \
  --install-location "/" \
  "$COMP_DIR/token-workshed-source.pkg"

cat >"$DIST_FILE" <<DISTXML
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
  <title>token-workshed NextCore Installer</title>
  <domains enable_anywhere="false" enable_currentUserHome="false" enable_localSystem="true"/>
  <options customize="never" require-scripts="true" hostArchitectures="arm64"/>
  <allowed-os-versions><os-version min="14.0"/></allowed-os-versions>
  <choices-outline>
    <line choice="source"/>
    <line choice="runtime"/>
    <line choice="ui"/>
  </choices-outline>
  <choice id="runtime" title="token-workshed NextCore SGLang Runtime" description="Install the pinned SGLang runtime into its dedicated virtual environment.">
    <pkg-ref id="com.taylorhe.tokenworkshed.nextcore.runtime"/>
  </choice>
  <choice id="ui" title="token-workshed NextCore" description="Install the token-workshed NextCore app into /Applications/nextcore.app.">
    <pkg-ref id="com.taylorhe.tokenworkshed.nextcore.ui"/>
  </choice>
  <choice id="source" title="token-workshed NextCore Source Bundle" description="Install the source and integrations into /Library/Application Support/token-workshed-nextcore/source.">
    <pkg-ref id="com.taylorhe.tokenworkshed.nextcore.source"/>
  </choice>
  <pkg-ref id="com.taylorhe.tokenworkshed.nextcore.runtime" version="$VERSION" onConclusion="none">token-workshed-runtime.pkg</pkg-ref>
  <pkg-ref id="com.taylorhe.tokenworkshed.nextcore.ui" version="$VERSION" onConclusion="none">token-workshed-ui.pkg</pkg-ref>
  <pkg-ref id="com.taylorhe.tokenworkshed.nextcore.source" version="$VERSION" onConclusion="none">token-workshed-source.pkg</pkg-ref>
</installer-gui-script>
DISTXML

productbuild \
  --distribution "$DIST_FILE" \
  --package-path "$COMP_DIR" \
  "$OUT_PKG"

echo "Built installer: $OUT_PKG"

cat >"$UPDATE_DIST_FILE" <<UPDATE_DISTXML
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
  <title>token-workshed NextCore Update</title>
  <domains enable_anywhere="false" enable_currentUserHome="false" enable_localSystem="true"/>
  <options customize="never" require-scripts="true" hostArchitectures="arm64"/>
  <allowed-os-versions><os-version min="14.0"/></allowed-os-versions>
  <choices-outline>
    <line choice="source"/>
    <line choice="ui"/>
  </choices-outline>
  <choice id="ui" title="token-workshed NextCore" description="Update the NextCore app in /Applications/nextcore.app.">
    <pkg-ref id="com.taylorhe.tokenworkshed.nextcore.ui"/>
  </choice>
  <choice id="source" title="token-workshed NextCore Source" description="Update the app source in /Library/Application Support/token-workshed-nextcore/source.">
    <pkg-ref id="com.taylorhe.tokenworkshed.nextcore.source"/>
  </choice>
  <pkg-ref id="com.taylorhe.tokenworkshed.nextcore.ui" version="$VERSION" onConclusion="none">token-workshed-ui.pkg</pkg-ref>
  <pkg-ref id="com.taylorhe.tokenworkshed.nextcore.source" version="$VERSION" onConclusion="none">token-workshed-source.pkg</pkg-ref>
</installer-gui-script>
UPDATE_DISTXML

productbuild \
  --distribution "$UPDATE_DIST_FILE" \
  --package-path "$COMP_DIR" \
  "$UPDATE_OUT_PKG"

echo "Built UI/source update package (does not reinstall SGLang Runtime): $UPDATE_OUT_PKG"
