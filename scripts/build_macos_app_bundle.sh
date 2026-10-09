#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_NAME="token-workshed.app"
APP_PATH="$ROOT_DIR/dist/$APP_NAME"
MACOS_DIR="$APP_PATH/Contents/MacOS"
RESOURCES_DIR="$APP_PATH/Contents/Resources"
COMPANION_APP_NAME="token-workshed-ide.app"
COMPANION_APP_PATH="$ROOT_DIR/dist/$COMPANION_APP_NAME"
APP_BUNDLE_ID="${TOKEN_WORKSHED_APP_BUNDLE_ID:-com.taylorhe.tokenworkshed}"
COMPANION_BUNDLE_ID="${TOKEN_WORKSHED_IDE_BUNDLE_ID:-com.taylorhe.tokenworkshed.ide}"
COMPANION_MACOS_DIR="$COMPANION_APP_PATH/Contents/MacOS"
COMPANION_RESOURCES_DIR="$COMPANION_APP_PATH/Contents/Resources"
NATIVE_UI_SRC="${TOKEN_WORKSHED_NATIVE_UI_SOURCE_BIN:-$ROOT_DIR/native-ui/target/release/token-workshed-native-ui}"
NATIVE_UI_DEST="$MACOS_DIR/token-workshed-native-ui"
LAUNCHER_SRC="$ROOT_DIR/native-ui/src/macos_app_launcher.swift"
BACKUP_DIR="$ROOT_DIR/build/app-backups"

if [[ ! -x "$NATIVE_UI_SRC" ]]; then
  echo "Missing native UI binary: $NATIVE_UI_SRC"
  echo "Build it first with: cd native-ui && cargo build --release"
  exit 1
fi

mkdir -p "$BACKUP_DIR"
mkdir -p "$ROOT_DIR/build/swift-module-cache"
if [[ -e "$APP_PATH" ]]; then
  backup_name="token-workshed.app.$(date +%Y%m%d-%H%M%S).bak"
  mv "$APP_PATH" "$BACKUP_DIR/$backup_name"
  echo "Moved existing app bundle to: $BACKUP_DIR/$backup_name"
fi

mkdir -p "$MACOS_DIR" "$RESOURCES_DIR"

cp "$NATIVE_UI_SRC" "$NATIVE_UI_DEST"
chmod +x "$NATIVE_UI_DEST"

cat >"$MACOS_DIR/token-workshed-runner" <<'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail

MACOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV_SOURCE_DIR="$(cd "$MACOS_DIR/../../../.." && pwd)"
INSTALLED_SOURCE_DIR="/Library/Application Support/token-workshed/token-workshed-0.2.0"
INSTALLED_PYTHON="/Library/Application Support/token-workshed/runtime/.venv/bin/python3"

SOURCE_DIR="${TOKEN_WORKSHED_SOURCE_DIR:-}"
if [[ -z "$SOURCE_DIR" ]]; then
  if [[ -f "$DEV_SOURCE_DIR/vllm_mlx/desktop_ui.py" ]]; then
    SOURCE_DIR="$DEV_SOURCE_DIR"
  else
    SOURCE_DIR="$INSTALLED_SOURCE_DIR"
  fi
fi

if [[ ! -f "$SOURCE_DIR/vllm_mlx/desktop_ui.py" ]]; then
  osascript -e "display dialog \"token-workshed source not found: $SOURCE_DIR\" buttons {\"OK\"} default button 1" >/dev/null 2>&1 || true
  echo "token-workshed source not found: $SOURCE_DIR" >&2
  exit 1
fi

export TOKEN_WORKSHED_NATIVE_UI_BIN="${TOKEN_WORKSHED_NATIVE_UI_BIN:-$MACOS_DIR/token-workshed-native-ui}"

cd "$SOURCE_DIR"

if [[ -n "${TOKEN_WORKSHED_PYTHON_BIN:-}" ]]; then
  export PYTHONPATH="$SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
  exec "$TOKEN_WORKSHED_PYTHON_BIN" -m vllm_mlx.desktop_ui "$@"
fi

DEV_PYTHON="$DEV_SOURCE_DIR/.venv/bin/python"
if [[ -x "$DEV_PYTHON" ]]; then
  export PYTHONPATH="$SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
  exec "$DEV_PYTHON" -m vllm_mlx.desktop_ui "$@"
fi

if [[ -x "$INSTALLED_PYTHON" ]]; then
  export PYTHONPATH="$SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
  exec "$INSTALLED_PYTHON" -m vllm_mlx.desktop_ui "$@"
fi

if command -v uv >/dev/null 2>&1 && [[ -f "$SOURCE_DIR/uv.lock" ]]; then
  exec uv run python -m vllm_mlx.desktop_ui "$@"
fi

if command -v python3 >/dev/null 2>&1; then
  export PYTHONPATH="$SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
  exec python3 -m vllm_mlx.desktop_ui "$@"
fi

echo "No usable Python runtime found. Install uv or run the installer runtime step first." >&2
exit 1
LAUNCHER
chmod +x "$MACOS_DIR/token-workshed-runner"
swiftc -O -module-cache-path "$ROOT_DIR/build/swift-module-cache" "$LAUNCHER_SRC" -o "$MACOS_DIR/token-workshed"
chmod +x "$MACOS_DIR/token-workshed"
printf 'APPL????' > "$APP_PATH/Contents/PkgInfo"

cat >"$APP_PATH/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>token-workshed</string>
  <key>CFBundleIdentifier</key>
  <string>${APP_BUNDLE_ID}</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>token-workshed</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.2.0</string>
  <key>CFBundleVersion</key>
  <string>0.2.0</string>
  <key>CFBundleURLTypes</key>
  <array>
    <dict>
      <key>CFBundleURLName</key>
      <string>Token Workshed IDE launcher</string>
      <key>CFBundleURLSchemes</key>
      <array>
        <string>tokenworkshed</string>
      </array>
    </dict>
  </array>
  <key>LSMinimumSystemVersion</key>
  <string>13.0</string>
  <key>NSPrincipalClass</key>
  <string>NSApplication</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
PLIST

# The Swift launcher is linked with a temporary ad-hoc signature, while this
# script writes Info.plist and nested executables afterward. Seal the finished
# local bundle so Launch Services can verify its executable and metadata.
/usr/bin/codesign --force --deep --sign - "$APP_PATH"

echo "Built app bundle: $APP_PATH"
echo "Native UI binary: $NATIVE_UI_DEST"

if [[ -e "$COMPANION_APP_PATH" ]]; then
  companion_backup_name="token-workshed-ide.app.$(date +%Y%m%d-%H%M%S).bak"
  mv "$COMPANION_APP_PATH" "$BACKUP_DIR/$companion_backup_name"
  echo "Moved existing companion bundle to: $BACKUP_DIR/$companion_backup_name"
fi

mkdir -p "$COMPANION_MACOS_DIR" "$COMPANION_RESOURCES_DIR"
cp "$NATIVE_UI_SRC" "$COMPANION_MACOS_DIR/token-workshed-native-ui"
chmod +x "$COMPANION_MACOS_DIR/token-workshed-native-ui"

cat >"$COMPANION_MACOS_DIR/token-workshed-ide-runner" <<'COMPANION_LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail

MACOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_BASE="${TOKEN_WORKSHED_IDE_API_BASE:-http://127.0.0.1:7862}"
SERVER_URL="${TOKEN_WORKSHED_IDE_SERVER_URL:-http://127.0.0.1:8000}"
exec "$MACOS_DIR/token-workshed-native-ui" \
  --api-base "$API_BASE" \
  --server-url "$SERVER_URL" \
  --initial-page chat \
  --ide-companion \
  "$@"
COMPANION_LAUNCHER
chmod +x "$COMPANION_MACOS_DIR/token-workshed-ide-runner"
swiftc -O -module-cache-path "$ROOT_DIR/build/swift-module-cache" "$LAUNCHER_SRC" -o "$COMPANION_MACOS_DIR/token-workshed-ide"
chmod +x "$COMPANION_MACOS_DIR/token-workshed-ide"
printf 'APPL????' > "$COMPANION_APP_PATH/Contents/PkgInfo"

cat >"$COMPANION_APP_PATH/Contents/Info.plist" <<COMPANION_PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>token-workshed-ide</string>
  <key>CFBundleIdentifier</key>
  <string>${COMPANION_BUNDLE_ID}</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>Token Workshed IDE</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>0.2.0</string>
  <key>CFBundleVersion</key>
  <string>0.2.0</string>
  <key>LSMinimumSystemVersion</key>
  <string>13.0</string>
  <key>NSPrincipalClass</key>
  <string>NSApplication</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
COMPANION_PLIST

/usr/bin/codesign --force --deep --sign - "$COMPANION_APP_PATH"

echo "Built IDE Companion bundle: $COMPANION_APP_PATH"
