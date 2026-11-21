#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export MACOSX_DEPLOYMENT_TARGET=14.0
APP_NAME="nextcore.app"
APP_PATH="$ROOT_DIR/dist/$APP_NAME"
MACOS_DIR="$APP_PATH/Contents/MacOS"
RESOURCES_DIR="$APP_PATH/Contents/Resources"
APP_ICON_SOURCE="$ROOT_DIR/assets/brand/nextcore-app-icon.png"
ROBO_SERVER_APP_SOURCE="$ROOT_DIR/dist/token-workshed-roboserver.app"
ICON_ASSET_CATALOG="$ROOT_DIR/build/NextCoreAssets.xcassets"
ICONSET_DIR="$ICON_ASSET_CATALOG/AppIcon.appiconset"
ICON_COMPILE_DIR="$ROOT_DIR/build/nextcore-icon-compiled"
ICON_INFO_PLIST="$ROOT_DIR/build/nextcore-icon-info.plist"
UI_HELPER_APP_PATH="$APP_PATH/Contents/Helpers/NextCore UI.app"
UI_HELPER_MACOS_DIR="$UI_HELPER_APP_PATH/Contents/MacOS"
COMPANION_APP_NAME="token-workshed NextCore IDE.app"
COMPANION_APP_PATH="$ROOT_DIR/dist/$COMPANION_APP_NAME"
APP_BUNDLE_ID="${TOKEN_WORKSHED_APP_BUNDLE_ID:-com.taylorhe.tokenworkshed.nextcore}"
COMPANION_BUNDLE_ID="${TOKEN_WORKSHED_IDE_BUNDLE_ID:-com.taylorhe.tokenworkshed.nextcore.ide}"
APP_VERSION="${TOKEN_WORKSHED_VERSION:-0.2.1}"
COMPANION_MACOS_DIR="$COMPANION_APP_PATH/Contents/MacOS"
COMPANION_RESOURCES_DIR="$COMPANION_APP_PATH/Contents/Resources"
NATIVE_UI_SRC="${TOKEN_WORKSHED_NATIVE_UI_SOURCE_BIN:-$ROOT_DIR/native-ui/target/release/token-workshed-native-ui}"
NATIVE_UI_DEST="$UI_HELPER_MACOS_DIR/token-workshed-native-ui"
LAUNCHER_SRC="$ROOT_DIR/native-ui/src/macos_app_launcher.swift"
BACKUP_DIR="$ROOT_DIR/build/app-backups"

if [[ ! "$APP_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Invalid app version: $APP_VERSION (expected MAJOR.MINOR.PATCH)." >&2
  exit 2
fi

if [[ ! -x "$NATIVE_UI_SRC" ]]; then
  echo "Missing native UI binary: $NATIVE_UI_SRC"
  echo "Build it first with: cd native-ui && cargo build --release"
  exit 1
fi

if [[ ! -f "$APP_ICON_SOURCE" ]]; then
  echo "Missing NextCore app icon: $APP_ICON_SOURCE" >&2
  exit 1
fi
if [[ ! -d "$ROBO_SERVER_APP_SOURCE" ]]; then
  echo "Build the RoboServer app first: bash scripts/build_roboserver_app_bundle.sh" >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR"
mkdir -p "$ROOT_DIR/build/swift-module-cache"
if [[ -e "$APP_PATH" ]]; then
  backup_name="nextcore.app.$(date +%Y%m%d-%H%M%S).bak"
  mv "$APP_PATH" "$BACKUP_DIR/$backup_name"
  echo "Moved existing app bundle to: $BACKUP_DIR/$backup_name"
fi

mkdir -p "$MACOS_DIR" "$RESOURCES_DIR" "$UI_HELPER_MACOS_DIR"
ditto "$ROBO_SERVER_APP_SOURCE" "$RESOURCES_DIR/token-workshed-roboserver.app"

rm -rf "$ICON_ASSET_CATALOG" "$ICON_COMPILE_DIR"
mkdir -p "$ICONSET_DIR" "$ICON_COMPILE_DIR"
sips -z 16 16 "$APP_ICON_SOURCE" --out "$ICONSET_DIR/icon_16x16.png" >/dev/null
sips -z 32 32 "$APP_ICON_SOURCE" --out "$ICONSET_DIR/icon_16x16@2x.png" >/dev/null
sips -z 32 32 "$APP_ICON_SOURCE" --out "$ICONSET_DIR/icon_32x32.png" >/dev/null
sips -z 64 64 "$APP_ICON_SOURCE" --out "$ICONSET_DIR/icon_32x32@2x.png" >/dev/null
sips -z 128 128 "$APP_ICON_SOURCE" --out "$ICONSET_DIR/icon_128x128.png" >/dev/null
sips -z 256 256 "$APP_ICON_SOURCE" --out "$ICONSET_DIR/icon_128x128@2x.png" >/dev/null
sips -z 256 256 "$APP_ICON_SOURCE" --out "$ICONSET_DIR/icon_256x256.png" >/dev/null
sips -z 512 512 "$APP_ICON_SOURCE" --out "$ICONSET_DIR/icon_256x256@2x.png" >/dev/null
sips -z 512 512 "$APP_ICON_SOURCE" --out "$ICONSET_DIR/icon_512x512.png" >/dev/null
sips -z 1024 1024 "$APP_ICON_SOURCE" --out "$ICONSET_DIR/icon_512x512@2x.png" >/dev/null
cat >"$ICONSET_DIR/Contents.json" <<'JSON'
{
  "images": [
    { "filename": "icon_16x16.png", "idiom": "mac", "scale": "1x", "size": "16x16" },
    { "filename": "icon_16x16@2x.png", "idiom": "mac", "scale": "2x", "size": "16x16" },
    { "filename": "icon_32x32.png", "idiom": "mac", "scale": "1x", "size": "32x32" },
    { "filename": "icon_32x32@2x.png", "idiom": "mac", "scale": "2x", "size": "32x32" },
    { "filename": "icon_128x128.png", "idiom": "mac", "scale": "1x", "size": "128x128" },
    { "filename": "icon_128x128@2x.png", "idiom": "mac", "scale": "2x", "size": "128x128" },
    { "filename": "icon_256x256.png", "idiom": "mac", "scale": "1x", "size": "256x256" },
    { "filename": "icon_256x256@2x.png", "idiom": "mac", "scale": "2x", "size": "256x256" },
    { "filename": "icon_512x512.png", "idiom": "mac", "scale": "1x", "size": "512x512" },
    { "filename": "icon_512x512@2x.png", "idiom": "mac", "scale": "2x", "size": "512x512" }
  ],
  "info": { "author": "xcode", "version": 1 }
}
JSON
xcrun actool \
  --compile "$ICON_COMPILE_DIR" \
  --platform macosx \
  --minimum-deployment-target 14.0 \
  --target-device mac \
  --app-icon AppIcon \
  --output-partial-info-plist "$ICON_INFO_PLIST" \
  "$ICON_ASSET_CATALOG" >/dev/null
cp "$ICON_COMPILE_DIR/AppIcon.icns" "$RESOURCES_DIR/AppIcon.icns"
cp "$ICON_COMPILE_DIR/Assets.car" "$RESOURCES_DIR/Assets.car"

cp "$NATIVE_UI_SRC" "$NATIVE_UI_DEST"
chmod +x "$NATIVE_UI_DEST"

printf 'APPL????' > "$UI_HELPER_APP_PATH/Contents/PkgInfo"
cat >"$UI_HELPER_APP_PATH/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>token-workshed-native-ui</string>
  <key>CFBundleIdentifier</key>
  <string>${APP_BUNDLE_ID}.ui</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>token-workshed NextCore</string>
  <key>CFBundleDisplayName</key>
  <string>token-workshed NextCore</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>${APP_VERSION}</string>
  <key>CFBundleVersion</key>
  <string>${APP_VERSION}</string>
  <key>LSUIElement</key>
  <true/>
</dict>
</plist>
PLIST

cat >"$MACOS_DIR/nextcore-runner" <<'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail

MACOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_BUNDLE_PATH="$(cd "$MACOS_DIR/../.." && pwd)"
DEV_SOURCE_DIR="$(cd "$MACOS_DIR/../../../.." && pwd)"
APP_VERSION="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleShortVersionString' "$MACOS_DIR/../Info.plist")"
APP_BUNDLE_ID="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$MACOS_DIR/../Info.plist")"
INSTALLED_SOURCE_DIR="/Library/Application Support/token-workshed-nextcore/source"
INSTALLED_PYTHON="/Library/Application Support/token-workshed-nextcore/runtime/.venv/bin/python3"

# CUA Driver's embedded runtime uses this only as a diagnostic label; macOS
# TCC attribution still comes from launching the Python process inside this app.
export TOKEN_WORKSHED_APP_BUNDLE_ID="${TOKEN_WORKSHED_APP_BUNDLE_ID:-$APP_BUNDLE_ID}"
export TOKEN_WORKSHED_NEXTCORE_APP_BUNDLE="${TOKEN_WORKSHED_NEXTCORE_APP_BUNDLE:-$APP_BUNDLE_PATH}"

SOURCE_DIR="${TOKEN_WORKSHED_SOURCE_DIR:-}"
if [[ -z "$SOURCE_DIR" ]]; then
  if [[ -f "$DEV_SOURCE_DIR/token_workshed_nextcore/desktop_ui.py" ]]; then
    SOURCE_DIR="$DEV_SOURCE_DIR"
  else
    SOURCE_DIR="$INSTALLED_SOURCE_DIR"
  fi
fi

if [[ ! -f "$SOURCE_DIR/token_workshed_nextcore/desktop_ui.py" ]]; then
  osascript -e "display dialog \"token-workshed NextCore source not found: $SOURCE_DIR\" buttons {\"OK\"} default button 1" >/dev/null 2>&1 || true
  echo "token-workshed NextCore source not found: $SOURCE_DIR" >&2
  exit 1
fi

export TOKEN_WORKSHED_NATIVE_UI_BIN="${TOKEN_WORKSHED_NATIVE_UI_BIN:-$MACOS_DIR/../Helpers/NextCore UI.app/Contents/MacOS/token-workshed-native-ui}"

cd "$SOURCE_DIR"

if [[ -n "${TOKEN_WORKSHED_PYTHON_BIN:-}" ]]; then
  export PYTHONPATH="$SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
  if [[ "${1:-}" == "--agent-scheduler" ]]; then
    shift
    exec "$TOKEN_WORKSHED_PYTHON_BIN" -m token_workshed_nextcore.agent.scheduler "$@"
  fi
  exec "$TOKEN_WORKSHED_PYTHON_BIN" -m token_workshed_nextcore.desktop_ui "$@"
fi

DEV_PYTHON="$DEV_SOURCE_DIR/.venv/bin/python"
python_has_pinned_sglang() {
  local candidate="$1"
  [[ -x "$candidate" ]] || return 1
  "$candidate" - <<'PY' >/dev/null 2>&1
import sys
import importlib.util
from importlib.metadata import PackageNotFoundError, version

try:
    installed = version("sglang")
except PackageNotFoundError:
    raise SystemExit(1)
has_runtime = importlib.util.find_spec("sglang") is not None
raise SystemExit(
    0
    if installed == "0.5.22" and sys.version_info[:2] == (3, 12) and has_runtime
    else 1
)
PY
}

if python_has_pinned_sglang "$DEV_PYTHON"; then
  export PYTHONPATH="$SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
  if [[ "${1:-}" == "--agent-scheduler" ]]; then
    shift
    exec "$DEV_PYTHON" -m token_workshed_nextcore.agent.scheduler "$@"
  fi
  exec "$DEV_PYTHON" -m token_workshed_nextcore.desktop_ui "$@"
fi

if python_has_pinned_sglang "$INSTALLED_PYTHON"; then
  export PYTHONPATH="$SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
  if [[ "${1:-}" == "--agent-scheduler" ]]; then
    shift
    exec "$INSTALLED_PYTHON" -m token_workshed_nextcore.agent.scheduler "$@"
  fi
  exec "$INSTALLED_PYTHON" -m token_workshed_nextcore.desktop_ui "$@"
fi

echo "No Python 3.12 runtime with pinned SGLang 0.5.22 was found." >&2
echo "Install the NextCore runtime or set TOKEN_WORKSHED_PYTHON_BIN explicitly." >&2
exit 1
LAUNCHER
chmod +x "$MACOS_DIR/nextcore-runner"
swiftc -target arm64-apple-macosx14.0 -O -module-cache-path "$ROOT_DIR/build/swift-module-cache" "$LAUNCHER_SRC" -o "$MACOS_DIR/nextcore"
chmod +x "$MACOS_DIR/nextcore"
printf 'APPL????' > "$APP_PATH/Contents/PkgInfo"

cat >"$APP_PATH/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>nextcore</string>
  <key>CFBundleIdentifier</key>
  <string>${APP_BUNDLE_ID}</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>token-workshed NextCore</string>
  <key>CFBundleDisplayName</key>
  <string>token-workshed NextCore</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>${APP_VERSION}</string>
  <key>CFBundleVersion</key>
  <string>${APP_VERSION}</string>
  <key>CFBundleIconFile</key>
  <string>AppIcon</string>
  <key>CFBundleIconName</key>
  <string>AppIcon</string>
  <key>CFBundleURLTypes</key>
  <array>
    <dict>
      <key>CFBundleURLName</key>
      <string>token-workshed NextCore IDE launcher</string>
      <key>CFBundleURLSchemes</key>
      <array>
        <string>tokenworkshednextcore</string>
      </array>
    </dict>
  </array>
  <key>LSMinimumSystemVersion</key>
  <string>14.0</string>
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

if [[ "${TOKEN_WORKSHED_SKIP_IDE_COMPANION:-0}" == "1" ]]; then
  echo "Skipping IDE Companion bundle as requested."
else
  if [[ -e "$COMPANION_APP_PATH" ]]; then
    companion_backup_name="token-workshed NextCore IDE.app.$(date +%Y%m%d-%H%M%S).bak"
    mv "$COMPANION_APP_PATH" "$BACKUP_DIR/$companion_backup_name"
    echo "Moved existing companion bundle to: $BACKUP_DIR/$companion_backup_name"
  fi

mkdir -p "$COMPANION_MACOS_DIR" "$COMPANION_RESOURCES_DIR"
cp "$NATIVE_UI_SRC" "$COMPANION_MACOS_DIR/token-workshed-native-ui"
chmod +x "$COMPANION_MACOS_DIR/token-workshed-native-ui"

cat >"$COMPANION_MACOS_DIR/nextcore-ide-runner" <<'COMPANION_LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail

MACOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_BASE="${TOKEN_WORKSHED_IDE_API_BASE:-http://127.0.0.1:7872}"
SERVER_URL="${TOKEN_WORKSHED_IDE_SERVER_URL:-http://127.0.0.1:8010}"
exec "$MACOS_DIR/token-workshed-native-ui" \
  --api-base "$API_BASE" \
  --server-url "$SERVER_URL" \
  --initial-page chat \
  --ide-companion \
  "$@"
COMPANION_LAUNCHER
chmod +x "$COMPANION_MACOS_DIR/nextcore-ide-runner"
swiftc -target arm64-apple-macosx14.0 -O -module-cache-path "$ROOT_DIR/build/swift-module-cache" "$LAUNCHER_SRC" -o "$COMPANION_MACOS_DIR/nextcore-ide"
chmod +x "$COMPANION_MACOS_DIR/nextcore-ide"
printf 'APPL????' > "$COMPANION_APP_PATH/Contents/PkgInfo"

cat >"$COMPANION_APP_PATH/Contents/Info.plist" <<COMPANION_PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key>
  <string>en</string>
  <key>CFBundleExecutable</key>
  <string>nextcore-ide</string>
  <key>CFBundleIdentifier</key>
  <string>${COMPANION_BUNDLE_ID}</string>
  <key>CFBundleInfoDictionaryVersion</key>
  <string>6.0</string>
  <key>CFBundleName</key>
  <string>token-workshed NextCore IDE</string>
  <key>CFBundleDisplayName</key>
  <string>token-workshed NextCore IDE</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>${APP_VERSION}</string>
  <key>CFBundleVersion</key>
  <string>${APP_VERSION}</string>
  <key>LSMinimumSystemVersion</key>
  <string>14.0</string>
  <key>NSPrincipalClass</key>
  <string>NSApplication</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
COMPANION_PLIST

/usr/bin/codesign --force --deep --sign - "$COMPANION_APP_PATH"

echo "Built IDE Companion bundle: $COMPANION_APP_PATH"
fi
