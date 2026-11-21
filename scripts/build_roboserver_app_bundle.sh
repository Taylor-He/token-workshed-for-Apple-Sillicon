#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_PATH="$ROOT_DIR/dist/token-workshed-roboserver.app"
CONTENTS="$APP_PATH/Contents"
MACOS_DIR="$CONTENTS/MacOS"
RESOURCES_DIR="$CONTENTS/Resources"
SOURCE_DIR="$RESOURCES_DIR/source"
ICON_SOURCE="$ROOT_DIR/assets/brand/nextcore-app-icon.png"
ICONSET="$ROOT_DIR/build/roboserver-icon.iconset"
BIN_SOURCE="$ROOT_DIR/native-ui/target/release/token-workshed-roboserver"
BACKUP_DIR="$ROOT_DIR/build/roboserver-app-backups"

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  echo "RoboServer app bundles are currently built for Apple Silicon macOS only." >&2
  exit 2
fi
if [[ ! -f "$ICON_SOURCE" ]]; then
  echo "Missing app icon: $ICON_SOURCE" >&2
  exit 1
fi

mkdir -p "$BACKUP_DIR" "$ROOT_DIR/build/swift-module-cache"
if [[ -e "$APP_PATH" ]]; then
  backup_name="token-workshed-roboserver.app.$(date +%Y%m%d-%H%M%S).bak"
  mv "$APP_PATH" "$BACKUP_DIR/$backup_name"
  echo "Moved previous RoboServer app to: $BACKUP_DIR/$backup_name"
fi

export MACOSX_DEPLOYMENT_TARGET=14.0
(
  cd "$ROOT_DIR/native-ui"
  cargo build --release --offline --bin token-workshed-roboserver
)

mkdir -p "$MACOS_DIR" "$RESOURCES_DIR" "$SOURCE_DIR/scripts" "$ICONSET"
cp "$BIN_SOURCE" "$MACOS_DIR/token-workshed-roboserver"
chmod +x "$MACOS_DIR/token-workshed-roboserver"
cp "$ROOT_DIR/scripts/roboserver-runner" "$RESOURCES_DIR/roboserver-runner"
cp "$ROOT_DIR/scripts/install_sglang_runtime.sh" "$SOURCE_DIR/scripts/install_sglang_runtime.sh"
chmod +x "$RESOURCES_DIR/roboserver-runner" "$SOURCE_DIR/scripts/install_sglang_runtime.sh"
cp "$ROOT_DIR/pyproject.toml" "$ROOT_DIR/README.md" "$SOURCE_DIR/"
mkdir -p "$SOURCE_DIR/token_workshed_nextcore"
rsync -a \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  "$ROOT_DIR/token_workshed_nextcore/" "$SOURCE_DIR/token_workshed_nextcore/"

for entry in \
  '16 16 icon_16x16.png' \
  '32 32 icon_16x16@2x.png' \
  '32 32 icon_32x32.png' \
  '64 64 icon_32x32@2x.png' \
  '128 128 icon_128x128.png' \
  '256 256 icon_128x128@2x.png' \
  '256 256 icon_256x256.png' \
  '512 512 icon_256x256@2x.png' \
  '512 512 icon_512x512.png' \
  '1024 1024 icon_512x512@2x.png'; do
  read -r width height file <<<"$entry"
  sips -z "$height" "$width" "$ICON_SOURCE" --out "$ICONSET/$file" >/dev/null
done
cat >"$ICONSET/Contents.json" <<'JSON'
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
# `actool` expects an asset-catalog wrapper around the iconset.
ASSETS="$ROOT_DIR/build/roboserver-icon.xcassets"
rm -rf "$ASSETS" "$ROOT_DIR/build/roboserver-icon-compiled"
mkdir -p "$ASSETS/AppIcon.appiconset" "$ROOT_DIR/build/roboserver-icon-compiled"
cp "$ICONSET"/*.png "$ASSETS/AppIcon.appiconset/"
cp "$ICONSET/Contents.json" "$ASSETS/AppIcon.appiconset/Contents.json"
xcrun actool \
  --compile "$ROOT_DIR/build/roboserver-icon-compiled" \
  --platform macosx \
  --minimum-deployment-target 14.0 \
  --target-device mac \
  --app-icon AppIcon \
  --output-partial-info-plist "$ROOT_DIR/build/roboserver-icon-info.plist" \
  "$ASSETS" >/dev/null
cp "$ROOT_DIR/build/roboserver-icon-compiled/AppIcon.icns" "$RESOURCES_DIR/AppIcon.icns"

printf 'APPL????' > "$CONTENTS/PkgInfo"
cat >"$CONTENTS/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleExecutable</key><string>token-workshed-roboserver</string>
  <key>CFBundleIdentifier</key><string>com.taylorhe.tokenworkshed.roboserver</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>token-workshed RoboServer</string>
  <key>CFBundleDisplayName</key><string>token-workshed RoboServer</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>0.1.0</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSPrincipalClass</key><string>NSApplication</string>
  <key>NSHighResolutionCapable</key><true/>
</dict>
</plist>
PLIST

/usr/bin/codesign --force --deep --sign - "$APP_PATH"
/usr/bin/codesign --verify --deep --strict --verbose=2 "$APP_PATH"
echo "Built RoboServer app bundle: $APP_PATH"
