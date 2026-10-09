#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-run}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP_BUNDLE="$ROOT_DIR/dist/token-workshed.app"
NATIVE_BINARY="$ROOT_DIR/native-ui/target/release/token-workshed-native-ui"
INITIAL_PAGE="${TOKEN_WORKSHED_INITIAL_PAGE:-chat}"
BUILD_PROFILE="${TOKEN_WORKSHED_BUILD_PROFILE:-release}"
case "$BUILD_PROFILE" in
  release) ;;
  debug) NATIVE_BINARY="$ROOT_DIR/native-ui/target/debug/token-workshed-native-ui" ;;
  *) echo "TOKEN_WORKSHED_BUILD_PROFILE must be release or debug." >&2; exit 2 ;;
esac

stop_native_ui() {
  local process_id executable
  while read -r process_id executable; do
    if [[ "$executable" == "$APP_BUNDLE/Contents/MacOS/token-workshed-native-ui" || "$executable" == "$NATIVE_BINARY" ]]; then
      kill -TERM "$process_id" >/dev/null 2>&1 || true
    fi
  done < <(/bin/ps -axo pid=,comm=)
}

bundle_native_running() {
  local process_id executable
  while read -r process_id executable; do
    if [[ "$executable" == "$APP_BUNDLE/Contents/MacOS/token-workshed-native-ui" ]]; then
      return 0
    fi
  done < <(/bin/ps -axo pid=,comm=)
  return 1
}

build() {
  if [[ "$BUILD_PROFILE" == "debug" ]]; then
    (cd "$ROOT_DIR/native-ui" && cargo build --offline --locked)
  else
    (cd "$ROOT_DIR/native-ui" && cargo build --release --offline --locked)
  fi
  TOKEN_WORKSHED_APP_BUNDLE_ID="${TOKEN_WORKSHED_APP_BUNDLE_ID:-com.taylorhe.tokenworkshed.dev020}" \
    TOKEN_WORKSHED_IDE_BUNDLE_ID="${TOKEN_WORKSHED_IDE_BUNDLE_ID:-com.taylorhe.tokenworkshed.dev020.ide}" \
    TOKEN_WORKSHED_NATIVE_UI_SOURCE_BIN="$NATIVE_BINARY" \
    bash "$ROOT_DIR/scripts/build_macos_app_bundle.sh"
}

open_app() {
  /usr/bin/open -n "$APP_BUNDLE" --args --initial-page "$INITIAL_PAGE"
}

stop_native_ui
build

case "$MODE" in
  run)
    open_app
    ;;
  --debug|debug)
    lldb -- "$NATIVE_BINARY"
    ;;
  --logs|logs)
    open_app
    exec /usr/bin/log stream --info --style compact --predicate 'process == "token-workshed-native-ui"'
    ;;
  --telemetry|telemetry)
    open_app
    exec /usr/bin/log stream --info --style compact --predicate 'process == "token-workshed-native-ui"'
    ;;
  --verify|verify)
    open_app
    for attempt in {1..30}; do
      if bundle_native_running; then
        exit 0
      fi
      sleep 1
    done
    echo "The current checkout's native UI did not start within 30 seconds." >&2
    exit 1
    ;;
  *)
    echo "usage: $0 [run|--debug|--logs|--telemetry|--verify]" >&2
    exit 2
    ;;
esac
