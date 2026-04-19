#!/usr/bin/env bash
# Generate Xcode project and build TeamAI.app in Debug.
#
# Primary path: `xcodegen generate && xcodebuild ... build`
# Fallback path: direct swiftc + manual .app bundle assembly. Used when
# xcodebuild fails its preflight plug-in load (e.g. partially-initialized
# Xcode install where DVTDownloads/IDESimulatorFoundation are mismatched).
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

if ! command -v xcodegen >/dev/null 2>&1; then
  echo "[build] xcodegen not found. Install with: brew install xcodegen" >&2
  exit 1
fi

if [ -d "/Applications/Xcode.app" ] && [ -z "${DEVELOPER_DIR:-}" ]; then
  export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
fi

cd "$ROOT_DIR"
xcodegen generate

LOG=$(mktemp)
trap 'rm -f "$LOG"' EXIT

CONFIG="${CONFIGURATION:-Debug}"

if xcodebuild \
  -project TeamAI.xcodeproj \
  -scheme TeamAI \
  -configuration "$CONFIG" \
  -destination 'platform=macOS' \
  CODE_SIGN_IDENTITY="-" \
  CODE_SIGNING_REQUIRED=NO \
  CODE_SIGNING_ALLOWED=NO \
  build > "$LOG" 2>&1; then
  DERIVED=$(xcodebuild -project TeamAI.xcodeproj -scheme TeamAI -configuration "$CONFIG" -showBuildSettings 2>/dev/null | awk '/ BUILT_PRODUCTS_DIR /{print $3; exit}')
  echo "[build] xcodebuild OK — TeamAI.app in: $DERIVED"
  exit 0
fi

if grep -q "A required plugin failed to load" "$LOG"; then
  echo "[build] xcodebuild plug-ins failed to load. Falling back to direct swiftc."
  echo "[build] To fix the underlying Xcode install once: sudo xcodebuild -runFirstLaunch"
else
  echo "[build] xcodebuild failed:" >&2
  tail -40 "$LOG" >&2
  exit 1
fi

OUT="$ROOT_DIR/build/TeamAI.app"
rm -rf "$ROOT_DIR/build"
mkdir -p "$OUT/Contents/MacOS" "$OUT/Contents/Resources"
xcrun -sdk macosx swiftc \
  -target arm64-apple-macos13.0 \
  -O \
  -o "$OUT/Contents/MacOS/TeamAI" \
  "$ROOT_DIR"/TeamAI/*.swift
cp "$ROOT_DIR/TeamAI/Info.plist" "$OUT/Contents/Info.plist"
xattr -cr "$OUT"
codesign --force --sign - \
  --entitlements "$ROOT_DIR/TeamAI/TeamAI.entitlements" \
  --deep "$OUT" >/dev/null
codesign --verify --strict "$OUT"
echo "[build] swiftc fallback OK — TeamAI.app in: $OUT"
