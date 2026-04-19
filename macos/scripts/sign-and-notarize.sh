#!/usr/bin/env bash
# Sign TeamAI.app with Developer ID and submit for notarization.
#
# Required environment variables:
#   DEVELOPER_ID         e.g. "Developer ID Application: Your Name (TEAMID)"
#   APPLE_ID_TEAM        Apple developer team id (10-char alphanumeric)
#
# Optional environment variables:
#   APP_PATH             Path to TeamAI.app (default: derived from xcodebuild)
#   NOTARY_PROFILE       Keychain profile created with `xcrun notarytool store-credentials`
#                        Falls back to APPLE_ID + APPLE_APP_PASSWORD if unset.
#   APPLE_ID             Apple ID email (only used if NOTARY_PROFILE unset)
#   APPLE_APP_PASSWORD   App-specific password (only used if NOTARY_PROFILE unset)
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ROOT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

: "${DEVELOPER_ID:?DEVELOPER_ID is required, e.g. 'Developer ID Application: Name (TEAMID)'}"
: "${APPLE_ID_TEAM:?APPLE_ID_TEAM is required (10-char team id)}"

if [ -d "/Applications/Xcode.app" ] && [ -z "${DEVELOPER_DIR:-}" ]; then
  export DEVELOPER_DIR="/Applications/Xcode.app/Contents/Developer"
fi

cd "$ROOT_DIR"

if [ -z "${APP_PATH:-}" ]; then
  DERIVED=$(xcodebuild -project TeamAI.xcodeproj -scheme TeamAI -configuration Release -showBuildSettings 2>/dev/null \
    | awk '/ BUILT_PRODUCTS_DIR /{print $3; exit}')
  APP_PATH="$DERIVED/TeamAI.app"
fi

if [ ! -d "$APP_PATH" ]; then
  echo "[sign] TeamAI.app not found at $APP_PATH — run scripts/build.sh first." >&2
  exit 1
fi

ENT="TeamAI/TeamAI.entitlements"
echo "[sign] Codesign with $DEVELOPER_ID"
codesign --force --deep --options=runtime --timestamp \
  --entitlements "$ENT" \
  --sign "$DEVELOPER_ID" "$APP_PATH"

codesign --verify --strict --deep --verbose=2 "$APP_PATH"

ZIP="${APP_PATH%/*}/TeamAI-notarize.zip"
echo "[notarize] Creating ditto zip at $ZIP"
ditto -c -k --keepParent "$APP_PATH" "$ZIP"

echo "[notarize] Submitting to notarytool"
if [ -n "${NOTARY_PROFILE:-}" ]; then
  xcrun notarytool submit "$ZIP" --keychain-profile "$NOTARY_PROFILE" --wait
else
  : "${APPLE_ID:?APPLE_ID required when NOTARY_PROFILE is unset}"
  : "${APPLE_APP_PASSWORD:?APPLE_APP_PASSWORD required when NOTARY_PROFILE is unset}"
  xcrun notarytool submit "$ZIP" \
    --apple-id "$APPLE_ID" \
    --team-id "$APPLE_ID_TEAM" \
    --password "$APPLE_APP_PASSWORD" \
    --wait
fi

echo "[notarize] Stapling ticket"
xcrun stapler staple "$APP_PATH"
xcrun stapler validate "$APP_PATH"
echo "[notarize] Done — $APP_PATH"
