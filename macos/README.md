# TeamAI macOS App (Phase 1)

Native menu-bar wrapper that embeds the existing teamAI FastAPI dashboard in a
WKWebView and manages the local Python daemon process.

## What it does

- Lives in the menu bar (`LSUIElement=true`) with a colored status dot:
  - gray = starting / stopped
  - green = ready (`/healthz` returns 200)
  - red = error (Python missing, daemon never went healthy, etc.)
- Spawns the daemon on launch via `python -m teamai daemon start ...`.
- Polls `/healthz` every 500 ms (up to 30 s) and flips to *ready*.
- Polls `/v1/dashboard/summary` every 5 s and fires native notifications when:
  - a queued/running job transitions to `completed` or `failed`
  - the pending-approval count grows
  - `safety.posture` flips
- Subscribes to `/v1/jobs/<id>/events/stream` for active jobs (SSE channel
  reserved for future progress UI; final transitions are detected by the poll).
- Settings window for port, Python path, workspace, and Launch-at-Login
  (`SMAppService.mainApp`, requires macOS 13+).
- On quit: SIGTERM → wait 5 s → SIGKILL on the daemon PID
  (`~/.teamai/daemon.pid`).

## Build

```bash
brew install xcodegen
cd macos
./scripts/build.sh
```

`build.sh` runs `xcodegen generate` and then `xcodebuild -scheme TeamAI -configuration Debug build`.
The unsigned `.app` lives in
`~/Library/Developer/Xcode/DerivedData/TeamAI-.../Build/Products/Debug/TeamAI.app`.

If you only have the Xcode Command Line Tools installed, point at the full
Xcode bundle first:

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
```

## Run

Double-click `TeamAI.app`, or:

```bash
open ~/Library/Developer/Xcode/DerivedData/TeamAI-*/Build/Products/Debug/TeamAI.app
```

The menu-bar item appears, the daemon starts, and **Open Dashboard** opens a
1280×800 window pointed at `http://127.0.0.1:8000/dashboard`.

The first launch asks for permission to deliver notifications. The daemon
spawn uses the project virtualenv at `<workspace>/.venv/bin/python` if it
exists; otherwise it falls back to the first system Python found (`/opt/homebrew/bin/python3` → `/usr/local/bin/python3` → `/usr/bin/python3`).
You can override this and the workspace path via **Settings…**.

## Verify

```bash
curl -s http://127.0.0.1:8000/healthz | jq .
pgrep -fa "teamai daemon"
```

After quitting the app, `pgrep -f "teamai daemon"` should return nothing.

## Sign + notarize (Release)

```bash
export DEVELOPER_ID="Developer ID Application: Your Name (TEAMID)"
export APPLE_ID_TEAM="TEAMID"
# Either store credentials once via:
#   xcrun notarytool store-credentials teamai-notary --apple-id you@example.com --team-id $APPLE_ID_TEAM
# and then:
export NOTARY_PROFILE=teamai-notary
# Or fall back to:
#   export APPLE_ID=you@example.com
#   export APPLE_APP_PASSWORD=app-specific-password

# 1. Release build
xcodegen generate
xcodebuild -project TeamAI.xcodeproj -scheme TeamAI -configuration Release build

# 2. Sign + notarize + staple
./scripts/sign-and-notarize.sh
```

## Constraints (Phase 1)

- macOS 13 Ventura minimum (`SMAppService.mainApp`).
- No App Sandbox.
- No bundled Python — the app shells out to whatever `pythonPath` resolves to.
- No external Swift dependencies. Only `Foundation`, `AppKit`, `SwiftUI`,
  `WebKit`, `UserNotifications`, `ServiceManagement`.

## Phase-2 followups

- App Sandbox + signed entitlements
- Bundle a Python runtime inside the `.app`
- SSE-driven approval-pending stream (right now it relies on the 5 s summary)
- iOS companion target
- In-app approval apply/reject UI (currently the embedded web dashboard handles
  approvals)
