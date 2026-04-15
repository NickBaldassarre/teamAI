"""
Desktop control bridge using Claude Computer Use.

Unlike the patch-producing bridges (Codex, Gemini, Local), this bridge
executes OS-level actions — mouse clicks, keyboard input, app interactions —
via the Anthropic Computer Use API.  It follows the same structural patterns
(engine name, result dataclass, approval-gated execution) but does not
subclass AgentBridge because it does not produce unified diffs.

Usage:
    from teamai.integrations.desktop_bridge import DesktopBridge

    bridge = DesktopBridge()
    result = bridge.execute(task="Open Safari and search for 'MLX benchmarks'")
"""
from __future__ import annotations

import base64
import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_DESKTOP_MODEL = "claude-sonnet-4-20250514"
MAX_STEPS = 50
SCREENSHOT_DELAY = 1.0  # seconds to wait after an action before capturing

# Actions that require extra caution
_DESTRUCTIVE_PATTERNS = (
    "rm ", "delete", "trash", "empty trash", "format",
    "sudo", "shutdown", "restart", "kill",
)


@dataclass(frozen=True)
class DesktopAction:
    """A single action taken by the desktop bridge."""
    step: int
    action_type: str  # screenshot, click, type, key, scroll, move, wait
    parameters: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class DesktopExecutionResult:
    """Result of a desktop bridge execution."""
    engine: str = "desktop"
    model: str = ""
    task: str = ""
    status: str = "completed"  # completed | stopped | failed
    action_log: list[DesktopAction] = field(default_factory=list)
    total_steps: int = 0
    summary: str = ""
    error: str | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "model": self.model,
            "task": self.task,
            "status": self.status,
            "total_steps": self.total_steps,
            "summary": self.summary,
            "error": self.error,
            "action_log": [
                {
                    "step": a.step,
                    "action_type": a.action_type,
                    "parameters": {k: v for k, v in a.parameters.items() if k != "screenshot_data"},
                    "timestamp": a.timestamp,
                }
                for a in self.action_log
            ],
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class DesktopBridge:
    """Executes OS-level tasks via Claude Computer Use.

    The bridge loops through the Anthropic Computer Use protocol:
    1. Take a screenshot of the current display
    2. Send the task + screenshot to Claude with computer_use tool
    3. Execute the requested action (click, type, key, etc.)
    4. Repeat until Claude signals completion or the step limit is reached
    """

    engine = "desktop"
    default_model = DEFAULT_DESKTOP_MODEL

    def execute(
        self,
        task: str,
        *,
        model: str | None = None,
        max_steps: int = MAX_STEPS,
        display_width: int = 1920,
        display_height: int = 1080,
        require_approval_for_destructive: bool = True,
    ) -> DesktopExecutionResult:
        """Execute a desktop task and return a structured result."""
        model_name = (model or self.default_model).strip()
        client = _create_anthropic_client()
        result = DesktopExecutionResult(
            model=model_name,
            task=task,
        )

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task},
                ],
            },
        ]

        tools = [
            {
                "type": "computer_20250124",
                "name": "computer",
                "display_width_px": display_width,
                "display_height_px": display_height,
                "display_number": 1,
            },
        ]

        step = 0
        while step < max_steps:
            try:
                response = client.messages.create(
                    model=model_name,
                    max_tokens=4096,
                    messages=messages,
                    tools=tools,
                    system=(
                        "You are a desktop automation agent. Execute the user's task "
                        "by interacting with the computer. Use the computer tool to "
                        "take screenshots, click, type, and use keyboard shortcuts. "
                        "When the task is complete, respond with a text summary of "
                        "what you accomplished. Be precise with coordinates."
                    ),
                    betas=["computer-use-2025-01-24"],
                )
            except Exception as exc:
                result.status = "failed"
                result.error = str(exc)
                break

            # Process response content blocks
            tool_uses: list[dict[str, Any]] = []
            text_response = ""

            for block in response.content:
                block_type = getattr(block, "type", "")
                if block_type == "text":
                    text_response += getattr(block, "text", "")
                elif block_type == "tool_use":
                    tool_uses.append({
                        "id": getattr(block, "id", ""),
                        "name": getattr(block, "name", ""),
                        "input": getattr(block, "input", {}),
                    })

            # If model responded with text only (no tool use), it's done
            if not tool_uses:
                result.summary = text_response.strip()
                result.status = "completed"
                break

            # Execute each tool use and collect results
            tool_results: list[dict[str, Any]] = []
            for tool_use in tool_uses:
                step += 1
                action_input = tool_use.get("input", {})
                action_type = action_input.get("action", "unknown")

                # Safety check for destructive actions
                if require_approval_for_destructive and action_type == "type":
                    typed_text = action_input.get("text", "").lower()
                    if any(pat in typed_text for pat in _DESTRUCTIVE_PATTERNS):
                        logger.warning(
                            "Desktop bridge blocked destructive action at step %d: %s",
                            step, typed_text[:100],
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "content": "Action blocked: potentially destructive command detected. "
                                       "This action requires manual approval.",
                        })
                        result.action_log.append(DesktopAction(
                            step=step,
                            action_type="blocked",
                            parameters={"reason": "destructive_pattern", "text": typed_text[:100]},
                            timestamp=datetime.now(timezone.utc).isoformat(),
                        ))
                        continue

                # Execute the action
                action_result = _execute_computer_action(action_input)

                result.action_log.append(DesktopAction(
                    step=step,
                    action_type=action_type,
                    parameters=action_input,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                ))

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use["id"],
                    "content": action_result,
                })

            # Append assistant response and tool results to conversation
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            # Check stop reason
            if response.stop_reason == "end_turn":
                result.summary = text_response.strip() or "Task completed."
                result.status = "completed"
                break

        else:
            result.status = "stopped"
            result.summary = f"Reached maximum step limit ({max_steps})."

        result.total_steps = step
        result.completed_at = datetime.now(timezone.utc)
        return result


# ---------------------------------------------------------------------------
# Computer action execution (macOS)
# ---------------------------------------------------------------------------

def _execute_computer_action(action_input: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute a computer use action and return the tool result content."""
    action = action_input.get("action", "")

    if action == "screenshot":
        return _take_screenshot()

    if action == "mouse_move":
        x, y = action_input.get("coordinate", [0, 0])
        _run_cliclick(f"m:{x},{y}")
        time.sleep(0.3)
        return _take_screenshot()

    if action == "left_click":
        x, y = action_input.get("coordinate", [0, 0])
        _run_cliclick(f"c:{x},{y}")
        time.sleep(SCREENSHOT_DELAY)
        return _take_screenshot()

    if action == "right_click":
        x, y = action_input.get("coordinate", [0, 0])
        _run_cliclick(f"rc:{x},{y}")
        time.sleep(SCREENSHOT_DELAY)
        return _take_screenshot()

    if action == "double_click":
        x, y = action_input.get("coordinate", [0, 0])
        _run_cliclick(f"dc:{x},{y}")
        time.sleep(SCREENSHOT_DELAY)
        return _take_screenshot()

    if action == "left_click_drag":
        start = action_input.get("start_coordinate", [0, 0])
        end = action_input.get("coordinate", [0, 0])
        _run_cliclick(f"dd:{start[0]},{start[1]}", f"du:{end[0]},{end[1]}")
        time.sleep(SCREENSHOT_DELAY)
        return _take_screenshot()

    if action == "type":
        text = action_input.get("text", "")
        _run_cliclick(f"t:{text}")
        time.sleep(0.5)
        return _take_screenshot()

    if action == "key":
        key_combo = action_input.get("text", "")
        _press_key(key_combo)
        time.sleep(SCREENSHOT_DELAY)
        return _take_screenshot()

    if action == "scroll":
        x, y = action_input.get("coordinate", [0, 0])
        direction = action_input.get("direction", "down")
        amount = action_input.get("amount", 3)
        delta = -amount if direction == "down" else amount
        _run_cliclick(f"m:{x},{y}")
        # Use AppleScript for scroll since cliclick doesn't support it natively
        _run_applescript(f'tell application "System Events" to scroll area 1 by {delta}')
        time.sleep(SCREENSHOT_DELAY)
        return _take_screenshot()

    if action == "wait":
        duration = action_input.get("duration", 1)
        time.sleep(min(duration, 10))
        return _take_screenshot()

    return [{"type": "text", "text": f"Unknown action: {action}"}]


def _take_screenshot() -> list[dict[str, Any]]:
    """Capture the screen and return base64-encoded image content."""
    tmp_path = Path("/tmp/teamai_desktop_screenshot.png")
    try:
        subprocess.run(
            ["screencapture", "-x", "-C", str(tmp_path)],
            capture_output=True,
            timeout=10,
        )
        if tmp_path.exists():
            data = tmp_path.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            return [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                },
            ]
    except Exception as exc:
        logger.warning("Screenshot failed: %s", exc)
    return [{"type": "text", "text": "Screenshot capture failed."}]


def _run_cliclick(*args: str) -> None:
    """Run cliclick for mouse/keyboard simulation.

    cliclick is a lightweight macOS CLI tool for simulating input events.
    Install via: brew install cliclick
    """
    try:
        subprocess.run(
            ["cliclick"] + list(args),
            capture_output=True,
            timeout=5,
        )
    except FileNotFoundError:
        # Fall back to AppleScript for basic operations
        logger.warning("cliclick not found. Install via: brew install cliclick")
        _fallback_applescript_click(args)
    except subprocess.TimeoutExpired:
        logger.warning("cliclick timed out for args: %s", args)


def _fallback_applescript_click(args: tuple[str, ...]) -> None:
    """Minimal AppleScript fallback for click operations when cliclick isn't installed."""
    for arg in args:
        if arg.startswith("c:"):
            coords = arg[2:].split(",")
            if len(coords) == 2:
                x, y = coords
                _run_applescript(
                    f'tell application "System Events" to click at {{{x}, {y}}}'
                )
        elif arg.startswith("t:"):
            text = arg[2:]
            _run_applescript(
                f'tell application "System Events" to keystroke "{text}"'
            )


def _press_key(key_combo: str) -> None:
    """Press a key or key combination using AppleScript.

    Handles combos like 'Return', 'cmd+c', 'ctrl+shift+t'.
    """
    parts = [p.strip().lower() for p in key_combo.split("+")]
    key = parts[-1]
    modifiers = parts[:-1]

    # Map modifier names to AppleScript modifier keywords
    mod_map = {
        "cmd": "command down",
        "command": "command down",
        "ctrl": "control down",
        "control": "control down",
        "alt": "option down",
        "option": "option down",
        "shift": "shift down",
    }

    # Map special key names to AppleScript key codes
    key_code_map = {
        "return": 36, "enter": 36,
        "tab": 48,
        "delete": 51, "backspace": 51,
        "escape": 53, "esc": 53,
        "space": 49,
        "up": 126, "down": 125, "left": 123, "right": 124,
        "home": 115, "end": 119,
        "pageup": 116, "pagedown": 121,
        "f1": 122, "f2": 120, "f3": 99, "f4": 118,
        "f5": 96, "f6": 97, "f7": 98, "f8": 100,
    }

    modifier_clause = ""
    if modifiers:
        using = ", ".join(mod_map.get(m, f"{m} down") for m in modifiers)
        modifier_clause = f" using {{{using}}}"

    if key in key_code_map:
        script = f'tell application "System Events" to key code {key_code_map[key]}{modifier_clause}'
    else:
        script = f'tell application "System Events" to keystroke "{key}"{modifier_clause}'

    _run_applescript(script)


def _run_applescript(script: str) -> str:
    """Execute an AppleScript and return stdout."""
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception as exc:
        logger.warning("AppleScript failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# Anthropic client
# ---------------------------------------------------------------------------

def _create_anthropic_client() -> Any:
    api_key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export it before using the desktop bridge."
        )

    try:
        from anthropic import Anthropic
    except ImportError as exc:
        raise RuntimeError(
            "The Anthropic SDK is not installed. Install it: pip install anthropic"
        ) from exc

    return Anthropic(api_key=api_key)
