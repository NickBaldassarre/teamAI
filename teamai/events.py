from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from .schemas import RunEvent

_ROUND_STAGE_PATTERN = re.compile(r"Round\s+(?P<round>\d+)/(?P<total>\d+):\s+(?P<stage>.+)")


def build_run_event(*, sequence: int, message: str) -> RunEvent:
    timestamp = datetime.now(timezone.utc)
    kind = "progress"
    round_number: int | None = None
    stage: str | None = None
    terminal = False
    data: dict[str, Any] = {}

    stripped = message.strip()
    round_match = _ROUND_STAGE_PATTERN.match(stripped)
    if stripped.startswith("Starting run in "):
        kind = "run_started"
    elif stripped.startswith("Task route: "):
        kind = "task_route_selected"
        task_route = stripped.split(":", maxsplit=1)[1].strip()
        if task_route:
            data["task_route"] = task_route
            stage = task_route
    elif stripped.startswith("Completed: "):
        kind = "run_completed"
        terminal = True
        stop_reason = stripped.split(":", maxsplit=1)[1].strip()
        if stop_reason:
            data["stop_reason"] = stop_reason
    elif stripped.startswith("Stopped: "):
        kind = "run_stopped"
        terminal = True
        stop_reason = stripped.split(":", maxsplit=1)[1].strip()
        if stop_reason:
            data["stop_reason"] = stop_reason
    elif stripped.startswith("Failed: "):
        kind = "run_failed"
        terminal = True
        stop_reason = stripped.split(":", maxsplit=1)[1].strip()
        if stop_reason:
            data["stop_reason"] = stop_reason
    elif round_match:
        round_number = int(round_match.group("round"))
        stage = _slugify_stage(round_match.group("stage"))
        kind = "round_stage"
        if stage == "executing_tool_actions":
            kind = "tool_execution"
        data["total_rounds"] = int(round_match.group("total"))
        data["raw_stage"] = round_match.group("stage").strip()

    return RunEvent(
        sequence=sequence,
        timestamp=timestamp,
        kind=kind,
        message=message,
        round_number=round_number,
        stage=stage,
        terminal=terminal,
        data=data,
    )


def build_status_event(
    *,
    sequence: int,
    kind: str,
    message: str,
    terminal: bool = False,
    data: dict[str, Any] | None = None,
) -> RunEvent:
    return RunEvent(
        sequence=sequence,
        timestamp=datetime.now(timezone.utc),
        kind=kind,
        message=message,
        terminal=terminal,
        data=data or {},
    )


def render_sse_event(*, event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


def _slugify_stage(stage: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", stage.lower()).strip("_")
    return normalized or "progress"
