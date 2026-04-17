"""
Inter-agent conversation channel for teamAI.

Provides a persistent, append-only JSONL-backed channel so any agent
can post conversational turns and any other agent can read them during
task execution.  Each task_id gets its own file at::

    .teamai/conversation/<task_id>/turns.jsonl

Usage::

    from teamai.conversation import ConversationChannel

    ch = ConversationChannel(workspace=Path("/my/project"))
    turn = ch.post(
        task_id="t1",
        agent_id="planner",
        role="observation",
        content="Found 3 test failures in auth module.",
    )
    recent = ch.read_latest("t1", count=10)
"""
from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Turn model
# ---------------------------------------------------------------------------

TurnRole = Literal[
    "observation",
    "request",
    "response",
    "handoff",
    "escalation",
    "status_update",
]


class ConversationTurn(BaseModel):
    """A single turn in an inter-agent conversation."""

    turn_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    task_id: str
    agent_id: str
    role: TurnRole
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


# ---------------------------------------------------------------------------
# Conversation directory layout
# ---------------------------------------------------------------------------

_CONVERSATION_DIR = "conversation"
_TURNS_FILE = "turns.jsonl"


def _turns_path(workspace: Path, task_id: str) -> Path:
    return workspace / ".teamai" / _CONVERSATION_DIR / task_id / _TURNS_FILE


# ---------------------------------------------------------------------------
# ConversationChannel
# ---------------------------------------------------------------------------


class ConversationChannel:
    """Thread-safe, append-only conversation store backed by JSONL files.

    One file per *task_id* keeps things simple for concurrent reads and
    makes cleanup straightforward (delete the directory).
    """

    def __init__(self, workspace: Path) -> None:
        self._workspace = workspace
        # Per-task locks so concurrent writes to different tasks don't block
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    # -- internal helpers ---------------------------------------------------

    def _lock_for(self, task_id: str) -> threading.Lock:
        """Return (or lazily create) a lock for *task_id*."""
        with self._global_lock:
            if task_id not in self._locks:
                self._locks[task_id] = threading.Lock()
            return self._locks[task_id]

    @staticmethod
    def _serialize_turn(turn: ConversationTurn) -> str:
        return turn.model_dump_json()

    @staticmethod
    def _deserialize_line(line: str) -> ConversationTurn | None:
        """Parse a single JSONL line, returning *None* on corruption."""
        stripped = line.strip()
        if not stripped:
            return None
        try:
            return ConversationTurn.model_validate_json(stripped)
        except Exception:  # noqa: BLE001
            logger.debug("Skipping corrupt conversation line: %.120s", stripped)
            return None

    def _read_all_turns(self, task_id: str) -> list[ConversationTurn]:
        """Read every valid turn from the JSONL file for *task_id*."""
        path = _turns_path(self._workspace, task_id)
        if not path.exists():
            return []
        turns: list[ConversationTurn] = []
        try:
            with path.open("r", encoding="utf-8") as fh:
                for raw_line in fh:
                    turn = self._deserialize_line(raw_line)
                    if turn is not None:
                        turns.append(turn)
        except OSError:
            logger.warning("Could not read conversation file: %s", path)
        return turns

    # -- public API ---------------------------------------------------------

    def post(
        self,
        *,
        task_id: str,
        agent_id: str,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append a turn and return its dict representation."""
        turn = ConversationTurn(
            task_id=task_id,
            agent_id=agent_id,
            role=role,  # type: ignore[arg-type]  # validated by Pydantic
            content=content,
            metadata=metadata or {},
        )
        path = _turns_path(self._workspace, task_id)

        lock = self._lock_for(task_id)
        with lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(self._serialize_turn(turn) + "\n")

        return turn.model_dump(mode="json")

    def read(
        self,
        task_id: str,
        *,
        after_turn: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Read turns for *task_id*, optionally after a cursor turn_id."""
        lock = self._lock_for(task_id)
        with lock:
            turns = self._read_all_turns(task_id)

        if after_turn is not None:
            # Find the cursor index and slice everything after it.
            cursor_idx: int | None = None
            for idx, t in enumerate(turns):
                if t.turn_id == after_turn:
                    cursor_idx = idx
                    break
            if cursor_idx is not None:
                turns = turns[cursor_idx + 1 :]
            else:
                # Unknown cursor -- return from the beginning so no data
                # is silently lost.
                pass

        return [t.model_dump(mode="json") for t in turns[:limit]]

    def read_latest(
        self,
        task_id: str,
        *,
        count: int = 5,
    ) -> list[dict[str, Any]]:
        """Return the *count* most recent turns (oldest first)."""
        lock = self._lock_for(task_id)
        with lock:
            turns = self._read_all_turns(task_id)
        return [t.model_dump(mode="json") for t in turns[-count:]]

    def summary(self, task_id: str) -> dict[str, Any]:
        """Return a lightweight summary of the conversation for *task_id*."""
        lock = self._lock_for(task_id)
        with lock:
            turns = self._read_all_turns(task_id)

        if not turns:
            return {
                "task_id": task_id,
                "turn_count": 0,
                "agents": [],
                "latest_timestamp": None,
            }

        agents = sorted({t.agent_id for t in turns})
        latest_ts = max(t.timestamp for t in turns)
        return {
            "task_id": task_id,
            "turn_count": len(turns),
            "agents": agents,
            "latest_timestamp": latest_ts.isoformat(),
        }

    def list_conversations(self) -> list[str]:
        """Return task_ids that have conversation directories."""
        conv_root = self._workspace / ".teamai" / _CONVERSATION_DIR
        if not conv_root.is_dir():
            return []
        task_ids: list[str] = []
        try:
            for child in sorted(conv_root.iterdir()):
                if child.is_dir() and (child / _TURNS_FILE).exists():
                    task_ids.append(child.name)
        except OSError:
            logger.warning("Could not list conversation directory: %s", conv_root)
        return task_ids
