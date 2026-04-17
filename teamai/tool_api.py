from __future__ import annotations

import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from .config import Settings
from .schemas import ToolAction, ToolExecutionResult
from .tools import WorkspaceTools

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ToolExecuteRequest(BaseModel):
    tool_action: ToolAction
    task_id: str
    execution_mode: str = "read_only"


class ToolBatchRequest(BaseModel):
    actions: list[ToolAction]
    task_id: str
    execution_mode: str = "read_only"


class ToolDescription(BaseModel):
    name: str
    description: str
    args: dict[str, str] = Field(default_factory=dict)


class ToolListResponse(BaseModel):
    tools: list[ToolDescription]


class ToolExecuteResponse(BaseModel):
    result: ToolExecutionResult


class ToolBatchResponse(BaseModel):
    results: list[ToolExecutionResult]


# ---------------------------------------------------------------------------
# Tool catalogue (static metadata surfaced by GET /v1/tools)
# ---------------------------------------------------------------------------

_TOOL_CATALOGUE: list[ToolDescription] = [
    ToolDescription(
        name="list_files",
        description="List files in a directory.",
        args={"path": ".", "recursive": "false", "max_entries": "50"},
    ),
    ToolDescription(
        name="search_text",
        description="Search for a text pattern in files using ripgrep.",
        args={"pattern": "<required>", "path": ".", "max_matches": "40"},
    ),
    ToolDescription(
        name="read_file",
        description="Read the contents of a file (line range).",
        args={"path": "<required>", "start_line": "1", "end_line": "200"},
    ),
    ToolDescription(
        name="run_command",
        description="Run a read-only shell command from an allowlist.",
        args={"command": "<required>", "cwd": "."},
    ),
    ToolDescription(
        name="write_file",
        description="Write content to a file (requires workspace_write mode).",
        args={"path": "<required>", "content": "<required>"},
    ),
    ToolDescription(
        name="replace_in_file",
        description="Replace text in a file (requires workspace_write mode).",
        args={"path": "<required>", "old_text": "<required>", "new_text": "<required>", "replace_all": "false"},
    ),
    ToolDescription(
        name="apply_patch",
        description="Apply a patch to a file (requires workspace_write mode).",
        args={"path": "<required>", "patch": "<required>"},
    ),
    ToolDescription(
        name="run_checks",
        description="Run project checks (linting, tests) on given paths.",
        args={"paths": "[]"},
    ),
    ToolDescription(
        name="create_branch",
        description="Create a new git branch.",
        args={"name": "<required>"},
    ),
    ToolDescription(
        name="git_add",
        description="Stage files with git add.",
        args={"paths": "[]"},
    ),
    ToolDescription(
        name="git_commit",
        description="Create a git commit.",
        args={"message": "<required>"},
    ),
    ToolDescription(
        name="git_restore",
        description="Restore files with git restore.",
        args={"paths": "[]"},
    ),
    ToolDescription(
        name="spawn_agent",
        description="Spawn a sub-agent to handle a subtask.",
        args={"task": "<required>", "execution_mode": "read_only"},
    ),
]

# ---------------------------------------------------------------------------
# In-memory rate limiter: max 20 calls per minute per task_id
# ---------------------------------------------------------------------------

_RATE_LIMIT_MAX_CALLS = 20
_RATE_LIMIT_WINDOW_SECONDS = 60.0


class _RateLimiter:
    """Simple sliding-window counter per task_id."""

    def __init__(self, max_calls: int = _RATE_LIMIT_MAX_CALLS, window: float = _RATE_LIMIT_WINDOW_SECONDS) -> None:
        self._max_calls = max_calls
        self._window = window
        self._calls: dict[str, list[float]] = defaultdict(list)

    def check(self, task_id: str, count: int = 1) -> None:
        """Raise HTTPException(429) if the task_id would exceed the limit."""
        now = time.monotonic()
        cutoff = now - self._window
        timestamps = self._calls[task_id]
        # Prune expired entries
        self._calls[task_id] = [ts for ts in timestamps if ts > cutoff]
        if len(self._calls[task_id]) + count > self._max_calls:
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Rate limit exceeded for task_id={task_id!r}: "
                    f"max {self._max_calls} tool calls per {self._window:.0f}s."
                ),
            )

    def record(self, task_id: str, count: int = 1) -> None:
        now = time.monotonic()
        self._calls[task_id].extend([now] * count)


# ---------------------------------------------------------------------------
# Auth / security dependency
# ---------------------------------------------------------------------------

_LOCALHOST_ADDRS = {"127.0.0.1", "::1"}


def _build_auth_dependency(token: str | None):  # noqa: ANN202
    """Return a FastAPI dependency that enforces bearer-token or localhost access."""

    async def _verify(request: Request) -> None:
        if token:
            auth_header = request.headers.get("authorization", "")
            if not auth_header.startswith("Bearer ") or auth_header[7:] != token:
                raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")
            return

        # No token configured -- restrict to localhost only.
        client = request.client
        if client is None or client.host not in _LOCALHOST_ADDRS:
            remote = client.host if client else "unknown"
            raise HTTPException(
                status_code=403,
                detail=(
                    f"Tool API access denied from {remote}. "
                    "Set TEAMAI_TOOL_API_TOKEN to allow remote access."
                ),
            )

    return _verify


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_tool_router(settings: Settings, workspace: Path) -> APIRouter:
    """Create a FastAPI APIRouter that exposes workspace tools over HTTP.

    Parameters
    ----------
    settings:
        The application-wide Settings instance (passed through to WorkspaceTools).
    workspace:
        Resolved workspace directory. All tool path arguments are sandboxed
        to this directory by WorkspaceTools._resolve_path.
    """
    token = os.getenv("TEAMAI_TOOL_API_TOKEN", "").strip() or None
    auth = _build_auth_dependency(token)

    router = APIRouter(
        prefix="/v1/tools",
        tags=["tools"],
        dependencies=[Depends(auth)],
    )

    tools = WorkspaceTools(settings)
    limiter = _RateLimiter()

    # ----- GET /v1/tools ---------------------------------------------------

    @router.get("", response_model=ToolListResponse)
    def list_tools() -> ToolListResponse:
        """Return the catalogue of available workspace tools."""
        return ToolListResponse(tools=_TOOL_CATALOGUE)

    # ----- POST /v1/tools/execute ------------------------------------------

    @router.post("/execute", response_model=ToolExecuteResponse)
    def execute_tool(body: ToolExecuteRequest) -> ToolExecuteResponse:
        """Execute a single tool action inside the workspace sandbox."""
        limiter.check(body.task_id)
        limiter.record(body.task_id)

        logger.info(
            "tool_api: execute tool=%s task_id=%s mode=%s",
            body.tool_action.tool,
            body.task_id,
            body.execution_mode,
        )

        results = tools.execute_actions(
            [body.tool_action],
            workspace=workspace,
            execution_mode=body.execution_mode,
        )
        return ToolExecuteResponse(result=results[0])

    # ----- POST /v1/tools/batch --------------------------------------------

    @router.post("/batch", response_model=ToolBatchResponse)
    def execute_batch(body: ToolBatchRequest) -> ToolBatchResponse:
        """Execute a list of tool actions sequentially and return all results."""
        if not body.actions:
            raise HTTPException(status_code=400, detail="actions list must not be empty.")

        limiter.check(body.task_id, count=len(body.actions))
        limiter.record(body.task_id, count=len(body.actions))

        logger.info(
            "tool_api: batch task_id=%s mode=%s actions=%d",
            body.task_id,
            body.execution_mode,
            len(body.actions),
        )

        results = tools.execute_actions(
            body.actions,
            workspace=workspace,
            execution_mode=body.execution_mode,
        )
        return ToolBatchResponse(results=results)

    return router
