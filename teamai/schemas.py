from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolAction(BaseModel):
    tool: Literal[
        "list_files",
        "search_text",
        "read_file",
        "run_command",
        "write_file",
        "replace_in_file",
    ]
    reason: str = ""
    args: dict[str, Any] = Field(default_factory=dict)


class PlannerTurn(BaseModel):
    summary: str
    should_stop: bool = False
    final_answer: str | None = None
    actions: list[ToolAction] = Field(default_factory=list)


class VerifierVerdict(BaseModel):
    done: bool = False
    confidence: float = 0.0
    summary: str
    next_focus: str | None = None


class ToolExecutionResult(BaseModel):
    tool: str
    success: bool
    output: str = ""
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunEvent(BaseModel):
    sequence: int
    timestamp: datetime
    kind: str
    message: str
    round_number: int | None = None
    stage: str | None = None
    terminal: bool = False
    data: dict[str, Any] = Field(default_factory=dict)


class RoundRecord(BaseModel):
    round_number: int
    strategist: str
    critic: str
    planner: PlannerTurn
    tool_results: list[ToolExecutionResult] = Field(default_factory=list)
    verifier: VerifierVerdict


class RunRequest(BaseModel):
    task: str
    workspace_path: str | None = None
    max_rounds: int | None = None
    max_actions_per_round: int | None = None
    max_tokens_per_turn: int | None = None
    temperature: float | None = None
    execution_mode: Literal["read_only", "workspace_write"] = "read_only"
    continuation_context: dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    status: Literal["completed", "stopped", "failed"]
    model_id: str
    workspace: str
    execution_mode: Literal["read_only", "workspace_write"]
    task_route: str = "multi_agent_loop"
    stop_reason: str
    final_answer: str
    transcript: str
    rounds: list[RoundRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    started_at: datetime
    completed_at: datetime


class HandoffPacket(BaseModel):
    goal: str
    status: Literal["completed", "stopped", "failed"]
    workspace: str
    model_id: str
    task_route: str = "multi_agent_loop"
    stop_reason: str
    summary: str
    primary_task: str | None = None
    next_tasks: list[str] = Field(default_factory=list)
    key_paths: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    suggested_codex_prompt: str


class JobResponse(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: RunResult | None = None
    error: str | None = None
