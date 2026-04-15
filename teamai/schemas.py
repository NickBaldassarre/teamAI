from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field

WritePolicy = Literal[
    "read_only",
    "propose_only",
    "auto_apply_low_risk",
    "auto_apply_scoped",
    "full_auto",
]

FailureType = Literal[
    "syntax_error",
    "import_error",
    "type_error",
    "assertion_failure",
    "formatting_failure",
    "missing_dependency",
    "flaky_test",
    "timeout",
    "unknown",
]

ComplexityLevel = Literal["low", "medium", "high"]


class ToolAction(BaseModel):
    tool: Literal[
        "list_files",
        "search_text",
        "read_file",
        "run_command",
        "write_file",
        "replace_in_file",
        "apply_patch",
        "run_checks",
        "create_branch",
        "git_add",
        "git_commit",
        "git_restore",
        "spawn_agent",
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


class MutationReceipt(BaseModel):
    mode: Literal["approval", "direct_apply", "blocked"]
    policy: WritePolicy
    phase: Literal["sandbox", "workspace"]
    applied: bool = False
    approval_required: bool = False
    approval_id: str | None = None
    blocked: bool = False
    blocked_reason: str | None = None
    file_count: int = 0
    line_count: int = 0
    changed_paths: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "low"
    risk_reasons: list[str] = Field(default_factory=list)
    verifier_confidence: float | None = None
    tests_passed: bool | None = None


class CheckExecution(BaseModel):
    command: list[str] = Field(default_factory=list)
    scope: str = "repo"
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class FailureDiagnosis(BaseModel):
    failure_type: FailureType
    summary: str
    strategy: str
    retryable: bool = True
    raw_output: str = ""


class RepairAttempt(BaseModel):
    round_number: int
    failure_type: FailureType
    strategy: str
    status: Literal["planned", "applied", "passed", "failed", "escalated"]
    notes: str = ""


class VerifierOutput(BaseModel):
    source: Literal["model", "operational", "handoff"]
    passed: bool
    confidence: float
    summary: str
    next_focus: str | None = None


class HandoffArtifact(BaseModel):
    engine: str
    summary: str
    diff: str
    rationale: str
    risks: list[str] = Field(default_factory=list)
    recommended_checks: list[list[str]] = Field(default_factory=list)
    confidence: float = 0.0


class HandoffRevisionRequest(BaseModel):
    reasons: list[
        Literal[
            "failing_tests",
            "policy_violations",
            "incorrect_scope",
            "verifier_objections",
            "path_scope_violations",
            "rejected_patch",
        ]
    ] = Field(
        default_factory=list
    )
    details: str = ""


class RoutingTraceEntry(BaseModel):
    stage: Literal["initial_route", "inline_escalation", "verified_handoff", "merge", "push"]
    capability: str
    model_id: str
    agent_id: str | None = None
    complexity: ComplexityLevel = "low"
    score: float | None = None
    reasons: list[str] = Field(default_factory=list)
    escalation_reason: str | None = None
    outcome: str | None = None
    retries: int = 0
    verifier_confidence: float | None = None


class RepoIndex(BaseModel):
    workspace: str
    file_tree_summary: list[str] = Field(default_factory=list)
    import_graph: dict[str, list[str]] = Field(default_factory=dict)
    symbol_map: dict[str, list[str]] = Field(default_factory=dict)
    test_map: dict[str, list[str]] = Field(default_factory=dict)
    config_entrypoints: list[str] = Field(default_factory=list)
    total_files: int = 0
    python_files: int = 0
    has_tests: bool = False


class AutonomousRunState(BaseModel):
    task_id: str
    workspace_id: str
    workspace_path: str
    sandbox_path: str | None = None
    policy_mode: WritePolicy
    complexity: ComplexityLevel = "low"
    round_number: int = 0
    files_changed: list[str] = Field(default_factory=list)
    checks_run: list[CheckExecution] = Field(default_factory=list)
    failures_encountered: list[FailureDiagnosis] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    repair_attempts: list[RepairAttempt] = Field(default_factory=list)
    verifier_outputs: list[VerifierOutput] = Field(default_factory=list)
    handoffs: list[HandoffArtifact] = Field(default_factory=list)
    routing_trace: list[RoutingTraceEntry] = Field(default_factory=list)
    final_outcome: str | None = None
    branch_name: str | None = None
    commit_sha: str | None = None
    pushed_remote: str | None = None
    pushed_branch: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)


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
    reasoning_source: Literal["model", "deterministic"] = "model"
    planner_json_repairs: int = 0
    verifier_json_repairs: int = 0


class RunRequest(BaseModel):
    task: str
    workspace_path: str | None = None
    max_rounds: int | None = None
    max_actions_per_round: int | None = None
    max_tokens_per_turn: int | None = None
    temperature: float | None = None
    execution_mode: Literal["read_only", "workspace_write"] = "read_only"
    write_policy: WritePolicy | None = None
    max_repair_attempts: int | None = None
    auto_commit: bool = False
    auto_push: bool = False
    push_remote: str = "origin"
    push_branch_name: str | None = None
    handoff_engine: Literal["codex", "gemini", "local"] | None = None
    handoff_model: str | None = None
    max_handoff_revision_attempts: int | None = None
    continuation_context: dict[str, Any] = Field(default_factory=dict)
    parent_task_id: str | None = None
    spawn_depth: int = 0
    max_spawn_depth: int = 3


class CodexHandoffPayload(BaseModel):
    original_task: str
    core_dependencies: list[str] = Field(default_factory=list)
    distilled_context: dict[str, str] = Field(default_factory=dict)
    recommended_codex_action: str
    structured_context: dict[str, Any] = Field(default_factory=dict)


class RunResult(BaseModel):
    status: Literal["completed", "stopped", "failed"]
    model_id: str
    workspace: str
    execution_mode: Literal["read_only", "workspace_write"]
    write_policy: WritePolicy = "read_only"
    task_route: str = "multi_agent_loop"
    stop_reason: str
    final_answer: str
    transcript: str
    rounds: list[RoundRecord] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    codex_payload: CodexHandoffPayload | None = None
    run_state: AutonomousRunState | None = None
    commit_metadata: dict[str, Any] = Field(default_factory=dict)
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


# ---------------------------------------------------------------------------
# Multi-agent spawning & team orchestration
# ---------------------------------------------------------------------------

class SpawnedTaskRecord(BaseModel):
    """Tracks a single spawned agent within a team execution."""
    spawn_id: str
    parent_id: str | None = None
    task: str
    agent_id: str | None = None
    execution_mode: Literal["read_only", "workspace_write"] = "read_only"
    depends_on: list[str] = Field(default_factory=list)
    status: Literal["pending", "running", "completed", "failed", "blocked"] = "pending"
    job_id: str | None = None
    result_summary: str | None = None
    error: str | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None


class TeamPlan(BaseModel):
    """A decomposed task plan: a DAG of subtasks with dependencies."""
    team_id: str
    goal: str
    tasks: list[SpawnedTaskRecord] = Field(default_factory=list)
    status: Literal["planning", "executing", "synthesizing", "completed", "failed"] = "planning"
    synthesis: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    spawn_depth: int = 0

    def ready_tasks(self) -> list[SpawnedTaskRecord]:
        """Return tasks whose dependencies are all satisfied."""
        completed_ids = {t.spawn_id for t in self.tasks if t.status == "completed"}
        return [
            t for t in self.tasks
            if t.status == "pending"
            and all(dep in completed_ids for dep in t.depends_on)
        ]

    def is_blocked(self) -> bool:
        """True when no tasks can run and some are still pending."""
        failed_ids = {t.spawn_id for t in self.tasks if t.status == "failed"}
        for t in self.tasks:
            if t.status == "pending" and any(dep in failed_ids for dep in t.depends_on):
                return True
        return False

    def all_terminal(self) -> bool:
        return all(t.status in ("completed", "failed") for t in self.tasks)
