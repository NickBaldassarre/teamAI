from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from .approvals import PatchApprovalStore
from .config import Settings
from .handoff import build_handoff_packet
from .schemas import RunRequest, RunResult, ToolExecutionResult
from .supervisor import ClosedLoopSupervisor


class EvalFixtureFile(BaseModel):
    path: str
    content: str


class EvalExpectations(BaseModel):
    allowed_statuses: list[str] = Field(default_factory=list)
    allowed_task_routes: list[str] = Field(default_factory=list)
    allowed_stop_reasons: list[str] = Field(default_factory=list)
    final_answer_contains: list[str] = Field(default_factory=list)
    warning_contains: list[str] = Field(default_factory=list)
    warning_excludes: list[str] = Field(default_factory=list)
    primary_task_contains: list[str] = Field(default_factory=list)
    key_paths_include: list[str] = Field(default_factory=list)
    local_completion: bool | None = None
    handoff: bool | None = None
    handoff_completed: bool | None = None
    approval_required: bool | None = None
    verification_success: bool | None = None


class EvalCase(BaseModel):
    case_id: str
    task: str
    workspace_path: str | None = None
    execution_mode: Literal["read_only", "workspace_write"] = "read_only"
    max_rounds: int | None = None
    max_actions_per_round: int | None = None
    max_tokens_per_turn: int | None = None
    temperature: float | None = None
    setup_files: list[EvalFixtureFile] = Field(default_factory=list)
    expectations: EvalExpectations = Field(default_factory=EvalExpectations)


class EvalSuite(BaseModel):
    name: str = "eval_suite"
    description: str = ""
    workspace_path: str | None = None
    cleanup_approvals: bool = True
    cases: list[EvalCase] = Field(default_factory=list)


class EvalCaseMetrics(BaseModel):
    duration_seconds: float
    rounds_count: int
    tool_actions_total: int
    tool_actions_successful: int
    tool_success_rate: float
    local_completion: bool
    routed_to_handoff: bool
    handoff_completed: bool
    approval_required: bool
    verification_attempted: bool
    verification_success: bool
    approval_ids: list[str] = Field(default_factory=list)


class EvalCaseReport(BaseModel):
    case_id: str
    task: str
    passed: bool
    failures: list[str] = Field(default_factory=list)
    metrics: EvalCaseMetrics
    result_status: str
    task_route: str
    stop_reason: str
    final_answer: str
    warnings: list[str] = Field(default_factory=list)
    primary_task: str | None = None
    key_paths: list[str] = Field(default_factory=list)
    approval_cleanup_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class EvalSuiteMetrics(BaseModel):
    total_cases: int
    passed_cases: int
    failed_cases: int
    pass_rate: float
    local_completion_rate: float
    handoff_rate: float
    handoff_completion_rate: float
    approval_rate: float
    verification_attempt_rate: float
    verification_success_rate: float
    average_rounds: float
    average_tool_success_rate: float
    average_duration_seconds: float
    task_route_counts: dict[str, int] = Field(default_factory=dict)
    stop_reason_counts: dict[str, int] = Field(default_factory=dict)


class EvalSuiteReport(BaseModel):
    name: str
    description: str = ""
    started_at: datetime
    completed_at: datetime
    metrics: EvalSuiteMetrics
    cases: list[EvalCaseReport] = Field(default_factory=list)


EvalRunner = Callable[[RunRequest, Settings], RunResult]


def load_eval_suite(path: Path) -> EvalSuite:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = {"cases": payload}
    return EvalSuite.model_validate(payload)


def run_eval_suite(
    *,
    settings: Settings,
    suite: EvalSuite,
    workspace_override: str | None = None,
    allow_write_cases: bool = False,
    runner: EvalRunner | None = None,
) -> EvalSuiteReport:
    started_at = datetime.now(timezone.utc)
    reports: list[EvalCaseReport] = []

    for case in suite.cases:
        workspace_request = workspace_override or case.workspace_path or suite.workspace_path
        workspace = settings.resolve_workspace(workspace_request)
        snapshots = _apply_setup_files(workspace=workspace, setup_files=case.setup_files)
        cleanup_ids: list[str] = []
        try:
            case_settings = settings
            if case.execution_mode == "workspace_write" and allow_write_cases and not settings.allow_writes:
                case_settings = replace(settings, allow_writes=True)

            request = RunRequest(
                task=case.task,
                workspace_path=str(workspace),
                max_rounds=case.max_rounds,
                max_actions_per_round=case.max_actions_per_round,
                max_tokens_per_turn=case.max_tokens_per_turn,
                temperature=case.temperature,
                execution_mode=case.execution_mode,
            )
            case_runner = runner or _default_runner
            result = case_runner(request, case_settings)
            handoff = build_handoff_packet(task=case.task, result=result)
            metrics = _build_case_metrics(result)
            failures = _evaluate_expectations(case=case, result=result, handoff_primary_task=handoff.primary_task, handoff_key_paths=handoff.key_paths, metrics=metrics)
            if suite.cleanup_approvals and metrics.approval_ids:
                cleanup_ids = _cleanup_approvals(workspace=workspace, approval_ids=metrics.approval_ids)
            reports.append(
                EvalCaseReport(
                    case_id=case.case_id,
                    task=case.task,
                    passed=not failures,
                    failures=failures,
                    metrics=metrics,
                    result_status=result.status,
                    task_route=result.task_route,
                    stop_reason=result.stop_reason,
                    final_answer=result.final_answer,
                    warnings=result.warnings,
                    primary_task=handoff.primary_task,
                    key_paths=handoff.key_paths,
                    approval_cleanup_ids=cleanup_ids,
                )
            )
        except Exception as exc:
            reports.append(
                EvalCaseReport(
                    case_id=case.case_id,
                    task=case.task,
                    passed=False,
                    failures=[f"Case execution failed: {exc}"],
                    metrics=EvalCaseMetrics(
                        duration_seconds=0.0,
                        rounds_count=0,
                        tool_actions_total=0,
                        tool_actions_successful=0,
                        tool_success_rate=0.0,
                        local_completion=False,
                        routed_to_handoff=False,
                        handoff_completed=False,
                        approval_required=False,
                        verification_attempted=False,
                        verification_success=False,
                    ),
                    result_status="failed",
                    task_route="failed",
                    stop_reason="case_execution_failed",
                    final_answer="",
                    error=str(exc),
                )
            )
        finally:
            _restore_setup_files(snapshots)

    completed_at = datetime.now(timezone.utc)
    return EvalSuiteReport(
        name=suite.name,
        description=suite.description,
        started_at=started_at,
        completed_at=completed_at,
        metrics=_build_suite_metrics(reports),
        cases=reports,
    )


def render_eval_markdown(report: EvalSuiteReport) -> str:
    metrics = report.metrics
    lines = [
        f"# Eval Report: {report.name}",
        "",
        f"Cases: {metrics.passed_cases}/{metrics.total_cases} passed",
        "",
        "## Summary",
        "",
        f"- Pass rate: {_format_percent(metrics.pass_rate)}",
        f"- Local completion rate: {_format_percent(metrics.local_completion_rate)}",
        f"- Handoff rate: {_format_percent(metrics.handoff_rate)}",
        f"- Handoff completion rate: {_format_percent(metrics.handoff_completion_rate)}",
        f"- Approval rate: {_format_percent(metrics.approval_rate)}",
        f"- Verification attempt rate: {_format_percent(metrics.verification_attempt_rate)}",
        f"- Verification success rate: {_format_percent(metrics.verification_success_rate)}",
        f"- Average rounds: {metrics.average_rounds:.2f}",
        f"- Average tool success rate: {_format_percent(metrics.average_tool_success_rate)}",
        f"- Average duration: {metrics.average_duration_seconds:.2f}s",
        "",
        "## Cases",
        "",
    ]
    for case in report.cases:
        lines.append(f"- `{case.case_id}`: {'PASS' if case.passed else 'FAIL'} ({case.task_route} / {case.stop_reason})")
        if case.failures:
            for failure in case.failures:
                lines.append(f"  - {failure}")
    return "\n".join(lines)


def _default_runner(request: RunRequest, settings: Settings) -> RunResult:
    return ClosedLoopSupervisor(settings).run(request)


def _build_case_metrics(result: RunResult) -> EvalCaseMetrics:
    tool_results = [tool_result for round_record in result.rounds for tool_result in round_record.tool_results]
    tool_actions_total = len(tool_results)
    tool_actions_successful = sum(1 for tool_result in tool_results if tool_result.success)
    routed_to_handoff = result.task_route == "codex_handoff"
    handoff_completed = result.stop_reason == "codex_handoff_synthesized"
    approval_required = result.stop_reason == "approval_required"
    verification_results = [tool_result for tool_result in tool_results if _is_verification_result(tool_result)]
    verification_attempted = bool(verification_results)
    verification_success = verification_attempted and all(tool_result.success for tool_result in verification_results)
    approval_ids = sorted(
        {
            str(tool_result.metadata.get("approval_id", "")).strip()
            for tool_result in tool_results
            if str(tool_result.metadata.get("approval_id", "")).strip()
        }
    )
    duration_seconds = max(0.0, (result.completed_at - result.started_at).total_seconds())
    return EvalCaseMetrics(
        duration_seconds=duration_seconds,
        rounds_count=len(result.rounds),
        tool_actions_total=tool_actions_total,
        tool_actions_successful=tool_actions_successful,
        tool_success_rate=(tool_actions_successful / tool_actions_total) if tool_actions_total else 1.0,
        local_completion=result.status == "completed" and not routed_to_handoff,
        routed_to_handoff=routed_to_handoff,
        handoff_completed=handoff_completed,
        approval_required=approval_required,
        verification_attempted=verification_attempted,
        verification_success=verification_success,
        approval_ids=approval_ids,
    )


def _evaluate_expectations(
    *,
    case: EvalCase,
    result: RunResult,
    handoff_primary_task: str | None,
    handoff_key_paths: list[str],
    metrics: EvalCaseMetrics,
) -> list[str]:
    expectations = case.expectations
    failures: list[str] = []
    warning_blob = "\n".join(result.warnings).lower()
    primary_task = (handoff_primary_task or "").lower()
    final_answer = result.final_answer.lower()
    key_paths = [path.lower() for path in handoff_key_paths]

    if expectations.allowed_statuses and result.status not in expectations.allowed_statuses:
        failures.append(f"Expected status in {expectations.allowed_statuses}, got `{result.status}`.")
    if expectations.allowed_task_routes and result.task_route not in expectations.allowed_task_routes:
        failures.append(f"Expected task_route in {expectations.allowed_task_routes}, got `{result.task_route}`.")
    if expectations.allowed_stop_reasons and result.stop_reason not in expectations.allowed_stop_reasons:
        failures.append(f"Expected stop_reason in {expectations.allowed_stop_reasons}, got `{result.stop_reason}`.")

    for needle in expectations.final_answer_contains:
        if needle.lower() not in final_answer:
            failures.append(f"Expected final answer to mention `{needle}`.")
    for needle in expectations.warning_contains:
        if needle.lower() not in warning_blob:
            failures.append(f"Expected warnings to mention `{needle}`.")
    for needle in expectations.warning_excludes:
        if needle.lower() in warning_blob:
            failures.append(f"Did not expect warnings to mention `{needle}`.")
    for needle in expectations.primary_task_contains:
        if needle.lower() not in primary_task:
            failures.append(f"Expected primary task to mention `{needle}`.")
    for needle in expectations.key_paths_include:
        if not any(needle.lower() in path for path in key_paths):
            failures.append(f"Expected handoff key paths to include `{needle}`.")

    _check_expected_bool(failures, "local completion", expectations.local_completion, metrics.local_completion)
    _check_expected_bool(failures, "handoff route", expectations.handoff, metrics.routed_to_handoff)
    _check_expected_bool(failures, "handoff completion", expectations.handoff_completed, metrics.handoff_completed)
    _check_expected_bool(failures, "approval_required", expectations.approval_required, metrics.approval_required)
    _check_expected_bool(failures, "verification success", expectations.verification_success, metrics.verification_success)
    return failures


def _check_expected_bool(failures: list[str], label: str, expected: bool | None, actual: bool) -> None:
    if expected is None:
        return
    if expected != actual:
        failures.append(f"Expected {label} to be `{expected}`, got `{actual}`.")


def _build_suite_metrics(reports: list[EvalCaseReport]) -> EvalSuiteMetrics:
    total_cases = len(reports)
    passed_cases = sum(1 for report in reports if report.passed)
    failed_cases = total_cases - passed_cases
    local_completion_count = sum(1 for report in reports if report.metrics.local_completion)
    handoff_count = sum(1 for report in reports if report.metrics.routed_to_handoff)
    handoff_completed_count = sum(1 for report in reports if report.metrics.handoff_completed)
    approval_count = sum(1 for report in reports if report.metrics.approval_required)
    verification_attempt_count = sum(1 for report in reports if report.metrics.verification_attempted)
    verification_success_count = sum(1 for report in reports if report.metrics.verification_success)

    task_route_counts: dict[str, int] = {}
    stop_reason_counts: dict[str, int] = {}
    for report in reports:
        task_route_counts[report.task_route] = task_route_counts.get(report.task_route, 0) + 1
        stop_reason_counts[report.stop_reason] = stop_reason_counts.get(report.stop_reason, 0) + 1

    return EvalSuiteMetrics(
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        pass_rate=(passed_cases / total_cases) if total_cases else 1.0,
        local_completion_rate=(local_completion_count / total_cases) if total_cases else 0.0,
        handoff_rate=(handoff_count / total_cases) if total_cases else 0.0,
        handoff_completion_rate=(handoff_completed_count / handoff_count) if handoff_count else 0.0,
        approval_rate=(approval_count / total_cases) if total_cases else 0.0,
        verification_attempt_rate=(verification_attempt_count / total_cases) if total_cases else 0.0,
        verification_success_rate=(
            verification_success_count / verification_attempt_count if verification_attempt_count else 0.0
        ),
        average_rounds=(
            sum(report.metrics.rounds_count for report in reports) / total_cases if total_cases else 0.0
        ),
        average_tool_success_rate=(
            sum(report.metrics.tool_success_rate for report in reports) / total_cases if total_cases else 0.0
        ),
        average_duration_seconds=(
            sum(report.metrics.duration_seconds for report in reports) / total_cases if total_cases else 0.0
        ),
        task_route_counts=task_route_counts,
        stop_reason_counts=stop_reason_counts,
    )


def _apply_setup_files(*, workspace: Path, setup_files: list[EvalFixtureFile]) -> list[tuple[Path, bool, str]]:
    snapshots: list[tuple[Path, bool, str]] = []
    for fixture in setup_files:
        target = (workspace / fixture.path).resolve()
        if workspace.resolve() not in {target, *target.parents}:
            raise ValueError(f"Eval fixture path escapes workspace root: {target}")
        existed = target.exists()
        previous_text = target.read_text(encoding="utf-8") if existed else ""
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(fixture.content, encoding="utf-8")
        snapshots.append((target, existed, previous_text))
    return snapshots


def _restore_setup_files(snapshots: list[tuple[Path, bool, str]]) -> None:
    for target, existed, previous_text in reversed(snapshots):
        if existed:
            target.write_text(previous_text, encoding="utf-8")
        elif target.exists():
            target.unlink()


def _cleanup_approvals(*, workspace: Path, approval_ids: list[str]) -> list[str]:
    store = PatchApprovalStore()
    cleaned: list[str] = []
    for approval_id in approval_ids:
        try:
            store.reject(
                workspace=workspace,
                approval_id=approval_id,
                reason="Cleaned up automatically by the eval harness.",
            )
        except ValueError:
            continue
        cleaned.append(approval_id)
    return cleaned


def _is_verification_result(tool_result: ToolExecutionResult) -> bool:
    if tool_result.tool != "run_command":
        return False
    command = tool_result.metadata.get("command")
    if not isinstance(command, list):
        return False
    rendered = " ".join(str(part) for part in command).lower()
    return "pytest" in rendered or "unittest" in rendered


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"
