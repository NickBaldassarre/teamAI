from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from .approvals import PatchApprovalStore
from .bridge import BridgeArtifacts, BridgeLaunchConfig, build_run_command, prepare_bridge_config
from .config import Settings
from .handoff import build_handoff_packet
from .memory import WorkspaceMemoryStore
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


class EvalRuntimeHealth(BaseModel):
    status: Literal["healthy", "unavailable", "unknown"] = "unknown"
    reason: str = "not_checked"
    summary: str = "Runtime health was not checked."
    checked_at: datetime
    warnings: list[str] = Field(default_factory=list)


class EvalCaseReport(BaseModel):
    case_id: str
    task: str
    workspace: str
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
    runner_mode: str = "in_process"
    memory_profile: str = "default"
    guardrail_notes: list[str] = Field(default_factory=list)
    failure_classification: Literal["passed", "infra_runtime", "agent_behavior", "case_timeout", "harness_failure"] = "passed"
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
    actionable_cases: int
    actionable_pass_rate: float
    infra_failure_cases: int
    infra_failure_rate: float
    agent_failure_cases: int
    timeout_failure_cases: int
    harness_failure_cases: int
    failure_classification_counts: dict[str, int] = Field(default_factory=dict)
    task_route_counts: dict[str, int] = Field(default_factory=dict)
    stop_reason_counts: dict[str, int] = Field(default_factory=dict)


class EvalSuiteReport(BaseModel):
    name: str
    description: str = ""
    started_at: datetime
    completed_at: datetime
    runtime_health: EvalRuntimeHealth
    metrics: EvalSuiteMetrics
    cases: list[EvalCaseReport] = Field(default_factory=list)


EvalRunner = Callable[[RunRequest, Settings], RunResult]
EvalSubprocessRunner = Callable[[list[str], dict[str, str], Path, float | None], subprocess.CompletedProcess[str]]
EvalRunnerMode = Literal["in_process", "isolated_subprocess"]
DEFAULT_ISOLATED_CASE_TIMEOUT_SECONDS = 180
DEFAULT_RUNTIME_HEALTH_TIMEOUT_SECONDS = 20


@dataclass(frozen=True)
class EvalCaseExecution:
    result: RunResult | None
    runner_mode: str
    memory_profile: str = "default"
    guardrail_notes: tuple[str, ...] = ()
    timeout_seconds: int | None = None
    error: str | None = None
    result_status: str = "failed"
    task_route: str = "failed"
    stop_reason: str = "case_execution_failed"


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
    runner_mode: EvalRunnerMode = "in_process",
    per_case_timeout_seconds: int | None = None,
    project_root: Path | None = None,
    python_executable: Path | None = None,
    subprocess_runner: EvalSubprocessRunner | None = None,
) -> EvalSuiteReport:
    started_at = datetime.now(timezone.utc)
    reports: list[EvalCaseReport] = []
    effective_project_root = (project_root or Path.cwd()).resolve()
    effective_python = (python_executable or Path(sys.executable)).resolve()
    runtime_health = _run_runtime_health_preflight(
        settings=settings,
        runner_mode=runner_mode,
        project_root=effective_project_root,
        python_executable=effective_python,
        subprocess_runner=subprocess_runner or _default_subprocess_runner,
    )

    for case in suite.cases:
        workspace_request = workspace_override or case.workspace_path or suite.workspace_path
        workspace = settings.resolve_workspace(workspace_request)
        snapshots = _apply_setup_files(workspace=workspace, setup_files=case.setup_files)
        cleanup_ids: list[str] = []
        try:
            case_settings = settings
            inject_write_env = case.execution_mode == "workspace_write" and allow_write_cases and not settings.allow_writes
            if inject_write_env:
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
            execution = _execute_eval_case(
                request=request,
                settings=case_settings,
                runner=runner,
                runner_mode=runner_mode,
                inject_write_env=inject_write_env,
                project_root=effective_project_root,
                python_executable=effective_python,
                per_case_timeout_seconds=per_case_timeout_seconds,
                subprocess_runner=subprocess_runner,
            )
            if execution.result is None:
                reports.append(
                    EvalCaseReport(
                        case_id=case.case_id,
                        task=case.task,
                        workspace=str(workspace),
                        passed=False,
                    failures=[execution.error or "Case execution failed."],
                    metrics=_empty_case_metrics(),
                    result_status=execution.result_status,
                    task_route=execution.task_route,
                    stop_reason=execution.stop_reason,
                        final_answer="",
                        runner_mode=execution.runner_mode,
                        memory_profile=execution.memory_profile,
                        guardrail_notes=list(execution.guardrail_notes),
                        failure_classification=(
                            "case_timeout" if execution.stop_reason == "case_timeout" else "harness_failure"
                        ),
                        error=execution.error,
                    )
                )
                continue
            result = execution.result
            handoff = build_handoff_packet(task=case.task, result=result)
            metrics = _build_case_metrics(result)
            failure_classification = _classify_case_failure(
                result=result,
                error=execution.error,
                runtime_health=runtime_health,
            )
            failures = _build_case_failures(
                case=case,
                result=result,
                handoff_primary_task=handoff.primary_task,
                handoff_key_paths=handoff.key_paths,
                metrics=metrics,
                failure_classification=failure_classification,
                runtime_health=runtime_health,
            )
            if suite.cleanup_approvals and metrics.approval_ids:
                cleanup_ids = _cleanup_approvals(workspace=workspace, approval_ids=metrics.approval_ids)
            reports.append(
                EvalCaseReport(
                    case_id=case.case_id,
                    task=case.task,
                    workspace=str(workspace),
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
                    runner_mode=execution.runner_mode,
                    memory_profile=execution.memory_profile,
                    guardrail_notes=list(execution.guardrail_notes),
                    failure_classification=failure_classification,
                )
            )
        except Exception as exc:
            reports.append(
                EvalCaseReport(
                    case_id=case.case_id,
                    task=case.task,
                    workspace=str(workspace),
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
                    runner_mode=runner_mode,
                    memory_profile="default",
                    failure_classification="harness_failure",
                    error=str(exc),
                )
            )
        finally:
            _restore_setup_files(snapshots)

    completed_at = datetime.now(timezone.utc)
    report = EvalSuiteReport(
        name=suite.name,
        description=suite.description,
        started_at=started_at,
        completed_at=completed_at,
        runtime_health=runtime_health,
        metrics=_build_suite_metrics(reports),
        cases=reports,
    )
    _persist_eval_feedback(suite=suite, report=report)
    return report


def render_eval_markdown(report: EvalSuiteReport) -> str:
    metrics = report.metrics
    lines = [
        f"# Eval Report: {report.name}",
        "",
        f"Cases: {metrics.passed_cases}/{metrics.total_cases} passed",
        "",
        "## Summary",
        "",
        f"- Runtime health: {report.runtime_health.status} ({report.runtime_health.reason})",
        f"- Runtime summary: {report.runtime_health.summary}",
        f"- Pass rate: {_format_percent(metrics.pass_rate)}",
        f"- Actionable pass rate: {_format_percent(metrics.actionable_pass_rate)}",
        f"- Local completion rate: {_format_percent(metrics.local_completion_rate)}",
        f"- Handoff rate: {_format_percent(metrics.handoff_rate)}",
        f"- Handoff completion rate: {_format_percent(metrics.handoff_completion_rate)}",
        f"- Approval rate: {_format_percent(metrics.approval_rate)}",
        f"- Verification attempt rate: {_format_percent(metrics.verification_attempt_rate)}",
        f"- Verification success rate: {_format_percent(metrics.verification_success_rate)}",
        f"- Infra failure rate: {_format_percent(metrics.infra_failure_rate)}",
        f"- Infra failure cases: {metrics.infra_failure_cases}",
        f"- Agent-behavior failure cases: {metrics.agent_failure_cases}",
        f"- Timeout failure cases: {metrics.timeout_failure_cases}",
        f"- Harness failure cases: {metrics.harness_failure_cases}",
        f"- Average rounds: {metrics.average_rounds:.2f}",
        f"- Average tool success rate: {_format_percent(metrics.average_tool_success_rate)}",
        f"- Average duration: {metrics.average_duration_seconds:.2f}s",
        "",
        "## Cases",
        "",
    ]
    for case in report.cases:
        lines.append(
            f"- `{case.case_id}`: {'PASS' if case.passed else 'FAIL'} "
            f"({case.task_route} / {case.stop_reason} / {case.failure_classification})"
        )
        if case.failures:
            for failure in case.failures:
                lines.append(f"  - {failure}")
    return "\n".join(lines)


def _default_runner(request: RunRequest, settings: Settings) -> RunResult:
    return ClosedLoopSupervisor(settings).run(request)


def _run_runtime_health_preflight(
    *,
    settings: Settings,
    runner_mode: EvalRunnerMode,
    project_root: Path,
    python_executable: Path,
    subprocess_runner: EvalSubprocessRunner,
) -> EvalRuntimeHealth:
    checked_at = datetime.now(timezone.utc)
    if runner_mode != "isolated_subprocess":
        return EvalRuntimeHealth(
            status="unknown",
            reason="not_checked_in_process_mode",
            summary="Skipped MLX runtime preflight because the eval suite is running in-process.",
            checked_at=checked_at,
        )

    env = dict(os.environ)
    preflight_script = """
import json

payload = {"status": "healthy", "reason": "mlx_import_ok", "summary": "MLX import preflight passed."}
try:
    import mlx_vlm  # noqa: F401
except Exception as exc:
    payload = {
        "status": "unavailable",
        "reason": "mlx_import_failed",
        "summary": f"MLX import preflight failed: {exc}",
    }

print(json.dumps(payload))
""".strip()
    command = [
        str(python_executable),
        "-c",
        preflight_script,
    ]
    try:
        completed = subprocess_runner(command, env, project_root, DEFAULT_RUNTIME_HEALTH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        return EvalRuntimeHealth(
            status="unknown",
            reason="mlx_import_timeout",
            summary="MLX import preflight timed out before the eval suite started.",
            checked_at=checked_at,
            warnings=["Runtime health preflight timed out; runtime-dependent failures may be environmental."],
        )

    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        return EvalRuntimeHealth(
            status="unknown",
            reason="preflight_subprocess_failed",
            summary="MLX runtime preflight could not complete cleanly.",
            checked_at=checked_at,
            warnings=[_tail_text(completed.stderr or completed.stdout or "Preflight subprocess failed.")],
        )

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return EvalRuntimeHealth(
            status="unknown",
            reason="preflight_invalid_payload",
            summary="MLX runtime preflight returned an unreadable payload.",
            checked_at=checked_at,
            warnings=[_tail_text(stdout or completed.stderr or "Preflight payload was not valid JSON.")],
        )

    status = str(payload.get("status", "unknown")).strip() or "unknown"
    reason = str(payload.get("reason", "unknown")).strip() or "unknown"
    summary = str(payload.get("summary", "Runtime health preflight completed.")).strip() or "Runtime health preflight completed."
    if status not in {"healthy", "unavailable", "unknown"}:
        status = "unknown"
    warnings: list[str] = []
    if status != "healthy":
        warnings.append(summary)
    return EvalRuntimeHealth(
        status=status,  # type: ignore[arg-type]
        reason=reason,
        summary=summary,
        checked_at=checked_at,
        warnings=warnings,
    )


def _execute_eval_case(
    *,
    request: RunRequest,
    settings: Settings,
    runner: EvalRunner | None,
    runner_mode: EvalRunnerMode,
    inject_write_env: bool,
    project_root: Path,
    python_executable: Path,
    per_case_timeout_seconds: int | None,
    subprocess_runner: EvalSubprocessRunner | None,
) -> EvalCaseExecution:
    if runner_mode == "isolated_subprocess":
        if runner is not None:
            raise ValueError("Custom eval runners are only supported with `runner_mode='in_process'`.")
        timeout_seconds = per_case_timeout_seconds or DEFAULT_ISOLATED_CASE_TIMEOUT_SECONDS
        return _run_case_in_subprocess(
            request=request,
            settings=settings,
            inject_write_env=inject_write_env,
            project_root=project_root,
            python_executable=python_executable,
            timeout_seconds=timeout_seconds,
            subprocess_runner=subprocess_runner or _default_subprocess_runner,
        )

    case_runner = runner or _default_runner
    result = case_runner(request, settings)
    return EvalCaseExecution(
        result=result,
        runner_mode="in_process",
        result_status=result.status,
        task_route=result.task_route,
        stop_reason=result.stop_reason,
    )


def _run_case_in_subprocess(
    *,
    request: RunRequest,
    settings: Settings,
    inject_write_env: bool,
    project_root: Path,
    python_executable: Path,
    timeout_seconds: int,
    subprocess_runner: EvalSubprocessRunner,
) -> EvalCaseExecution:
    case_id_slug = _slugify_case_id(request.task)
    with tempfile.TemporaryDirectory(prefix=f"teamai-eval-{case_id_slug}-") as temp_dir:
        temp_path = Path(temp_dir)
        artifacts = BridgeArtifacts(
            handoff_file=temp_path / "handoff.json",
            status_file=temp_path / "status.json",
            log_file=temp_path / "run.log",
            script_file=temp_path / "launch.sh",
        )
        bridge_config = prepare_bridge_config(
            BridgeLaunchConfig(
                task=request.task,
                project_root=project_root,
                python_executable=python_executable,
                workspace=request.workspace_path,
                max_rounds=request.max_rounds,
                max_actions=request.max_actions_per_round,
                max_tokens=request.max_tokens_per_turn,
                temperature=request.temperature,
                execution_mode=request.execution_mode,
                inject_write_env=inject_write_env,
                terminal_app="Terminal",
                artifacts=artifacts,
            )
        )
        result_file = temp_path / "run-result.json"
        command = build_run_command(bridge_config, output_format="full_json", output_file=result_file)
        env = dict(os.environ)
        if bridge_config.inject_write_env and bridge_config.execution_mode == "workspace_write":
            env["TEAMAI_ALLOW_WRITES"] = "true"

        try:
            completed = subprocess_runner(command, env, project_root, timeout_seconds)
        except subprocess.TimeoutExpired:
            return EvalCaseExecution(
                result=None,
                runner_mode="isolated_subprocess",
                memory_profile=bridge_config.memory_profile,
                guardrail_notes=bridge_config.guardrail_notes,
                timeout_seconds=timeout_seconds,
                error=(
                    f"Case timed out after {timeout_seconds} seconds while running in isolated subprocess mode "
                    f"under the `{bridge_config.memory_profile}` memory profile."
                ),
                stop_reason="case_timeout",
            )

        if result_file.exists():
            try:
                result = RunResult.model_validate_json(result_file.read_text(encoding="utf-8"))
            except Exception as exc:
                return EvalCaseExecution(
                    result=None,
                    runner_mode="isolated_subprocess",
                    memory_profile=bridge_config.memory_profile,
                    guardrail_notes=bridge_config.guardrail_notes,
                    timeout_seconds=timeout_seconds,
                    error=(
                        "Case subprocess wrote an unreadable result payload: "
                        f"{exc}. stderr: {_tail_text(completed.stderr)}"
                    ),
                )
            return EvalCaseExecution(
                result=result,
                runner_mode="isolated_subprocess",
                memory_profile=bridge_config.memory_profile,
                guardrail_notes=bridge_config.guardrail_notes,
                timeout_seconds=timeout_seconds,
                result_status=result.status,
                task_route=result.task_route,
                stop_reason=result.stop_reason,
            )

        error_lines = []
        if completed.returncode != 0:
            error_lines.append(f"exit code {completed.returncode}")
        if completed.stderr.strip():
            error_lines.append(f"stderr: {_tail_text(completed.stderr)}")
        if completed.stdout.strip():
            error_lines.append(f"stdout: {_tail_text(completed.stdout)}")
        if not error_lines:
            error_lines.append("no result file was written")
        return EvalCaseExecution(
            result=None,
            runner_mode="isolated_subprocess",
            memory_profile=bridge_config.memory_profile,
            guardrail_notes=bridge_config.guardrail_notes,
            timeout_seconds=timeout_seconds,
            error="Case subprocess failed in isolated mode: " + "; ".join(error_lines) + ".",
        )


def _default_subprocess_runner(
    command: list[str],
    env: dict[str, str],
    cwd: Path,
    timeout_seconds: float | None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def _empty_case_metrics() -> EvalCaseMetrics:
    return EvalCaseMetrics(
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
    )


def _slugify_case_id(value: str) -> str:
    lowered = "".join(ch.lower() if ch.isalnum() else "-" for ch in value)
    collapsed = "-".join(part for part in lowered.split("-") if part)
    return collapsed[:40] or "case"


def _tail_text(text: str, *, max_chars: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return "..." + compact[-max_chars:]


def _classify_case_failure(
    *,
    result: RunResult,
    error: str | None,
    runtime_health: EvalRuntimeHealth,
) -> Literal["passed", "infra_runtime", "agent_behavior", "case_timeout", "harness_failure"]:
    del error
    if result.status == "completed":
        return "passed"
    if _looks_like_runtime_failure(result.stop_reason, result.warnings, result.final_answer):
        return "infra_runtime"
    return "agent_behavior"


def _build_case_failures(
    *,
    case: EvalCase,
    result: RunResult,
    handoff_primary_task: str | None,
    handoff_key_paths: list[str],
    metrics: EvalCaseMetrics,
    failure_classification: Literal["passed", "infra_runtime", "agent_behavior", "case_timeout", "harness_failure"],
    runtime_health: EvalRuntimeHealth,
) -> list[str]:
    if failure_classification == "passed":
        return _evaluate_expectations(
            case=case,
            result=result,
            handoff_primary_task=handoff_primary_task,
            handoff_key_paths=handoff_key_paths,
            metrics=metrics,
        )

    if failure_classification == "infra_runtime":
        detail = (
            f"Case was not scored as an agent-behavior failure because the local runtime failed first "
            f"({result.stop_reason})."
        )
        warnings_blob = " ".join(result.warnings).strip()
        if warnings_blob:
            detail = f"{detail} Runtime detail: {_tail_text(warnings_blob)}"
        if runtime_health.status != "healthy":
            detail = f"{detail} Preflight: {runtime_health.summary}"
        return [detail]

    if failure_classification == "case_timeout":
        return [
            "Case timed out before it produced a result. Treat this as a stability or routing issue, not as a clean agent-quality signal."
        ]

    if failure_classification == "harness_failure":
        return ["Case execution failed inside the eval harness before a usable run result was produced."]

    return _evaluate_expectations(
        case=case,
        result=result,
        handoff_primary_task=handoff_primary_task,
        handoff_key_paths=handoff_key_paths,
        metrics=metrics,
    )


def _looks_like_runtime_failure(stop_reason: str, warnings: list[str], final_answer: str) -> bool:
    if stop_reason == "model_backend_error":
        return True
    combined = "\n".join([*warnings, final_answer]).lower()
    markers = [
        "failed to import mlx runtime",
        "could not import mlx_vlm",
        "metal initialization failed",
        "failed to load model",
        "mlx generation failed",
        "insufficient memory",
        "outofmemory",
        "out of memory",
    ]
    return any(marker in combined for marker in markers)


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
    infra_failure_cases = sum(1 for report in reports if report.failure_classification == "infra_runtime")
    agent_failure_cases = sum(1 for report in reports if report.failure_classification == "agent_behavior")
    timeout_failure_cases = sum(1 for report in reports if report.failure_classification == "case_timeout")
    harness_failure_cases = sum(1 for report in reports if report.failure_classification == "harness_failure")
    actionable_cases = max(total_cases - infra_failure_cases - harness_failure_cases, 0)

    task_route_counts: dict[str, int] = {}
    stop_reason_counts: dict[str, int] = {}
    failure_classification_counts: dict[str, int] = {}
    for report in reports:
        task_route_counts[report.task_route] = task_route_counts.get(report.task_route, 0) + 1
        stop_reason_counts[report.stop_reason] = stop_reason_counts.get(report.stop_reason, 0) + 1
        failure_classification_counts[report.failure_classification] = (
            failure_classification_counts.get(report.failure_classification, 0) + 1
        )

    return EvalSuiteMetrics(
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        pass_rate=(passed_cases / total_cases) if total_cases else 1.0,
        actionable_cases=actionable_cases,
        actionable_pass_rate=(passed_cases / actionable_cases) if actionable_cases else 0.0,
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
        infra_failure_cases=infra_failure_cases,
        infra_failure_rate=(infra_failure_cases / total_cases) if total_cases else 0.0,
        agent_failure_cases=agent_failure_cases,
        timeout_failure_cases=timeout_failure_cases,
        harness_failure_cases=harness_failure_cases,
        failure_classification_counts=failure_classification_counts,
        task_route_counts=task_route_counts,
        stop_reason_counts=stop_reason_counts,
    )


def _persist_eval_feedback(*, suite: EvalSuite, report: EvalSuiteReport) -> None:
    store = WorkspaceMemoryStore()
    workspace_groups: dict[str, list[EvalCaseReport]] = {}
    for case_report in report.cases:
        workspace_groups.setdefault(case_report.workspace, []).append(case_report)

    for workspace_text, workspace_reports in workspace_groups.items():
        workspace_metrics = _build_suite_metrics(workspace_reports)
        store.persist_eval_feedback(
            workspace=Path(workspace_text),
            suite_name=suite.name,
            completed_at=report.completed_at,
            metrics=workspace_metrics.model_dump(mode="python"),
            cases=[case_report.model_dump(mode="python") for case_report in workspace_reports],
            description=suite.description,
            runtime_health=report.runtime_health.model_dump(mode="json"),
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
