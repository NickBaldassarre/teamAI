from __future__ import annotations

import json
import os
import re
import textwrap
import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from uuid import uuid4

from pydantic import ValidationError

from .agent_registry import AgentRegistry, RoutingDecision, TaskSignature
from .autonomy import (
    AutonomousRunStateStore,
    ContextPackager,
    PatchExecutionContext,
    PatchExecutor,
    RepoIndexer,
    SafeCommandExecutor,
    build_commit_message,
    classify_failure,
    collect_workspace_changes,
    compute_complexity,
    derive_write_policy,
    infer_task_scopes,
    new_run_state,
    render_change_diff,
    update_run_state_metrics,
)
from .config import Settings
from .distillation import generate_semantic_skeleton
from .events import build_run_event
from .handoff import build_handoff_packet
from .json_utils import JsonExtractionError, extract_json_object
from .model_backend import MLXModelBackend, ModelBackendError
from .memory import WorkspaceMemorySnapshot, WorkspaceMemoryStore
from .patch_utils import extract_patch_targets
from .prompts import (
    CRITIC_SYSTEM_PROMPT,
    JSON_REPAIR_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT_TEMPLATE,
    PLANNER_JSON_SCHEMA,
    STRATEGIST_SYSTEM_PROMPT,
    VERIFIER_SYSTEM_PROMPT,
    VERIFIER_JSON_SCHEMA,
    build_round_context,
)
from .schemas import (
    AutonomousRunState,
    CheckExecution,
    CodexHandoffPayload,
    FailureDiagnosis,
    HandoffArtifact,
    PlannerTurn,
    RepoIndex,
    RepairAttempt,
    RoundRecord,
    RoutingTraceEntry,
    RunRequest,
    RunEvent,
    RunResult,
    ToolAction,
    ToolExecutionResult,
    VerifierOutput,
    VerifierVerdict,
)
from .sandbox import Sandbox
from .tools import WorkspaceTools


_VALID_ROUTES: frozenset[str] = frozenset({
    "repository_inspection",
    "deterministic_patch",
    "explicit_write_loop",
    "codex_handoff",
    "multi_agent_loop",
})

# Marker prepended to any final answer assembled by deterministic synthesis
# rather than written by the local model. The marker travels with the answer
# wherever it is persisted (transcripts, run history, handoff payloads, eval
# reports) so downstream readers cannot conflate templated text with model
# reasoning.
DETERMINISTIC_SYNTHESIS_LABEL = "[deterministic-synthesis]"

# Placeholder rendered in transcripts when a deterministic route does not
# invoke the model for the strategist/critic step. The on-disk RoundRecord
# stores empty strings; the transcript renderer substitutes this notice so
# readers can immediately tell that no model reasoning was generated.
DETERMINISTIC_PERSONA_TRANSCRIPT_NOTICE = (
    "(deterministic route — no model strategist/critic was generated for this round)"
)


class ClosedLoopSupervisor:
    def __init__(self, settings: Settings, backend: MLXModelBackend | None = None) -> None:
        self._settings = settings
        self._backend = backend or MLXModelBackend(settings)
        self._owns_backend = backend is None
        self._backend_by_model: dict[str, MLXModelBackend] = {settings.model_id: self._backend}
        self._memory = WorkspaceMemoryStore()
        self._tools = WorkspaceTools(settings)
        self._registry = AgentRegistry.load()
        self._repo_indexer = RepoIndexer()
        self._context_packager = ContextPackager()
        self._patch_executor = PatchExecutor()
        self._safe_commands = SafeCommandExecutor(timeout_seconds=settings.command_timeout_seconds)
        self._run_state_store = AutonomousRunStateStore()
        self._last_routing_decision: RoutingDecision | None = None
        self._active_model_id = settings.model_id
        self._last_planner_json_repairs = 0
        self._last_verifier_json_repairs = 0

    @property
    def model_loaded(self) -> bool:
        return self._backend.model_loaded

    def generate_raw(
        self,
        *,
        system: str,
        user: str,
        max_tokens: int = 512,
        temperature: float = 0.3,
    ) -> str:
        """Low-level model call without the council loop.

        Used by ``AgentTeam`` for task decomposition and synthesis —
        situations where the full strategist/critic/planner pipeline is
        unnecessary and a single prompt-response suffices.
        """
        return self._ask_model(
            system_prompt=system,
            user_prompt=user,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def run(
        self,
        request: RunRequest,
        progress_callback: Callable[[str], None] | None = None,
        event_callback: Callable[[RunEvent], None] | None = None,
    ) -> RunResult:
        started_at = datetime.now(timezone.utc)
        workspace = self._settings.resolve_workspace(request.workspace_path)
        max_rounds = request.max_rounds or self._settings.max_rounds
        max_actions = request.max_actions_per_round or self._settings.max_actions_per_round
        max_tokens = request.max_tokens_per_turn or self._settings.max_tokens_per_turn
        temperature = request.temperature if request.temperature is not None else self._settings.temperature
        execution_mode = request.execution_mode
        write_policy = derive_write_policy(
            execution_mode=execution_mode,
            requested_policy=request.write_policy,
        )
        continuation_context = request.continuation_context or {}

        warnings: list[str] = []
        round_records: list[RoundRecord] = []
        final_answer = ""
        stop_reason = "max_rounds_reached"
        status: RunResult["status"] = "stopped"
        event_sequence = 0
        self._last_routing_decision = None
        self._active_model_id = self._settings.model_id
        self._backend = self._backend_by_model.get(self._settings.model_id, self._backend)
        self._last_planner_json_repairs = 0
        self._last_verifier_json_repairs = 0

        if execution_mode == "workspace_write" and write_policy not in {"read_only", "propose_only"}:
            preview_route = self._classify_task_route(
                task=request.task,
                execution_mode=execution_mode,
                workspace=workspace,
                continuation_context=continuation_context,
            )
            if preview_route == "codex_handoff" and not self._can_inline_escalate(
                request=request,
                task_route=preview_route,
            ):
                execution_mode = "read_only"
                write_policy = "read_only"
            else:
                return self._run_autonomous_workspace_write(
                    request=request,
                    workspace=workspace,
                    started_at=started_at,
                    max_rounds=max_rounds,
                    max_actions=max_actions,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    write_policy=write_policy,
                    progress_callback=progress_callback,
                    event_callback=event_callback,
                )

        def emit_progress(message: str) -> None:
            nonlocal event_sequence
            self._emit_progress(progress_callback, message)
            if event_callback is None:
                return
            event_sequence += 1
            event_callback(build_run_event(sequence=event_sequence, message=message))

        if execution_mode == "workspace_write" and not self._settings.allow_writes:
            task_route = "write_disabled_preflight"
            stop_reason = "write_disabled_preflight"
            status = "failed"
            final_answer = (
                "Run refused: `workspace_write` was requested, but `TEAMAI_ALLOW_WRITES` is false in local "
                "configuration. Enable writes first, rerun in `read_only`, or use `bridge-launch --inject-write-env` "
                "for an explicitly approved bridge run."
            )
            warnings.append(final_answer)
            emit_progress(
                f"Starting run in {workspace} "
                f"(mode={execution_mode}, max_rounds={max_rounds}, max_actions={max_actions})"
            )
            emit_progress(f"Task route: {task_route}")
            emit_progress(f"Failed: {stop_reason}")
            completed_at = datetime.now(timezone.utc)
            try:
                self._memory.persist_run(
                    workspace=workspace,
                    task=request.task,
                    status=status,
                    stop_reason=stop_reason,
                    final_answer=final_answer,
                    warnings=warnings,
                    completed_at=completed_at,
                    model_id=self._active_model_id,
                    task_route=task_route,
                    execution_mode=execution_mode,
                    rounds=round_records,
                )
            except Exception as exc:
                warnings.append(f"Failed to persist workspace memory: {exc}")
            return RunResult(
                status=status,
                model_id=self._active_model_id,
                workspace=str(workspace),
                execution_mode=execution_mode,
                write_policy=write_policy,
                task_route=task_route,
                stop_reason=stop_reason,
                final_answer=final_answer,
                transcript=self._render_transcript(round_records, request.task, workspace, warnings),
                rounds=round_records,
                warnings=warnings,
                started_at=started_at,
                completed_at=completed_at,
            )

        task_route = self._classify_task_route(
            task=request.task,
            execution_mode=execution_mode,
            workspace=workspace,
            continuation_context=continuation_context,
        )
        if task_route == "codex_handoff":
            warnings.append(
                "Broad coding task routed to reconnaissance for a Codex handoff instead of local autonomous implementation."
            )
            if execution_mode == "workspace_write":
                warnings.append(
                    "Requested `workspace_write` for a broad coding task; using `read_only` reconnaissance instead."
                )
                execution_mode = "read_only"

        emit_progress(
            f"Starting run in {workspace} "
            f"(mode={execution_mode}, write_policy={write_policy}, max_rounds={max_rounds}, max_actions={max_actions})"
        )
        emit_progress(f"Task route: {task_route}")

        try:
            memory_snapshot = self._memory.load_snapshot(
                workspace,
                task=request.task,
                task_route=task_route,
                continuation_context=continuation_context,
            )
            if continuation_context:
                emit_progress("Continuation: scoped verification before resuming the task")
                probe_round = self._build_continuation_probe_round(
                    workspace=workspace,
                    continuation_context=continuation_context,
                )
                if probe_round is not None:
                    round_records.append(probe_round)
                    continuation_completion = self._maybe_complete_after_continuation_probe(
                        task=request.task,
                        workspace=workspace,
                        continuation_context=continuation_context,
                        probe_round=probe_round,
                    )
                    if continuation_completion is not None:
                        final_answer, stop_reason, status = continuation_completion
                        emit_progress(f"Completed: {stop_reason}")
            if task_route == "deterministic_patch":
                (
                    deterministic_rounds,
                    final_answer,
                    stop_reason,
                    status,
                ) = self._run_deterministic_patch_route(
                    task=request.task,
                    workspace=workspace,
                    execution_mode=execution_mode,
                )
                round_records.extend(deterministic_rounds)
                if final_answer:
                    if status == "failed":
                        emit_progress(f"Failed: {stop_reason}")
                    elif status == "completed":
                        emit_progress(f"Completed: {stop_reason}")
                    else:
                        emit_progress(f"Stopped: {stop_reason}")

            if (
                task_route == "repository_inspection"
                and not final_answer
                and self._can_bootstrap_repository_inspection(workspace)
            ):
                (
                    inspection_rounds,
                    final_answer,
                    stop_reason,
                    status,
                ) = self._run_repository_inspection_route(
                    task=request.task,
                    workspace=workspace,
                    max_rounds=max_rounds,
                    max_actions=max_actions,
                    warnings=warnings,
                    emit_progress=emit_progress,
                )
                round_records.extend(inspection_rounds)
                if final_answer:
                    if status == "failed":
                        emit_progress(f"Failed: {stop_reason}")
                    elif status == "completed":
                        emit_progress(f"Completed: {stop_reason}")
                    else:
                        emit_progress(f"Stopped: {stop_reason}")

            for round_number in range(1, max_rounds + 1):
                if final_answer:
                    break
                emit_progress(f"Round {round_number}/{max_rounds}: building context")
                context = self._build_context(
                    task=request.task,
                    workspace=workspace,
                    round_number=round_number,
                    task_route=task_route,
                    memory_snapshot=memory_snapshot,
                    continuation_context=continuation_context,
                    previous_rounds=round_records,
                )

                emit_progress(f"Round {round_number}/{max_rounds}: strategist")
                strategist = self._ask_model(
                    system_prompt=STRATEGIST_SYSTEM_PROMPT,
                    user_prompt=context,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                critic_context = (
                    f"{context}\n\nStrategist output:\n{strategist}\n\n"
                    "Respond with critique and missing considerations."
                )
                emit_progress(f"Round {round_number}/{max_rounds}: critic")
                critic = self._ask_model(
                    system_prompt=CRITIC_SYSTEM_PROMPT,
                    user_prompt=critic_context,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                planner_context = (
                    f"{context}\n\nStrategist output:\n{strategist}\n\n"
                    f"Critic output:\n{critic}\n\n"
                    "Create the next action plan."
                )
                emit_progress(f"Round {round_number}/{max_rounds}: planner")
                planner = self._plan(
                    task=request.task,
                    user_prompt=planner_context,
                    workspace=workspace,
                    previous_rounds=round_records,
                    execution_mode=execution_mode,
                    write_policy=write_policy,
                    task_route=task_route,
                    max_actions=max_actions,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    warnings=warnings,
                )
                planner_json_repairs = self._last_planner_json_repairs

                if planner.actions:
                    emit_progress(
                        f"Round {round_number}/{max_rounds}: executing {len(planner.actions[:max_actions])} tool action(s)"
                    )
                else:
                    emit_progress(f"Round {round_number}/{max_rounds}: no tool actions")
                tool_results = self._tools.execute_actions(
                    planner.actions[:max_actions],
                    workspace=workspace,
                    execution_mode=execution_mode,
                    write_policy=write_policy,
                    approval_context={
                        "task": request.task,
                        "execution_mode": execution_mode,
                    },
                )

                verifier_context = self._build_verifier_context(
                    task=request.task,
                    workspace=workspace,
                    strategist=strategist,
                    critic=critic,
                    planner=planner,
                    tool_results=tool_results,
                )
                emit_progress(f"Round {round_number}/{max_rounds}: verifier")
                verifier = self._verify(
                    verifier_context,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    warnings=warnings,
                )
                verifier_json_repairs = self._last_verifier_json_repairs

                pending_approvals = self._collect_pending_approvals(tool_results, workspace)
                if pending_approvals:
                    round_records.append(
                        RoundRecord(
                            round_number=round_number,
                            strategist=strategist,
                            critic=critic,
                            planner=planner,
                            tool_results=tool_results,
                            verifier=verifier,
                            planner_json_repairs=planner_json_repairs,
                            verifier_json_repairs=verifier_json_repairs,
                        )
                    )
                    final_answer = self._build_approval_required_answer(pending_approvals)
                    stop_reason = "approval_required"
                    status = "stopped"
                    emit_progress(f"Stopped: {stop_reason}")
                    break

                round_records.append(
                    RoundRecord(
                        round_number=round_number,
                        strategist=strategist,
                        critic=critic,
                        planner=planner,
                        tool_results=tool_results,
                        verifier=verifier,
                        planner_json_repairs=planner_json_repairs,
                        verifier_json_repairs=verifier_json_repairs,
                    )
                )

                if (
                    execution_mode == "workspace_write"
                    and self._is_explicit_write_task(request.task)
                    and planner.should_stop
                    and planner.final_answer
                ):
                    warnings.append(
                        "Planner declared the workspace_write task complete without a concrete patch application step; continuing."
                    )
                elif planner.should_stop and planner.final_answer:
                    final_answer = planner.final_answer
                    stop_reason = "planner_declared_complete"
                    status = "completed"
                    emit_progress(f"Completed: {stop_reason}")
                    break

                if verifier.done:
                    final_answer = planner.final_answer or verifier.summary
                    stop_reason = "verifier_declared_complete"
                    status = "completed"
                    emit_progress(f"Completed: {stop_reason}")
                    break

                reroute_reason = self._local_drift_reroute_reason(
                    task=request.task,
                    workspace=workspace,
                    task_route=task_route,
                    round_records=round_records,
                )
                if reroute_reason:
                    warnings.append(reroute_reason)
                    task_route = "codex_handoff"
                    emit_progress(f"Task route: {task_route}")
                    final_answer = self._build_local_drift_handoff_answer(
                        task=request.task,
                        rounds=round_records,
                        workspace=workspace,
                        reroute_reason=reroute_reason,
                    )
                    stop_reason = "local_drift_rerouted"
                    status = "completed"
                    emit_progress(f"Completed: {stop_reason}")
                    break

                synthesized_handoff = self._maybe_synthesize_codex_handoff_answer(
                    task=request.task,
                    rounds=round_records,
                    workspace=workspace,
                    task_route=task_route,
                )
                if synthesized_handoff:
                    final_answer = synthesized_handoff
                    stop_reason = "codex_handoff_synthesized"
                    status = "completed"
                    emit_progress(f"Completed: {stop_reason}")
                    break

                synthesized_answer = self._maybe_synthesize_repository_answer(
                    task=request.task,
                    rounds=round_records,
                    workspace=workspace,
                    allow_partial=self._should_allow_early_partial_repository_synthesis(
                        task=request.task,
                        rounds=round_records,
                        max_rounds=max_rounds,
                    ),
                )
                if synthesized_answer:
                    final_answer = synthesized_answer
                    stop_reason = "inspection_synthesized"
                    status = "completed"
                    emit_progress(f"Completed: {stop_reason}")
                    break

            if not final_answer:
                synthesized_handoff = self._maybe_synthesize_codex_handoff_answer(
                    task=request.task,
                    rounds=round_records,
                    workspace=workspace,
                    task_route=task_route,
                )
                if synthesized_handoff:
                    final_answer = synthesized_handoff
                    stop_reason = "codex_handoff_synthesized"
                    status = "completed"
                    emit_progress(f"Completed: {stop_reason}")
                else:
                    synthesized_answer = self._maybe_synthesize_repository_answer(
                        task=request.task,
                        rounds=round_records,
                        workspace=workspace,
                    )
                    if synthesized_answer:
                        final_answer = synthesized_answer
                        stop_reason = "inspection_synthesized"
                        status = "completed"
                        emit_progress(f"Completed: {stop_reason}")
                    else:
                        partial_synthesized_answer = self._maybe_synthesize_repository_answer(
                            task=request.task,
                            rounds=round_records,
                            workspace=workspace,
                            allow_partial=True,
                        )
                        if partial_synthesized_answer:
                            warnings.append("Inspection run hit max rounds; used partial synthesis from gathered evidence.")
                            final_answer = partial_synthesized_answer
                            stop_reason = "inspection_synthesized"
                            status = "completed"
                            emit_progress(f"Completed: {stop_reason}")
                        else:
                            final_answer = self._build_fallback_answer(round_records, request.task)
                            emit_progress(f"Stopped: {stop_reason}")
        except ModelBackendError as exc:
            status = "failed"
            stop_reason = "model_backend_error"
            warnings.append(str(exc))
            final_answer = "The local model backend failed before the loop could finish."
            emit_progress(f"Failed: {stop_reason}")

        if self._rounds_are_fully_deterministic(round_records):
            final_answer = self._label_deterministic_synthesis(final_answer)

        completed_at = datetime.now(timezone.utc)
        try:
            self._memory.persist_run(
                workspace=workspace,
                task=request.task,
                status=status,
                stop_reason=stop_reason,
                final_answer=final_answer,
                warnings=warnings,
                completed_at=completed_at,
                model_id=self._active_model_id,
                task_route=task_route,
                execution_mode=execution_mode,
                rounds=round_records,
            )
        except Exception as exc:
            warnings.append(f"Failed to persist workspace memory: {exc}")

        self._log_telemetry(round_records, task_route)

        provisional_result = RunResult(
            status=status,
            model_id=self._active_model_id,
            workspace=str(workspace),
            execution_mode=execution_mode,
            write_policy=write_policy,
            task_route=task_route,
            stop_reason=stop_reason,
            final_answer=final_answer,
            transcript=self._render_transcript(round_records, request.task, workspace, warnings),
            rounds=round_records,
            warnings=warnings,
            started_at=started_at,
            completed_at=completed_at,
        )
        codex_payload = self._maybe_generate_codex_payload(
            task=request.task,
            result=provisional_result,
            workspace=workspace,
            rounds=round_records,
            warnings=warnings,
        )
        if codex_payload is None:
            if provisional_result.warnings != warnings:
                return provisional_result.model_copy(update={"warnings": warnings})
            return provisional_result

        return provisional_result.model_copy(
            update={
                "warnings": warnings,
                "codex_payload": codex_payload,
            }
        )

    @staticmethod
    def _emit_progress(
        callback: Callable[[str], None] | None,
        message: str,
    ) -> None:
        if callback is not None:
            callback(message)

    def _run_autonomous_workspace_write(
        self,
        *,
        request: RunRequest,
        workspace: Path,
        started_at: datetime,
        max_rounds: int,
        max_actions: int,
        max_tokens: int,
        temperature: float,
        write_policy: str,
        progress_callback: Callable[[str], None] | None,
        event_callback: Callable[[RunEvent], None] | None,
    ) -> RunResult:
        warnings: list[str] = []
        round_records: list[RoundRecord] = []
        final_answer = ""
        stop_reason = "max_rounds_reached"
        status: RunResult["status"] = "stopped"
        event_sequence = 0

        def emit_progress(message: str) -> None:
            nonlocal event_sequence
            self._emit_progress(progress_callback, message)
            if event_callback is None:
                return
            event_sequence += 1
            event_callback(build_run_event(sequence=event_sequence, message=message))

        if not self._settings.allow_writes:
            final_answer = (
                "Run refused: autonomous write policy was requested, but TEAMAI_ALLOW_WRITES is false in local configuration."
            )
            warnings.append(final_answer)
            completed_at = datetime.now(timezone.utc)
            return RunResult(
                status="failed",
                model_id=self._active_model_id,
                workspace=str(workspace),
                execution_mode=request.execution_mode,
                write_policy=write_policy,  # type: ignore[arg-type]
                task_route="write_disabled_preflight",
                stop_reason="write_disabled_preflight",
                final_answer=final_answer,
                transcript=self._render_transcript(round_records, request.task, workspace, warnings),
                rounds=round_records,
                warnings=warnings,
                started_at=started_at,
                completed_at=completed_at,
            )

        repo_index = self._repo_indexer.build(workspace)
        scopes = infer_task_scopes(task=request.task, workspace=workspace, repo_index=repo_index)
        complexity = compute_complexity(
            task=request.task,
            repo_index=repo_index,
            expected_files_touched=max(len(scopes), 1),
        )
        run_state = new_run_state(workspace=workspace, policy=write_policy, complexity=complexity)
        task_route = self._classify_task_route(
            task=request.task,
            execution_mode=request.execution_mode,
            workspace=workspace,
            continuation_context=request.continuation_context,
        )
        emit_progress(
            f"Starting sandboxed autonomous run in {workspace} "
            f"(mode={request.execution_mode}, write_policy={write_policy}, max_rounds={max_rounds}, max_actions={max_actions})"
        )
        emit_progress(f"Task route: {task_route}")

        with Sandbox(workspace, preserve_git=True) as sandbox:
            sandbox_workspace = sandbox.path
            run_state = run_state.model_copy(update={"sandbox_path": str(sandbox_workspace)})
            memory_snapshot = self._memory.load_snapshot(
                workspace,
                task=request.task,
                task_route=task_route,
                continuation_context=request.continuation_context,
            )
            sandbox_repo_index = self._repo_indexer.build(sandbox_workspace)
            allowed_scopes = infer_task_scopes(task=request.task, workspace=sandbox_workspace, repo_index=sandbox_repo_index)
            self._append_routing_trace(
                run_state,
                stage="initial_route",
                capability=task_route,
                model_id=self._active_model_id,
                agent_id=getattr(self._last_routing_decision.agent, "id", None) if self._last_routing_decision else None,
                score=self._last_routing_decision.score if self._last_routing_decision else None,
                reasons=self._last_routing_decision.reasons if self._last_routing_decision else (),
                complexity=run_state.complexity,
                outcome="selected",
            )
            json_repair_rounds = 0
            active_agent = self._active_agent_entry()
            if (
                getattr(active_agent, "is_local", False)
                and run_state.complexity == "high"
                and int(getattr(active_agent, "max_context_tokens", 0) or 0) < 4096
            ):
                escalated_route = self._try_escalate_to_stronger_local(
                    task_route=task_route,
                    run_state=run_state,
                    reason="repository_complexity_above_local_threshold",
                    emit_progress=emit_progress,
                    warnings=warnings,
                )
                if escalated_route is not None:
                    task_route = escalated_route

            if task_route == "codex_handoff" and not final_answer:
                escalated_route = self._try_escalate_to_stronger_local(
                    task_route=task_route,
                    run_state=run_state,
                    reason="broad_task_requires_inline_escalation",
                    emit_progress=emit_progress,
                    warnings=warnings,
                )
                if escalated_route is not None:
                    task_route = escalated_route
                else:
                    handoff_result = self._execute_inline_verified_handoff(
                        task=request.task,
                        request=request,
                        sandbox_workspace=sandbox_workspace,
                        repo_index=sandbox_repo_index,
                        task_scopes=allowed_scopes,
                        round_records=round_records,
                        run_state=run_state,
                        reason="broad_task_requires_inline_verified_handoff",
                        emit_progress=emit_progress,
                        warnings=warnings,
                    )
                    if handoff_result.get("accepted", False):
                        final_answer = str(
                            handoff_result.get("summary")
                            or "Inline verified handoff produced a patch that passed checks."
                        )
                        stop_reason = "inline_verified_handoff_verified"
                        status = "completed"
                    else:
                        stop_reason = str(handoff_result.get("stop_reason", "verified_handoff_rejected"))
                        status = "stopped"
                        final_answer = str(
                            handoff_result.get("final_answer")
                            or "Inline verified handoff could not produce an acceptable patch."
                        )

            if not final_answer:
                for round_number in range(1, max_rounds + 1):
                    run_state.round_number = round_number
                    emit_progress(f"Round {round_number}/{max_rounds}: building context")
                    latest_failure = run_state.failures_encountered[-1] if run_state.failures_encountered else None
                    current_diff = self._current_diff_text(
                        workspace=workspace,
                        sandbox_workspace=sandbox_workspace,
                    )
                    context = self._build_context(
                        task=request.task,
                        workspace=sandbox_workspace,
                        round_number=round_number,
                        task_route=task_route,
                        memory_snapshot=memory_snapshot,
                        continuation_context=request.continuation_context,
                        previous_rounds=round_records,
                        repo_index=sandbox_repo_index,
                        task_scopes=allowed_scopes,
                        changed_paths=tuple(run_state.files_changed),
                        failure_output=latest_failure.raw_output if latest_failure is not None else "",
                        prior_failed_repairs=tuple(
                            f"{attempt.status}: {attempt.strategy}"
                            for attempt in run_state.repair_attempts[-4:]
                        ),
                        current_diff=current_diff,
                    )

                    strategist = ""
                    critic = ""
                    planner: PlannerTurn
                    if round_number == 1 and task_route == "deterministic_patch":
                        compiled = self._compile_small_write_action_from_task(task=request.task, workspace=sandbox_workspace)
                        if compiled is None:
                            planner = PlannerTurn(
                                summary="Autonomous route could not compile a deterministic patch.",
                                should_stop=False,
                                final_answer=None,
                                actions=[],
                            )
                        else:
                            planner = PlannerTurn(
                                summary="Deterministic autonomous route compiled the requested patch directly in sandbox.",
                                should_stop=False,
                                final_answer=None,
                                actions=[compiled],
                            )
                    else:
                        emit_progress(f"Round {round_number}/{max_rounds}: strategist")
                        strategist = self._ask_model(
                            system_prompt=STRATEGIST_SYSTEM_PROMPT,
                            user_prompt=context,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        critic_context = (
                            f"{context}\n\nStrategist output:\n{strategist}\n\n"
                            "Respond with critique and missing considerations."
                        )
                        emit_progress(f"Round {round_number}/{max_rounds}: critic")
                        critic = self._ask_model(
                            system_prompt=CRITIC_SYSTEM_PROMPT,
                            user_prompt=critic_context,
                            max_tokens=max_tokens,
                            temperature=temperature,
                        )
                        planner_context = (
                            f"{context}\n\nStrategist output:\n{strategist}\n\n"
                            f"Critic output:\n{critic}\n\n"
                            "Create the next action plan."
                        )
                        emit_progress(f"Round {round_number}/{max_rounds}: planner")
                        planner = self._plan(
                            task=request.task,
                            user_prompt=planner_context,
                            workspace=sandbox_workspace,
                            previous_rounds=round_records,
                            execution_mode=request.execution_mode,
                            write_policy=write_policy,
                            task_route=task_route,
                            max_actions=max_actions,
                            max_tokens=max_tokens,
                            temperature=temperature,
                            warnings=warnings,
                        )

                    if self._last_planner_json_repairs > 0:
                        json_repair_rounds += 1
                    else:
                        json_repair_rounds = 0

                    emit_progress(
                        f"Round {round_number}/{max_rounds}: executing {len(planner.actions[:max_actions])} tool action(s)"
                    )
                    tool_results = self._tools.execute_actions(
                        planner.actions[:max_actions],
                        workspace=sandbox_workspace,
                        execution_mode=request.execution_mode,
                        approval_context={
                            "task": request.task,
                            "execution_mode": request.execution_mode,
                        },
                        write_policy=write_policy,
                        patch_context=PatchExecutionContext(
                            policy=write_policy,  # type: ignore[arg-type]
                            phase="sandbox",
                            allowed_path_scopes=allowed_scopes,
                        ),
                    )

                    changed_paths = self._collect_directly_applied_paths(tool_results)
                    check_records: list[CheckExecution] = []
                    if changed_paths:
                        run_state.files_changed = list(dict.fromkeys([*run_state.files_changed, *changed_paths]))
                        check_result = self._tools.execute_actions(
                            [
                                ToolAction(
                                    tool="run_checks",
                                    reason="Validate the edited files before merging them back to the workspace.",
                                    args={"paths": changed_paths},
                                )
                            ],
                            workspace=sandbox_workspace,
                            execution_mode="read_only",
                        )[0]
                        tool_results.append(check_result)
                        raw_checks = check_result.metadata.get("checks", [])
                        if isinstance(raw_checks, list):
                            for item in raw_checks:
                                try:
                                    check_records.append(CheckExecution.model_validate(item))
                                except ValidationError:
                                    continue
                        run_state.checks_run.extend(check_records)
                        diagnosis = classify_failure(check_records)
                        if not check_result.success:
                            if diagnosis is None:
                                diagnosis = FailureDiagnosis(
                                    failure_type="unknown",
                                    summary="Sandbox checks failed, but the failure could not be classified cleanly.",
                                    strategy="Inspect the failing check output, repair the smallest broken surface, and rerun the narrowest check first.",
                                    raw_output=check_result.output,
                                )
                            run_state.failures_encountered.append(diagnosis)
                            run_state.hypotheses.append(diagnosis.strategy)
                            run_state.repair_attempts.append(
                                RepairAttempt(
                                    round_number=round_number,
                                    failure_type=diagnosis.failure_type,
                                    strategy=diagnosis.strategy,
                                    status="failed",
                                    notes=diagnosis.summary,
                                )
                            )
                            verifier = VerifierVerdict(
                                done=False,
                                confidence=0.25,
                                summary=diagnosis.summary,
                                next_focus=diagnosis.strategy,
                            )
                            round_records.append(
                                RoundRecord(
                                    round_number=round_number,
                                    strategist=strategist,
                                    critic=critic,
                                    planner=planner,
                                    tool_results=tool_results,
                                    verifier=verifier,
                                    reasoning_source="deterministic" if task_route == "deterministic_patch" and round_number == 1 else "model",
                                )
                            )
                            max_repairs = request.max_repair_attempts or 2
                            if (
                                len(run_state.failures_encountered) >= max_repairs
                                or diagnosis.failure_type in {"missing_dependency", "unknown"}
                            ):
                                escalated_route = self._try_escalate_to_stronger_local(
                                    task_route=task_route,
                                    run_state=run_state,
                                    reason="repair_budget_exhausted",
                                    emit_progress=emit_progress,
                                    warnings=warnings,
                                )
                                if escalated_route is not None:
                                    task_route = escalated_route
                                    continue
                                handoff_result = self._execute_inline_verified_handoff(
                                    task=request.task,
                                    request=request,
                                    sandbox_workspace=sandbox_workspace,
                                    repo_index=sandbox_repo_index,
                                    task_scopes=allowed_scopes,
                                    round_records=round_records,
                                    run_state=run_state,
                                    reason="repair_budget_exhausted",
                                    emit_progress=emit_progress,
                                    warnings=warnings,
                                )
                                if handoff_result.get("accepted", False):
                                    final_answer = str(
                                        handoff_result.get("summary")
                                        or "Inline verified handoff recovered the autonomous run."
                                    )
                                    stop_reason = "inline_verified_handoff_verified"
                                    status = "completed"
                                else:
                                    warnings.append(diagnosis.summary)
                                    handoff_stop = str(handoff_result.get("stop_reason", "repair_budget_exhausted"))
                                    stop_reason = (
                                        "repair_budget_exhausted"
                                        if handoff_stop in {"verified_handoff_unavailable", "verified_handoff_failed"}
                                        else handoff_stop
                                    )
                                    status = "stopped"
                                    final_answer = str(
                                        handoff_result.get("final_answer")
                                        or (
                                            f"Autonomous repair loop escalated after {len(run_state.failures_encountered)} "
                                            f"failed verification attempt(s): {diagnosis.summary}"
                                        )
                                    )
                                break
                            continue

                    if (
                        task_route == "deterministic_patch"
                        and changed_paths
                        and all(check.returncode == 0 for check in check_records)
                    ):
                        verifier = VerifierVerdict(
                            done=True,
                            confidence=0.95,
                            summary="Scoped sandbox checks passed after the deterministic patch.",
                            next_focus="none",
                        )
                        run_state.verifier_outputs.append(
                            VerifierOutput(
                                source="operational",
                                passed=True,
                                confidence=0.95,
                                summary=verifier.summary,
                            )
                        )
                        round_records.append(
                            RoundRecord(
                                round_number=round_number,
                                strategist=strategist,
                                critic=critic,
                                planner=planner,
                                tool_results=tool_results,
                                verifier=verifier,
                                reasoning_source="deterministic",
                            )
                        )
                        final_answer = planner.final_answer or verifier.summary
                        stop_reason = "autonomous_checks_passed"
                        status = "completed"
                        break

                    verifier_context = self._build_verifier_context(
                        task=request.task,
                        workspace=sandbox_workspace,
                        strategist=strategist,
                        critic=critic,
                        planner=planner,
                        tool_results=tool_results,
                        repo_index=sandbox_repo_index,
                        task_scopes=allowed_scopes,
                        changed_paths=tuple(changed_paths),
                        failure_output=latest_failure.raw_output if latest_failure is not None else "",
                        prior_failed_repairs=tuple(
                            f"{attempt.status}: {attempt.strategy}"
                            for attempt in run_state.repair_attempts[-4:]
                        ),
                        current_diff=current_diff,
                    )
                    emit_progress(f"Round {round_number}/{max_rounds}: verifier")
                    verifier = self._verify(
                        verifier_context,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        warnings=warnings,
                    )
                    if self._last_verifier_json_repairs > 0:
                        json_repair_rounds += 1
                    else:
                        json_repair_rounds = 0 if self._last_planner_json_repairs == 0 else json_repair_rounds
                    operational_confidence = 0.85 if changed_paths and (not check_records or all(check.returncode == 0 for check in check_records)) else 0.0
                    if changed_paths and operational_confidence:
                        run_state.verifier_outputs.append(
                            VerifierOutput(
                                source="operational",
                                passed=True,
                                confidence=operational_confidence,
                                summary="Scoped sandbox checks passed after the latest patch.",
                            )
                        )
                    run_state.verifier_outputs.append(
                        VerifierOutput(
                            source="model",
                            passed=verifier.done,
                            confidence=verifier.confidence,
                            summary=verifier.summary,
                            next_focus=verifier.next_focus,
                        )
                    )
                    round_records.append(
                        RoundRecord(
                            round_number=round_number,
                            strategist=strategist,
                            critic=critic,
                            planner=planner,
                            tool_results=tool_results,
                            verifier=verifier,
                            reasoning_source="deterministic" if task_route == "deterministic_patch" and round_number == 1 else "model",
                        )
                    )

                    if not verifier.done and (
                        json_repair_rounds >= 2
                        or (
                            verifier.confidence < 0.4
                            and len(run_state.failures_encountered) >= max(1, request.max_repair_attempts or 2)
                        )
                    ):
                        escalation_reason = (
                            "planner_json_malformed_repeatedly" if json_repair_rounds >= 2 else "verifier_confidence_below_threshold"
                        )
                        escalated_route = self._try_escalate_to_stronger_local(
                            task_route=task_route,
                            run_state=run_state,
                            reason=escalation_reason,
                            emit_progress=emit_progress,
                            warnings=warnings,
                        )
                        if escalated_route is not None:
                            task_route = escalated_route
                            continue
                        handoff_result = self._execute_inline_verified_handoff(
                            task=request.task,
                            request=request,
                            sandbox_workspace=sandbox_workspace,
                            repo_index=sandbox_repo_index,
                            task_scopes=allowed_scopes,
                            round_records=round_records,
                            run_state=run_state,
                            reason=escalation_reason,
                            emit_progress=emit_progress,
                            warnings=warnings,
                        )
                        if handoff_result.get("accepted", False):
                            final_answer = str(
                                handoff_result.get("summary")
                                or "Inline verified handoff completed the task."
                            )
                            stop_reason = "inline_verified_handoff_verified"
                            status = "completed"
                        else:
                            stop_reason = str(
                                handoff_result.get("stop_reason", "inline_escalation_failed")
                            )
                            status = "stopped"
                            final_answer = str(
                                handoff_result.get("final_answer")
                                or "Inline escalation could not recover the run."
                            )
                        break

                    if changed_paths and all(check.returncode == 0 for check in check_records):
                        if planner.should_stop and planner.final_answer:
                            final_answer = planner.final_answer
                        elif verifier.done or self._is_explicit_write_task(request.task):
                            final_answer = planner.final_answer or verifier.summary or "Autonomous sandbox checks passed."
                        if final_answer:
                            stop_reason = "autonomous_checks_passed"
                            status = "completed"
                            break

                    if planner.should_stop and planner.final_answer and not changed_paths:
                        final_answer = planner.final_answer
                        stop_reason = "planner_declared_complete"
                        status = "completed"
                        break

            if status == "completed" and run_state.files_changed:
                run_state = update_run_state_metrics(run_state)
                if final_answer and self._rounds_are_fully_deterministic(round_records):
                    final_answer = self._label_deterministic_synthesis(final_answer)
                merged = self._merge_autonomous_changes(
                    workspace=workspace,
                    sandbox_workspace=sandbox_workspace,
                    task=request.task,
                    run_state=run_state,
                    write_policy=write_policy,
                    allowed_scopes=scopes,
                    auto_commit=request.auto_commit or request.auto_push,
                    auto_push=request.auto_push,
                    push_remote=request.push_remote,
                    push_branch_name=request.push_branch_name,
                )
                warnings.extend(merged.get("warnings", []))
                commit_metadata = merged.get("commit_metadata", {})
                if not merged.get("applied", False):
                    status = "stopped"
                    stop_reason = merged.get("stop_reason", "approval_required")
                    final_answer = str(merged.get("final_answer", final_answer))
                    if final_answer and self._rounds_are_fully_deterministic(round_records):
                        final_answer = self._label_deterministic_synthesis(final_answer)
                else:
                    if not final_answer:
                        final_answer = "Applied the sandboxed changes to the workspace."
                    run_state = run_state.model_copy(update={"final_outcome": stop_reason})
                    completed_at = datetime.now(timezone.utc)
                    persisted_state = self._run_state_store.persist(workspace=workspace, state=run_state)
                    warnings.append(f"Autonomous run state: {persisted_state}")
                    self._memory.persist_run(
                        workspace=workspace,
                        task=request.task,
                        status=status,
                        stop_reason=stop_reason,
                        final_answer=final_answer,
                        warnings=warnings,
                        completed_at=completed_at,
                        model_id=self._active_model_id,
                        task_route=task_route,
                        execution_mode=request.execution_mode,
                        rounds=round_records,
                        run_state=run_state,
                    )
                    return RunResult(
                        status=status,
                        model_id=self._active_model_id,
                        workspace=str(workspace),
                        execution_mode=request.execution_mode,
                        write_policy=write_policy,  # type: ignore[arg-type]
                        task_route=task_route,
                        stop_reason=stop_reason,
                        final_answer=final_answer,
                        transcript=self._render_transcript(round_records, request.task, workspace, warnings),
                        rounds=round_records,
                        warnings=warnings,
                        run_state=run_state,
                        commit_metadata=commit_metadata if isinstance(commit_metadata, dict) else {},
                        started_at=started_at,
                        completed_at=completed_at,
                    )

        run_state = update_run_state_metrics(run_state)
        run_state = run_state.model_copy(update={"final_outcome": stop_reason})
        completed_at = datetime.now(timezone.utc)
        persisted_state = self._run_state_store.persist(workspace=workspace, state=run_state)
        warnings.append(f"Autonomous run state: {persisted_state}")
        self._memory.persist_run(
            workspace=workspace,
            task=request.task,
            status=status,
            stop_reason=stop_reason,
            final_answer=final_answer or self._build_fallback_answer(round_records, request.task),
            warnings=warnings,
            completed_at=completed_at,
            model_id=self._active_model_id,
            task_route=task_route,
            execution_mode=request.execution_mode,
            rounds=round_records,
            run_state=run_state,
        )
        return RunResult(
            status=status,
            model_id=self._active_model_id,
            workspace=str(workspace),
            execution_mode=request.execution_mode,
            write_policy=write_policy,  # type: ignore[arg-type]
            task_route=task_route,
            stop_reason=stop_reason,
            final_answer=final_answer or self._build_fallback_answer(round_records, request.task),
            transcript=self._render_transcript(round_records, request.task, workspace, warnings),
            rounds=round_records,
            warnings=warnings,
            run_state=run_state,
            started_at=started_at,
            completed_at=completed_at,
        )

    @staticmethod
    def _collect_directly_applied_paths(tool_results: list[ToolExecutionResult]) -> list[str]:
        changed_paths: list[str] = []
        for result in tool_results:
            receipt = result.metadata.get("mutation_receipt")
            if not isinstance(receipt, dict):
                continue
            if not receipt.get("applied", False):
                continue
            for path in receipt.get("changed_paths", []) or []:
                path_str = str(path)
                if path_str and path_str not in changed_paths:
                    changed_paths.append(path_str)
        return changed_paths

    @staticmethod
    def _latest_check_batch_passed(checks: list[CheckExecution]) -> bool:
        if not checks:
            return False
        saw_green = False
        for check in reversed(checks):
            if check.returncode != 0:
                return saw_green
            saw_green = True
        return saw_green

    def _merge_autonomous_changes(
        self,
        *,
        workspace: Path,
        sandbox_workspace: Path,
        task: str,
        run_state: AutonomousRunState,
        write_policy: str,
        allowed_scopes: tuple[str, ...],
        auto_commit: bool,
        auto_push: bool,
        push_remote: str,
        push_branch_name: str | None,
    ) -> dict[str, object]:
        changes = collect_workspace_changes(source_root=workspace, modified_root=sandbox_workspace)
        if not changes:
            return {
                "applied": False,
                "stop_reason": "no_changes_detected",
                "final_answer": "Autonomous run completed its sandbox loop, but no file changes were produced.",
                "warnings": [],
            }
        verifier_confidence = max(
            [output.confidence for output in run_state.verifier_outputs if output.passed] or [0.0]
        )
        latest_checks_green = self._latest_check_batch_passed(run_state.checks_run)
        receipt = self._patch_executor.execute_bundle(
            workspace=workspace,
            changes=changes,
            reason=f"Autonomous run {run_state.task_id} for: {task}",
            source_tool="autonomous_supervisor",
            context=PatchExecutionContext(
                policy=write_policy,  # type: ignore[arg-type]
                phase="workspace",
                allowed_path_scopes=allowed_scopes,
                tests_passed=latest_checks_green,
                verifier_confidence=verifier_confidence,
            ),
            continuation=None,
        )
        warnings: list[str] = []
        commit_metadata: dict[str, object] = {}
        final_answer = ""
        stop_reason = "autonomous_merge_complete"
        requested_branch_name = (push_branch_name or "").strip()
        blocked_push_branch = requested_branch_name in {"main", "master", "default", "trunk"}
        if receipt.approval_required and receipt.approval_id:
            final_answer = self._build_approval_required_answer(
                [
                    {
                        "approval_id": receipt.approval_id,
                        "path": receipt.changed_paths[0] if receipt.changed_paths else "(unknown path)",
                        "tool": "autonomous_merge",
                    }
                ]
            )
            stop_reason = "approval_required"
            return {
                "applied": False,
                "stop_reason": stop_reason,
                "final_answer": final_answer,
                "warnings": warnings,
            }
        if not receipt.applied:
            return {
                "applied": False,
                "stop_reason": "policy_violation",
                "final_answer": receipt.blocked_reason or "Autonomous merge was blocked by policy.",
                "warnings": warnings,
            }
        self._append_routing_trace(
            run_state,
            stage="merge",
            capability="autonomous_merge",
            model_id=self._active_model_id,
            complexity=run_state.complexity,
            outcome="applied",
        )

        if auto_commit and (workspace / ".git").exists():
            branch_name = (
                requested_branch_name
                if requested_branch_name and not blocked_push_branch
                else self._default_push_branch_name(run_state.task_id)
            )
            if blocked_push_branch:
                warnings.append(
                    f"Requested push branch `{requested_branch_name}` is protected; using local feature branch `{branch_name}` and skipping push."
                )
            branch_result = self._safe_commands.create_branch(workspace=workspace, branch_name=branch_name)
            if branch_result.returncode != 0:
                warnings.append(branch_result.stderr.strip() or "Autonomous branch creation failed.")
            else:
                add_result = self._safe_commands.git_add(workspace=workspace, paths=receipt.changed_paths)
                if add_result.returncode != 0:
                    warnings.append(add_result.stderr.strip() or "Autonomous git add failed.")
                else:
                    commit_message = build_commit_message(task=task, state=run_state)
                    commit_result = self._safe_commands.git_commit(workspace=workspace, message=commit_message)
                    if commit_result.returncode != 0:
                        warnings.append(commit_result.stderr.strip() or "Autonomous git commit failed.")
                    else:
                        head = self._safe_commands.current_head(workspace=workspace)
                        run_state.branch_name = branch_name
                        run_state.commit_sha = head
                        commit_metadata = {
                            "branch": branch_name,
                            "commit_sha": head,
                            "message": commit_message,
                            "review_ready": True,
                        }
                        if auto_push:
                            if not latest_checks_green:
                                warnings.append("Autonomous push skipped because verification is not green.")
                            elif blocked_push_branch:
                                warnings.append(
                                    f"Autonomous push blocked for protected branch `{requested_branch_name}`."
                                )
                            elif not self._settings.allow_git_push:
                                warnings.append(
                                    "Autonomous push requested, but TEAMAI_ALLOW_GIT_PUSH is false; leaving a review-ready local branch."
                                )
                            else:
                                push_result = self._safe_commands.git_push(
                                    workspace=workspace,
                                    remote=push_remote,
                                    branch_name=branch_name,
                                )
                                if push_result.returncode != 0:
                                    warnings.append(push_result.stderr.strip() or "Autonomous git push failed.")
                                else:
                                    run_state.pushed_remote = push_remote
                                    run_state.pushed_branch = branch_name
                                    commit_metadata.update(
                                        {
                                            "pushed": True,
                                            "remote": push_remote,
                                        }
                                    )
                                    self._append_routing_trace(
                                        run_state,
                                        stage="push",
                                        capability="git_push",
                                        model_id=self._active_model_id,
                                        complexity=run_state.complexity,
                                        outcome="pushed",
                                    )
        elif auto_push:
            warnings.append("Autonomous push requested, but no local git commit was created.")
        return {
            "applied": True,
            "stop_reason": stop_reason,
            "final_answer": final_answer,
            "warnings": warnings,
            "commit_metadata": commit_metadata,
        }

    def _maybe_generate_codex_payload(
        self,
        *,
        task: str,
        result: RunResult,
        workspace: Path,
        rounds: list[RoundRecord],
        warnings: list[str],
    ) -> CodexHandoffPayload | None:
        if result.task_route != "codex_handoff":
            return None
        if result.status == "failed":
            return None

        handoff = build_handoff_packet(task=task, result=result)
        prioritized_files = self._rank_codex_handoff_paths(
            task=task,
            paths=[
                *handoff.key_paths,
                *self._priority_candidates(
                    rounds,
                    workspace,
                    task=task,
                    task_route="codex_handoff",
                ),
            ],
        )
        if not prioritized_files:
            return None

        recommended_action = handoff.primary_task or next(
            (candidate for candidate in handoff.next_tasks if candidate.strip()),
            "",
        )
        try:
            return generate_semantic_skeleton(
                task=task,
                workspace=workspace,
                prioritized_files=prioritized_files,
                backend=self._backend,
                recommended_codex_action=recommended_action,
                max_tokens=min(128, self._settings.max_tokens_per_turn),
                warnings=warnings,
            )
        except Exception as exc:
            warnings.append(f"Semantic skeleton generation failed: {exc}")
            return None

    def _ask_model(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        response = self._backend.generate_messages(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=False,
        )
        return self._sanitize_model_text(response.text)

    def _plan(
        self,
        task: str,
        user_prompt: str,
        *,
        workspace: Path,
        previous_rounds: list[RoundRecord],
        execution_mode: str,
        write_policy: str | None = None,
        max_actions: int,
        max_tokens: int,
        temperature: float,
        warnings: list[str],
        task_route: str = "multi_agent_loop",
    ) -> PlannerTurn:
        effective_route = (
            task_route
            if task_route != "multi_agent_loop"
            else self._classify_task_route(task=task, execution_mode=execution_mode, workspace=workspace)
        )
        json_repairs = 0
        planner_prompt = PLANNER_SYSTEM_PROMPT_TEMPLATE.format(
            tool_manifest=self._tools.describe_tools(execution_mode=execution_mode, write_policy=write_policy),
            execution_mode=execution_mode,
            max_actions=max_actions,
        )
        raw = self._ask_model(
            system_prompt=planner_prompt,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=min(temperature, 0.1),
        )
        try:
            payload = extract_json_object(raw)
            planner = PlannerTurn.model_validate(payload)
        except (JsonExtractionError, ValidationError) as exc:
            repaired_raw = self._repair_json_response(
                raw_response=raw,
                schema=PLANNER_JSON_SCHEMA,
                max_tokens=max_tokens,
            )
            try:
                payload = extract_json_object(repaired_raw)
                planner = PlannerTurn.model_validate(payload)
                json_repairs = 1
                warnings.append(f"Planner JSON required repair: {exc}")
            except (JsonExtractionError, ValidationError) as repair_exc:
                json_repairs = 1
                warnings.append(
                    f"Planner JSON could not be parsed cleanly: {exc}; repair failed: {repair_exc}"
                )
                planner = self._heuristic_plan_from_context(
                    task=task,
                    raw_response=raw,
                    user_prompt=user_prompt,
                    workspace=workspace,
                    previous_rounds=previous_rounds,
                    max_actions=max_actions,
                    execution_mode=execution_mode,
                    task_route=effective_route,
                )
                if planner.actions:
                    warnings.append("Planner JSON failed; used heuristic fallback action synthesis.")
                else:
                    planner = PlannerTurn(
                        summary="Planner output was not valid JSON; continuing without actions.",
                        should_stop=False,
                        final_answer=None,
                        actions=[],
                    )

        if len(planner.actions) > max_actions:
            warnings.append(
                f"Planner requested {len(planner.actions)} actions; truncating to {max_actions}."
            )
            planner = planner.model_copy(update={"actions": planner.actions[:max_actions]})

        filtered_actions = self._remove_repeated_actions(
            planner.actions,
            workspace=workspace,
            previous_rounds=previous_rounds,
            warnings=warnings,
        )
        filtered_actions = self._remove_invalid_actions(
            filtered_actions,
            workspace=workspace,
            warnings=warnings,
        )
        if len(filtered_actions) != len(planner.actions):
            planner = planner.model_copy(update={"actions": filtered_actions})

        explicit_write_task = execution_mode == "workspace_write" and self._is_explicit_write_task(task)

        if explicit_write_task and not any(
            self._action_matches_explicit_write_task(
                action,
                task=task,
                workspace=workspace,
            )
            for action in planner.actions
        ):
            fallback = self._heuristic_plan_from_context(
                task=task,
                raw_response=raw,
                user_prompt=user_prompt,
                workspace=workspace,
                previous_rounds=previous_rounds,
                max_actions=max_actions,
                execution_mode=execution_mode,
                task_route=effective_route,
            )
            if fallback.actions:
                warnings.append("Explicit write task required a concrete patch action; used heuristic write fallback.")
                planner = fallback

        if not planner.should_stop and not planner.actions:
            fallback = self._heuristic_plan_from_context(
                task=task,
                raw_response=raw,
                user_prompt=user_prompt,
                workspace=workspace,
                previous_rounds=previous_rounds,
                max_actions=max_actions,
                execution_mode=execution_mode,
                task_route=effective_route,
            )
            if fallback.actions:
                warnings.append("Planner had no novel actions; used heuristic fallback action synthesis.")
                planner = fallback

        if effective_route in {"repository_inspection", "codex_handoff"} and not planner.should_stop:
            supplemented_actions = self._supplement_inspection_actions(
                planner.actions,
                task=task,
                workspace=workspace,
                previous_rounds=previous_rounds,
                max_actions=max_actions,
                task_route=effective_route,
            )
            if len(supplemented_actions) != len(planner.actions):
                warnings.append(
                    f"Inspection task detected; expanded plan to {len(supplemented_actions)} read-only action(s)."
                )
                planner = planner.model_copy(update={"actions": supplemented_actions})

        self._last_planner_json_repairs = json_repairs
        return planner

    def _verify(
        self,
        user_prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        warnings: list[str],
    ) -> VerifierVerdict:
        json_repairs = 0
        raw = self._ask_model(
            system_prompt=VERIFIER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=max_tokens,
            temperature=min(temperature, 0.1),
        )
        try:
            payload = extract_json_object(raw)
            verdict = VerifierVerdict.model_validate(payload)
        except (JsonExtractionError, ValidationError) as exc:
            repaired_raw = self._repair_json_response(
                raw_response=raw,
                schema=VERIFIER_JSON_SCHEMA,
                max_tokens=max_tokens,
            )
            try:
                payload = extract_json_object(repaired_raw)
                verdict = VerifierVerdict.model_validate(payload)
                json_repairs = 1
                warnings.append(f"Verifier JSON required repair: {exc}")
            except (JsonExtractionError, ValidationError) as repair_exc:
                json_repairs = 1
                warnings.append(
                    f"Verifier JSON could not be parsed cleanly: {exc}; repair failed: {repair_exc}"
                )
                verdict = VerifierVerdict(
                    done=False,
                    confidence=0.0,
                    summary="Verifier output was invalid JSON; assuming the task is incomplete.",
                    next_focus="Recover structured planning and keep gathering evidence.",
                )
        verdict.confidence = min(max(verdict.confidence, 0.0), 1.0)
        self._last_verifier_json_repairs = json_repairs
        return verdict

    def _repair_json_response(
        self,
        *,
        raw_response: str,
        schema: str,
        max_tokens: int,
    ) -> str:
        repair_prompt = (
            "Repair the following model output into one valid JSON object.\n\n"
            f"Original output:\n{raw_response}"
        )
        repaired = self._ask_model(
            system_prompt=JSON_REPAIR_SYSTEM_PROMPT.format(schema=schema),
            user_prompt=repair_prompt,
            max_tokens=max(256, max_tokens),
            temperature=0.0,
        )
        return repaired.strip()

    def _build_context(
        self,
        *,
        task: str,
        workspace: Path,
        round_number: int,
        task_route: str,
        memory_snapshot: WorkspaceMemorySnapshot,
        continuation_context: dict[str, object],
        previous_rounds: list[RoundRecord],
        repo_index: RepoIndex | None = None,
        task_scopes: tuple[str, ...] = (),
        changed_paths: tuple[str, ...] = (),
        failure_output: str = "",
        prior_failed_repairs: tuple[str, ...] = (),
        current_diff: str = "",
    ) -> str:
        previous = []
        for record in previous_rounds[-2:]:
            previous.append(
                f"Round {record.round_number}: "
                f"planner={record.planner.summary}; verifier={record.verifier.summary}"
            )
        previous_rounds_text = "\n".join(previous) if previous else "No prior rounds."

        recent_actions = self._render_recent_actions(previous_rounds, workspace)
        suggested_paths = self._render_suggested_paths(
            previous_rounds,
            workspace,
            task=task,
            task_route=task_route,
        )

        latest_observations = "No tool observations yet."
        if previous_rounds:
            latest = previous_rounds[-1].tool_results
            latest_observations = self._render_tool_observations(latest) or latest_observations

        bundle = self._context_packager.build(
            task=task,
            workspace=workspace,
            repo_index=repo_index,
            task_scopes=task_scopes,
            observed_paths=self._observed_paths_from_rounds(previous_rounds, workspace),
            changed_paths=changed_paths,
            failure_output=failure_output,
            prior_failed_repairs=prior_failed_repairs,
        )
        if current_diff:
            bundle = self._context_packager.with_diff(bundle, diff_text=current_diff)

        return build_round_context(
            task=task,
            workspace=str(workspace),
            round_number=round_number,
            continuation_context=self._render_continuation_context(continuation_context),
            persistent_memory=memory_snapshot.memory_text,
            persisted_runs=memory_snapshot.recent_runs_text,
            improvement_notes=memory_snapshot.improvement_notes_text,
            global_memory=memory_snapshot.global_memory_text,
            previous_rounds=previous_rounds_text,
            latest_observations=latest_observations,
            recent_actions=recent_actions,
            suggested_paths=suggested_paths,
            context_package=self._context_packager.render(bundle),
        )

    def _build_verifier_context(
        self,
        *,
        task: str,
        workspace: Path,
        strategist: str,
        critic: str,
        planner: PlannerTurn,
        tool_results: list[ToolExecutionResult],
        repo_index: RepoIndex | None = None,
        task_scopes: tuple[str, ...] = (),
        changed_paths: tuple[str, ...] = (),
        failure_output: str = "",
        prior_failed_repairs: tuple[str, ...] = (),
        current_diff: str = "",
    ) -> str:
        bundle = self._context_packager.build(
            task=task,
            workspace=workspace,
            repo_index=repo_index,
            task_scopes=task_scopes,
            observed_paths=self._tool_result_paths(tool_results, workspace),
            changed_paths=changed_paths,
            failure_output=failure_output,
            prior_failed_repairs=prior_failed_repairs,
        )
        if current_diff:
            bundle = self._context_packager.with_diff(bundle, diff_text=current_diff)
        return (
            f"Task:\n{task}\n\n"
            f"Workspace:\n{workspace}\n\n"
            f"Strategist:\n{strategist}\n\n"
            f"Critic:\n{critic}\n\n"
            f"Planner summary:\n{planner.summary}\n\n"
            f"Candidate final answer:\n{planner.final_answer or '(none)'}\n\n"
            f"Deterministic context package:\n{self._context_packager.render(bundle)}\n\n"
            f"Tool results:\n{self._render_tool_observations(tool_results)}"
        )

    def _observed_paths_from_rounds(self, rounds: list[RoundRecord], workspace: Path) -> tuple[str, ...]:
        observed: list[str] = []
        for record in rounds:
            for result in record.tool_results:
                raw_path = str(result.metadata.get("path", "")).strip()
                if not raw_path:
                    continue
                normalized = self._normalize_path_arg(raw_path, workspace)
                if normalized and normalized not in observed:
                    observed.append(normalized)
        return tuple(observed)

    def _tool_result_paths(self, tool_results: list[ToolExecutionResult], workspace: Path) -> tuple[str, ...]:
        observed: list[str] = []
        for result in tool_results:
            raw_path = str(result.metadata.get("path", "")).strip()
            if not raw_path:
                continue
            normalized = self._normalize_path_arg(raw_path, workspace)
            if normalized and normalized not in observed:
                observed.append(normalized)
        return tuple(observed)

    def _build_continuation_probe_round(
        self,
        *,
        workspace: Path,
        continuation_context: dict[str, object],
    ) -> RoundRecord | None:
        actions = self._build_continuation_probe_actions(continuation_context)
        if not actions:
            return None

        tool_results = self._tools.execute_actions(
            actions,
            workspace=workspace,
            execution_mode="read_only",
        )
        failures = [result for result in tool_results if not result.success]
        verifier_summary = "Scoped continuation verification collected focused evidence for the resumed task."
        next_focus = "Continue from the applied patch without recreating it."
        if failures:
            verifier_summary = "Scoped continuation verification found an issue that should be addressed before more edits."
            next_focus = "Review the failing scoped verification result first, then continue the remaining task."

        # Deterministic continuation probe: the strategist/critic step is
        # intentionally not delegated to the model. Persona fields are blank
        # and `reasoning_source="deterministic"` labels the round so the
        # transcript and downstream artifacts cannot conflate this with model
        # reasoning.
        return RoundRecord(
            round_number=0,
            strategist="",
            critic="",
            planner=PlannerTurn(
                summary="Ran deterministic post-approval verification before continuing.",
                should_stop=False,
                final_answer=None,
                actions=actions,
            ),
            tool_results=tool_results,
            verifier=VerifierVerdict(
                done=False,
                confidence=0.6 if not failures else 0.3,
                summary=verifier_summary,
                next_focus=next_focus,
            ),
            reasoning_source="deterministic",
        )

    def _maybe_complete_after_continuation_probe(
        self,
        *,
        task: str,
        workspace: Path,
        continuation_context: dict[str, object],
        probe_round: RoundRecord,
    ) -> tuple[str, str, RunResult["status"]] | None:
        if any(not result.success for result in probe_round.tool_results):
            return None

        verification_results = [result for result in probe_round.tool_results if result.tool == "run_command"]
        if not verification_results or not all(result.success for result in verification_results):
            return None

        changed_paths = [
            self._normalize_path_arg(path, workspace)
            for path in continuation_context.get("changed_paths", [])
            if str(path).strip()
        ]
        if not changed_paths:
            return None

        original_task = str(continuation_context.get("original_task", "")).strip() or task
        explicit_targets = self._extract_file_targets(original_task, workspace)
        if not explicit_targets:
            return None
        if not all(target in changed_paths for target in explicit_targets):
            return None

        verified_commands = []
        for command in continuation_context.get("suggested_commands", []):
            if isinstance(command, list) and command:
                verified_commands.append(" ".join(str(part) for part in command))

        changed_label = ", ".join(changed_paths)
        command_label = verified_commands[0] if verified_commands else "the directly related verification command"
        final_answer = (
            f"Verified the approved patch for {changed_label}. "
            f"Read the changed files and confirmed `{command_label}` succeeded. "
            "No remaining work was detected for the approved task."
        )
        return final_answer, "verifier_declared_complete", "completed"

    @staticmethod
    def _build_continuation_probe_actions(continuation_context: dict[str, object]) -> list[ToolAction]:
        actions: list[ToolAction] = []
        seen_paths: set[str] = set()
        raw_paths = continuation_context.get("suggested_read_paths")
        if isinstance(raw_paths, list):
            for value in raw_paths[:2]:
                path = str(value).strip()
                if not path or path in seen_paths:
                    continue
                seen_paths.add(path)
                actions.append(
                    ToolAction(
                        tool="read_file",
                        reason="Verify the applied patch landed before continuing the task.",
                        args={"path": path, "start_line": 1, "end_line": 200},
                    )
                )

        raw_commands = continuation_context.get("suggested_commands")
        if isinstance(raw_commands, list):
            for value in raw_commands[:1]:
                if not isinstance(value, list) or not value:
                    continue
                actions.append(
                    ToolAction(
                        tool="run_command",
                        reason="Run the most directly related verification command before continuing.",
                        args={"command": [str(part) for part in value], "cwd": "."},
                    )
                )

        return actions

    @staticmethod
    def _render_continuation_context(continuation_context: dict[str, object]) -> str:
        if not continuation_context:
            return "No continuation context."

        lines: list[str] = []
        approval_id = str(continuation_context.get("approval_id", "")).strip()
        if approval_id:
            lines.append(f"Approval ID: {approval_id}")
        path = str(continuation_context.get("path", "")).strip()
        if path:
            lines.append(f"Changed path: {path}")
        source_tool = str(continuation_context.get("source_tool", "")).strip()
        if source_tool:
            lines.append(f"Source tool: {source_tool}")
        verification_focus = str(continuation_context.get("verification_focus", "")).strip()
        if verification_focus:
            lines.append(f"Verification focus: {verification_focus}")

        commands = continuation_context.get("suggested_commands")
        rendered_commands: list[str] = []
        if isinstance(commands, list):
            for command in commands[:2]:
                if isinstance(command, list) and command:
                    rendered_commands.append(" ".join(str(part) for part in command))
        if rendered_commands:
            lines.append("Suggested commands:")
            lines.extend(f"- {command}" for command in rendered_commands)

        return "\n".join(lines) if lines else "No continuation context."

    @staticmethod
    def _render_tool_observations(tool_results: list[ToolExecutionResult]) -> str:
        rendered = []
        for result in tool_results:
            status = "ok" if result.success else "error"
            body = result.output or result.error or "(no output)"
            approval_status = str(result.metadata.get("approval_status", "")).strip()
            approval_id = str(result.metadata.get("approval_id", "")).strip()
            if approval_status == "pending" and approval_id:
                body = (
                    f"{body}\n"
                    f"Approval required: pending patch {approval_id}. No file changes were applied yet."
                )
            rendered.append(f"[{result.tool} | {status}]\n{body}")
        return "\n\n".join(rendered)

    @staticmethod
    def _collect_pending_approvals(
        tool_results: list[ToolExecutionResult],
        workspace: Path,
    ) -> list[dict[str, str]]:
        pending: list[dict[str, str]] = []
        for result in tool_results:
            if str(result.metadata.get("approval_status", "")).strip() != "pending":
                continue
            approval_id = str(result.metadata.get("approval_id", "")).strip()
            raw_path = str(result.metadata.get("path", "")).strip()
            relative_path = raw_path
            if raw_path:
                try:
                    relative_path = str(Path(raw_path).resolve().relative_to(workspace.resolve()))
                except Exception:
                    relative_path = raw_path
            pending.append(
                {
                    "approval_id": approval_id,
                    "path": relative_path or "(unknown path)",
                    "tool": result.tool,
                }
            )
        return pending

    @staticmethod
    def _build_approval_required_answer(pending_approvals: list[dict[str, str]]) -> str:
        lines = [
            "Patch approval required before any proposed file changes can be applied.",
            "",
            "Pending approvals:",
        ]
        for item in pending_approvals:
            approval_id = item.get("approval_id", "")
            path = item.get("path", "(unknown path)")
            lines.append(f"- {approval_id} | {path}")
        lines.extend(
            [
                "",
                "No file changes were applied yet.",
                "Review a patch with `teamai approvals show <approval_id>` and apply it with `teamai approvals apply <approval_id>`.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _build_fallback_answer(rounds: list[RoundRecord], task: str) -> str:
        if not rounds:
            return f"No rounds completed for task: {task}"
        last_round = rounds[-1]
        return (
            f"Stopped before a verified completion. "
            f"Latest planner summary: {last_round.planner.summary}\n\n"
            f"Latest verifier summary: {last_round.verifier.summary}"
        )

    @staticmethod
    def _rounds_are_fully_deterministic(rounds: list[RoundRecord]) -> bool:
        return bool(rounds) and all(record.reasoning_source == "deterministic" for record in rounds)

    @staticmethod
    def _label_deterministic_synthesis(text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return text
        if stripped.startswith(DETERMINISTIC_SYNTHESIS_LABEL):
            return stripped
        return f"{DETERMINISTIC_SYNTHESIS_LABEL}\n{stripped}"

    @staticmethod
    def _render_round_persona_text(text: str, *, reasoning_source: str) -> str:
        stripped = text.strip()
        if stripped or reasoning_source != "deterministic":
            return text
        return DETERMINISTIC_PERSONA_TRANSCRIPT_NOTICE

    def _run_deterministic_patch_route(
        self,
        *,
        task: str,
        workspace: Path,
        execution_mode: str,
    ) -> tuple[list[RoundRecord], str, str, RunResult["status"]]:
        action = self._compile_small_write_action_from_task(task=task, workspace=workspace)
        if action is None:
            return ([], "", "max_rounds_reached", "stopped")

        # Deterministic route: the model is intentionally not invoked for the
        # strategist/critic step. The persona fields are left empty and the
        # round is labeled `reasoning_source="deterministic"` so downstream
        # consumers cannot mistake template text for model reasoning.
        planner = PlannerTurn(
            summary="Deterministic task routing compiled the requested patch action without invoking the local model.",
            should_stop=False,
            final_answer=None,
            actions=[action],
        )
        tool_results = self._tools.execute_actions(
            [action],
            workspace=workspace,
            execution_mode=execution_mode,
            approval_context={
                "task": task,
                "execution_mode": execution_mode,
            },
        )
        pending_approvals = self._collect_pending_approvals(tool_results, workspace)
        if pending_approvals:
            verifier = VerifierVerdict(
                done=False,
                confidence=0.95,
                summary="Deterministic patch compiler produced a pending approval; no model verifier was run.",
                next_focus="Review and apply the pending patch approval, then continue the task.",
            )
            return (
                [
                    RoundRecord(
                        round_number=1,
                        strategist="",
                        critic="",
                        planner=planner,
                        tool_results=tool_results,
                        verifier=verifier,
                        reasoning_source="deterministic",
                    )
                ],
                self._build_approval_required_answer(pending_approvals),
                "approval_required",
                "stopped",
            )

        verifier = VerifierVerdict(
            done=False,
            confidence=0.0,
            summary="Deterministic patch compiler ran but did not produce a pending approval; no model verifier was run.",
            next_focus="Inspect the tool output and repair the write path before retrying.",
        )
        error_text = tool_results[0].error or "The deterministic patch route did not create an approval artifact."
        return (
            [
                RoundRecord(
                    round_number=1,
                    strategist="",
                    critic="",
                    planner=planner,
                    tool_results=tool_results,
                    verifier=verifier,
                    reasoning_source="deterministic",
                )
            ],
            error_text,
            "deterministic_route_failed",
            "failed",
        )

    def _run_repository_inspection_route(
        self,
        *,
        task: str,
        workspace: Path,
        max_rounds: int,
        max_actions: int,
        warnings: list[str],
        emit_progress: Callable[[str], None],
    ) -> tuple[list[RoundRecord], str, str, RunResult["status"]]:
        rounds: list[RoundRecord] = []

        for round_number in range(1, max_rounds + 1):
            emit_progress(f"Round {round_number}/{max_rounds}: building context")
            emit_progress(f"Round {round_number}/{max_rounds}: strategist")
            seed_action = self._next_repository_inspection_seed_action(
                task=task,
                workspace=workspace,
                previous_rounds=rounds,
            )
            if seed_action is None:
                break

            emit_progress(f"Round {round_number}/{max_rounds}: critic")
            actions = self._supplement_inspection_actions(
                [seed_action],
                task=task,
                workspace=workspace,
                previous_rounds=rounds,
                max_actions=max_actions,
                task_route="repository_inspection",
            )[:max_actions]
            emit_progress(f"Round {round_number}/{max_rounds}: planner")
            emit_progress(f"Round {round_number}/{max_rounds}: executing {len(actions)} tool action(s)")
            tool_results = self._tools.execute_actions(
                actions,
                workspace=workspace,
                execution_mode="read_only",
                approval_context={
                    "task": task,
                    "execution_mode": "read_only",
                },
            )
            emit_progress(f"Round {round_number}/{max_rounds}: verifier")
            verifier = VerifierVerdict(
                done=False,
                confidence=min(0.25 + 0.2 * round_number, 0.85),
                summary="Deterministic inspection collected repository context; no model verifier was run.",
                next_focus="Read the highest-signal runtime files and synthesize the next engineering tasks.",
            )
            # Deterministic route: strategist/critic intentionally left empty
            # and the round is labeled `reasoning_source="deterministic"`.
            rounds.append(
                RoundRecord(
                    round_number=round_number,
                    strategist="",
                    critic="",
                    planner=PlannerTurn(
                        summary="Deterministic inspection bootstrap selected the next read-only repository actions.",
                        should_stop=False,
                        final_answer=None,
                        actions=actions,
                    ),
                    tool_results=tool_results,
                    verifier=verifier,
                    reasoning_source="deterministic",
                )
            )

            synthesized = self._maybe_synthesize_repository_answer(
                task=task,
                rounds=rounds,
                workspace=workspace,
                allow_partial=self._should_allow_early_partial_repository_synthesis(
                    task=task,
                    rounds=rounds,
                    max_rounds=max_rounds,
                ),
            )
            if synthesized:
                return (rounds, synthesized, "inspection_synthesized", "completed")

        strict_synthesized = self._maybe_synthesize_repository_answer(
            task=task,
            rounds=rounds,
            workspace=workspace,
        )
        if strict_synthesized:
            return (rounds, strict_synthesized, "inspection_synthesized", "completed")

        partial_synthesized = self._maybe_synthesize_repository_answer(
            task=task,
            rounds=rounds,
            workspace=workspace,
            allow_partial=True,
        )
        if partial_synthesized:
            warnings.append("Inspection run hit max rounds; used partial synthesis from gathered evidence.")
            return (rounds, partial_synthesized, "inspection_synthesized", "completed")

        return (rounds, self._build_fallback_answer(rounds, task), "max_rounds_reached", "stopped")

    def _next_repository_inspection_seed_action(
        self,
        *,
        task: str,
        workspace: Path,
        previous_rounds: list[RoundRecord],
    ) -> ToolAction | None:
        successful = self._successful_action_signatures(previous_rounds, workspace)
        for candidate in [
            "README.md",
            "teamai/config.py",
            "pyproject.toml",
            "teamai/supervisor.py",
            "teamai/cli.py",
            "teamai",
        ]:
            action = self._candidate_to_action(candidate, task, workspace)
            if action is None:
                continue
            if self._action_signature(action, workspace) in successful:
                continue
            return action

        for candidate in self._priority_candidates(
            previous_rounds,
            workspace,
            task=task,
            task_route="repository_inspection",
        ):
            action = self._candidate_to_action(candidate, task, workspace)
            if action is None:
                continue
            if self._action_signature(action, workspace) in successful:
                continue
            return action
        return None

    @staticmethod
    def _can_bootstrap_repository_inspection(workspace: Path) -> bool:
        return any(
            (workspace / candidate).exists()
            for candidate in ["README.md", "pyproject.toml", "PROJECT_MEMORY.md", "teamai"]
        )

    @staticmethod
    def _render_transcript(
        rounds: list[RoundRecord],
        task: str,
        workspace: Path,
        warnings: list[str],
    ) -> str:
        chunks = [f"TASK\n{task}", f"WORKSPACE\n{workspace}"]
        if warnings:
            chunks.append("WARNINGS\n" + "\n".join(f"- {warning}" for warning in warnings))

        for record in rounds:
            chunks.append(f"ROUND {record.round_number}\nReasoning Source\n{record.reasoning_source}")
            chunks.append(
                "ROUND "
                f"{record.round_number}\nStrategist\n"
                f"{ClosedLoopSupervisor._render_round_persona_text(record.strategist, reasoning_source=record.reasoning_source)}"
            )
            chunks.append(
                "ROUND "
                f"{record.round_number}\nCritic\n"
                f"{ClosedLoopSupervisor._render_round_persona_text(record.critic, reasoning_source=record.reasoning_source)}"
            )
            chunks.append(
                f"ROUND {record.round_number}\nPlanner\n{json.dumps(record.planner.model_dump(), indent=2)}"
            )
            tool_dump = [result.model_dump() for result in record.tool_results]
            chunks.append(f"ROUND {record.round_number}\nTool Results\n{json.dumps(tool_dump, indent=2)}")
            chunks.append(
                f"ROUND {record.round_number}\nVerifier\n{json.dumps(record.verifier.model_dump(), indent=2)}"
            )

        return "\n\n".join(chunks)

    @staticmethod
    def _sanitize_model_text(text: str) -> str:
        cleaned = text.strip()
        cleaned = cleaned.replace("<|channel>thought", "").replace("<|channel|>thought", "")
        cleaned = cleaned.replace("<think>", "").replace("</think>", "")
        return cleaned.strip()

    def _render_recent_actions(self, previous_rounds: list[RoundRecord], workspace: Path) -> str:
        rendered: list[str] = []
        for record in previous_rounds[-4:]:
            for action, result in zip(record.planner.actions, record.tool_results):
                if not result.success:
                    continue
                rendered.append(f"- {self._action_signature(action, workspace)}")
        return "\n".join(rendered[-6:]) if rendered else "No successful actions yet."

    def _render_suggested_paths(
        self,
        previous_rounds: list[RoundRecord],
        workspace: Path,
        *,
        task: str,
        task_route: str,
    ) -> str:
        candidates = self._priority_candidates(
            previous_rounds,
            workspace,
            task=task,
            task_route=task_route,
        )
        if not candidates:
            return "No obvious next paths."
        return "\n".join(f"- {candidate}" for candidate in candidates[:8])

    def _remove_repeated_actions(
        self,
        actions: list[ToolAction],
        *,
        workspace: Path,
        previous_rounds: list[RoundRecord],
        warnings: list[str],
    ) -> list[ToolAction]:
        successful = self._successful_action_signatures(previous_rounds, workspace)
        filtered: list[ToolAction] = []
        for action in actions:
            signature = self._action_signature(action, workspace)
            if signature in successful:
                warnings.append(f"Skipping repeated successful action: {signature}")
                continue
            filtered.append(action)
        return filtered

    def _remove_invalid_actions(
        self,
        actions: list[ToolAction],
        *,
        workspace: Path,
        warnings: list[str],
    ) -> list[ToolAction]:
        filtered: list[ToolAction] = []
        for action in actions:
            if not self._action_has_required_args(action):
                warnings.append(f"Skipping invalid action arguments: {action.tool}")
                continue
            if self._action_needs_existing_target(action):
                target_kind = self._action_target_kind(action, workspace)
                if target_kind is None:
                    warnings.append(f"Skipping invalid action target: {self._action_signature(action, workspace)}")
                    continue
                if not self._action_target_matches_tool(action, target_kind):
                    warnings.append(
                        f"Skipping incompatible action target: {self._action_signature(action, workspace)}"
                    )
                    continue
            filtered.append(action)
        return filtered

    def _successful_action_signatures(
        self,
        previous_rounds: list[RoundRecord],
        workspace: Path,
    ) -> set[str]:
        signatures: set[str] = set()
        for record in previous_rounds:
            for action, result in zip(record.planner.actions, record.tool_results):
                if result.success:
                    signatures.add(self._action_signature(action, workspace))
        return signatures

    def _action_signature(self, action: ToolAction, workspace: Path) -> str:
        tool = action.tool
        if tool == "write_file":
            changes = action.args.get("changes")
            if isinstance(changes, list) and changes:
                bundle_paths = []
                for entry in changes:
                    if not isinstance(entry, dict):
                        continue
                    normalized_path = self._normalize_path_arg(entry.get("path", "."), workspace)
                    bundle_paths.append(
                        f"{normalized_path}:{self._digest_signature_value(entry.get('content', ''))}"
                    )
                if bundle_paths:
                    return f"{tool}:{'|'.join(bundle_paths)}"
            content = str(action.args.get("content", ""))
            path = self._normalize_path_arg(action.args.get("path", "."), workspace)
            return f"{tool}:{path}:{self._digest_signature_value(content)}"
        if tool in {"list_files", "read_file"}:
            return f"{tool}:{self._normalize_path_arg(action.args.get('path', '.'), workspace)}"
        if tool == "replace_in_file":
            path = self._normalize_path_arg(action.args.get("path", "."), workspace)
            old_text = str(action.args.get("old_text", ""))
            new_text = str(action.args.get("new_text", ""))
            replace_all = bool(action.args.get("replace_all", False))
            return (
                f"{tool}:{path}:"
                f"{self._digest_signature_value(old_text)}:"
                f"{self._digest_signature_value(new_text)}:"
                f"{replace_all}"
            )
        if tool == "search_text":
            pattern = str(action.args.get("pattern", "")).strip()
            path = self._normalize_path_arg(action.args.get("path", "."), workspace)
            return f"{tool}:{path}:{pattern}"
        if tool == "run_command":
            command = action.args.get("command", "")
            cwd = self._normalize_path_arg(action.args.get("cwd", "."), workspace)
            return f"{tool}:{cwd}:{command}"
        return tool

    @staticmethod
    def _digest_signature_value(value: object) -> str:
        digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()
        return digest[:12]

    @staticmethod
    def _action_needs_existing_target(action: ToolAction) -> bool:
        return action.tool in {"list_files", "search_text", "read_file", "replace_in_file", "run_command"}

    @staticmethod
    def _action_has_required_args(action: ToolAction) -> bool:
        args = action.args
        if action.tool == "search_text":
            return bool(str(args.get("pattern", "")).strip())
        if action.tool == "read_file":
            return bool(str(args.get("path", "")).strip())
        if action.tool == "run_command":
            command = args.get("command", "")
            if isinstance(command, list):
                return any(str(part).strip() for part in command)
            return bool(str(command).strip())
        if action.tool == "write_file":
            changes = args.get("changes")
            if isinstance(changes, list):
                return any(
                    isinstance(change, dict)
                    and bool(str(change.get("path", "")).strip())
                    and "content" in change
                    for change in changes
                )
            return "path" in args and "content" in args
        if action.tool == "replace_in_file":
            return "path" in args and "old_text" in args and "new_text" in args
        return True

    def _action_target_kind(self, action: ToolAction, workspace: Path) -> str | None:
        try:
            candidate = action.args.get("cwd", ".") if action.tool == "run_command" else action.args.get("path", ".")
            raw = Path(str(candidate)).expanduser()
            resolved = raw.resolve() if raw.is_absolute() else (workspace / raw).resolve()
            if not resolved.exists():
                return None
            if resolved.is_dir():
                return "dir"
            if resolved.is_file():
                return "file"
            return "other"
        except Exception:
            return None

    @staticmethod
    def _action_target_matches_tool(action: ToolAction, target_kind: str) -> bool:
        if action.tool == "list_files":
            return target_kind == "dir"
        if action.tool == "run_command":
            return target_kind == "dir"
        if action.tool in {"read_file", "replace_in_file"}:
            return target_kind == "file"
        if action.tool == "search_text":
            return target_kind in {"file", "dir"}
        return True

    def _normalize_path_arg(self, candidate: object, workspace: Path) -> str:
        raw = str(candidate or ".").strip()
        if raw not in {".", "/"}:
            raw = raw.rstrip("/") or "."
        path = Path(raw).expanduser()
        try:
            resolved = path.resolve() if path.is_absolute() else (workspace / path).resolve()
            return str(resolved.relative_to(workspace))
        except Exception:
            return raw

    def _heuristic_plan_from_context(
        self,
        *,
        task: str,
        raw_response: str,
        user_prompt: str,
        workspace: Path,
        previous_rounds: list[RoundRecord],
        max_actions: int,
        execution_mode: str = "read_only",
        task_route: str = "multi_agent_loop",
    ) -> PlannerTurn:
        effective_route = (
            task_route
            if task_route != "multi_agent_loop"
            else self._classify_task_route(task=task, execution_mode=execution_mode, workspace=workspace)
        )
        write_action = self._heuristic_write_action_from_task(
            task=task,
            workspace=workspace,
            previous_rounds=previous_rounds,
            execution_mode=execution_mode,
        )
        if write_action is not None:
            return PlannerTurn(
                summary=f"Heuristic fallback selected `{write_action.tool}` for `{write_action.args.get('path', '.')}`.",
                should_stop=False,
                final_answer=None,
                actions=[write_action][:max_actions],
            )

        successful = self._successful_action_signatures(previous_rounds, workspace)
        text = f"{raw_response}\n{user_prompt}"
        for candidate in self._extract_candidate_paths(text):
            action = self._candidate_to_action(candidate, text, workspace)
            if action is None:
                continue
            signature = self._action_signature(action, workspace)
            if signature in successful:
                continue
            return PlannerTurn(
                summary=f"Heuristic fallback selected `{action.tool}` for `{candidate}`.",
                should_stop=False,
                final_answer=None,
                actions=[action][:max_actions],
            )

        for candidate in self._priority_candidates(
            previous_rounds,
            workspace,
            task=task,
            task_route=effective_route,
        ):
            action = self._candidate_to_action(candidate, text, workspace)
            if action is None:
                continue
            signature = self._action_signature(action, workspace)
            if signature in successful:
                continue
            return PlannerTurn(
                summary=f"Heuristic fallback selected `{action.tool}` for `{candidate}`.",
                should_stop=False,
                final_answer=None,
                actions=[action][:max_actions],
            )

        return PlannerTurn(
            summary="Heuristic fallback could not find a novel next action.",
            should_stop=False,
            final_answer=None,
            actions=[],
        )

    @staticmethod
    def _is_repository_inspection_task(task: str) -> bool:
        text = task.lower()
        if ClosedLoopSupervisor._is_explicit_write_task(text):
            return False
        inspection_markers = ["inspect", "review", "analyze", "understand", "summarize"]
        repo_markers = ["repository", "repo", "codebase", "project", "workspace"]
        task_markers = ["engineering task", "implementation step", "next step", "project state"]
        return (
            any(marker in text for marker in inspection_markers)
            and (any(marker in text for marker in repo_markers) or any(marker in text for marker in task_markers))
        )

    @staticmethod
    def _is_broad_coding_task(task: str) -> bool:
        text = task.lower()
        if ClosedLoopSupervisor._is_explicit_write_task(text):
            return False
        if ClosedLoopSupervisor._is_repository_inspection_task(text):
            return False
        return bool(
            re.search(
                (
                    r"\b("
                    r"implement|fix|debug|refactor|build|create|add support|wire up|make .* work|ship|complete|"
                    r"improve|harden|optimize|extend|upgrade|stabilize|tighten"
                    r")\b"
                ),
                text,
            )
        )

    def _local_drift_reroute_reason(
        self,
        *,
        task: str,
        workspace: Path,
        task_route: str,
        round_records: list[RoundRecord],
    ) -> str | None:
        actual_rounds = [record for record in round_records if record.round_number > 0]
        if len(actual_rounds) < 2:
            return None
        if task_route not in {"explicit_write_loop", "multi_agent_loop"}:
            return None
        if self._is_repository_inspection_task(task):
            return None

        recent = actual_rounds[-2:]
        low_confidence = all(record.verifier.confidence <= 0.35 for record in recent)
        if not low_confidence:
            return None

        recent_without_success = all(not any(result.success for result in record.tool_results) for record in recent)
        recent_without_actions = all(not record.planner.actions for record in recent)
        repeated_focus = self._recent_focus_is_repeating(recent)
        if not (recent_without_success or recent_without_actions or repeated_focus):
            return None

        if task_route == "explicit_write_loop":
            target_path = self._extract_primary_file_target(task, workspace)
            target_observed = (
                target_path is None
                or self._path_was_successfully_read(round_records=actual_rounds, workspace=workspace, path=target_path)
            )
            proposed_write = any(
                action.tool in {"write_file", "replace_in_file"}
                for record in actual_rounds
                for action in record.planner.actions
            )
            if target_observed and not proposed_write:
                target_label = target_path or "the target file"
                return (
                    "Local write loop drifted after reading "
                    f"{target_label} without producing a concrete compiler-safe patch; rerouting to a Codex handoff."
                )
            return None

        if self._is_broad_coding_task(task):
            return (
                "Local planning drifted across repeated low-confidence rounds on a broad coding task; "
                "rerouting to a Codex handoff."
            )
        return None

    @staticmethod
    def _recent_focus_is_repeating(rounds: list[RoundRecord]) -> bool:
        focuses = [(record.verifier.next_focus or "").strip().lower() for record in rounds if (record.verifier.next_focus or "").strip()]
        return len(focuses) >= 2 and len(set(focuses[-2:])) == 1

    def _path_was_successfully_read(
        self,
        *,
        round_records: list[RoundRecord],
        workspace: Path,
        path: str,
    ) -> bool:
        normalized_path = self._normalize_path_arg(path, workspace)
        for record in round_records:
            for result in record.tool_results:
                if not result.success or result.tool != "read_file":
                    continue
                raw_path = str(result.metadata.get("path", "")).strip()
                if not raw_path:
                    continue
                if self._normalize_path_arg(raw_path, workspace) == normalized_path:
                    return True
        return False

    def _classify_task_route(
        self,
        *,
        task: str,
        execution_mode: str,
        workspace: Path,
        continuation_context: dict[str, object] | None = None,
    ) -> str:
        signature = self._build_task_signature(
            task=task,
            execution_mode=execution_mode,
            workspace=workspace,
            continuation_context=continuation_context or {},
        )
        decision = self._registry.pick_best(
            task_signature=signature,
            prefer_local=True,
            env_check=False,
        )
        if isinstance(decision, RoutingDecision) and decision.capability in _VALID_ROUTES:
            self._apply_routing_decision(decision)
            return decision.capability

        self._last_routing_decision = None
        self._active_model_id = self._settings.model_id
        self._backend = self._backend_by_model.get(self._settings.model_id, self._backend)
        return "multi_agent_loop"

    def _build_task_signature(
        self,
        *,
        task: str,
        execution_mode: str,
        workspace: Path,
        continuation_context: dict[str, object],
    ) -> TaskSignature:
        compiled = None
        if execution_mode == "workspace_write":
            compiled = self._compile_small_write_action_from_task(task=task, workspace=workspace)

        repo_index = self._repo_indexer.build(workspace)
        expected_files_touched = max(len(infer_task_scopes(task=task, workspace=workspace, repo_index=repo_index)), 1)
        route_health = self._memory.load_routing_health(
            workspace,
            recent_window=self._registry.routing.recent_success_window,
            broken_repair_rate=self._registry.routing.broken_repair_rate,
        )
        broad_coding = self._is_broad_coding_task(task)
        memory_pressure = any(health.memory_pressure for health in route_health.values())
        if broad_coding and self._settings.max_tokens_per_turn >= 256:
            memory_pressure = True
        complexity = compute_complexity(
            task=task,
            repo_index=repo_index,
            expected_files_touched=expected_files_touched,
        )
        policy_scores = self._memory.load_policy_scores(
            workspace,
            task=task,
            complexity=complexity,
        )
        agent_policy_scores = self._memory.load_agent_policy_scores(
            workspace,
            task=task,
            complexity=complexity,
        )

        return TaskSignature(
            task=task,
            execution_mode=execution_mode,
            complexity=complexity,
            continuation=bool(continuation_context),
            deterministic_candidate=compiled is not None,
            explicit_write=execution_mode == "workspace_write" and self._is_explicit_write_task(task),
            repository_inspection=self._is_repository_inspection_task(task),
            broad_coding=broad_coding,
            memory_pressure=memory_pressure,
            expected_files_touched=expected_files_touched,
            has_tests=repo_index.has_tests,
            policy_scores=policy_scores,
            agent_policy_scores=agent_policy_scores,
            route_health=route_health,
        )

    def _apply_routing_decision(self, decision: RoutingDecision) -> None:
        self._last_routing_decision = decision
        self._active_model_id = self._settings.model_id
        self._backend = self._backend_by_model.get(self._settings.model_id, self._backend)

        if decision.agent.type != "local_mlx":
            return
        if decision.capability == "codex_handoff":
            return
        if not self._settings.model_router:
            return

        target_model = decision.agent.model_id.strip() or self._settings.model_id
        self._active_model_id = target_model
        if target_model == self._settings.model_id or not self._owns_backend:
            return
        if target_model not in self._backend_by_model:
            override_settings = self._settings.__class__(
                model_id=target_model,
                model_revision=self._settings.model_revision,
                force_download=self._settings.force_download,
                trust_remote_code=self._settings.trust_remote_code,
                enable_thinking=self._settings.enable_thinking,
                workspace_root=self._settings.workspace_root,
                max_rounds=self._settings.max_rounds,
                max_actions_per_round=self._settings.max_actions_per_round,
                max_tokens_per_turn=self._settings.max_tokens_per_turn,
                temperature=self._settings.temperature,
                allow_shell=self._settings.allow_shell,
                allow_writes=self._settings.allow_writes,
                command_timeout_seconds=self._settings.command_timeout_seconds,
                max_file_bytes=self._settings.max_file_bytes,
                max_command_output_chars=self._settings.max_command_output_chars,
                host=self._settings.host,
                port=self._settings.port,
                model_router=self._settings.model_router,
            )
            self._backend_by_model[target_model] = MLXModelBackend(override_settings)
        self._backend = self._backend_by_model[target_model]

    def _active_agent_entry(self) -> object | None:
        for agent in self._registry.agents:
            if agent.model_id == self._active_model_id:
                return agent
        return None

    @staticmethod
    def _latest_verifier_confidence(run_state: AutonomousRunState) -> float | None:
        if not run_state.verifier_outputs:
            return None
        return run_state.verifier_outputs[-1].confidence

    def _append_routing_trace(
        self,
        run_state: AutonomousRunState,
        *,
        stage: str,
        capability: str,
        model_id: str,
        complexity: str,
        agent_id: str | None = None,
        score: float | None = None,
        reasons: tuple[str, ...] | list[str] = (),
        escalation_reason: str | None = None,
        outcome: str | None = None,
    ) -> None:
        run_state.routing_trace.append(
            RoutingTraceEntry(
                stage=stage,  # type: ignore[arg-type]
                capability=capability,
                model_id=model_id,
                agent_id=agent_id,
                complexity=complexity,  # type: ignore[arg-type]
                score=score,
                reasons=[str(reason) for reason in reasons],
                escalation_reason=escalation_reason,
                outcome=outcome,
                retries=len(run_state.failures_encountered),
                verifier_confidence=self._latest_verifier_confidence(run_state),
            )
        )

    def _activate_backend_for_model(self, model_id: str) -> bool:
        target_model = model_id.strip()
        if not target_model:
            return False
        backend = self._backend_by_model.get(target_model)
        if backend is None:
            if not self._owns_backend:
                return False
            override_settings = self._settings.__class__(
                model_id=target_model,
                model_revision=self._settings.model_revision,
                force_download=self._settings.force_download,
                trust_remote_code=self._settings.trust_remote_code,
                enable_thinking=self._settings.enable_thinking,
                workspace_root=self._settings.workspace_root,
                max_rounds=self._settings.max_rounds,
                max_actions_per_round=self._settings.max_actions_per_round,
                max_tokens_per_turn=self._settings.max_tokens_per_turn,
                temperature=self._settings.temperature,
                allow_shell=self._settings.allow_shell,
                allow_writes=self._settings.allow_writes,
                command_timeout_seconds=self._settings.command_timeout_seconds,
                max_file_bytes=self._settings.max_file_bytes,
                max_command_output_chars=self._settings.max_command_output_chars,
                host=self._settings.host,
                port=self._settings.port,
                model_router=self._settings.model_router,
                allow_git_push=self._settings.allow_git_push,
                default_handoff_engine=self._settings.default_handoff_engine,
                max_inline_handoff_revisions=self._settings.max_inline_handoff_revisions,
            )
            backend = MLXModelBackend(override_settings)
            self._backend_by_model[target_model] = backend
        self._active_model_id = target_model
        self._backend = backend
        return True

    def _try_escalate_to_stronger_local(
        self,
        *,
        task_route: str,
        run_state: AutonomousRunState,
        reason: str,
        emit_progress: Callable[[str], None],
        warnings: list[str],
    ) -> str | None:
        attempted_models = {
            entry.model_id
            for entry in run_state.routing_trace
            if entry.stage == "inline_escalation" and entry.model_id
        }
        candidates = self._registry.stronger_local_candidates(
            current_model_id=self._active_model_id,
            capabilities=[task_route, "explicit_write_loop", "multi_agent_loop"],
            env_check=False,
        )
        for agent in candidates:
            if agent.model_id in attempted_models:
                continue
            previous_model_id = self._active_model_id
            if not self._activate_backend_for_model(agent.model_id):
                continue
            warnings.append(
                f"Inline escalation switched from `{previous_model_id}` to stronger local model `{agent.model_id}` "
                f"because {reason}."
            )
            emit_progress(f"Inline escalation: switching to stronger local model `{agent.model_id}`")
            run_state.repair_attempts.append(
                RepairAttempt(
                    round_number=run_state.round_number,
                    failure_type="unknown",
                    strategy=f"Escalate to stronger local model because {reason}.",
                    status="escalated",
                    notes=agent.id,
                )
            )
            next_route = "explicit_write_loop" if task_route == "codex_handoff" else task_route
            self._append_routing_trace(
                run_state,
                stage="inline_escalation",
                capability=next_route,
                model_id=agent.model_id,
                agent_id=agent.id,
                complexity=run_state.complexity,
                escalation_reason=reason,
                outcome="stronger_local_selected",
            )
            return next_route
        return None

    @staticmethod
    def _bridge_engine_from_agent(agent: object | None) -> str | None:
        if agent is None:
            return None
        agent_type = str(getattr(agent, "type", "")).strip().lower()
        agent_id = str(getattr(agent, "id", "")).strip().lower()
        if agent_type == "openai" or agent_id.startswith("codex"):
            return "codex"
        if agent_type == "google" or agent_id.startswith("gemini"):
            return "gemini"
        if agent_type == "local_mlx" or agent_id.startswith("local"):
            return "local"
        return None

    def _select_verified_handoff_target(self, request: RunRequest) -> tuple[str, str | None] | None:
        requested_engine = (request.handoff_engine or "").strip().lower()
        requested_model = (request.handoff_model or "").strip() or None
        if requested_engine in {"codex", "gemini", "local"}:
            return requested_engine, requested_model

        default_cloud = self._registry.pick_cloud(env_check=True)
        engine = self._bridge_engine_from_agent(default_cloud)
        if engine is not None:
            model = requested_model or (str(getattr(default_cloud, "model_id", "")).strip() or None)
            return engine, model

        fallback_engine = (self._settings.default_handoff_engine or "").strip().lower()
        fallback_allowed = (
            fallback_engine == "local"
            or (fallback_engine == "codex" and bool(os.getenv("OPENAI_API_KEY", "").strip()))
            or (fallback_engine == "gemini" and bool(os.getenv("GOOGLE_API_KEY", "").strip()))
        )
        if fallback_engine in {"codex", "gemini", "local"} and fallback_allowed:
            return fallback_engine, requested_model
        return None

    def _has_available_stronger_local(self, *, task_route: str) -> bool:
        candidates = self._registry.stronger_local_candidates(
            current_model_id=self._active_model_id,
            capabilities=[task_route, "explicit_write_loop", "multi_agent_loop"],
            env_check=False,
        )
        return any(self._owns_backend or agent.model_id in self._backend_by_model for agent in candidates)

    def _can_inline_escalate(self, *, request: RunRequest, task_route: str) -> bool:
        return self._has_available_stronger_local(task_route=task_route) or self._select_verified_handoff_target(request) is not None

    @staticmethod
    def _default_push_branch_name(task_id: str) -> str:
        return f"teamai/{task_id}"

    def _current_diff_text(self, *, workspace: Path, sandbox_workspace: Path) -> str:
        return render_change_diff(collect_workspace_changes(source_root=workspace, modified_root=sandbox_workspace))

    def _build_inline_handoff_payload(
        self,
        *,
        task: str,
        workspace: Path,
        repo_index: RepoIndex,
        task_scopes: tuple[str, ...],
        round_records: list[RoundRecord],
        run_state: AutonomousRunState,
        current_diff: str,
    ) -> CodexHandoffPayload:
        observed_paths = self._observed_paths_from_rounds(round_records, workspace)
        latest_failure = run_state.failures_encountered[-1] if run_state.failures_encountered else None
        repair_notes = tuple(
            f"{attempt.status}: {attempt.strategy} ({attempt.notes})".strip()
            for attempt in run_state.repair_attempts[-4:]
        )
        bundle = self._context_packager.build(
            task=task,
            workspace=workspace,
            repo_index=repo_index,
            task_scopes=task_scopes,
            observed_paths=observed_paths,
            changed_paths=tuple(run_state.files_changed),
            failure_output=latest_failure.raw_output if latest_failure is not None else "",
            prior_failed_repairs=repair_notes,
        )
        bundle = self._context_packager.with_diff(bundle, diff_text=current_diff)
        core_dependencies = list(bundle.relevant_paths[:4]) or list(task_scopes[:4]) or list(self._task_relevant_candidates(task, workspace)[:4])
        distilled_context: dict[str, str] = {}
        for path in core_dependencies:
            detail_parts: list[str] = []
            if path in bundle.symbol_definitions:
                detail_parts.append("symbols: " + ", ".join(bundle.symbol_definitions[path][:4]))
            if path in bundle.nearest_imports:
                detail_parts.append("imports: " + ", ".join(bundle.nearest_imports[path][:4]))
            if path in bundle.failing_tests:
                compact_test = " ".join(bundle.failing_tests[path].split())
                detail_parts.append(f"test excerpt: {compact_test[:220]}")
            if path in bundle.dependent_call_sites:
                detail_parts.append("dependent call sites: " + " | ".join(bundle.dependent_call_sites[path][:3]))
            if not detail_parts:
                try:
                    compact = " ".join((workspace / path).read_text(encoding="utf-8")[:400].split())
                except OSError:
                    compact = "No local excerpt available."
                detail_parts.append(compact[:240])
            distilled_context[path] = " ".join(detail_parts)

        structured_context = {
            "task": task,
            "repo_summary": bundle.repo_summary,
            "files_inspected": list(observed_paths),
            "current_diff": bundle.current_diff,
            "failing_checks": [
                {
                    "command": check.command,
                    "returncode": check.returncode,
                    "stdout_excerpt": (check.stdout or "")[-400:],
                    "stderr_excerpt": (check.stderr or "")[-400:],
                }
                for check in run_state.checks_run[-3:]
            ],
            "failure_classifications": [
                {
                    "type": failure.failure_type,
                    "summary": failure.summary,
                    "strategy": failure.strategy,
                }
                for failure in run_state.failures_encountered[-3:]
            ],
            "verifier_output": [
                {
                    "source": output.source,
                    "passed": output.passed,
                    "confidence": output.confidence,
                    "summary": output.summary,
                    "next_focus": output.next_focus,
                }
                for output in run_state.verifier_outputs[-4:]
            ],
            "prior_repair_attempts": [attempt.model_dump(mode="json") for attempt in run_state.repair_attempts[-4:]],
            "context_package": self._context_packager.render(bundle),
        }
        recommended_action = (
            "Produce a minimal verified patch that keeps scope narrow, fixes the failing checks, and preserves "
            "all unrelated behavior."
        )
        return CodexHandoffPayload(
            original_task=task,
            core_dependencies=core_dependencies,
            distilled_context=distilled_context,
            recommended_codex_action=recommended_action,
            structured_context=structured_context,
        )

    def _apply_verified_patch_inline(
        self,
        *,
        patch_file: Path,
        workspace: Path,
    ) -> tuple[bool, str, list[str]]:
        result = subprocess.run(
            ["patch", "-p1", "-E", "-i", str(patch_file.resolve())],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            check=False,
        )
        targets = [target.primary_path for target in extract_patch_targets(patch_file.read_text(encoding="utf-8"))]
        output = (result.stdout or "").strip()
        if result.stderr:
            output = f"{output}\n{result.stderr.strip()}".strip()
        return result.returncode == 0, output, [path for path in targets if path]

    def _execute_inline_verified_handoff(
        self,
        *,
        task: str,
        request: RunRequest,
        sandbox_workspace: Path,
        repo_index: RepoIndex,
        task_scopes: tuple[str, ...],
        round_records: list[RoundRecord],
        run_state: AutonomousRunState,
        reason: str,
        emit_progress: Callable[[str], None],
        warnings: list[str],
    ) -> dict[str, object]:
        target = self._select_verified_handoff_target(request)
        if target is None:
            return {"accepted": False, "stop_reason": "verified_handoff_unavailable"}

        engine, model = target
        from .integrations import get_bridge

        payload = self._build_inline_handoff_payload(
            task=task,
            workspace=sandbox_workspace,
            repo_index=repo_index,
            task_scopes=task_scopes,
            round_records=round_records,
            run_state=run_state,
            current_diff=self._current_diff_text(workspace=Path(run_state.workspace_path), sandbox_workspace=sandbox_workspace),
        )
        payload_dir = sandbox_workspace / ".teamai"
        payload_dir.mkdir(parents=True, exist_ok=True)
        payload_file = payload_dir / f"inline-handoff-{run_state.task_id}.json"
        patch_file = payload_dir / f"inline-handoff-{run_state.task_id}.patch"
        payload_file.write_text(payload.model_dump_json(indent=2) + "\n", encoding="utf-8")
        emit_progress(f"Inline escalation: executing verified `{engine}` handoff")
        bridge = get_bridge(engine)
        try:
            verified = bridge.execute_verified(
                project_root=sandbox_workspace,
                payload_file=payload_file,
                patch_file=patch_file,
                model=model,
                max_revision_attempts=(
                    request.max_handoff_revision_attempts
                    if request.max_handoff_revision_attempts is not None
                    else self._settings.max_inline_handoff_revisions
                ),
                create_approval=False,
            )
        except Exception as exc:
            warnings.append(f"Inline verified handoff failed before verification: {exc}")
            return {
                "accepted": False,
                "stop_reason": "verified_handoff_failed",
                "final_answer": f"Inline verified handoff failed: {exc}",
            }

        artifact = verified.artifact or HandoffArtifact(
            engine=engine,
            summary=f"Inline {engine} handoff patch.",
            diff=verified.execution.patch_text,
            rationale="Inline verified handoff fallback artifact.",
            confidence=0.6,
        )
        run_state.handoffs.append(artifact)
        run_state.verifier_outputs.append(
            VerifierOutput(
                source="handoff",
                passed=bool(verified.accepted and verified.verification.success),
                confidence=artifact.confidence,
                summary=artifact.summary,
                next_focus=(verified.revision_requests[-1].details if verified.revision_requests else None),
            )
        )
        self._append_routing_trace(
            run_state,
            stage="verified_handoff",
            capability="verified_handoff_execution",
            model_id=verified.execution.model,
            complexity=run_state.complexity,
            escalation_reason=reason,
            outcome="accepted" if verified.accepted else "rejected",
        )
        if not verified.accepted:
            warnings.append(
                "Inline verified handoff was rejected after revision requests: "
                + (verified.revision_requests[-1].details if verified.revision_requests else "verification failed")
            )
            return {
                "accepted": False,
                "stop_reason": "verified_handoff_rejected",
                "final_answer": (
                    "Inline verified handoff could not produce an acceptable patch. "
                    + (verified.revision_requests[-1].details if verified.revision_requests else "Verification failed.")
                ),
            }

        applied, apply_output, changed_paths = self._apply_verified_patch_inline(
            patch_file=verified.execution.patch_file,
            workspace=sandbox_workspace,
        )
        if not applied:
            warnings.append(apply_output or "Inline verified patch could not be applied to the active sandbox.")
            return {
                "accepted": False,
                "stop_reason": "verified_handoff_apply_failed",
                "final_answer": "Inline verified handoff succeeded in isolation but the patch could not be applied to the active sandbox.",
            }

        normalized_changed_paths = [
            self._normalize_path_arg(path, sandbox_workspace)
            for path in changed_paths
            if str(path).strip()
        ]
        run_state.files_changed = list(dict.fromkeys([*run_state.files_changed, *normalized_changed_paths]))
        check_records = self._safe_commands.run_checks(
            workspace=sandbox_workspace,
            changed_paths=normalized_changed_paths,
            repo_index=repo_index,
        )
        run_state.checks_run.extend(check_records)
        check_success = bool(check_records) and all(check.returncode == 0 for check in check_records)
        if not check_success:
            diagnosis = classify_failure(check_records)
            if diagnosis is not None:
                run_state.failures_encountered.append(diagnosis)
                run_state.repair_attempts.append(
                    RepairAttempt(
                        round_number=run_state.round_number,
                        failure_type=diagnosis.failure_type,
                        strategy=diagnosis.strategy,
                        status="failed",
                        notes="inline_verified_handoff_post_apply",
                    )
                )
            return {
                "accepted": False,
                "stop_reason": "verified_handoff_post_apply_checks_failed",
                "final_answer": "Inline verified handoff patch applied, but the active sandbox checks still failed.",
                "changed_paths": normalized_changed_paths,
            }

        warnings.append(
            f"Inline verified handoff via `{engine}` succeeded after {verified.revision_count} revision request(s)."
        )
        return {
            "accepted": True,
            "changed_paths": normalized_changed_paths,
            "summary": artifact.summary,
        }

    @staticmethod
    def _is_explicit_write_task(task: str) -> bool:
        text = task.lower()
        has_write_verb = bool(
            re.search(r"\b(update|edit|modify|replace|rewrite|append|insert|write|add)\b", text)
        )
        has_file_target = bool(re.search(r"\b[\w./-]+\.\w+\b", text))
        write_markers = [
            "workspace_write",
            "patch approval",
            "replace_in_file",
            "write_file",
        ]
        return (has_write_verb and has_file_target) or any(marker in text for marker in write_markers)

    def _supplement_inspection_actions(
        self,
        actions: list[ToolAction],
        *,
        task: str,
        workspace: Path,
        previous_rounds: list[RoundRecord],
        max_actions: int,
        task_route: str,
    ) -> list[ToolAction]:
        if len(actions) >= max_actions:
            return actions
        if any(action.tool not in {"list_files", "read_file"} for action in actions):
            return actions

        supplemented = actions[:]
        seen_signatures = self._successful_action_signatures(previous_rounds, workspace)
        seen_signatures.update(self._action_signature(action, workspace) for action in supplemented)

        for candidate in self._priority_candidates(
            previous_rounds,
            workspace,
            task=task,
            task_route=task_route,
            current_action_signatures=seen_signatures,
        ):
            action = self._candidate_to_action(candidate, task, workspace)
            if action is None:
                continue
            signature = self._action_signature(action, workspace)
            if signature in seen_signatures:
                continue
            supplemented.append(action)
            seen_signatures.add(signature)
            if len(supplemented) >= max_actions:
                break

        return supplemented

    def _priority_candidates(
        self,
        previous_rounds: list[RoundRecord],
        workspace: Path,
        *,
        task: str,
        task_route: str,
        current_action_signatures: set[str] | None = None,
    ) -> list[str]:
        successful = self._successful_action_signatures(previous_rounds, workspace)
        available_signatures = successful | (current_action_signatures or set())
        candidates: list[str] = []

        def add(candidate: str) -> None:
            normalized = candidate.rstrip("/")
            resolved = (workspace / normalized).resolve()
            if not resolved.exists():
                return
            tool = "list_files" if resolved.is_dir() else "read_file"
            signature = f"{tool}:{normalized or '.'}"
            if signature in available_signatures:
                return
            if normalized not in candidates:
                candidates.append(normalized)

        config_read = "read_file:teamai/config.py" in available_signatures
        cli_read = "read_file:teamai/cli.py" in available_signatures
        supervisor_read = "read_file:teamai/supervisor.py" in available_signatures

        if task_route == "codex_handoff":
            for candidate in self._task_relevant_candidates(task, workspace):
                add(candidate)

        if task_route == "repository_inspection":
            add("README.md")
            add("teamai/config.py")
            add("teamai/supervisor.py")
            add("teamai/cli.py")
            add("teamai")
        else:
            for candidate in ["README.md", "pyproject.toml", "setup.py", "PROJECT_MEMORY.md"]:
                add(candidate)
            add("teamai")

        if config_read:
            add("teamai/cli.py")
            add("teamai/supervisor.py")
            add("teamai/api.py")

        if cli_read or supervisor_read:
            add("teamai/model_backend.py")
            add("teamai/tools.py")
            add("teamai/prompts.py")

        for candidate in [
            "teamai/config.py",
            "teamai/cli.py",
            "teamai/supervisor.py",
            "teamai/model_backend.py",
            "teamai/tools.py",
            "teamai/api.py",
            "teamai/jobs.py",
            "teamai/schemas.py",
            "tests",
            "tests/test_supervisor.py",
            "tests/test_tools.py",
        ]:
            add(candidate)

        return candidates

    def _should_allow_early_partial_repository_synthesis(
        self,
        *,
        task: str,
        rounds: list[RoundRecord],
        max_rounds: int,
    ) -> bool:
        if not self._is_repository_inspection_task(task):
            return False
        if len(rounds) < 2:
            return False
        return True

    def _task_relevant_candidates(self, task: str, workspace: Path) -> list[str]:
        text = task.lower()
        candidates: list[str] = []
        seen: set[str] = set()

        def add(candidate: str) -> None:
            normalized = self._normalize_path_arg(candidate, workspace)
            resolved = (workspace / normalized).resolve()
            if not resolved.exists():
                return
            if normalized in seen:
                return
            seen.add(normalized)
            candidates.append(normalized)

        for candidate in self._extract_candidate_paths(task):
            add(candidate)

        if any(marker in text for marker in ["stream", "streaming", "event output", "progress output"]):
            for candidate in [
                "teamai/cli.py",
                "teamai/api.py",
                "teamai/jobs.py",
                "teamai/schemas.py",
                "teamai/supervisor.py",
            ]:
                add(candidate)

        if "cli" in text:
            for candidate in ["teamai/cli.py", "teamai/__main__.py"]:
                add(candidate)

        if "api" in text:
            for candidate in ["teamai/api.py", "teamai/jobs.py", "teamai/schemas.py"]:
                add(candidate)

        if any(marker in text for marker in ["bridge", "handoff", "terminal"]):
            for candidate in [
                "teamai/bridge.py",
                "teamai/cli.py",
                "teamai/handoff.py",
                "tests/test_bridge.py",
                "tests/test_handoff.py",
            ]:
                add(candidate)

        if any(marker in text for marker in ["approval", "patch", "write path", "coarse write", "workspace_write", "deterministic"]):
            for candidate in [
                "teamai/tools.py",
                "teamai/approvals.py",
                "teamai/supervisor.py",
                "tests/test_tools.py",
                "tests/test_approvals.py",
                "tests/test_supervisor.py",
            ]:
                add(candidate)

        if any(marker in text for marker in ["memory", "history", "persist", "cross-run"]):
            for candidate in [
                "teamai/memory.py",
                "teamai/prompts.py",
                "teamai/supervisor.py",
                "tests/test_memory.py",
            ]:
                add(candidate)

        if any(
            marker in text
            for marker in [
                "learned-note",
                "learned note",
                "improvement note",
                "self-improvement",
                "self improvement",
                "decay",
                "prune",
                "pruning",
                "stale lesson",
                "stale note",
            ]
        ):
            for candidate in [
                "teamai/memory.py",
                "tests/test_memory.py",
                "teamai/prompts.py",
                "teamai/handoff.py",
                "teamai/bridge.py",
                "teamai/supervisor.py",
            ]:
                add(candidate)

        if any(marker in text for marker in ["json", "planner", "verifier", "prompt", "structured output"]):
            for candidate in [
                "teamai/prompts.py",
                "teamai/schemas.py",
                "teamai/supervisor.py",
                "tests/test_supervisor.py",
            ]:
                add(candidate)

        if any(marker in text for marker in ["routing", "route", "implement", "fix", "refactor", "debug"]):
            for candidate in [
                "teamai/supervisor.py",
                "teamai/cli.py",
                "teamai/api.py",
                "teamai/tools.py",
                "teamai/schemas.py",
            ]:
                add(candidate)

        return candidates

    def _heuristic_write_action_from_task(
        self,
        *,
        task: str,
        workspace: Path,
        previous_rounds: list[RoundRecord],
        execution_mode: str,
    ) -> ToolAction | None:
        if execution_mode != "workspace_write" or not self._is_explicit_write_task(task):
            return None

        compiled = self._compile_small_write_action_from_task(task=task, workspace=workspace)
        if compiled is not None:
            return compiled

        target_path = self._extract_primary_file_target(task, workspace)
        if target_path is None:
            return None

        contents = self._collect_read_file_outputs(previous_rounds, workspace)
        raw_file_text = contents.get(target_path)
        if not raw_file_text:
            return None

        file_text = self._strip_read_file_line_numbers(raw_file_text)
        sentence = self._extract_task_sentence(task)
        anchor = self._extract_task_anchor(task)
        if not sentence or not anchor:
            return None

        paragraph = self._find_paragraph_starting_with(file_text, anchor)
        if paragraph is None or sentence in paragraph or sentence in file_text:
            return None

        return ToolAction(
            tool="replace_in_file",
            reason="Propose the explicitly requested patch approval for the target file.",
            args={
                "path": target_path,
                "old_text": paragraph,
                "new_text": f"{paragraph}\n\n{sentence}",
                "replace_all": False,
            },
        )

    def _compile_python_function_and_unittest_bundle_action(
        self,
        *,
        task: str,
        workspace: Path,
    ) -> ToolAction | None:
        targets = self._extract_file_targets(task, workspace)
        if len(targets) < 2:
            return None

        source_path = next((path for path in targets if path.endswith(".py") and "test" not in Path(path).name.lower()), None)
        test_path = next((path for path in targets if path.endswith(".py") and "test" in path.lower()), None)
        if source_path is None or test_path is None:
            return None

        function_name = self._extract_function_update_target(task)
        return_template = self._extract_python_string_normalizer_template(task)
        expectation = self._extract_unittest_io_expectation(task)
        if function_name is None or return_template is None or expectation is None:
            return None

        try:
            source_text = (workspace / source_path).read_text(encoding="utf-8")
            test_text = (workspace / test_path).read_text(encoding="utf-8")
        except Exception:
            return None

        updated_source = self._build_python_function_return_update(
            file_text=source_text,
            function_name=function_name,
            return_template=return_template,
        )
        updated_test = self._build_python_unittest_expectation_update(
            file_text=test_text,
            function_name=function_name,
            input_value=expectation[0],
            output_value=expectation[1],
        )

        changes: list[dict[str, str]] = []
        if updated_source is not None:
            changes.append({"path": source_path, "content": updated_source})
        if updated_test is not None:
            changes.append({"path": test_path, "content": updated_test})
        if not changes:
            return None

        return ToolAction(
            tool="write_file",
            reason="Compile the explicit multi-file code-and-test request into a bundled patch approval.",
            args={"changes": changes},
        )

    def _compile_small_write_action_from_task(
        self,
        *,
        task: str,
        workspace: Path,
    ) -> ToolAction | None:
        bundled = self._compile_python_function_and_unittest_bundle_action(task=task, workspace=workspace)
        if bundled is not None:
            return bundled

        target_path = self._extract_primary_file_target(task, workspace)
        if target_path is None:
            return None

        target = (workspace / target_path).resolve()
        try:
            file_text = target.read_text(encoding="utf-8")
        except Exception:
            return None

        for compiler in (
            self._compile_paragraph_insert_action,
            self._compile_assignment_update_action,
            self._compile_import_insert_action,
            self._compile_test_method_insert_action,
            self._compile_replace_all_action,
            self._compile_exact_replace_action,
            self._compile_anchor_insert_action,
            self._compile_append_action,
        ):
            compiled = compiler(task=task, target_path=target_path, file_text=file_text)
            if compiled is not None:
                return compiled
        return None

    def _compile_paragraph_insert_action(
        self,
        *,
        task: str,
        target_path: str,
        file_text: str,
    ) -> ToolAction | None:
        if "paragraph" not in task.lower():
            return None

        payload = self._extract_task_payload(task)
        anchor = self._extract_task_anchor(task)
        if payload is None or anchor is None:
            return None

        _, inserted_text = payload
        paragraph = self._find_paragraph_starting_with(file_text, anchor)
        if paragraph is None:
            return None

        updated_paragraph = f"{paragraph}\n\n{inserted_text}"
        if updated_paragraph in file_text or inserted_text in paragraph:
            return None

        return ToolAction(
            tool="replace_in_file",
            reason="Compile the explicit paragraph insertion into a deterministic patch approval.",
            args={
                "path": target_path,
                "old_text": paragraph,
                "new_text": updated_paragraph,
                "replace_all": False,
            },
        )

    def _compile_assignment_update_action(
        self,
        *,
        task: str,
        target_path: str,
        file_text: str,
    ) -> ToolAction | None:
        assignment = self._extract_assignment_update_values(task)
        if assignment is None:
            return None

        key, raw_value = assignment
        updated_text = self._build_assignment_updated_file_text(
            file_text=file_text,
            target_path=target_path,
            key=key,
            raw_value=raw_value,
        )
        if updated_text is None or updated_text == file_text:
            return None

        return ToolAction(
            tool="write_file",
            reason="Compile the explicit assignment update into a deterministic patch approval.",
            args={
                "path": target_path,
                "content": updated_text,
            },
        )

    def _compile_import_insert_action(
        self,
        *,
        task: str,
        target_path: str,
        file_text: str,
    ) -> ToolAction | None:
        if not target_path.endswith(".py"):
            return None

        import_statement = self._extract_import_statement(task)
        if import_statement is None:
            return None

        updated_text = self._build_python_import_inserted_text(
            file_text=file_text,
            import_statement=import_statement,
        )
        if updated_text is None or updated_text == file_text:
            return None

        return ToolAction(
            tool="write_file",
            reason="Compile the explicit import insertion into a deterministic patch approval.",
            args={
                "path": target_path,
                "content": updated_text,
            },
        )

    def _compile_test_method_insert_action(
        self,
        *,
        task: str,
        target_path: str,
        file_text: str,
    ) -> ToolAction | None:
        if not target_path.endswith(".py"):
            return None

        class_insert = self._extract_class_block_insert_values(task)
        if class_insert is None:
            return None

        class_name, block_text = class_insert
        updated_text = self._build_class_block_inserted_text(
            file_text=file_text,
            class_name=class_name,
            block_text=block_text,
        )
        if updated_text is None or updated_text == file_text:
            return None

        return ToolAction(
            tool="write_file",
            reason="Compile the explicit test-class insertion into a deterministic patch approval.",
            args={
                "path": target_path,
                "content": updated_text,
            },
        )

    def _compile_exact_replace_action(
        self,
        *,
        task: str,
        target_path: str,
        file_text: str,
    ) -> ToolAction | None:
        replace_values = self._extract_task_replace_values(task)
        if replace_values is None:
            return None

        old_text, new_text, replace_all = replace_values
        if replace_all or not old_text or old_text == new_text or old_text not in file_text:
            return None

        return ToolAction(
            tool="replace_in_file",
            reason="Compile the explicit replace request into a deterministic patch approval.",
            args={
                "path": target_path,
                "old_text": old_text,
                "new_text": new_text,
                "replace_all": False,
            },
        )

    def _compile_replace_all_action(
        self,
        *,
        task: str,
        target_path: str,
        file_text: str,
    ) -> ToolAction | None:
        replace_values = self._extract_task_replace_values(task)
        if replace_values is None:
            return None

        old_text, new_text, replace_all = replace_values
        if not replace_all or not old_text or old_text == new_text:
            return None

        occurrences = file_text.count(old_text)
        if occurrences < 1:
            return None

        return ToolAction(
            tool="replace_in_file",
            reason="Compile the explicit replace-all request into a deterministic patch approval.",
            args={
                "path": target_path,
                "old_text": old_text,
                "new_text": new_text,
                "replace_all": True,
            },
        )

    def _compile_anchor_insert_action(
        self,
        *,
        task: str,
        target_path: str,
        file_text: str,
    ) -> ToolAction | None:
        anchor_insert = self._extract_anchor_insert_values(task)
        if anchor_insert is None:
            return None

        kind, inserted_text, position, anchor = anchor_insert
        if anchor not in file_text:
            return None

        delimiter = self._insertion_delimiter(kind, inserted_text)
        if position == "before":
            replacement = f"{inserted_text}{delimiter}{anchor}"
        else:
            replacement = f"{anchor}{delimiter}{inserted_text}"

        if replacement in file_text:
            return None

        return ToolAction(
            tool="replace_in_file",
            reason="Compile the explicit anchored insertion into a deterministic patch approval.",
            args={
                "path": target_path,
                "old_text": anchor,
                "new_text": replacement,
                "replace_all": False,
            },
        )

    def _compile_append_action(
        self,
        *,
        task: str,
        target_path: str,
        file_text: str,
    ) -> ToolAction | None:
        append_values = self._extract_append_values(task)
        if append_values is None:
            return None

        kind, appended_text = append_values
        updated_text = self._build_appended_file_text(file_text=file_text, appended_text=appended_text, kind=kind)
        if updated_text is None or updated_text == file_text:
            return None

        return ToolAction(
            tool="write_file",
            reason="Compile the explicit append request into a deterministic patch approval.",
            args={
                "path": target_path,
                "content": updated_text,
            },
        )

    def _action_matches_explicit_write_task(
        self,
        action: ToolAction,
        *,
        task: str,
        workspace: Path,
    ) -> bool:
        if action.tool not in {"write_file", "replace_in_file"}:
            return False

        expected_action = self._compile_small_write_action_from_task(task=task, workspace=workspace)
        if expected_action is not None:
            return self._write_actions_match(action, expected_action, workspace)

        target_path = self._extract_primary_file_target(task, workspace)
        normalized_path = self._normalize_path_arg(action.args.get("path", "."), workspace)
        if target_path is not None and normalized_path != target_path:
            return False

        sentence = self._extract_task_sentence(task)
        if not sentence:
            return True

        if action.tool == "write_file":
            return sentence in str(action.args.get("content", ""))

        new_text = str(action.args.get("new_text", ""))
        return sentence in new_text

    def _write_actions_match(
        self,
        action: ToolAction,
        expected_action: ToolAction,
        workspace: Path,
    ) -> bool:
        if action.tool != expected_action.tool:
            return False

        actual_path = self._normalize_path_arg(action.args.get("path", "."), workspace)
        expected_path = self._normalize_path_arg(expected_action.args.get("path", "."), workspace)
        if actual_path != expected_path:
            return False

        if action.tool == "write_file":
            expected_bundle = self._normalize_write_bundle(expected_action.args, workspace)
            actual_bundle = self._normalize_write_bundle(action.args, workspace)
            if expected_bundle is not None or actual_bundle is not None:
                return actual_bundle == expected_bundle
            return str(action.args.get("content", "")) == str(expected_action.args.get("content", ""))

        return (
            str(action.args.get("old_text", "")) == str(expected_action.args.get("old_text", ""))
            and str(action.args.get("new_text", "")) == str(expected_action.args.get("new_text", ""))
            and bool(action.args.get("replace_all", False)) == bool(expected_action.args.get("replace_all", False))
        )

    def _normalize_write_bundle(
        self,
        args: dict[str, object],
        workspace: Path,
    ) -> list[tuple[str, str]] | None:
        raw_changes = args.get("changes")
        if not isinstance(raw_changes, list):
            return None
        normalized: list[tuple[str, str]] = []
        for entry in raw_changes:
            if not isinstance(entry, dict):
                continue
            path = self._normalize_path_arg(entry.get("path", "."), workspace)
            normalized.append((path, str(entry.get("content", ""))))
        return normalized

    def _extract_file_targets(self, task: str, workspace: Path) -> list[str]:
        targets: list[str] = []
        for candidate in self._extract_candidate_paths(task):
            normalized = self._normalize_path_arg(candidate, workspace)
            resolved = (workspace / normalized).resolve()
            if resolved.exists() and resolved.is_file() and normalized not in targets:
                targets.append(normalized)
        return targets

    def _extract_primary_file_target(self, task: str, workspace: Path) -> str | None:
        targets = self._extract_file_targets(task, workspace)
        return targets[0] if targets else None

    @staticmethod
    def _extract_function_update_target(task: str) -> str | None:
        match = re.search(r"\bupdate\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+so it\b", task, flags=re.IGNORECASE)
        if match:
            return match.group("name").strip()
        return None

    @staticmethod
    def _extract_python_string_normalizer_template(task: str) -> str | None:
        lowered = task.lower()
        trims_whitespace = (
            "trim whitespace" in lowered
            or "trims whitespace" in lowered
            or "trim surrounding whitespace" in lowered
        )
        title_cases = (
            "title-cases each word" in lowered
            or "title-case each word" in lowered
            or "title cases each word" in lowered
            or "title case each word" in lowered
        )
        if trims_whitespace and title_cases:
            return '" ".join(part.capitalize() for part in {param}.split())'
        return None

    @staticmethod
    def _extract_unittest_io_expectation(task: str) -> tuple[str, str] | None:
        match = re.search(
            r"proves?\s+(?P<input_quote>['\"`])(?P<input>.+?)(?P=input_quote)\s+becomes\s+"
            r"(?P<output_quote>['\"`])(?P<output>.+?)(?P=output_quote)",
            task,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group("input"), match.group("output")
        return None

    @staticmethod
    def _build_python_function_return_update(
        *,
        file_text: str,
        function_name: str,
        return_template: str,
    ) -> str | None:
        lines = file_text.splitlines(keepends=True)
        function_pattern = re.compile(
            rf"^(?P<indent>\s*)def\s+{re.escape(function_name)}\s*\((?P<params>[^)]*)\)\s*(?:->\s*[^:]+)?:\s*$"
        )
        for index, line in enumerate(lines):
            match = function_pattern.match(line.rstrip("\n"))
            if not match:
                continue
            indent = match.group("indent")
            params = match.group("params")
            first_param = params.split(",", 1)[0].strip() if params.strip() else "value"
            param_name = first_param.split(":", 1)[0].split("=", 1)[0].strip() or "value"
            body_start = index + 1
            body_end = len(lines)
            for candidate_index in range(body_start, len(lines)):
                stripped = lines[candidate_index].strip()
                if not stripped:
                    continue
                current_indent = len(lines[candidate_index]) - len(lines[candidate_index].lstrip(" "))
                if current_indent <= len(indent):
                    body_end = candidate_index
                    break
            body_indent = f"{indent}    "
            new_body_line = f"{body_indent}return {return_template.format(param=param_name)}\n"
            existing_body = "".join(lines[body_start:body_end]).strip()
            if existing_body == new_body_line.strip():
                return None
            return "".join([*lines[:body_start], new_body_line, *lines[body_end:]])
        return None

    def _build_python_unittest_expectation_update(
        self,
        *,
        file_text: str,
        function_name: str,
        input_value: str,
        output_value: str,
    ) -> str | None:
        assertion = f"self.assertEqual({function_name}({input_value!r}), {output_value!r})"
        if assertion in file_text:
            return None

        class_match = re.search(
            r"^class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?:.*\.)?TestCase\):\s*$",
            file_text,
            flags=re.MULTILINE,
        )
        if class_match is None:
            return None

        method_name = f"test_{function_name}_normalizes_whitespace_and_title_case"
        block_text = (
            f"def {method_name}(self) -> None:\n"
            f"    {assertion}"
        )
        return self._build_class_block_inserted_text(
            file_text=file_text,
            class_name=class_match.group("name"),
            block_text=block_text,
        )

    @staticmethod
    def _strip_read_file_line_numbers(text: str) -> str:
        return "\n".join(re.sub(r"^\d{4}:\s?", "", line) for line in text.splitlines())

    @staticmethod
    def _extract_task_sentence(task: str) -> str | None:
        match = re.search(r"(?:sentence|line|text)\s+['\"]([^'\"]+)['\"]", task, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _extract_task_payload(task: str) -> tuple[str, str] | None:
        fenced_block = ClosedLoopSupervisor._extract_task_fenced_block(task)
        if fenced_block is not None:
            return "block", fenced_block

        match = re.search(
            r"(sentence|line|text)\s+(?P<quote>['\"`])(?P<payload>.+?)(?P=quote)",
            task,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).lower(), match.group("payload").strip()

        match = re.search(
            r"(?:append|insert|add)\s+(?:the\s+)?(?:exact\s+|literal\s+|verbatim\s+)?"
            r"(?P<quote>['\"`])(?P<payload>.+?)(?P=quote)",
            task,
            flags=re.IGNORECASE,
        )
        if match:
            return "text", match.group("payload").strip()
        return None

    @staticmethod
    def _extract_task_replace_values(task: str) -> tuple[str, str, bool] | None:
        match = re.search(
            r"replace(?P<all>\s+all(?:\s+occurrences?)?\s+of|\s+every\s+occurrence\s+of)?(?:\s+the\s+(?:text|line|sentence))?\s+"
            r"(?P<old_quote>['\"`])(?P<old>.+?)(?P=old_quote)\s+with\s+"
            r"(?P<new_quote>['\"`])(?P<new>.+?)(?P=new_quote)",
            task,
            flags=re.IGNORECASE,
        )
        if match:
            return (
                match.group("old").strip(),
                match.group("new").strip(),
                bool(match.group("all")),
            )
        return None

    @staticmethod
    def _extract_import_statement(task: str) -> str | None:
        match = re.search(
            r"(?:add|insert)\s+(?:the\s+)?(?:(?:import|statement)\s+)?"
            r"(?P<quote>['\"`])(?P<statement>(?:from\s+[^\n]+?\s+import\s+[^\n]+?|import\s+[^\n]+?))(?P=quote)\s+"
            r"(?:to|into)\b",
            task,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group("statement").strip()
        return None

    @staticmethod
    def _extract_assignment_update_values(task: str) -> tuple[str, str] | None:
        match = re.search(
            r"\b(?:set|change|update)\s+(?P<name>[A-Za-z_][\w.-]*)\s*(?:=|to)\s*(?P<value>.+?)\s+(?:in|inside)\s+[\w./-]+\b",
            task,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        value = match.group("value").strip().rstrip(".,")
        return match.group("name").strip(), value

    @staticmethod
    def _extract_anchor_insert_values(task: str) -> tuple[str, str, str, str] | None:
        fenced_block = ClosedLoopSupervisor._extract_task_fenced_block(task)
        if fenced_block is not None:
            match = re.search(
                r"(?:insert|add)\s+(?:the\s+)?(?:following\s+)?block\s+"
                r"(?:immediately\s+)?(?P<position>before|after)\s+(?:the\s+)?"
                r"(?:(?:sentence|line|text|block)\s+)?(?P<anchor_quote>['\"`])(?P<anchor>.+?)(?P=anchor_quote)",
                task,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match:
                return (
                    "block",
                    fenced_block,
                    match.group("position").lower(),
                    match.group("anchor").strip(),
                )

        match = re.search(
            r"(?:insert|add)\s+(?:the\s+)?(?:(?P<kind>sentence|line|text)\s+)?"
            r"(?P<payload_quote>['\"`])(?P<payload>.+?)(?P=payload_quote)\s+"
            r"(?:immediately\s+)?(?P<position>before|after)\s+(?:the\s+)?"
            r"(?:(?:sentence|line|text)\s+)?(?P<anchor_quote>['\"`])(?P<anchor>.+?)(?P=anchor_quote)",
            task,
            flags=re.IGNORECASE,
        )
        if match:
            kind = (match.group("kind") or "text").lower()
            return (
                kind,
                match.group("payload").strip(),
                match.group("position").lower(),
                match.group("anchor").strip(),
            )
        return None

    @staticmethod
    def _extract_class_block_insert_values(task: str) -> tuple[str, str] | None:
        fenced_block = ClosedLoopSupervisor._extract_task_fenced_block(task)
        if fenced_block is None:
            return None

        match = re.search(
            r"(?:add|insert)\s+(?:the\s+)?(?:following\s+)?(?:test|method|block)\s+to\s+class\s+(?P<class_name>[A-Za-z_][A-Za-z0-9_]*)\b",
            task,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group("class_name").strip(), fenced_block
        return None

    @staticmethod
    def _extract_append_values(task: str) -> tuple[str, str] | None:
        fenced_block = ClosedLoopSupervisor._extract_task_fenced_block(task)
        if fenced_block is not None:
            match = re.search(
                r"append\s+(?:the\s+)?(?:following\s+)?block\s+(?:to|at\s+the\s+end\s+of)\b",
                task,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if match:
                return "block", fenced_block

        match = re.search(
            r"append\s+(?:the\s+)?(?:exact\s+|literal\s+|verbatim\s+)?(?:(?P<kind>sentence|line|text)\s+)?"
            r"(?P<quote>['\"`])(?P<payload>.+?)(?P=quote)\s+"
            r"(?:to|at\s+the\s+end\s+of)\b",
            task,
            flags=re.IGNORECASE,
        )
        if match:
            return (match.group("kind") or "text").lower(), match.group("payload").strip()
        return None

    @staticmethod
    def _extract_task_fenced_block(task: str) -> str | None:
        match = re.search(
            r"```(?:[\w.+-]+)?\n(?P<block>[\s\S]+?)```",
            task,
            flags=re.MULTILINE,
        )
        if match:
            return match.group("block").rstrip("\n")
        return None

    @staticmethod
    def _insertion_delimiter(kind: str, inserted_text: str) -> str:
        if "\n" in inserted_text or kind == "block":
            return "\n"
        if kind == "line":
            return "\n"
        if kind == "sentence":
            return " "
        return ""

    @staticmethod
    def _build_appended_file_text(
        *,
        file_text: str,
        appended_text: str,
        kind: str,
    ) -> str | None:
        if kind == "block":
            normalized_appended = appended_text.rstrip("\n")
            if normalized_appended and normalized_appended in file_text:
                return None
            separator = ""
            if file_text and not file_text.endswith("\n"):
                separator = "\n"
            elif file_text and not file_text.endswith("\n\n"):
                separator = "\n"
            updated = f"{file_text}{separator}{normalized_appended}"
            return updated if updated.endswith("\n") else f"{updated}\n"

        if kind == "line":
            if file_text.rstrip("\n").endswith(appended_text):
                return None
            prefix = "" if not file_text or file_text.endswith("\n") else "\n"
            updated = f"{file_text}{prefix}{appended_text}"
            return updated if updated.endswith("\n") else f"{updated}\n"

        if kind == "sentence":
            if file_text.rstrip().endswith(appended_text):
                return None
            if not file_text:
                return appended_text
            separator = "\n" if file_text.endswith("\n") else " "
            return f"{file_text}{separator}{appended_text}"

        if file_text.endswith(appended_text):
            return None
        return f"{file_text}{appended_text}"

    @staticmethod
    def _build_python_import_inserted_text(
        *,
        file_text: str,
        import_statement: str,
    ) -> str | None:
        lines = file_text.splitlines(keepends=True)
        normalized_statement = import_statement.strip()
        existing_lines = {line.strip() for line in lines}
        if normalized_statement in existing_lines:
            return None

        insert_at = 0
        if insert_at < len(lines) and lines[insert_at].startswith("#!"):
            insert_at += 1
        if insert_at < len(lines) and re.search(r"coding[:=]", lines[insert_at]):
            insert_at += 1

        while insert_at < len(lines) and not lines[insert_at].strip():
            insert_at += 1

        if insert_at < len(lines):
            stripped = lines[insert_at].lstrip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = stripped[:3]
                insert_at += 1
                if quote not in stripped[3:]:
                    while insert_at < len(lines):
                        if quote in lines[insert_at]:
                            insert_at += 1
                            break
                        insert_at += 1
                while insert_at < len(lines) and not lines[insert_at].strip():
                    insert_at += 1

        import_insert_at: int | None = None
        last_import = -1
        scan_index = insert_at
        while scan_index < len(lines):
            stripped = lines[scan_index].strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                last_import = scan_index
                scan_index += 1
                continue
            if last_import >= 0 and (not stripped or stripped.startswith("#")):
                scan_index += 1
                continue
            break
        if last_import >= 0:
            import_insert_at = last_import + 1

        insertion_index = import_insert_at if import_insert_at is not None else insert_at
        inserted_lines = [f"{normalized_statement}\n"]
        if import_insert_at is None and insertion_index < len(lines) and lines[insertion_index].strip():
            inserted_lines.append("\n")

        return "".join(lines[:insertion_index] + inserted_lines + lines[insertion_index:])

    @staticmethod
    def _build_assignment_updated_file_text(
        *,
        file_text: str,
        target_path: str,
        key: str,
        raw_value: str,
    ) -> str | None:
        suffix = Path(target_path).suffix.lower()
        is_env_style = target_path.startswith(".env") or suffix == ".env"
        if is_env_style:
            separators = ["="]
        elif suffix in {".yaml", ".yml"}:
            separators = [":"]
        else:
            separators = ["=", ":"]

        lines = file_text.splitlines(keepends=True)
        for index, line in enumerate(lines):
            line_without_newline = line.rstrip("\n")
            for separator in separators:
                pattern = re.compile(
                    rf"^(?P<prefix>\s*{re.escape(key)}\s*{re.escape(separator)}\s*)(?P<value>.*?)(?P<comment>\s+#.*)?$"
                )
                match = pattern.match(line_without_newline)
                if not match:
                    continue
                existing_value = match.group("value").rstrip()
                replacement_value = ClosedLoopSupervisor._normalize_assignment_value(
                    raw_value=raw_value,
                    existing_value=existing_value,
                    separator=separator,
                    target_path=target_path,
                )
                updated_line = f"{match.group('prefix')}{replacement_value}{match.group('comment') or ''}"
                newline = "\n" if line.endswith("\n") else ""
                new_lines = lines[:]
                new_lines[index] = f"{updated_line}{newline}"
                return "".join(new_lines)

        if is_env_style:
            replacement_value = ClosedLoopSupervisor._normalize_assignment_value(
                raw_value=raw_value,
                existing_value="",
                separator="=",
                target_path=target_path,
            )
            appended_line = f"{key}={replacement_value}"
            existing_lines = {line.strip() for line in file_text.splitlines()}
            if appended_line in existing_lines:
                return None
            separator = "" if not file_text or file_text.endswith("\n") else "\n"
            updated_text = f"{file_text}{separator}{appended_line}"
            return updated_text if updated_text.endswith("\n") else f"{updated_text}\n"
        return None

    @staticmethod
    def _normalize_assignment_value(
        *,
        raw_value: str,
        existing_value: str,
        separator: str,
        target_path: str,
    ) -> str:
        candidate = raw_value.strip()
        if not candidate:
            return candidate

        if (
            (candidate.startswith('"') and candidate.endswith('"'))
            or (candidate.startswith("'") and candidate.endswith("'"))
        ):
            return candidate

        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", candidate):
            return candidate

        lowered = candidate.lower()
        if lowered in {"true", "false", "none", "null"}:
            if separator == "=" and target_path.endswith(".py"):
                return lowered.capitalize() if lowered in {"true", "false", "none"} else "None"
            return lowered

        if existing_value[:1] in {'"', "'"} and existing_value[-1:] == existing_value[:1]:
            quote = existing_value[:1]
            return f"{quote}{candidate}{quote}"

        return candidate

    @staticmethod
    def _build_class_block_inserted_text(
        *,
        file_text: str,
        class_name: str,
        block_text: str,
    ) -> str | None:
        lines = file_text.splitlines(keepends=True)
        class_pattern = re.compile(rf"^(?P<indent>\s*)class\s+{re.escape(class_name)}\b.*:\s*$")

        class_index = -1
        class_indent = ""
        for index, line in enumerate(lines):
            match = class_pattern.match(line.rstrip("\n"))
            if match:
                class_index = index
                class_indent = match.group("indent")
                break
        if class_index < 0:
            return None

        class_end = len(lines)
        for index in range(class_index + 1, len(lines)):
            stripped = lines[index].strip()
            if not stripped:
                continue
            current_indent = len(lines[index]) - len(lines[index].lstrip(" "))
            if current_indent <= len(class_indent):
                class_end = index
                break

        dedented_block = textwrap.dedent(block_text).strip("\n")
        if not dedented_block:
            return None

        body_indent = f"{class_indent}    "
        normalized_lines = []
        for line in dedented_block.splitlines():
            if line.strip():
                normalized_lines.append(f"{body_indent}{line.rstrip()}\n")
            else:
                normalized_lines.append("\n")
        normalized_block = "".join(normalized_lines).rstrip("\n")
        class_slice = "".join(lines[class_index:class_end])
        if normalized_block in class_slice:
            return None

        prefix = "".join(lines[:class_end])
        suffix = "".join(lines[class_end:])
        separator_before = ""
        if prefix and not prefix.endswith("\n\n"):
            separator_before = "\n" if prefix.endswith("\n") else "\n\n"
        separator_after = ""
        if suffix and not suffix.startswith("\n"):
            separator_after = "\n"

        return f"{prefix}{separator_before}{normalized_block}\n{separator_after}{suffix}"

    @staticmethod
    def _extract_task_anchor(task: str) -> str | None:
        match = re.search(r"(?:starts|begins)\s+with\s+['\"]([^'\"]+)['\"]", task, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return None

    @staticmethod
    def _find_paragraph_starting_with(text: str, anchor: str) -> str | None:
        normalized_anchor = ClosedLoopSupervisor._normalize_paragraph_anchor(anchor)
        for paragraph in re.split(r"\n\s*\n", text):
            candidate = paragraph.strip()
            if ClosedLoopSupervisor._normalize_paragraph_anchor(candidate).startswith(normalized_anchor):
                return candidate
        return None

    @staticmethod
    def _normalize_paragraph_anchor(text: str) -> str:
        collapsed = re.sub(r"\s+", " ", text.replace("`", " ").strip().lower())
        return collapsed.strip()

    def _maybe_synthesize_repository_answer(
        self,
        *,
        task: str,
        rounds: list[RoundRecord],
        workspace: Path,
        allow_partial: bool = False,
    ) -> str | None:
        if not self._is_repository_inspection_task(task):
            return None

        successful = self._successful_action_signatures(rounds, workspace)
        has_readme = "read_file:README.md" in successful
        has_package_listing = "list_files:teamai" in successful
        has_config = "read_file:teamai/config.py" in successful
        has_runtime_anchor = "read_file:teamai/cli.py" in successful or "read_file:teamai/supervisor.py" in successful

        strict_ready = has_readme and has_package_listing and has_config and has_runtime_anchor
        partial_ready = has_readme and has_runtime_anchor and (has_package_listing or has_config)
        if not strict_ready:
            if not allow_partial or not partial_ready:
                return None

        contents = self._collect_read_file_outputs(rounds, workspace)
        readme_text = contents.get("README.md", "")
        pyproject_text = contents.get("pyproject.toml", "")
        memory_text = contents.get("PROJECT_MEMORY.md", "")
        combined_text = "\n".join([readme_text, pyproject_text, memory_text]).lower()
        implemented = self._implemented_feature_flags(workspace, contents)

        current_state_parts: list[str] = []
        if has_readme and pyproject_text:
            current_state_parts.append(
                "The repo is already packaged as a local-first Python application for MLX-based orchestration."
            )
        if "teamai/supervisor.py" in contents:
            current_state_parts.append(
                "The core strategist/critic/planner/verifier loop is implemented in the supervisor."
            )
        if "teamai/model_backend.py" in contents:
            current_state_parts.append(
                "The MLX backend is wired in with lazy model loading and explicit load/generation error handling."
            )
        if "teamai/tools.py" in contents:
            current_state_parts.append(
                "Workspace inspection tools and write restrictions are already scaffolded behind a sandboxed tool layer."
            )
        if "teamai/config.py" in contents:
            current_state_parts.append(
                "Runtime settings are centralized with environment validation, workspace scoping, and safety limits."
            )
        if implemented["persistent_memory"]:
            current_state_parts.append(
                "Persistent run history and cross-run workspace memory are already implemented through `.teamai/` state files and prompt injection."
            )

        tasks: list[str] = []
        if (
            any(keyword in combined_text for keyword in ["persistent memory", "persistent run history", "run history"])
            and not implemented["persistent_memory"]
        ):
            tasks.append(
                "Add persistent run history and memory so the loop can carry useful context across rounds and across separate runs."
            )

        if any(
            keyword in combined_text
            for keyword in [
                "patch-oriented editing tools",
                "safer patch-oriented write tools",
                "approval checkpoints",
                "before destructive changes",
            ]
        ):
            tasks.append(
                "Replace the coarse write path with patch-oriented editing tools and approval checkpoints before destructive changes."
            )

        if any(keyword in combined_text for keyword in ["streaming events", "streaming event output", "streaming output"]):
            tasks.append(
                "Add streaming event output across the CLI, API, and job flow so long runs expose persona and tool progress in real time."
            )

        if "teamai/model_backend.py" in contents:
            tasks.append(
                "Harden the MLX backend around model load, generation failures, and clearer operator-facing recovery guidance."
            )

        if any(keyword in combined_text for keyword in ["json planning / verification", "json planning", "verification"]):
            tasks.append(
                "Keep hardening structured planner and verifier outputs so longer inspection and execution runs stay reliable."
            )

        if not tasks:
            tasks.append(
                "Inspect and tighten `teamai/supervisor.py`, since it appears to be the main coordination point for the closed-loop behavior."
            )
            tasks.append(
                "Inspect and harden `teamai/model_backend.py`, since the MLX integration is the highest-risk runtime dependency after the CLI/config layer."
            )
            tasks.append(
                "Add higher-level tests for full repository-inspection runs so the system proves it can reach actionable next steps."
            )

        deduped_tasks: list[str] = []
        for task_item in tasks:
            if task_item not in deduped_tasks:
                deduped_tasks.append(task_item)

        current_state = " ".join(current_state_parts) if current_state_parts else "The repo structure and core runtime pieces are in place."
        tasks_section = "\n".join(f"- {task_item}" for task_item in deduped_tasks[:4])
        return f"Current state: {current_state}\n\nNext engineering tasks:\n{tasks_section}"

    def _maybe_synthesize_codex_handoff_answer(
        self,
        *,
        task: str,
        rounds: list[RoundRecord],
        workspace: Path,
        task_route: str,
    ) -> str | None:
        if task_route != "codex_handoff":
            return None

        raw_key_paths: list[str] = []
        seen_paths: set[str] = set()
        next_focuses: list[str] = []
        seen_focuses: set[str] = set()
        evidence_count = 0

        for record in rounds:
            for result in record.tool_results:
                if not result.success:
                    continue
                evidence_count += 1
                raw_path = str(result.metadata.get("path", "")).strip()
                if raw_path:
                    normalized = self._normalize_path_arg(raw_path, workspace)
                    if normalized not in seen_paths:
                        seen_paths.add(normalized)
                        raw_key_paths.append(normalized)
            next_focus = (record.verifier.next_focus or "").strip()
            if next_focus and next_focus not in seen_focuses:
                seen_focuses.add(next_focus)
                next_focuses.append(next_focus.rstrip("."))

        if len(raw_key_paths) < 2:
            fallback_paths = self._rank_codex_handoff_paths(
                task=task,
                paths=self._task_relevant_candidates(task, workspace),
            )
            for candidate in fallback_paths:
                if candidate in seen_paths:
                    continue
                seen_paths.add(candidate)
                raw_key_paths.append(candidate)
                if len(raw_key_paths) >= 8:
                    break

        key_paths = self._rank_codex_handoff_paths(task=task, paths=raw_key_paths)
        if len(key_paths) < 2:
            for candidate in self._rank_codex_handoff_paths(
                task=task,
                paths=self._task_relevant_candidates(task, workspace),
            ):
                if candidate in key_paths:
                    continue
                key_paths.append(candidate)
                if len(key_paths) >= 4:
                    break
        lead_task = None
        for focus in next_focuses:
            normalized_focus = self._normalize_codex_handoff_focus(focus, workspace=workspace)
            if normalized_focus:
                lead_task = normalized_focus
                break

        if evidence_count < 2 and not (key_paths or lead_task):
            return None

        lines = [
            "Current state: The local run treated this as a broad coding task and gathered reconnaissance instead of attempting autonomous implementation.",
            "",
            "Next engineering tasks:",
        ]
        if lead_task:
            lines.append(f"- {self._ensure_sentence(lead_task)}")
        if key_paths and not self._lead_task_covers_paths(lead_task, key_paths[:2]):
            lines.append(f"- Inspect the most relevant paths first: {', '.join(key_paths[:4])}.")
        lines.append(f"- Implement the requested change in Codex after verifying the scoped plan for: {task}")
        return "\n".join(lines)

    def _normalize_codex_handoff_focus(
        self,
        focus: str,
        *,
        workspace: Path,
    ) -> str | None:
        compact = " ".join(focus.split()).strip().rstrip(".")
        if not compact:
            return None

        lowered = compact.lower()
        normalized_paths: list[str] = []
        seen_paths: set[str] = set()
        for candidate in self._extract_candidate_paths(compact):
            normalized = self._normalize_path_arg(candidate, workspace)
            resolved = (workspace / normalized).resolve()
            if not resolved.exists() or normalized in seen_paths:
                continue
            seen_paths.add(normalized)
            normalized_paths.append(normalized)

        if normalized_paths:
            if len(normalized_paths) == 1:
                return f"Inspect {normalized_paths[0]} before implementing the requested change"
            if len(normalized_paths) == 2:
                return (
                    f"Inspect {normalized_paths[0]} and {normalized_paths[1]} "
                    "before implementing the requested change"
                )
            return f"Inspect the most relevant paths first: {', '.join(normalized_paths[:4])}"

        if lowered.startswith(("inspect ", "review ", "read ", "trace ", "verify ", "map ", "compare ", "reproduce ")):
            return compact
        if lowered.startswith(("implement ", "fix ", "debug ", "update ")):
            return compact
        return None

    @staticmethod
    def _ensure_sentence(text: str) -> str:
        stripped = text.strip()
        if stripped.endswith((".", "!", "?")):
            return stripped
        return f"{stripped}."

    @staticmethod
    def _lead_task_covers_paths(lead_task: str | None, paths: list[str]) -> bool:
        if not lead_task:
            return False
        lowered = lead_task.lower()
        return all(path.lower() in lowered for path in paths if path)

    def _rank_codex_handoff_paths(self, *, task: str, paths: list[str]) -> list[str]:
        task_lower = task.lower()
        unique_paths: list[str] = []
        seen: set[str] = set()
        for path in paths:
            normalized = path.rstrip("/")
            if normalized in seen:
                continue
            seen.add(normalized)
            unique_paths.append(normalized)

        file_paths = [path for path in unique_paths if path not in {".", "teamai", "tests"} and "." in Path(path).name]
        preferred_pool = file_paths or [path for path in unique_paths if path != "."]

        ranked = sorted(
            preferred_pool,
            key=lambda path: (-self._score_handoff_path(task_lower=task_lower, path=path), unique_paths.index(path)),
        )
        return ranked[:12]

    @staticmethod
    def _score_handoff_path(*, task_lower: str, path: str) -> int:
        path_lower = path.lower()
        score = 0
        if any(marker in task_lower for marker in ["inspect", "explore", "identify", "next tasks", "broad"]):
            if path_lower == "readme.md":
                score += 10
            elif path_lower == "teamai/config.py":
                score += 8
            elif path_lower == "project_memory.md":
                score += 6
        if any(marker in task_lower for marker in ["bridge", "handoff", "terminal"]) and any(
            marker in path_lower for marker in ["bridge.py", "handoff.py", "test_bridge.py", "test_handoff.py"]
        ):
            score += 6
        if any(marker in task_lower for marker in ["memory", "history", "persist", "cross-run"]) and any(
            marker in path_lower for marker in ["memory.py", "test_memory.py", "prompts.py"]
        ):
            score += 6
        if any(
            marker in task_lower
            for marker in [
                "learned-note",
                "learned note",
                "improvement note",
                "self-improvement",
                "self improvement",
                "decay",
                "prune",
                "pruning",
                "stale lesson",
                "stale note",
            ]
        ):
            if path_lower.endswith("teamai/memory.py"):
                score += 10
            elif path_lower.endswith("tests/test_memory.py"):
                score += 8
            elif any(
                marker in path_lower
                for marker in ["prompts.py", "handoff.py", "bridge.py", "supervisor.py"]
            ):
                score += 4
        if any(marker in task_lower for marker in ["stream", "streaming", "event output", "progress output"]) and any(
            marker in path_lower for marker in ["cli.py", "api.py", "jobs.py", "schemas.py", "supervisor.py"]
        ):
            score += 6
        if any(marker in task_lower for marker in ["approval", "patch", "write path", "workspace_write", "deterministic"]) and any(
            marker in path_lower for marker in ["tools.py", "approvals.py", "supervisor.py", "test_tools.py", "test_approvals.py", "test_supervisor.py"]
        ):
            score += 6
        if any(marker in task_lower for marker in ["json", "planner", "verifier", "prompt", "structured output"]) and any(
            marker in path_lower for marker in ["prompts.py", "schemas.py", "supervisor.py", "test_supervisor.py"]
        ):
            score += 5
        return score

    def _build_local_drift_handoff_answer(
        self,
        *,
        task: str,
        rounds: list[RoundRecord],
        workspace: Path,
        reroute_reason: str,
    ) -> str:
        synthesized = self._maybe_synthesize_codex_handoff_answer(
            task=task,
            rounds=rounds,
            workspace=workspace,
            task_route="codex_handoff",
        )
        if synthesized:
            return synthesized.replace(
                "Current state: The local run treated this as a broad coding task and gathered reconnaissance instead of attempting autonomous implementation.",
                "Current state: The local run started locally, gathered partial evidence, and then rerouted after it started drifting beyond the reliable local path.",
            )

        relevant_paths = self._task_relevant_candidates(task, workspace)
        lines = [
            "Current state: The local run started locally but rerouted after it stopped making reliable scoped progress.",
            "",
            "Next engineering tasks:",
            f"- {reroute_reason}",
        ]
        if relevant_paths:
            lines.append(f"- Inspect the most relevant paths first: {', '.join(relevant_paths[:4])}.")
        lines.append(f"- Continue the requested change in Codex after verifying the scoped plan for: {task}")
        return "\n".join(lines)

    def _implemented_feature_flags(
        self,
        workspace: Path,
        contents: dict[str, str],
    ) -> dict[str, bool]:
        return {
            "persistent_memory": (
                self._workspace_text_contains(
                    workspace,
                    contents,
                    "teamai/memory.py",
                    ["WorkspaceMemoryStore", "RUN_HISTORY_FILE_NAME", "MEMORY_FILE_NAME"],
                )
                and self._workspace_text_contains(
                    workspace,
                    contents,
                    "teamai/supervisor.py",
                    ["WorkspaceMemoryStore", "load_snapshot", "persist_run"],
                )
                and self._workspace_text_contains(
                    workspace,
                    contents,
                    "teamai/prompts.py",
                    ["Persistent workspace memory:", "Recent persisted runs:"],
                )
            ),
        }

    def _collect_read_file_outputs(
        self,
        rounds: list[RoundRecord],
        workspace: Path,
    ) -> dict[str, str]:
        outputs: dict[str, str] = {}
        for record in rounds:
            for action, result in zip(record.planner.actions, record.tool_results):
                if action.tool != "read_file" or not result.success:
                    continue
                path = self._normalize_path_arg(action.args.get("path", "."), workspace)
                outputs[path] = result.output
        return outputs

    @staticmethod
    def _workspace_text_contains(
        workspace: Path,
        contents: dict[str, str],
        path: str,
        needles: list[str],
    ) -> bool:
        text = contents.get(path)
        if text is not None and all(needle in text for needle in needles):
            return True

        candidate = workspace / path
        if not candidate.exists() or not candidate.is_file():
            return False
        try:
            file_text = candidate.read_text(encoding="utf-8")
        except Exception:
            return False
        return all(needle in file_text for needle in needles)

    @staticmethod
    def _extract_candidate_paths(text: str) -> list[str]:
        patterns = [
            re.compile(r"`([^`]+)`"),
            re.compile(r"'([^']+)'"),
            re.compile(r'"([^"]+)"'),
        ]
        candidates: list[str] = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                if not value:
                    continue
                if "/" in value or "." in value:
                    candidates.append(value)

        unquoted_path_pattern = re.compile(
            r"(?<![`'\"\w])"
            r"((?:\.[A-Za-z0-9_-]+|[A-Za-z0-9_-]+)(?:/[A-Za-z0-9._-]+)+/?|(?:\.[A-Za-z0-9_-]+|[A-Za-z0-9_-]+)\.[A-Za-z0-9_-]+)"
        )
        for match in unquoted_path_pattern.finditer(text):
            candidates.append(match.group(1).strip())

        for common in [
            "README.md",
            "pyproject.toml",
            "setup.py",
            "PROJECT_MEMORY.md",
            ".env",
            ".env.example",
            "teamai/",
            "teamai/model_backend.py",
            "teamai/supervisor.py",
            "teamai/tools.py",
            "teamai/api.py",
            "teamai/cli.py",
            "tests/",
        ]:
            if common in text:
                candidates.append(common)

        ordered: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.rstrip(",.:;")
            if normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)
        return ordered

    def _candidate_to_action(self, candidate: str, text: str, workspace: Path) -> ToolAction | None:
        normalized = candidate.strip().strip("`'\"")
        if not normalized:
            return None
        if not self._looks_like_candidate_path(normalized):
            return None
        try:
            path = Path(normalized.rstrip("/"))
            if not path.is_absolute():
                resolved = (workspace / path).resolve()
            else:
                resolved = path.resolve()
            if not resolved.exists():
                return None
        except OSError:
            return None
        except ValueError:
            return None

        text_lower = text.lower()
        relative = self._normalize_path_arg(normalized, workspace)
        if resolved.is_dir():
            return ToolAction(
                tool="list_files",
                reason=f"Inspect directory structure for `{relative}`.",
                args={"path": relative},
            )
        if "search" in text_lower and normalized in text:
            return ToolAction(
                tool="read_file",
                reason=f"Read `{relative}` to inspect the referenced content directly.",
                args={"path": relative},
            )
        return ToolAction(
            tool="read_file",
            reason=f"Read `{relative}` because it was explicitly referenced in planning context.",
            args={"path": relative},
        )

    @staticmethod
    def _looks_like_candidate_path(candidate: str) -> bool:
        if len(candidate) > 240:
            return False
        if any(char in candidate for char in "\n\r\t{}[]"):
            return False
        if ": " in candidate:
            return False
        return True
    def _log_telemetry(self, rounds: list[RoundRecord], task_route: str) -> None:
        if os.getenv("TEAMAI_TELEMETRY") != "1":
            return
        try:
            log_dir = Path("LOGS")
            log_dir.mkdir(exist_ok=True)
            log_file = log_dir / "telemetry.jsonl"

            tools_used = [res.tool for r in rounds for res in r.tool_results]
            total = len(tools_used)
            if total == 0:
                mix = {"search_text": 0.0, "read_file": 0.0, "list_files": 0.0}
            else:
                mix = {
                    "search_text": round(tools_used.count("search_text") / total * 100, 1),
                    "read_file": round(tools_used.count("read_file") / total * 100, 1),
                    "list_files": round(tools_used.count("list_files") / total * 100, 1),
                }

            unique_files = len({res.metadata.get("path") for r in rounds for res in r.tool_results if "path" in res.metadata})

            final_conf = 0.0
            if rounds and getattr(rounds[-1], "verifier", None):
                final_conf = getattr(rounds[-1].verifier, "confidence", 0.0)
            json_repair_count = sum(
                max(int(getattr(record, "planner_json_repairs", 0) or 0), 0)
                + max(int(getattr(record, "verifier_json_repairs", 0) or 0), 0)
                for record in rounds
            )
            structured_response_count = sum(2 for record in rounds if record.reasoning_source == "model")

            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "task_route": task_route,
                "total_rounds": len(rounds),
                "tool_mix": mix,
                "unique_files_touched": unique_files,
                "synthesis_confidence": final_conf,
                "json_repair_count": json_repair_count,
                "structured_response_count": structured_response_count,
                "json_repair_rate": (
                    json_repair_count / structured_response_count if structured_response_count else 0.0
                ),
            }

            with log_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass
