"""
Context assembly for teamAI.

Extracted from supervisor.py.  Builds the textual context windows fed to
the planner and verifier model calls, plus the deterministic
continuation-probe round that verifies approved patches before re-entering
the council loop.

The ``ContextBuilder`` owns no model backends or tool instances; it
receives them as explicit parameters so the supervisor can manage
lifecycle and concurrency.

Usage::

    from teamai.context_builder import ContextBuilder

    ctx = ContextBuilder(
        context_packager=packager,
        normalize_path=supervisor._normalize_path_arg,
    )
    context_text = ctx.build_context(
        task=task, workspace=workspace, ...,
        render_recent_actions=supervisor._render_recent_actions,
        render_suggested_paths=supervisor._render_suggested_paths,
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .autonomy import ContextPackager
from .memory import WorkspaceMemorySnapshot
from .prompts import build_round_context
from .schemas import (
    PlannerTurn,
    RepoIndex,
    RoundRecord,
    ToolAction,
    ToolExecutionResult,
    VerifierVerdict,
)


class ContextBuilder:
    """Assembles context strings for planner and verifier model calls."""

    def __init__(
        self,
        context_packager: ContextPackager,
        normalize_path: Callable[[str, Path], str],
    ) -> None:
        self._context_packager = context_packager
        self._normalize_path = normalize_path

    # ------------------------------------------------------------------
    # Planner / verifier context
    # ------------------------------------------------------------------

    def build_context(
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
        render_recent_actions: Callable[[list[RoundRecord], Path], str],
        render_suggested_paths: Callable[[list[RoundRecord], Path, str, str], str],
    ) -> str:
        previous = []
        for record in previous_rounds[-2:]:
            previous.append(
                f"Round {record.round_number}: "
                f"planner={record.planner.summary}; verifier={record.verifier.summary}"
            )
        previous_rounds_text = "\n".join(previous) if previous else "No prior rounds."

        recent_actions = render_recent_actions(previous_rounds, workspace)
        suggested_paths = render_suggested_paths(
            previous_rounds,
            workspace,
            task=task,
            task_route=task_route,
        )

        latest_observations = "No tool observations yet."
        if previous_rounds:
            latest = previous_rounds[-1].tool_results
            latest_observations = self.render_tool_observations(latest) or latest_observations

        bundle = self._context_packager.build(
            task=task,
            workspace=workspace,
            repo_index=repo_index,
            task_scopes=task_scopes,
            observed_paths=self.observed_paths_from_rounds(previous_rounds, workspace),
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
            continuation_context=self.render_continuation_context(continuation_context),
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

    def build_verifier_context(
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
            observed_paths=self.tool_result_paths(tool_results, workspace),
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
            f"Tool results:\n{self.render_tool_observations(tool_results)}"
        )

    # ------------------------------------------------------------------
    # Path extraction from rounds
    # ------------------------------------------------------------------

    def observed_paths_from_rounds(self, rounds: list[RoundRecord], workspace: Path) -> tuple[str, ...]:
        observed: list[str] = []
        for record in rounds:
            for result in record.tool_results:
                raw_path = str(result.metadata.get("path", "")).strip()
                if not raw_path:
                    continue
                normalized = self._normalize_path(raw_path, workspace)
                if normalized and normalized not in observed:
                    observed.append(normalized)
        return tuple(observed)

    def tool_result_paths(self, tool_results: list[ToolExecutionResult], workspace: Path) -> tuple[str, ...]:
        observed: list[str] = []
        for result in tool_results:
            raw_path = str(result.metadata.get("path", "")).strip()
            if not raw_path:
                continue
            normalized = self._normalize_path(raw_path, workspace)
            if normalized and normalized not in observed:
                observed.append(normalized)
        return tuple(observed)

    # ------------------------------------------------------------------
    # Continuation-probe logic
    # ------------------------------------------------------------------

    def build_continuation_probe_round(
        self,
        *,
        workspace: Path,
        continuation_context: dict[str, object],
        execute_actions: Callable[[list[ToolAction], Path, str], list[ToolExecutionResult]],
    ) -> RoundRecord | None:
        actions = self.build_continuation_probe_actions(continuation_context)
        if not actions:
            return None

        tool_results = execute_actions(actions, workspace, "read_only")
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

    def maybe_complete_after_continuation_probe(
        self,
        *,
        task: str,
        workspace: Path,
        continuation_context: dict[str, object],
        probe_round: RoundRecord,
        extract_file_targets: Callable[[str, Path], list[str]],
    ) -> tuple[str, str, str] | None:
        if any(not result.success for result in probe_round.tool_results):
            return None

        verification_results = [result for result in probe_round.tool_results if result.tool == "run_command"]
        if not verification_results or not all(result.success for result in verification_results):
            return None

        changed_paths = [
            self._normalize_path(path, workspace)
            for path in continuation_context.get("changed_paths", [])
            if str(path).strip()
        ]
        if not changed_paths:
            return None

        original_task = str(continuation_context.get("original_task", "")).strip() or task
        explicit_targets = extract_file_targets(original_task, workspace)
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
    def build_continuation_probe_actions(continuation_context: dict[str, object]) -> list[ToolAction]:
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
    def render_continuation_context(continuation_context: dict[str, object]) -> str:
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
    def render_tool_observations(tool_results: list[ToolExecutionResult]) -> str:
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
