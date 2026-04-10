from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from .schemas import RoundRecord


STATE_DIR_NAME = ".teamai"
RUN_HISTORY_FILE_NAME = "run-history.jsonl"
MEMORY_FILE_NAME = "memory.md"
MAX_HISTORY_ENTRIES = 50
MAX_CONTEXT_RUNS = 5
MAX_MEMORY_CHARS = 4_000
MAX_IMPROVEMENT_NOTES = 6
MAX_IMPROVEMENT_NOTE_CHARS = 1_500
MIN_IMPROVEMENT_NOTE_SCORE = 6
SPECIALIZED_NOTE_STALE_AGE = 2
LOW_SIGNAL_SINGLETON_STALE_AGE = 2
MAX_EVAL_FAILURE_CASES = 4

# Global memory constants
GLOBAL_STATE_DIR = Path("~/.teamai")
GLOBAL_MEMORY_FILE_NAME = "global-memory.md"
MAX_GLOBAL_MEMORY_NOTES = 20
MAX_GLOBAL_MEMORY_CHARS = 4_000
# Pattern to detect project-specific file references that make a note non-generalizable
_SPECIFIC_FILE_RE = re.compile(
    r"\b[\w./-]+/[\w.-]+\b|\b\w+\.(py|js|ts|md|yaml|json|txt|sh|toml|cfg|lock)\b"
)


@dataclass(frozen=True)
class WorkspaceMemorySnapshot:
    memory_text: str
    recent_runs_text: str
    improvement_notes_text: str
    global_memory_text: str = field(default="")


class WorkspaceMemoryStore:
    def load_snapshot(
        self,
        workspace: Path,
        *,
        task: str = "",
        task_route: str = "",
        continuation_context: dict[str, object] | None = None,
    ) -> WorkspaceMemorySnapshot:
        state_dir = self._state_dir(workspace)
        memory_text = "No persistent workspace memory yet."
        memory_path = state_dir / MEMORY_FILE_NAME
        if memory_path.exists():
            memory_text = memory_path.read_text(encoding="utf-8").strip() or memory_text
            memory_text = memory_text[:MAX_MEMORY_CHARS]

        records = self._load_history_records(workspace)
        recent_runs_text = self._render_recent_runs_text(records[-MAX_CONTEXT_RUNS:])
        improvement_notes_text = self._render_improvement_notes_text(
            records,
            task=task,
            task_route=task_route,
            continuation_context=continuation_context or {},
        )
        global_memory_text = GlobalMemoryStore().load()
        return WorkspaceMemorySnapshot(
            memory_text=memory_text,
            recent_runs_text=recent_runs_text,
            improvement_notes_text=improvement_notes_text,
            global_memory_text=global_memory_text,
        )

    def persist_run(
        self,
        *,
        workspace: Path,
        task: str,
        status: str,
        stop_reason: str,
        final_answer: str,
        warnings: list[str],
        completed_at: datetime,
        model_id: str,
        task_route: str = "multi_agent_loop",
        execution_mode: str = "read_only",
        rounds: list[RoundRecord] | None = None,
    ) -> None:
        state_dir = self._state_dir(workspace)
        state_dir.mkdir(parents=True, exist_ok=True)

        rounds = rounds or []
        summary, next_tasks = self._extract_summary_and_tasks(final_answer)
        successful_action_count, failed_action_count, saw_unittest = self._count_tool_results(rounds)
        approval_created = stop_reason == "approval_required" or self._has_pending_approval(rounds)
        improvement_notes = self._derive_improvement_notes(
            task=task,
            task_route=task_route,
            execution_mode=execution_mode,
            stop_reason=stop_reason,
            warnings=warnings,
            successful_action_count=successful_action_count,
            failed_action_count=failed_action_count,
            approval_created=approval_created,
            saw_unittest=saw_unittest,
        )
        records = self._load_history_records(workspace)
        records.append(
            {
                "completed_at": completed_at.isoformat(),
                "task": task,
                "status": status,
                "stop_reason": stop_reason,
                "task_route": task_route,
                "execution_mode": execution_mode,
                "summary": summary,
                "next_tasks": next_tasks,
                "warnings": warnings,
                "model_id": model_id,
                "successful_action_count": successful_action_count,
                "failed_action_count": failed_action_count,
                "approval_created": approval_created,
                "improvement_notes": improvement_notes,
            }
        )
        records = records[-MAX_HISTORY_ENTRIES:]

        history_path = state_dir / RUN_HISTORY_FILE_NAME
        history_payload = "\n".join(json.dumps(record, ensure_ascii=True) for record in records)
        history_path.write_text(history_payload + ("\n" if history_payload else ""), encoding="utf-8")

        memory_path = state_dir / MEMORY_FILE_NAME
        memory_path.write_text(self._render_memory_markdown(records), encoding="utf-8")

        GlobalMemoryStore().update(improvement_notes)

    def persist_eval_feedback(
        self,
        *,
        workspace: Path,
        suite_name: str,
        completed_at: datetime,
        metrics: dict[str, object],
        cases: list[dict[str, object]],
        description: str = "",
        runtime_health: dict[str, object] | None = None,
    ) -> None:
        state_dir = self._state_dir(workspace)
        state_dir.mkdir(parents=True, exist_ok=True)

        summary = self._summarize_eval_feedback(
            suite_name=suite_name,
            metrics=metrics,
            cases=cases,
            description=description,
            runtime_health=runtime_health,
        )
        next_tasks = self._derive_eval_next_tasks(metrics=metrics, cases=cases, runtime_health=runtime_health)
        improvement_notes = self._derive_eval_feedback_notes(metrics=metrics, cases=cases, runtime_health=runtime_health)
        warnings = self._derive_eval_feedback_warnings(cases, runtime_health=runtime_health)

        total_cases = int(metrics.get("total_cases", len(cases)) or len(cases))
        passed_cases = int(metrics.get("passed_cases", 0) or 0)
        failed_cases = int(metrics.get("failed_cases", total_cases - passed_cases) or 0)

        records = self._load_history_records(workspace)
        records.append(
            {
                "source": "eval_suite",
                "completed_at": completed_at.isoformat(),
                "task": f"Eval suite: {suite_name}",
                "status": "completed",
                "stop_reason": "eval_feedback_recorded",
                "task_route": "eval_feedback",
                "execution_mode": "read_only",
                "summary": summary,
                "next_tasks": next_tasks,
                "warnings": warnings,
                "model_id": "eval_harness",
                "successful_action_count": passed_cases,
                "failed_action_count": failed_cases,
                "approval_created": self._safe_float_from_mapping(metrics, "approval_rate") > 0.0,
                "improvement_notes": improvement_notes,
                "eval_metrics": metrics,
                "runtime_health": runtime_health or {},
                "total_cases": total_cases,
                "passed_cases": passed_cases,
                "failed_cases": failed_cases,
                "description": description.strip(),
            }
        )
        records = records[-MAX_HISTORY_ENTRIES:]

        history_path = state_dir / RUN_HISTORY_FILE_NAME
        history_payload = "\n".join(json.dumps(record, ensure_ascii=True) for record in records)
        history_path.write_text(history_payload + ("\n" if history_payload else ""), encoding="utf-8")

        memory_path = state_dir / MEMORY_FILE_NAME
        memory_path.write_text(self._render_memory_markdown(records), encoding="utf-8")

    def _load_history_records(self, workspace: Path) -> list[dict[str, object]]:
        history_path = self._state_dir(workspace) / RUN_HISTORY_FILE_NAME
        if not history_path.exists():
            return []

        records: list[dict[str, object]] = []
        for line in history_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
        return records[-MAX_HISTORY_ENTRIES:]

    @staticmethod
    def _state_dir(workspace: Path) -> Path:
        return workspace / STATE_DIR_NAME

    @staticmethod
    def _extract_summary_and_tasks(final_answer: str) -> tuple[str, list[str]]:
        marker = "Next engineering tasks:"
        if marker not in final_answer:
            summary = final_answer.strip() or "No summary available."
            return summary, []

        summary_part, tasks_part = final_answer.split(marker, maxsplit=1)
        summary = summary_part.strip() or "No summary available."
        tasks: list[str] = []
        for line in tasks_part.splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                tasks.append(stripped[2:].strip())
        return summary, tasks

    @staticmethod
    def _render_recent_runs_text(records: list[dict[str, object]]) -> str:
        if not records:
            return "No persisted runs yet."

        rendered: list[str] = []
        for record in records[-MAX_CONTEXT_RUNS:]:
            completed_at = str(record.get("completed_at", "unknown-time"))
            status = str(record.get("status", "unknown-status"))
            task_route = str(record.get("task_route", "unknown-route"))
            task = str(record.get("task", "unknown-task"))
            stop_reason = str(record.get("stop_reason", "unknown-stop"))
            summary = str(record.get("summary", "")).strip()
            rendered.append(f"- {completed_at} | {status} | route={task_route} | {task}")
            if str(record.get("source", "")).strip() == "eval_suite":
                passed_cases = int(record.get("passed_cases", 0) or 0)
                failed_cases = int(record.get("failed_cases", 0) or 0)
                rendered.append(
                    f"  stop: {stop_reason}; eval cases: {passed_cases} passed / {failed_cases} failed"
                )
            else:
                successful_action_count = int(record.get("successful_action_count", 0) or 0)
                failed_action_count = int(record.get("failed_action_count", 0) or 0)
                rendered.append(
                    f"  stop: {stop_reason}; actions: {successful_action_count} ok / {failed_action_count} failed"
                )
            if summary:
                rendered.append(f"  summary: {summary[:200]}")
        return "\n".join(rendered)

    def _render_improvement_notes_text(
        self,
        records: list[dict[str, object]],
        *,
        task: str = "",
        task_route: str = "",
        continuation_context: dict[str, object] | None = None,
    ) -> str:
        if not records:
            return "No local improvement notes yet."

        recent_records = records[-MAX_CONTEXT_RUNS:]
        latest = recent_records[-1]
        current_focus_tags = self._current_focus_tags(
            task=task,
            task_route=task_route,
            continuation_context=continuation_context or {},
        )
        note_stats: dict[str, list[int]] = {}
        for index, record in enumerate(recent_records):
            raw_notes = record.get("improvement_notes", [])
            if not isinstance(raw_notes, list):
                continue
            for raw_note in raw_notes:
                note = str(raw_note).strip()
                if not note:
                    continue
                note_stats.setdefault(note, []).append(index)

        lines = [*self._render_latest_outcome_lines(latest)]
        if current_focus_tags:
            lines.append(f"Current task bias: {self._describe_focus_tags(current_focus_tags)}")

        if note_stats:
            lines.append("Bias toward these learned behaviors:")
            ordered_notes = sorted(
                note_stats.items(),
                key=lambda item: (
                    -self._score_improvement_note(
                        item[0],
                        occurrence_indices=item[1],
                        latest_record=latest,
                        total_records=len(recent_records),
                        current_focus_tags=current_focus_tags,
                    ),
                    -len(item[1]),
                    -item[1][-1],
                    item[0],
                ),
            )
            filtered_notes = [
                note
                for note, indices in ordered_notes
                if self._score_improvement_note(
                    note,
                    occurrence_indices=indices,
                    latest_record=latest,
                    total_records=len(recent_records),
                    current_focus_tags=current_focus_tags,
                )
                >= MIN_IMPROVEMENT_NOTE_SCORE
            ]
            notes_to_render = filtered_notes or [note for note, _ in ordered_notes[:2]]
            lines.extend(f"- {note}" for note in notes_to_render[:MAX_IMPROVEMENT_NOTES])
        else:
            lines.append("No stable behavior notes yet.")

        rendered = "\n".join(lines).strip()
        return rendered[:MAX_IMPROVEMENT_NOTE_CHARS]

    @classmethod
    def _score_improvement_note(
        cls,
        note: str,
        *,
        occurrence_indices: list[int],
        latest_record: dict[str, object],
        total_records: int,
        current_focus_tags: set[str],
    ) -> int:
        note_lower = note.lower()
        latest_route = str(latest_record.get("task_route", "")).strip()
        latest_stop_reason = str(latest_record.get("stop_reason", "")).strip()
        note_tags = cls._improvement_note_tags(note_lower)
        count = len(occurrence_indices)
        last_seen_index = occurrence_indices[-1]
        latest_index = total_records - 1
        age = latest_index - last_seen_index

        score = cls._base_improvement_note_score(note_lower)
        score += count * 2
        score += cls._recency_reinforcement_score(occurrence_indices, total_records=total_records)
        if note_tags & current_focus_tags:
            score += 6
            if "verification" in current_focus_tags and "verification" in note_tags:
                score += 2
            if "inspection" in current_focus_tags and "inspection" in note_tags:
                score += 2
            if "patch_writes" in current_focus_tags and "patch_writes" in note_tags:
                score += 2
        score -= cls._staleness_penalty(
            note_tags=note_tags,
            age=age,
            count=count,
            current_focus_tags=current_focus_tags,
        )
        if latest_route == "codex_handoff" and "codex handoff" in note_lower:
            score += 3
        if latest_route == "deterministic_patch" and (
            "deterministic patch route" in note_lower
            or "approval_required" in note_lower
            or "approved patch" in note_lower
        ):
            score += 3
        if str(latest_record.get("source", "")).strip() == "eval_suite" and "eval suite" in note_lower:
            score += 3
        if latest_stop_reason == "local_drift_rerouted" and "reroute earlier" in note_lower:
            score += 3
        if latest_stop_reason == "inspection_synthesized" and "repository inspection" in note_lower:
            score += 2
        return score

    @staticmethod
    def _recency_reinforcement_score(occurrence_indices: list[int], *, total_records: int) -> int:
        latest_index = total_records - 1
        score = 0
        for index in occurrence_indices:
            age = latest_index - index
            score += max(0, 4 - age)
        return score

    @classmethod
    def _staleness_penalty(
        cls,
        *,
        note_tags: set[str],
        age: int,
        count: int,
        current_focus_tags: set[str],
    ) -> int:
        penalty = max(0, age - 1) * 3
        specialized_tags = {"inspection", "patch_writes", "verification", "continuation", "codex_handoff", "evaluation"}
        if count == 1 and age >= LOW_SIGNAL_SINGLETON_STALE_AGE:
            penalty += 4
        if note_tags & specialized_tags and age >= SPECIALIZED_NOTE_STALE_AGE:
            penalty += 3
        if current_focus_tags and note_tags & specialized_tags and not (note_tags & current_focus_tags):
            penalty += 5
        return penalty

    @staticmethod
    def _base_improvement_note_score(note_lower: str) -> int:
        if "eval suite" in note_lower or "evaluation harness" in note_lower:
            return 6
        if "codex handoff" in note_lower or "deterministic patch route" in note_lower:
            return 6
        if "approval_required" in note_lower or "reroute earlier" in note_lower:
            return 6
        if "strict and compact json" in note_lower:
            return 5
        if "tool failures start to dominate" in note_lower:
            return 5
        if "most specific related unittest" in note_lower:
            return 4
        if "file-targeted actions" in note_lower:
            return 3
        if "repository inspection tasks" in note_lower:
            return 3
        return 2

    @staticmethod
    def _improvement_note_tags(note_lower: str) -> set[str]:
        tags: set[str] = set()
        if "eval suite" in note_lower or "evaluation harness" in note_lower:
            tags.add("evaluation")
        if "codex handoff" in note_lower:
            tags.add("codex_handoff")
        if (
            "deterministic patch route" in note_lower
            or "approval_required" in note_lower
            or "workspace_write mode" in note_lower
        ):
            tags.add("patch_writes")
        if "repository inspection tasks" in note_lower:
            tags.add("inspection")
        if (
            "most specific related unittest" in note_lower
            or "approved patch" in note_lower
            or "verification" in note_lower
            or "pytest" in note_lower
            or "unittest" in note_lower
        ):
            tags.update({"continuation", "verification"})
        if "strict and compact json" in note_lower:
            tags.add("structured_output")
        if "file-targeted actions" in note_lower or "tool failures start to dominate" in note_lower:
            tags.add("efficiency")
        return tags

    @classmethod
    def _current_focus_tags(
        cls,
        *,
        task: str,
        task_route: str,
        continuation_context: dict[str, object],
    ) -> set[str]:
        tags: set[str] = set()
        lowered_task = task.lower()
        if task_route == "repository_inspection":
            tags.add("inspection")
        if task_route == "codex_handoff":
            tags.add("codex_handoff")
        if task_route in {"deterministic_patch", "explicit_write_loop"}:
            tags.add("patch_writes")
        if task_route == "eval_feedback":
            tags.add("evaluation")
        if continuation_context:
            tags.add("continuation")
            tags.add("verification")
        if "inspect this repository" in lowered_task or "identify the next engineering tasks" in lowered_task:
            tags.add("inspection")
        if any(marker in lowered_task for marker in ["eval", "evaluation", "benchmark", "regression", "suite"]):
            tags.add("evaluation")
        if any(marker in lowered_task for marker in ["implement", "improve", "harden", "optimize", "refactor"]):
            tags.add("codex_handoff")
        if "test" in lowered_task or "verify" in lowered_task or "unittest" in lowered_task:
            tags.add("verification")
        if cls._looks_like_explicit_write_request(lowered_task):
            tags.add("patch_writes")
        return tags

    @staticmethod
    def _looks_like_explicit_write_request(lowered_task: str) -> bool:
        return any(
            marker in lowered_task
            for marker in [
                "replace the text",
                "append",
                "insert",
                "update ",
                "set ",
                "write ",
                "workspace_write",
            ]
        )

    @staticmethod
    def _describe_focus_tags(tags: set[str]) -> str:
        ordered: list[str] = []
        mapping = [
            ("inspection", "inspection lessons first"),
            ("patch_writes", "patch and approval lessons first"),
            ("verification", "verification and continuation lessons first"),
            ("evaluation", "evaluation-feedback lessons first"),
            ("codex_handoff", "Codex-handoff lessons first"),
        ]
        for tag, label in mapping:
            if tag in tags:
                ordered.append(label)
        return ", ".join(ordered) if ordered else "general efficiency lessons first"

    @classmethod
    def _render_latest_outcome_lines(cls, latest: dict[str, object]) -> list[str]:
        if str(latest.get("source", "")).strip() == "eval_suite":
            total_cases = int(latest.get("total_cases", 0) or 0)
            passed_cases = int(latest.get("passed_cases", 0) or 0)
            failed_cases = int(latest.get("failed_cases", 0) or 0)
            eval_metrics = latest.get("eval_metrics", {})
            local_completion_rate = cls._safe_float_from_mapping(eval_metrics, "local_completion_rate")
            handoff_rate = cls._safe_float_from_mapping(eval_metrics, "handoff_rate")
            verification_success_rate = cls._safe_float_from_mapping(eval_metrics, "verification_success_rate")
            lines = [
                f"Latest eval outcome: {passed_cases}/{total_cases} case(s) passed; {failed_cases} failed.",
                (
                    "Latest eval metrics: "
                    f"local completion {cls._format_rate(local_completion_rate)}, "
                    f"handoff {cls._format_rate(handoff_rate)}, "
                    f"verification success {cls._format_rate(verification_success_rate)}."
                ),
            ]
            runtime_health = latest.get("runtime_health", {})
            if isinstance(runtime_health, dict):
                status = str(runtime_health.get("status", "")).strip()
                summary = str(runtime_health.get("summary", "")).strip()
                if status:
                    lines.append(f"Latest runtime health: {status}. {summary}".strip())
            return lines

        return [
            (
                "Latest route outcome: "
                f"{latest.get('task_route', 'unknown-route')} -> {latest.get('stop_reason', 'unknown-stop')}."
            ),
            (
                "Latest tool reliability: "
                f"{int(latest.get('successful_action_count', 0) or 0)} successful action(s), "
                f"{int(latest.get('failed_action_count', 0) or 0)} failed action(s)."
            ),
        ]

    @staticmethod
    def _safe_float_from_mapping(mapping: object, key: str) -> float:
        if not isinstance(mapping, dict):
            return 0.0
        try:
            return float(mapping.get(key, 0.0) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _format_rate(value: float) -> str:
        return f"{value * 100:.1f}%"

    @classmethod
    def _summarize_eval_feedback(
        cls,
        *,
        suite_name: str,
        metrics: dict[str, object],
        cases: list[dict[str, object]],
        description: str = "",
        runtime_health: dict[str, object] | None = None,
    ) -> str:
        total_cases = int(metrics.get("total_cases", len(cases)) or len(cases))
        passed_cases = int(metrics.get("passed_cases", 0) or 0)
        failed_cases = int(metrics.get("failed_cases", total_cases - passed_cases) or 0)
        infra_failure_cases = int(metrics.get("infra_failure_cases", 0) or 0)
        local_completion_rate = cls._safe_float_from_mapping(metrics, "local_completion_rate")
        handoff_rate = cls._safe_float_from_mapping(metrics, "handoff_rate")
        verification_success_rate = cls._safe_float_from_mapping(metrics, "verification_success_rate")
        summary = (
            f"Eval suite `{suite_name}` completed with {passed_cases}/{total_cases} case(s) passing "
            f"({failed_cases} failed). "
            f"Local completion was {cls._format_rate(local_completion_rate)}, "
            f"handoff rate was {cls._format_rate(handoff_rate)}, "
            f"and verification success was {cls._format_rate(verification_success_rate)}."
        )
        if infra_failure_cases:
            summary = (
                f"{summary} {infra_failure_cases}/{total_cases} case(s) failed because the local runtime was unavailable, "
                "so treat those results as infrastructure health signals instead of agent-quality regressions."
            )
        failing_case_ids = [str(case.get("case_id", "")).strip() for case in cases if not bool(case.get("passed", False))]
        failing_case_ids = [case_id for case_id in failing_case_ids if case_id]
        if failing_case_ids:
            summary = f"{summary} Failing cases: {', '.join(failing_case_ids[:MAX_EVAL_FAILURE_CASES])}."
        elif description.strip():
            summary = f"{summary} {description.strip()}"
        if isinstance(runtime_health, dict):
            health_status = str(runtime_health.get("status", "")).strip()
            health_summary = str(runtime_health.get("summary", "")).strip()
            if health_status and health_status != "healthy" and health_summary:
                summary = f"{summary} Runtime preflight: {health_summary}"
        return summary

    @classmethod
    def _derive_eval_next_tasks(
        cls,
        *,
        metrics: dict[str, object],
        cases: list[dict[str, object]],
        runtime_health: dict[str, object] | None = None,
    ) -> list[str]:
        tasks: list[str] = []
        failed_cases = [case for case in cases if not bool(case.get("passed", False))]
        failures_text = "\n".join(
            failure
            for case in failed_cases
            for failure in case.get("failures", [])
            if isinstance(failure, str)
        ).lower()
        failing_routes = {str(case.get("task_route", "")).strip() for case in failed_cases}

        local_completion_rate = cls._safe_float_from_mapping(metrics, "local_completion_rate")
        handoff_rate = cls._safe_float_from_mapping(metrics, "handoff_rate")
        verification_success_rate = cls._safe_float_from_mapping(metrics, "verification_success_rate")
        verification_attempt_rate = cls._safe_float_from_mapping(metrics, "verification_attempt_rate")
        average_tool_success_rate = cls._safe_float_from_mapping(metrics, "average_tool_success_rate")
        infra_failure_cases = int(metrics.get("infra_failure_cases", 0) or 0)
        total_cases = int(metrics.get("total_cases", len(cases)) or len(cases))
        runtime_unhealthy = isinstance(runtime_health, dict) and str(runtime_health.get("status", "")).strip() == "unavailable"

        if infra_failure_cases:
            tasks.append("Restore local MLX runtime health and rerun the eval suite before treating these failures as agent-behavior regressions.")
        if runtime_unhealthy:
            tasks.append("Harden or expand MLX runtime preflight checks so eval reports flag backend availability before scoring behavior.")
        if infra_failure_cases >= total_cases > 0:
            return tasks[:MAX_IMPROVEMENT_NOTES]

        if "json" in failures_text:
            tasks.append("Reduce structured-output repair overhead in planner and verifier responses.")
        if verification_attempt_rate < 1.0 or verification_success_rate < 1.0:
            tasks.append("Strengthen scoped verification so eval and continuation cases reliably run the most relevant tests.")
        if "deterministic_patch" in failing_routes or "approval_required" in failures_text:
            tasks.append("Harden the deterministic patch and approval flow for explicit narrow edit cases.")
        if handoff_rate > local_completion_rate:
            tasks.append("Improve decomposition and reconnaissance ranking so more eval cases stay local before escalating to a Codex handoff.")
        if average_tool_success_rate < 0.95:
            tasks.append("Reduce tool failures and repeated low-signal actions before expanding local autonomy.")
        if not tasks and failed_cases:
            tasks.append("Review the failed eval cases and tighten routing or note selection where the expectations drifted.")
        if not tasks:
            tasks.append("Expand the eval suite with more representative cases so learned-note changes are measured under broader pressure.")
        return tasks[:MAX_IMPROVEMENT_NOTES]

    @classmethod
    def _derive_eval_feedback_notes(
        cls,
        *,
        metrics: dict[str, object],
        cases: list[dict[str, object]],
        runtime_health: dict[str, object] | None = None,
    ) -> list[str]:
        notes: list[str] = []
        local_completion_rate = cls._safe_float_from_mapping(metrics, "local_completion_rate")
        handoff_rate = cls._safe_float_from_mapping(metrics, "handoff_rate")
        handoff_completion_rate = cls._safe_float_from_mapping(metrics, "handoff_completion_rate")
        approval_rate = cls._safe_float_from_mapping(metrics, "approval_rate")
        verification_attempt_rate = cls._safe_float_from_mapping(metrics, "verification_attempt_rate")
        verification_success_rate = cls._safe_float_from_mapping(metrics, "verification_success_rate")
        infra_failure_cases = int(metrics.get("infra_failure_cases", 0) or 0)
        total_cases = int(metrics.get("total_cases", len(cases)) or len(cases))
        runtime_unhealthy = isinstance(runtime_health, dict) and str(runtime_health.get("status", "")).strip() == "unavailable"

        failures_text = "\n".join(
            failure
            for case in cases
            if not bool(case.get("passed", False))
            for failure in case.get("failures", [])
            if isinstance(failure, str)
        ).lower()
        failing_routes = {str(case.get("task_route", "")).strip() for case in cases if not bool(case.get("passed", False))}

        if infra_failure_cases:
            notes.append(
                "The eval suite hit local-runtime failures; do not treat backend availability or Metal/MLX startup problems as agent-behavior regressions."
            )
        if runtime_unhealthy or infra_failure_cases >= total_cases > 0:
            notes.append(
                "Runtime-health checks should stay visible in the eval scoreboard so learning signals only come from cases where the local model actually got a fair attempt."
            )
        if infra_failure_cases >= total_cases > 0:
            return notes[:MAX_IMPROVEMENT_NOTES]

        if approval_rate > 0.0 and "deterministic_patch" not in failing_routes:
            notes.append(
                "The eval suite confirmed the deterministic patch route can stop cleanly at approval_required for narrow edit tasks."
            )
        if handoff_rate > 0.0 and handoff_completion_rate >= 1.0:
            notes.append(
                "The eval suite confirmed broad or ambiguous implementation work still behaves best as read-only reconnaissance plus a Codex handoff."
            )
        if local_completion_rate > 0.0:
            notes.append(
                "The eval suite showed local completion improves when tasks stay explicit, scoped, and easy to verify."
            )
        if verification_attempt_rate < 1.0 or verification_success_rate < 1.0:
            notes.append(
                "The eval suite showed verification should stay grounded in direct unittest or pytest execution before trusting a follow-up local pass."
            )
        if "json" in failures_text:
            notes.append(
                "The eval suite still loses effective rounds to JSON repair; keep structured outputs strict and compact before expanding autonomy."
            )
        if "codex_handoff" in failing_routes or handoff_rate > local_completion_rate:
            notes.append(
                "The eval suite suggests remote-load reduction depends on improving decomposition and reconnaissance so fewer broad cases fall through to a Codex handoff."
            )
        if "deterministic_patch" in failing_routes or "approval_required" in failures_text:
            notes.append(
                "The eval suite exposed approval or deterministic patch gaps; tighten narrow write routing before asking the local model to do broader edits."
            )
        if not notes:
            notes.append(
                "The eval suite is currently the best source of grounded self-improvement signal; keep using its outcomes to rank, prune, and route learned behaviors."
            )

        deduped: list[str] = []
        seen: set[str] = set()
        for note in notes:
            if note in seen:
                continue
            seen.add(note)
            deduped.append(note)
        return deduped[:MAX_IMPROVEMENT_NOTES]

    @staticmethod
    def _derive_eval_feedback_warnings(
        cases: list[dict[str, object]],
        runtime_health: dict[str, object] | None = None,
    ) -> list[str]:
        warnings: list[str] = []
        seen: set[str] = set()
        if isinstance(runtime_health, dict):
            status = str(runtime_health.get("status", "")).strip()
            summary = str(runtime_health.get("summary", "")).strip()
            if status and status != "healthy" and summary:
                warnings.append(f"runtime-health: {summary}")
                seen.add(warnings[-1])
        for case in cases:
            if bool(case.get("passed", False)):
                continue
            case_id = str(case.get("case_id", "")).strip() or "unknown-case"
            for failure in case.get("failures", []):
                if not isinstance(failure, str):
                    continue
                entry = f"{case_id}: {failure}"
                if entry in seen:
                    continue
                seen.add(entry)
                warnings.append(entry)
                if len(warnings) >= MAX_EVAL_FAILURE_CASES:
                    return warnings
        return warnings

    def _render_memory_markdown(self, records: list[dict[str, object]]) -> str:
        if not records:
            return "# Workspace Memory\n\nNo runs recorded yet.\n"

        latest = records[-1]
        lines = [
            "# Workspace Memory",
            "",
            f"Last updated: {latest.get('completed_at', 'unknown-time')}",
            f"Latest goal: {latest.get('task', 'unknown-task')}",
            f"Latest route: {latest.get('task_route', 'unknown-route')} ({latest.get('execution_mode', 'unknown-mode')})",
            f"Latest result: {latest.get('status', 'unknown-status')} ({latest.get('stop_reason', 'unknown-stop')})",
            "",
            "## Current State",
            str(latest.get("summary", "No summary available.")).strip() or "No summary available.",
        ]

        lines.extend(["", "## Local Improvement Notes"])
        lines.extend(self._render_improvement_notes_text(records).splitlines())

        next_tasks = latest.get("next_tasks", [])
        if isinstance(next_tasks, list) and next_tasks:
            lines.extend(["", "## Open Tasks"])
            lines.extend(f"- {str(task)}" for task in next_tasks[:6])

        lines.extend(["", "## Recent Runs"])
        for record in records[-MAX_CONTEXT_RUNS:]:
            lines.append(
                f"- {record.get('completed_at', 'unknown-time')} | {record.get('status', 'unknown-status')} | {record.get('task', 'unknown-task')}"
            )

        rendered = "\n".join(lines).strip() + "\n"
        return rendered[:MAX_MEMORY_CHARS]

    @staticmethod
    def _count_tool_results(rounds: list[RoundRecord]) -> tuple[int, int, bool]:
        successful_action_count = 0
        failed_action_count = 0
        saw_unittest = False
        for record in rounds:
            for result in record.tool_results:
                if result.success:
                    successful_action_count += 1
                else:
                    failed_action_count += 1
                if result.tool != "run_command":
                    continue
                command = str(result.metadata.get("command", "")).strip().lower()
                if "unittest" in command:
                    saw_unittest = True
        return successful_action_count, failed_action_count, saw_unittest

    @staticmethod
    def _has_pending_approval(rounds: list[RoundRecord]) -> bool:
        for record in rounds:
            for result in record.tool_results:
                if str(result.metadata.get("approval_status", "")).strip() == "pending":
                    return True
        return False

    @staticmethod
    def _derive_improvement_notes(
        *,
        task: str,
        task_route: str,
        execution_mode: str,
        stop_reason: str,
        warnings: list[str],
        successful_action_count: int,
        failed_action_count: int,
        approval_created: bool,
        saw_unittest: bool,
    ) -> list[str]:
        notes: list[str] = []
        lowered_task = task.lower()
        lowered_warnings = [warning.lower() for warning in warnings]

        if task_route == "deterministic_patch":
            notes.append(
                "Narrow explicit edit requests work best through the deterministic patch route; compile a concrete patch before falling back to broader planning."
            )
        if task_route == "codex_handoff":
            notes.append(
                "Broad or ambiguous implementation requests work better as read-only reconnaissance plus a Codex handoff than as autonomous local edits."
            )
        if stop_reason == "local_drift_rerouted":
            notes.append(
                "When repeated low-confidence write rounds still do not produce a concrete patch, reroute earlier to a Codex handoff instead of exhausting the local round budget."
            )
        if stop_reason == "inspection_synthesized" or "inspect this repository" in lowered_task:
            notes.append(
                "For repository inspection tasks, synthesize concrete next tasks once enough evidence is gathered instead of spending extra rounds on orientation."
            )
        if approval_created and execution_mode == "workspace_write":
            notes.append(
                "In workspace_write mode, stop cleanly at approval_required and resume after review instead of claiming the file is already changed."
            )
        if saw_unittest:
            notes.append(
                "After an approved patch, reread the changed file and run the most specific related unittest first to ground the next local pass."
            )
        if any("json required repair" in warning for warning in lowered_warnings):
            notes.append(
                "Keep planner and verifier outputs strict and compact JSON; malformed structured output adds repair overhead and costs effective rounds."
            )
        if any(
            "planner had no novel actions" in warning or "skipping repeated successful action" in warning
            for warning in lowered_warnings
        ):
            notes.append(
                "Prefer novel, file-targeted actions over repeated directory listings or recycled successful actions."
            )
        if failed_action_count and failed_action_count >= max(successful_action_count, 1):
            notes.append(
                "When tool failures start to dominate a run, narrow the scope or switch to reconnaissance instead of pushing deeper locally."
            )

        deduped: list[str] = []
        seen: set[str] = set()
        for note in notes:
            if note in seen:
                continue
            seen.add(note)
            deduped.append(note)
        return deduped[:MAX_IMPROVEMENT_NOTES]


class GlobalMemoryStore:
    """Cross-workspace persistent memory stored at ~/.teamai/global-memory.md.

    Collects generalizable lessons (ones that don't reference specific file paths
    or project-specific terms) from every workspace run and makes them available
    to all future runs regardless of workspace.
    """

    def __init__(self) -> None:
        self._state_dir: Path = GLOBAL_STATE_DIR.expanduser()
        self._memory_path: Path = self._state_dir / GLOBAL_MEMORY_FILE_NAME

    def load(self) -> str:
        if not self._memory_path.exists():
            return ""
        text = self._memory_path.read_text(encoding="utf-8").strip()
        return text[:MAX_GLOBAL_MEMORY_CHARS]

    def update(self, improvement_notes: list[str]) -> None:
        generalizable = [note for note in improvement_notes if self._is_generalizable(note)]
        if not generalizable:
            return
        self._state_dir.mkdir(parents=True, exist_ok=True)
        existing = self._load_notes()
        added = 0
        for note in generalizable:
            if note not in existing:
                existing.append(note)
                added += 1
        if not added:
            return
        trimmed = existing[-MAX_GLOBAL_MEMORY_NOTES:]
        self._memory_path.write_text(self._render(trimmed), encoding="utf-8")

    def _load_notes(self) -> list[str]:
        if not self._memory_path.exists():
            return []
        notes: list[str] = []
        for line in self._memory_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith("- "):
                notes.append(stripped[2:].strip())
        return notes

    @staticmethod
    def _render(notes: list[str]) -> str:
        lines = ["# Global teamAI Lessons", ""]
        lines.extend(f"- {note}" for note in notes)
        return "\n".join(lines) + "\n"

    @staticmethod
    def _is_generalizable(note: str) -> bool:
        """Return True if the note doesn't reference project-specific file paths."""
        if len(note.strip()) < 30:
            return False
        return not bool(_SPECIFIC_FILE_RE.search(note.lower()))
