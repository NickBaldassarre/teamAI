from __future__ import annotations

import json
from dataclasses import dataclass
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


@dataclass(frozen=True)
class WorkspaceMemorySnapshot:
    memory_text: str
    recent_runs_text: str
    improvement_notes_text: str


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
        return WorkspaceMemorySnapshot(
            memory_text=memory_text,
            recent_runs_text=recent_runs_text,
            improvement_notes_text=improvement_notes_text,
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
            successful_action_count = int(record.get("successful_action_count", 0) or 0)
            failed_action_count = int(record.get("failed_action_count", 0) or 0)
            summary = str(record.get("summary", "")).strip()
            rendered.append(f"- {completed_at} | {status} | route={task_route} | {task}")
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

        lines = [
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
        specialized_tags = {"inspection", "patch_writes", "verification", "continuation", "codex_handoff"}
        if count == 1 and age >= LOW_SIGNAL_SINGLETON_STALE_AGE:
            penalty += 4
        if note_tags & specialized_tags and age >= SPECIALIZED_NOTE_STALE_AGE:
            penalty += 3
        if current_focus_tags and note_tags & specialized_tags and not (note_tags & current_focus_tags):
            penalty += 5
        return penalty

    @staticmethod
    def _base_improvement_note_score(note_lower: str) -> int:
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
        if "most specific related unittest" in note_lower or "approved patch" in note_lower:
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
        if continuation_context:
            tags.add("continuation")
            tags.add("verification")
        if "inspect this repository" in lowered_task or "identify the next engineering tasks" in lowered_task:
            tags.add("inspection")
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
            ("codex_handoff", "Codex-handoff lessons first"),
        ]
        for tag, label in mapping:
            if tag in tags:
                ordered.append(label)
        return ", ".join(ordered) if ordered else "general efficiency lessons first"

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
