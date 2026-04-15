from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .agent_registry import RouteHealth
from .schemas import ModelPerformanceRecord, RateLimitState


if TYPE_CHECKING:
    from .schemas import AutonomousRunState, RoundRecord


STATE_DIR_NAME = ".teamai"
RUN_HISTORY_FILE_NAME = "run-history.jsonl"
MEMORY_FILE_NAME = "memory.md"
POLICY_MEMORY_FILE_NAME = "policy-memory.json"
AGENT_POLICY_MEMORY_FILE_NAME = "agent-policy-memory.json"
MODEL_PERFORMANCE_FILE_NAME = "model-performance.json"
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
GLOBAL_RATE_LIMITS_FILE_NAME = "rate_limits.json"
MAX_GLOBAL_MEMORY_NOTES = 20
MAX_GLOBAL_MEMORY_CHARS = 4_000
MODEL_PERFORMANCE_EMA_ALPHA = 0.35
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
        run_state: AutonomousRunState | None = None,
    ) -> None:
        state_dir = self._state_dir(workspace)
        state_dir.mkdir(parents=True, exist_ok=True)

        rounds = rounds or []
        summary, next_tasks = self._extract_summary_and_tasks(final_answer)
        successful_action_count, failed_action_count, saw_unittest = self._count_tool_results(rounds)
        approval_created = stop_reason == "approval_required" or self._has_pending_approval(rounds)
        json_repair_count, structured_response_count = self._count_json_repairs(rounds)
        json_repair_rate = (
            json_repair_count / structured_response_count if structured_response_count else 0.0
        )
        memory_pressure = any("memory pressure" in warning.lower() for warning in warnings)
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
        retry_count = 0.0
        verifier_disagreement_rate = 0.0
        if run_state is not None:
            retry_count = float(run_state.metrics.get("retry_count", 0.0))
            verifier_disagreement_rate = float(run_state.metrics.get("verifier_disagreement_rate", 0.0))
        task_tags = sorted(self._task_tags(task))
        records.append(
            {
                "completed_at": completed_at.isoformat(),
                "task": task,
                "task_tags": task_tags,
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
                "json_repair_count": json_repair_count,
                "structured_response_count": structured_response_count,
                "json_repair_rate": round(json_repair_rate, 4),
                "memory_pressure": memory_pressure,
                "retry_count": retry_count,
                "verifier_disagreement_rate": verifier_disagreement_rate,
                "policy_mode": getattr(run_state, "policy_mode", execution_mode),
                "improvement_notes": improvement_notes,
            }
        )
        records = records[-MAX_HISTORY_ENTRIES:]

        history_path = state_dir / RUN_HISTORY_FILE_NAME
        history_payload = "\n".join(json.dumps(record, ensure_ascii=True) for record in records)
        history_path.write_text(history_payload + ("\n" if history_payload else ""), encoding="utf-8")

        memory_path = state_dir / MEMORY_FILE_NAME
        memory_path.write_text(self._render_memory_markdown(records), encoding="utf-8")
        self._write_policy_memory(workspace, records)

        try:
            GlobalMemoryStore().update(improvement_notes)
        except OSError:
            pass

    def load_routing_health(
        self,
        workspace: Path,
        *,
        recent_window: int = 8,
        broken_repair_rate: float = 0.35,
    ) -> dict[str, RouteHealth]:
        records = self._load_history_records(workspace)
        per_route: dict[str, list[dict[str, object]]] = {}

        for record in reversed(records):
            route = str(record.get("task_route", "")).strip()
            if not route or route == "eval_feedback":
                continue
            bucket = per_route.setdefault(route, [])
            if len(bucket) >= max(recent_window, 1):
                continue
            bucket.append(record)

        health: dict[str, RouteHealth] = {}
        for route, reversed_records in per_route.items():
            route_records = list(reversed(reversed_records))
            recent_runs = len(route_records)
            success_count = sum(1 for record in route_records if self._record_counts_as_success(record))
            repair_count = sum(self._record_json_repair_count(record) for record in route_records)
            structured_count = sum(self._record_structured_response_count(record) for record in route_records)
            repair_rate = repair_count / structured_count if structured_count else 0.0
            average_retries = (
                sum(float(record.get("retry_count", 0.0) or 0.0) for record in route_records) / recent_runs
                if recent_runs
                else 0.0
            )
            verifier_disagreement_rate = (
                sum(float(record.get("verifier_disagreement_rate", 0.0) or 0.0) for record in route_records) / recent_runs
                if recent_runs
                else 0.0
            )
            memory_pressure = any(self._record_memory_pressure(record) for record in route_records)
            health[route] = RouteHealth(
                capability=route,
                recent_runs=recent_runs,
                success_rate=(success_count / recent_runs) if recent_runs else 0.0,
                repair_rate=repair_rate,
                average_retries=average_retries,
                verifier_disagreement_rate=verifier_disagreement_rate,
                broken=structured_count > 0 and repair_rate >= broken_repair_rate,
                memory_pressure=memory_pressure,
            )
        return health

    def load_policy_scores(
        self,
        workspace: Path,
        *,
        task: str,
        complexity: str,
    ) -> dict[str, float]:
        policy_path = self._state_dir(workspace) / POLICY_MEMORY_FILE_NAME
        if not policy_path.exists():
            return {}
        try:
            payload = json.loads(policy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}

        task_tags = self._task_tags(task)
        scores: dict[str, float] = {}
        for capability, entries in payload.items():
            if not isinstance(entries, list):
                continue
            adjustment = 0.0
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                tags = {str(tag) for tag in (entry.get("task_tags") or [])}
                overlap = len(task_tags & tags)
                if task_tags and overlap == 0:
                    continue
                success_rate = float(entry.get("success_rate", 0.0) or 0.0)
                retry_penalty = float(entry.get("average_retries", 0.0) or 0.0)
                disagreement_penalty = float(entry.get("verifier_disagreement_rate", 0.0) or 0.0)
                complexity_match = 1.0 if str(entry.get("complexity", "")) == complexity else 0.5
                adjustment += (success_rate * 4.0 * complexity_match) + (overlap * 1.5) - retry_penalty - disagreement_penalty
            if adjustment:
                scores[str(capability)] = round(adjustment, 2)
        return scores

    def load_agent_policy_scores(
        self,
        workspace: Path,
        *,
        task: str,
        complexity: str,
    ) -> dict[str, float]:
        policy_path = self._state_dir(workspace) / AGENT_POLICY_MEMORY_FILE_NAME
        if not policy_path.exists():
            return {}
        try:
            payload = json.loads(policy_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}

        task_tags = self._task_tags(task)
        scores: dict[str, float] = {}
        for key, entries in payload.items():
            if not isinstance(entries, list):
                continue
            adjustment = 0.0
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                tags = {str(tag) for tag in (entry.get("task_tags") or [])}
                overlap = len(task_tags & tags)
                if task_tags and overlap == 0:
                    continue
                success_rate = float(entry.get("success_rate", 0.0) or 0.0)
                retry_penalty = float(entry.get("average_retries", 0.0) or 0.0)
                disagreement_penalty = float(entry.get("verifier_disagreement_rate", 0.0) or 0.0)
                complexity_match = 1.0 if str(entry.get("complexity", "")) == complexity else 0.5
                adjustment += (success_rate * 5.0 * complexity_match) + overlap - retry_penalty - disagreement_penalty
            if adjustment:
                scores[str(key)] = round(adjustment, 2)
        return scores

    def get_rate_limits(self) -> dict[str, RateLimitState]:
        rate_limit_path = self._global_state_dir() / GLOBAL_RATE_LIMITS_FILE_NAME
        if not rate_limit_path.exists():
            return {}
        try:
            payload = json.loads(rate_limit_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}

        states: dict[str, RateLimitState] = {}
        for model_id, entry in payload.items():
            if not isinstance(entry, dict):
                continue
            try:
                state = RateLimitState.model_validate(entry)
            except Exception:
                continue
            states[str(model_id)] = self._normalize_rate_limit_state(state)
        return states

    def update_rate_limits(
        self,
        *,
        model_id: str,
        state: RateLimitState,
    ) -> RateLimitState:
        normalized_model_id = model_id.strip() or state.model_id.strip()
        if not normalized_model_id:
            raise ValueError("model_id is required when persisting rate-limit state.")

        current = self.get_rate_limits().get(normalized_model_id)
        merged = state.model_copy(deep=True)
        merged.model_id = normalized_model_id

        if current is not None:
            merged.requests_made = current.requests_made + max(merged.requests_made, 0)
            merged.tokens_in = current.tokens_in + max(merged.tokens_in, 0)
            merged.tokens_out = current.tokens_out + max(merged.tokens_out, 0)

            for field_name in (
                "requests_limit",
                "remaining_requests",
                "tokens_limit",
                "remaining_tokens",
                "daily_requests_limit",
                "remaining_daily_requests",
                "daily_tokens_limit",
                "remaining_daily_tokens",
                "window_headroom",
                "daily_headroom",
            ):
                if getattr(merged, field_name) is None:
                    setattr(merged, field_name, getattr(current, field_name))
            if not merged.provider:
                merged.provider = current.provider
            if merged.source == "usage_only":
                merged.source = current.source
            if merged.quota_pressure <= 0.0 and current.quota_pressure > 0.0:
                merged.quota_pressure = current.quota_pressure

        merged = self._normalize_rate_limit_state(merged)

        rate_limit_path = self._global_state_dir() / GLOBAL_RATE_LIMITS_FILE_NAME
        rate_limit_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            key: value.model_dump(mode="json")
            for key, value in {**self.get_rate_limits(), normalized_model_id: merged}.items()
        }
        rate_limit_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return merged

    def get_model_stats(
        self,
        workspace: Path,
        *,
        task_signature_hash: str,
    ) -> dict[str, ModelPerformanceRecord]:
        performance_path = self._state_dir(workspace) / MODEL_PERFORMANCE_FILE_NAME
        if not performance_path.exists():
            return {}
        try:
            payload = json.loads(performance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if not isinstance(payload, dict):
            return {}

        bucket = payload.get(task_signature_hash)
        if not isinstance(bucket, dict):
            return {}

        records: dict[str, ModelPerformanceRecord] = {}
        for model_id, entry in bucket.items():
            if not isinstance(entry, dict):
                continue
            try:
                records[str(model_id)] = ModelPerformanceRecord.model_validate(entry)
            except Exception:
                continue
        return records

    def update_model_stats(
        self,
        workspace: Path,
        *,
        task_signature_hash: str,
        model_id: str,
        success: bool,
        latency_ms: float | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        total_tokens: int | None = None,
        cost: float | None = None,
        quota_pressure: float | None = None,
        status: str | None = None,
    ) -> ModelPerformanceRecord:
        performance_path = self._state_dir(workspace) / MODEL_PERFORMANCE_FILE_NAME
        performance_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            payload = json.loads(performance_path.read_text(encoding="utf-8")) if performance_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            payload = {}

        bucket = payload.setdefault(task_signature_hash, {})
        if not isinstance(bucket, dict):
            bucket = {}
            payload[task_signature_hash] = bucket

        existing_raw = bucket.get(model_id)
        existing = None
        if isinstance(existing_raw, dict):
            try:
                existing = ModelPerformanceRecord.model_validate(existing_raw)
            except Exception:
                existing = None

        record = existing or ModelPerformanceRecord(
            task_signature_hash=task_signature_hash,
            model_id=model_id,
        )
        sample_count = max(record.sample_count, 0) + 1
        record = record.model_copy(deep=True)
        record.sample_count = sample_count
        record.success_ema = self._ema(record.success_ema, 1.0 if success else 0.0, previous_count=sample_count - 1)
        if latency_ms is not None:
            record.latency_ema_ms = self._ema(record.latency_ema_ms, float(latency_ms), previous_count=sample_count - 1)
        if prompt_tokens is not None:
            record.prompt_tokens_ema = self._ema(record.prompt_tokens_ema, float(prompt_tokens), previous_count=sample_count - 1)
        if completion_tokens is not None:
            record.completion_tokens_ema = self._ema(record.completion_tokens_ema, float(completion_tokens), previous_count=sample_count - 1)
        if total_tokens is not None:
            record.total_tokens_ema = self._ema(record.total_tokens_ema, float(total_tokens), previous_count=sample_count - 1)
        if cost is not None:
            record.cost_ema = self._ema(record.cost_ema, float(cost), previous_count=sample_count - 1)
        if quota_pressure is not None:
            record.quota_pressure_ema = self._ema(record.quota_pressure_ema, float(quota_pressure), previous_count=sample_count - 1)
        record.last_status = status
        record.updated_at = datetime.now(timezone.utc)

        bucket[model_id] = record.model_dump(mode="json")
        performance_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return record

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

    def _write_policy_memory(self, workspace: Path, records: list[dict[str, object]]) -> None:
        by_capability: dict[str, list[dict[str, object]]] = {}
        for record in records:
            capability = str(record.get("task_route", "")).strip()
            if not capability or capability == "eval_feedback":
                continue
            by_capability.setdefault(capability, []).append(record)

        payload: dict[str, list[dict[str, object]]] = {}
        for capability, route_records in by_capability.items():
            if not route_records:
                continue
            successes = sum(1 for record in route_records if self._record_counts_as_success(record))
            retry_avg = sum(float(record.get("retry_count", 0.0) or 0.0) for record in route_records) / len(route_records)
            disagreement_avg = (
                sum(float(record.get("verifier_disagreement_rate", 0.0) or 0.0) for record in route_records)
                / len(route_records)
            )
            tags: list[str] = []
            for record in route_records[-6:]:
                for tag in record.get("task_tags", []) if isinstance(record.get("task_tags"), list) else []:
                    tag_str = str(tag)
                    if tag_str not in tags:
                        tags.append(tag_str)
            payload[capability] = [
                {
                    "task_tags": tags,
                    "success_rate": successes / len(route_records),
                    "average_retries": round(retry_avg, 4),
                    "verifier_disagreement_rate": round(disagreement_avg, 4),
                    "complexity": self._infer_complexity_from_record(route_records[-1]),
                }
            ]

        policy_path = self._state_dir(workspace) / POLICY_MEMORY_FILE_NAME
        policy_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self._write_agent_policy_memory(workspace, records)

    def _write_agent_policy_memory(self, workspace: Path, records: list[dict[str, object]]) -> None:
        by_model: dict[str, list[dict[str, object]]] = {}
        for record in records:
            model_id = str(record.get("model_id", "")).strip()
            if not model_id or str(record.get("task_route", "")).strip() == "eval_feedback":
                continue
            by_model.setdefault(model_id, []).append(record)

        payload: dict[str, list[dict[str, object]]] = {}
        for model_id, model_records in by_model.items():
            successes = sum(1 for record in model_records if self._record_counts_as_success(record))
            retry_avg = sum(float(record.get("retry_count", 0.0) or 0.0) for record in model_records) / len(model_records)
            disagreement_avg = (
                sum(float(record.get("verifier_disagreement_rate", 0.0) or 0.0) for record in model_records)
                / len(model_records)
            )
            tags: list[str] = []
            for record in model_records[-6:]:
                for tag in record.get("task_tags", []) if isinstance(record.get("task_tags"), list) else []:
                    tag_str = str(tag)
                    if tag_str not in tags:
                        tags.append(tag_str)
            payload[model_id] = [
                {
                    "task_tags": tags,
                    "success_rate": successes / len(model_records),
                    "average_retries": round(retry_avg, 4),
                    "verifier_disagreement_rate": round(disagreement_avg, 4),
                    "complexity": self._infer_complexity_from_record(model_records[-1]),
                }
            ]

        agent_policy_path = self._state_dir(workspace) / AGENT_POLICY_MEMORY_FILE_NAME
        agent_policy_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

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
    def _record_counts_as_success(record: dict[str, object]) -> bool:
        status = str(record.get("status", "")).strip()
        stop_reason = str(record.get("stop_reason", "")).strip()
        return status == "completed" or stop_reason == "approval_required"

    @staticmethod
    def _record_json_repair_count(record: dict[str, object]) -> int:
        try:
            stored = int(record.get("json_repair_count", 0) or 0)
        except (TypeError, ValueError):
            stored = 0
        if stored > 0:
            return stored

        warnings = record.get("warnings") or []
        if isinstance(warnings, list):
            return sum(
                1
                for warning in warnings
                if isinstance(warning, str) and "json required repair" in warning.lower()
            )
        return 0

    @staticmethod
    def _record_structured_response_count(record: dict[str, object]) -> int:
        try:
            stored = int(record.get("structured_response_count", 0) or 0)
        except (TypeError, ValueError):
            stored = 0
        if stored > 0:
            return stored

        task_route = str(record.get("task_route", "")).strip()
        if task_route in {"deterministic_patch", "repository_inspection", "write_disabled_preflight"}:
            return 0
        return 2

    @staticmethod
    def _record_memory_pressure(record: dict[str, object]) -> bool:
        if bool(record.get("memory_pressure", False)):
            return True
        warnings = record.get("warnings") or []
        return isinstance(warnings, list) and any(
            isinstance(warning, str) and "memory pressure" in warning.lower()
            for warning in warnings
        )

    @staticmethod
    def _infer_complexity_from_record(record: dict[str, object]) -> str:
        task = str(record.get("task", "")).lower()
        if any(marker in task for marker in ("refactor", "architecture", "multi-file", "cross-file", "end-to-end")):
            return "high"
        if any(marker in task for marker in ("implement", "fix", "repair", "update", "wire")):
            return "medium"
        return "low"

    @staticmethod
    def _task_tags(task: str) -> set[str]:
        lowered = task.lower()
        tags: set[str] = set()
        for marker in (
            "typescript",
            "python",
            "refactor",
            "import",
            "test",
            "cli",
            "api",
            "routing",
            "handoff",
            "memory",
            "write",
            "approval",
            "sandbox",
        ):
            if marker in lowered:
                tags.add(marker)
        for match in re.finditer(r"\b[\w./-]+\.(py|ts|tsx|js|jsx|md|json|toml|yaml|yml)\b", lowered):
            tags.add(match.group(0))
        return tags

    @staticmethod
    def _state_dir(workspace: Path) -> Path:
        return workspace / STATE_DIR_NAME

    @staticmethod
    def _global_state_dir() -> Path:
        return GLOBAL_STATE_DIR.expanduser()

    @classmethod
    def _normalize_rate_limit_state(cls, state: RateLimitState) -> RateLimitState:
        normalized = state.model_copy(deep=True)
        normalized.window_headroom = cls._headroom(normalized.remaining_requests, normalized.requests_limit)
        token_headroom = cls._headroom(normalized.remaining_tokens, normalized.tokens_limit)
        if token_headroom is not None:
            normalized.window_headroom = (
                min(normalized.window_headroom, token_headroom)
                if normalized.window_headroom is not None
                else token_headroom
            )
        normalized.daily_headroom = cls._headroom(
            normalized.remaining_daily_requests,
            normalized.daily_requests_limit,
        )
        daily_token_headroom = cls._headroom(
            normalized.remaining_daily_tokens,
            normalized.daily_tokens_limit,
        )
        if daily_token_headroom is not None:
            normalized.daily_headroom = (
                min(normalized.daily_headroom, daily_token_headroom)
                if normalized.daily_headroom is not None
                else daily_token_headroom
            )

        headrooms = [value for value in (normalized.window_headroom, normalized.daily_headroom) if value is not None]
        normalized.quota_pressure = round(1.0 - min(headrooms), 4) if headrooms else max(normalized.quota_pressure, 0.0)
        normalized.observed_at = datetime.now(timezone.utc)
        return normalized

    @staticmethod
    def _headroom(remaining: int | None, limit: int | None) -> float | None:
        if remaining is None or limit in {None, 0}:
            return None
        return max(0.0, min(1.0, float(remaining) / float(limit)))

    @staticmethod
    def _ema(previous: float, value: float, *, previous_count: int) -> float:
        if previous_count <= 0:
            return round(value, 4)
        alpha = MODEL_PERFORMANCE_EMA_ALPHA
        return round((alpha * value) + ((1.0 - alpha) * previous), 4)

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
    def _count_json_repairs(rounds: list[RoundRecord]) -> tuple[int, int]:
        total_repairs = 0
        structured_turns = 0
        for record in rounds:
            total_repairs += max(int(getattr(record, "planner_json_repairs", 0) or 0), 0)
            total_repairs += max(int(getattr(record, "verifier_json_repairs", 0) or 0), 0)
            if getattr(record, "reasoning_source", "model") == "model":
                structured_turns += 2
        return total_repairs, structured_turns

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
