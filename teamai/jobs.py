from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from .events import build_status_event
from .schemas import JobResponse, RunEvent, RunRequest, RunResult


@dataclass
class _JobRecord:
    job_id: str
    request: RunRequest
    status: str
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: RunResult | None = None
    error: str | None = None
    events: list[RunEvent] | None = None
    next_event_sequence: int = 1


class InMemoryJobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._records: dict[str, _JobRecord] = {}

    def create(self, request: RunRequest) -> JobResponse:
        with self._lock:
            job_id = f"job_{uuid.uuid4().hex[:12]}"
            record = _JobRecord(
                job_id=job_id,
                request=request,
                status="queued",
                created_at=datetime.now(timezone.utc),
                events=[],
            )
            self._append_record_event(
                record,
                build_status_event(
                    sequence=0,
                    kind="job_queued",
                    message="Job queued.",
                    data={"job_id": job_id},
                ),
            )
            self._records[job_id] = record
            return self._to_response(record)

    def mark_running(self, job_id: str) -> None:
        with self._condition:
            record = self._records[job_id]
            record.status = "running"
            record.started_at = datetime.now(timezone.utc)
            self._append_record_event(
                record,
                build_status_event(
                    sequence=0,
                    kind="job_running",
                    message="Job running.",
                    data={"job_id": job_id},
                ),
            )
            self._condition.notify_all()

    def mark_completed(self, job_id: str, result: RunResult) -> None:
        with self._condition:
            record = self._records[job_id]
            record.status = "completed"
            record.result = result
            record.completed_at = datetime.now(timezone.utc)
            self._append_record_event(
                record,
                build_status_event(
                    sequence=0,
                    kind="job_completed",
                    message="Job completed.",
                    terminal=True,
                    data={"job_id": job_id, "stop_reason": result.stop_reason},
                ),
            )
            self._condition.notify_all()

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._condition:
            record = self._records[job_id]
            record.status = "failed"
            record.error = error
            record.completed_at = datetime.now(timezone.utc)
            self._append_record_event(
                record,
                build_status_event(
                    sequence=0,
                    kind="job_failed",
                    message="Job failed.",
                    terminal=True,
                    data={"job_id": job_id, "error": error},
                ),
            )
            self._condition.notify_all()

    def get(self, job_id: str) -> JobResponse:
        with self._lock:
            return self._to_response(self._records[job_id])

    def append_event(self, job_id: str, event: RunEvent) -> None:
        with self._condition:
            record = self._records[job_id]
            self._append_record_event(record, event)
            self._condition.notify_all()

    def list_events(self, job_id: str, *, after_sequence: int = 0) -> list[RunEvent]:
        with self._lock:
            record = self._records[job_id]
            events = record.events or []
            return [event.model_copy(deep=True) for event in events if event.sequence > after_sequence]

    def wait_for_events(
        self,
        job_id: str,
        *,
        after_sequence: int = 0,
        timeout: float = 15.0,
    ) -> list[RunEvent]:
        with self._condition:
            record = self._records[job_id]
            if self._events_after(record, after_sequence):
                return [event.model_copy(deep=True) for event in self._events_after(record, after_sequence)]
            self._condition.wait(timeout=timeout)
            return [event.model_copy(deep=True) for event in self._events_after(record, after_sequence)]

    def is_terminal(self, job_id: str) -> bool:
        with self._lock:
            return self._records[job_id].status in {"completed", "failed"}

    @staticmethod
    def _to_response(record: _JobRecord) -> JobResponse:
        return JobResponse(
            job_id=record.job_id,
            status=record.status,  # type: ignore[arg-type]
            created_at=record.created_at,
            started_at=record.started_at,
            completed_at=record.completed_at,
            result=record.result,
            error=record.error,
        )

    @staticmethod
    def _events_after(record: _JobRecord, after_sequence: int) -> list[RunEvent]:
        return [event for event in (record.events or []) if event.sequence > after_sequence]

    @staticmethod
    def _append_record_event(record: _JobRecord, event: RunEvent) -> None:
        copied = event.model_copy(
            update={
                "sequence": record.next_event_sequence,
            },
            deep=True,
        )
        record.next_event_sequence += 1
        if record.events is None:
            record.events = []
        record.events.append(copied)
