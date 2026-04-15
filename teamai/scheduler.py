"""
Recurring task scheduler for the teamAI daemon.

Reads schedule definitions from ``~/.teamai/schedules.json`` and fires
matching tasks through the daemon's job API at the right times.  Runs as a
background thread inside the FastAPI process so it shares the loaded model.

Schedule file format (``~/.teamai/schedules.json``)::

    [
      {
        "id": "nightly-evals",
        "cron": "0 3 * * *",
        "task": "Run the evaluation suite and save results",
        "execution_mode": "read_only",
        "enabled": true
      }
    ]

Cron fields: minute hour day-of-month month day-of-week (0=Sun or 7=Sun).
Supports: ``*``, integers, ranges (``1-5``), steps (``*/15``), and lists
(``1,15,30``).
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

GLOBAL_STATE_DIR = Path("~/.teamai")
SCHEDULES_FILE = "schedules.json"


# ---------------------------------------------------------------------------
# Cron expression parser (minimal, no deps)
# ---------------------------------------------------------------------------

def _parse_cron_field(expr: str, lo: int, hi: int) -> set[int]:
    """Parse a single cron field into a set of matching integers."""
    values: set[int] = set()
    for part in expr.split(","):
        part = part.strip()
        if part == "*":
            values.update(range(lo, hi + 1))
        elif "/" in part:
            base, step_s = part.split("/", 1)
            step = int(step_s)
            start = lo if base == "*" else int(base)
            values.update(range(start, hi + 1, step))
        elif "-" in part:
            a, b = part.split("-", 1)
            values.update(range(int(a), int(b) + 1))
        else:
            values.add(int(part))
    return values


def cron_matches(cron_expr: str, dt: datetime) -> bool:
    """Return True if *dt* matches the five-field cron expression."""
    fields = cron_expr.strip().split()
    if len(fields) != 5:
        raise ValueError(f"Cron expression must have exactly 5 fields, got {len(fields)}: {cron_expr!r}")

    minute, hour, dom, month, dow = fields
    minutes = _parse_cron_field(minute, 0, 59)
    hours = _parse_cron_field(hour, 0, 23)
    doms = _parse_cron_field(dom, 1, 31)
    months = _parse_cron_field(month, 1, 12)
    dows_raw = _parse_cron_field(dow, 0, 7)
    # Normalize: 7 → 0 (Sunday)
    dows = {d % 7 for d in dows_raw}

    return (
        dt.minute in minutes
        and dt.hour in hours
        and dt.day in doms
        and dt.month in months
        and dt.weekday() in _iso_weekday_to_cron(dows)
    )


def _iso_weekday_to_cron(cron_dows: set[int]) -> set[int]:
    """Convert cron day-of-week (0=Sun) to Python weekday (0=Mon)."""
    mapping = {0: 6, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
    return {mapping[d] for d in cron_dows if d in mapping}


# ---------------------------------------------------------------------------
# Schedule data model
# ---------------------------------------------------------------------------

@dataclass
class ScheduleEntry:
    id: str
    cron: str
    task: str
    execution_mode: str = "read_only"
    workspace: str | None = None
    max_rounds: int | None = None
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class ScheduleFireRecord:
    schedule_id: str
    fired_at: datetime
    job_id: str | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _schedules_path() -> Path:
    return GLOBAL_STATE_DIR.expanduser() / SCHEDULES_FILE


def load_schedules() -> list[ScheduleEntry]:
    """Load schedule entries from disk."""
    path = _schedules_path()
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Failed to read schedules file: %s", path)
        return []
    entries: list[ScheduleEntry] = []
    for item in raw:
        if isinstance(item, dict) and "id" in item and "cron" in item and "task" in item:
            entries.append(ScheduleEntry(
                id=item["id"],
                cron=item["cron"],
                task=item["task"],
                execution_mode=item.get("execution_mode", "read_only"),
                workspace=item.get("workspace"),
                max_rounds=item.get("max_rounds"),
                enabled=item.get("enabled", True),
            ))
    return entries


def save_schedules(entries: list[ScheduleEntry]) -> None:
    """Persist schedule entries to disk."""
    path = _schedules_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([e.to_dict() for e in entries], indent=2) + "\n",
        encoding="utf-8",
    )


def add_schedule(
    *,
    task: str,
    cron: str,
    schedule_id: str | None = None,
    execution_mode: str = "read_only",
    workspace: str | None = None,
    max_rounds: int | None = None,
    enabled: bool = True,
) -> ScheduleEntry:
    """Add a new schedule entry and persist it."""
    # Validate cron expression
    cron_matches(cron, datetime.now(timezone.utc))

    entry = ScheduleEntry(
        id=schedule_id or f"sched_{uuid.uuid4().hex[:8]}",
        cron=cron,
        task=task,
        execution_mode=execution_mode,
        workspace=workspace,
        max_rounds=max_rounds,
        enabled=enabled,
    )
    entries = load_schedules()
    # Replace existing entry with same id
    entries = [e for e in entries if e.id != entry.id]
    entries.append(entry)
    save_schedules(entries)
    return entry


def remove_schedule(schedule_id: str) -> bool:
    """Remove a schedule entry by id. Returns True if found and removed."""
    entries = load_schedules()
    filtered = [e for e in entries if e.id != schedule_id]
    if len(filtered) == len(entries):
        return False
    save_schedules(filtered)
    return True


# ---------------------------------------------------------------------------
# Scheduler thread
# ---------------------------------------------------------------------------

class TaskScheduler:
    """Background thread that checks schedules once per minute and fires jobs.

    ``fire_callback`` receives a :class:`ScheduleEntry` and should submit
    the task (e.g. via the daemon HTTP API or directly through the
    supervisor).  It must return a job-id string or raise on failure.
    """

    def __init__(self, fire_callback: Callable[[ScheduleEntry], str | None]) -> None:
        self._fire = fire_callback
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._fire_log: list[ScheduleFireRecord] = []
        self._last_fire_minute: dict[str, str] = {}  # schedule_id -> "YYYY-MM-DD HH:MM"

    @property
    def fire_log(self) -> list[ScheduleFireRecord]:
        return list(self._fire_log)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="teamai-scheduler")
        self._thread.start()
        logger.info("Scheduler started.")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        logger.info("Scheduler stopped.")

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("Scheduler tick failed.")
            # Sleep until the next minute boundary (± a small buffer)
            now = datetime.now(timezone.utc)
            seconds_to_next_minute = 60 - now.second
            self._stop_event.wait(timeout=seconds_to_next_minute + 0.5)

    def _tick(self) -> None:
        now = datetime.now(timezone.utc)
        minute_key = now.strftime("%Y-%m-%d %H:%M")
        entries = load_schedules()

        for entry in entries:
            if not entry.enabled:
                continue
            # Don't fire the same schedule twice in the same minute
            if self._last_fire_minute.get(entry.id) == minute_key:
                continue
            try:
                if not cron_matches(entry.cron, now):
                    continue
            except ValueError:
                logger.warning("Invalid cron expression for schedule %s: %s", entry.id, entry.cron)
                continue

            self._last_fire_minute[entry.id] = minute_key
            logger.info("Firing scheduled task %s: %s", entry.id, entry.task)

            record = ScheduleFireRecord(schedule_id=entry.id, fired_at=now)
            try:
                job_id = self._fire(entry)
                record.job_id = job_id
            except Exception as exc:
                record.error = str(exc)
                logger.exception("Failed to fire schedule %s", entry.id)
            self._fire_log.append(record)
