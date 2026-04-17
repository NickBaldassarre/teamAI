"""
Multi-agent spawning and team orchestration for teamAI.

Two layers:

1. **AgentSpawner** — low-level primitive.  Spawns a single agent run
   (synchronous or threaded), tracks parent-child relationships, and
   enforces depth limits.  Exposed to the planner as the ``spawn_agent``
   tool so any agent can delegate mid-execution.

2. **AgentTeam** — high-level orchestrator.  Decomposes a complex goal
   into a dependency DAG of subtasks, assigns agents via the registry,
   executes the DAG with parallelism where the graph allows, and
   synthesises a unified result.

Both use ``ClosedLoopSupervisor.run()`` internally — the supervisor
stays single-task-focused and this module adds the coordination layer.

Usage:

    from teamai.spawn import AgentTeam

    team = AgentTeam.from_env()
    result = team.run("Build a user auth system with tests")
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .schemas import (
    RunRequest,
    RunResult,
    SpawnedTaskRecord,
    TeamPlan,
)

logger = logging.getLogger(__name__)

MAX_SPAWN_DEPTH = 3
MAX_TEAM_TASKS = 12
_TEAM_WORKER_THREADS = 4

# Prompt templates live here to avoid coupling to the supervisor's prompts.
_DECOMPOSE_SYSTEM = """\
You are a task decomposition engine for a multi-agent coding system.
Break the user's goal into concrete, independently-executable subtasks.

Rules:
- Each subtask must be a self-contained instruction an agent can act on.
- Identify dependencies: if subtask B needs the output of subtask A,
  list A's id in B's depends_on array.
- Subtasks with no dependencies can run in parallel.
- Mark each subtask as read_only or workspace_write.
- Estimate complexity: low, medium, or high.
- Use at most {max_tasks} subtasks.
- Return ONLY valid JSON, no markdown fences or commentary.

Output format:
{{
  "subtasks": [
    {{
      "id": "t1",
      "task": "Clear instruction for what to do",
      "depends_on": [],
      "execution_mode": "read_only",
      "complexity": "low"
    }}
  ]
}}"""

_SYNTHESIZE_SYSTEM = """\
You are a synthesis engine for a multi-agent coding system.
Multiple agents have each completed a subtask of a larger goal.
Summarise what was accomplished, noting any failures or open items.
Be concise — focus on outcomes, not process."""


# ---------------------------------------------------------------------------
# AgentSpawner — single-agent spawn primitive
# ---------------------------------------------------------------------------

class AgentSpawner:
    """Spawn individual agent runs with depth tracking.

    Each spawn creates a ``ClosedLoopSupervisor`` (or reuses the one
    passed in) and calls ``run()`` directly.  Async spawns use a
    daemon thread and store results in ``_results``.
    """

    def __init__(
        self,
        *,
        settings: Any | None = None,
        supervisor: Any | None = None,
        event_callback: Callable | None = None,
    ) -> None:
        self._settings = settings
        self._supervisor = supervisor
        self._event_callback = event_callback
        self._lock = threading.Lock()
        self._results: dict[str, RunResult | None] = {}
        self._errors: dict[str, str] = {}
        self._events: dict[str, threading.Event] = {}

    def _get_supervisor(self) -> Any:
        if self._supervisor is not None:
            return self._supervisor
        from .config import Settings
        from .supervisor import ClosedLoopSupervisor
        settings = self._settings or Settings.from_env()
        self._supervisor = ClosedLoopSupervisor(settings)
        return self._supervisor

    # ---- synchronous spawn ----

    def spawn_sync(
        self,
        task: str,
        *,
        workspace_path: str | None = None,
        execution_mode: str = "read_only",
        parent_task_id: str | None = None,
        spawn_depth: int = 0,
        max_spawn_depth: int = MAX_SPAWN_DEPTH,
        max_rounds: int | None = None,
        handoff_engine: str | None = None,
        agent_id: str | None = None,
    ) -> RunResult:
        """Spawn and block until the agent completes."""
        if spawn_depth >= max_spawn_depth:
            raise SpawnDepthExceeded(
                f"Spawn depth {spawn_depth} reached limit {max_spawn_depth}."
            )

        request = RunRequest(
            task=task,
            workspace_path=workspace_path,
            execution_mode=execution_mode,
            parent_task_id=parent_task_id,
            spawn_depth=spawn_depth,
            max_spawn_depth=max_spawn_depth,
            max_rounds=max_rounds,
            handoff_engine=handoff_engine,
        )
        supervisor = self._get_supervisor()
        return supervisor.run(request, event_callback=self._event_callback)

    # ---- async (threaded) spawn ----

    def spawn_async(
        self,
        task: str,
        *,
        spawn_id: str | None = None,
        workspace_path: str | None = None,
        execution_mode: str = "read_only",
        parent_task_id: str | None = None,
        spawn_depth: int = 0,
        max_spawn_depth: int = MAX_SPAWN_DEPTH,
        max_rounds: int | None = None,
        handoff_engine: str | None = None,
    ) -> str:
        """Spawn in a background thread and return the spawn_id immediately."""
        sid = spawn_id or f"spawn_{uuid.uuid4().hex[:8]}"
        done_event = threading.Event()
        with self._lock:
            self._results[sid] = None
            self._events[sid] = done_event

        def _worker() -> None:
            try:
                result = self.spawn_sync(
                    task,
                    workspace_path=workspace_path,
                    execution_mode=execution_mode,
                    parent_task_id=parent_task_id,
                    spawn_depth=spawn_depth,
                    max_spawn_depth=max_spawn_depth,
                    max_rounds=max_rounds,
                    handoff_engine=handoff_engine,
                )
                with self._lock:
                    self._results[sid] = result
            except Exception as exc:
                with self._lock:
                    self._errors[sid] = str(exc)
                logger.exception("Spawned agent %s failed", sid)
            finally:
                done_event.set()

        threading.Thread(target=_worker, daemon=True, name=f"teamai-{sid}").start()
        return sid

    def wait(self, spawn_id: str, *, timeout: float | None = None) -> RunResult | None:
        """Block until a spawned agent finishes. Returns the result or None on timeout."""
        event = self._events.get(spawn_id)
        if event is None:
            raise KeyError(f"Unknown spawn_id: {spawn_id}")
        event.wait(timeout=timeout)
        with self._lock:
            if spawn_id in self._errors:
                raise SpawnFailed(self._errors[spawn_id])
            return self._results.get(spawn_id)

    def get_result(self, spawn_id: str) -> RunResult | None:
        with self._lock:
            return self._results.get(spawn_id)

    def get_error(self, spawn_id: str) -> str | None:
        with self._lock:
            return self._errors.get(spawn_id)


# ---------------------------------------------------------------------------
# AgentTeam — DAG-based multi-agent orchestrator
# ---------------------------------------------------------------------------

class AgentTeam:
    """Decomposes a goal into subtasks, spawns agents in dependency order,
    and synthesises a unified result.

    The decomposition uses the local model (or a cloud model if provided)
    to break the goal into a DAG of subtasks.  Subtasks whose
    dependencies are met run in parallel via a thread pool.
    """

    def __init__(
        self,
        *,
        settings: Any | None = None,
        supervisor: Any | None = None,
        progress_callback: Callable[[str], None] | None = None,
        max_tasks: int = MAX_TEAM_TASKS,
        max_workers: int = _TEAM_WORKER_THREADS,
    ) -> None:
        self._settings = settings
        self._supervisor = supervisor
        self._progress = progress_callback or (lambda msg: None)
        self._max_tasks = max_tasks
        self._max_workers = max_workers
        self._spawner: AgentSpawner | None = None

    @classmethod
    def from_env(
        cls,
        *,
        progress_callback: Callable[[str], None] | None = None,
    ) -> AgentTeam:
        from .config import Settings
        return cls(settings=Settings.from_env(), progress_callback=progress_callback)

    def _get_settings(self) -> Any:
        if self._settings is not None:
            return self._settings
        from .config import Settings
        self._settings = Settings.from_env()
        return self._settings

    def _get_supervisor(self) -> Any:
        if self._supervisor is not None:
            return self._supervisor
        from .supervisor import ClosedLoopSupervisor
        self._supervisor = ClosedLoopSupervisor(self._get_settings())
        return self._supervisor

    def _get_spawner(self) -> AgentSpawner:
        if self._spawner is None:
            self._spawner = AgentSpawner(
                settings=self._get_settings(),
                supervisor=self._get_supervisor(),
            )
        return self._spawner

    # ---- full pipeline ----

    def run(
        self,
        goal: str,
        *,
        workspace_path: str | None = None,
        spawn_depth: int = 0,
        max_spawn_depth: int = MAX_SPAWN_DEPTH,
    ) -> TeamPlan:
        """Decompose → execute → synthesise.  Returns the completed plan."""
        plan = self.decompose(goal, workspace_path=workspace_path, spawn_depth=spawn_depth)
        plan = self.execute(
            plan,
            workspace_path=workspace_path,
            spawn_depth=spawn_depth,
            max_spawn_depth=max_spawn_depth,
        )
        plan = self.synthesize(plan, workspace_path=workspace_path)
        return plan

    # ---- decomposition ----

    def decompose(
        self,
        goal: str,
        *,
        workspace_path: str | None = None,
        spawn_depth: int = 0,
    ) -> TeamPlan:
        """Use the model to break *goal* into a DAG of subtasks."""
        self._progress(f"Decomposing goal into subtasks: {goal[:80]}")

        team_id = f"team_{uuid.uuid4().hex[:8]}"
        plan = TeamPlan(
            team_id=team_id,
            goal=goal,
            status="planning",
            spawn_depth=spawn_depth,
        )

        subtasks_json = self._call_model_for_decomposition(goal, workspace_path)
        if subtasks_json is None:
            # Model couldn't decompose — create a single-task plan
            plan.tasks = [
                SpawnedTaskRecord(
                    spawn_id="t1",
                    task=goal,
                    execution_mode="read_only",
                ),
            ]
            self._progress("Could not decompose; running as single task.")
        else:
            plan.tasks = self._parse_decomposition(subtasks_json, team_id)
            task_count = len(plan.tasks)
            parallel = sum(1 for t in plan.tasks if not t.depends_on)
            self._progress(
                f"Plan: {task_count} subtasks, "
                f"{parallel} can start immediately."
            )

        plan.status = "executing"
        return plan

    def _call_model_for_decomposition(
        self,
        goal: str,
        workspace_path: str | None,
    ) -> str | None:
        """Ask the model to produce a subtask JSON."""
        supervisor = self._get_supervisor()

        system_prompt = _DECOMPOSE_SYSTEM.format(max_tasks=self._max_tasks)
        user_prompt = f"Goal: {goal}"
        if workspace_path:
            user_prompt += f"\nWorkspace: {workspace_path}"

        try:
            raw = supervisor.generate_raw(
                system=system_prompt,
                user=user_prompt,
                max_tokens=1024,
                temperature=0.2,
            )
            return raw.strip()
        except Exception:
            logger.exception("Decomposition model call failed")
            return None

    def _parse_decomposition(
        self,
        raw_json: str,
        team_id: str,
    ) -> list[SpawnedTaskRecord]:
        """Parse the model's JSON output into SpawnedTaskRecords."""
        # Strip markdown fences if present
        text = raw_json.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Decomposition output was not valid JSON: %.200s", text)
            return []

        raw_tasks = data.get("subtasks", []) if isinstance(data, dict) else []
        records: list[SpawnedTaskRecord] = []
        for item in raw_tasks[: self._max_tasks]:
            if not isinstance(item, dict) or "task" not in item:
                continue
            records.append(
                SpawnedTaskRecord(
                    spawn_id=str(item.get("id", f"t{len(records) + 1}")),
                    parent_id=team_id,
                    task=str(item["task"]),
                    execution_mode=str(item.get("execution_mode", "read_only")),
                    depends_on=[str(d) for d in item.get("depends_on", [])],
                )
            )
        return records

    # ---- execution ----

    def execute(
        self,
        plan: TeamPlan,
        *,
        workspace_path: str | None = None,
        spawn_depth: int = 0,
        max_spawn_depth: int = MAX_SPAWN_DEPTH,
    ) -> TeamPlan:
        """Execute the DAG: run ready tasks in parallel, wait, repeat."""
        spawner = self._get_spawner()
        plan.status = "executing"

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="teamai-team",
        ) as pool:
            while not plan.all_terminal():
                ready = plan.ready_tasks()
                if not ready:
                    if plan.is_blocked():
                        self._progress("Plan blocked: a dependency failed.")
                        # Mark remaining pending tasks as blocked
                        failed_ids = {t.spawn_id for t in plan.tasks if t.status == "failed"}
                        for t in plan.tasks:
                            if t.status == "pending" and any(d in failed_ids for d in t.depends_on):
                                t.status = "blocked"
                                t.error = "Dependency failed."
                        break
                    # Should not happen, but guard against infinite loop
                    break

                futures: dict[concurrent.futures.Future[RunResult | None], SpawnedTaskRecord] = {}
                for task_record in ready:
                    task_record.status = "running"
                    task_record.started_at = datetime.now(timezone.utc)
                    self._progress(f"Spawning: {task_record.spawn_id} — {task_record.task[:60]}")

                    future = pool.submit(
                        self._execute_subtask,
                        spawner=spawner,
                        task_record=task_record,
                        workspace_path=workspace_path,
                        spawn_depth=spawn_depth + 1,
                        max_spawn_depth=max_spawn_depth,
                    )
                    futures[future] = task_record

                for future in concurrent.futures.as_completed(futures):
                    task_record = futures[future]
                    try:
                        result = future.result()
                        task_record.completed_at = datetime.now(timezone.utc)
                        if result is not None:
                            task_record.status = "completed" if result.status == "completed" else "failed"
                            task_record.result_summary = result.final_answer[:2000] if result.final_answer else None
                            if result.status != "completed":
                                task_record.error = result.stop_reason
                        else:
                            task_record.status = "failed"
                            task_record.error = "No result returned."
                    except Exception as exc:
                        task_record.status = "failed"
                        task_record.error = str(exc)
                        task_record.completed_at = datetime.now(timezone.utc)

                    status_icon = "+" if task_record.status == "completed" else "x"
                    self._progress(
                        f"[{status_icon}] {task_record.spawn_id}: {task_record.status}"
                    )

        return plan

    @staticmethod
    def _execute_subtask(
        *,
        spawner: AgentSpawner,
        task_record: SpawnedTaskRecord,
        workspace_path: str | None,
        spawn_depth: int,
        max_spawn_depth: int,
    ) -> RunResult | None:
        return spawner.spawn_sync(
            task_record.task,
            workspace_path=workspace_path,
            execution_mode=task_record.execution_mode,
            parent_task_id=task_record.parent_id,
            spawn_depth=spawn_depth,
            max_spawn_depth=max_spawn_depth,
        )

    # ---- synthesis ----

    def synthesize(
        self,
        plan: TeamPlan,
        *,
        workspace_path: str | None = None,
    ) -> TeamPlan:
        """Aggregate subtask results into a unified summary."""
        plan.status = "synthesizing"
        self._progress("Synthesising results from all agents...")

        completed = [t for t in plan.tasks if t.status == "completed"]
        failed = [t for t in plan.tasks if t.status in ("failed", "blocked")]

        if not completed and failed:
            plan.synthesis = "All subtasks failed. No results to synthesise."
            plan.status = "failed"
            plan.completed_at = datetime.now(timezone.utc)
            return plan

        # Build a context block from all subtask results
        parts: list[str] = [f"Goal: {plan.goal}\n"]
        for t in plan.tasks:
            status_line = f"[{t.status}] {t.spawn_id}: {t.task}"
            if t.result_summary:
                status_line += f"\n  Result: {t.result_summary[:500]}"
            if t.error:
                status_line += f"\n  Error: {t.error[:200]}"
            parts.append(status_line)

        context = "\n\n".join(parts)

        try:
            supervisor = self._get_supervisor()
            synthesis = supervisor.generate_raw(
                system=_SYNTHESIZE_SYSTEM,
                user=context,
                max_tokens=1024,
                temperature=0.3,
            )
            plan.synthesis = synthesis.strip()
        except Exception:
            # Fall back to a mechanical summary if the model is unavailable
            logger.exception("Synthesis model call failed, using mechanical summary")
            lines = [f"Team completed {len(completed)}/{len(plan.tasks)} subtasks."]
            for t in completed:
                lines.append(f"  - {t.spawn_id}: {(t.result_summary or 'done')[:100]}")
            for t in failed:
                lines.append(f"  - {t.spawn_id}: FAILED — {(t.error or 'unknown')[:100]}")
            plan.synthesis = "\n".join(lines)

        plan.status = "completed" if not failed else "partially_completed"
        plan.completed_at = datetime.now(timezone.utc)
        return plan


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class SpawnDepthExceeded(RuntimeError):
    """Raised when a spawn would exceed the maximum nesting depth."""


class SpawnFailed(RuntimeError):
    """Raised when a spawned agent failed."""
