from __future__ import annotations

import json
import queue
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .config import ConfigError, Settings
from .conversation import ConversationChannel
from .events import render_sse_event
from .jobs import InMemoryJobStore
from .scheduler import (
    ScheduleEntry,
    TaskScheduler,
    add_schedule,
    load_schedules,
    remove_schedule,
)
from .schemas import JobResponse, RunEvent, RunRequest, RunResult, TeamPlan
from .supervisor import ClosedLoopSupervisor
from .tool_api import create_tool_router


def create_app(
    settings: Settings | None = None,
    *,
    supervisor: ClosedLoopSupervisor | None = None,
    jobs: InMemoryJobStore | None = None,
) -> FastAPI:
    app_settings = settings or Settings.from_env()
    app_supervisor = supervisor or ClosedLoopSupervisor(app_settings)
    job_store = jobs or InMemoryJobStore()

    # Scheduler is created here so the lifespan can reference it, but
    # the fire callback is wired up below after route definitions.
    _scheduler_holder: list[TaskScheduler | None] = [None]

    @asynccontextmanager
    async def _lifespan(app: FastAPI):  # type: ignore[no-untyped-def]
        sched = _scheduler_holder[0]
        if sched is not None:
            sched.start()
        yield
        if sched is not None:
            sched.stop()

    app = FastAPI(
        title="teamAI Local Loop",
        version="0.1.0",
        description="Local-first closed-loop orchestration for Apple Silicon using MLX.",
        lifespan=_lifespan,
    )

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "model_id": app_settings.model_id,
            "model_loaded": app_supervisor.model_loaded,
            "workspace_root": str(app_settings.workspace_root),
        }

    @app.post("/v1/run", response_model=RunResult)
    def run_once(request: RunRequest) -> RunResult:
        try:
            return app_supervisor.run(request)
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/run/stream")
    def run_once_stream(request: RunRequest) -> StreamingResponse:
        event_queue: queue.Queue[tuple[str, dict[str, object] | None]] = queue.Queue()

        def _worker() -> None:
            try:
                result = app_supervisor.run(
                    request,
                    event_callback=lambda event: event_queue.put(("run_event", event.model_dump(mode="json"))),
                )
                event_queue.put(("run_result", result.model_dump(mode="json")))
            except ConfigError as exc:
                event_queue.put(("run_error", {"error": str(exc)}))
            except Exception as exc:  # pragma: no cover - defensive surface
                event_queue.put(("run_error", {"error": str(exc)}))
            finally:
                event_queue.put(("done", None))

        def _stream() -> object:
            while True:
                event_name, payload = event_queue.get()
                if event_name == "done":
                    break
                if payload is None:
                    continue
                yield render_sse_event(event=event_name, payload=payload)

        threading.Thread(target=_worker, daemon=True).start()
        return StreamingResponse(_stream(), media_type="text/event-stream")

    @app.post("/v1/jobs", response_model=JobResponse)
    def create_job(request: RunRequest) -> JobResponse:
        record = job_store.create(request)

        def _worker() -> None:
            job_store.mark_running(record.job_id)
            try:
                result = app_supervisor.run(
                    request,
                    event_callback=lambda event: job_store.append_event(record.job_id, event),
                )
                job_store.mark_completed(record.job_id, result)
            except Exception as exc:
                job_store.mark_failed(record.job_id, str(exc))

        threading.Thread(target=_worker, daemon=True).start()
        return record

    @app.get("/v1/jobs/{job_id}", response_model=JobResponse)
    def get_job(job_id: str) -> JobResponse:
        try:
            return job_store.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    @app.get("/v1/jobs/{job_id}/events", response_model=list[RunEvent])
    def get_job_events(job_id: str, after_sequence: int = 0) -> list[RunEvent]:
        try:
            return job_store.list_events(job_id, after_sequence=after_sequence)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    @app.get("/v1/jobs/{job_id}/events/stream")
    def stream_job_events(job_id: str, after_sequence: int = 0) -> StreamingResponse:
        try:
            job_store.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

        def _stream() -> object:
            next_sequence = after_sequence
            while True:
                events = job_store.wait_for_events(job_id, after_sequence=next_sequence, timeout=5.0)
                if events:
                    for event in events:
                        next_sequence = event.sequence
                        yield render_sse_event(event="run_event", payload=event.model_dump(mode="json"))
                elif job_store.is_terminal(job_id):
                    break
                else:
                    yield ": keep-alive\n\n"

                if job_store.is_terminal(job_id):
                    remaining = job_store.list_events(job_id, after_sequence=next_sequence)
                    if not remaining:
                        break

        return StreamingResponse(_stream(), media_type="text/event-stream")

    # ---------------------------------------------------------------- scheduler
    def _fire_scheduled_task(entry: ScheduleEntry) -> str | None:
        """Submit a scheduled task as a background job. Returns job_id."""
        request = RunRequest(
            task=entry.task,
            workspace_path=entry.workspace,
            execution_mode=entry.execution_mode,  # type: ignore[arg-type]
            max_rounds=entry.max_rounds,
        )
        record = job_store.create(request)

        def _worker() -> None:
            job_store.mark_running(record.job_id)
            try:
                result = app_supervisor.run(
                    request,
                    event_callback=lambda event: job_store.append_event(record.job_id, event),
                )
                job_store.mark_completed(record.job_id, result)
            except Exception as exc:
                job_store.mark_failed(record.job_id, str(exc))

        threading.Thread(target=_worker, daemon=True).start()
        return record.job_id

    scheduler = TaskScheduler(fire_callback=_fire_scheduled_task)
    _scheduler_holder[0] = scheduler

    @app.get("/v1/schedules")
    def list_schedules() -> list[dict[str, object]]:
        return [e.to_dict() for e in load_schedules()]

    @app.post("/v1/schedules")
    def create_schedule(body: dict[str, object]) -> dict[str, object]:
        task = body.get("task")
        cron = body.get("cron")
        if not task or not cron:
            raise HTTPException(status_code=400, detail="'task' and 'cron' are required.")
        entry = add_schedule(
            task=str(task),
            cron=str(cron),
            schedule_id=str(body["id"]) if "id" in body else None,
            execution_mode=str(body.get("execution_mode", "read_only")),
            workspace=str(body["workspace"]) if "workspace" in body else None,
            max_rounds=int(body["max_rounds"]) if "max_rounds" in body else None,
            enabled=bool(body.get("enabled", True)),
        )
        return entry.to_dict()

    @app.delete("/v1/schedules/{schedule_id}")
    def delete_schedule(schedule_id: str) -> dict[str, object]:
        if not remove_schedule(schedule_id):
            raise HTTPException(status_code=404, detail="Schedule not found.")
        return {"status": "removed", "id": schedule_id}

    @app.get("/v1/schedules/log")
    def schedule_fire_log() -> list[dict[str, object]]:
        return [
            {
                "schedule_id": r.schedule_id,
                "fired_at": r.fired_at.isoformat(),
                "job_id": r.job_id,
                "error": r.error,
            }
            for r in scheduler.fire_log
        ]

    # ------------------------------------------------------------- team
    _team_store: dict[str, TeamPlan] = {}

    @app.post("/v1/team")
    def create_team(body: dict[str, object]) -> dict[str, object]:
        goal = body.get("goal")
        if not goal:
            raise HTTPException(status_code=400, detail="'goal' is required.")

        from .spawn import AgentTeam

        team = AgentTeam(
            settings=app_settings,
            supervisor=app_supervisor,
        )
        plan = team.decompose(
            str(goal),
            workspace_path=str(body.get("workspace_path", "")) or None,
        )
        _team_store[plan.team_id] = plan

        # Execute in a background thread so the endpoint returns immediately
        def _run_team() -> None:
            try:
                team.execute(
                    plan,
                    workspace_path=str(body.get("workspace_path", "")) or None,
                )
                team.synthesize(plan)
            except Exception as exc:
                plan.status = "failed"
                plan.synthesis = f"Team execution failed: {exc}"

        threading.Thread(target=_run_team, daemon=True).start()
        return plan.model_dump(mode="json")

    @app.get("/v1/team/{team_id}")
    def get_team(team_id: str) -> dict[str, object]:
        plan = _team_store.get(team_id)
        if plan is None:
            raise HTTPException(status_code=404, detail="Team not found.")
        return plan.model_dump(mode="json")

    @app.get("/v1/team/{team_id}/tasks")
    def get_team_tasks(team_id: str) -> list[dict[str, object]]:
        plan = _team_store.get(team_id)
        if plan is None:
            raise HTTPException(status_code=404, detail="Team not found.")
        return [t.model_dump(mode="json") for t in plan.tasks]

    # ------------------------------------------------------------- tool API
    workspace = app_settings.resolve_workspace(None)
    tool_router = create_tool_router(settings=app_settings, workspace=workspace)
    app.include_router(tool_router)

    # ------------------------------------------------------------- conversation
    conversation = ConversationChannel(workspace)

    @app.post("/v1/conversation/{task_id}/post")
    def post_turn(task_id: str, body: dict[str, object]) -> dict[str, object]:
        agent_id = body.get("agent_id")
        role = body.get("role")
        content = body.get("content")
        if not agent_id or not role or not content:
            raise HTTPException(status_code=400, detail="'agent_id', 'role', and 'content' are required.")
        metadata = body.get("metadata") or {}
        return conversation.post(
            task_id=task_id,
            agent_id=str(agent_id),
            role=str(role),
            content=str(content),
            metadata=metadata if isinstance(metadata, dict) else {},
        )

    @app.get("/v1/conversation/{task_id}")
    def get_conversation(task_id: str, after_turn: str | None = None, limit: int = 50) -> list[dict[str, object]]:
        return conversation.read(task_id, after_turn=after_turn, limit=limit)

    @app.get("/v1/conversation/{task_id}/summary")
    def get_conversation_summary(task_id: str) -> dict[str, object]:
        return conversation.summary(task_id)

    @app.get("/v1/conversations")
    def list_conversations() -> list[str]:
        return conversation.list_conversations()

    return app


app = create_app()
