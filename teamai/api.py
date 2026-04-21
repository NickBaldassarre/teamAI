from __future__ import annotations

import json
import queue
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse

from .approvals import PatchApprovalStore
from .config import ConfigError, Settings
from .conversation import ConversationChannel
from .dashboard import render_dashboard_html
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


def _build_supervisor_factory(
    settings: Settings,
    supervisor: Any | None,
) -> tuple[Callable[[], Any], Any]:
    template = supervisor or ClosedLoopSupervisor(settings)
    if hasattr(template, "isolated_copy"):
        return lambda: template.isolated_copy(), template
    return lambda: template, template


def create_app(
    settings: Settings | None = None,
    *,
    supervisor: Any | None = None,
    jobs: InMemoryJobStore | None = None,
) -> FastAPI:
    app_settings = settings or Settings.from_env()
    build_supervisor, supervisor_template = _build_supervisor_factory(app_settings, supervisor)
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

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/dashboard")

    @app.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
    def dashboard() -> HTMLResponse:
        return HTMLResponse(render_dashboard_html())

    @app.get("/healthz")
    def healthz() -> dict[str, object]:
        return {
            "status": "ok",
            "model_id": app_settings.model_id,
            "model_loaded": getattr(supervisor_template, "model_loaded", False),
            "workspace_root": str(app_settings.workspace_root),
        }

    @app.get("/v1/dashboard/summary")
    def dashboard_summary(limit: int = 8) -> dict[str, object]:
        job_limit = max(1, min(limit, 25))
        recent_jobs = [job.model_dump(mode="json") for job in job_store.list_recent(limit=job_limit)]
        schedules = [entry.to_dict() for entry in load_schedules()]
        team_items = [
            {
                "team_id": plan.team_id,
                "goal": plan.goal,
                "status": plan.status,
                "task_count": len(plan.tasks),
                "completed_tasks": sum(1 for task in plan.tasks if task.status == "completed"),
                "created_at": plan.created_at.isoformat(),
                "completed_at": plan.completed_at.isoformat() if plan.completed_at else None,
            }
            for plan in sorted(
                _team_store.values(),
                key=lambda item: item.created_at,
                reverse=True,
            )[:job_limit]
        ]
        conversation_ids = conversation.list_conversations()
        conversation_items = [
            conversation.summary(task_id)
            for task_id in sorted(
                conversation_ids,
                key=lambda task_id: conversation.summary(task_id).get("latest_timestamp") or "",
                reverse=True,
            )[:job_limit]
        ]
        pending_approvals = _list_pending_approvals(limit=job_limit)
        return {
            "health": {
                "status": "ok",
                "model_id": app_settings.model_id,
                "model_loaded": getattr(supervisor_template, "model_loaded", False),
                "workspace_root": str(app_settings.workspace_root),
                "server_time": datetime.now(timezone.utc).isoformat(),
            },
            "safety": {
                "allow_writes": bool(app_settings.allow_writes),
                "allow_shell": bool(app_settings.allow_shell),
                "posture": "writes_enabled" if app_settings.allow_writes else "read_only",
            },
            "jobs": {
                "counts": job_store.status_counts(),
                "recent": recent_jobs,
            },
            "schedules": {
                "count": len(schedules),
                "items": schedules[:job_limit],
            },
            "teams": {
                "count": len(_team_store),
                "items": team_items,
            },
            "conversations": {
                "count": len(conversation_ids),
                "items": conversation_items,
            },
            "approvals": {
                "count": len(pending_approvals),
                "items": pending_approvals,
            },
        }

    @app.post("/v1/run", response_model=RunResult)
    def run_once(request: RunRequest) -> RunResult:
        try:
            return build_supervisor().run(request)
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/run/stream")
    def run_once_stream(request: RunRequest) -> StreamingResponse:
        event_queue: queue.Queue[tuple[str, dict[str, object] | None]] = queue.Queue()

        def _worker() -> None:
            try:
                result = build_supervisor().run(
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
                result = build_supervisor().run(
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
                result = build_supervisor().run(
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
        if not app_settings.allow_writes:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Creating or updating schedules is disabled because TEAMAI_ALLOW_WRITES is false. "
                    "Set TEAMAI_ALLOW_WRITES=true to manage scheduled automation."
                ),
            )
        task = body.get("task")
        cron = body.get("cron")
        if not task or not cron:
            raise HTTPException(status_code=400, detail="'task' and 'cron' are required.")
        try:
            entry = add_schedule(
                task=str(task),
                cron=str(cron),
                schedule_id=str(body["id"]) if "id" in body else None,
                execution_mode=str(body.get("execution_mode", "read_only")),
                workspace=str(body["workspace"]) if "workspace" in body else None,
                max_rounds=int(body["max_rounds"]) if "max_rounds" in body else None,
                enabled=bool(body.get("enabled", True)),
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"Invalid schedule: {exc}") from exc
        return entry.to_dict()

    @app.delete("/v1/schedules/{schedule_id}")
    def delete_schedule(schedule_id: str) -> dict[str, object]:
        if not app_settings.allow_writes:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Deleting schedules is disabled because TEAMAI_ALLOW_WRITES is false. "
                    "Set TEAMAI_ALLOW_WRITES=true to manage scheduled automation."
                ),
            )
        if not remove_schedule(schedule_id):
            raise HTTPException(status_code=404, detail="Schedule not found.")
        return {"status": "removed", "id": schedule_id}

    @app.post("/v1/schedules/{schedule_id}/toggle")
    def toggle_schedule(schedule_id: str) -> dict[str, object]:
        if not app_settings.allow_writes:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Toggling schedules is disabled because TEAMAI_ALLOW_WRITES is false. "
                    "Set TEAMAI_ALLOW_WRITES=true to manage scheduled automation."
                ),
            )
        existing = next((e for e in load_schedules() if e.id == schedule_id), None)
        if existing is None:
            raise HTTPException(status_code=404, detail="Schedule not found.")
        updated = add_schedule(
            task=existing.task,
            cron=existing.cron,
            schedule_id=existing.id,
            execution_mode=existing.execution_mode,
            workspace=existing.workspace,
            max_rounds=existing.max_rounds,
            enabled=not existing.enabled,
        )
        return {"status": "toggled", "schedule": updated.to_dict()}

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
            supervisor_factory=build_supervisor,
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

    approval_store = PatchApprovalStore()

    def _list_pending_approvals(*, limit: int) -> list[dict[str, object]]:
        try:
            records = approval_store.list(workspace=workspace, include_all=False)
        except Exception:  # pragma: no cover - defensive: corrupt approval files
            return []
        summaries = [approval_store.summarize(record) for record in records[:limit]]
        return summaries

    @app.get("/v1/approvals/{approval_id}")
    def get_approval(approval_id: str) -> dict[str, object]:
        try:
            payload = approval_store.get(workspace=workspace, approval_id=approval_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return approval_store.summarize(payload, include_diff=True)

    @app.post("/v1/approvals/{approval_id}/apply")
    def apply_approval(approval_id: str) -> dict[str, object]:
        if not app_settings.allow_writes:
            raise HTTPException(
                status_code=403,
                detail=(
                    "Applying approvals is disabled because TEAMAI_ALLOW_WRITES is false. "
                    "Set TEAMAI_ALLOW_WRITES=true to permit approved patches to touch the workspace."
                ),
            )
        try:
            payload = approval_store.apply(workspace=workspace, approval_id=approval_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "status": "applied",
            "approval": approval_store.summarize(payload),
        }

    @app.post("/v1/approvals/{approval_id}/reject")
    def reject_approval(approval_id: str, body: dict[str, object] | None = None) -> dict[str, object]:
        reason_raw = (body or {}).get("reason") if isinstance(body, dict) else None
        reason = str(reason_raw).strip() if isinstance(reason_raw, str) else None
        try:
            payload = approval_store.reject(
                workspace=workspace,
                approval_id=approval_id,
                reason=reason,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {
            "status": "rejected",
            "approval": approval_store.summarize(payload),
        }

    @app.post("/v1/approvals/prune-stale")
    def prune_stale_approvals() -> dict[str, object]:
        try:
            pruned = approval_store.prune_stale(workspace=workspace)
        except Exception as exc:  # pragma: no cover - defensive: corrupt approval files
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return {
            "status": "pruned",
            "count": len(pruned),
            "approvals": [approval_store.summarize(record) for record in pruned],
        }

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
