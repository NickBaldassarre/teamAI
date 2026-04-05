from __future__ import annotations

import queue
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .config import ConfigError, Settings
from .events import render_sse_event
from .jobs import InMemoryJobStore
from .schemas import JobResponse, RunEvent, RunRequest, RunResult
from .supervisor import ClosedLoopSupervisor


def create_app(
    settings: Settings | None = None,
    *,
    supervisor: ClosedLoopSupervisor | None = None,
    jobs: InMemoryJobStore | None = None,
) -> FastAPI:
    app_settings = settings or Settings.from_env()
    app_supervisor = supervisor or ClosedLoopSupervisor(app_settings)
    job_store = jobs or InMemoryJobStore()

    app = FastAPI(
        title="teamAI Local Loop",
        version="0.1.0",
        description="Local-first closed-loop orchestration for Apple Silicon using MLX.",
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

    return app


app = create_app()
