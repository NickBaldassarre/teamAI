from __future__ import annotations

import json
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from teamai.api import create_app
from teamai.config import Settings
from teamai.events import build_run_event
from teamai.jobs import InMemoryJobStore
from teamai.schemas import RunRequest, RunResult


class _FakeSupervisor:
    def __init__(self) -> None:
        self.model_loaded = False

    def run(self, request, progress_callback=None, event_callback=None):  # noqa: ANN001
        messages = [
            "Starting run in /tmp/demo (mode=read_only, max_rounds=1, max_actions=1)",
            "Task route: repository_inspection",
            "Round 1/1: planner",
            "Completed: verifier_declared_complete",
        ]
        sequence = 0
        for message in messages:
            if progress_callback is not None:
                progress_callback(message)
            if event_callback is not None:
                sequence += 1
                event_callback(build_run_event(sequence=sequence, message=message))
        return RunResult(
            status="completed",
            model_id="dummy-model",
            workspace="/tmp/demo",
            execution_mode=request.execution_mode,
            stop_reason="verifier_declared_complete",
            final_answer="Done.",
            transcript="demo transcript",
            warnings=[],
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )


class APIStreamingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.settings = Settings(
            model_id="dummy",
            model_revision=None,
            force_download=False,
            trust_remote_code=False,
            enable_thinking=False,
            workspace_root=self.workspace,
            max_rounds=2,
            max_actions_per_round=2,
            max_tokens_per_turn=64,
            temperature=0.3,
            allow_shell=False,
            allow_writes=False,
            command_timeout_seconds=5,
            max_file_bytes=10_000,
            max_command_output_chars=10_000,
            host="127.0.0.1",
            port=8000,
        )
        self.jobs = InMemoryJobStore()
        self.client = TestClient(create_app(self.settings, supervisor=_FakeSupervisor(), jobs=self.jobs))

    def tearDown(self) -> None:
        self.client.close()
        self.temp_dir.cleanup()

    def test_run_stream_endpoint_emits_events_and_final_result(self) -> None:
        response = self.client.post(
            "/v1/run/stream",
            json={"task": "Inspect repo.", "workspace_path": ".", "execution_mode": "read_only"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: run_event", response.text)
        self.assertIn("event: run_result", response.text)
        self.assertIn('"kind": "task_route_selected"', response.text)
        self.assertIn('"stop_reason": "verifier_declared_complete"', response.text)

    def test_job_event_endpoints_store_and_stream_events(self) -> None:
        create_response = self.client.post(
            "/v1/jobs",
            json={"task": "Inspect repo.", "workspace_path": ".", "execution_mode": "read_only"},
        )
        self.assertEqual(create_response.status_code, 200)
        job_id = create_response.json()["job_id"]

        for _ in range(50):
            job_response = self.client.get(f"/v1/jobs/{job_id}")
            if job_response.json()["status"] == "completed":
                break
            time.sleep(0.01)
        else:  # pragma: no cover - defensive timeout
            self.fail("job did not complete in time")

        events_response = self.client.get(f"/v1/jobs/{job_id}/events")
        self.assertEqual(events_response.status_code, 200)
        events = events_response.json()
        kinds = [event["kind"] for event in events]
        self.assertIn("job_queued", kinds)
        self.assertIn("job_running", kinds)
        self.assertIn("task_route_selected", kinds)
        self.assertIn("job_completed", kinds)

        stream_response = self.client.get(f"/v1/jobs/{job_id}/events/stream")
        self.assertEqual(stream_response.status_code, 200)
        self.assertIn("event: run_event", stream_response.text)
        self.assertIn('"kind": "job_completed"', stream_response.text)


if __name__ == "__main__":
    unittest.main()
