from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

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


class _TemplateSupervisor:
    def __init__(self) -> None:
        self.model_loaded = False
        self.clone_count = 0

    def isolated_copy(self):  # noqa: ANN202
        self.clone_count += 1
        return _IsolatedSupervisor(self.clone_count)

    def run(self, request, progress_callback=None, event_callback=None):  # noqa: ANN001
        raise AssertionError("template supervisor should not execute runs directly")


class _IsolatedSupervisor:
    def __init__(self, clone_id: int) -> None:
        self._clone_id = clone_id
        self.model_loaded = False

    def run(self, request, progress_callback=None, event_callback=None):  # noqa: ANN001
        return RunResult(
            status="completed",
            model_id=f"child-{self._clone_id}",
            workspace="/tmp/demo",
            execution_mode=request.execution_mode,
            stop_reason="verifier_declared_complete",
            final_answer=f"child-{self._clone_id}",
            transcript="demo transcript",
            warnings=[],
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )


class APIStreamingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.tool_api_token = "test-token"
        self.tool_api_env = patch.dict(
            os.environ,
            {"TEAMAI_TOOL_API_TOKEN": self.tool_api_token},
            clear=False,
        )
        self.tool_api_env.start()
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
        self.tool_api_env.stop()
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

    def test_tool_catalogue_and_execute_endpoint_require_bearer_token(self) -> None:
        (self.workspace / "README.md").write_text("# demo\n", encoding="utf-8")

        unauthorized = self.client.get("/v1/tools")
        self.assertEqual(unauthorized.status_code, 401)

        headers = {"Authorization": f"Bearer {self.tool_api_token}"}
        catalogue = self.client.get("/v1/tools", headers=headers)
        self.assertEqual(catalogue.status_code, 200)
        tool_names = [tool["name"] for tool in catalogue.json()["tools"]]
        self.assertIn("list_files", tool_names)
        self.assertIn("spawn_agent", tool_names)

        execute = self.client.post(
            "/v1/tools/execute",
            headers=headers,
            json={
                "task_id": "task-demo",
                "execution_mode": "read_only",
                "tool_action": {
                    "tool": "list_files",
                    "reason": "Inspect the workspace root.",
                    "args": {"path": ".", "max_entries": 20},
                },
            },
        )
        self.assertEqual(execute.status_code, 200)
        result = execute.json()["result"]
        self.assertTrue(result["success"])
        self.assertIn("README.md", result["output"])

    def test_conversation_endpoints_store_and_summarize_turns(self) -> None:
        post_response = self.client.post(
            "/v1/conversation/task-demo/post",
            json={
                "agent_id": "planner",
                "role": "observation",
                "content": "Found the highest-signal runtime files.",
                "metadata": {"round": 1},
            },
        )
        self.assertEqual(post_response.status_code, 200)
        created = post_response.json()
        self.assertEqual(created["task_id"], "task-demo")
        self.assertEqual(created["agent_id"], "planner")

        turns_response = self.client.get("/v1/conversation/task-demo")
        self.assertEqual(turns_response.status_code, 200)
        turns = turns_response.json()
        self.assertEqual(len(turns), 1)
        self.assertEqual(turns[0]["content"], "Found the highest-signal runtime files.")

        summary_response = self.client.get("/v1/conversation/task-demo/summary")
        self.assertEqual(summary_response.status_code, 200)
        summary = summary_response.json()
        self.assertEqual(summary["task_id"], "task-demo")
        self.assertEqual(summary["turn_count"], 1)
        self.assertEqual(summary["agents"], ["planner"])

        list_response = self.client.get("/v1/conversations")
        self.assertEqual(list_response.status_code, 200)
        self.assertIn("task-demo", list_response.json())

    def test_dashboard_endpoints_render_control_surface_and_summary(self) -> None:
        dashboard_response = self.client.get("/dashboard")
        self.assertEqual(dashboard_response.status_code, 200)
        self.assertIn("teamAI Console", dashboard_response.text)
        self.assertIn("Pending Approvals", dashboard_response.text)
        self.assertIn("writes: read-only", dashboard_response.text)

        self.client.post(
            "/v1/conversation/task-dashboard/post",
            json={
                "agent_id": "planner",
                "role": "status_update",
                "content": "Dashboard inbox seeded.",
            },
        )
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

        with patch("teamai.api.load_schedules", return_value=[]):
            summary_response = self.client.get("/v1/dashboard/summary")

        self.assertEqual(summary_response.status_code, 200)
        payload = summary_response.json()
        self.assertEqual(payload["health"]["status"], "ok")
        self.assertGreaterEqual(payload["jobs"]["counts"]["completed"], 1)
        self.assertTrue(any(item["job_id"] == job_id for item in payload["jobs"]["recent"]))
        self.assertEqual(payload["conversations"]["count"], 1)
        self.assertEqual(payload["conversations"]["items"][0]["task_id"], "task-dashboard")
        self.assertIn("safety", payload)
        self.assertEqual(payload["safety"]["allow_writes"], False)
        self.assertEqual(payload["safety"]["posture"], "read_only")
        self.assertIn("approvals", payload)
        self.assertEqual(payload["approvals"]["count"], 0)
        self.assertEqual(payload["approvals"]["items"], [])

    def test_dashboard_summary_surfaces_pending_approvals(self) -> None:
        from teamai.approvals import PatchApprovalStore

        store = PatchApprovalStore()
        workspace = self.workspace
        target = workspace / "note.md"
        target.write_text("before\n", encoding="utf-8")
        store.create(
            workspace=workspace,
            path=target,
            before_text="before\n",
            after_text="after\n",
            before_exists=True,
            reason="Smoke-test approval visibility.",
            source_tool="patch_compiler",
        )

        with patch("teamai.api.load_schedules", return_value=[]):
            summary_response = self.client.get("/v1/dashboard/summary")

        self.assertEqual(summary_response.status_code, 200)
        payload = summary_response.json()
        self.assertEqual(payload["approvals"]["count"], 1)
        approval_item = payload["approvals"]["items"][0]
        self.assertEqual(approval_item["status"], "pending")
        self.assertEqual(approval_item["source_tool"], "patch_compiler")
        self.assertEqual(approval_item["change_count"], 1)
        self.assertEqual(approval_item["reason"], "Smoke-test approval visibility.")

    def test_run_endpoint_clones_supervisor_per_request_when_supported(self) -> None:
        self.client.close()
        template = _TemplateSupervisor()
        self.client = TestClient(create_app(self.settings, supervisor=template, jobs=self.jobs))

        first = self.client.post(
            "/v1/run",
            json={"task": "Inspect repo.", "workspace_path": ".", "execution_mode": "read_only"},
        )
        second = self.client.post(
            "/v1/run",
            json={"task": "Inspect repo again.", "workspace_path": ".", "execution_mode": "read_only"},
        )

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["final_answer"], "child-1")
        self.assertEqual(second.json()["final_answer"], "child-2")
        self.assertEqual(template.clone_count, 2)


if __name__ == "__main__":
    unittest.main()
