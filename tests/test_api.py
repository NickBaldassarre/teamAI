from __future__ import annotations

import dataclasses
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
        self._runtime_settings_patch = patch(
            "teamai.runtime_state._runtime_settings_path",
            return_value=self.workspace / "runtime_settings.json",
        )
        self._runtime_settings_patch.start()
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
        self._runtime_settings_patch.stop()
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

    def _build_client_with_settings(self, *, allow_writes: bool) -> TestClient:
        self.client.close()
        settings = dataclasses.replace(self.settings, allow_writes=allow_writes)
        client = TestClient(create_app(settings, supervisor=_FakeSupervisor(), jobs=self.jobs))
        self.client = client
        return client

    def _seed_pending_approval(self) -> str:
        from teamai.approvals import PatchApprovalStore

        store = PatchApprovalStore()
        target = self.workspace / "note.md"
        target.write_text("before\n", encoding="utf-8")
        payload = store.create(
            workspace=self.workspace,
            path=target,
            before_text="before\n",
            after_text="after\n",
            before_exists=True,
            reason="Dashboard action test.",
            source_tool="patch_compiler",
        )
        return str(payload["approval_id"])

    def test_approval_get_endpoint_returns_summary_with_unified_diff(self) -> None:
        approval_id = self._seed_pending_approval()

        response = self.client.get(f"/v1/approvals/{approval_id}")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["approval_id"], approval_id)
        self.assertEqual(body["status"], "pending")
        self.assertEqual(body["source_tool"], "patch_compiler")
        diff = body.get("diff", "")
        self.assertIn("--- a/", diff)
        self.assertIn("+++ b/", diff)
        self.assertIn("-before", diff)
        self.assertIn("+after", diff)

    def test_approval_get_endpoint_returns_404_when_missing(self) -> None:
        response = self.client.get("/v1/approvals/does-not-exist")

        self.assertEqual(response.status_code, 404)

    def test_approval_apply_endpoint_writes_target_when_writes_enabled(self) -> None:
        self._build_client_with_settings(allow_writes=True)
        approval_id = self._seed_pending_approval()

        response = self.client.post(f"/v1/approvals/{approval_id}/apply")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "applied")
        self.assertEqual(body["approval"]["status"], "applied")
        self.assertEqual((self.workspace / "note.md").read_text(encoding="utf-8"), "after\n")

    def test_approval_apply_endpoint_returns_403_when_writes_disabled(self) -> None:
        approval_id = self._seed_pending_approval()

        response = self.client.post(f"/v1/approvals/{approval_id}/apply")

        self.assertEqual(response.status_code, 403)
        self.assertIn("TEAMAI_ALLOW_WRITES", response.json()["detail"])
        # Workspace file must remain untouched by a blocked apply.
        self.assertEqual((self.workspace / "note.md").read_text(encoding="utf-8"), "before\n")

    def test_approval_apply_endpoint_returns_404_when_missing(self) -> None:
        self._build_client_with_settings(allow_writes=True)

        response = self.client.post("/v1/approvals/does-not-exist/apply")

        self.assertEqual(response.status_code, 404)

    def test_approval_apply_endpoint_returns_409_when_stale(self) -> None:
        self._build_client_with_settings(allow_writes=True)
        approval_id = self._seed_pending_approval()
        # Mutate the target so apply() detects drift and marks it stale.
        (self.workspace / "note.md").write_text("drifted\n", encoding="utf-8")

        response = self.client.post(f"/v1/approvals/{approval_id}/apply")

        self.assertEqual(response.status_code, 409)
        self.assertIn("stale", response.json()["detail"])

    def test_approval_reject_endpoint_marks_rejected(self) -> None:
        approval_id = self._seed_pending_approval()

        response = self.client.post(
            f"/v1/approvals/{approval_id}/reject",
            json={"reason": "not needed"},
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "rejected")
        self.assertEqual(body["approval"]["status"], "rejected")
        # Target untouched by reject.
        self.assertEqual((self.workspace / "note.md").read_text(encoding="utf-8"), "before\n")

    def test_approval_reject_endpoint_returns_404_when_missing(self) -> None:
        response = self.client.post(
            "/v1/approvals/does-not-exist/reject",
            json={"reason": "never existed"},
        )

        self.assertEqual(response.status_code, 404)

    def test_approval_prune_stale_endpoint_removes_stale_records(self) -> None:
        from teamai.approvals import PatchApprovalStore

        store = PatchApprovalStore()
        approval_id = self._seed_pending_approval()

        # Flip the record to `stale` directly (mirroring what apply() does on drift).
        approval_path = self.workspace / ".teamai" / "approvals" / f"{approval_id}.json"
        payload = json.loads(approval_path.read_text(encoding="utf-8"))
        payload["status"] = "stale"
        payload["stale_reason"] = "Synthetic drift for test."
        approval_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        response = self.client.post("/v1/approvals/prune-stale")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "pruned")
        self.assertEqual(body["count"], 1)
        self.assertFalse(approval_path.exists())
        # Listing confirms the record is gone.
        self.assertEqual(store.list(workspace=self.workspace, include_all=True), [])

    def test_approval_prune_stale_endpoint_noop_when_nothing_stale(self) -> None:
        self._seed_pending_approval()

        response = self.client.post("/v1/approvals/prune-stale")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["count"], 0)
        self.assertEqual(body["approvals"], [])

    def _isolated_schedules_path(self) -> Path:
        return self.workspace / "schedules.json"

    def test_schedule_create_endpoint_returns_403_when_writes_disabled(self) -> None:
        with patch(
            "teamai.scheduler._schedules_path",
            return_value=self._isolated_schedules_path(),
        ):
            response = self.client.post(
                "/v1/schedules",
                json={"task": "Run evals", "cron": "0 3 * * *"},
            )

        self.assertEqual(response.status_code, 403)
        self.assertIn("TEAMAI_ALLOW_WRITES", response.json()["detail"])
        self.assertFalse(self._isolated_schedules_path().exists())

    def test_schedule_create_endpoint_persists_entry_when_writes_enabled(self) -> None:
        self._build_client_with_settings(allow_writes=True)
        with patch(
            "teamai.scheduler._schedules_path",
            return_value=self._isolated_schedules_path(),
        ):
            response = self.client.post(
                "/v1/schedules",
                json={
                    "task": "Run evals",
                    "cron": "0 3 * * *",
                    "execution_mode": "read_only",
                },
            )

            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual(body["task"], "Run evals")
            self.assertEqual(body["cron"], "0 3 * * *")
            self.assertTrue(body["id"].startswith("sched_"))
            self.assertTrue(self._isolated_schedules_path().exists())

    def test_schedule_create_endpoint_returns_400_on_invalid_cron(self) -> None:
        self._build_client_with_settings(allow_writes=True)
        with patch(
            "teamai.scheduler._schedules_path",
            return_value=self._isolated_schedules_path(),
        ):
            response = self.client.post(
                "/v1/schedules",
                json={"task": "Run evals", "cron": "not-a-cron"},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid schedule", response.json()["detail"])

    def test_schedule_create_endpoint_returns_400_when_task_missing(self) -> None:
        self._build_client_with_settings(allow_writes=True)
        with patch(
            "teamai.scheduler._schedules_path",
            return_value=self._isolated_schedules_path(),
        ):
            response = self.client.post(
                "/v1/schedules",
                json={"cron": "0 3 * * *"},
            )

        self.assertEqual(response.status_code, 400)

    def test_schedule_delete_endpoint_returns_403_when_writes_disabled(self) -> None:
        with patch(
            "teamai.scheduler._schedules_path",
            return_value=self._isolated_schedules_path(),
        ):
            response = self.client.delete("/v1/schedules/sched_anything")

        self.assertEqual(response.status_code, 403)
        self.assertIn("TEAMAI_ALLOW_WRITES", response.json()["detail"])

    def test_schedule_delete_endpoint_removes_entry_when_writes_enabled(self) -> None:
        self._build_client_with_settings(allow_writes=True)
        with patch(
            "teamai.scheduler._schedules_path",
            return_value=self._isolated_schedules_path(),
        ):
            created = self.client.post(
                "/v1/schedules",
                json={"task": "Run evals", "cron": "0 3 * * *"},
            ).json()
            schedule_id = created["id"]

            response = self.client.delete(f"/v1/schedules/{schedule_id}")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"status": "removed", "id": schedule_id})
            listed = self.client.get("/v1/schedules").json()
            self.assertEqual(listed, [])

    def test_schedule_delete_endpoint_returns_404_when_missing(self) -> None:
        self._build_client_with_settings(allow_writes=True)
        with patch(
            "teamai.scheduler._schedules_path",
            return_value=self._isolated_schedules_path(),
        ):
            response = self.client.delete("/v1/schedules/does-not-exist")

        self.assertEqual(response.status_code, 404)

    def test_schedule_toggle_endpoint_flips_enabled_state(self) -> None:
        self._build_client_with_settings(allow_writes=True)
        with patch(
            "teamai.scheduler._schedules_path",
            return_value=self._isolated_schedules_path(),
        ):
            created = self.client.post(
                "/v1/schedules",
                json={"task": "Run evals", "cron": "0 3 * * *", "enabled": True},
            ).json()
            schedule_id = created["id"]

            first = self.client.post(f"/v1/schedules/{schedule_id}/toggle")
            self.assertEqual(first.status_code, 200)
            body = first.json()
            self.assertEqual(body["status"], "toggled")
            self.assertFalse(body["schedule"]["enabled"])

            second = self.client.post(f"/v1/schedules/{schedule_id}/toggle")
            self.assertEqual(second.status_code, 200)
            self.assertTrue(second.json()["schedule"]["enabled"])

    def test_schedule_toggle_endpoint_returns_403_when_writes_disabled(self) -> None:
        with patch(
            "teamai.scheduler._schedules_path",
            return_value=self._isolated_schedules_path(),
        ):
            response = self.client.post("/v1/schedules/sched_anything/toggle")

        self.assertEqual(response.status_code, 403)
        self.assertIn("TEAMAI_ALLOW_WRITES", response.json()["detail"])

    def test_schedule_toggle_endpoint_returns_404_when_missing(self) -> None:
        self._build_client_with_settings(allow_writes=True)
        with patch(
            "teamai.scheduler._schedules_path",
            return_value=self._isolated_schedules_path(),
        ):
            response = self.client.post("/v1/schedules/does-not-exist/toggle")

        self.assertEqual(response.status_code, 404)

    def test_cancel_job_endpoint_marks_queued_record_cancelled(self) -> None:
        # Seed a queued record directly — no worker thread runs, so the
        # record stays queued until the cancel endpoint flips it.
        request = RunRequest(
            task="inspect",
            workspace_path=".",
            execution_mode="read_only",
        )
        record = self.jobs.create(request)

        response = self.client.post(f"/v1/jobs/{record.job_id}/cancel")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["job_id"], record.job_id)
        self.assertEqual(body["status"], "cancelled")
        self.assertEqual(self.jobs.get(record.job_id).status, "cancelled")

    def test_cancel_job_endpoint_returns_409_when_terminal(self) -> None:
        request = RunRequest(
            task="inspect",
            workspace_path=".",
            execution_mode="read_only",
        )
        record = self.jobs.create(request)
        result = RunResult(
            status="completed",
            model_id="dummy",
            workspace=str(self.workspace),
            execution_mode="read_only",
            stop_reason="verifier_declared_complete",
            final_answer="done",
            transcript="demo transcript",
            warnings=[],
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
        self.jobs.mark_running(record.job_id)
        self.jobs.mark_completed(record.job_id, result)

        response = self.client.post(f"/v1/jobs/{record.job_id}/cancel")

        self.assertEqual(response.status_code, 409)
        self.assertIn("terminal", response.json()["detail"])

    def test_cancel_job_endpoint_returns_404_when_missing(self) -> None:
        response = self.client.post("/v1/jobs/does-not-exist/cancel")

        self.assertEqual(response.status_code, 404)

    def test_transcript_endpoint_returns_text_when_completed(self) -> None:
        create_response = self.client.post(
            "/v1/jobs",
            json={"task": "inspect", "workspace_path": ".", "execution_mode": "read_only"},
        )
        self.assertEqual(create_response.status_code, 200)
        job_id = create_response.json()["job_id"]
        for _ in range(50):
            if self.client.get(f"/v1/jobs/{job_id}").json()["status"] == "completed":
                break
            time.sleep(0.01)
        else:  # pragma: no cover - defensive timeout
            self.fail("job did not complete in time")

        response = self.client.get(f"/v1/jobs/{job_id}/transcript")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["job_id"], job_id)
        self.assertEqual(body["status"], "completed")
        self.assertEqual(body["transcript"], "demo transcript")

    def test_transcript_endpoint_returns_409_when_not_completed(self) -> None:
        request = RunRequest(
            task="inspect",
            workspace_path=".",
            execution_mode="read_only",
        )
        record = self.jobs.create(request)

        response = self.client.get(f"/v1/jobs/{record.job_id}/transcript")

        self.assertEqual(response.status_code, 409)
        self.assertIn("queued", response.json()["detail"])

    def test_transcript_endpoint_returns_404_when_missing(self) -> None:
        response = self.client.get("/v1/jobs/does-not-exist/transcript")

        self.assertEqual(response.status_code, 404)

    def test_settings_allow_writes_toggle_cycles_write_gates(self) -> None:
        approval_id = self._seed_pending_approval()

        # Initially blocked: setUp builds the app with allow_writes=False.
        pre = self.client.post(f"/v1/approvals/{approval_id}/apply")
        self.assertEqual(pre.status_code, 403)

        toggle_on = self.client.post(
            "/v1/settings/allow_writes",
            json={"allow_writes": True},
        )
        self.assertEqual(toggle_on.status_code, 200)
        body_on = toggle_on.json()
        self.assertTrue(body_on["allow_writes"])
        self.assertEqual(body_on["posture"], "writes_enabled")

        # Same approval now applies because the endpoint gate reads the
        # mutated settings via closure at request time.
        apply_response = self.client.post(f"/v1/approvals/{approval_id}/apply")
        self.assertEqual(apply_response.status_code, 200)
        self.assertEqual(
            (self.workspace / "note.md").read_text(encoding="utf-8"),
            "after\n",
        )

        # Dashboard summary should reflect the toggled-on posture.
        with patch("teamai.api.load_schedules", return_value=[]):
            summary_on = self.client.get("/v1/dashboard/summary?limit=1")
        self.assertEqual(summary_on.status_code, 200)
        self.assertTrue(summary_on.json()["safety"]["allow_writes"])

        toggle_off = self.client.post(
            "/v1/settings/allow_writes",
            json={"allow_writes": False},
        )
        self.assertEqual(toggle_off.status_code, 200)
        body_off = toggle_off.json()
        self.assertFalse(body_off["allow_writes"])
        self.assertEqual(body_off["posture"], "read_only")

        # Schedule creation should 403 again now that writes are off.
        with patch(
            "teamai.scheduler._schedules_path",
            return_value=self._isolated_schedules_path(),
        ):
            sched_response = self.client.post(
                "/v1/schedules",
                json={"task": "Run evals", "cron": "0 3 * * *"},
            )
        self.assertEqual(sched_response.status_code, 403)

    def test_settings_allow_writes_rejects_invalid_body(self) -> None:
        missing = self.client.post("/v1/settings/allow_writes", json={})
        self.assertEqual(missing.status_code, 422)
        self.assertIn("allow_writes", missing.json()["detail"])

        wrong_type = self.client.post(
            "/v1/settings/allow_writes",
            json={"allow_writes": "yes"},
        )
        self.assertEqual(wrong_type.status_code, 422)

        # State must be unchanged after rejections.
        with patch("teamai.api.load_schedules", return_value=[]):
            summary = self.client.get("/v1/dashboard/summary?limit=1")
        self.assertEqual(summary.status_code, 200)
        self.assertFalse(summary.json()["safety"]["allow_writes"])

    def test_runtime_settings_round_trip_overlays_env_value(self) -> None:
        from teamai.runtime_state import (
            load_runtime_settings,
            save_runtime_settings,
        )

        save_runtime_settings({"allow_writes": True})
        self.assertEqual(load_runtime_settings(), {"allow_writes": True})

        # Settings.from_env honours the persisted value even when the env
        # var defaults to False.
        with patch.dict(os.environ, {"TEAMAI_ALLOW_WRITES": "0"}, clear=False):
            reloaded = Settings.from_env()
        self.assertTrue(reloaded.allow_writes)

        save_runtime_settings({"allow_writes": False})
        with patch.dict(os.environ, {"TEAMAI_ALLOW_WRITES": "1"}, clear=False):
            reloaded_off = Settings.from_env()
        self.assertFalse(reloaded_off.allow_writes)

    def test_settings_allow_writes_endpoint_persists_through_restart(self) -> None:
        # Toggle ON via the endpoint.
        toggle = self.client.post(
            "/v1/settings/allow_writes",
            json={"allow_writes": True},
        )
        self.assertEqual(toggle.status_code, 200)
        self.assertTrue((self.workspace / "runtime_settings.json").exists())

        # A fresh Settings.from_env (analogous to a daemon restart) sees the
        # persisted toggle even though the env var stays at its default.
        with patch.dict(os.environ, {"TEAMAI_ALLOW_WRITES": "0"}, clear=False):
            restarted = Settings.from_env()
        self.assertTrue(restarted.allow_writes)

    def test_team_endpoint_rejects_missing_goal(self) -> None:
        # The teams-launch dashboard form posts {goal, workspace_path?} —
        # the endpoint must reject empty bodies before spawning the
        # background AgentTeam thread, otherwise an empty goal would
        # leak into decomposition.
        missing = self.client.post("/v1/team", json={})
        self.assertEqual(missing.status_code, 400)
        self.assertIn("goal", missing.json()["detail"])

        empty_goal = self.client.post("/v1/team", json={"goal": ""})
        self.assertEqual(empty_goal.status_code, 400)

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
