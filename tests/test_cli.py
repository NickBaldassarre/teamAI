from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from teamai.cli import _build_run_stream_handlers, _run_autopilot_workflow, main
from teamai.config import Settings
from teamai.schemas import CodexHandoffPayload, RunEvent, RunResult


class CLIStreamingTest(unittest.TestCase):
    def test_stream_handlers_write_jsonl_to_stderr_and_log_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            stderr = io.StringIO()
            progress_callback, event_callback, close_stream = _build_run_stream_handlers(
                project_root=project_root,
                stream_format="jsonl",
                event_log_file="events.jsonl",
            )
            event = RunEvent(
                sequence=1,
                timestamp=datetime.now(timezone.utc),
                kind="round_stage",
                message="Round 1/2: planner",
                round_number=1,
                stage="planner",
                data={"total_rounds": 2},
            )
            try:
                with redirect_stderr(stderr):
                    assert progress_callback is not None
                    progress_callback("Round 1/2: planner")
                    assert event_callback is not None
                    event_callback(event)
            finally:
                close_stream()

            stderr_payload = stderr.getvalue().strip().splitlines()
            self.assertEqual(len(stderr_payload), 1)
            self.assertEqual(json.loads(stderr_payload[0])["kind"], "round_stage")

            log_lines = (project_root / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(log_lines), 1)
            self.assertEqual(json.loads(log_lines[0])["stage"], "planner")

    def test_run_command_writes_codex_payload_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            settings = Settings(
                model_id="dummy",
                model_revision=None,
                force_download=False,
                trust_remote_code=False,
                enable_thinking=False,
                workspace_root=workspace,
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
            result = RunResult(
                status="completed",
                model_id="dummy",
                workspace=str(workspace),
                execution_mode="read_only",
                task_route="codex_handoff",
                stop_reason="codex_handoff_synthesized",
                final_answer="Current state: Ready.\n\nNext engineering tasks:\n- Inspect teamai/cli.py.\n",
                transcript="demo transcript",
                warnings=[],
                codex_payload=CodexHandoffPayload(
                    original_task="Inspect repo.",
                    core_dependencies=["teamai/cli.py", "teamai/api.py"],
                    distilled_context={
                        "teamai/cli.py": "CLI entrypoint summary.",
                        "teamai/api.py": "API entrypoint summary.",
                    },
                    recommended_codex_action="Inspect teamai/cli.py and teamai/api.py before implementing the change.",
                ),
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )

            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch("teamai.config.Settings.from_env", return_value=settings), patch(
                "teamai.supervisor.ClosedLoopSupervisor.run",
                return_value=result,
            ), patch("sys.argv", ["teamai", "run", "Inspect repo.", "--workspace", "."]), redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main()

            self.assertEqual(exit_code, 0)
            payload_path = workspace / ".teamai" / "codex_payload.json"
            self.assertTrue(payload_path.exists())
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["original_task"], "Inspect repo.")
            self.assertEqual(payload["core_dependencies"], ["teamai/cli.py", "teamai/api.py"])
            self.assertIn("semantic skeleton", stderr.getvalue().lower())

    def test_run_command_uses_autopilot_output_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            settings = Settings(
                model_id="dummy",
                model_revision=None,
                force_download=False,
                trust_remote_code=False,
                enable_thinking=False,
                workspace_root=workspace,
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
            result = RunResult(
                status="completed",
                model_id="dummy",
                workspace=str(workspace),
                execution_mode="read_only",
                task_route="codex_handoff",
                stop_reason="codex_handoff_synthesized",
                final_answer="Current state: Ready.\n\nNext engineering tasks:\n- Inspect teamai/cli.py.\n",
                transcript="demo transcript",
                warnings=[],
                codex_payload=CodexHandoffPayload(
                    original_task="Implement broad change.",
                    core_dependencies=["teamai/cli.py"],
                    distilled_context={"teamai/cli.py": "CLI entrypoint summary."},
                    recommended_codex_action="Inspect teamai/cli.py before implementing the change.",
                ),
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )

            stdout = io.StringIO()
            with patch("teamai.config.Settings.from_env", return_value=settings), patch(
                "teamai.supervisor.ClosedLoopSupervisor.run",
                return_value=result,
            ), patch(
                "teamai.cli._run_autopilot_workflow",
                return_value={
                    "requested_cycles": 2,
                    "completed_cycles": 1,
                    "status": "completed",
                    "stop_reason": "task_completed",
                    "cycles": [],
                    "final_result": result.model_dump(mode="json"),
                    "success": True,
                    "summary": "Autopilot summary",
                },
            ), patch(
                "sys.argv",
                [
                    "teamai",
                    "run",
                    "Implement broad change.",
                    "--workspace",
                    ".",
                    "--autopilot-cycles",
                    "2",
                ],
            ), redirect_stdout(stdout):
                exit_code = main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertIn("run", payload)
            self.assertIn("autopilot", payload)
            self.assertEqual(payload["autopilot"]["status"], "completed")

    def test_dashboard_command_starts_service_without_opening_browser_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            settings = Settings(
                model_id="dummy",
                model_revision=None,
                force_download=False,
                trust_remote_code=False,
                enable_thinking=False,
                workspace_root=workspace,
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
                port=8123,
            )

            stderr = io.StringIO()
            with patch("teamai.config.Settings.from_env", return_value=settings), patch(
                "teamai.api.create_app",
                return_value="app",
            ) as create_app, patch("uvicorn.run") as uvicorn_run, patch(
                "teamai.cli._schedule_dashboard_open"
            ) as schedule_open, patch(
                "sys.argv",
                ["teamai", "dashboard", "--no-open-browser"],
            ), redirect_stderr(stderr):
                exit_code = main()

            self.assertEqual(exit_code, 0)
            create_app.assert_called_once_with(settings)
            uvicorn_run.assert_called_once_with("app", host="127.0.0.1", port=8123, reload=False)
            schedule_open.assert_not_called()
            self.assertIn("http://127.0.0.1:8123/dashboard", stderr.getvalue())

    def test_run_autopilot_workflow_completes_after_verified_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            initial_result = RunResult(
                status="completed",
                model_id="dummy",
                workspace=str(workspace),
                execution_mode="read_only",
                task_route="codex_handoff",
                stop_reason="codex_handoff_synthesized",
                final_answer="Current state: Ready.\n\nNext engineering tasks:\n- Implement broad change.\n",
                transcript="demo transcript",
                warnings=[],
                codex_payload=CodexHandoffPayload(
                    original_task="Implement broad change.",
                    core_dependencies=["teamai/cli.py"],
                    distilled_context={"teamai/cli.py": "CLI entrypoint summary."},
                    recommended_codex_action="Inspect teamai/cli.py before implementing the change.",
                ),
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )
            final_result = RunResult(
                status="completed",
                model_id="dummy",
                workspace=str(workspace),
                execution_mode="workspace_write",
                task_route="multi_agent_loop",
                stop_reason="verifier_declared_complete",
                final_answer="Implemented broad change.",
                transcript="done",
                warnings=[],
                codex_payload=None,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )

            with patch(
                "teamai.cli._execute_verified_handoff_workflow",
                return_value={
                    "engine": "codex",
                    "model": "gpt-5.4",
                    "payload_path": str(workspace / ".teamai" / "codex_payload.json"),
                    "patch_path": str(workspace / ".teamai" / "codex_solution.patch"),
                    "verification": {"success": True},
                    "approval": {"approval_id": "approval123"},
                    "approval_error": None,
                    "failure_context_path": None,
                    "summary": "Handoff execution summary",
                    "success": True,
                },
            ), patch(
                "teamai.cli._apply_approval_and_continue",
                return_value={
                    "approval": {"approval_id": "approval123", "status": "applied"},
                    "approval_summary": {"approval_id": "approval123", "status": "applied"},
                    "continuation_task": "Continue the task.",
                    "continuation_context": {"verification_focus": "Verify the patch."},
                    "continuation_result": final_result,
                },
            ):
                payload = _run_autopilot_workflow(
                    project_root=workspace,
                    settings=SimpleNamespace(allow_writes=False),
                    initial_result=initial_result,
                    initial_payload_path=workspace / ".teamai" / "codex_payload.json",
                    workspace_path=str(workspace),
                    max_rounds=2,
                    max_actions=2,
                    max_tokens=64,
                    temperature=0.3,
                    handoff_engine="codex",
                    handoff_model=None,
                    handoff_patch_file=".teamai/codex_solution.patch",
                    max_cycles=2,
                )

            self.assertTrue(payload["success"])
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["stop_reason"], "task_completed")
            self.assertEqual(payload["completed_cycles"], 1)
            self.assertEqual(payload["cycles"][0]["approval"]["approval_id"], "approval123")

    def test_run_autopilot_workflow_accepts_local_handoff_engine(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            initial_result = RunResult(
                status="completed",
                model_id="dummy",
                workspace=str(workspace),
                execution_mode="read_only",
                task_route="codex_handoff",
                stop_reason="codex_handoff_synthesized",
                final_answer="Current state: Ready.\n\nNext engineering tasks:\n- Implement broad change.\n",
                transcript="demo transcript",
                warnings=[],
                codex_payload=CodexHandoffPayload(
                    original_task="Implement broad change.",
                    core_dependencies=["teamai/cli.py"],
                    distilled_context={"teamai/cli.py": "CLI entrypoint summary."},
                    recommended_codex_action="Inspect teamai/cli.py before implementing the change.",
                ),
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )
            final_result = RunResult(
                status="completed",
                model_id="dummy",
                workspace=str(workspace),
                execution_mode="workspace_write",
                task_route="multi_agent_loop",
                stop_reason="verifier_declared_complete",
                final_answer="Implemented broad change.",
                transcript="done",
                warnings=[],
                codex_payload=None,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )

            with patch(
                "teamai.cli._execute_verified_handoff_workflow",
                return_value={
                    "engine": "local",
                    "model": "mlx-community/gemma-4-2b-it-4bit",
                    "payload_path": str(workspace / ".teamai" / "codex_payload.json"),
                    "patch_path": str(workspace / ".teamai" / "local_solution.patch"),
                    "verification": {"success": True},
                    "approval": {"approval_id": "approval123"},
                    "approval_error": None,
                    "failure_context_path": None,
                    "summary": "Local handoff execution summary",
                    "success": True,
                },
            ), patch(
                "teamai.cli._apply_approval_and_continue",
                return_value={
                    "approval": {"approval_id": "approval123", "status": "applied"},
                    "approval_summary": {"approval_id": "approval123", "status": "applied"},
                    "continuation_task": "Continue the task.",
                    "continuation_context": {"verification_focus": "Verify the patch."},
                    "continuation_result": final_result,
                },
            ):
                payload = _run_autopilot_workflow(
                    project_root=workspace,
                    settings=SimpleNamespace(allow_writes=False),
                    initial_result=initial_result,
                    initial_payload_path=workspace / ".teamai" / "codex_payload.json",
                    workspace_path=str(workspace),
                    max_rounds=2,
                    max_actions=2,
                    max_tokens=64,
                    temperature=0.3,
                    handoff_engine="local",
                    handoff_model="mlx-community/gemma-4-2b-it-4bit",
                    handoff_patch_file=".teamai/local_solution.patch",
                    max_cycles=1,
                )

            self.assertTrue(payload["success"])
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["cycles"][0]["handoff"]["engine"], "local")

    def test_run_autopilot_workflow_stops_when_no_handoff_is_needed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            result = RunResult(
                status="completed",
                model_id="dummy",
                workspace=str(workspace),
                execution_mode="read_only",
                task_route="repository_inspection",
                stop_reason="inspection_synthesized",
                final_answer="Repository inspected.",
                transcript="done",
                warnings=[],
                codex_payload=None,
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )

            payload = _run_autopilot_workflow(
                project_root=workspace,
                settings=SimpleNamespace(allow_writes=False),
                initial_result=result,
                initial_payload_path=None,
                workspace_path=str(workspace),
                max_rounds=2,
                max_actions=2,
                max_tokens=64,
                temperature=0.3,
                handoff_engine="codex",
                handoff_model=None,
                handoff_patch_file=".teamai/codex_solution.patch",
                max_cycles=2,
            )

            self.assertTrue(payload["success"])
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["stop_reason"], "completed_without_handoff")
            self.assertEqual(payload["completed_cycles"], 0)

    def test_run_autopilot_workflow_fails_when_verified_handoff_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            initial_result = RunResult(
                status="completed",
                model_id="dummy",
                workspace=str(workspace),
                execution_mode="read_only",
                task_route="codex_handoff",
                stop_reason="codex_handoff_synthesized",
                final_answer="Current state: Ready.\n\nNext engineering tasks:\n- Implement broad change.\n",
                transcript="demo transcript",
                warnings=[],
                codex_payload=CodexHandoffPayload(
                    original_task="Implement broad change.",
                    core_dependencies=["teamai/cli.py"],
                    distilled_context={"teamai/cli.py": "CLI entrypoint summary."},
                    recommended_codex_action="Inspect teamai/cli.py before implementing the change.",
                ),
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )

            with patch(
                "teamai.cli._execute_verified_handoff_workflow",
                return_value={
                    "engine": "codex",
                    "model": "gpt-5.4",
                    "payload_path": str(workspace / ".teamai" / "codex_payload.json"),
                    "patch_path": str(workspace / ".teamai" / "codex_solution.patch"),
                    "verification": {"success": False},
                    "approval": None,
                    "approval_error": None,
                    "failure_context_path": str(workspace / ".teamai" / "failure_context.log"),
                    "summary": "Handoff execution summary",
                    "success": False,
                },
            ):
                payload = _run_autopilot_workflow(
                    project_root=workspace,
                    settings=SimpleNamespace(allow_writes=False),
                    initial_result=initial_result,
                    initial_payload_path=workspace / ".teamai" / "codex_payload.json",
                    workspace_path=str(workspace),
                    max_rounds=2,
                    max_actions=2,
                    max_tokens=64,
                    temperature=0.3,
                    handoff_engine="codex",
                    handoff_model=None,
                    handoff_patch_file=".teamai/codex_solution.patch",
                    max_cycles=2,
                )

            self.assertFalse(payload["success"])
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["stop_reason"], "verified_handoff_failed")
            self.assertEqual(payload["completed_cycles"], 1)

    def test_run_command_auto_executes_verified_handoff_for_broad_task(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            settings = Settings(
                model_id="dummy",
                model_revision=None,
                force_download=False,
                trust_remote_code=False,
                enable_thinking=False,
                workspace_root=workspace,
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
            result = RunResult(
                status="completed",
                model_id="dummy",
                workspace=str(workspace),
                execution_mode="read_only",
                task_route="codex_handoff",
                stop_reason="codex_handoff_synthesized",
                final_answer="Current state: Ready.\n\nNext engineering tasks:\n- Inspect teamai/cli.py.\n",
                transcript="demo transcript",
                warnings=[],
                codex_payload=CodexHandoffPayload(
                    original_task="Implement a broad change.",
                    core_dependencies=["teamai/cli.py", "teamai/api.py"],
                    distilled_context={
                        "teamai/cli.py": "CLI entrypoint summary.",
                        "teamai/api.py": "API entrypoint summary.",
                    },
                    recommended_codex_action="Inspect teamai/cli.py and teamai/api.py before implementing the change.",
                ),
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )

            stdout = io.StringIO()
            stderr = io.StringIO()
            with patch("teamai.config.Settings.from_env", return_value=settings), patch(
                "teamai.supervisor.ClosedLoopSupervisor.run",
                return_value=result,
            ), patch(
                "teamai.cli._execute_verified_handoff_workflow",
                return_value={
                    "engine": "codex",
                    "model": "gpt-5.4",
                    "payload_path": str(workspace / ".teamai" / "codex_payload.json"),
                    "patch_path": str(workspace / ".teamai" / "codex_solution.patch"),
                    "verification": {"success": True},
                    "approval": {"approval_id": "approval123"},
                    "approval_error": None,
                    "failure_context_path": None,
                    "summary": "Handoff execution summary\n- Approval: approval123",
                    "success": True,
                },
            ), patch(
                "sys.argv",
                [
                    "teamai",
                    "run",
                    "Implement a broad change.",
                    "--workspace",
                    ".",
                    "--auto-execute-handoff",
                ],
            ), redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(stdout.getvalue())
            self.assertIn("run", payload)
            self.assertIn("auto_handoff", payload)
            self.assertEqual(payload["auto_handoff"]["approval"]["approval_id"], "approval123")
            self.assertTrue(payload["auto_handoff"]["success"])
            self.assertIn("auto handoff", stderr.getvalue().lower())

    def test_run_command_auto_handoff_failure_sets_nonzero_exit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            settings = Settings(
                model_id="dummy",
                model_revision=None,
                force_download=False,
                trust_remote_code=False,
                enable_thinking=False,
                workspace_root=workspace,
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
            result = RunResult(
                status="completed",
                model_id="dummy",
                workspace=str(workspace),
                execution_mode="read_only",
                task_route="codex_handoff",
                stop_reason="codex_handoff_synthesized",
                final_answer="Current state: Ready.\n\nNext engineering tasks:\n- Inspect teamai/cli.py.\n",
                transcript="demo transcript",
                warnings=[],
                codex_payload=CodexHandoffPayload(
                    original_task="Implement a broad change.",
                    core_dependencies=["teamai/cli.py"],
                    distilled_context={"teamai/cli.py": "CLI entrypoint summary."},
                    recommended_codex_action="Inspect teamai/cli.py before implementing the change.",
                ),
                started_at=datetime.now(timezone.utc),
                completed_at=datetime.now(timezone.utc),
            )

            stdout = io.StringIO()
            with patch("teamai.config.Settings.from_env", return_value=settings), patch(
                "teamai.supervisor.ClosedLoopSupervisor.run",
                return_value=result,
            ), patch(
                "teamai.cli._execute_verified_handoff_workflow",
                return_value={
                    "engine": "codex",
                    "model": "gpt-5.4",
                    "payload_path": str(workspace / ".teamai" / "codex_payload.json"),
                    "patch_path": str(workspace / ".teamai" / "codex_solution.patch"),
                    "verification": {"success": False},
                    "approval": None,
                    "approval_error": None,
                    "failure_context_path": str(workspace / ".teamai" / "failure_context.log"),
                    "summary": "Handoff execution summary\n- Sandbox verification: failed",
                    "success": False,
                },
            ), patch(
                "sys.argv",
                [
                    "teamai",
                    "run",
                    "Implement a broad change.",
                    "--workspace",
                    ".",
                    "--auto-execute-handoff",
                ],
            ), redirect_stdout(stdout):
                exit_code = main()

            self.assertEqual(exit_code, 1)
            payload = json.loads(stdout.getvalue())
            self.assertFalse(payload["auto_handoff"]["success"])

    def test_execute_handoff_command_reports_verified_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            failure_log = project_root / ".teamai" / "failure_context.log"
            failure_log.parent.mkdir(parents=True, exist_ok=True)
            failure_log.write_text("stale failure log\n", encoding="utf-8")

            stdout = io.StringIO()
            with patch(
                "teamai.cli._execute_verified_handoff_workflow",
                return_value={
                    "engine": "codex",
                    "model": "gpt-5.4",
                    "payload_path": str(project_root / ".teamai" / "codex_payload.json"),
                    "patch_path": str(project_root / ".teamai" / "codex_solution.patch"),
                    "verification": {
                        "success": True,
                        "patch_returncode": 0,
                        "test_returncode": 0,
                        "commands_run": [],
                    },
                    "approval": {
                        "approval_id": "approval123",
                        "change_count": 1,
                    },
                    "approval_error": None,
                    "failure_context_path": None,
                    "summary": "\n".join(
                        [
                            "Handoff execution summary",
                            "- Engine: codex",
                            "- Model: gpt-5.4",
                            f"- Payload: {project_root / '.teamai' / 'codex_payload.json'}",
                            f"- Patch: {project_root / '.teamai' / 'codex_solution.patch'}",
                            "- Patch files: 1",
                            "- Patch lines: 5",
                            "- Sandbox verification: passed",
                            "- Verification detail: patch applied and sandbox tests passed",
                            "- Test exit code: 0",
                            "- Approval: approval123",
                            "- Approval scope: 1 file(s)",
                            "teamai approvals show approval123",
                            "teamai approvals apply approval123",
                        ]
                    ),
                    "success": True,
                },
            ), patch("sys.argv", ["teamai", "execute-handoff"]), patch("pathlib.Path.cwd", return_value=project_root), redirect_stdout(stdout):
                exit_code = main()

            self.assertEqual(exit_code, 0)
            rendered = stdout.getvalue().strip()
            self.assertIn("Handoff execution summary", rendered)
            self.assertIn("- Engine: codex", rendered)
            self.assertIn("- Model: gpt-5.4", rendered)
            self.assertIn(f"- Payload: {project_root / '.teamai' / 'codex_payload.json'}", rendered)
            self.assertIn(f"- Patch: {project_root / '.teamai' / 'codex_solution.patch'}", rendered)
            self.assertIn("- Patch files: 1", rendered)
            self.assertIn("- Patch lines: 5", rendered)
            self.assertIn("- Sandbox verification: passed", rendered)
            self.assertIn("- Verification detail: patch applied and sandbox tests passed", rendered)
            self.assertIn("- Test exit code: 0", rendered)
            self.assertIn("- Approval: approval123", rendered)
            self.assertIn("- Approval scope: 1 file(s)", rendered)
            self.assertIn("teamai approvals show approval123", rendered)
            self.assertIn("teamai approvals apply approval123", rendered)

    def test_execute_handoff_command_reports_failed_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            failure_log = project_root / ".teamai" / "failure_context.log"
            failure_log.parent.mkdir(parents=True, exist_ok=True)
            failure_log.write_text("tests failed\n", encoding="utf-8")
            stdout = io.StringIO()
            with patch(
                "teamai.cli._execute_verified_handoff_workflow",
                return_value={
                    "engine": "codex",
                    "model": "gpt-5.4",
                    "payload_path": str(project_root / ".teamai" / "codex_payload.json"),
                    "patch_path": str(project_root / ".teamai" / "codex_solution.patch"),
                    "verification": {
                        "success": False,
                        "patch_returncode": 0,
                        "test_returncode": 1,
                        "commands_run": [],
                    },
                    "approval": None,
                    "approval_error": None,
                    "failure_context_path": str(failure_log),
                    "summary": "\n".join(
                        [
                            "Handoff execution summary",
                            "- Patch files: 1",
                            "- Sandbox verification: failed",
                            "- Verification detail: patch applied, but sandbox tests failed",
                            "- Test exit code: 1",
                            f"- Failure log: {failure_log}",
                            "sandbox test failures before retrying",
                        ]
                    ),
                    "success": False,
                },
            ), patch("sys.argv", ["teamai", "execute-handoff"]), patch("pathlib.Path.cwd", return_value=project_root), redirect_stdout(stdout):
                exit_code = main()

            self.assertEqual(exit_code, 1)
            rendered = stdout.getvalue().strip()
            self.assertIn("Handoff execution summary", rendered)
            self.assertIn("- Patch files: 1", rendered)
            self.assertIn("- Sandbox verification: failed", rendered)
            self.assertIn("- Verification detail: patch applied, but sandbox tests failed", rendered)
            self.assertIn("- Test exit code: 1", rendered)
            self.assertIn(f"- Failure log: {failure_log}", rendered)
            self.assertIn("sandbox test failures before retrying", rendered)
            self.assertTrue(failure_log.exists())
            self.assertEqual(failure_log.read_text(encoding="utf-8"), "tests failed\n")


if __name__ == "__main__":
    unittest.main()
