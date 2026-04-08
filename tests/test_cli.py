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

from teamai.cli import _build_run_stream_handlers, main
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

    def test_execute_handoff_command_reports_verified_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            failure_log = project_root / ".teamai" / "failure_context.log"
            failure_log.parent.mkdir(parents=True, exist_ok=True)
            failure_log.write_text("stale failure log\n", encoding="utf-8")

            stdout = io.StringIO()
            with patch(
                "teamai.integrations.codex_bridge.execute_verified_codex_handoff",
                return_value=SimpleNamespace(
                    execution=SimpleNamespace(
                        model="gpt-5.4",
                        payload_file=project_root / ".teamai" / "codex_payload.json",
                        patch_file=project_root / ".teamai" / "codex_solution.patch",
                        patch_text=(
                            "diff --git a/demo.txt b/demo.txt\n"
                            "--- a/demo.txt\n"
                            "+++ b/demo.txt\n"
                            "@@ -0,0 +1 @@\n"
                            "+patched\n"
                        ),
                    ),
                    verification=SimpleNamespace(success=True, patch_returncode=0, test_returncode=0),
                    failure_context_file=failure_log,
                ),
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
            self.assertIn("Ready for human review", rendered)
            self.assertFalse(failure_log.exists())

    def test_execute_handoff_command_reports_failed_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            failure_log = project_root / ".teamai" / "failure_context.log"
            stdout = io.StringIO()
            with patch(
                "teamai.integrations.codex_bridge.execute_verified_codex_handoff",
                return_value=SimpleNamespace(
                    execution=SimpleNamespace(
                        model="gpt-5.4",
                        payload_file=project_root / ".teamai" / "codex_payload.json",
                        patch_file=project_root / ".teamai" / "codex_solution.patch",
                        patch_text=(
                            "diff --git a/demo.txt b/demo.txt\n"
                            "--- a/demo.txt\n"
                            "+++ b/demo.txt\n"
                            "@@ -0,0 +1 @@\n"
                            "+patched\n"
                        ),
                    ),
                    verification=SimpleNamespace(
                        success=False,
                        patch_returncode=0,
                        test_returncode=1,
                        log_output="tests failed\n",
                    ),
                    failure_context_file=failure_log,
                ),
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
