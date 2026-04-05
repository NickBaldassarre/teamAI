from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from teamai.approvals import PatchApprovalStore
from teamai.cli import main
from teamai.schemas import RunResult


class PatchApprovalStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        (self.workspace / "example.txt").write_text("hello\nworld\n", encoding="utf-8")
        self.store = PatchApprovalStore()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_reject_marks_pending_approval_as_rejected(self) -> None:
        approval = self.store.create(
            workspace=self.workspace,
            path=Path("example.txt"),
            before_text="hello\nworld\n",
            after_text="hello\nteamai\n",
            before_exists=True,
            reason="propose a replacement",
            source_tool="replace_in_file",
        )

        rejected = self.store.reject(
            workspace=self.workspace,
            approval_id=str(approval["approval_id"]),
            reason="superseded by a newer patch",
        )

        self.assertEqual(rejected["status"], "rejected")
        self.assertIn("rejected_at", rejected)
        self.assertEqual(rejected["rejection_reason"], "superseded by a newer patch")

    def test_prune_stale_removes_stale_artifacts(self) -> None:
        approval = self.store.create(
            workspace=self.workspace,
            path=Path("example.txt"),
            before_text="hello\nworld\n",
            after_text="hello\nteamai\n",
            before_exists=True,
            reason="propose a replacement",
            source_tool="replace_in_file",
        )

        (self.workspace / "example.txt").write_text("hello\nchanged\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stale"):
            self.store.apply(workspace=self.workspace, approval_id=str(approval["approval_id"]))

        pruned = self.store.prune_stale(workspace=self.workspace)

        self.assertEqual(len(pruned), 1)
        self.assertEqual(pruned[0]["status"], "stale")
        approval_path = self.workspace / ".teamai" / "approvals" / f"{approval['approval_id']}.json"
        self.assertFalse(approval_path.exists())

    def test_build_continuation_task_uses_original_goal_when_available(self) -> None:
        approval = self.store.create(
            workspace=self.workspace,
            path=Path("example.txt"),
            before_text="hello\nworld\n",
            after_text="hello\nteamai\n",
            before_exists=True,
            reason="propose a replacement",
            source_tool="replace_in_file",
            continuation={
                "original_task": "Update example.txt and finish the task.",
                "requested_execution_mode": "workspace_write",
            },
        )

        task = self.store.build_continuation_task(approval)

        self.assertIn("Update example.txt and finish the task.", task)
        self.assertIn(str(approval["approval_id"]), task)
        self.assertEqual(self.store.continuation_execution_mode(approval), "workspace_write")

    def test_build_continuation_context_includes_scoped_verification_details(self) -> None:
        (self.workspace / "teamai").mkdir()
        (self.workspace / "teamai" / "api.py").write_text("print('demo')\n", encoding="utf-8")
        (self.workspace / "tests").mkdir()
        (self.workspace / "tests" / "test_api.py").write_text(
            "import unittest\n\n\nclass DemoTest(unittest.TestCase):\n    def test_ok(self) -> None:\n        self.assertTrue(True)\n",
            encoding="utf-8",
        )
        approval = self.store.create(
            workspace=self.workspace,
            path=Path("teamai/api.py"),
            before_text="print('demo')\n",
            after_text="print('streaming')\n",
            before_exists=True,
            reason="update the API surface",
            source_tool="replace_in_file",
            continuation={
                "original_task": "Update teamai/api.py and continue.",
                "requested_execution_mode": "workspace_write",
            },
        )

        context = self.store.build_continuation_context(approval, workspace=self.workspace)
        task = self.store.build_continuation_task(approval, continuation_context=context)

        self.assertEqual(context["path"], "teamai/api.py")
        self.assertIn("tests/test_api.py", context["suggested_read_paths"])
        self.assertEqual(context["suggested_commands"][0], ["python", "-m", "unittest", "tests.test_api"])
        self.assertIn("Verification focus:", task)
        self.assertIn("python -m unittest tests.test_api", task)


class ApprovalsCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        (self.workspace / "example.txt").write_text("hello\nworld\n", encoding="utf-8")
        self.store = PatchApprovalStore()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _run_cli(self, argv: list[str]) -> tuple[int, dict]:
        stdout = StringIO()
        with (
            patch.object(sys, "argv", argv),
            patch.dict(os.environ, {"TEAMAI_WORKSPACE_ROOT": str(self.workspace)}, clear=False),
            redirect_stdout(stdout),
        ):
            exit_code = main()
        return exit_code, json.loads(stdout.getvalue())

    def test_cli_reject_command(self) -> None:
        approval = self.store.create(
            workspace=self.workspace,
            path=Path("example.txt"),
            before_text="hello\nworld\n",
            after_text="hello\nteamai\n",
            before_exists=True,
            reason="propose a replacement",
            source_tool="replace_in_file",
        )

        exit_code, payload = self._run_cli(
            [
                "teamai",
                "approvals",
                "reject",
                "--workspace",
                ".",
                str(approval["approval_id"]),
                "--reason",
                "bad patch",
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["status"], "rejected")
        self.assertEqual(payload["rejection_reason"], "bad patch")

    def test_cli_prune_stale_command(self) -> None:
        approval = self.store.create(
            workspace=self.workspace,
            path=Path("example.txt"),
            before_text="hello\nworld\n",
            after_text="hello\nteamai\n",
            before_exists=True,
            reason="propose a replacement",
            source_tool="replace_in_file",
        )
        (self.workspace / "example.txt").write_text("hello\nchanged\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "stale"):
            self.store.apply(workspace=self.workspace, approval_id=str(approval["approval_id"]))

        exit_code, payload = self._run_cli(
            [
                "teamai",
                "approvals",
                "prune-stale",
                "--workspace",
                ".",
            ]
        )

        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["pruned"][0]["status"], "stale")
        approval_path = self.workspace / ".teamai" / "approvals" / f"{approval['approval_id']}.json"
        self.assertFalse(approval_path.exists())

    def test_cli_apply_continue_resumes_original_task_with_write_mode(self) -> None:
        approval = self.store.create(
            workspace=self.workspace,
            path=Path("example.txt"),
            before_text="hello\nworld\n",
            after_text="hello\nteamai\n",
            before_exists=True,
            reason="propose a replacement",
            source_tool="replace_in_file",
            continuation={
                "original_task": "Update example.txt and finish the task.",
                "requested_execution_mode": "workspace_write",
            },
        )
        captured: dict[str, object] = {}
        workspace = self.workspace

        class DummySupervisor:
            def __init__(self, settings) -> None:  # noqa: ANN001
                captured["allow_writes"] = settings.allow_writes

            def run(self, request, progress_callback=None):  # noqa: ANN001
                captured["request"] = request
                return RunResult(
                    status="completed",
                    model_id="dummy",
                    workspace=str(workspace),
                    execution_mode=request.execution_mode,
                    stop_reason="verifier_declared_complete",
                    final_answer="Continuation finished cleanly.",
                    transcript="",
                    rounds=[],
                    warnings=[],
                    started_at=datetime.now(timezone.utc),
                    completed_at=datetime.now(timezone.utc),
                )

        stdout = StringIO()
        with (
            patch.object(sys, "argv", ["teamai", "approvals", "apply", "--workspace", ".", str(approval["approval_id"]), "--continue"]),
            patch.dict(
                os.environ,
                {"TEAMAI_WORKSPACE_ROOT": str(self.workspace), "TEAMAI_ALLOW_WRITES": "false"},
                clear=False,
            ),
            patch("teamai.supervisor.ClosedLoopSupervisor", DummySupervisor),
            redirect_stdout(stdout),
        ):
            exit_code = main()

        payload = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(payload["approval"]["status"], "applied")
        self.assertEqual(payload["continuation"]["status"], "completed")
        self.assertEqual(captured["allow_writes"], True)
        self.assertEqual(captured["request"].execution_mode, "workspace_write")
        self.assertIn("Do not recreate the same patch.", captured["request"].task)
        self.assertEqual(captured["request"].continuation_context["path"], "example.txt")
        self.assertIn("verification_focus", captured["request"].continuation_context)
        self.assertEqual((self.workspace / "example.txt").read_text(encoding="utf-8"), "hello\nteamai\n")


if __name__ == "__main__":
    unittest.main()
