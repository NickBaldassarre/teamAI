from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from teamai.approvals import PatchApprovalStore
from teamai.config import Settings
from teamai.schemas import ToolAction
from teamai.tools import WorkspaceTools


class WorkspaceToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        (self.workspace / "example.txt").write_text("hello\nworld\n", encoding="utf-8")
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
            temperature=0.1,
            allow_shell=False,
            allow_writes=False,
            command_timeout_seconds=5,
            max_file_bytes=10_000,
            max_command_output_chars=10_000,
            host="127.0.0.1",
            port=8000,
        )
        self.tools = WorkspaceTools(self.settings)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_read_file(self) -> None:
        result = self.tools.execute_actions(
            [ToolAction(tool="read_file", args={"path": "example.txt"}, reason="inspect")],
            workspace=self.workspace,
            execution_mode="read_only",
        )[0]
        self.assertTrue(result.success)
        self.assertIn("hello", result.output)

    def test_write_is_blocked_in_read_only_mode(self) -> None:
        result = self.tools.execute_actions(
            [
                ToolAction(
                    tool="write_file",
                    args={"path": "new.txt", "content": "x"},
                    reason="write",
                )
            ],
            workspace=self.workspace,
            execution_mode="read_only",
        )[0]
        self.assertFalse(result.success)
        self.assertIn("workspace_write", result.error or "")

    def test_write_file_creates_pending_approval_without_mutating_workspace(self) -> None:
        writable_settings = Settings(
            model_id="dummy",
            model_revision=None,
            force_download=False,
            trust_remote_code=False,
            enable_thinking=False,
            workspace_root=self.workspace,
            max_rounds=2,
            max_actions_per_round=2,
            max_tokens_per_turn=64,
            temperature=0.1,
            allow_shell=False,
            allow_writes=True,
            command_timeout_seconds=5,
            max_file_bytes=10_000,
            max_command_output_chars=10_000,
            host="127.0.0.1",
            port=8000,
        )
        tools = WorkspaceTools(writable_settings)

        result = tools.execute_actions(
            [
                ToolAction(
                    tool="write_file",
                    args={"path": "new.txt", "content": "new contents\n"},
                    reason="create a pending patch",
                )
            ],
            workspace=self.workspace,
            execution_mode="workspace_write",
        )[0]

        self.assertTrue(result.success)
        self.assertFalse((self.workspace / "new.txt").exists())
        self.assertEqual(result.metadata["approval_status"], "pending")

        approval_id = str(result.metadata["approval_id"])
        approval = PatchApprovalStore().get(workspace=self.workspace, approval_id=approval_id)
        self.assertEqual(approval["status"], "pending")
        self.assertEqual(approval["path"], "new.txt")
        self.assertIn("teamai approvals apply", result.output)

    def test_write_file_records_continuation_context_for_follow_up_run(self) -> None:
        writable_settings = Settings(
            model_id="dummy",
            model_revision=None,
            force_download=False,
            trust_remote_code=False,
            enable_thinking=False,
            workspace_root=self.workspace,
            max_rounds=2,
            max_actions_per_round=2,
            max_tokens_per_turn=64,
            temperature=0.1,
            allow_shell=False,
            allow_writes=True,
            command_timeout_seconds=5,
            max_file_bytes=10_000,
            max_command_output_chars=10_000,
            host="127.0.0.1",
            port=8000,
        )
        tools = WorkspaceTools(writable_settings)

        result = tools.execute_actions(
            [
                ToolAction(
                    tool="write_file",
                    args={"path": "new.txt", "content": "new contents\n"},
                    reason="create a pending patch",
                )
            ],
            workspace=self.workspace,
            execution_mode="workspace_write",
            approval_context={
                "task": "Update new.txt and then continue the task.",
                "execution_mode": "workspace_write",
            },
        )[0]

        approval_id = str(result.metadata["approval_id"])
        approval = PatchApprovalStore().get(workspace=self.workspace, approval_id=approval_id)
        self.assertEqual(
            approval["continuation"]["original_task"],
            "Update new.txt and then continue the task.",
        )
        self.assertEqual(approval["continuation"]["requested_execution_mode"], "workspace_write")
        self.assertEqual(approval["continuation"]["target_path"], "new.txt")
        self.assertEqual(approval["continuation"]["source_tool"], "write_file")

    def test_replace_in_file_applies_only_after_explicit_approval(self) -> None:
        writable_settings = Settings(
            model_id="dummy",
            model_revision=None,
            force_download=False,
            trust_remote_code=False,
            enable_thinking=False,
            workspace_root=self.workspace,
            max_rounds=2,
            max_actions_per_round=2,
            max_tokens_per_turn=64,
            temperature=0.1,
            allow_shell=False,
            allow_writes=True,
            command_timeout_seconds=5,
            max_file_bytes=10_000,
            max_command_output_chars=10_000,
            host="127.0.0.1",
            port=8000,
        )
        tools = WorkspaceTools(writable_settings)

        result = tools.execute_actions(
            [
                ToolAction(
                    tool="replace_in_file",
                    args={"path": "example.txt", "old_text": "world", "new_text": "teamai"},
                    reason="prepare a patch",
                )
            ],
            workspace=self.workspace,
            execution_mode="workspace_write",
        )[0]

        self.assertTrue(result.success)
        self.assertEqual((self.workspace / "example.txt").read_text(encoding="utf-8"), "hello\nworld\n")

        approval_id = str(result.metadata["approval_id"])
        applied = PatchApprovalStore().apply(workspace=self.workspace, approval_id=approval_id)
        self.assertEqual(applied["status"], "applied")
        self.assertEqual((self.workspace / "example.txt").read_text(encoding="utf-8"), "hello\nteamai\n")

    def test_list_files_skips_build_artifacts(self) -> None:
        (self.workspace / "build").mkdir()
        (self.workspace / "build" / "junk.txt").write_text("x", encoding="utf-8")
        (self.workspace / "demo.egg-info").mkdir()

        result = self.tools.execute_actions(
            [ToolAction(tool="list_files", args={"path": "."}, reason="inspect")],
            workspace=self.workspace,
            execution_mode="read_only",
        )[0]

        self.assertTrue(result.success)
        self.assertNotIn("build/", result.output)
        self.assertNotIn("demo.egg-info/", result.output)


if __name__ == "__main__":
    unittest.main()
