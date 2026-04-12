from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from teamai.approvals import PatchApprovalStore
from teamai.autonomy import PatchExecutionContext
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

    def test_write_file_bundle_creates_one_pending_approval_for_multiple_files(self) -> None:
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
        (self.workspace / "tests").mkdir()
        (self.workspace / "tests" / "test_example.py").write_text("old test\n", encoding="utf-8")
        tools = WorkspaceTools(writable_settings)

        result = tools.execute_actions(
            [
                ToolAction(
                    tool="write_file",
                    args={
                        "changes": [
                            {"path": "example.txt", "content": "hello\nteamai\n"},
                            {"path": "tests/test_example.py", "content": "new test\n"},
                        ]
                    },
                    reason="prepare a bundled patch",
                )
            ],
            workspace=self.workspace,
            execution_mode="workspace_write",
            approval_context={
                "task": "Update the implementation and its directly related unittest.",
                "execution_mode": "workspace_write",
            },
        )[0]

        self.assertTrue(result.success)
        approval = PatchApprovalStore().get(workspace=self.workspace, approval_id=str(result.metadata["approval_id"]))
        self.assertEqual(approval["change_count"], 2)
        self.assertEqual(approval["path"], "<multiple files>")
        self.assertEqual(approval["changed_paths"], ["example.txt", "tests/test_example.py"])
        self.assertEqual(result.metadata["change_count"], 2)
        self.assertEqual(result.metadata["changed_paths"], ["example.txt", "tests/test_example.py"])
        self.assertEqual((self.workspace / "example.txt").read_text(encoding="utf-8"), "hello\nworld\n")
        self.assertEqual((self.workspace / "tests" / "test_example.py").read_text(encoding="utf-8"), "old test\n")

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

    def test_write_file_auto_apply_policy_writes_directly_when_risk_is_low(self) -> None:
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
                    args={"path": "example.txt", "content": "hello\nteamai\n"},
                    reason="apply a narrow low-risk change",
                )
            ],
            workspace=self.workspace,
            execution_mode="workspace_write",
            write_policy="auto_apply_low_risk",
            patch_context=PatchExecutionContext(policy="auto_apply_low_risk", phase="sandbox"),
        )[0]

        self.assertTrue(result.success)
        self.assertEqual((self.workspace / "example.txt").read_text(encoding="utf-8"), "hello\nteamai\n")
        self.assertFalse(result.metadata["requires_approval"])

    def test_write_file_auto_apply_escalates_high_risk_manifest_changes(self) -> None:
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
        (self.workspace / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
        tools = WorkspaceTools(writable_settings)

        result = tools.execute_actions(
            [
                ToolAction(
                    tool="write_file",
                    args={"path": "pyproject.toml", "content": "[project]\nname='demo'\nversion='0.2.0'\n"},
                    reason="change manifest metadata",
                )
            ],
            workspace=self.workspace,
            execution_mode="workspace_write",
            write_policy="auto_apply_low_risk",
            patch_context=PatchExecutionContext(
                policy="auto_apply_low_risk",
                phase="workspace",
                tests_passed=True,
                verifier_confidence=0.9,
            ),
        )[0]

        self.assertTrue(result.success)
        self.assertTrue(result.metadata["requires_approval"])
        self.assertNotIn("version='0.2.0'", (self.workspace / "pyproject.toml").read_text(encoding="utf-8"))

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
