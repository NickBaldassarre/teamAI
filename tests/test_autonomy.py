from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from teamai.autonomy import (
    ContextPackager,
    PatchExecutionContext,
    PatchExecutor,
    RepoIndexer,
    SafeCommandExecutor,
    build_check_commands,
    derive_write_policy,
)


class PatchExecutorPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        (self.workspace / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
        (self.workspace / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
        self.executor = PatchExecutor()

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_propose_only_path_creates_pending_approval_without_mutating_file(self) -> None:
        receipt = self.executor.execute_bundle(
            workspace=self.workspace,
            changes=[
                {
                    "path": "app.py",
                    "before_exists": True,
                    "after_exists": True,
                    "before_text": "def answer():\n    return 1\n",
                    "after_text": "def answer():\n    return 2\n",
                }
            ],
            reason="propose a change",
            source_tool="write_file",
            context=PatchExecutionContext(policy="propose_only", phase="workspace"),
        )

        self.assertTrue(receipt.approval_required)
        self.assertFalse(receipt.applied)
        self.assertEqual((self.workspace / "app.py").read_text(encoding="utf-8"), "def answer():\n    return 1\n")

    def test_auto_apply_low_risk_path_writes_directly_in_sandbox_phase(self) -> None:
        receipt = self.executor.execute_bundle(
            workspace=self.workspace,
            changes=[
                {
                    "path": "app.py",
                    "before_exists": True,
                    "after_exists": True,
                    "before_text": "def answer():\n    return 1\n",
                    "after_text": "def answer():\n    return 2\n",
                }
            ],
            reason="apply a safe change",
            source_tool="write_file",
            context=PatchExecutionContext(policy="auto_apply_low_risk", phase="sandbox"),
        )

        self.assertTrue(receipt.applied)
        self.assertFalse(receipt.approval_required)
        self.assertEqual((self.workspace / "app.py").read_text(encoding="utf-8"), "def answer():\n    return 2\n")

    def test_high_risk_workspace_change_escalates_even_under_auto_apply(self) -> None:
        receipt = self.executor.execute_bundle(
            workspace=self.workspace,
            changes=[
                {
                    "path": "pyproject.toml",
                    "before_exists": True,
                    "after_exists": True,
                    "before_text": "[project]\nname='demo'\n",
                    "after_text": "[project]\nname='demo'\nversion='0.2.0'\n",
                }
            ],
            reason="change dependency manifest metadata",
            source_tool="write_file",
            context=PatchExecutionContext(
                policy="auto_apply_low_risk",
                phase="workspace",
                tests_passed=True,
                verifier_confidence=0.95,
            ),
        )

        self.assertTrue(receipt.approval_required)
        self.assertFalse(receipt.applied)
        self.assertIn("high_risk_path:pyproject.toml", receipt.risk_reasons)
        self.assertNotIn("version='0.2.0'", (self.workspace / "pyproject.toml").read_text(encoding="utf-8"))

    def test_workspace_write_defaults_to_safe_auto_apply_policy(self) -> None:
        self.assertEqual(
            derive_write_policy(execution_mode="workspace_write", requested_policy=None),
            "auto_apply_low_risk",
        )


class RepoIndexerTest(unittest.TestCase):
    def test_repo_indexer_builds_symbol_and_test_maps(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "teamai").mkdir()
            (workspace / "tests").mkdir()
            (workspace / "teamai" / "sample.py").write_text(
                "import math\n\n\nclass Demo:\n    pass\n\n\ndef normalize(value: str) -> str:\n    return value.strip()\n",
                encoding="utf-8",
            )
            (workspace / "tests" / "test_sample.py").write_text(
                "import unittest\nfrom teamai.sample import normalize\n",
                encoding="utf-8",
            )
            (workspace / "README.md").write_text("# demo\n", encoding="utf-8")

            index = RepoIndexer().build(workspace)

        self.assertIn("teamai/sample.py", index.symbol_map)
        self.assertIn("Demo", index.symbol_map["teamai/sample.py"])
        self.assertIn("normalize", index.symbol_map["teamai/sample.py"])
        self.assertEqual(index.test_map["teamai/sample.py"], ["tests/test_sample.py"])
        self.assertIn("README.md", index.config_entrypoints)

    def test_build_check_commands_prefers_related_unittest_before_repo_suite(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "calc.py").write_text("def answer():\n    return 42\n", encoding="utf-8")
            (workspace / "tests").mkdir()
            (workspace / "tests" / "test_calc.py").write_text(
                "import unittest\nfrom calc import answer\n\n\nclass CalcTest(unittest.TestCase):\n    def test_answer(self) -> None:\n        self.assertEqual(answer(), 42)\n",
                encoding="utf-8",
            )
            commands = build_check_commands(workspace=workspace, changed_paths=["calc.py"])

        self.assertEqual(commands[0][-1], "tests.test_calc")

    def test_context_packager_collects_related_tests_imports_and_call_sites(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "teamai").mkdir()
            (workspace / "tests").mkdir()
            (workspace / "teamai" / "sample.py").write_text(
                "from teamai.helpers import helper_value\n\n\ndef normalize(value: str) -> str:\n    return helper_value(value).strip()\n",
                encoding="utf-8",
            )
            (workspace / "teamai" / "helpers.py").write_text(
                "def helper_value(value: str) -> str:\n    return value.lower()\n",
                encoding="utf-8",
            )
            (workspace / "teamai" / "service.py").write_text(
                "from teamai.sample import normalize\n\n\ndef handle(value: str) -> str:\n    return normalize(value)\n",
                encoding="utf-8",
            )
            (workspace / "tests" / "test_sample.py").write_text(
                "import unittest\nfrom teamai.sample import normalize\n\n\nclass SampleTest(unittest.TestCase):\n    def test_normalize(self) -> None:\n        self.assertEqual(normalize(' X '), 'x')\n",
                encoding="utf-8",
            )

            repo_index = RepoIndexer().build(workspace)
            bundle = ContextPackager().build(
                task="Fix normalize in teamai/sample.py so tests pass.",
                workspace=workspace,
                repo_index=repo_index,
                task_scopes=("teamai/sample.py",),
                observed_paths=("teamai/sample.py",),
                changed_paths=("teamai/sample.py",),
                failure_output="AssertionError: normalize returned the wrong value in tests.test_sample",
                prior_failed_repairs=("failed: adjust normalize narrowly",),
            )

        self.assertIn("teamai/sample.py", bundle.relevant_paths)
        self.assertIn("tests/test_sample.py", bundle.failing_tests)
        self.assertIn("teamai/sample.py", bundle.nearest_imports)
        self.assertTrue(bundle.dependent_call_sites)
        self.assertIn("failed:", bundle.prior_failed_repairs[0])


class SafeCommandExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        (self.workspace / "tracked.txt").write_text("initial\n", encoding="utf-8")
        subprocess.run(["git", "init"], cwd=self.workspace, capture_output=True, text=True, check=True)
        subprocess.run(["git", "config", "user.email", "teamai@example.com"], cwd=self.workspace, capture_output=True, text=True, check=True)
        subprocess.run(["git", "config", "user.name", "teamAI"], cwd=self.workspace, capture_output=True, text=True, check=True)
        subprocess.run(["git", "add", "tracked.txt"], cwd=self.workspace, capture_output=True, text=True, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.workspace, capture_output=True, text=True, check=True)
        self.executor = SafeCommandExecutor(timeout_seconds=10)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_invalid_branch_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.executor.create_branch(workspace=self.workspace, branch_name="../nope")

    def test_git_add_and_commit_work_for_valid_paths(self) -> None:
        (self.workspace / "tracked.txt").write_text("updated\n", encoding="utf-8")

        branch_result = self.executor.create_branch(workspace=self.workspace, branch_name="teamai/test-safe-tools")
        add_result = self.executor.git_add(workspace=self.workspace, paths=["tracked.txt"])
        commit_result = self.executor.git_commit(workspace=self.workspace, message="safe tool commit")

        self.assertEqual(branch_result.returncode, 0)
        self.assertEqual(add_result.returncode, 0)
        self.assertEqual(commit_result.returncode, 0)
        self.assertIn("safe tool commit", subprocess.run(["git", "log", "--oneline", "-1"], cwd=self.workspace, capture_output=True, text=True, check=True).stdout)

    def test_git_push_blocks_protected_default_branch(self) -> None:
        push_result = self.executor.git_push(
            workspace=self.workspace,
            remote="origin",
            branch_name="main",
        )

        self.assertNotEqual(push_result.returncode, 0)
        self.assertIn("protected or default branch", push_result.stderr.lower())
