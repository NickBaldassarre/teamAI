from __future__ import annotations
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from teamai.autonomy import new_run_state
from teamai.config import Settings
from teamai.integrations.bridge_base import BridgeExecutionResult, VerifiedBridgeExecutionResult
from teamai.model_backend import ModelResponse
from teamai.schemas import CheckExecution, HandoffArtifact, HandoffRevisionRequest, RunRequest, VerifierOutput
from teamai.supervisor import ClosedLoopSupervisor
from teamai.verification import VerificationResult


class FakeBackend:
    def __init__(self, responses: list[str], *, repeat_last: bool = False) -> None:
        self._responses = responses[:]
        self._repeat_last = repeat_last
        self._last_response = responses[-1] if responses else ""

    @property
    def model_loaded(self) -> bool:
        return True

    def generate_messages(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        enable_thinking: bool | None = None,
    ) -> ModelResponse:
        if not self._responses:
            if not self._repeat_last:
                raise AssertionError("No fake responses left for backend.")
            text = self._last_response
        else:
            text = self._responses.pop(0)
            self._last_response = text
        return ModelResponse(
            text=text,
            prompt_tokens=1,
            generation_tokens=1,
            total_tokens=2,
            prompt_tps=1.0,
            generation_tps=1.0,
            peak_memory_gb=0.0,
        )


class FakeVerifiedBridge:
    def __init__(
        self,
        *,
        patch_text: str,
        accepted: bool = True,
        revision_count: int = 0,
        revision_requests: tuple[HandoffRevisionRequest, ...] = (),
        summary: str = "Bridge patch",
    ) -> None:
        self.patch_text = patch_text
        self.accepted = accepted
        self.revision_count = revision_count
        self.revision_requests = revision_requests
        self.summary = summary
        self.calls = 0

    def execute_verified(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        project_root = Path(kwargs["project_root"])
        payload_path = project_root / kwargs["payload_file"]
        patch_path = project_root / kwargs["patch_file"]
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(self.patch_text, encoding="utf-8")
        failure_context = project_root / ".teamai" / "inline-failure.log"
        failure_context.parent.mkdir(parents=True, exist_ok=True)
        if self.accepted:
            if failure_context.exists():
                failure_context.unlink()
        else:
            failure_context.write_text("revision context\n", encoding="utf-8")
        return VerifiedBridgeExecutionResult(
            execution=BridgeExecutionResult(
                engine="local",
                model=str(kwargs.get("model") or "bridge-model"),
                payload_file=payload_path,
                patch_file=patch_path,
                prompt="inline verified handoff",
                patch_text=self.patch_text,
            ),
            verification=VerificationResult(
                success=self.accepted,
                log_output="all green" if self.accepted else "tests failed",
                patch_returncode=0,
                test_returncode=0 if self.accepted else 1,
            ),
            failure_context_file=failure_context,
            approval=None,
            approval_error=None,
            artifact=HandoffArtifact(
                engine="local",
                summary=self.summary,
                diff=self.patch_text,
                rationale="Apply a verified inline patch.",
                confidence=0.92 if self.accepted else 0.3,
            ),
            revision_requests=self.revision_requests,
            accepted=self.accepted,
            revision_count=self.revision_count,
        )


class AutonomousSupervisorTest(unittest.TestCase):
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
            max_rounds=3,
            max_actions_per_round=2,
            max_tokens_per_turn=128,
            temperature=0.1,
            allow_shell=False,
            allow_writes=True,
            command_timeout_seconds=10,
            max_file_bytes=10_000,
            max_command_output_chars=10_000,
            host="127.0.0.1",
            port=8000,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _create_python_fixture(self) -> None:
        (self.workspace / "calc.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
        (self.workspace / "tests").mkdir(exist_ok=True)
        (self.workspace / "tests" / "test_calc.py").write_text(
            "import unittest\nfrom calc import answer\n\n\nclass CalcTest(unittest.TestCase):\n    def test_answer(self) -> None:\n        self.assertEqual(answer(), 42)\n",
            encoding="utf-8",
        )

    def _init_git_repo(self) -> None:
        subprocess.run(["git", "init"], cwd=self.workspace, capture_output=True, text=True, check=True)
        subprocess.run(["git", "config", "user.email", "teamai@example.com"], cwd=self.workspace, capture_output=True, text=True, check=True)
        subprocess.run(["git", "config", "user.name", "teamAI"], cwd=self.workspace, capture_output=True, text=True, check=True)
        subprocess.run(["git", "add", "calc.py", "tests/test_calc.py"], cwd=self.workspace, capture_output=True, text=True, check=True)
        subprocess.run(["git", "commit", "-m", "init"], cwd=self.workspace, capture_output=True, text=True, check=True)

    def test_autonomous_loop_repairs_after_first_failed_patch(self) -> None:
        self._create_python_fixture()
        backend = FakeBackend(
            [
                "Read calc.py and patch it.",
                "Prefer a direct write and then validate with tests.",
                (
                    '{"summary":"Apply an initial patch.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"write_file","reason":"Update calc.py.","args":{"path":"calc.py","content":"def answer():\\n    return 0\\n"}}]}'
                ),
                "The failing test points back to calc.py.",
                "Fix only the incorrect return value and rerun checks.",
                (
                    '{"summary":"Repair calc.py.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"write_file","reason":"Repair calc.py.","args":{"path":"calc.py","content":"def answer():\\n    return 42\\n"}}]}'
                ),
                '{"done":true,"confidence":0.9,"summary":"Scoped checks passed; the task is complete.","next_focus":"none"}',
            ]
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=backend)

        result = supervisor.run(
            RunRequest(
                task="Update calc.py so answer() returns the correct value and tests pass.",
                workspace_path=".",
                execution_mode="workspace_write",
                write_policy="auto_apply_low_risk",
                max_repair_attempts=2,
            )
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.stop_reason, "autonomous_checks_passed")
        self.assertEqual((self.workspace / "calc.py").read_text(encoding="utf-8"), "def answer():\n    return 42\n")
        assert result.run_state is not None
        self.assertEqual(len(result.run_state.failures_encountered), 1)
        self.assertGreaterEqual(result.run_state.metrics.get("retry_count", 0.0), 1.0)

    def test_default_safe_policy_auto_applies_low_risk_change_without_explicit_override(self) -> None:
        self._create_python_fixture()
        backend = FakeBackend(
            [
                "Update calc.py directly.",
                "Keep the change narrow.",
                (
                    '{"summary":"Apply the correct patch.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"write_file","reason":"Fix calc.py.","args":{"path":"calc.py","content":"def answer():\\n    return 42\\n"}}]}'
                ),
                '{"done":true,"confidence":0.95,"summary":"Scoped checks passed; the task is complete.","next_focus":"none"}',
            ]
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=backend)

        result = supervisor.run(
            RunRequest(
                task="Update calc.py so answer() returns the correct value and tests pass.",
                workspace_path=".",
                execution_mode="workspace_write",
            )
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual((self.workspace / "calc.py").read_text(encoding="utf-8"), "def answer():\n    return 42\n")

    def test_autonomous_retry_budget_exhaustion_keeps_primary_workspace_unchanged(self) -> None:
        self._create_python_fixture()
        backend = FakeBackend(
            [
                "Patch calc.py.",
                "Use a small write and validate it.",
                (
                    '{"summary":"Apply an incorrect patch.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"write_file","reason":"Break calc.py.","args":{"path":"calc.py","content":"def answer():\\n    return 0\\n"}}]}'
                ),
            ]
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=backend)

        result = supervisor.run(
            RunRequest(
                task="Update calc.py so answer() returns the correct value and tests pass.",
                workspace_path=".",
                execution_mode="workspace_write",
                write_policy="auto_apply_low_risk",
                max_repair_attempts=1,
            )
        )

        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stop_reason, "repair_budget_exhausted")
        self.assertEqual((self.workspace / "calc.py").read_text(encoding="utf-8"), "def answer():\n    return 1\n")
        assert result.run_state is not None
        self.assertEqual(len(result.run_state.failures_encountered), 1)

    def test_successful_autonomous_run_can_end_with_commit(self) -> None:
        self._create_python_fixture()
        self._init_git_repo()

        backend = FakeBackend(
            [
                "Update calc.py directly.",
                "Keep the change narrow and validate it.",
                (
                    '{"summary":"Apply the correct patch.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"write_file","reason":"Fix calc.py.","args":{"path":"calc.py","content":"def answer():\\n    return 42\\n"}}]}'
                ),
                '{"done":true,"confidence":0.95,"summary":"Scoped checks passed; the task is complete.","next_focus":"none"}',
            ]
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=backend)

        result = supervisor.run(
            RunRequest(
                task="Update calc.py so answer() returns the correct value and tests pass.",
                workspace_path=".",
                execution_mode="workspace_write",
                write_policy="auto_apply_low_risk",
                auto_commit=True,
                max_repair_attempts=1,
            )
        )

        self.assertEqual(result.status, "completed")
        self.assertTrue(result.commit_metadata)
        current_branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=self.workspace,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertTrue(current_branch.startswith("teamai/run_"))
        latest_log = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=self.workspace,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertIn("teamai:", latest_log)

    def test_inline_escalation_switches_to_stronger_local_model_and_commits(self) -> None:
        self._create_python_fixture()
        self._init_git_repo()
        escalation_settings = replace(
            self.settings,
            model_id="mlx-community/gemma-4-2b-it-4bit",
            model_router=False,
        )
        weak_backend = FakeBackend(
            [
                "Inspect calc.py and patch it.",
                "Try a narrow change first.",
                (
                    '{"summary":"Apply an incorrect patch.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"write_file","reason":"Patch calc.py.","args":{"path":"calc.py","content":"def answer():\\n    return 0\\n"}}]}'
                ),
            ]
        )
        strong_backend = FakeBackend(
            [
                "Use the failing assertion and repair only calc.py.",
                "Keep the patch minimal and verified.",
                (
                    '{"summary":"Repair calc.py.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"write_file","reason":"Repair calc.py.","args":{"path":"calc.py","content":"def answer():\\n    return 42\\n"}}]}'
                ),
                '{"done":true,"confidence":0.96,"summary":"Scoped checks passed after escalation.","next_focus":"none"}',
            ]
        )
        supervisor = ClosedLoopSupervisor(escalation_settings, backend=weak_backend)
        supervisor._backend_by_model["mlx-community/gemma-4-12b-it-4bit"] = strong_backend  # noqa: SLF001

        result = supervisor.run(
            RunRequest(
                task="Update calc.py so answer() returns the correct value and tests pass.",
                workspace_path=".",
                execution_mode="workspace_write",
                write_policy="auto_apply_low_risk",
                max_repair_attempts=1,
                auto_commit=True,
            )
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.model_id, "mlx-community/gemma-4-12b-it-4bit")
        self.assertEqual((self.workspace / "calc.py").read_text(encoding="utf-8"), "def answer():\n    return 42\n")
        assert result.run_state is not None
        escalation_entries = [entry for entry in result.run_state.routing_trace if entry.stage == "inline_escalation"]
        self.assertTrue(escalation_entries)
        self.assertEqual(escalation_entries[-1].model_id, "mlx-community/gemma-4-12b-it-4bit")
        self.assertTrue(result.commit_metadata)

    def test_inline_verified_handoff_consumes_revised_patch_without_restart(self) -> None:
        self._create_python_fixture()
        handoff_settings = replace(
            self.settings,
            model_id="mlx-community/gemma-4-12b-it-4bit",
            model_router=False,
        )
        backend = FakeBackend(
            [
                "Inspect calc.py and attempt a fix.",
                "Try the smallest plausible patch.",
                (
                    '{"summary":"Apply an incorrect patch.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"write_file","reason":"Patch calc.py.","args":{"path":"calc.py","content":"def answer():\\n    return 0\\n"}}]}'
                ),
            ]
        )
        bridge = FakeVerifiedBridge(
            patch_text=(
                "diff --git a/calc.py b/calc.py\n"
                "--- a/calc.py\n"
                "+++ b/calc.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def answer():\n"
                "-    return 0\n"
                "+    return 42\n"
            ),
            accepted=True,
            revision_count=1,
            revision_requests=(
                HandoffRevisionRequest(
                    reasons=["failing_tests"],
                    details="Failing test excerpts:\nAssertionError: 0 != 42",
                ),
            ),
            summary="Revised bridge patch fixed calc.py.",
        )
        supervisor = ClosedLoopSupervisor(handoff_settings, backend=backend)

        with patch("teamai.integrations.get_bridge", return_value=bridge):
            result = supervisor.run(
                RunRequest(
                    task="Update calc.py so answer() returns the correct value and tests pass.",
                    workspace_path=".",
                    execution_mode="workspace_write",
                    write_policy="auto_apply_low_risk",
                    max_repair_attempts=1,
                    handoff_engine="local",
                )
            )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.stop_reason, "inline_verified_handoff_verified")
        self.assertEqual((self.workspace / "calc.py").read_text(encoding="utf-8"), "def answer():\n    return 42\n")
        assert result.run_state is not None
        self.assertEqual(len(result.run_state.handoffs), 1)
        self.assertEqual(bridge.calls, 1)
        self.assertTrue(any("1 revision request" in warning for warning in result.warnings))

    def test_auto_push_pushes_review_branch_to_local_remote(self) -> None:
        self._create_python_fixture()
        self._init_git_repo()
        remote_dir = tempfile.TemporaryDirectory()
        self.addCleanup(remote_dir.cleanup)
        remote_path = Path(remote_dir.name) / "remote.git"
        subprocess.run(["git", "init", "--bare", str(remote_path)], capture_output=True, text=True, check=True)
        subprocess.run(["git", "remote", "add", "origin", str(remote_path)], cwd=self.workspace, capture_output=True, text=True, check=True)

        push_settings = replace(self.settings, allow_git_push=True)
        backend = FakeBackend(
            [
                "Update calc.py directly.",
                "Keep the change narrow and validated.",
                (
                    '{"summary":"Apply the correct patch.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"write_file","reason":"Fix calc.py.","args":{"path":"calc.py","content":"def answer():\\n    return 42\\n"}}]}'
                ),
                '{"done":true,"confidence":0.95,"summary":"Scoped checks passed; the task is complete.","next_focus":"none"}',
            ]
        )
        supervisor = ClosedLoopSupervisor(push_settings, backend=backend)

        result = supervisor.run(
            RunRequest(
                task="Update calc.py so answer() returns the correct value and tests pass.",
                workspace_path=".",
                execution_mode="workspace_write",
                write_policy="auto_apply_low_risk",
                auto_push=True,
            )
        )

        self.assertEqual(result.status, "completed")
        assert result.run_state is not None
        self.assertEqual(result.run_state.pushed_remote, "origin")
        self.assertTrue(result.run_state.pushed_branch)
        remote_branches = subprocess.run(
            ["git", "--git-dir", str(remote_path), "branch", "--list"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertIn(result.run_state.pushed_branch, remote_branches)

    def test_auto_push_is_blocked_when_verification_remains_unresolved(self) -> None:
        self._create_python_fixture()
        self._init_git_repo()
        remote_dir = tempfile.TemporaryDirectory()
        self.addCleanup(remote_dir.cleanup)
        remote_path = Path(remote_dir.name) / "remote.git"
        subprocess.run(["git", "init", "--bare", str(remote_path)], capture_output=True, text=True, check=True)
        subprocess.run(["git", "remote", "add", "origin", str(remote_path)], cwd=self.workspace, capture_output=True, text=True, check=True)
        sandbox_dir = tempfile.TemporaryDirectory()
        self.addCleanup(sandbox_dir.cleanup)
        sandbox_workspace = Path(sandbox_dir.name)
        for relative in ["calc.py", "tests/test_calc.py"]:
            target = sandbox_workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((self.workspace / relative).read_text(encoding="utf-8"), encoding="utf-8")
        (sandbox_workspace / "calc.py").write_text("def answer():\n    return 42\n", encoding="utf-8")

        supervisor = ClosedLoopSupervisor(replace(self.settings, allow_git_push=True), backend=FakeBackend([]))
        run_state = new_run_state(workspace=self.workspace, policy="auto_apply_low_risk")
        run_state.files_changed = ["calc.py"]
        run_state.checks_run = [
            CheckExecution(command=["python3", "-m", "unittest", "tests.test_calc"], scope="repo", returncode=1, stdout="", stderr="AssertionError")
        ]
        run_state.verifier_outputs = [
            VerifierOutput(source="model", passed=False, confidence=0.2, summary="Checks are still failing.")
        ]

        merged = supervisor._merge_autonomous_changes(  # noqa: SLF001
            workspace=self.workspace,
            sandbox_workspace=sandbox_workspace,
            task="Update calc.py so answer() returns the correct value and tests pass.",
            run_state=run_state,
            write_policy="auto_apply_low_risk",
            allowed_scopes=("calc.py",),
            auto_commit=True,
            auto_push=True,
            push_remote="origin",
            push_branch_name=None,
        )

        self.assertEqual(merged["stop_reason"], "approval_required")
        remote_branches = subprocess.run(
            ["git", "--git-dir", str(remote_path), "branch", "--list"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        self.assertNotIn("teamai/", remote_branches)

    def test_auto_push_blocks_protected_branch_override(self) -> None:
        self._create_python_fixture()
        self._init_git_repo()
        remote_dir = tempfile.TemporaryDirectory()
        self.addCleanup(remote_dir.cleanup)
        remote_path = Path(remote_dir.name) / "remote.git"
        subprocess.run(["git", "init", "--bare", str(remote_path)], capture_output=True, text=True, check=True)
        subprocess.run(["git", "remote", "add", "origin", str(remote_path)], cwd=self.workspace, capture_output=True, text=True, check=True)

        backend = FakeBackend(
            [
                "Update calc.py directly.",
                "Keep the change narrow and validated.",
                (
                    '{"summary":"Apply the correct patch.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"write_file","reason":"Fix calc.py.","args":{"path":"calc.py","content":"def answer():\\n    return 42\\n"}}]}'
                ),
                '{"done":true,"confidence":0.95,"summary":"Scoped checks passed; the task is complete.","next_focus":"none"}',
            ]
        )
        supervisor = ClosedLoopSupervisor(replace(self.settings, allow_git_push=True), backend=backend)

        result = supervisor.run(
            RunRequest(
                task="Update calc.py so answer() returns the correct value and tests pass.",
                workspace_path=".",
                execution_mode="workspace_write",
                auto_push=True,
                push_branch_name="main",
            )
        )

        self.assertEqual(result.status, "completed")
        assert result.run_state is not None
        self.assertIsNone(result.run_state.pushed_branch)
        self.assertTrue(any("protected" in warning.lower() for warning in result.warnings))

    # ------------------------------------------------------------------ verify_before_commit

    def _prepare_verify_gate_sandbox(self) -> tuple[Path, Path]:
        """Create a git-backed workspace and a sandbox with a pending calc.py change.

        Returns (workspace, sandbox_workspace) for direct ``_merge_autonomous_changes``
        testing — bypasses the supervisor loop so we can drive check-state precisely.
        """
        self._create_python_fixture()
        self._init_git_repo()
        sandbox_dir = tempfile.TemporaryDirectory()
        self.addCleanup(sandbox_dir.cleanup)
        sandbox_workspace = Path(sandbox_dir.name)
        (sandbox_workspace / "calc.py").write_text(
            "def answer():\n    return 99\n", encoding="utf-8"
        )
        (sandbox_workspace / "tests").mkdir(exist_ok=True)
        (sandbox_workspace / "tests" / "test_calc.py").write_text(
            (self.workspace / "tests" / "test_calc.py").read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        return self.workspace, sandbox_workspace

    def test_verify_before_commit_blocks_merge_when_checks_not_green(self) -> None:
        """With no green checks recorded, verify_before_commit short-circuits the merge.

        The primary workspace is never touched (no patch application, no approval created,
        no commit).  stop_reason flags the verification failure so callers can act on it.
        """
        workspace, sandbox_workspace = self._prepare_verify_gate_sandbox()
        supervisor = ClosedLoopSupervisor(self.settings, backend=FakeBackend([], repeat_last=True))
        run_state = new_run_state(workspace=workspace, policy="auto_apply_low_risk")
        run_state.files_changed = ["calc.py"]
        # No checks_run entries → latest_checks_green resolves to False, tripping the gate.

        merged = supervisor._merge_autonomous_changes(  # noqa: SLF001
            workspace=workspace,
            sandbox_workspace=sandbox_workspace,
            task="Verify-gate: mutate calc.py without running the checks.",
            run_state=run_state,
            write_policy="auto_apply_low_risk",
            allowed_scopes=(),
            auto_commit=True,
            auto_push=False,
            push_remote="origin",
            push_branch_name=None,
            verify_before_commit=True,
        )

        self.assertFalse(merged["applied"])
        self.assertEqual(merged["stop_reason"], "verification_failed")
        self.assertTrue(
            any("verify_before_commit" in warning for warning in merged["warnings"]),
            f"Expected a verify_before_commit warning, got: {merged['warnings']!r}",
        )
        # Primary workspace was never touched: calc.py still has its pre-run content.
        self.assertEqual(
            (workspace / "calc.py").read_text(encoding="utf-8"),
            "def answer():\n    return 1\n",
        )
        # No feature branch was created.
        current_branch = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertNotIn("teamai/", current_branch)
        # No second commit — history still has only the init commit.
        commit_count = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            cwd=workspace,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        self.assertEqual(commit_count, "1")
        # A verify_gate entry landed in the routing trace for observability.
        verify_entries = [entry for entry in run_state.routing_trace if entry.stage == "verify_gate"]
        self.assertEqual(len(verify_entries), 1)
        self.assertEqual(verify_entries[0].outcome, "verification_failed")

    def test_verify_before_commit_permits_merge_when_checks_passed(self) -> None:
        """A green check lets the gate pass; normal merge proceeds."""
        workspace, sandbox_workspace = self._prepare_verify_gate_sandbox()
        supervisor = ClosedLoopSupervisor(self.settings, backend=FakeBackend([], repeat_last=True))
        run_state = new_run_state(workspace=workspace, policy="auto_apply_low_risk")
        run_state.files_changed = ["calc.py"]
        run_state.checks_run = [
            CheckExecution(
                command=["python3", "-m", "unittest", "discover", "-s", "tests"],
                scope="repo",
                returncode=0,
                stdout="OK\n",
                stderr="",
            )
        ]
        # Push verifier confidence past the auto_apply_low_risk threshold (0.55) so the
        # merge auto-applies rather than stopping at an approval gate.
        run_state.verifier_outputs = [
            VerifierOutput(
                source="model",
                passed=True,
                confidence=0.9,
                summary="Scoped checks passed.",
                failure_types=[],
                mismatches=[],
                next_focus="none",
            )
        ]

        merged = supervisor._merge_autonomous_changes(  # noqa: SLF001
            workspace=workspace,
            sandbox_workspace=sandbox_workspace,
            task="Verify-gate: mutate calc.py with a green check recorded.",
            run_state=run_state,
            write_policy="auto_apply_low_risk",
            allowed_scopes=(),
            auto_commit=True,
            auto_push=False,
            push_remote="origin",
            push_branch_name=None,
            verify_before_commit=True,
        )

        self.assertTrue(merged["applied"])
        self.assertNotEqual(merged["stop_reason"], "verification_failed")
        self.assertTrue(merged.get("commit_metadata"))
        self.assertEqual(
            (workspace / "calc.py").read_text(encoding="utf-8"),
            "def answer():\n    return 99\n",
        )

    def test_verify_before_commit_default_false_does_not_block_merge(self) -> None:
        """Regression guard: the flag is opt-in, so default behavior is unchanged.

        Without verify_before_commit, the merge proceeds through the policy as before —
        an approval may be created for risk reasons, but the stop_reason is never
        ``verification_failed``.
        """
        workspace, sandbox_workspace = self._prepare_verify_gate_sandbox()
        supervisor = ClosedLoopSupervisor(self.settings, backend=FakeBackend([], repeat_last=True))
        run_state = new_run_state(workspace=workspace, policy="auto_apply_low_risk")
        run_state.files_changed = ["calc.py"]

        merged = supervisor._merge_autonomous_changes(  # noqa: SLF001
            workspace=workspace,
            sandbox_workspace=sandbox_workspace,
            task="Verify-gate default: mutate calc.py without the new flag set.",
            run_state=run_state,
            write_policy="auto_apply_low_risk",
            allowed_scopes=(),
            auto_commit=True,
            auto_push=False,
            push_remote="origin",
            push_branch_name=None,
        )

        self.assertNotEqual(merged["stop_reason"], "verification_failed")
        # No verify_gate routing entry when the flag is off.
        self.assertFalse(
            [entry for entry in run_state.routing_trace if entry.stage == "verify_gate"],
            "verify_gate routing entry should only appear when the flag is on.",
        )
