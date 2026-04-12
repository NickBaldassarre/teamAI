from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from teamai.integrations.bridge_base import AgentBridge
from teamai.verification import VerificationResult


class _FakeBridge(AgentBridge):
    engine = "fake"
    default_model = "fake-model"

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses[:]
        self.prompts: list[str] = []

    def _request_patch(
        self,
        *,
        prompt: str,
        model: str,
        payload,
        payload_path: Path,
        project_root: Path,
    ) -> str:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("No fake handoff response left.")
        return self._responses.pop(0)


class VerifiedBridgeLoopTest(unittest.TestCase):
    def _write_payload(self, root: Path, *, dependencies: list[str] | None = None) -> Path:
        payload_path = root / ".teamai" / "codex_payload.json"
        payload_path.parent.mkdir(parents=True, exist_ok=True)
        payload_path.write_text(
            json.dumps(
                {
                    "original_task": "Update the CLI output path.",
                    "core_dependencies": dependencies or ["teamai/cli.py"],
                    "distilled_context": {"teamai/cli.py": "CLI entrypoint summary."},
                    "recommended_codex_action": "Inspect teamai/cli.py before implementing the change.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return payload_path

    def test_verified_handoff_accepts_structured_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            self._write_payload(project_root)
            bridge = _FakeBridge(
                [
                    json.dumps(
                        {
                            "summary": "Fix the CLI output formatting.",
                            "diff": (
                                "diff --git a/teamai/cli.py b/teamai/cli.py\n"
                                "--- a/teamai/cli.py\n"
                                "+++ b/teamai/cli.py\n"
                                "@@ -1,1 +1,2 @@\n"
                                " from __future__ import annotations\n"
                                "+# patched\n"
                            ),
                            "rationale": "Keep the change scoped to the CLI entrypoint.",
                            "risks": ["low"],
                            "recommended_checks": [["python3", "-m", "unittest", "tests.test_cli"]],
                            "confidence": 0.91,
                        }
                    )
                ]
            )
            verification = VerificationResult(
                success=True,
                log_output="all green",
                patch_returncode=0,
                test_returncode=0,
            )

            with patch("teamai.integrations.bridge_base.verify_patch", return_value=verification), patch(
                "teamai.integrations.bridge_base.PatchApprovalStore.create_bundle_from_patch",
                return_value={"approval_id": "approval123", "change_count": 1},
            ), patch("teamai.integrations.bridge_base.Sandbox") as sandbox_cls:
                sandbox_cls.return_value.__enter__.return_value = type("_Sandbox", (), {"path": project_root})()
                result = bridge.execute_verified(
                    project_root=project_root,
                    payload_file=".teamai/codex_payload.json",
                    patch_file=".teamai/fake.patch",
                )

            self.assertTrue(result.accepted)
            self.assertIsNotNone(result.artifact)
            assert result.artifact is not None
            self.assertEqual(result.artifact.summary, "Fix the CLI output formatting.")
            self.assertEqual(result.revision_requests, ())
            self.assertEqual(result.revision_count, 0)
            self.assertEqual(result.approval, {"approval_id": "approval123", "change_count": 1})
            sandbox_cls.assert_called_once_with(project_root.resolve())

    def test_verified_handoff_rejects_policy_violations_with_revision_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            self._write_payload(project_root, dependencies=["teamai/cli.py"])
            bridge = _FakeBridge(
                [
                    json.dumps(
                        {
                            "summary": "Rewrite dependency config.",
                            "diff": (
                                "diff --git a/pyproject.toml b/pyproject.toml\n"
                                "--- a/pyproject.toml\n"
                                "+++ b/pyproject.toml\n"
                                "@@ -0,0 +1,1 @@\n"
                                "+[tool.example]\n"
                            ),
                            "rationale": "This changes packaging defaults.",
                            "confidence": 0.88,
                        }
                    )
                ]
            )
            verification = VerificationResult(
                success=True,
                log_output="tests passed",
                patch_returncode=0,
                test_returncode=0,
            )

            with patch("teamai.integrations.bridge_base.verify_patch", return_value=verification), patch(
                "teamai.integrations.bridge_base.Sandbox"
            ) as sandbox_cls:
                sandbox_cls.return_value.__enter__.return_value = type("_Sandbox", (), {"path": project_root})()
                result = bridge.execute_verified(
                    project_root=project_root,
                    payload_file=".teamai/codex_payload.json",
                    patch_file=".teamai/fake.patch",
                )

            self.assertFalse(result.accepted)
            self.assertEqual(result.revision_count, 1)
            self.assertEqual(result.approval, None)
            assert result.revision_requests
            reasons = set(result.revision_requests[0].reasons)
            self.assertIn("policy_violations", reasons)
            self.assertIn("incorrect_scope", reasons)
            self.assertIn("Revision required", result.failure_context_file.read_text(encoding="utf-8"))

    def test_verified_handoff_requests_revision_then_accepts_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            self._write_payload(project_root)
            bridge = _FakeBridge(
                [
                    json.dumps(
                        {
                            "summary": "First attempt.",
                            "diff": (
                                "diff --git a/teamai/cli.py b/teamai/cli.py\n"
                                "--- a/teamai/cli.py\n"
                                "+++ b/teamai/cli.py\n"
                                "@@ -1,1 +1,2 @@\n"
                                " from __future__ import annotations\n"
                                "+# broken\n"
                            ),
                            "rationale": "Try a quick patch first.",
                            "confidence": 0.9,
                        }
                    ),
                    json.dumps(
                        {
                            "summary": "Second attempt.",
                            "diff": (
                                "diff --git a/teamai/cli.py b/teamai/cli.py\n"
                                "--- a/teamai/cli.py\n"
                                "+++ b/teamai/cli.py\n"
                                "@@ -1,1 +1,2 @@\n"
                                " from __future__ import annotations\n"
                                "+# fixed\n"
                            ),
                            "rationale": "Apply the requested revision and keep scope narrow.",
                            "confidence": 0.94,
                        }
                    ),
                ]
            )
            verifications = [
                VerificationResult(
                    success=False,
                    log_output="tests failed",
                    patch_returncode=0,
                    test_returncode=1,
                ),
                VerificationResult(
                    success=True,
                    log_output="all green",
                    patch_returncode=0,
                    test_returncode=0,
                ),
            ]

            with patch("teamai.integrations.bridge_base.verify_patch", side_effect=verifications), patch(
                "teamai.integrations.bridge_base.PatchApprovalStore.create_bundle_from_patch",
                return_value={"approval_id": "approval123", "change_count": 1},
            ), patch("teamai.integrations.bridge_base.Sandbox") as sandbox_cls:
                sandbox_cls.return_value.__enter__.return_value = type("_Sandbox", (), {"path": project_root})()
                result = bridge.execute_verified(
                    project_root=project_root,
                    payload_file=".teamai/codex_payload.json",
                    patch_file=".teamai/fake.patch",
                    max_revision_attempts=1,
                )

            self.assertTrue(result.accepted)
            self.assertEqual(result.revision_count, 1)
            self.assertEqual(len(result.revision_requests), 1)
            self.assertIn("failing_tests", result.revision_requests[0].reasons)
            self.assertIn("Revision request", bridge.prompts[1])
            self.assertEqual(result.approval, {"approval_id": "approval123", "change_count": 1})


if __name__ == "__main__":
    unittest.main()
