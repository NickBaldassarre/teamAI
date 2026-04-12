from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from teamai.integrations.bridge_base import AgentBridge
from teamai.integrations.local_bridge import execute_local_handoff, execute_verified_local_handoff
from teamai.verification import VerificationResult


class LocalBridgeTest(unittest.TestCase):
    def test_local_bridge_implements_agent_bridge_contract(self) -> None:
        from teamai.integrations.local_bridge import LocalMLXBridge

        self.assertTrue(issubclass(LocalMLXBridge, AgentBridge))

    def test_execute_local_handoff_uses_mlx_backend_and_writes_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            payload_path = project_root / ".teamai" / "codex_payload.json"
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_text(
                json.dumps(
                    {
                        "original_task": "Implement a local patch.",
                        "core_dependencies": ["teamai/cli.py"],
                        "distilled_context": {"teamai/cli.py": "CLI summary."},
                        "recommended_codex_action": "Inspect teamai/cli.py before implementing the change.",
                    }
                ),
                encoding="utf-8",
            )

            fake_response = type(
                "_Response",
                (),
                {"text": "diff --git a/demo.txt b/demo.txt\n--- a/demo.txt\n+++ b/demo.txt\n@@ -0,0 +1 @@\n+patched\n"},
            )()

            with patch("teamai.integrations.local_bridge.Settings.from_env") as from_env, patch(
                "teamai.integrations.local_bridge.MLXModelBackend"
            ) as backend_cls:
                from_env.return_value = type(
                    "_Settings",
                    (),
                    {
                        "model_id": "mlx-community/gemma-4-2b-it-4bit",
                        "model_revision": None,
                        "force_download": False,
                        "trust_remote_code": False,
                        "enable_thinking": False,
                        "workspace_root": project_root,
                        "max_rounds": 2,
                        "max_actions_per_round": 2,
                        "max_tokens_per_turn": 256,
                        "temperature": 0.0,
                        "allow_shell": False,
                        "allow_writes": False,
                        "command_timeout_seconds": 5,
                        "max_file_bytes": 4096,
                        "max_command_output_chars": 4096,
                        "host": "127.0.0.1",
                        "port": 8000,
                        "model_router": False,
                    },
                )()
                backend_cls.return_value.generate_messages.return_value = fake_response

                result = execute_local_handoff(project_root=project_root)

            self.assertEqual(result.engine, "local")
            self.assertTrue(result.patch_file.exists())
            self.assertIn("diff --git a/demo.txt b/demo.txt", result.patch_text)

    def test_execute_verified_local_handoff_uses_shared_verification_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            payload_path = project_root / ".teamai" / "codex_payload.json"
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_text(
                json.dumps(
                    {
                        "original_task": "Implement a local patch.",
                        "core_dependencies": ["teamai/cli.py"],
                        "distilled_context": {"teamai/cli.py": "CLI summary."},
                        "recommended_codex_action": "Inspect teamai/cli.py before implementing the change.",
                    }
                ),
                encoding="utf-8",
            )

            fake_response = type(
                "_Response",
                (),
                {"text": "diff --git a/demo.txt b/demo.txt\n--- a/demo.txt\n+++ b/demo.txt\n@@ -0,0 +1 @@\n+patched\n"},
            )()
            verification = VerificationResult(success=True, log_output="ok", patch_returncode=0, test_returncode=0)

            with patch("teamai.integrations.local_bridge.Settings.from_env") as from_env, patch(
                "teamai.integrations.local_bridge.MLXModelBackend"
            ) as backend_cls, patch(
                "teamai.integrations.bridge_base.verify_patch",
                return_value=verification,
            ), patch(
                "teamai.integrations.bridge_base.PatchApprovalStore.create_bundle_from_patch",
                return_value={"approval_id": "approval123"},
            ), patch("teamai.integrations.bridge_base.Sandbox") as sandbox_cls:
                from_env.return_value = type(
                    "_Settings",
                    (),
                    {
                        "model_id": "mlx-community/gemma-4-2b-it-4bit",
                        "model_revision": None,
                        "force_download": False,
                        "trust_remote_code": False,
                        "enable_thinking": False,
                        "workspace_root": project_root,
                        "max_rounds": 2,
                        "max_actions_per_round": 2,
                        "max_tokens_per_turn": 256,
                        "temperature": 0.0,
                        "allow_shell": False,
                        "allow_writes": False,
                        "command_timeout_seconds": 5,
                        "max_file_bytes": 4096,
                        "max_command_output_chars": 4096,
                        "host": "127.0.0.1",
                        "port": 8000,
                        "model_router": False,
                    },
                )()
                backend_cls.return_value.generate_messages.return_value = fake_response
                sandbox_cls.return_value.__enter__.return_value = type("_Sandbox", (), {"path": project_root})()

                result = execute_verified_local_handoff(project_root=project_root)

            self.assertTrue(result.verification.success)
            self.assertEqual(result.execution.engine, "local")
            self.assertEqual(result.approval, {"approval_id": "approval123"})


if __name__ == "__main__":
    unittest.main()
