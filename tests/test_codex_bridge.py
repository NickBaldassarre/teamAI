from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from teamai.codex_prompts import build_codex_handoff_prompt
from teamai.integrations.codex_bridge import execute_codex_handoff, execute_verified_codex_handoff
from teamai.schemas import CodexHandoffPayload
from teamai.verification import VerificationResult


class _FakeResponses:
    def __init__(self, output_text: str) -> None:
        self._output_text = output_text
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)

        class _Response:
            def __init__(self, output_text: str) -> None:
                self.output_text = output_text

        return _Response(self._output_text)


class _FakeOpenAIClient:
    def __init__(self, output_text: str) -> None:
        self.responses = _FakeResponses(output_text)


class CodexPromptTest(unittest.TestCase):
    def test_build_codex_handoff_prompt_embeds_constraints_and_distilled_context(self) -> None:
        payload = CodexHandoffPayload(
            original_task="Improve streaming event output across the CLI and API.",
            core_dependencies=["teamai/cli.py", "teamai/api.py"],
            distilled_context={
                "teamai/cli.py": "CLI entrypoint summary.",
                "teamai/api.py": "API entrypoint summary.",
            },
            recommended_codex_action="Inspect teamai/cli.py and teamai/api.py before implementing the change.",
        )

        prompt = build_codex_handoff_prompt(payload)

        self.assertIn("You are the Lead Architect.", prompt)
        self.assertIn("Do not ask for more file context.", prompt)
        self.assertIn("unified diff format (.patch)", prompt)
        self.assertIn("Original task:\nImprove streaming event output across the CLI and API.", prompt)
        self.assertIn("[teamai/cli.py]\nCLI entrypoint summary.", prompt)
        self.assertIn("Recommended Codex action:\nInspect teamai/cli.py and teamai/api.py", prompt)


class CodexBridgeExecutionTest(unittest.TestCase):
    def test_execute_codex_handoff_reads_payload_formats_prompt_and_writes_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            payload_path = project_root / ".teamai" / "codex_payload.json"
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "original_task": "Improve streaming event output across the CLI and API.",
                "core_dependencies": ["teamai/cli.py", "teamai/api.py"],
                "distilled_context": {
                    "teamai/cli.py": "CLI entrypoint summary.",
                    "teamai/api.py": "API entrypoint summary.",
                },
                "recommended_codex_action": "Inspect teamai/cli.py and teamai/api.py before implementing the change.",
            }
            payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

            client = _FakeOpenAIClient(
                "diff --git a/teamai/cli.py b/teamai/cli.py\n"
                "--- a/teamai/cli.py\n"
                "+++ b/teamai/cli.py\n"
                "@@ -1,1 +1,2 @@\n"
                " from __future__ import annotations\n"
                "+# patched\n"
            )

            with patch("teamai.integrations.codex_bridge._create_openai_client", return_value=client):
                result = execute_codex_handoff(project_root=project_root)

            self.assertEqual(result.model, "gpt-5.4")
            self.assertEqual(result.payload_file, payload_path.resolve())
            self.assertTrue(result.patch_file.exists())
            self.assertIn("diff --git a/teamai/cli.py b/teamai/cli.py", result.patch_text)
            self.assertIn("Do not ask for more file context.", result.prompt)
            self.assertEqual(len(client.responses.calls), 1)
            request = client.responses.calls[0]
            self.assertEqual(request["model"], "gpt-5.4")
            input_messages = request["input"]
            self.assertIsInstance(input_messages, list)
            self.assertIn("Improve streaming event output across the CLI and API.", input_messages[0]["content"])

    def test_execute_verified_codex_handoff_clears_stale_failure_context_after_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            payload_path = project_root / ".teamai" / "codex_payload.json"
            failure_context_path = project_root / ".teamai" / "failure_context.log"
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_text(
                json.dumps(
                    {
                        "original_task": "Inspect repo.",
                        "core_dependencies": ["teamai/cli.py"],
                        "distilled_context": {"teamai/cli.py": "CLI entrypoint summary."},
                        "recommended_codex_action": "Inspect teamai/cli.py before implementing the change.",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            failure_context_path.write_text("stale failure log\n", encoding="utf-8")
            client = _FakeOpenAIClient(
                "diff --git a/teamai/cli.py b/teamai/cli.py\n"
                "--- a/teamai/cli.py\n"
                "+++ b/teamai/cli.py\n"
                "@@ -1,1 +1,2 @@\n"
                " from __future__ import annotations\n"
                "+# patched\n"
            )
            verification = VerificationResult(
                success=True,
                log_output="all green",
                patch_returncode=0,
                test_returncode=0,
            )

            with patch("teamai.integrations.codex_bridge._create_openai_client", return_value=client), patch(
                "teamai.integrations.codex_bridge.verify_patch",
                return_value=verification,
            ), patch("teamai.integrations.codex_bridge.Sandbox") as sandbox_cls:
                sandbox_cls.return_value.__enter__.return_value = object()
                result = execute_verified_codex_handoff(project_root=project_root)

            self.assertTrue(result.verification.success)
            self.assertFalse(failure_context_path.exists())
            sandbox_cls.assert_called_once_with(project_root.resolve())

    def test_execute_verified_codex_handoff_persists_failure_context_after_failed_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            payload_path = project_root / ".teamai" / "codex_payload.json"
            failure_context_path = project_root / ".teamai" / "failure_context.log"
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_text(
                json.dumps(
                    {
                        "original_task": "Inspect repo.",
                        "core_dependencies": ["teamai/cli.py"],
                        "distilled_context": {"teamai/cli.py": "CLI entrypoint summary."},
                        "recommended_codex_action": "Inspect teamai/cli.py before implementing the change.",
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            client = _FakeOpenAIClient(
                "diff --git a/teamai/cli.py b/teamai/cli.py\n"
                "--- a/teamai/cli.py\n"
                "+++ b/teamai/cli.py\n"
                "@@ -1,1 +1,2 @@\n"
                " from __future__ import annotations\n"
                "+# patched\n"
            )
            verification = VerificationResult(
                success=False,
                log_output="tests failed",
                patch_returncode=0,
                test_returncode=1,
            )

            with patch("teamai.integrations.codex_bridge._create_openai_client", return_value=client), patch(
                "teamai.integrations.codex_bridge.verify_patch",
                return_value=verification,
            ), patch("teamai.integrations.codex_bridge.Sandbox") as sandbox_cls:
                sandbox_cls.return_value.__enter__.return_value = object()
                result = execute_verified_codex_handoff(project_root=project_root)

            self.assertFalse(result.verification.success)
            self.assertEqual(failure_context_path.read_text(encoding="utf-8"), "tests failed\n")
            sandbox_cls.assert_called_once_with(project_root.resolve())


if __name__ == "__main__":
    unittest.main()
