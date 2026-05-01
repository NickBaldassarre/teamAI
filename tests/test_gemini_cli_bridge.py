from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from teamai.integrations import get_bridge
from teamai.integrations.bridge_base import AgentBridge
from teamai.integrations.gemini_cli_bridge import (
    DEFAULT_GEMINI_CLI_MODEL,
    GeminiCLIBridge,
    _parse_gemini_cli_output,
    execute_gemini_cli_handoff,
    execute_verified_gemini_cli_handoff,
)
from teamai.verification import VerificationResult


PATCH_TEXT = (
    "diff --git a/demo.txt b/demo.txt\n"
    "--- a/demo.txt\n"
    "+++ b/demo.txt\n"
    "@@ -0,0 +1 @@\n"
    "+patched\n"
)


def _write_payload(project_root: Path) -> Path:
    payload_path = project_root / ".teamai" / "codex_payload.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(
        json.dumps(
            {
                "original_task": "Improve streaming output.",
                "core_dependencies": ["teamai/cli.py"],
                "distilled_context": {"teamai/cli.py": "CLI summary."},
                "recommended_codex_action": "Inspect teamai/cli.py before implementing the change.",
            }
        ),
        encoding="utf-8",
    )
    return payload_path


class GeminiCLIBridgeTest(unittest.TestCase):
    def test_bridge_implements_agent_bridge_contract(self) -> None:
        self.assertTrue(issubclass(GeminiCLIBridge, AgentBridge))
        self.assertEqual(GeminiCLIBridge.engine, "gemini-cli")
        self.assertEqual(GeminiCLIBridge.default_model, DEFAULT_GEMINI_CLI_MODEL)

    def test_get_bridge_returns_gemini_cli_for_both_aliases(self) -> None:
        for alias in ("gemini-cli", "gemini_cli", "GEMINI-CLI"):
            self.assertIsInstance(get_bridge(alias), GeminiCLIBridge)

    def test_execute_gemini_cli_handoff_invokes_subprocess_and_writes_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            payload_path = _write_payload(project_root)

            captured: dict[str, object] = {}

            def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                captured["cmd"] = cmd
                captured["kwargs"] = kwargs
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "response": PATCH_TEXT,
                            "usage": {
                                "prompt_tokens": 12,
                                "completion_tokens": 5,
                                "total_tokens": 17,
                            },
                        }
                    ),
                    stderr="",
                )

            with patch(
                "teamai.integrations.gemini_cli_bridge.subprocess.run",
                side_effect=fake_run,
            ), patch(
                "teamai.integrations.gemini_cli_bridge.shutil.which",
                return_value="/usr/local/bin/gemini",
            ):
                result = execute_gemini_cli_handoff(
                    project_root=project_root,
                    payload_file=payload_path,
                )

            self.assertEqual(result.engine, "gemini-cli")
            self.assertEqual(result.model, DEFAULT_GEMINI_CLI_MODEL)
            self.assertTrue(result.patch_file.exists())
            self.assertIn("diff --git a/demo.txt b/demo.txt", result.patch_text)
            self.assertEqual(result.prompt_tokens, 12)
            self.assertEqual(result.completion_tokens, 5)
            self.assertEqual(result.total_tokens, 17)

            cmd = captured["cmd"]
            self.assertIsInstance(cmd, list)
            self.assertEqual(cmd[0], "/usr/local/bin/gemini")
            self.assertIn("--prompt", cmd)
            self.assertIn("--model", cmd)
            self.assertEqual(cmd[cmd.index("--model") + 1], DEFAULT_GEMINI_CLI_MODEL)
            self.assertIn("--output-format", cmd)
            self.assertEqual(cmd[cmd.index("--output-format") + 1], "json")
            kwargs = captured["kwargs"]
            self.assertIsInstance(kwargs, dict)
            self.assertEqual(kwargs["cwd"], str(project_root.resolve()))

    def test_execute_gemini_cli_handoff_handles_plain_text_stdout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_payload(project_root)

            def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                return subprocess.CompletedProcess(
                    args=cmd, returncode=0, stdout=PATCH_TEXT, stderr=""
                )

            with patch(
                "teamai.integrations.gemini_cli_bridge.subprocess.run",
                side_effect=fake_run,
            ), patch(
                "teamai.integrations.gemini_cli_bridge.shutil.which",
                return_value="/usr/local/bin/gemini",
            ):
                result = execute_gemini_cli_handoff(project_root=project_root)

            self.assertIn("diff --git a/demo.txt b/demo.txt", result.patch_text)
            self.assertIsNone(result.prompt_tokens)
            self.assertIsNone(result.completion_tokens)

    def test_execute_gemini_cli_handoff_raises_when_subprocess_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_payload(project_root)

            def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=2,
                    stdout="",
                    stderr="failed to load local Gemma\n",
                )

            with patch(
                "teamai.integrations.gemini_cli_bridge.subprocess.run",
                side_effect=fake_run,
            ), patch(
                "teamai.integrations.gemini_cli_bridge.shutil.which",
                return_value="/usr/local/bin/gemini",
            ):
                with self.assertRaisesRegex(RuntimeError, "Gemini CLI exited with code 2"):
                    execute_gemini_cli_handoff(project_root=project_root)

    def test_execute_gemini_cli_handoff_raises_when_executable_missing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_payload(project_root)

            def fake_run(*args, **kwargs):  # type: ignore[no-untyped-def]
                raise FileNotFoundError("gemini")

            with patch(
                "teamai.integrations.gemini_cli_bridge.subprocess.run",
                side_effect=fake_run,
            ), patch(
                "teamai.integrations.gemini_cli_bridge.shutil.which",
                return_value=None,
            ), patch.dict("os.environ", {}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "was not found on PATH"):
                    execute_gemini_cli_handoff(project_root=project_root)

    def test_execute_verified_gemini_cli_handoff_uses_shared_verification_flow(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_payload(project_root)

            def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout=json.dumps({"response": PATCH_TEXT}),
                    stderr="",
                )

            verification = VerificationResult(
                success=True, log_output="ok", patch_returncode=0, test_returncode=0
            )

            with patch(
                "teamai.integrations.gemini_cli_bridge.subprocess.run",
                side_effect=fake_run,
            ), patch(
                "teamai.integrations.gemini_cli_bridge.shutil.which",
                return_value="/usr/local/bin/gemini",
            ), patch(
                "teamai.integrations.bridge_base.verify_patch",
                return_value=verification,
            ), patch(
                "teamai.integrations.bridge_base.PatchApprovalStore.create_bundle_from_patch",
                return_value={"approval_id": "approval-cli-1"},
            ), patch("teamai.integrations.bridge_base.Sandbox") as sandbox_cls:
                sandbox_cls.return_value.__enter__.return_value = type(
                    "_Sandbox", (), {"path": project_root}
                )()
                result = execute_verified_gemini_cli_handoff(project_root=project_root)

            self.assertTrue(result.verification.success)
            self.assertEqual(result.execution.engine, "gemini-cli")
            self.assertEqual(result.approval, {"approval_id": "approval-cli-1"})

    def test_model_override_flows_through_command(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            _write_payload(project_root)

            captured: dict[str, object] = {}

            def fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
                captured["cmd"] = cmd
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=0,
                    stdout=json.dumps({"response": PATCH_TEXT}),
                    stderr="",
                )

            with patch(
                "teamai.integrations.gemini_cli_bridge.subprocess.run",
                side_effect=fake_run,
            ), patch(
                "teamai.integrations.gemini_cli_bridge.shutil.which",
                return_value="/usr/local/bin/gemini",
            ):
                execute_gemini_cli_handoff(
                    project_root=project_root, model="gemma-3-12b-it"
                )

            cmd = captured["cmd"]
            self.assertEqual(cmd[cmd.index("--model") + 1], "gemma-3-12b-it")


class ParseGeminiCLIOutputTest(unittest.TestCase):
    def test_parses_top_level_response_field(self) -> None:
        text, prompt_tokens, completion_tokens, total_tokens = _parse_gemini_cli_output(
            json.dumps(
                {
                    "response": "hello",
                    "usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                    },
                }
            )
        )
        self.assertEqual(text, "hello")
        self.assertEqual((prompt_tokens, completion_tokens, total_tokens), (3, 1, 4))

    def test_parses_streamed_jsonl_last_event(self) -> None:
        stream = "\n".join(
            [
                json.dumps({"event": "start"}),
                json.dumps({"event": "token", "text": "ignored"}),
                json.dumps({"response": "final answer", "usage": {"total_tokens": 9}}),
            ]
        )
        text, prompt_tokens, completion_tokens, total_tokens = _parse_gemini_cli_output(stream)
        self.assertEqual(text, "final answer")
        self.assertEqual(total_tokens, 9)

    def test_falls_back_to_raw_text_when_not_json(self) -> None:
        text, prompt_tokens, completion_tokens, total_tokens = _parse_gemini_cli_output(
            "diff --git a/x b/x\n"
        )
        self.assertIn("diff --git a/x b/x", text)
        self.assertIsNone(prompt_tokens)
        self.assertIsNone(completion_tokens)
        self.assertIsNone(total_tokens)


if __name__ == "__main__":
    unittest.main()
