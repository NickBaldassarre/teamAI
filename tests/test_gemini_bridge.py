from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from teamai.integrations.gemini_bridge import execute_gemini_handoff


class GeminiBridgeTest(unittest.TestCase):
    def test_execute_gemini_handoff_requires_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            payload_path = project_root / ".teamai" / "codex_payload.json"
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_text("{}", encoding="utf-8")

            with patch.dict("os.environ", {}, clear=False):
                with self.assertRaisesRegex(RuntimeError, "GEMINI_API_KEY is not set"):
                    execute_gemini_handoff(
                        project_root=project_root,
                        payload_file=payload_path,
                    )

    def test_execute_gemini_handoff_resolves_relative_paths_and_writes_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            payload_path = project_root / ".teamai" / "codex_payload.json"
            patch_path = project_root / ".teamai" / "codex_solution.patch"
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_text(
                json.dumps(
                    {
                        "original_task": "Improve streaming output.",
                        "core_dependencies": ["teamai/cli.py"],
                        "distilled_context": {"teamai/cli.py": "CLI summary."},
                    }
                ),
                encoding="utf-8",
            )

            captured: dict[str, object] = {}

            class _FakeClient:
                def __init__(self, *, api_key: str) -> None:
                    captured["api_key"] = api_key
                    self.models = self

                def generate_content(self, **kwargs):  # type: ignore[no-untyped-def]
                    captured["request"] = kwargs
                    return SimpleNamespace(
                        text=(
                            "```diff\n"
                            "diff --git a/teamai/cli.py b/teamai/cli.py\n"
                            "--- a/teamai/cli.py\n"
                            "+++ b/teamai/cli.py\n"
                            "@@ -1,1 +1,2 @@\n"
                            "+# patched\n"
                            "```\n"
                        )
                    )

            class _GenerateContentConfig:
                def __init__(self, **kwargs) -> None:
                    self.kwargs = kwargs

            fake_google = types.ModuleType("google")
            fake_genai = types.ModuleType("google.genai")
            fake_types = types.ModuleType("google.genai.types")
            fake_genai.Client = _FakeClient
            fake_types.GenerateContentConfig = _GenerateContentConfig
            fake_genai.types = fake_types
            fake_google.genai = fake_genai

            with patch.dict(
                sys.modules,
                {
                    "google": fake_google,
                    "google.genai": fake_genai,
                    "google.genai.types": fake_types,
                },
            ), patch.dict("os.environ", {"GEMINI_API_KEY": "test-key"}, clear=False):
                returned_path = execute_gemini_handoff(
                    project_root=project_root,
                    payload_file=".teamai/codex_payload.json",
                    patch_file=".teamai/codex_solution.patch",
                )

            self.assertEqual(returned_path.model, "gemini-2.5-pro")
            self.assertEqual(returned_path.payload_file.resolve(), payload_path.resolve())
            self.assertEqual(returned_path.patch_file.resolve(), patch_path.resolve())
            self.assertTrue(patch_path.exists())
            self.assertIn("diff --git a/teamai/cli.py b/teamai/cli.py", patch_path.read_text(encoding="utf-8"))
            self.assertIn("TASK: Improve streaming output.", returned_path.prompt)
            self.assertIn("diff --git a/teamai/cli.py b/teamai/cli.py", returned_path.patch_text)
            self.assertEqual(captured["api_key"], "test-key")
            request = captured["request"]
            self.assertIsInstance(request, dict)
            self.assertEqual(request["model"], "gemini-2.5-pro")


if __name__ == "__main__":
    unittest.main()
