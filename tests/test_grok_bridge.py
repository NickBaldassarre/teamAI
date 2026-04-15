from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from teamai.integrations.grok_bridge import execute_grok_handoff


class _FakeResponses:
    def __init__(self, output_text: str) -> None:
        self._output_text = output_text
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)

        class _Usage:
            input_tokens = 120
            output_tokens = 30
            total_tokens = 150

        class _Response:
            def __init__(self, output_text: str) -> None:
                self.output_text = output_text
                self.usage = _Usage()
                self.headers = {
                    "x-ratelimit-limit-requests": "100",
                    "x-ratelimit-remaining-requests": "55",
                    "x-ratelimit-limit-tokens": "1000",
                    "x-ratelimit-remaining-tokens": "700",
                }

        return _Response(self._output_text)


class _FakeXAIClient:
    def __init__(self, output_text: str) -> None:
        self.responses = _FakeResponses(output_text)


class GrokBridgeTest(unittest.TestCase):
    def test_execute_grok_handoff_requires_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            payload_path = project_root / ".teamai" / "codex_payload.json"
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_text(
                json.dumps(
                    {
                        "original_task": "Improve routing.",
                        "core_dependencies": ["teamai/supervisor.py"],
                        "distilled_context": {"teamai/supervisor.py": "Supervisor summary."},
                        "recommended_codex_action": "Inspect teamai/supervisor.py before implementing the change.",
                    }
                ),
                encoding="utf-8",
            )

            with patch.dict("os.environ", {}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "XAI_API_KEY is not set"):
                    execute_grok_handoff(project_root=project_root, payload_file=payload_path)

    def test_execute_grok_handoff_writes_patch_and_updates_rate_limit_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            project_root = Path(temp_dir)
            payload_path = project_root / ".teamai" / "codex_payload.json"
            patch_path = project_root / ".teamai" / "grok_solution.patch"
            payload_path.parent.mkdir(parents=True, exist_ok=True)
            payload_path.write_text(
                json.dumps(
                    {
                        "original_task": "Improve routing.",
                        "core_dependencies": ["teamai/supervisor.py"],
                        "distilled_context": {"teamai/supervisor.py": "Supervisor summary."},
                        "recommended_codex_action": "Inspect teamai/supervisor.py before implementing the change.",
                    }
                ),
                encoding="utf-8",
            )
            client = _FakeXAIClient(
                "diff --git a/teamai/supervisor.py b/teamai/supervisor.py\n"
                "--- a/teamai/supervisor.py\n"
                "+++ b/teamai/supervisor.py\n"
                "@@ -1,1 +1,2 @@\n"
                " from __future__ import annotations\n"
                "+# patched\n"
            )

            with patch("teamai.integrations.grok_bridge._create_xai_client", return_value=client), patch.dict(
                "os.environ",
                {"HOME": temp_dir, "XAI_API_KEY": "test-key"},
                clear=False,
            ):
                result = execute_grok_handoff(project_root=project_root)

            self.assertEqual(result.engine, "grok")
            self.assertEqual(result.model, "grok-4-1-fast-reasoning")
            self.assertTrue(patch_path.exists())
            self.assertEqual(result.prompt_tokens, 120)
            self.assertEqual(result.completion_tokens, 30)
            self.assertEqual(result.total_tokens, 150)
            self.assertIsNotNone(result.rate_limit_state)

            rate_limit_path = Path(temp_dir) / ".teamai" / "rate_limits.json"
            self.assertTrue(rate_limit_path.exists())
            stored = json.loads(rate_limit_path.read_text(encoding="utf-8"))
            self.assertIn("grok-4-1-fast-reasoning", stored)


if __name__ == "__main__":
    unittest.main()
