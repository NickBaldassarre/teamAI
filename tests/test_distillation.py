from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from teamai.distillation import generate_semantic_skeleton
from teamai.model_backend import ModelResponse
from teamai.schemas import CodexHandoffPayload


class FakeBackend:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses[:]
        self.calls: list[list[dict[str, str]]] = []

    def generate_messages(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        enable_thinking: bool | None = None,
    ) -> ModelResponse:
        del max_tokens, temperature, enable_thinking
        self.calls.append(messages)
        if not self._responses:
            raise AssertionError("No fake responses left for backend.")
        return ModelResponse(
            text=self._responses.pop(0),
            prompt_tokens=1,
            generation_tokens=1,
            total_tokens=2,
            prompt_tps=1.0,
            generation_tps=1.0,
            peak_memory_gb=0.0,
        )


class SemanticSkeletonSchemaTest(unittest.TestCase):
    def test_codex_handoff_payload_requires_recommended_action(self) -> None:
        with self.assertRaises(ValidationError):
            CodexHandoffPayload.model_validate(
                {
                    "original_task": "Inspect repo.",
                    "core_dependencies": ["README.md"],
                    "distilled_context": {"README.md": "Summary."},
                }
            )

    def test_generate_semantic_skeleton_distills_prioritized_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "README.md").write_text("# teamAI\nHigh-signal overview.\n", encoding="utf-8")
            (workspace / "teamai").mkdir()
            (workspace / "teamai" / "config.py").write_text(
                "DEFAULT_MODEL_ID = 'mlx-community/gemma-4-2b-it-4bit'\n",
                encoding="utf-8",
            )

            backend = FakeBackend(
                [
                    "README explains the local-first orchestration goal and the key runtime workflow.",
                    "config centralizes the Gemma scout model choice and safety-oriented runtime settings.",
                ]
            )

            payload = generate_semantic_skeleton(
                task="Improve the local scout handoff path.",
                workspace=workspace,
                prioritized_files=["teamai", "README.md", "teamai/config.py", "README.md"],
                backend=backend,  # type: ignore[arg-type]
                recommended_codex_action="Inspect README.md and teamai/config.py before changing the handoff flow.",
            )

            self.assertEqual(payload.original_task, "Improve the local scout handoff path.")
            self.assertEqual(payload.core_dependencies, ["README.md", "teamai/config.py"])
            self.assertEqual(
                payload.recommended_codex_action,
                "Inspect README.md and teamai/config.py before changing the handoff flow.",
            )
            self.assertIn("README.md", payload.distilled_context)
            self.assertIn("teamai/config.py", payload.distilled_context)
            self.assertEqual(len(backend.calls), 2)
            self.assertIn("File:\nREADME.md", backend.calls[0][1]["content"])
            self.assertIn("File:\nteamai/config.py", backend.calls[1][1]["content"])


if __name__ == "__main__":
    unittest.main()
