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


class FailingBackend:
    def generate_messages(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        enable_thinking: bool | None = None,
    ) -> ModelResponse:
        del messages, max_tokens, temperature, enable_thinking
        raise RuntimeError("mlx unavailable")


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

    def test_generate_semantic_skeleton_falls_back_to_heuristic_summaries_when_backend_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "README.md").write_text("# teamAI\nHigh-signal overview.\n", encoding="utf-8")
            (workspace / "teamai").mkdir()
            (workspace / "teamai" / "cli.py").write_text(
                'def build_parser() -> None:\n    parser.add_parser("run")\n    parser.add_parser("execute-handoff")\n\n\ndef main() -> int:\n    return 0\n',
                encoding="utf-8",
            )
            warnings: list[str] = []

            payload = generate_semantic_skeleton(
                task="Improve the local scout handoff path.",
                workspace=workspace,
                prioritized_files=["README.md", "teamai/cli.py"],
                backend=FailingBackend(),  # type: ignore[arg-type]
                recommended_codex_action="Inspect README.md and teamai/cli.py before changing the handoff flow.",
                warnings=warnings,
            )

            self.assertEqual(payload.core_dependencies, ["README.md", "teamai/cli.py"])
            self.assertIn("README.md", payload.distilled_context)
            self.assertIn("teamai/cli.py", payload.distilled_context)
            self.assertIn("repository guide", payload.distilled_context["README.md"])
            self.assertIn("`build_parser` and `main`", payload.distilled_context["teamai/cli.py"])
            self.assertIn("CLI subcommands", payload.distilled_context["teamai/cli.py"])
            self.assertIn("`run` and `execute-handoff`", payload.distilled_context["teamai/cli.py"])
            self.assertTrue(any("fallback used for README.md" in warning for warning in warnings))

    def test_generate_semantic_skeleton_heuristics_surface_routes_dependencies_and_env_knobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            (workspace / "teamai").mkdir()
            (workspace / "pyproject.toml").write_text(
                '[project]\n'
                'name = "teamai"\n'
                'dependencies = [\n'
                '  "fastapi>=0.115,<1.0",\n'
                '  "uvicorn[standard]>=0.30,<1.0",\n'
                ']\n\n'
                '[project.scripts]\n'
                'teamai = "teamai.cli:main"\n',
                encoding="utf-8",
            )
            (workspace / "teamai" / "api.py").write_text(
                "from fastapi import FastAPI\n"
                "from fastapi.responses import StreamingResponse\n\n"
                "app = FastAPI()\n"
                'STREAM_MODE = "TEAMAI_STREAM_FORMAT"\n\n'
                '@app.post("/v1/runs")\n'
                "def create_run() -> StreamingResponse:\n"
                "    return StreamingResponse(iter([b'{}']))\n\n"
                '@app.get("/v1/jobs/{job_id}/events")\n'
                "def stream_job_events() -> StreamingResponse:\n"
                "    return StreamingResponse(iter([b'{}']))\n",
                encoding="utf-8",
            )

            payload = generate_semantic_skeleton(
                task="Improve operator-facing streaming updates.",
                workspace=workspace,
                prioritized_files=["pyproject.toml", "teamai/api.py"],
                backend=FailingBackend(),  # type: ignore[arg-type]
                warnings=[],
            )

            self.assertIn("runtime dependencies", payload.distilled_context["pyproject.toml"])
            self.assertIn("`fastapi`", payload.distilled_context["pyproject.toml"])
            self.assertIn("entrypoints like `teamai`", payload.distilled_context["pyproject.toml"])
            self.assertIn("HTTP routes", payload.distilled_context["teamai/api.py"])
            self.assertIn("`/v1/runs` and `/v1/jobs/{job_id}/events`", payload.distilled_context["teamai/api.py"])
            self.assertIn("streaming response paths", payload.distilled_context["teamai/api.py"])
            self.assertIn("env knobs", payload.distilled_context["teamai/api.py"])
            self.assertIn("`TEAMAI_STREAM_FORMAT`", payload.distilled_context["teamai/api.py"])


if __name__ == "__main__":
    unittest.main()
