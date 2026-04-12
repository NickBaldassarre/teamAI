from __future__ import annotations

import json
import os
from pathlib import Path

from ..schemas import CodexHandoffPayload
from .bridge_base import AgentBridge, BridgeExecutionResult

DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"

GeminiHandoffExecutionResult = BridgeExecutionResult


class GeminiBridge(AgentBridge):
    engine = "gemini"
    default_model = DEFAULT_GEMINI_MODEL

    def _request_patch(
        self,
        *,
        prompt: str,
        model: str,
        payload: CodexHandoffPayload,
        payload_path: Path,
        project_root: Path,
    ) -> str:
        api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Export it before running `teamai execute-handoff --engine gemini`.")

        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise RuntimeError(
                "The Google GenAI SDK is not installed. Install it before using `teamai execute-handoff --engine gemini`."
            ) from exc

        prompt_payload = (
            f"TASK: {payload.original_task}\n\n"
            f"CORE DEPENDENCIES:\n{json.dumps(payload.core_dependencies, indent=2)}\n\n"
            f"DISTILLED CONTEXT:\n{json.dumps(payload.distilled_context, indent=2)}\n"
        )
        system_instruction = (
            "You are the execution engine for a multi-agent orchestrator.\n"
            "Your constraints:\n"
            "- Read the task and context.\n"
            "- Return ONLY a strict git unified diff patch.\n"
            "- Every file change MUST begin with standard git headers (e.g., `--- a/filepath` and `+++ b/filepath`).\n"
            "- Do not wrap the patch in markdown or conversational filler.\n"
        )

        client = genai.Client(api_key=api_key)
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt_payload,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0,
                ),
            )
        except Exception as exc:
            raise RuntimeError(f"Gemini API request failed: {exc}") from exc
        return str(getattr(response, "text", "") or "")


def execute_gemini_handoff(
    payload_file: str | Path | None = None,
    patch_file: str | Path | None = None,
    **kwargs,
) -> GeminiHandoffExecutionResult:
    project_root = Path(kwargs.get("project_root") or Path.cwd()).resolve()
    return GeminiBridge().execute(
        project_root=project_root,
        payload_file=kwargs.get("payload_path") or payload_file or ".teamai/codex_payload.json",
        patch_file=kwargs.get("patch_path") or patch_file or ".teamai/codex_solution.patch",
        model=kwargs.get("model") or kwargs.get("model_name") or DEFAULT_GEMINI_MODEL,
    )
