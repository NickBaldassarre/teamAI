from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..schemas import CodexHandoffPayload
from .bridge_base import AgentBridge, BridgeExecutionResult, VerifiedBridgeExecutionResult

DEFAULT_CODEX_MODEL = "gpt-5.4"
DEFAULT_CODEX_PAYLOAD_FILE = ".teamai/codex_payload.json"
DEFAULT_CODEX_PATCH_FILE = ".teamai/codex_solution.patch"
DEFAULT_CODEX_FAILURE_CONTEXT_FILE = ".teamai/failure_context.log"

CodexHandoffExecutionResult = BridgeExecutionResult
VerifiedCodexHandoffExecutionResult = VerifiedBridgeExecutionResult


class CodexBridge(AgentBridge):
    engine = "codex"
    default_model = DEFAULT_CODEX_MODEL

    def _request_patch(
        self,
        *,
        prompt: str,
        model: str,
        payload: CodexHandoffPayload,
        payload_path: Path,
        project_root: Path,
    ) -> str:
        client = _create_openai_client()
        response = client.responses.create(
            model=model,
            input=[
                {"role": "user", "content": prompt},
            ],
        )
        return _extract_response_text(response)


def execute_codex_handoff(
    *,
    project_root: Path,
    payload_file: str | Path = DEFAULT_CODEX_PAYLOAD_FILE,
    patch_file: str | Path = DEFAULT_CODEX_PATCH_FILE,
    model: str | None = None,
) -> CodexHandoffExecutionResult:
    return CodexBridge().execute(
        project_root=project_root,
        payload_file=payload_file,
        patch_file=patch_file,
        model=model or os.getenv("TEAMAI_CODEX_MODEL", "").strip() or DEFAULT_CODEX_MODEL,
    )


def execute_verified_codex_handoff(
    *,
    project_root: Path,
    payload_file: str | Path = DEFAULT_CODEX_PAYLOAD_FILE,
    patch_file: str | Path = DEFAULT_CODEX_PATCH_FILE,
    failure_context_file: str | Path = DEFAULT_CODEX_FAILURE_CONTEXT_FILE,
    model: str | None = None,
) -> VerifiedCodexHandoffExecutionResult:
    return CodexBridge().execute_verified(
        project_root=project_root,
        payload_file=payload_file,
        patch_file=patch_file,
        failure_context_file=failure_context_file,
        model=model or os.getenv("TEAMAI_CODEX_MODEL", "").strip() or DEFAULT_CODEX_MODEL,
    )


def _create_openai_client() -> Any:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Export it before running `teamai execute-handoff`.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "The OpenAI SDK is not installed. Install the project dependencies again so `openai` is available."
        ) from exc

    base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    if base_url:
        return OpenAI(api_key=api_key, base_url=base_url)
    return OpenAI(api_key=api_key)


def _extract_response_text(response: Any) -> str:
    direct_text = str(getattr(response, "output_text", "") or "").strip()
    if direct_text:
        return direct_text

    output = getattr(response, "output", None)
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            content = getattr(item, "content", None)
            if not isinstance(content, list):
                continue
            for entry in content:
                text = str(getattr(entry, "text", "") or "").strip()
                if text:
                    chunks.append(text)
        if chunks:
            return "\n".join(chunks)

    raise RuntimeError("Codex response did not contain any text output.")
