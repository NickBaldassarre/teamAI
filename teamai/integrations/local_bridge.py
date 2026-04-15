from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from ..config import Settings
from ..model_backend import MLXModelBackend
from ..schemas import CodexHandoffPayload
from .bridge_base import AgentBridge, BridgeExecutionResult, BridgeModelResponse, VerifiedBridgeExecutionResult

DEFAULT_LOCAL_BRIDGE_PAYLOAD_FILE = ".teamai/codex_payload.json"
DEFAULT_LOCAL_BRIDGE_PATCH_FILE = ".teamai/local_solution.patch"
DEFAULT_LOCAL_BRIDGE_FAILURE_CONTEXT_FILE = ".teamai/failure_context.log"
LOCAL_BRIDGE_SYSTEM_PROMPT = (
    "You are the local patch execution engine for a supervised coding orchestrator.\n"
    "Read the task context and return ONLY a strict unified diff patch.\n"
    "Do not add markdown fences, explanations, or prose outside the patch.\n"
    "Every changed file must include standard diff headers."
)

LocalHandoffExecutionResult = BridgeExecutionResult
VerifiedLocalHandoffExecutionResult = VerifiedBridgeExecutionResult


class LocalMLXBridge(AgentBridge):
    engine = "local"
    default_model = ""

    def _request_patch(
        self,
        *,
        prompt: str,
        model: str,
        payload: CodexHandoffPayload,
        payload_path: Path,
        project_root: Path,
    ) -> BridgeModelResponse:
        settings = Settings.from_env()
        backend_settings = settings if settings.model_id == model else replace(settings, model_id=model)
        backend = MLXModelBackend(backend_settings)
        response = backend.generate_messages(
            messages=[
                {"role": "system", "content": LOCAL_BRIDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max(512, backend_settings.max_tokens_per_turn),
            temperature=0.0,
            enable_thinking=False,
        )
        return BridgeModelResponse(
            patch_text=response.text,
            prompt_tokens=getattr(response, "prompt_tokens", None),
            completion_tokens=getattr(response, "generation_tokens", None),
            total_tokens=getattr(response, "total_tokens", None),
        )


def execute_local_handoff(
    *,
    project_root: Path,
    payload_file: str | Path = DEFAULT_LOCAL_BRIDGE_PAYLOAD_FILE,
    patch_file: str | Path = DEFAULT_LOCAL_BRIDGE_PATCH_FILE,
    model: str | None = None,
) -> LocalHandoffExecutionResult:
    settings = Settings.from_env()
    return LocalMLXBridge().execute(
        project_root=project_root,
        payload_file=payload_file,
        patch_file=patch_file,
        model=model or settings.model_id,
    )


def execute_verified_local_handoff(
    *,
    project_root: Path,
    payload_file: str | Path = DEFAULT_LOCAL_BRIDGE_PAYLOAD_FILE,
    patch_file: str | Path = DEFAULT_LOCAL_BRIDGE_PATCH_FILE,
    failure_context_file: str | Path = DEFAULT_LOCAL_BRIDGE_FAILURE_CONTEXT_FILE,
    model: str | None = None,
) -> VerifiedLocalHandoffExecutionResult:
    settings = Settings.from_env()
    return LocalMLXBridge().execute_verified(
        project_root=project_root,
        payload_file=payload_file,
        patch_file=patch_file,
        failure_context_file=failure_context_file,
        model=model or settings.model_id,
    )
