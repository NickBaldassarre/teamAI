from __future__ import annotations

from .bridge_base import (
    AgentBridge,
    BridgeExecutionResult,
    BridgeModelResponse,
    VerifiedBridgeExecutionResult,
)
from .codex_bridge import (
    DEFAULT_CODEX_FAILURE_CONTEXT_FILE,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_PATCH_FILE,
    DEFAULT_CODEX_PAYLOAD_FILE,
    CodexBridge,
    CodexHandoffExecutionResult,
    VerifiedCodexHandoffExecutionResult,
    execute_codex_handoff,
    execute_verified_codex_handoff,
)
from .gemini_bridge import (
    DEFAULT_GEMINI_MODEL,
    GeminiBridge,
    GeminiHandoffExecutionResult,
    execute_gemini_handoff,
)
from .grok_bridge import (
    DEFAULT_GROK_FAILURE_CONTEXT_FILE,
    DEFAULT_GROK_MODEL,
    DEFAULT_GROK_PATCH_FILE,
    DEFAULT_GROK_PAYLOAD_FILE,
    GrokBridge,
    GrokHandoffExecutionResult,
    VerifiedGrokHandoffExecutionResult,
    execute_grok_handoff,
    execute_verified_grok_handoff,
)
from .local_bridge import (
    DEFAULT_LOCAL_BRIDGE_FAILURE_CONTEXT_FILE,
    DEFAULT_LOCAL_BRIDGE_PATCH_FILE,
    DEFAULT_LOCAL_BRIDGE_PAYLOAD_FILE,
    LocalMLXBridge,
    execute_local_handoff,
    execute_verified_local_handoff,
)


def get_bridge(engine: str) -> AgentBridge:
    normalized = engine.strip().lower()
    if normalized == "codex":
        return CodexBridge()
    if normalized == "gemini":
        return GeminiBridge()
    if normalized == "grok":
        return GrokBridge()
    if normalized == "local":
        return LocalMLXBridge()
    raise ValueError(f"Unknown handoff engine: {engine}")


__all__ = [
    "AgentBridge",
    "BridgeExecutionResult",
    "BridgeModelResponse",
    "CodexBridge",
    "CodexHandoffExecutionResult",
    "DEFAULT_CODEX_FAILURE_CONTEXT_FILE",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_CODEX_PATCH_FILE",
    "DEFAULT_CODEX_PAYLOAD_FILE",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_GROK_FAILURE_CONTEXT_FILE",
    "DEFAULT_GROK_MODEL",
    "DEFAULT_GROK_PATCH_FILE",
    "DEFAULT_GROK_PAYLOAD_FILE",
    "DEFAULT_LOCAL_BRIDGE_FAILURE_CONTEXT_FILE",
    "DEFAULT_LOCAL_BRIDGE_PATCH_FILE",
    "DEFAULT_LOCAL_BRIDGE_PAYLOAD_FILE",
    "GeminiBridge",
    "GeminiHandoffExecutionResult",
    "GrokBridge",
    "GrokHandoffExecutionResult",
    "LocalMLXBridge",
    "VerifiedBridgeExecutionResult",
    "VerifiedCodexHandoffExecutionResult",
    "VerifiedGrokHandoffExecutionResult",
    "execute_codex_handoff",
    "execute_gemini_handoff",
    "execute_grok_handoff",
    "execute_local_handoff",
    "execute_verified_codex_handoff",
    "execute_verified_grok_handoff",
    "execute_verified_local_handoff",
    "get_bridge",
]
