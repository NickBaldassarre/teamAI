from __future__ import annotations

from .bridge_base import (
    AgentBridge,
    BridgeExecutionResult,
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
    if normalized == "local":
        return LocalMLXBridge()
    raise ValueError(f"Unknown handoff engine: {engine}")


__all__ = [
    "AgentBridge",
    "BridgeExecutionResult",
    "CodexBridge",
    "CodexHandoffExecutionResult",
    "DEFAULT_CODEX_FAILURE_CONTEXT_FILE",
    "DEFAULT_CODEX_MODEL",
    "DEFAULT_CODEX_PATCH_FILE",
    "DEFAULT_CODEX_PAYLOAD_FILE",
    "DEFAULT_GEMINI_MODEL",
    "DEFAULT_LOCAL_BRIDGE_FAILURE_CONTEXT_FILE",
    "DEFAULT_LOCAL_BRIDGE_PATCH_FILE",
    "DEFAULT_LOCAL_BRIDGE_PAYLOAD_FILE",
    "GeminiBridge",
    "GeminiHandoffExecutionResult",
    "LocalMLXBridge",
    "VerifiedBridgeExecutionResult",
    "VerifiedCodexHandoffExecutionResult",
    "execute_codex_handoff",
    "execute_gemini_handoff",
    "execute_local_handoff",
    "execute_verified_codex_handoff",
    "execute_verified_local_handoff",
    "get_bridge",
]
