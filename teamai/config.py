from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_MODEL_ID = "mlx-community/gemma-4-2b-it-4bit"
DEFAULT_MAX_ROUNDS = 4
DEFAULT_MAX_ACTIONS_PER_ROUND = 3
DEFAULT_MAX_TOKENS_PER_TURN = 320
DEFAULT_TEMPERATURE = 0.3
DEFAULT_COMMAND_TIMEOUT_SECONDS = 30
DEFAULT_MAX_FILE_BYTES = 50_000
DEFAULT_MAX_COMMAND_OUTPUT_CHARS = 12_000
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


class ConfigError(ValueError):
    """Raised when the local configuration is invalid."""


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, str(default)).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"`{name}` must be an integer. Received: {raw!r}") from exc
    if value < minimum:
        raise ConfigError(f"`{name}` must be at least {minimum}. Received: {value}.")
    return value


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name, str(default)).strip()
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"`{name}` must be a float. Received: {raw!r}") from exc
    if not (minimum <= value <= maximum):
        raise ConfigError(
            f"`{name}` must be between {minimum} and {maximum}. Received: {value}."
        )
    return value


@dataclass(frozen=True)
class Settings:
    model_id: str
    model_revision: str | None
    force_download: bool
    trust_remote_code: bool
    enable_thinking: bool
    workspace_root: Path
    max_rounds: int
    max_actions_per_round: int
    max_tokens_per_turn: int
    temperature: float
    allow_shell: bool
    allow_writes: bool
    command_timeout_seconds: int
    max_file_bytes: int
    max_command_output_chars: int
    host: str
    port: int
    model_router: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(override=False)

        workspace_root = Path(
            os.getenv("TEAMAI_WORKSPACE_ROOT", ".").strip() or "."
        ).expanduser()

        return cls(
            model_id=os.getenv("TEAMAI_MODEL_ID", DEFAULT_MODEL_ID).strip()
            or DEFAULT_MODEL_ID,
            model_revision=os.getenv("TEAMAI_MODEL_REVISION", "").strip() or None,
            force_download=_env_bool("TEAMAI_FORCE_DOWNLOAD", False),
            trust_remote_code=_env_bool("TEAMAI_TRUST_REMOTE_CODE", False),
            enable_thinking=_env_bool("TEAMAI_ENABLE_THINKING", False),
            workspace_root=workspace_root.resolve(),
            max_rounds=_env_int("TEAMAI_MAX_ROUNDS", DEFAULT_MAX_ROUNDS, minimum=1),
            max_actions_per_round=_env_int(
                "TEAMAI_MAX_ACTIONS_PER_ROUND",
                DEFAULT_MAX_ACTIONS_PER_ROUND,
                minimum=1,
            ),
            max_tokens_per_turn=_env_int(
                "TEAMAI_MAX_TOKENS_PER_TURN",
                DEFAULT_MAX_TOKENS_PER_TURN,
                minimum=32,
            ),
            temperature=_env_float("TEAMAI_TEMPERATURE", DEFAULT_TEMPERATURE, 0.0, 2.0),
            allow_shell=_env_bool("TEAMAI_ALLOW_SHELL", True),
            allow_writes=_env_bool("TEAMAI_ALLOW_WRITES", False),
            command_timeout_seconds=_env_int(
                "TEAMAI_COMMAND_TIMEOUT_SECONDS",
                DEFAULT_COMMAND_TIMEOUT_SECONDS,
                minimum=1,
            ),
            max_file_bytes=_env_int(
                "TEAMAI_MAX_FILE_BYTES",
                DEFAULT_MAX_FILE_BYTES,
                minimum=1024,
            ),
            max_command_output_chars=_env_int(
                "TEAMAI_MAX_COMMAND_OUTPUT_CHARS",
                DEFAULT_MAX_COMMAND_OUTPUT_CHARS,
                minimum=1000,
            ),
            host=os.getenv("TEAMAI_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST,
            port=_env_int("TEAMAI_PORT", DEFAULT_PORT, minimum=1),
            model_router=_env_bool("TEAMAI_MODEL_ROUTER", False),
        )

    def resolve_workspace(self, requested_path: str | None) -> Path:
        root = self.workspace_root.resolve()
        candidate = root
        if requested_path:
            incoming = Path(requested_path).expanduser()
            candidate = incoming if incoming.is_absolute() else root / incoming

        resolved = candidate.resolve()
        if not resolved.exists():
            raise ConfigError(f"Workspace path does not exist: {resolved}")
        if not resolved.is_dir():
            raise ConfigError(f"Workspace path is not a directory: {resolved}")
        if root not in {resolved, *resolved.parents}:
            raise ConfigError(
                f"Workspace path {resolved} must stay inside workspace root {root}."
            )
        return resolved
