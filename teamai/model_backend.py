from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from .config import Settings


class ModelBackendError(RuntimeError):
    """Raised when the MLX backend cannot generate a response."""


@dataclass(frozen=True)
class ModelResponse:
    text: str
    prompt_tokens: int
    generation_tokens: int
    total_tokens: int
    prompt_tps: float
    generation_tps: float
    peak_memory_gb: float


class MLXModelBackend:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._model: Any | None = None
        self._processor: Any | None = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None and self._processor is not None

    def generate_messages(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        enable_thinking: bool | None = None,
    ) -> ModelResponse:
        with self._lock:
            try:
                self._ensure_loaded()
                from mlx_vlm import generate as mlx_generate
                from mlx_vlm.prompt_utils import apply_chat_template
            except Exception as exc:
                raise ModelBackendError(
                    "Failed to import MLX runtime. This usually means MLX is not installed "
                    "correctly or Metal initialization failed on this machine."
                ) from exc

            try:
                prompt = apply_chat_template(
                    self._processor,
                    self._model.config,
                    messages,
                    add_generation_prompt=True,
                    enable_thinking=self._settings.enable_thinking
                    if enable_thinking is None
                    else enable_thinking,
                )
                result = mlx_generate(
                    self._model,
                    self._processor,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    verbose=False,
                )
            except Exception as exc:
                raise ModelBackendError(f"MLX generation failed: {exc}") from exc

            return ModelResponse(
                text=result.text.strip(),
                prompt_tokens=result.prompt_tokens,
                generation_tokens=result.generation_tokens,
                total_tokens=result.total_tokens,
                prompt_tps=result.prompt_tps,
                generation_tps=result.generation_tps,
                peak_memory_gb=result.peak_memory,
            )

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return

        try:
            from mlx_vlm import load as mlx_load
        except Exception as exc:
            raise ModelBackendError(
                "Could not import mlx_vlm. Install the project dependencies first."
            ) from exc

        load_kwargs: dict[str, Any] = {
            "force_download": self._settings.force_download,
            "trust_remote_code": self._settings.trust_remote_code,
        }
        if self._settings.model_revision:
            load_kwargs["revision"] = self._settings.model_revision

        try:
            self._model, self._processor = mlx_load(
                self._settings.model_id,
                lazy=False,
                **load_kwargs,
            )
        except Exception as exc:
            raise ModelBackendError(
                f"Failed to load model `{self._settings.model_id}` with MLX."
            ) from exc

