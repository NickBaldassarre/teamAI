from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from time import perf_counter

from ..schemas import CodexHandoffPayload
from .bridge_base import (
    AgentBridge,
    BridgeExecutionResult,
    BridgeModelResponse,
    VerifiedBridgeExecutionResult,
    build_rate_limit_state,
)

DEFAULT_GEMINI_CLI_MODEL = "gemma-3-4b-it"
DEFAULT_GEMINI_CLI_BIN = "gemini"
DEFAULT_GEMINI_CLI_TIMEOUT_SECONDS = 600
DEFAULT_GEMINI_CLI_PAYLOAD_FILE = ".teamai/codex_payload.json"
DEFAULT_GEMINI_CLI_PATCH_FILE = ".teamai/gemini_cli_solution.patch"
DEFAULT_GEMINI_CLI_FAILURE_CONTEXT_FILE = ".teamai/failure_context.log"

GEMINI_CLI_SYSTEM_PROMPT = (
    "You are the local execution engine for a supervised multi-agent orchestrator running "
    "through the Gemini CLI against a locally-hosted Gemma model.\n"
    "Your constraints:\n"
    "- Read the task and context carefully.\n"
    "- Return ONLY a strict git unified diff patch.\n"
    "- Every changed file MUST begin with standard git headers (e.g., `--- a/path` and `+++ b/path`).\n"
    "- Do not wrap the patch in markdown fences, prose, or commentary outside the diff.\n"
)

GeminiCLIHandoffExecutionResult = BridgeExecutionResult
VerifiedGeminiCLIHandoffExecutionResult = VerifiedBridgeExecutionResult


class GeminiCLIBridge(AgentBridge):
    engine = "gemini-cli"
    default_model = DEFAULT_GEMINI_CLI_MODEL

    def _request_patch(
        self,
        *,
        prompt: str,
        model: str,
        payload: CodexHandoffPayload,
        payload_path: Path,
        project_root: Path,
    ) -> BridgeModelResponse:
        executable = _resolve_gemini_cli_executable()
        timeout_seconds = _resolve_timeout_seconds()
        full_prompt = f"{GEMINI_CLI_SYSTEM_PROMPT}\n\n{prompt}"

        cmd: list[str] = [
            executable,
            "--prompt",
            full_prompt,
            "--model",
            model,
            "--output-format",
            "json",
            "--yolo",
        ]

        env = _build_subprocess_env()
        started = perf_counter()
        try:
            completed = subprocess.run(  # noqa: S603 - explicit list, no shell
                cmd,
                cwd=str(project_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"The Gemini CLI executable `{executable}` was not found on PATH. "
                "Install it via `npm install -g @google/gemini-cli` (or set TEAMAI_GEMINI_CLI_BIN) "
                "and run `gemini gemma setup` to enable the local Gemma model."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Gemini CLI call timed out after {timeout_seconds}s while invoking model `{model}`."
            ) from exc
        latency_ms = (perf_counter() - started) * 1000.0

        if completed.returncode != 0:
            stderr_excerpt = (completed.stderr or "").strip().splitlines()[-20:]
            raise RuntimeError(
                "Gemini CLI exited with code "
                f"{completed.returncode}: {os.linesep.join(stderr_excerpt) or '<no stderr>'}"
            )

        text, prompt_tokens, completion_tokens, total_tokens = _parse_gemini_cli_output(
            completed.stdout or ""
        )
        if not text.strip():
            raise RuntimeError("Gemini CLI returned an empty response.")
        return BridgeModelResponse(
            patch_text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            rate_limit_state=build_rate_limit_state(
                provider="gemini-cli",
                model_id=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                headers={},
                source="usage_only",
            ),
            metadata={
                "executable": executable,
                "stderr_tail": "\n".join(
                    (completed.stderr or "").strip().splitlines()[-5:]
                ),
            },
        )


def execute_gemini_cli_handoff(
    *,
    project_root: Path,
    payload_file: str | Path = DEFAULT_GEMINI_CLI_PAYLOAD_FILE,
    patch_file: str | Path = DEFAULT_GEMINI_CLI_PATCH_FILE,
    model: str | None = None,
) -> GeminiCLIHandoffExecutionResult:
    return GeminiCLIBridge().execute(
        project_root=project_root,
        payload_file=payload_file,
        patch_file=patch_file,
        model=model or _resolve_default_model(),
    )


def execute_verified_gemini_cli_handoff(
    *,
    project_root: Path,
    payload_file: str | Path = DEFAULT_GEMINI_CLI_PAYLOAD_FILE,
    patch_file: str | Path = DEFAULT_GEMINI_CLI_PATCH_FILE,
    failure_context_file: str | Path = DEFAULT_GEMINI_CLI_FAILURE_CONTEXT_FILE,
    model: str | None = None,
) -> VerifiedGeminiCLIHandoffExecutionResult:
    return GeminiCLIBridge().execute_verified(
        project_root=project_root,
        payload_file=payload_file,
        patch_file=patch_file,
        failure_context_file=failure_context_file,
        model=model or _resolve_default_model(),
    )


def _resolve_default_model() -> str:
    override = os.getenv("TEAMAI_GEMINI_CLI_MODEL", "").strip()
    if override:
        return override
    env_model = os.getenv("GEMINI_MODEL", "").strip()
    if env_model:
        return env_model
    return DEFAULT_GEMINI_CLI_MODEL


def _resolve_gemini_cli_executable() -> str:
    override = os.getenv("TEAMAI_GEMINI_CLI_BIN", "").strip()
    if override:
        return override
    discovered = shutil.which(DEFAULT_GEMINI_CLI_BIN)
    if discovered:
        return discovered
    return DEFAULT_GEMINI_CLI_BIN


def _resolve_timeout_seconds() -> int:
    raw = os.getenv("TEAMAI_GEMINI_CLI_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return DEFAULT_GEMINI_CLI_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_GEMINI_CLI_TIMEOUT_SECONDS
    return max(1, value)


def _build_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("GEMINI_NO_TELEMETRY", "1")
    env.setdefault("GEMINI_NONINTERACTIVE", "1")
    return env


def _parse_gemini_cli_output(
    raw: str,
) -> tuple[str, int | None, int | None, int | None]:
    stripped = raw.strip()
    if not stripped:
        return "", None, None, None

    parsed = _try_parse_json(stripped)
    if parsed is None and "\n" in stripped:
        for candidate in reversed(stripped.splitlines()):
            parsed = _try_parse_json(candidate.strip())
            if parsed is not None:
                break

    if isinstance(parsed, dict):
        text = _extract_text_from_json(parsed)
        prompt_tokens, completion_tokens, total_tokens = _extract_usage_from_json(parsed)
        if text:
            return text, prompt_tokens, completion_tokens, total_tokens

    return stripped, None, None, None


def _try_parse_json(text: str) -> object | None:
    if not text or text[0] not in "{[":
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_text_from_json(payload: dict) -> str:
    for key in ("response", "text", "output_text", "content"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value

    output = payload.get("output")
    if isinstance(output, list):
        chunks: list[str] = []
        for item in output:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    chunks.append(content)
                elif isinstance(content, list):
                    for entry in content:
                        if isinstance(entry, dict):
                            text_part = entry.get("text") or entry.get("content")
                            if isinstance(text_part, str) and text_part.strip():
                                chunks.append(text_part)
            elif isinstance(item, str) and item.strip():
                chunks.append(item)
        if chunks:
            return "\n".join(chunks)
    return ""


def _extract_usage_from_json(payload: dict) -> tuple[int | None, int | None, int | None]:
    usage = payload.get("usage") or payload.get("usage_metadata") or {}
    if not isinstance(usage, dict):
        return None, None, None
    prompt_tokens = _coerce_int(
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or usage.get("prompt_token_count")
    )
    completion_tokens = _coerce_int(
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or usage.get("candidates_token_count")
    )
    total_tokens = _coerce_int(
        usage.get("total_tokens") or usage.get("total_token_count")
    )
    if total_tokens is None and (prompt_tokens is not None or completion_tokens is not None):
        total_tokens = (prompt_tokens or 0) + (completion_tokens or 0)
    return prompt_tokens, completion_tokens, total_tokens


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None
