from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_GEMINI_MODEL = "gemini-2.5-pro"


@dataclass(frozen=True)
class GeminiHandoffExecutionResult:
    model: str
    payload_file: Path
    patch_file: Path
    prompt: str
    patch_text: str


def execute_gemini_handoff(
    payload_file: str | Path | None = None,
    patch_file: str | Path | None = None,
    **kwargs,
) -> GeminiHandoffExecutionResult:
    """Read the local scout payload, request a patch from Gemini, and write it to disk."""
    project_root = Path(kwargs.get("project_root") or Path.cwd()).resolve()
    payload_path = _resolve_project_path(project_root, kwargs.get("payload_path") or payload_file or ".teamai/codex_payload.json")
    patch_path = _resolve_project_path(project_root, kwargs.get("patch_path") or patch_file or ".teamai/codex_solution.patch")
    model_name = kwargs.get("model") or kwargs.get("model_name") or DEFAULT_GEMINI_MODEL

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

    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Gemini handoff payload file was not found: {payload_path}") from exc

    prompt = (
        f"TASK: {payload.get('original_task')}\n\n"
        f"CORE DEPENDENCIES:\n{json.dumps(payload.get('core_dependencies', []), indent=2)}\n\n"
        f"DISTILLED CONTEXT:\n{json.dumps(payload.get('distilled_context', {}), indent=2)}\n"
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
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.0,
            ),
        )
    except Exception as exc:
        raise RuntimeError(f"Gemini API request failed: {exc}") from exc

    patch_text = _sanitize_patch_output(str(getattr(response, "text", "") or ""))
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(_ensure_trailing_newline(patch_text), encoding="utf-8")
    return GeminiHandoffExecutionResult(
        model=model_name,
        payload_file=payload_path,
        patch_file=patch_path,
        prompt=prompt,
        patch_text=patch_text,
    )


def _sanitize_patch_output(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()

    if not any(marker in stripped for marker in ("diff --git", "--- ", "+++ ")):
        raise RuntimeError("Gemini response did not look like a unified diff patch.")
    return stripped


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else f"{text}\n"


def _resolve_project_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else project_root / path
