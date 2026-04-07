from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..codex_prompts import build_codex_handoff_prompt
from ..schemas import CodexHandoffPayload


DEFAULT_CODEX_MODEL = "gpt-5.4"
DEFAULT_CODEX_PAYLOAD_FILE = ".teamai/codex_payload.json"
DEFAULT_CODEX_PATCH_FILE = ".teamai/codex_solution.patch"


@dataclass(frozen=True)
class CodexHandoffExecutionResult:
    model: str
    payload_file: Path
    patch_file: Path
    prompt: str
    patch_text: str


def execute_codex_handoff(
    *,
    project_root: Path,
    payload_file: str | Path = DEFAULT_CODEX_PAYLOAD_FILE,
    patch_file: str | Path = DEFAULT_CODEX_PATCH_FILE,
    model: str | None = None,
) -> CodexHandoffExecutionResult:
    project_root = project_root.resolve()
    payload_path = _resolve_project_path(project_root, payload_file)
    patch_path = _resolve_project_path(project_root, patch_file)

    payload = CodexHandoffPayload.model_validate_json(payload_path.read_text(encoding="utf-8"))
    prompt = build_codex_handoff_prompt(payload)
    client = _create_openai_client()
    model_name = (model or os.getenv("TEAMAI_CODEX_MODEL", "").strip() or DEFAULT_CODEX_MODEL)
    response = client.responses.create(
        model=model_name,
        input=[
            {"role": "user", "content": prompt},
        ],
    )
raw_text = _extract_response_text(response)
    print(f"\n--- RAW CLOUD RESPONSE ---\n{raw_text}\n--------------------------\n")
    patch_text = _sanitize_patch_output(raw_text)
    patch_path.parent.mkdir(parents=True, exist_ok=True)
    patch_path.write_text(patch_text + ("\n" if not patch_text.endswith("\n") else ""), encoding="utf-8")

    return CodexHandoffExecutionResult(
        model=model_name,
        payload_file=payload_path,
        patch_file=patch_path,
        prompt=prompt,
        patch_text=patch_text,
    )


def _resolve_project_path(project_root: Path, raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else project_root / path


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


def _sanitize_patch_output(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()

    if not any(marker in stripped for marker in ("diff --git", "--- ", "+++ ")):
        raise RuntimeError("Codex response did not look like a unified diff patch.")
    return stripped
