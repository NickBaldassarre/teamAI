from __future__ import annotations

from pathlib import Path

from .model_backend import MLXModelBackend
from .schemas import CodexHandoffPayload


DEFAULT_SEMANTIC_SKELETON_MAX_FILES = 4
DEFAULT_SEMANTIC_SKELETON_MAX_TOKENS = 96
DEFAULT_SEMANTIC_SKELETON_MAX_FILE_CHARS = 6_000

DISTILLATION_SYSTEM_PROMPT = (
    "You compress local repository context for a stronger remote coding agent. "
    "Return 2-4 plain-text sentences. Focus on logic, constraints, change surface, "
    "and nearby risks that matter for the task. Omit boilerplate, imports, and "
    "formatting noise. Do not use markdown, bullets, or code fences."
)


def generate_semantic_skeleton(
    *,
    task: str,
    workspace: Path,
    prioritized_files: list[str],
    backend: MLXModelBackend,
    recommended_codex_action: str | None = None,
    max_files: int = DEFAULT_SEMANTIC_SKELETON_MAX_FILES,
    max_tokens: int = DEFAULT_SEMANTIC_SKELETON_MAX_TOKENS,
) -> CodexHandoffPayload:
    workspace = workspace.resolve()
    core_dependencies = _select_core_dependencies(
        workspace=workspace,
        prioritized_files=prioritized_files,
        max_files=max_files,
    )
    distilled_context: dict[str, str] = {}
    for relative_path in core_dependencies:
        file_text = _read_distillation_text(workspace / relative_path)
        distilled_context[relative_path] = _distill_file_summary(
            task=task,
            file_path=relative_path,
            file_text=file_text,
            backend=backend,
            max_tokens=max_tokens,
        )

    action = (recommended_codex_action or "").strip() or _fallback_recommended_action(
        task=task,
        core_dependencies=core_dependencies,
    )
    return CodexHandoffPayload(
        original_task=task,
        core_dependencies=core_dependencies,
        distilled_context=distilled_context,
        recommended_codex_action=action,
    )


def _select_core_dependencies(
    *,
    workspace: Path,
    prioritized_files: list[str],
    max_files: int,
) -> list[str]:
    dependencies: list[str] = []
    seen: set[str] = set()
    for candidate in prioritized_files:
        normalized = _normalize_candidate_path(candidate, workspace)
        if not normalized or normalized in seen:
            continue
        target = (workspace / normalized).resolve()
        if not target.exists() or not target.is_file():
            continue
        seen.add(normalized)
        dependencies.append(normalized)
        if len(dependencies) >= max_files:
            break
    return dependencies


def _normalize_candidate_path(candidate: str, workspace: Path) -> str | None:
    raw = candidate.strip()
    if not raw:
        return None

    path = Path(raw)
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(workspace))
        except ValueError:
            return None
    return str(path)


def _read_distillation_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if len(text) <= DEFAULT_SEMANTIC_SKELETON_MAX_FILE_CHARS:
        return text

    head = text[: DEFAULT_SEMANTIC_SKELETON_MAX_FILE_CHARS // 2]
    tail = text[-DEFAULT_SEMANTIC_SKELETON_MAX_FILE_CHARS // 2 :]
    return (
        f"{head}\n\n"
        "[... file truncated for local semantic distillation ...]\n\n"
        f"{tail}"
    )


def _distill_file_summary(
    *,
    task: str,
    file_path: str,
    file_text: str,
    backend: MLXModelBackend,
    max_tokens: int,
) -> str:
    response = backend.generate_messages(
        messages=[
            {"role": "system", "content": DISTILLATION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task:\n{task}\n\n"
                    f"File:\n{file_path}\n\n"
                    "Summarize only the logic and constraints from this file that matter "
                    "for completing the task.\n\n"
                    f"File contents:\n{file_text}"
                ),
            },
        ],
        max_tokens=max_tokens,
        temperature=0.0,
        enable_thinking=False,
    )
    compact = " ".join(response.text.split()).strip()
    return compact or "No distilled summary was produced for this file."


def _fallback_recommended_action(*, task: str, core_dependencies: list[str]) -> str:
    if not core_dependencies:
        return f"Inspect the local reconnaissance output and implement the requested change for: {task}"
    if len(core_dependencies) == 1:
        return f"Inspect {core_dependencies[0]} and then implement the requested change for: {task}"
    return (
        f"Inspect {core_dependencies[0]} and {core_dependencies[1]} and then implement the "
        f"requested change for: {task}"
    )
