from __future__ import annotations

import re
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
    warnings: list[str] | None = None,
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
            warnings=warnings,
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
    warnings: list[str] | None = None,
) -> str:
    try:
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
        if compact:
            return compact
        if warnings is not None:
            warnings.append(
                f"Semantic distillation returned an empty summary for {file_path}; used heuristic fallback."
            )
    except Exception as exc:
        if warnings is not None:
            warnings.append(
                f"Semantic distillation fallback used for {file_path}: {exc}"
            )

    return _heuristic_file_summary(task=task, file_path=file_path, file_text=file_text)


def _fallback_recommended_action(*, task: str, core_dependencies: list[str]) -> str:
    if not core_dependencies:
        return f"Inspect the local reconnaissance output and implement the requested change for: {task}"
    if len(core_dependencies) == 1:
        return f"Inspect {core_dependencies[0]} and then implement the requested change for: {task}"
    return (
        f"Inspect {core_dependencies[0]} and {core_dependencies[1]} and then implement the "
        f"requested change for: {task}"
    )


def _heuristic_file_summary(*, task: str, file_path: str, file_text: str) -> str:
    path = Path(file_path)
    kind = _describe_file_kind(path)
    subjects = _extract_subjects(path=path, file_text=file_text)
    topics = _extract_topics(task=task, path=path, file_text=file_text)
    detail_sentences = _build_signal_sentences(path=path, file_text=file_text)
    risk = _build_risk_sentence(path=path, topics=topics, file_text=file_text)

    lead = f"{file_path} is {kind}"
    if subjects:
        lead += f" centered on {subjects}"
    else:
        preview = _extract_preview(file_text)
        if preview:
            lead += f" with key context around {preview}"
    lead = lead.rstrip(".") + "."

    if not detail_sentences:
        if topics:
            detail_sentences = [
                f"It touches {topics}, so changes here are likely part of the scoped implementation surface."
            ]
        elif _file_text_is_truncated(file_text):
            detail_sentences = [
                "The local context is truncated, so inspect this file directly before making broader refactors."
            ]
        else:
            detail_sentences = [
                "Changes here are likely relevant to the requested task and should be reviewed directly before implementation."
            ]

    if not risk and _file_text_is_truncated(file_text):
        risk = "Inspect the full file directly before relying on this compressed view for broader refactors."

    sentences = [lead, *detail_sentences]
    if risk:
        sentences.append(risk)
    return " ".join(sentence.strip() for sentence in sentences if sentence.strip())


def _describe_file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    name = path.name.lower()

    if name == "readme.md":
        return "the main repository guide"
    if name == "pyproject.toml":
        return "the project packaging and dependency manifest"
    if name.startswith("test_") and suffix == ".py":
        return "a Python test module"
    if suffix == ".py":
        return "a Python module"
    if suffix == ".md":
        return "a markdown document"
    if suffix == ".toml":
        return "a TOML configuration file"
    if suffix == ".json":
        return "a JSON artifact"
    if suffix in {".yml", ".yaml"}:
        return "a YAML configuration file"
    return "a repository file"


def _extract_subjects(*, path: Path, file_text: str) -> str:
    if path.suffix.lower() == ".py":
        names = re.findall(r"^(?:class|def)\s+([A-Za-z_][A-Za-z0-9_]*)", file_text, flags=re.MULTILINE)
        if names:
            return _join_phrases([f"`{name}`" for name in names[:3]])

    if path.suffix.lower() == ".md":
        headings = [line.lstrip("#").strip() for line in file_text.splitlines() if line.startswith("#")]
        if headings:
            return _join_phrases([f"`{heading}`" for heading in headings[:2]])

    if path.name.lower() == "pyproject.toml":
        keys = re.findall(r"^\[([^\]]+)\]", file_text, flags=re.MULTILINE)
        if keys:
            return _join_phrases([f"`[{key}]`" for key in keys[:3]])

    return ""


def _extract_topics(*, task: str, path: Path, file_text: str) -> str:
    lowered = f"{task}\n{path.as_posix()}\n{file_text}".lower()
    topic_map = {
        "streaming": ["stream", "event", "jsonl", "sse"],
        "runtime selection": ["runtime", "python executable", ".venv", "doctor"],
        "handoff flow": ["handoff", "codex", "semantic skeleton", "distill"],
        "bridge execution": ["bridge", "terminal", "osascript"],
        "write approvals": ["approval", "patch", "workspace_write", "replace_in_file", "write_file"],
        "workspace memory": ["memory", "run-history", "improvement note"],
        "API and jobs": ["fastapi", "/v1/", "job", "streamingresponse"],
        "tests": ["unittest", "assert", "test_"],
        "model runtime": ["mlx", "model", "gemma", "generate_messages"],
        "configuration": ["settings", "env", "teamai_", "pyproject", "dependency"],
    }
    topics = [label for label, markers in topic_map.items() if any(marker in lowered for marker in markers)]
    return _join_phrases(topics[:3])


def _build_signal_sentences(*, path: Path, file_text: str) -> list[str]:
    signal_map: dict[str, str] = {}

    cli_commands = _extract_cli_commands(file_text)
    if cli_commands:
        signal_map["cli"] = f"It wires CLI subcommands such as {_format_code_list(cli_commands[:4])}."

    api_routes = _extract_api_routes(file_text)
    if api_routes:
        route_sentence = f"It exposes HTTP routes like {_format_code_list(api_routes[:4])}"
        if _contains_streaming_markers(file_text):
            route_sentence += ", including streaming response paths"
        signal_map["api"] = route_sentence + "."

    manifest_sentence = _build_manifest_sentence(path=path, file_text=file_text)
    if manifest_sentence:
        signal_map["manifest"] = manifest_sentence

    env_vars = _extract_env_vars(file_text)
    if env_vars:
        signal_map["env"] = f"It reads env knobs like {_format_code_list(env_vars[:4])}."

    test_names = _extract_test_names(file_text)
    if test_names:
        prefix = "It asserts behaviors like" if path.name.lower().startswith("test_") else "It exercises behaviors like"
        signal_map["tests"] = f"{prefix} {_format_code_list(test_names[:4])}."

    headings = _extract_markdown_headings(file_text)
    if headings:
        signal_map["headings"] = f"It documents workflows under sections like {_format_code_list(headings[:3])}."

    if _file_text_is_truncated(file_text):
        signal_map["truncated"] = (
            "The local copy is truncated, so inspect the full file before changing broader control flow."
        )

    ordered = _signal_priority(path)
    return [signal_map[key] for key in ordered if key in signal_map][:2]


def _build_risk_sentence(*, path: Path, topics: str, file_text: str) -> str:
    name = path.name.lower()
    file_path = path.as_posix().lower()

    if name == "readme.md":
        return "Keep this aligned with the real commands, defaults, and safety behavior so the operator path stays trustworthy."
    if name == "pyproject.toml":
        return "Dependency or packaging changes here can affect installability, smoke runs, and runtime parity."
    if name == "cli.py":
        return "Changes here usually need matching updates to CLI tests, docs, and bridge or eval entrypoints."
    if name == "api.py":
        return "Changes here usually affect schemas, streaming semantics, and background-job event contracts."
    if name == "config.py":
        return "Changes here affect defaults, workspace safety, and runtime selection across CLI, bridge, and eval flows."
    if name == "bridge.py":
        return "Changes here usually affect launch artifacts, status reporting, and memory-profile retry behavior."
    if name == "handoff.py":
        return "Changes here affect task shaping, primary-task selection, and the paths handed to remote execution."
    if name == "distillation.py":
        return "Changes here affect Codex payload quality and fallback behavior when local MLX distillation is unavailable."
    if name == "evals.py":
        return "Changes here affect scoring, runtime-failure classification, and the meaning of the smoke suite."
    if name == "verification.py":
        return "Changes here affect how remote patches are validated and what operators see when verification fails."
    if name.startswith("test_") and path.suffix.lower() == ".py":
        return "Intentional behavior changes here usually need matching implementation updates or explicitly revised expectations."
    if file_path.endswith("/prompts.py"):
        return "Changes here can shift planner and verifier behavior, so keep structured-output assumptions and routing tests aligned."
    if topics:
        return f"Changes here likely need matching updates around {topics}."
    if _file_text_is_truncated(file_text):
        return "Inspect the full file directly before relying on this compressed view for broader refactors."
    return ""


def _extract_cli_commands(file_text: str) -> list[str]:
    return _dedupe_preserve_order(
        re.findall(r"add_parser\(\s*['\"]([^'\"]+)['\"]", file_text)
    )


def _extract_api_routes(file_text: str) -> list[str]:
    return _dedupe_preserve_order(
        re.findall(r"@[A-Za-z_][A-Za-z0-9_]*\.(?:get|post|put|delete|patch)\(\s*['\"]([^'\"]+)['\"]", file_text)
    )


def _extract_env_vars(file_text: str) -> list[str]:
    return _dedupe_preserve_order(
        re.findall(r"\b(?:TEAMAI|OPENAI|GEMINI|CODEX|MLX)_[A-Z0-9_]+\b", file_text)
    )


def _extract_dependencies(*, path: Path, file_text: str) -> list[str]:
    if path.name.lower() != "pyproject.toml":
        return []
    dependencies = _extract_toml_string_array(file_text, key="dependencies")
    normalized = [_normalize_dependency_name(item) for item in dependencies]
    return _dedupe_preserve_order([item for item in normalized if item])


def _extract_project_scripts(*, path: Path, file_text: str) -> list[str]:
    if path.name.lower() != "pyproject.toml":
        return []
    section = _extract_toml_table(file_text, "project.scripts")
    if not section:
        return []
    return _dedupe_preserve_order(
        re.findall(r"^([A-Za-z0-9_.-]+)\s*=", section, flags=re.MULTILINE)
    )


def _extract_test_names(file_text: str) -> list[str]:
    return _dedupe_preserve_order(
        re.findall(r"^def\s+(test_[A-Za-z0-9_]+)", file_text, flags=re.MULTILINE)
    )


def _extract_markdown_headings(file_text: str) -> list[str]:
    headings = [line.lstrip("#").strip() for line in file_text.splitlines() if line.startswith("#")]
    return _dedupe_preserve_order(headings)


def _build_manifest_sentence(*, path: Path, file_text: str) -> str:
    dependencies = _extract_dependencies(path=path, file_text=file_text)
    project_scripts = _extract_project_scripts(path=path, file_text=file_text)
    if not dependencies and not project_scripts:
        return ""

    fragments: list[str] = []
    if dependencies:
        fragments.append(f"runtime dependencies such as {_format_code_list(dependencies[:4])}")
    if project_scripts:
        fragments.append(f"entrypoints like {_format_code_list(project_scripts[:4])}")
    return f"It defines {_join_phrases(fragments)}."


def _signal_priority(path: Path) -> list[str]:
    name = path.name.lower()
    if name == "pyproject.toml":
        return ["manifest", "env", "headings", "tests", "truncated"]
    if name == "readme.md":
        return ["headings", "env", "manifest", "truncated"]
    if name.startswith("test_") and path.suffix.lower() == ".py":
        return ["tests", "cli", "api", "env", "truncated"]
    if name == "cli.py":
        return ["cli", "env", "tests", "truncated"]
    if name == "api.py":
        return ["api", "env", "tests", "truncated"]
    if name in {"config.py", "bridge.py", "runtime.py", "supervisor.py"}:
        return ["env", "cli", "api", "tests", "manifest", "truncated"]
    return ["cli", "api", "env", "manifest", "tests", "headings", "truncated"]


def _format_code_list(items: list[str]) -> str:
    return _join_phrases([f"`{item}`" for item in items])


def _contains_streaming_markers(file_text: str) -> bool:
    lowered = file_text.lower()
    markers = ["streamingresponse", "eventsource", "jsonl", "sse", "yield "]
    return any(marker in lowered for marker in markers)


def _extract_toml_table(file_text: str, table_name: str) -> str:
    pattern = re.compile(rf"^\[{re.escape(table_name)}\]\s*$", flags=re.MULTILINE)
    match = pattern.search(file_text)
    if match is None:
        return ""
    remainder = file_text[match.end() :]
    next_header = re.search(r"^\[.+\]\s*$", remainder, flags=re.MULTILINE)
    if next_header is None:
        return remainder
    return remainder[: next_header.start()]


def _extract_toml_string_array(file_text: str, *, key: str) -> list[str]:
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=\s*\[", flags=re.MULTILINE)
    match = pattern.search(file_text)
    if match is None:
        return []

    items: list[str] = []
    current: list[str] = []
    in_string = False
    escape = False

    for char in file_text[match.end() :]:
        if in_string:
            if escape:
                current.append(char)
                escape = False
                continue
            if char == "\\":
                escape = True
                continue
            if char == '"':
                items.append("".join(current))
                current = []
                in_string = False
                continue
            current.append(char)
            continue
        if char == '"':
            in_string = True
            continue
        if char == "]":
            return items
    return items


def _normalize_dependency_name(raw_dependency: str) -> str:
    match = re.match(r"[A-Za-z0-9_.-]+", raw_dependency.strip())
    return match.group(0) if match else ""


def _extract_preview(file_text: str, *, max_chars: int = 80) -> str:
    for line in file_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("[... file truncated"):
            continue
        compact = " ".join(stripped.split())
        if len(compact) <= max_chars:
            return f"`{compact}`"
        return f"`{compact[:max_chars].rstrip()}...`"
    return ""


def _join_phrases(items: list[str]) -> str:
    unique = _dedupe_preserve_order(items)
    if not unique:
        return ""
    if len(unique) == 1:
        return unique[0]
    if len(unique) == 2:
        return f"{unique[0]} and {unique[1]}"
    return f"{', '.join(unique[:-1])}, and {unique[-1]}"


def _file_text_is_truncated(file_text: str) -> bool:
    return "[... file truncated for local semantic distillation ...]" in file_text


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    unique: list[str] = []
    for item in items:
        if not item or item in unique:
            continue
        unique.append(item)
    return unique
