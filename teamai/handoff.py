from __future__ import annotations

from pathlib import Path

from .schemas import HandoffPacket, RunResult, ToolExecutionResult


def build_handoff_packet(*, task: str, result: RunResult) -> HandoffPacket:
    summary, next_tasks = _extract_summary_and_tasks(result.final_answer)
    key_paths = _collect_key_paths(task=task, result=result)
    primary_task = next_tasks[0] if next_tasks else None
    evidence = _collect_evidence(result)
    open_questions = _collect_open_questions(
        result,
        summary=summary,
        next_tasks=next_tasks,
        evidence=evidence,
    )
    suggested_prompt = _build_suggested_codex_prompt(
        task=task,
        task_route=result.task_route,
        primary_task=primary_task,
        key_paths=key_paths,
        evidence=evidence,
        open_questions=open_questions,
    )
    return HandoffPacket(
        goal=task,
        status=result.status,
        workspace=result.workspace,
        model_id=result.model_id,
        task_route=result.task_route,
        stop_reason=result.stop_reason,
        summary=summary,
        primary_task=primary_task,
        next_tasks=next_tasks,
        key_paths=key_paths,
        evidence=evidence,
        open_questions=open_questions,
        warnings=result.warnings,
        suggested_codex_prompt=suggested_prompt,
    )


def render_handoff_markdown(packet: HandoffPacket) -> str:
    lines = [
        f"# Local Model Handoff",
        "",
        f"Goal: {packet.goal}",
        f"Status: {packet.status} ({packet.stop_reason})",
        f"Route: {packet.task_route}",
        f"Workspace: {packet.workspace}",
        f"Model: {packet.model_id}",
        "",
        "## Summary",
        packet.summary,
    ]
    if packet.primary_task:
        lines.extend(["", "## Primary Task", packet.primary_task])
    if packet.next_tasks:
        lines.extend(["", "## Next Tasks"])
        lines.extend(f"- {task}" for task in packet.next_tasks)
    if packet.key_paths:
        lines.extend(["", "## Key Paths"])
        lines.extend(f"- {path}" for path in packet.key_paths)
    if packet.evidence:
        lines.extend(["", "## Evidence"])
        lines.extend(f"- {item}" for item in packet.evidence)
    if packet.open_questions:
        lines.extend(["", "## Open Questions"])
        lines.extend(f"- {item}" for item in packet.open_questions)
    if packet.warnings:
        lines.extend(["", "## Warnings"])
        lines.extend(f"- {warning}" for warning in packet.warnings)
    lines.extend(["", "## Suggested Codex Prompt", packet.suggested_codex_prompt])
    return "\n".join(lines)


def _extract_summary_and_tasks(final_answer: str) -> tuple[str, list[str]]:
    marker = "Next engineering tasks:"
    if marker not in final_answer:
        summary = final_answer.strip() or "No summary available."
        return summary, []

    summary_part, task_part = final_answer.split(marker, maxsplit=1)
    summary = summary_part.strip() or "No summary available."
    tasks: list[str] = []
    for line in task_part.splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            tasks.append(stripped[2:].strip())
    return summary, tasks


def _collect_key_paths(*, task: str, result: RunResult) -> list[str]:
    workspace = Path(result.workspace)
    ordered: list[str] = []
    seen: set[str] = set()
    for record in result.rounds:
        for tool_result in record.tool_results:
            if not tool_result.success:
                continue
            raw_path = str(tool_result.metadata.get("path", "")).strip()
            if not raw_path:
                continue
            normalized = _normalize_path(raw_path, workspace)
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
    return _rank_key_paths_for_task(task=task, paths=ordered)[:12]


def _rank_key_paths_for_task(*, task: str, paths: list[str]) -> list[str]:
    task_lower = task.lower()
    unique_paths: list[str] = []
    seen: set[str] = set()
    for path in paths:
        normalized = path.rstrip("/")
        if normalized in seen:
            continue
        seen.add(normalized)
        unique_paths.append(normalized)

    file_paths = [path for path in unique_paths if path not in {".", "teamai", "tests"} and "." in Path(path).name]
    preferred_pool = file_paths or [path for path in unique_paths if path != "."]

    return sorted(
        preferred_pool,
        key=lambda path: (-_score_key_path(task_lower=task_lower, path=path), unique_paths.index(path)),
    )


def _score_key_path(*, task_lower: str, path: str) -> int:
    path_lower = path.lower()
    score = 0
    if any(marker in task_lower for marker in ["bridge", "handoff", "terminal"]) and any(
        marker in path_lower for marker in ["bridge.py", "handoff.py", "test_bridge.py", "test_handoff.py"]
    ):
        score += 6
    if any(marker in task_lower for marker in ["memory", "history", "persist", "cross-run"]) and any(
        marker in path_lower for marker in ["memory.py", "test_memory.py", "prompts.py"]
    ):
        score += 6
    if any(
        marker in task_lower
        for marker in [
            "learned-note",
            "learned note",
            "improvement note",
            "self-improvement",
            "self improvement",
            "decay",
            "prune",
            "pruning",
            "stale lesson",
            "stale note",
        ]
    ):
        if path_lower.endswith("teamai/memory.py"):
            score += 10
        elif path_lower.endswith("tests/test_memory.py"):
            score += 8
        elif any(marker in path_lower for marker in ["prompts.py", "handoff.py", "bridge.py", "supervisor.py"]):
            score += 4
    if any(marker in task_lower for marker in ["stream", "streaming", "event output", "progress output"]) and any(
        marker in path_lower for marker in ["cli.py", "api.py", "jobs.py", "schemas.py", "supervisor.py"]
    ):
        score += 6
    if any(marker in task_lower for marker in ["approval", "patch", "write path", "workspace_write"]) and any(
        marker in path_lower for marker in ["tools.py", "approvals.py", "supervisor.py", "test_tools.py", "test_approvals.py"]
    ):
        score += 6
    if any(marker in task_lower for marker in ["json", "planner", "verifier", "prompt", "structured output"]) and any(
        marker in path_lower for marker in ["prompts.py", "schemas.py", "supervisor.py", "test_supervisor.py"]
    ):
        score += 5
    return score


def _collect_evidence(result: RunResult) -> list[str]:
    evidence: list[str] = []
    seen: set[str] = set()
    workspace = Path(result.workspace)
    for record in result.rounds:
        for tool_result in record.tool_results:
            if not tool_result.success:
                continue
            item = _summarize_tool_result(tool_result, workspace=workspace)
            if not item or item in seen:
                continue
            seen.add(item)
            evidence.append(item)
            if len(evidence) >= 8:
                return evidence
        verifier_summary = record.verifier.summary.strip()
        if verifier_summary:
            item = f"Verifier summary: {verifier_summary}"
            if item not in seen:
                seen.add(item)
                evidence.append(item)
                if len(evidence) >= 8:
                    return evidence
    return evidence


def _collect_open_questions(
    result: RunResult,
    *,
    summary: str,
    next_tasks: list[str],
    evidence: list[str],
) -> list[str]:
    candidates: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    implemented_themes = _implemented_themes(summary=summary, evidence=evidence)
    pending_themes = _pending_themes(next_tasks)
    order = 0
    for record in result.rounds:
        next_focus = (record.verifier.next_focus or "").strip()
        if next_focus and next_focus not in seen:
            seen.add(next_focus)
            candidates.append(
                (
                    next_focus,
                    _score_open_question(
                        next_focus,
                        pending_themes=pending_themes,
                        implemented_themes=implemented_themes,
                    ),
                    order,
                )
            )
            order += 1
        for tool_result in record.tool_results:
            if tool_result.success or not tool_result.error:
                continue
            question = f"Resolve tool failure: {tool_result.error}"
            if question not in seen:
                seen.add(question)
                candidates.append(
                    (
                        question,
                        _score_open_question(
                            question,
                            pending_themes=pending_themes,
                            implemented_themes=implemented_themes,
                        ),
                        order,
                    )
                )
                order += 1
        if len(candidates) >= 6:
            break

    prioritized = sorted(candidates, key=lambda item: (-item[1], item[2]))
    filtered = [question for question, score, _ in prioritized if score >= 0]
    if filtered:
        return filtered[:6]
    return [question for question, _, _ in prioritized[:2]]


def _implemented_themes(*, summary: str, evidence: list[str]) -> set[str]:
    themes: set[str] = set()
    lowered_summary = summary.lower()
    if "already implemented" in lowered_summary:
        themes.update(_feature_tags_for_text(lowered_summary))
    for item in evidence:
        lowered = item.lower()
        if "already implemented" in lowered:
            themes.update(_feature_tags_for_text(lowered))
    return themes


def _pending_themes(next_tasks: list[str]) -> set[str]:
    themes: set[str] = set()
    for task in next_tasks:
        themes.update(_feature_tags_for_text(task))
    return themes


def _score_open_question(
    question: str,
    *,
    pending_themes: set[str],
    implemented_themes: set[str],
) -> int:
    lowered = question.lower()
    themes = _feature_tags_for_text(lowered)
    score = 0
    if question.startswith("Resolve tool failure:"):
        score += 2
    if themes & pending_themes:
        score += 3
    if themes & implemented_themes and not (themes & pending_themes):
        score -= 4
    if "_extract_summary_and_tasks" in lowered or "next_tasks" in lowered:
        score += 1
    if "highest-value next change" in lowered or "next engineering task" in lowered:
        score += 1
    return score


def _feature_tags_for_text(text: str) -> set[str]:
    lowered = text.lower()
    tags: set[str] = set()
    if any(
        marker in lowered
        for marker in [
            "persistent memory",
            "run history",
            "workspace memory",
            "cross-run",
            "memory.md",
            "persist_run",
            "load_snapshot",
            "teamai/memory.py",
        ]
    ):
        tags.add("persistent_memory")
    if any(
        marker in lowered
        for marker in [
            "patch-oriented",
            "approval checkpoint",
            "approval checkpoints",
            "destructive changes",
            "coarse write path",
            "write path",
        ]
    ):
        tags.add("patch_writes")
    if "streaming event" in lowered or "streaming output" in lowered or "streaming events" in lowered:
        tags.add("streaming")
    if any(
        marker in lowered
        for marker in [
            "mlx backend",
            "teamai/model_backend.py",
            "model load",
            "generation failures",
            "operator-facing recovery",
        ]
    ):
        tags.add("mlx_backend")
    if any(
        marker in lowered
        for marker in [
            "planner and verifier",
            "structured planner",
            "structured verifier",
            "json planning",
            "json verification",
            "json planning / verification",
            "_extract_summary_and_tasks",
            "next_tasks",
            "final_answer",
        ]
    ):
        tags.add("structured_outputs")
    return tags


def _summarize_tool_result(tool_result: ToolExecutionResult, *, workspace: Path) -> str:
    raw_path = str(tool_result.metadata.get("path", "")).strip()
    path = _normalize_path(raw_path, workspace) if raw_path else ""
    if tool_result.tool == "read_file" and path:
        return f"Read {path}."
    if tool_result.tool == "list_files" and path:
        return f"Listed {path}."
    if tool_result.tool == "search_text":
        pattern = str(tool_result.metadata.get("pattern", "")).strip()
        if path and pattern:
            return f"Searched {path} for {pattern!r}."
        if path:
            return f"Searched {path}."
    if tool_result.tool == "run_command":
        command = str(tool_result.metadata.get("command", "")).strip()
        if command:
            return f"Ran safe command: {command}."
    return ""


def _normalize_path(raw_path: str, workspace: Path) -> str:
    try:
        workspace = workspace.resolve()
        resolved = Path(raw_path).resolve()
        return str(resolved.relative_to(workspace))
    except Exception:
        return raw_path


def _build_suggested_codex_prompt(
    *,
    task: str,
    task_route: str,
    primary_task: str | None,
    key_paths: list[str],
    evidence: list[str],
    open_questions: list[str],
) -> str:
    parts = [
        "Continue from this local-model handoff.",
        f"Original goal: {task}",
    ]
    if task_route == "codex_handoff":
        parts.append(
            "Treat this as a Codex-lead implementation task; use the local run as reconnaissance instead of trusting it to carry the full implementation."
        )
    elif task_route == "deterministic_patch":
        parts.append("A deterministic patch route was selected, so review or apply the generated approval artifact first.")
    if primary_task:
        parts.append(f"Prioritize this next task first: {primary_task}")
    if key_paths:
        parts.append(f"Inspect these paths first: {', '.join(key_paths[:6])}")
    if evidence:
        parts.append(f"Use this evidence: {' '.join(evidence[:3])}")
    if open_questions:
        parts.append(f"Verify this first if still unresolved: {open_questions[0]}")
    parts.append("Then implement or review the highest-value next change.")
    return " ".join(parts)
