"""
Answer synthesis for teamAI.

Extracted from supervisor.py.  Contains all answer-building, fallback,
approval-required, and handoff-answer assembly logic.  The supervisor
delegates to an ``AnswerSynthesizer`` instance for anything that
constructs a final-answer string from round records, workspace state,
or routing context.

Usage::

    from teamai.synthesis import AnswerSynthesizer

    synth = AnswerSynthesizer(
        normalize_path=supervisor._normalize_path_arg,
        extract_candidate_paths=supervisor._extract_candidate_paths,
    )
    answer = synth.maybe_synthesize_repository_answer(
        task=task, rounds=rounds, workspace=workspace,
        successful_signatures=signatures,
    )
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from .schemas import RoundRecord, ToolExecutionResult
from .task_classifier import is_repository_inspection_task


# Marker prepended to any final answer assembled by deterministic synthesis
# rather than written by the local model.  The marker travels with the answer
# wherever it is persisted (transcripts, run history, handoff payloads, eval
# reports) so downstream readers cannot conflate templated text with model
# reasoning.
DETERMINISTIC_SYNTHESIS_LABEL = "[deterministic-synthesis]"

# Placeholder rendered in transcripts when a deterministic route does not
# invoke the model for the strategist/critic step.  The on-disk RoundRecord
# stores empty strings; the transcript renderer substitutes this notice so
# readers can immediately tell that no model reasoning was generated.
DETERMINISTIC_PERSONA_TRANSCRIPT_NOTICE = (
    "(deterministic route — no model strategist/critic was generated for this round)"
)


class AnswerSynthesizer:
    """Builds final-answer strings from round records and workspace state."""

    def __init__(
        self,
        normalize_path: Callable[[str, Path], str],
        extract_candidate_paths: Callable[[str], list[str]],
    ) -> None:
        self._normalize_path = normalize_path
        self._extract_candidate_paths = extract_candidate_paths

    # ------------------------------------------------------------------
    # Public API (mapped 1:1 from supervisor thin wrappers)
    # ------------------------------------------------------------------

    def maybe_synthesize_repository_answer(
        self,
        *,
        task: str,
        rounds: list[RoundRecord],
        workspace: Path,
        allow_partial: bool = False,
        successful_signatures: set[str],
    ) -> str | None:
        if not is_repository_inspection_task(task):
            return None

        has_readme = "read_file:README.md" in successful_signatures
        has_package_listing = "list_files:teamai" in successful_signatures
        has_config = "read_file:teamai/config.py" in successful_signatures
        has_runtime_anchor = (
            "read_file:teamai/cli.py" in successful_signatures
            or "read_file:teamai/supervisor.py" in successful_signatures
        )

        strict_ready = has_readme and has_package_listing and has_config and has_runtime_anchor
        partial_ready = has_readme and has_runtime_anchor and (has_package_listing or has_config)
        if not strict_ready:
            if not allow_partial or not partial_ready:
                return None

        contents = self._collect_read_file_outputs(rounds, workspace)
        readme_text = contents.get("README.md", "")
        pyproject_text = contents.get("pyproject.toml", "")
        memory_text = contents.get("PROJECT_MEMORY.md", "")
        combined_text = "\n".join([readme_text, pyproject_text, memory_text]).lower()
        implemented = self._implemented_feature_flags(workspace, contents)

        current_state_parts: list[str] = []
        if has_readme and pyproject_text:
            current_state_parts.append(
                "The repo is already packaged as a local-first Python application for MLX-based orchestration."
            )
        if "teamai/supervisor.py" in contents:
            current_state_parts.append(
                "The core strategist/critic/planner/verifier loop is implemented in the supervisor."
            )
        if "teamai/model_backend.py" in contents:
            current_state_parts.append(
                "The MLX backend is wired in with lazy model loading and explicit load/generation error handling."
            )
        if "teamai/tools.py" in contents:
            current_state_parts.append(
                "Workspace inspection tools and write restrictions are already scaffolded behind a sandboxed tool layer."
            )
        if "teamai/config.py" in contents:
            current_state_parts.append(
                "Runtime settings are centralized with environment validation, workspace scoping, and safety limits."
            )
        if implemented["persistent_memory"]:
            current_state_parts.append(
                "Persistent run history and cross-run workspace memory are already implemented through `.teamai/` state files and prompt injection."
            )

        tasks: list[str] = []
        if (
            any(keyword in combined_text for keyword in ["persistent memory", "persistent run history", "run history"])
            and not implemented["persistent_memory"]
        ):
            tasks.append(
                "Add persistent run history and memory so the loop can carry useful context across rounds and across separate runs."
            )

        if any(
            keyword in combined_text
            for keyword in [
                "patch-oriented editing tools",
                "safer patch-oriented write tools",
                "approval checkpoints",
                "before destructive changes",
            ]
        ):
            tasks.append(
                "Replace the coarse write path with patch-oriented editing tools and approval checkpoints before destructive changes."
            )

        if any(keyword in combined_text for keyword in ["streaming events", "streaming event output", "streaming output"]):
            tasks.append(
                "Add streaming event output across the CLI, API, and job flow so long runs expose persona and tool progress in real time."
            )

        if "teamai/model_backend.py" in contents:
            tasks.append(
                "Harden the MLX backend around model load, generation failures, and clearer operator-facing recovery guidance."
            )

        if any(keyword in combined_text for keyword in ["json planning / verification", "json planning", "verification"]):
            tasks.append(
                "Keep hardening structured planner and verifier outputs so longer inspection and execution runs stay reliable."
            )

        if not tasks:
            tasks.append(
                "Inspect and tighten `teamai/supervisor.py`, since it appears to be the main coordination point for the closed-loop behavior."
            )
            tasks.append(
                "Inspect and harden `teamai/model_backend.py`, since the MLX integration is the highest-risk runtime dependency after the CLI/config layer."
            )
            tasks.append(
                "Add higher-level tests for full repository-inspection runs so the system proves it can reach actionable next steps."
            )

        deduped_tasks: list[str] = []
        for task_item in tasks:
            if task_item not in deduped_tasks:
                deduped_tasks.append(task_item)

        current_state = " ".join(current_state_parts) if current_state_parts else "The repo structure and core runtime pieces are in place."
        tasks_section = "\n".join(f"- {task_item}" for task_item in deduped_tasks[:4])
        return f"Current state: {current_state}\n\nNext engineering tasks:\n{tasks_section}"

    def maybe_synthesize_codex_handoff_answer(
        self,
        *,
        task: str,
        rounds: list[RoundRecord],
        workspace: Path,
        task_route: str,
        task_relevant_candidates: list[str],
    ) -> str | None:
        if task_route != "codex_handoff":
            return None

        raw_key_paths: list[str] = []
        seen_paths: set[str] = set()
        next_focuses: list[str] = []
        seen_focuses: set[str] = set()
        evidence_count = 0

        for record in rounds:
            for result in record.tool_results:
                if not result.success:
                    continue
                evidence_count += 1
                raw_path = str(result.metadata.get("path", "")).strip()
                if raw_path:
                    normalized = self._normalize_path(raw_path, workspace)
                    if normalized not in seen_paths:
                        seen_paths.add(normalized)
                        raw_key_paths.append(normalized)
            next_focus = (record.verifier.next_focus or "").strip()
            if next_focus and next_focus not in seen_focuses:
                seen_focuses.add(next_focus)
                next_focuses.append(next_focus.rstrip("."))

        if len(raw_key_paths) < 2:
            fallback_paths = self._rank_codex_handoff_paths(
                task=task,
                paths=task_relevant_candidates,
            )
            for candidate in fallback_paths:
                if candidate in seen_paths:
                    continue
                seen_paths.add(candidate)
                raw_key_paths.append(candidate)
                if len(raw_key_paths) >= 8:
                    break

        key_paths = self._rank_codex_handoff_paths(task=task, paths=raw_key_paths)
        if len(key_paths) < 2:
            for candidate in self._rank_codex_handoff_paths(
                task=task,
                paths=task_relevant_candidates,
            ):
                if candidate in key_paths:
                    continue
                key_paths.append(candidate)
                if len(key_paths) >= 4:
                    break
        lead_task = None
        for focus in next_focuses:
            normalized_focus = self._normalize_codex_handoff_focus(focus, workspace=workspace)
            if normalized_focus:
                lead_task = normalized_focus
                break

        if evidence_count < 2 and not (key_paths or lead_task):
            return None

        lines = [
            "Current state: The local run treated this as a broad coding task and gathered reconnaissance instead of attempting autonomous implementation.",
            "",
            "Next engineering tasks:",
        ]
        if lead_task:
            lines.append(f"- {self._ensure_sentence(lead_task)}")
        if key_paths and not self._lead_task_covers_paths(lead_task, key_paths[:2]):
            lines.append(f"- Inspect the most relevant paths first: {', '.join(key_paths[:4])}.")
        lines.append(f"- Implement the requested change in Codex after verifying the scoped plan for: {task}")
        return "\n".join(lines)

    def build_local_drift_handoff_answer(
        self,
        *,
        task: str,
        rounds: list[RoundRecord],
        workspace: Path,
        reroute_reason: str,
        task_relevant_candidates: list[str],
    ) -> str:
        synthesized = self.maybe_synthesize_codex_handoff_answer(
            task=task,
            rounds=rounds,
            workspace=workspace,
            task_route="codex_handoff",
            task_relevant_candidates=task_relevant_candidates,
        )
        if synthesized:
            return synthesized.replace(
                "Current state: The local run treated this as a broad coding task and gathered reconnaissance instead of attempting autonomous implementation.",
                "Current state: The local run started locally, gathered partial evidence, and then rerouted after it started drifting beyond the reliable local path.",
            )

        relevant_paths = task_relevant_candidates
        lines = [
            "Current state: The local run started locally but rerouted after it stopped making reliable scoped progress.",
            "",
            "Next engineering tasks:",
            f"- {reroute_reason}",
        ]
        if relevant_paths:
            lines.append(f"- Inspect the most relevant paths first: {', '.join(relevant_paths[:4])}.")
        lines.append(f"- Continue the requested change in Codex after verifying the scoped plan for: {task}")
        return "\n".join(lines)

    def rank_codex_handoff_paths(self, *, task: str, paths: list[str]) -> list[str]:
        return self._rank_codex_handoff_paths(task=task, paths=paths)

    # ------------------------------------------------------------------
    # Static public methods
    # ------------------------------------------------------------------

    @staticmethod
    def build_approval_required_answer(pending_approvals: list[dict[str, str]]) -> str:
        lines = [
            "Patch approval required before any proposed file changes can be applied.",
            "",
            "Pending approvals:",
        ]
        for item in pending_approvals:
            approval_id = item.get("approval_id", "")
            path = item.get("path", "(unknown path)")
            lines.append(f"- {approval_id} | {path}")
        lines.extend(
            [
                "",
                "No file changes were applied yet.",
                "Review a patch with `teamai approvals show <approval_id>` and apply it with `teamai approvals apply <approval_id>`.",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def collect_pending_approvals(
        tool_results: list[ToolExecutionResult],
        workspace: Path,
    ) -> list[dict[str, str]]:
        pending: list[dict[str, str]] = []
        for result in tool_results:
            if str(result.metadata.get("approval_status", "")).strip() != "pending":
                continue
            approval_id = str(result.metadata.get("approval_id", "")).strip()
            raw_path = str(result.metadata.get("path", "")).strip()
            relative_path = raw_path
            if raw_path:
                try:
                    relative_path = str(Path(raw_path).resolve().relative_to(workspace.resolve()))
                except Exception:
                    relative_path = raw_path
            pending.append(
                {
                    "approval_id": approval_id,
                    "path": relative_path or "(unknown path)",
                    "tool": result.tool,
                }
            )
        return pending

    @staticmethod
    def build_fallback_answer(rounds: list[RoundRecord], task: str) -> str:
        if not rounds:
            return f"No rounds completed for task: {task}"
        last_round = rounds[-1]
        return (
            f"Stopped before a verified completion. "
            f"Latest planner summary: {last_round.planner.summary}\n\n"
            f"Latest verifier summary: {last_round.verifier.summary}"
        )

    @staticmethod
    def rounds_are_fully_deterministic(rounds: list[RoundRecord]) -> bool:
        return bool(rounds) and all(record.reasoning_source == "deterministic" for record in rounds)

    @staticmethod
    def label_deterministic_synthesis(text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return text
        if stripped.startswith(DETERMINISTIC_SYNTHESIS_LABEL):
            return stripped
        return f"{DETERMINISTIC_SYNTHESIS_LABEL}\n{stripped}"

    @staticmethod
    def render_round_persona_text(text: str, *, reasoning_source: str) -> str:
        stripped = text.strip()
        if stripped or reasoning_source != "deterministic":
            return text
        return DETERMINISTIC_PERSONA_TRANSCRIPT_NOTICE

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _normalize_codex_handoff_focus(
        self,
        focus: str,
        *,
        workspace: Path,
    ) -> str | None:
        compact = " ".join(focus.split()).strip().rstrip(".")
        if not compact:
            return None

        lowered = compact.lower()
        normalized_paths: list[str] = []
        seen_paths: set[str] = set()
        for candidate in self._extract_candidate_paths(compact):
            normalized = self._normalize_path(candidate, workspace)
            resolved = (workspace / normalized).resolve()
            if not resolved.exists() or normalized in seen_paths:
                continue
            seen_paths.add(normalized)
            normalized_paths.append(normalized)

        if normalized_paths:
            if len(normalized_paths) == 1:
                return f"Inspect {normalized_paths[0]} before implementing the requested change"
            if len(normalized_paths) == 2:
                return (
                    f"Inspect {normalized_paths[0]} and {normalized_paths[1]} "
                    "before implementing the requested change"
                )
            return f"Inspect the most relevant paths first: {', '.join(normalized_paths[:4])}"

        if lowered.startswith(("inspect ", "review ", "read ", "trace ", "verify ", "map ", "compare ", "reproduce ")):
            return compact
        if lowered.startswith(("implement ", "fix ", "debug ", "update ")):
            return compact
        return None

    @staticmethod
    def _ensure_sentence(text: str) -> str:
        stripped = text.strip()
        if stripped.endswith((".", "!", "?")):
            return stripped
        return f"{stripped}."

    @staticmethod
    def _lead_task_covers_paths(lead_task: str | None, paths: list[str]) -> bool:
        if not lead_task:
            return False
        lowered = lead_task.lower()
        return all(path.lower() in lowered for path in paths if path)

    def _rank_codex_handoff_paths(self, *, task: str, paths: list[str]) -> list[str]:
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

        ranked = sorted(
            preferred_pool,
            key=lambda path: (-self._score_handoff_path(task_lower=task_lower, path=path), unique_paths.index(path)),
        )
        return ranked[:12]

    @staticmethod
    def _score_handoff_path(*, task_lower: str, path: str) -> int:
        path_lower = path.lower()
        score = 0
        if any(marker in task_lower for marker in ["inspect", "explore", "identify", "next tasks", "broad"]):
            if path_lower == "readme.md":
                score += 10
            elif path_lower == "teamai/config.py":
                score += 8
            elif path_lower == "project_memory.md":
                score += 6
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
            elif any(
                marker in path_lower
                for marker in ["prompts.py", "handoff.py", "bridge.py", "supervisor.py"]
            ):
                score += 4
        if any(marker in task_lower for marker in ["stream", "streaming", "event output", "progress output"]) and any(
            marker in path_lower for marker in ["cli.py", "api.py", "jobs.py", "schemas.py", "supervisor.py"]
        ):
            score += 6
        if any(marker in task_lower for marker in ["approval", "patch", "write path", "workspace_write", "deterministic"]) and any(
            marker in path_lower for marker in ["tools.py", "approvals.py", "supervisor.py", "test_tools.py", "test_approvals.py", "test_supervisor.py"]
        ):
            score += 6
        if any(marker in task_lower for marker in ["json", "planner", "verifier", "prompt", "structured output"]) and any(
            marker in path_lower for marker in ["prompts.py", "schemas.py", "supervisor.py", "test_supervisor.py"]
        ):
            score += 5
        return score

    def _implemented_feature_flags(
        self,
        workspace: Path,
        contents: dict[str, str],
    ) -> dict[str, bool]:
        return {
            "persistent_memory": (
                self._workspace_text_contains(
                    workspace,
                    contents,
                    "teamai/memory.py",
                    ["WorkspaceMemoryStore", "RUN_HISTORY_FILE_NAME", "MEMORY_FILE_NAME"],
                )
                and self._workspace_text_contains(
                    workspace,
                    contents,
                    "teamai/supervisor.py",
                    ["WorkspaceMemoryStore", "load_snapshot", "persist_run"],
                )
                and self._workspace_text_contains(
                    workspace,
                    contents,
                    "teamai/prompts.py",
                    ["Persistent workspace memory:", "Recent persisted runs:"],
                )
            ),
        }

    def _collect_read_file_outputs(
        self,
        rounds: list[RoundRecord],
        workspace: Path,
    ) -> dict[str, str]:
        outputs: dict[str, str] = {}
        for record in rounds:
            for action, result in zip(record.planner.actions, record.tool_results):
                if action.tool != "read_file" or not result.success:
                    continue
                path = self._normalize_path(action.args.get("path", "."), workspace)
                outputs[path] = result.output
        return outputs

    @staticmethod
    def _workspace_text_contains(
        workspace: Path,
        contents: dict[str, str],
        path: str,
        needles: list[str],
    ) -> bool:
        text = contents.get(path)
        if text is not None and all(needle in text for needle in needles):
            return True

        candidate = workspace / path
        if not candidate.exists() or not candidate.is_file():
            return False
        try:
            file_text = candidate.read_text(encoding="utf-8")
        except Exception:
            return False
        return all(needle in file_text for needle in needles)
