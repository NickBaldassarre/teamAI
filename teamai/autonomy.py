from __future__ import annotations

import ast
import difflib
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence
from uuid import uuid4

from .approvals import PatchApprovalStore
from .schemas import (
    AutonomousRunState,
    CheckExecution,
    ComplexityLevel,
    FailureDiagnosis,
    FailureType,
    MutationReceipt,
    RepoIndex,
    RepairAttempt,
    VerifierOutput,
    WritePolicy,
)


_HIGH_RISK_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|/)(pyproject\.toml|package\.json|package-lock\.json|pnpm-lock\.yaml|yarn\.lock|poetry\.lock|requirements(\.txt)?|setup\.py|Pipfile(\.lock)?)$", re.IGNORECASE),
    re.compile(r"(^|/)(migrations?|alembic|schema|schemas?)(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)\.env(\..+)?$", re.IGNORECASE),
    re.compile(r"(^|/)(\.github|\.gitlab|\.circleci|vercel\.json|netlify\.toml|Dockerfile|docker-compose)", re.IGNORECASE),
)
_FORBIDDEN_BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".ico", ".so", ".dylib"}
_DEFAULT_ALLOWED_CONFIGS = (
    "README.md",
    "teamai/",
    "tests/",
    "pyproject.toml",
    "setup.py",
)
_BRANCH_RE = re.compile(r"^[A-Za-z0-9._/-]+$")
_PROTECTED_BRANCH_NAMES = {"main", "master", "default", "trunk"}


@dataclass(frozen=True)
class RelevantContextBundle:
    repo_summary: str
    relevant_paths: tuple[str, ...]
    symbol_definitions: dict[str, list[str]]
    failing_tests: dict[str, str]
    nearest_imports: dict[str, list[str]]
    dependent_call_sites: dict[str, list[str]]
    traceback_excerpt: str = ""
    current_diff: str = ""
    prior_failed_repairs: tuple[str, ...] = ()


@dataclass(frozen=True)
class PatchExecutionContext:
    policy: WritePolicy
    phase: str = "workspace"
    allowed_path_scopes: tuple[str, ...] = ()
    tests_passed: bool | None = None
    verifier_confidence: float | None = None


@dataclass(frozen=True)
class GitCommandResult:
    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


def derive_write_policy(*, execution_mode: str, requested_policy: WritePolicy | None) -> WritePolicy:
    if requested_policy is not None:
        return requested_policy
    if execution_mode == "workspace_write":
        return "auto_apply_low_risk"
    return "read_only"


class PatchExecutor:
    def __init__(self) -> None:
        self._approvals = PatchApprovalStore()

    def execute_bundle(
        self,
        *,
        workspace: Path,
        changes: list[dict[str, Any]],
        reason: str,
        source_tool: str,
        context: PatchExecutionContext,
        continuation: dict[str, Any] | None = None,
    ) -> MutationReceipt:
        normalized = [self._normalize_change(change) for change in changes if change]
        if not normalized:
            raise ValueError("Patch execution requires at least one file change.")

        file_count = len(normalized)
        line_count = sum(self._line_delta(change["before_text"], change["after_text"]) for change in normalized)
        changed_paths = [str(change["path"]) for change in normalized]
        risk_reasons = self._assess_risk(
            changes=normalized,
            file_count=file_count,
            line_count=line_count,
            context=context,
        )
        risk_level = self._risk_level(risk_reasons)

        if context.policy == "read_only":
            return MutationReceipt(
                mode="blocked",
                policy=context.policy,
                phase=str(context.phase),
                blocked=True,
                blocked_reason="Write policy is read_only.",
                file_count=file_count,
                line_count=line_count,
                changed_paths=changed_paths,
                risk_level=risk_level,
                risk_reasons=risk_reasons,
                verifier_confidence=context.verifier_confidence,
                tests_passed=context.tests_passed,
            )

        if self._should_require_approval(context=context, risk_reasons=risk_reasons):
            approval = self._approvals.create_bundle(
                workspace=workspace,
                changes=normalized,
                reason=reason,
                source_tool=source_tool,
                continuation=continuation,
            )
            return MutationReceipt(
                mode="approval",
                policy=context.policy,
                phase=str(context.phase),
                applied=False,
                approval_required=True,
                approval_id=str(approval["approval_id"]),
                file_count=file_count,
                line_count=line_count,
                changed_paths=changed_paths,
                risk_level=risk_level,
                risk_reasons=risk_reasons,
                verifier_confidence=context.verifier_confidence,
                tests_passed=context.tests_passed,
            )

        self._apply_changes(workspace=workspace, changes=normalized)
        return MutationReceipt(
            mode="direct_apply",
            policy=context.policy,
            phase=str(context.phase),
            applied=True,
            file_count=file_count,
            line_count=line_count,
            changed_paths=changed_paths,
            risk_level=risk_level,
            risk_reasons=risk_reasons,
            verifier_confidence=context.verifier_confidence,
            tests_passed=context.tests_passed,
        )

    @staticmethod
    def _normalize_change(change: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": str(change.get("path", "")).strip(),
            "before_exists": bool(change.get("before_exists", False)),
            "after_exists": bool(change.get("after_exists", True)),
            "before_text": str(change.get("before_text", "")),
            "after_text": str(change.get("after_text", "")),
        }

    @staticmethod
    def _line_delta(before_text: str, after_text: str) -> int:
        diff = difflib.unified_diff(before_text.splitlines(), after_text.splitlines())
        return sum(1 for line in diff if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))

    def _assess_risk(
        self,
        *,
        changes: list[dict[str, Any]],
        file_count: int,
        line_count: int,
        context: PatchExecutionContext,
    ) -> list[str]:
        reasons: list[str] = []
        max_files = 3 if context.policy == "auto_apply_low_risk" else 6
        max_lines = 80 if context.policy == "auto_apply_low_risk" else 240
        if file_count > max_files:
            reasons.append(f"changed_files>{max_files}")
        if line_count > max_lines:
            reasons.append(f"changed_lines>{max_lines}")

        for change in changes:
            path = str(change["path"])
            normalized_path = path.replace("\\", "/")
            if self._path_is_forbidden(normalized_path):
                reasons.append(f"forbidden_path:{normalized_path}")
            if self._path_is_high_risk(normalized_path):
                reasons.append(f"high_risk_path:{normalized_path}")
            if context.policy == "auto_apply_scoped" and context.allowed_path_scopes:
                if not any(
                    normalized_path == scope or normalized_path.startswith(f"{scope.rstrip('/')}/")
                    for scope in context.allowed_path_scopes
                ):
                    reasons.append(f"out_of_scope:{normalized_path}")
            if self._looks_binary(change):
                reasons.append(f"binary:{normalized_path}")
            if not bool(change.get("after_exists", True)) and self._large_deletion(change):
                reasons.append(f"large_deletion:{normalized_path}")

        if context.phase == "workspace":
            if context.tests_passed is not True:
                reasons.append("tests_not_green")
            confidence = context.verifier_confidence if context.verifier_confidence is not None else 0.0
            threshold = 0.55 if context.policy != "full_auto" else 0.35
            if confidence < threshold:
                reasons.append(f"verifier_confidence<{threshold:.2f}")
        return reasons

    @staticmethod
    def _path_is_high_risk(path: str) -> bool:
        return any(pattern.search(path) for pattern in _HIGH_RISK_PATTERNS)

    @staticmethod
    def _path_is_forbidden(path: str) -> bool:
        lowered = path.lower()
        return lowered.startswith(".git/") or lowered.endswith((".pem", ".key", ".crt"))

    @staticmethod
    def _looks_binary(change: dict[str, Any]) -> bool:
        path = str(change["path"])
        if Path(path).suffix.lower() in _FORBIDDEN_BINARY_SUFFIXES:
            return True
        return "\x00" in str(change.get("before_text", "")) or "\x00" in str(change.get("after_text", ""))

    @staticmethod
    def _large_deletion(change: dict[str, Any]) -> bool:
        before_lines = str(change.get("before_text", "")).splitlines()
        after_lines = str(change.get("after_text", "")).splitlines()
        return bool(before_lines) and not after_lines and len(before_lines) > 40

    @staticmethod
    def _risk_level(reasons: list[str]) -> str:
        if any(reason.startswith(("forbidden_path", "high_risk_path", "binary", "large_deletion")) for reason in reasons):
            return "high"
        if reasons:
            return "medium"
        return "low"

    @staticmethod
    def _should_require_approval(*, context: PatchExecutionContext, risk_reasons: list[str]) -> bool:
        if context.policy == "propose_only":
            return True
        if context.phase == "sandbox":
            return False
        return bool(risk_reasons)

    @staticmethod
    def _apply_changes(*, workspace: Path, changes: list[dict[str, Any]]) -> None:
        applied: list[tuple[Path, bool, str]] = []
        try:
            for change in changes:
                target = (workspace / str(change["path"])).resolve()
                before_exists = bool(change["before_exists"])
                before_text = str(change["before_text"])
                applied.append((target, before_exists, before_text))
                if bool(change["after_exists"]):
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(str(change["after_text"]), encoding="utf-8")
                elif target.exists():
                    target.unlink()
        except Exception:
            for target, before_exists, before_text in reversed(applied):
                if before_exists:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(before_text, encoding="utf-8")
                elif target.exists():
                    target.unlink()
            raise


class SafeCommandExecutor:
    def __init__(self, timeout_seconds: int = 30) -> None:
        self._timeout_seconds = timeout_seconds

    def run_checks(
        self,
        *,
        workspace: Path,
        changed_paths: Sequence[str],
        repo_index: RepoIndex | None = None,
    ) -> list[CheckExecution]:
        commands = build_check_commands(workspace=workspace, changed_paths=list(changed_paths), repo_index=repo_index)
        results: list[CheckExecution] = []
        for command in commands:
            results.append(self._run_checked_command(command=command, workspace=workspace))
            if results[-1].returncode != 0:
                break
        return results

    def create_branch(self, *, workspace: Path, branch_name: str) -> GitCommandResult:
        self._validate_branch(branch_name)
        return self._run_git(workspace=workspace, command=["git", "checkout", "-b", branch_name])

    def git_add(self, *, workspace: Path, paths: Sequence[str]) -> GitCommandResult:
        normalized = self._normalize_paths(workspace=workspace, paths=paths)
        return self._run_git(workspace=workspace, command=["git", "add", *normalized])

    def git_commit(self, *, workspace: Path, message: str) -> GitCommandResult:
        if not message.strip():
            raise ValueError("Commit message cannot be empty.")
        return self._run_git(workspace=workspace, command=["git", "commit", "-m", message.strip()])

    def git_restore(self, *, workspace: Path, paths: Sequence[str]) -> GitCommandResult:
        normalized = self._normalize_paths(workspace=workspace, paths=paths)
        return self._run_git(workspace=workspace, command=["git", "restore", "--worktree", "--staged", "--", *normalized])

    def current_head(self, *, workspace: Path) -> str | None:
        result = self._run_git(workspace=workspace, command=["git", "rev-parse", "HEAD"])
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    def current_branch(self, *, workspace: Path) -> str | None:
        result = self._run_git(workspace=workspace, command=["git", "branch", "--show-current"])
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    def git_push(
        self,
        *,
        workspace: Path,
        remote: str,
        branch_name: str,
        set_upstream: bool = True,
    ) -> GitCommandResult:
        self._validate_branch(branch_name)
        protected = self._protected_branch_names(workspace=workspace, remote=remote)
        if branch_name in protected:
            return GitCommandResult(
                command=("git", "push", remote, branch_name),
                returncode=1,
                stdout="",
                stderr=(
                    f"Refusing to push to protected or default branch `{branch_name}`. "
                    "Push only to a feature branch."
                ),
            )
        command = ["git", "push"]
        if set_upstream:
            command.append("--set-upstream")
        command.extend([remote, branch_name])
        return self._run_git(workspace=workspace, command=command)

    def _run_git(self, *, workspace: Path, command: list[str]) -> GitCommandResult:
        completed = subprocess.run(
            command,
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=self._timeout_seconds,
            check=False,
        )
        return GitCommandResult(
            command=tuple(command),
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def _run_checked_command(self, *, command: list[str], workspace: Path) -> CheckExecution:
        try:
            completed = subprocess.run(
                command,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=self._timeout_seconds,
                check=False,
            )
            return CheckExecution(
                command=command,
                scope="scoped" if len(command) > 4 else "repo",
                returncode=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        except subprocess.TimeoutExpired as exc:
            return CheckExecution(
                command=command,
                scope="scoped" if len(command) > 4 else "repo",
                returncode=124,
                stdout=exc.stdout or "",
                stderr=exc.stderr or "",
                timed_out=True,
            )

    @staticmethod
    def _validate_branch(branch_name: str) -> None:
        if not branch_name or branch_name.startswith("-") or ".." in branch_name or " " in branch_name:
            raise ValueError(f"Unsafe branch name: {branch_name!r}")
        if not _BRANCH_RE.fullmatch(branch_name):
            raise ValueError(f"Unsafe branch name: {branch_name!r}")

    def _protected_branch_names(self, *, workspace: Path, remote: str) -> set[str]:
        protected = set(_PROTECTED_BRANCH_NAMES)
        symbolic_ref = self._run_git(
            workspace=workspace,
            command=["git", "symbolic-ref", f"refs/remotes/{remote}/HEAD"],
        )
        if symbolic_ref.returncode == 0:
            ref = symbolic_ref.stdout.strip()
            if ref:
                protected.add(ref.rsplit("/", maxsplit=1)[-1])
        return protected

    @staticmethod
    def _normalize_paths(*, workspace: Path, paths: Sequence[str]) -> list[str]:
        workspace_root = workspace.resolve()
        normalized: list[str] = []
        for raw in paths:
            candidate = (workspace_root / raw).resolve()
            if workspace_root not in {candidate, *candidate.parents}:
                raise ValueError(f"Path escapes workspace: {raw}")
            normalized.append(str(candidate.relative_to(workspace_root)))
        if not normalized:
            raise ValueError("At least one path is required.")
        return normalized


class RepoIndexer:
    def build(self, workspace: Path) -> RepoIndex:
        workspace = workspace.resolve()
        file_tree_summary: list[str] = []
        import_graph: dict[str, list[str]] = {}
        symbol_map: dict[str, list[str]] = {}
        test_map: dict[str, list[str]] = {}
        config_entrypoints: list[str] = []
        total_files = 0
        python_files = 0

        for candidate in sorted(self._iter_files(workspace)):
            relative = str(candidate.relative_to(workspace))
            total_files += 1
            if len(file_tree_summary) < 48:
                file_tree_summary.append(relative)
            if candidate.suffix == ".py":
                python_files += 1
                imports, symbols = self._inspect_python_file(candidate)
                if imports:
                    import_graph[relative] = imports
                if symbols:
                    symbol_map[relative] = symbols
                related_tests = self._related_tests_for(workspace=workspace, target=Path(relative))
                if related_tests:
                    test_map[relative] = related_tests
            if relative in {"README.md", "pyproject.toml", "setup.py", "agents.yaml", "teamai/cli.py", "teamai/__main__.py", "teamai/api.py"}:
                config_entrypoints.append(relative)

        return RepoIndex(
            workspace=str(workspace),
            file_tree_summary=file_tree_summary,
            import_graph=import_graph,
            symbol_map=symbol_map,
            test_map=test_map,
            config_entrypoints=config_entrypoints,
            total_files=total_files,
            python_files=python_files,
            has_tests=any(path.startswith("tests/") for path in file_tree_summary),
        )

    @staticmethod
    def summarize(index: RepoIndex) -> str:
        lines = [
            f"Repo index: {index.total_files} files, {index.python_files} Python files.",
        ]
        if index.config_entrypoints:
            lines.append(f"Entrypoints/config: {', '.join(index.config_entrypoints[:6])}.")
        if index.test_map:
            preview = []
            for source, tests in list(index.test_map.items())[:4]:
                preview.append(f"{source} -> {', '.join(tests[:2])}")
            lines.append(f"Test map: {'; '.join(preview)}.")
        return "\n".join(lines)

    @staticmethod
    def _iter_files(workspace: Path) -> Iterable[Path]:
        ignored_dirs = {".git", ".venv", ".teamai", "__pycache__", "build", "dist"}
        for path in workspace.rglob("*"):
            if path.is_dir():
                continue
            try:
                relative_parts = path.relative_to(workspace).parts
            except ValueError:
                continue
            if any(part in ignored_dirs or part.endswith(".egg-info") for part in relative_parts):
                continue
            yield path

    @staticmethod
    def _inspect_python_file(path: Path) -> tuple[list[str], list[str]]:
        try:
            module = ast.parse(path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            return [], []
        imports: list[str] = []
        symbols: list[str] = []
        for node in ast.walk(module):
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if module_name:
                    imports.append(module_name)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.append(node.name)
        deduped_imports = list(dict.fromkeys(imports))
        deduped_symbols = list(dict.fromkeys(symbols))
        return deduped_imports, deduped_symbols

    @staticmethod
    def _related_tests_for(*, workspace: Path, target: Path) -> list[str]:
        if target.parts[:1] == ("tests",):
            return [str(target)]
        if target.suffix != ".py":
            return []
        candidates = [
            Path("tests") / f"test_{target.stem}.py",
        ]
        return [str(candidate) for candidate in candidates if (workspace / candidate).exists()]


class ContextPackager:
    def build(
        self,
        *,
        task: str,
        workspace: Path,
        repo_index: RepoIndex | None = None,
        task_scopes: Sequence[str] = (),
        observed_paths: Sequence[str] = (),
        changed_paths: Sequence[str] = (),
        failure_output: str = "",
        prior_failed_repairs: Sequence[str] = (),
    ) -> RelevantContextBundle:
        workspace = workspace.resolve()
        index = repo_index or RepoIndexer().build(workspace)
        relevant_paths = self._select_relevant_paths(
            task=task,
            workspace=workspace,
            repo_index=index,
            task_scopes=task_scopes,
            observed_paths=observed_paths,
            changed_paths=changed_paths,
        )
        symbol_definitions = {
            path: self._symbol_definition_snippets(workspace / path, index.symbol_map.get(path, []))
            for path in relevant_paths
            if index.symbol_map.get(path)
        }
        nearest_imports = {
            path: list(index.import_graph.get(path, [])[:6])
            for path in relevant_paths
            if index.import_graph.get(path)
        }
        failing_tests = self._select_failing_tests(
            workspace=workspace,
            repo_index=index,
            relevant_paths=relevant_paths,
            failure_output=failure_output,
        )
        dependent_call_sites = self._find_dependent_call_sites(
            workspace=workspace,
            repo_index=index,
            symbol_definitions=symbol_definitions,
            skip_paths={
                str(path)
                for path in ([*changed_paths] or [*task_scopes] or list(relevant_paths[:1]))
                if str(path).strip()
            },
        )
        return RelevantContextBundle(
            repo_summary=RepoIndexer.summarize(index),
            relevant_paths=tuple(relevant_paths),
            symbol_definitions=symbol_definitions,
            failing_tests=failing_tests,
            nearest_imports=nearest_imports,
            dependent_call_sites=dependent_call_sites,
            traceback_excerpt=self._trim_traceback_excerpt(failure_output),
            current_diff="",
            prior_failed_repairs=tuple(str(item).strip() for item in prior_failed_repairs if str(item).strip())[:4],
        )

    @staticmethod
    def with_diff(bundle: RelevantContextBundle, *, diff_text: str) -> RelevantContextBundle:
        return replace(bundle, current_diff=diff_text.strip())

    @staticmethod
    def render(bundle: RelevantContextBundle) -> str:
        lines = [
            f"Repo summary: {bundle.repo_summary}",
            (
                "Relevant paths: "
                + (", ".join(bundle.relevant_paths[:6]) if bundle.relevant_paths else "(none selected)")
            ),
        ]
        if bundle.symbol_definitions:
            lines.append("Symbol definitions:")
            for path, symbols in list(bundle.symbol_definitions.items())[:4]:
                lines.append(f"- {path}: {', '.join(symbols[:4])}")
        if bundle.nearest_imports:
            lines.append("Nearest imports:")
            for path, imports in list(bundle.nearest_imports.items())[:4]:
                lines.append(f"- {path}: {', '.join(imports[:4])}")
        if bundle.failing_tests:
            lines.append("Failing or closely related test sources:")
            for path, excerpt in list(bundle.failing_tests.items())[:2]:
                compact_excerpt = " ".join(excerpt.split())
                lines.append(f"- {path}: {compact_excerpt[:220]}")
        if bundle.dependent_call_sites:
            lines.append("Dependent call sites:")
            for path, callsites in list(bundle.dependent_call_sites.items())[:4]:
                lines.append(f"- {path}: {' | '.join(callsites[:3])}")
        if bundle.traceback_excerpt:
            lines.append("Traceback excerpt:")
            lines.append(bundle.traceback_excerpt)
        if bundle.current_diff:
            lines.append("Current diff:")
            lines.append(bundle.current_diff)
        if bundle.prior_failed_repairs:
            lines.append("Prior failed repairs:")
            lines.extend(f"- {item}" for item in bundle.prior_failed_repairs[:4])
        return "\n".join(lines)

    def _select_relevant_paths(
        self,
        *,
        task: str,
        workspace: Path,
        repo_index: RepoIndex,
        task_scopes: Sequence[str],
        observed_paths: Sequence[str],
        changed_paths: Sequence[str],
    ) -> list[str]:
        ordered: list[str] = []
        seen: set[str] = set()

        def add(path: str) -> None:
            normalized = path.strip().rstrip("/")
            if not normalized or normalized in seen:
                return
            target = (workspace / normalized).resolve()
            if workspace not in {target, *target.parents} or not target.exists() or not target.is_file():
                return
            seen.add(normalized)
            ordered.append(normalized)

        for candidate in [*changed_paths, *observed_paths, *task_scopes]:
            add(str(candidate))

        lowered_task = task.lower()
        task_tokens = {
            token
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", lowered_task)
            if token not in {"with", "from", "that", "this", "tests", "test", "update", "fix", "file", "files"}
        }
        for candidate in repo_index.file_tree_summary:
            lowered_path = candidate.lower()
            if any(token in lowered_path for token in task_tokens):
                add(candidate)
                if len(ordered) >= 6:
                    break

        for source_path in list(ordered):
            for related_test in repo_index.test_map.get(source_path, []):
                add(related_test)

        return ordered[:8]

    @staticmethod
    def _symbol_definition_snippets(path: Path, symbols: list[str]) -> list[str]:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return symbols[:6]
        snippets: list[str] = []
        for symbol in symbols[:6]:
            match = re.search(
                rf"^\s*(?:async\s+def|def|class)\s+{re.escape(symbol)}\b.*$",
                text,
                flags=re.MULTILINE,
            )
            if match:
                snippets.append(match.group(0).strip())
            else:
                snippets.append(symbol)
        return snippets

    def _select_failing_tests(
        self,
        *,
        workspace: Path,
        repo_index: RepoIndex,
        relevant_paths: Sequence[str],
        failure_output: str,
    ) -> dict[str, str]:
        candidates: list[str] = []
        seen: set[str] = set()
        for path in relevant_paths:
            for related_test in repo_index.test_map.get(path, []):
                if related_test not in seen:
                    seen.add(related_test)
                    candidates.append(related_test)

        for match in re.findall(r"tests\.([A-Za-z0-9_./]+)", failure_output):
            module_path = f"tests/{match.replace('.', '/')}.py"
            if module_path not in seen and (workspace / module_path).exists():
                seen.add(module_path)
                candidates.append(module_path)

        rendered: dict[str, str] = {}
        for path in candidates[:3]:
            try:
                rendered[path] = (workspace / path).read_text(encoding="utf-8")[:800]
            except OSError:
                continue
        return rendered

    def _find_dependent_call_sites(
        self,
        *,
        workspace: Path,
        repo_index: RepoIndex,
        symbol_definitions: dict[str, list[str]],
        skip_paths: set[str],
    ) -> dict[str, list[str]]:
        tracked_symbols: set[str] = set()
        for definitions in symbol_definitions.values():
            for definition in definitions:
                match = re.search(r"(?:def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", definition)
                if match:
                    tracked_symbols.add(match.group(1))
        if not tracked_symbols:
            return {}

        matches: dict[str, list[str]] = {}
        for candidate in repo_index.symbol_map.keys():
            if candidate in skip_paths:
                continue
            target = workspace / candidate
            try:
                lines = target.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            callsites: list[str] = []
            for line_number, line in enumerate(lines, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                for symbol in tracked_symbols:
                    if re.search(rf"\b{re.escape(symbol)}\b", line):
                        callsites.append(f"{candidate}:{line_number}:{stripped[:100]}")
                        break
                if len(callsites) >= 4:
                    break
            if callsites:
                matches[candidate] = callsites
        return matches

    @staticmethod
    def _trim_traceback_excerpt(failure_output: str) -> str:
        lines = [line.rstrip() for line in failure_output.splitlines() if line.strip()]
        if not lines:
            return ""
        excerpt = lines[-12:]
        return "\n".join(excerpt)


class AutonomousRunStateStore:
    def persist(self, *, workspace: Path, state: AutonomousRunState) -> Path:
        state_dir = workspace / ".teamai" / "run-states"
        state_dir.mkdir(parents=True, exist_ok=True)
        path = state_dir / f"{state.task_id}.json"
        path.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path


def build_check_commands(
    *,
    workspace: Path,
    changed_paths: list[str],
    repo_index: RepoIndex | None = None,
) -> list[list[str]]:
    python_command = preferred_python_command(workspace)
    if python_command is None:
        return []

    test_targets: list[str] = []
    seen_tests: set[str] = set()
    index = repo_index or RepoIndexer().build(workspace)

    for changed_path in changed_paths:
        for candidate in index.test_map.get(changed_path, []):
            if candidate in seen_tests:
                continue
            seen_tests.add(candidate)
            test_targets.append(candidate)

    commands: list[list[str]] = []
    for test_path in test_targets:
        commands.append([*python_command, "-m", "unittest", path_to_module(Path(test_path))])

    tests_dir = workspace / "tests"
    if tests_dir.exists() and tests_dir.is_dir():
        if repo_prefers_pytest(workspace):
            commands.append([*python_command, "-m", "pytest"])
        else:
            commands.append([*python_command, "-m", "unittest", "discover", "-s", "tests"])
        return _dedupe_commands(commands)

    changed_python_files = [path for path in changed_paths if path.endswith(".py") and (workspace / path).exists()]
    if changed_python_files:
        commands.append([*python_command, "-m", "py_compile", *changed_python_files])
    return _dedupe_commands(commands)


def preferred_python_command(workspace: Path) -> list[str] | None:
    for candidate in (
        workspace / ".venv" / "bin" / "python",
        workspace / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.exists():
            return [str(candidate)]
    if shutil.which("python3"):
        return ["python3"]
    if shutil.which("python"):
        return ["python"]
    return None


def path_to_module(path: Path) -> str:
    return ".".join(path.with_suffix("").parts)


def repo_prefers_pytest(workspace: Path) -> bool:
    if any((workspace / name).exists() for name in ("pytest.ini", ".pytest.ini", "conftest.py")):
        return True
    pyproject_path = workspace / "pyproject.toml"
    if not pyproject_path.exists():
        return False
    try:
        pyproject_text = pyproject_path.read_text(encoding="utf-8")
    except OSError:
        return False
    return any(marker in pyproject_text for marker in ("[tool.pytest", "pytest.ini_options"))


def classify_failure(checks: Sequence[CheckExecution]) -> FailureDiagnosis | None:
    if not checks:
        return None
    last = checks[-1]
    combined = "\n".join([last.stdout, last.stderr]).strip()
    lowered = combined.lower()
    if last.timed_out:
        return FailureDiagnosis(
            failure_type="timeout",
            summary="Scoped verification timed out.",
            strategy=repair_strategy_for("timeout"),
            raw_output=combined,
        )
    for marker in ("syntaxerror", "invalid syntax", "indentationerror"):
        if marker in lowered:
            return FailureDiagnosis(
                failure_type="syntax_error",
                summary="Verification found a syntax error in the edited code.",
                strategy=repair_strategy_for("syntax_error"),
                raw_output=combined,
            )
    if "modulenotfounderror" in lowered or "no module named" in lowered:
        failure_type: FailureType = "missing_dependency" if any(marker in lowered for marker in ("pip install", "requires", "dependency")) else "import_error"
        return FailureDiagnosis(
            failure_type=failure_type,
            summary="Verification failed because an import could not be resolved.",
            strategy=repair_strategy_for(failure_type),
            raw_output=combined,
        )
    if "typeerror" in lowered or "mypy" in lowered:
        return FailureDiagnosis(
            failure_type="type_error",
            summary="Verification found a type-level failure.",
            strategy=repair_strategy_for("type_error"),
            raw_output=combined,
        )
    if any(marker in lowered for marker in ("assertionerror", "failures=", "failed (failures", "expected", "assert ")):
        return FailureDiagnosis(
            failure_type="assertion_failure",
            summary="Verification failed an assertion in the targeted tests.",
            strategy=repair_strategy_for("assertion_failure"),
            raw_output=combined,
        )
    if any(marker in lowered for marker in ("would reformat", "ruff format", "black", "formatting")):
        return FailureDiagnosis(
            failure_type="formatting_failure",
            summary="Verification detected a formatting failure.",
            strategy=repair_strategy_for("formatting_failure"),
            raw_output=combined,
        )
    return FailureDiagnosis(
        failure_type="unknown",
        summary="Verification failed with an unknown error.",
        strategy=repair_strategy_for("unknown"),
        raw_output=combined,
    )


def repair_strategy_for(failure_type: FailureType) -> str:
    return {
        "syntax_error": "Re-open the edited file, fix the parse error, and rerun the narrowest possible check first.",
        "import_error": "Restore or add the missing import/module wiring and rerun the directly related tests.",
        "type_error": "Tighten the call signature or value shape that triggered the runtime/type mismatch.",
        "assertion_failure": "Use the failing assertion output to adjust only the affected behavior and keep scope narrow.",
        "formatting_failure": "Apply the repository formatting expectation or rewrite the changed block to match it.",
        "missing_dependency": "Do not install packages automatically; escalate because the fix needs a dependency change.",
        "flaky_test": "Retry once to confirm flakiness, then escalate if the result is unstable.",
        "timeout": "Narrow the verification scope and fix the blocking code path before rerunning.",
        "unknown": "Inspect the exact failing output, avoid broad rewrites, and attempt the smallest repair consistent with the log.",
    }[failure_type]


def compute_complexity(
    *,
    task: str,
    repo_index: RepoIndex,
    expected_files_touched: int,
    retrieved_context_size: int = 0,
    recent_failure_count: int = 0,
) -> ComplexityLevel:
    lowered = task.lower()
    broad_markers = ("refactor", "architecture", "multi-file", "cross-file", "broad", "system", "end-to-end")
    if (
        expected_files_touched >= 5
        or repo_index.total_files >= 250
        or retrieved_context_size >= 12_000
        or recent_failure_count >= 2
        or any(marker in lowered for marker in broad_markers)
    ):
        return "high"
    if (
        expected_files_touched >= 2
        or repo_index.has_tests
        or any(marker in lowered for marker in ("implement", "fix", "repair", "update", "wire"))
    ):
        return "medium"
    return "low"


def infer_task_scopes(*, task: str, workspace: Path, repo_index: RepoIndex | None = None) -> tuple[str, ...]:
    workspace_root = workspace.resolve()
    candidates: list[str] = []
    patterns = [
        re.compile(r"`([^`]+)`"),
        re.compile(r"'([^']+)'"),
        re.compile(r'"([^"]+)"'),
        re.compile(
            r"(?<![`'\"\w])((?:\.[A-Za-z0-9_-]+|[A-Za-z0-9_-]+)(?:/[A-Za-z0-9._-]+)+/?|(?:\.[A-Za-z0-9_-]+|[A-Za-z0-9_-]+)\.[A-Za-z0-9_-]+)"
        ),
    ]
    for pattern in patterns:
        for match in pattern.finditer(task):
            candidate = match.group(1).strip().rstrip(",.:;")
            if not candidate:
                continue
            resolved = (workspace_root / candidate).resolve()
            if workspace_root not in {resolved, *resolved.parents}:
                continue
            if not resolved.exists():
                continue
            relative = str(resolved.relative_to(workspace_root))
            if relative not in candidates:
                candidates.append(relative)

    index = repo_index or RepoIndexer().build(workspace)
    expanded = candidates[:]
    for candidate in candidates:
        for related_test in index.test_map.get(candidate, []):
            if related_test not in expanded:
                expanded.append(related_test)
    return tuple(expanded)


def build_commit_message(*, task: str, state: AutonomousRunState) -> str:
    summary = task.strip().splitlines()[0].strip().rstrip(".")
    if len(summary) > 60:
        summary = summary[:57].rstrip() + "..."
    if not summary:
        summary = "apply autonomous changes"
    changed = len(state.files_changed)
    return f"teamai: {summary} ({changed} file{'s' if changed != 1 else ''})"


def new_run_state(
    *,
    workspace: Path,
    policy: WritePolicy,
    complexity: ComplexityLevel = "low",
) -> AutonomousRunState:
    task_id = f"run_{uuid4().hex[:12]}"
    workspace_id = workspace.resolve().name or "workspace"
    return AutonomousRunState(
        task_id=task_id,
        workspace_id=workspace_id,
        workspace_path=str(workspace.resolve()),
        policy_mode=policy,
        complexity=complexity,
    )


def update_run_state_metrics(state: AutonomousRunState) -> AutonomousRunState:
    retries = sum(1 for attempt in state.repair_attempts if attempt.status in {"failed", "passed", "escalated"})
    successful_checks = sum(1 for check in state.checks_run if check.returncode == 0)
    failed_checks = sum(1 for check in state.checks_run if check.returncode != 0)
    disagreement_rate = 0.0
    if state.verifier_outputs:
        model_votes = [output.passed for output in state.verifier_outputs if output.source == "model"]
        operational_votes = [output.passed for output in state.verifier_outputs if output.source in {"operational", "handoff"}]
        if model_votes and operational_votes:
            disagreement_rate = 1.0 if model_votes[-1] != operational_votes[-1] else 0.0
    escalation_count = sum(1 for entry in state.routing_trace if entry.stage == "inline_escalation")
    final_verifier_confidence = state.verifier_outputs[-1].confidence if state.verifier_outputs else 0.0
    updated = state.model_copy(deep=True)
    updated.metrics = {
        "retry_count": float(retries),
        "successful_checks": float(successful_checks),
        "failed_checks": float(failed_checks),
        "verifier_disagreement_rate": disagreement_rate,
        "escalation_count": float(escalation_count),
        "routing_trace_count": float(len(state.routing_trace)),
        "final_verifier_confidence": float(final_verifier_confidence),
    }
    return updated


def collect_workspace_changes(*, source_root: Path, modified_root: Path) -> list[dict[str, Any]]:
    source_root = source_root.resolve()
    modified_root = modified_root.resolve()
    paths: set[str] = set()
    for root in (source_root, modified_root):
        for candidate in root.rglob("*"):
            if candidate.is_dir():
                continue
            relative = candidate.relative_to(root)
            if _skip_diff_path(relative):
                continue
            paths.add(str(relative))

    changes: list[dict[str, Any]] = []
    for relative in sorted(paths):
        source_path = source_root / relative
        modified_path = modified_root / relative
        source_exists = source_path.exists()
        modified_exists = modified_path.exists()
        before_text = _read_text_if_possible(source_path) if source_exists else ""
        after_text = _read_text_if_possible(modified_path) if modified_exists else ""
        if source_exists == modified_exists and before_text == after_text:
            continue
        changes.append(
            {
                "path": relative,
                "before_exists": source_exists,
                "after_exists": modified_exists,
                "before_text": before_text,
                "after_text": after_text,
            }
        )
    return changes


def render_change_diff(changes: Sequence[dict[str, Any]], *, max_chars: int = 4_000) -> str:
    sections: list[str] = []
    for change in changes:
        path = str(change.get("path", "")).strip() or "(unknown)"
        before_text = str(change.get("before_text", ""))
        after_text = str(change.get("after_text", ""))
        before_exists = bool(change.get("before_exists", False))
        after_exists = bool(change.get("after_exists", True))
        before_lines = before_text.splitlines()
        after_lines = after_text.splitlines()
        fromfile = f"a/{path}" if before_exists else "/dev/null"
        tofile = f"b/{path}" if after_exists else "/dev/null"
        diff = "\n".join(
            difflib.unified_diff(
                before_lines,
                after_lines,
                fromfile=fromfile,
                tofile=tofile,
                lineterm="",
            )
        ).strip()
        if diff:
            sections.append(diff)
    rendered = "\n\n".join(sections).strip()
    if len(rendered) <= max_chars:
        return rendered
    return rendered[:max_chars].rstrip() + "\n...[truncated]"


def _skip_diff_path(relative: Path) -> bool:
    parts = relative.parts
    if any(part in {".git", "__pycache__", ".venv", "build", "dist"} for part in parts):
        return True
    if not parts:
        return False
    if parts[0] != ".teamai":
        return False
    if len(parts) == 1:
        return False
    return parts[1] in {"approvals", "run-states"}


def _read_text_if_possible(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return ""


def _dedupe_commands(commands: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    deduped: list[list[str]] = []
    for command in commands:
        key = tuple(command)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(command)
    return deduped
