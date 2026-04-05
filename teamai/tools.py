from __future__ import annotations

import shlex
import shutil
import subprocess
from pathlib import Path

from .approvals import PatchApprovalStore
from .config import Settings
from .schemas import ToolAction, ToolExecutionResult


READ_ONLY_COMMAND_PREFIXES: tuple[tuple[str, ...], ...] = (
    ("pwd",),
    ("ls",),
    ("find",),
    ("rg",),
    ("cat",),
    ("sed",),
    ("head",),
    ("tail",),
    ("wc",),
    ("git", "status"),
    ("git", "diff"),
    ("git", "show"),
    ("git", "log"),
    ("git", "rev-parse"),
    ("git", "branch", "--show-current"),
    ("python3", "-m", "pytest"),
    ("python", "-m", "pytest"),
    ("python3", "-m", "unittest"),
    ("python", "-m", "unittest"),
    ("pytest",),
)

IGNORED_PATH_NAMES = {
    ".git",
    ".teamai",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}


class WorkspaceTools:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._approvals = PatchApprovalStore()

    def describe_tools(self, *, execution_mode: str) -> str:
        write_note = (
            "Write tools create pending patch approvals; no file changes are applied until explicitly approved."
            if execution_mode == "workspace_write" and self._settings.allow_writes
            else "Write tools are disabled."
        )
        shell_note = (
            "Shell commands are enabled with a strict read-only allowlist."
            if self._settings.allow_shell
            else "Shell commands are disabled."
        )
        return f"""- list_files(args: path='.', recursive=false, max_entries=50)
- search_text(args: pattern, path='.', max_matches=40)
- read_file(args: path, start_line=1, end_line=200)
- run_command(args: command, cwd='.')
- write_file(args: path, content) [{write_note}]
- replace_in_file(args: path, old_text, new_text, replace_all=false) [{write_note}]

{shell_note}
"""

    def execute_actions(
        self,
        actions: list[ToolAction],
        *,
        workspace: Path,
        execution_mode: str,
        approval_context: dict | None = None,
    ) -> list[ToolExecutionResult]:
        results: list[ToolExecutionResult] = []
        for action in actions:
            try:
                result = self._execute(
                    action,
                    workspace=workspace,
                    execution_mode=execution_mode,
                    approval_context=approval_context,
                )
            except Exception as exc:
                result = ToolExecutionResult(
                    tool=action.tool,
                    success=False,
                    error=str(exc),
                    output="",
                )
            results.append(result)
        return results

    def _execute(
        self,
        action: ToolAction,
        *,
        workspace: Path,
        execution_mode: str,
        approval_context: dict | None = None,
    ) -> ToolExecutionResult:
        if action.tool == "list_files":
            return self._list_files(action.args, workspace)
        if action.tool == "search_text":
            return self._search_text(action.args, workspace)
        if action.tool == "read_file":
            return self._read_file(action.args, workspace)
        if action.tool == "run_command":
            return self._run_command(action.args, workspace)
        if action.tool == "write_file":
            return self._write_file(
                action.args,
                workspace,
                execution_mode,
                action.reason,
                approval_context=approval_context,
            )
        if action.tool == "replace_in_file":
            return self._replace_in_file(
                action.args,
                workspace,
                execution_mode,
                action.reason,
                approval_context=approval_context,
            )
        raise ValueError(f"Unsupported tool: {action.tool}")

    def _list_files(self, args: dict, workspace: Path) -> ToolExecutionResult:
        path = self._resolve_path(args.get("path", "."), workspace)
        recursive = bool(args.get("recursive", False))
        max_entries = min(int(args.get("max_entries", 50)), 200)

        entries: list[str] = []
        iterator = path.rglob("*") if recursive else path.iterdir()
        for item in iterator:
            if self._should_skip_path(item, workspace):
                continue
            entries.append(self._display_path(item, workspace) + ("/" if item.is_dir() else ""))
            if len(entries) >= max_entries:
                break

        return ToolExecutionResult(
            tool="list_files",
            success=True,
            output="\n".join(sorted(entries)),
            metadata={"path": str(path), "count": len(entries)},
        )

    def _search_text(self, args: dict, workspace: Path) -> ToolExecutionResult:
        pattern = str(args.get("pattern", "")).strip()
        if not pattern:
            raise ValueError("`search_text` requires a non-empty `pattern`.")

        path = self._resolve_path(args.get("path", "."), workspace)
        max_matches = min(int(args.get("max_matches", 40)), 200)

        rg_binary = shutil.which("rg")
        if rg_binary:
            command = [
                rg_binary,
                "-n",
                "--hidden",
                "--max-count",
                str(max_matches),
                "--glob",
                "!.git",
                "--glob",
                "!.venv",
                pattern,
                str(path),
            ]
            completed = subprocess.run(
                command,
                cwd=str(workspace),
                capture_output=True,
                text=True,
                timeout=self._settings.command_timeout_seconds,
            )
            output = completed.stdout.strip() or completed.stderr.strip()
            return ToolExecutionResult(
                tool="search_text",
                success=completed.returncode in {0, 1},
                output=self._truncate(output),
                metadata={"path": str(path), "pattern": pattern},
            )

        matches: list[str] = []
        for file_path in path.rglob("*"):
            if not file_path.is_file() or self._should_skip_path(file_path, workspace):
                continue
            try:
                text = file_path.read_text(encoding="utf-8")
            except Exception:
                continue
            for line_number, line in enumerate(text.splitlines(), start=1):
                if pattern in line:
                    matches.append(f"{self._display_path(file_path, workspace)}:{line_number}:{line}")
                    if len(matches) >= max_matches:
                        break
            if len(matches) >= max_matches:
                break

        return ToolExecutionResult(
            tool="search_text",
            success=True,
            output=self._truncate("\n".join(matches)),
            metadata={"path": str(path), "pattern": pattern, "fallback": True},
        )

    def _read_file(self, args: dict, workspace: Path) -> ToolExecutionResult:
        path = self._resolve_path(args["path"], workspace)
        start_line = max(int(args.get("start_line", 1)), 1)
        end_line = max(int(args.get("end_line", 200)), start_line)

        raw = path.read_text(encoding="utf-8")
        limited = raw[: self._settings.max_file_bytes]
        lines = limited.splitlines()
        selected = lines[start_line - 1 : end_line]
        rendered = "\n".join(
            f"{index + start_line:04d}: {line}" for index, line in enumerate(selected)
        )

        return ToolExecutionResult(
            tool="read_file",
            success=True,
            output=rendered,
            metadata={
                "path": str(path),
                "start_line": start_line,
                "end_line": end_line,
                "truncated": len(raw) > len(limited),
            },
        )

    def _run_command(self, args: dict, workspace: Path) -> ToolExecutionResult:
        if not self._settings.allow_shell:
            raise ValueError("Shell execution is disabled by configuration.")

        raw_command = args.get("command")
        if isinstance(raw_command, list):
            argv = [str(part) for part in raw_command]
        else:
            argv = shlex.split(str(raw_command or ""))

        if not argv:
            raise ValueError("`run_command` requires a non-empty `command`.")
        if not self._is_allowed_command(argv):
            raise ValueError(f"Command is not allowed: {' '.join(argv)}")

        cwd = self._resolve_path(args.get("cwd", "."), workspace)
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=self._settings.command_timeout_seconds,
        )

        combined_output = completed.stdout
        if completed.stderr:
            combined_output = f"{combined_output}\n[stderr]\n{completed.stderr}".strip()

        return ToolExecutionResult(
            tool="run_command",
            success=completed.returncode == 0,
            output=self._truncate(combined_output),
            metadata={
                "command": argv,
                "cwd": str(cwd),
                "returncode": completed.returncode,
            },
            error=None if completed.returncode == 0 else f"Command exited with {completed.returncode}",
        )

    def _write_file(
        self,
        args: dict,
        workspace: Path,
        execution_mode: str,
        reason: str,
        *,
        approval_context: dict | None = None,
    ) -> ToolExecutionResult:
        self._assert_writes_enabled(execution_mode)
        path = self._resolve_path(args["path"], workspace)
        content = str(args.get("content", ""))
        before_exists = path.exists()
        before_text = path.read_text(encoding="utf-8") if before_exists else ""
        if before_text == content:
            raise ValueError(f"`write_file` would not change {path}")
        approval = self._approvals.create(
            workspace=workspace,
            path=Path(self._display_path(path, workspace)),
            before_text=before_text,
            after_text=content,
            before_exists=before_exists,
            reason=reason,
            source_tool="write_file",
            continuation=self._build_approval_continuation(
                approval_context=approval_context,
                execution_mode=execution_mode,
            ),
        )
        approval_id = str(approval["approval_id"])
        return ToolExecutionResult(
            tool="write_file",
            success=True,
            output=(
                f"Created pending patch approval {approval_id} for {self._display_path(path, workspace)}. "
                "No file changes were applied. "
                f"Review with `teamai approvals show {approval_id}` and apply with `teamai approvals apply {approval_id}`."
            ),
            metadata={
                "path": str(path),
                "bytes": len(content.encode("utf-8")),
                "approval_id": approval_id,
                "approval_status": "pending",
                "requires_approval": True,
            },
        )

    def _replace_in_file(
        self,
        args: dict,
        workspace: Path,
        execution_mode: str,
        reason: str,
        *,
        approval_context: dict | None = None,
    ) -> ToolExecutionResult:
        self._assert_writes_enabled(execution_mode)
        path = self._resolve_path(args["path"], workspace)
        old_text = str(args.get("old_text", ""))
        new_text = str(args.get("new_text", ""))
        replace_all = bool(args.get("replace_all", False))

        if not old_text:
            raise ValueError("`replace_in_file` requires non-empty `old_text`.")

        original = path.read_text(encoding="utf-8")
        occurrences = original.count(old_text)
        if occurrences == 0:
            raise ValueError(f"`old_text` was not found in {path}")

        updated = original.replace(old_text, new_text) if replace_all else original.replace(old_text, new_text, 1)
        if updated == original:
            raise ValueError(f"`replace_in_file` would not change {path}")
        approval = self._approvals.create(
            workspace=workspace,
            path=Path(self._display_path(path, workspace)),
            before_text=original,
            after_text=updated,
            before_exists=True,
            reason=reason,
            source_tool="replace_in_file",
            continuation=self._build_approval_continuation(
                approval_context=approval_context,
                execution_mode=execution_mode,
            ),
        )
        approval_id = str(approval["approval_id"])

        return ToolExecutionResult(
            tool="replace_in_file",
            success=True,
            output=(
                f"Created pending patch approval {approval_id} for {self._display_path(path, workspace)} "
                f"covering {occurrences if replace_all else 1} replacement(s). "
                "No file changes were applied. "
                f"Review with `teamai approvals show {approval_id}` and apply with `teamai approvals apply {approval_id}`."
            ),
            metadata={
                "path": str(path),
                "replace_all": replace_all,
                "occurrences": occurrences,
                "approval_id": approval_id,
                "approval_status": "pending",
                "requires_approval": True,
            },
        )

    def _assert_writes_enabled(self, execution_mode: str) -> None:
        if execution_mode != "workspace_write":
            raise ValueError("Write tools require `workspace_write` execution mode.")
        if not self._settings.allow_writes:
            raise ValueError("Write tools are disabled by configuration.")

    @staticmethod
    def _build_approval_continuation(
        *,
        approval_context: dict | None,
        execution_mode: str,
    ) -> dict | None:
        if not approval_context:
            return None

        original_task = str(approval_context.get("task", "")).strip()
        if not original_task:
            return None

        requested_mode = str(approval_context.get("execution_mode", execution_mode)).strip()
        if requested_mode not in {"read_only", "workspace_write"}:
            requested_mode = execution_mode

        return {
            "original_task": original_task,
            "requested_execution_mode": requested_mode,
        }

    def _resolve_path(self, candidate: str, workspace: Path) -> Path:
        workspace = workspace.resolve()
        raw = Path(candidate).expanduser()
        resolved = raw.resolve() if raw.is_absolute() else (workspace / raw).resolve()
        if workspace not in {resolved, *resolved.parents}:
            raise ValueError(f"Path escapes the workspace root: {resolved}")
        return resolved

    def _is_allowed_command(self, argv: list[str]) -> bool:
        return any(self._matches_prefix(argv, prefix) for prefix in READ_ONLY_COMMAND_PREFIXES)

    def _should_skip_path(self, path: Path, workspace: Path) -> bool:
        try:
            relative = path.relative_to(workspace)
            parts = relative.parts
        except ValueError:
            parts = path.parts

        for part in parts:
            if part in IGNORED_PATH_NAMES:
                return True
            if part.endswith(".egg-info"):
                return True
        return False

    @staticmethod
    def _matches_prefix(argv: list[str], prefix: tuple[str, ...]) -> bool:
        if len(argv) < len(prefix):
            return False
        return tuple(argv[: len(prefix)]) == prefix

    def _truncate(self, text: str) -> str:
        normalized = text.strip()
        if len(normalized) <= self._settings.max_command_output_chars:
            return normalized
        limit = self._settings.max_command_output_chars
        return normalized[:limit] + "\n...[truncated]"

    @staticmethod
    def _display_path(path: Path, workspace: Path) -> str:
        try:
            return str(path.resolve().relative_to(workspace.resolve()))
        except Exception:
            return str(path)
