from __future__ import annotations

import difflib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .memory import STATE_DIR_NAME
from .patch_utils import extract_patch_targets


APPROVALS_DIR_NAME = "approvals"


class PatchApprovalStore:
    def create(
        self,
        *,
        workspace: Path,
        path: Path,
        before_text: str,
        after_text: str,
        before_exists: bool,
        reason: str,
        source_tool: str,
        after_exists: bool = True,
        continuation: dict[str, Any] | None = None,
        diff_text: str | None = None,
    ) -> dict[str, Any]:
        change = {
            "path": str(path),
            "before_exists": before_exists,
            "after_exists": after_exists,
            "before_text": before_text,
            "after_text": after_text,
        }
        return self.create_bundle(
            workspace=workspace,
            changes=[change],
            reason=reason,
            source_tool=source_tool,
            continuation=continuation,
            diff_text=diff_text,
        )

    def create_bundle(
        self,
        *,
        workspace: Path,
        changes: list[dict[str, Any]],
        reason: str,
        source_tool: str,
        continuation: dict[str, Any] | None = None,
        diff_text: str | None = None,
    ) -> dict[str, Any]:
        normalized_changes = [self._normalize_change_payload(change) for change in changes if change]
        if not normalized_changes:
            raise ValueError("Patch approval bundle requires at least one change.")

        approval_id = uuid4().hex[:12]
        created_at = datetime.now(timezone.utc).isoformat()
        changed_paths = [str(change["path"]) for change in normalized_changes]
        primary_path = changed_paths[0] if len(changed_paths) == 1 else "<multiple files>"
        payload: dict[str, Any] = {
            "approval_id": approval_id,
            "created_at": created_at,
            "status": "pending",
            "path": primary_path,
            "source_tool": source_tool,
            "reason": reason.strip(),
            "diff": diff_text or self._build_combined_diff(normalized_changes),
            "change_count": len(normalized_changes),
            "changed_paths": changed_paths,
            "changes": normalized_changes,
        }
        if len(normalized_changes) == 1:
            change = normalized_changes[0]
            payload["before_exists"] = change["before_exists"]
            payload["after_exists"] = change["after_exists"]
            payload["before_text"] = change["before_text"]
            payload["after_text"] = change["after_text"]
        if continuation:
            payload["continuation"] = {
                **continuation,
                "target_path": primary_path,
                "source_tool": source_tool,
                "approval_reason": reason.strip(),
            }
        self._write_record(workspace, payload)
        return payload

    def create_bundle_from_patch(
        self,
        *,
        workspace: Path,
        sandbox_root: Path,
        patch_text: str,
        reason: str,
        source_tool: str,
        continuation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        changes: list[dict[str, Any]] = []
        for target in extract_patch_targets(patch_text):
            changes.extend(
                self._changes_from_patch_target(
                    workspace=workspace,
                    sandbox_root=sandbox_root,
                    before_path=target.before_path,
                    after_path=target.after_path,
                )
            )

        if not changes:
            raise ValueError("Verified patch approval creation could not infer any changed files from the patch.")

        return self.create_bundle(
            workspace=workspace,
            changes=changes,
            reason=reason,
            source_tool=source_tool,
            continuation=continuation,
            diff_text=patch_text,
        )

    def list(self, *, workspace: Path, include_all: bool = False) -> list[dict[str, Any]]:
        directory = self._approvals_dir(workspace)
        if not directory.exists():
            return []

        records: list[dict[str, Any]] = []
        for candidate in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if not include_all and payload.get("status") != "pending":
                continue
            records.append(payload)

        records.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return records

    def get(self, *, workspace: Path, approval_id: str) -> dict[str, Any]:
        path = self._approval_path(workspace, approval_id)
        if not path.exists():
            raise KeyError(f"Patch approval not found: {approval_id}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Patch approval payload is invalid: {approval_id}")
        return payload

    def apply(self, *, workspace: Path, approval_id: str) -> dict[str, Any]:
        payload = self.get(workspace=workspace, approval_id=approval_id)
        if payload.get("status") != "pending":
            raise ValueError(f"Patch approval `{approval_id}` is not pending.")

        changes = self._approval_changes(payload)
        for change in changes:
            target = self._resolve_change_target(workspace, change["path"])
            before_exists = bool(change.get("before_exists", False))
            current_exists = target.exists()
            if before_exists != current_exists:
                payload["status"] = "stale"
                payload["stale_reason"] = (
                    f"Target existence changed for {change['path']} since the patch was proposed."
                )
                self._write_record(workspace, payload)
                raise ValueError(f"Patch approval `{approval_id}` is stale because {change['path']} changed.")

            current_text = target.read_text(encoding="utf-8") if current_exists else ""
            before_text = str(change.get("before_text", ""))
            if current_text != before_text:
                payload["status"] = "stale"
                payload["stale_reason"] = (
                    f"Target contents changed for {change['path']} since the patch was proposed."
                )
                self._write_record(workspace, payload)
                raise ValueError(f"Patch approval `{approval_id}` is stale because {change['path']} changed.")

        for change in changes:
            target = self._resolve_change_target(workspace, change["path"])
            if bool(change.get("after_exists", True)):
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(str(change.get("after_text", "")), encoding="utf-8")
            elif target.exists():
                target.unlink()

        payload["status"] = "applied"
        payload["applied_at"] = datetime.now(timezone.utc).isoformat()
        self._write_record(workspace, payload)
        return payload

    def reject(
        self,
        *,
        workspace: Path,
        approval_id: str,
        reason: str | None = None,
    ) -> dict[str, Any]:
        payload = self.get(workspace=workspace, approval_id=approval_id)
        if payload.get("status") != "pending":
            raise ValueError(f"Patch approval `{approval_id}` is not pending.")

        payload["status"] = "rejected"
        payload["rejected_at"] = datetime.now(timezone.utc).isoformat()
        rejection_reason = (reason or "").strip()
        if rejection_reason:
            payload["rejection_reason"] = rejection_reason
        self._write_record(workspace, payload)
        return payload

    def prune_stale(self, *, workspace: Path) -> list[dict[str, Any]]:
        pruned: list[dict[str, Any]] = []
        for payload in self.list(workspace=workspace, include_all=True):
            if payload.get("status") != "stale":
                continue
            approval_id = str(payload.get("approval_id", "")).strip()
            if not approval_id:
                continue
            path = self._approval_path(workspace, approval_id)
            if path.exists():
                path.unlink()
            pruned.append(payload)
        return pruned

    @staticmethod
    def summarize(payload: dict[str, Any], *, include_diff: bool = False) -> dict[str, Any]:
        summary = {
            "approval_id": payload.get("approval_id"),
            "created_at": payload.get("created_at"),
            "applied_at": payload.get("applied_at"),
            "rejected_at": payload.get("rejected_at"),
            "status": payload.get("status"),
            "path": payload.get("path"),
            "source_tool": payload.get("source_tool"),
            "reason": payload.get("reason"),
        }
        change_count = int(payload.get("change_count", 1) or 1)
        summary["change_count"] = change_count
        changed_paths = payload.get("changed_paths")
        if isinstance(changed_paths, list) and changed_paths:
            summary["changed_paths"] = [str(path) for path in changed_paths]
        continuation = payload.get("continuation")
        if isinstance(continuation, dict):
            summary["continuation_available"] = True
            summary["continuation_execution_mode"] = PatchApprovalStore.continuation_execution_mode(payload)
        if include_diff:
            summary["diff"] = payload.get("diff", "")
        if "stale_reason" in payload:
            summary["stale_reason"] = payload["stale_reason"]
        if "rejection_reason" in payload:
            summary["rejection_reason"] = payload["rejection_reason"]
        return summary

    @staticmethod
    def continuation_execution_mode(payload: dict[str, Any]) -> str:
        continuation = payload.get("continuation")
        if isinstance(continuation, dict):
            mode = str(continuation.get("requested_execution_mode", "")).strip()
            if mode in {"read_only", "workspace_write"}:
                return mode
        return "read_only"

    @staticmethod
    def build_continuation_task(
        payload: dict[str, Any],
        *,
        continuation_context: dict[str, Any] | None = None,
    ) -> str:
        approval_id = str(payload.get("approval_id", "")).strip() or "<approval_id>"
        changed_paths = PatchApprovalStore._collect_changed_paths(payload)
        if len(changed_paths) > 1:
            path = f"{len(changed_paths)} files"
        else:
            path = changed_paths[0] if changed_paths else str(payload.get("path", "(unknown path)")).strip() or "(unknown path)"
        continuation = payload.get("continuation")
        original_task = ""
        if isinstance(continuation, dict):
            original_task = str(continuation.get("original_task", "")).strip()

        base = (
            f"Continue the previously approved task after applying patch approval {approval_id} for {path}. "
            "The approved patch has already been applied. "
            "Do not recreate the same patch. "
            "Verify the applied change first, then finish any remaining work."
        )
        if continuation_context:
            verification_focus = str(continuation_context.get("verification_focus", "")).strip()
            if verification_focus:
                base = f"{base} Verification focus: {verification_focus}"
            commands = continuation_context.get("suggested_commands")
            if isinstance(commands, list) and commands:
                rendered_commands = []
                for command in commands[:2]:
                    if isinstance(command, list) and command:
                        rendered_commands.append(" ".join(str(part) for part in command))
                if rendered_commands:
                    base = f"{base} Suggested verification commands: {'; '.join(rendered_commands)}."
        if not original_task:
            return base

        return (
            f"{base} "
            f"Original goal: {original_task}"
        )

    @classmethod
    def build_continuation_context(
        cls,
        payload: dict[str, Any],
        *,
        workspace: Path,
    ) -> dict[str, Any]:
        changed_paths = cls._collect_changed_paths(payload)
        path = changed_paths[0] if len(changed_paths) == 1 else str(payload.get("path", "")).strip()
        source_tool = str(payload.get("source_tool", "")).strip()
        reason = str(payload.get("reason", "")).strip()
        approval_id = str(payload.get("approval_id", "")).strip()
        continuation = payload.get("continuation")
        original_task = ""
        requested_mode = "read_only"
        if isinstance(continuation, dict):
            original_task = str(continuation.get("original_task", "")).strip()
            requested_mode = str(continuation.get("requested_execution_mode", "")).strip() or requested_mode

        read_paths: list[str] = []
        for changed_path in changed_paths[:3]:
            if changed_path and changed_path not in read_paths:
                read_paths.append(changed_path)

        suggested_commands: list[list[str]] = []
        related_test_paths: list[Path] = []
        seen_related_tests: set[str] = set()
        for changed_path in changed_paths[:3]:
            related_test_path = cls.related_test_path_for(workspace, Path(changed_path)) if changed_path else None
            if related_test_path is None:
                continue
            normalized_test = str(related_test_path).replace("\\", "/")
            if normalized_test in seen_related_tests:
                continue
            seen_related_tests.add(normalized_test)
            related_test_paths.append(related_test_path)
            if normalized_test not in read_paths:
                read_paths.append(normalized_test)
            suggested_commands.append(["python", "-m", "unittest", cls.path_to_module(related_test_path)])

        verification_focus = cls._build_verification_focus(
            path=path,
            source_tool=source_tool,
            reason=reason,
            has_related_tests=bool(related_test_paths),
            change_count=max(len(changed_paths), 1),
        )

        return {
            "approval_id": approval_id,
            "path": path,
            "changed_paths": changed_paths,
            "source_tool": source_tool,
            "approval_reason": reason,
            "original_task": original_task,
            "requested_execution_mode": requested_mode,
            "verification_focus": verification_focus,
            "suggested_read_paths": read_paths,
            "suggested_commands": suggested_commands,
        }

    @staticmethod
    def _build_diff(
        *,
        path: str,
        before_text: str,
        after_text: str,
        before_exists: bool,
        after_exists: bool = True,
    ) -> str:
        fromfile = f"a/{path}" if before_exists else "/dev/null"
        tofile = f"b/{path}" if after_exists else "/dev/null"
        diff = difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
        rendered = "".join(diff).strip()
        return rendered or "(no textual diff)"

    @classmethod
    def _build_combined_diff(cls, changes: list[dict[str, Any]]) -> str:
        rendered = [
            cls._build_diff(
                path=str(change["path"]),
                before_text=str(change.get("before_text", "")),
                after_text=str(change.get("after_text", "")),
                before_exists=bool(change.get("before_exists", False)),
                after_exists=bool(change.get("after_exists", True)),
            )
            for change in changes
        ]
        return "\n\n".join(part for part in rendered if part and part != "(no textual diff)") or "(no textual diff)"

    @staticmethod
    def _build_verification_focus(
        *,
        path: str,
        source_tool: str,
        reason: str,
        has_related_tests: bool,
        change_count: int = 1,
    ) -> str:
        path_text = path or ("the changed files" if change_count > 1 else "the changed file")
        if source_tool == "replace_in_file":
            focus = f"Read {path_text} and confirm the approved replacement landed without disturbing nearby content."
        elif source_tool == "write_file":
            focus = f"Read {path_text} and confirm the approved contents landed as intended."
        elif source_tool == "verified_codex_handoff":
            focus = f"Read {path_text} and confirm the verified cloud patch landed cleanly before doing any follow-up work."
        else:
            focus = f"Read {path_text} and confirm the approved patch landed cleanly."
        if has_related_tests:
            focus = f"{focus} Run the directly related unit test next."
        if reason:
            focus = f"{focus} Original patch reason: {reason}."
        return focus

    @classmethod
    def related_test_path_for(cls, workspace: Path, target_path: Path) -> Path | None:
        return cls._find_related_test_path(workspace, target_path)

    @classmethod
    def path_to_module(cls, path: Path) -> str:
        return cls._path_to_module(path)

    @staticmethod
    def _find_related_test_path(workspace: Path, target_path: Path) -> Path | None:
        normalized = Path(str(target_path).replace("\\", "/"))
        if normalized.suffix != ".py":
            return None

        if normalized.parts[:1] == ("tests",):
            candidate = workspace / normalized
            return normalized if candidate.exists() else None

        if normalized.parts[:1] == ("teamai",):
            candidate = Path("tests") / f"test_{normalized.stem}.py"
            if (workspace / candidate).exists():
                return candidate
        elif len(normalized.parts) == 1:
            candidate = Path("tests") / f"test_{normalized.stem}.py"
            if (workspace / candidate).exists():
                return candidate
        return None

    @staticmethod
    def _path_to_module(path: Path) -> str:
        normalized = Path(str(path).replace("\\", "/"))
        return ".".join(normalized.with_suffix("").parts)

    @staticmethod
    def _normalize_change_payload(change: dict[str, Any]) -> dict[str, Any]:
        return {
            "path": str(change.get("path", "")).strip(),
            "before_exists": bool(change.get("before_exists", False)),
            "after_exists": bool(change.get("after_exists", True)),
            "before_text": str(change.get("before_text", "")),
            "after_text": str(change.get("after_text", "")),
        }

    @classmethod
    def _approval_changes(cls, payload: dict[str, Any]) -> list[dict[str, Any]]:
        raw_changes = payload.get("changes")
        if isinstance(raw_changes, list) and raw_changes:
            return [cls._normalize_change_payload(change) for change in raw_changes if isinstance(change, dict)]

        return [
            cls._normalize_change_payload(
                {
                    "path": payload.get("path", ""),
                    "before_exists": payload.get("before_exists", False),
                    "after_exists": payload.get("after_exists", True),
                    "before_text": payload.get("before_text", ""),
                    "after_text": payload.get("after_text", ""),
                }
            )
        ]

    @staticmethod
    def _collect_changed_paths(payload: dict[str, Any]) -> list[str]:
        raw_paths = payload.get("changed_paths")
        if isinstance(raw_paths, list) and raw_paths:
            return [str(path).strip() for path in raw_paths if str(path).strip()]

        path = str(payload.get("path", "")).strip()
        return [path] if path else []

    @staticmethod
    def _resolve_change_target(workspace: Path, relative_path: str) -> Path:
        target = (workspace / Path(relative_path)).resolve()
        if workspace.resolve() not in {target, *target.parents}:
            raise ValueError(f"Patch approval path escapes the workspace root: {target}")
        return target

    @classmethod
    def _changes_from_patch_target(
        cls,
        *,
        workspace: Path,
        sandbox_root: Path,
        before_path: str | None,
        after_path: str | None,
    ) -> list[dict[str, Any]]:
        if before_path and after_path and before_path != after_path:
            return [
                cls._build_change_from_paths(
                    workspace=workspace,
                    sandbox_root=sandbox_root,
                    path=before_path,
                    before_path=before_path,
                    after_path=None,
                ),
                cls._build_change_from_paths(
                    workspace=workspace,
                    sandbox_root=sandbox_root,
                    path=after_path,
                    before_path=None,
                    after_path=after_path,
                ),
            ]

        path = after_path or before_path
        if not path:
            return []
        return [
            cls._build_change_from_paths(
                workspace=workspace,
                sandbox_root=sandbox_root,
                path=path,
                before_path=before_path or path,
                after_path=after_path or path,
            )
        ]

    @staticmethod
    def _build_change_from_paths(
        *,
        workspace: Path,
        sandbox_root: Path,
        path: str,
        before_path: str | None,
        after_path: str | None,
    ) -> dict[str, Any]:
        before_exists = before_path is not None
        after_exists = after_path is not None
        before_text = ""
        after_text = ""

        if before_path is not None:
            before_target = workspace / before_path
            if before_target.exists():
                before_text = before_target.read_text(encoding="utf-8")
        if after_path is not None:
            after_target = sandbox_root / after_path
            if after_target.exists():
                after_text = after_target.read_text(encoding="utf-8")

        return {
            "path": path,
            "before_exists": before_exists,
            "after_exists": after_exists,
            "before_text": before_text,
            "after_text": after_text,
        }

    def _write_record(self, workspace: Path, payload: dict[str, Any]) -> None:
        directory = self._approvals_dir(workspace)
        directory.mkdir(parents=True, exist_ok=True)
        path = self._approval_path(workspace, str(payload["approval_id"]))
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _approval_path(self, workspace: Path, approval_id: str) -> Path:
        return self._approvals_dir(workspace) / f"{approval_id}.json"

    @staticmethod
    def _approvals_dir(workspace: Path) -> Path:
        return workspace / STATE_DIR_NAME / APPROVALS_DIR_NAME
