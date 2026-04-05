from __future__ import annotations

import difflib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from .memory import STATE_DIR_NAME


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
        continuation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        approval_id = uuid4().hex[:12]
        relative_path = str(path)
        created_at = datetime.now(timezone.utc).isoformat()
        payload: dict[str, Any] = {
            "approval_id": approval_id,
            "created_at": created_at,
            "status": "pending",
            "path": relative_path,
            "before_exists": before_exists,
            "source_tool": source_tool,
            "reason": reason.strip(),
            "diff": self._build_diff(
                path=relative_path,
                before_text=before_text,
                after_text=after_text,
                before_exists=before_exists,
            ),
            "before_text": before_text,
            "after_text": after_text,
        }
        if continuation:
            payload["continuation"] = {
                **continuation,
                "target_path": relative_path,
                "source_tool": source_tool,
                "approval_reason": reason.strip(),
            }
        self._write_record(workspace, payload)
        return payload

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

        relative_path = Path(str(payload["path"]))
        target = (workspace / relative_path).resolve()
        if workspace.resolve() not in {target, *target.parents}:
            raise ValueError(f"Patch approval path escapes the workspace root: {target}")

        before_exists = bool(payload.get("before_exists", False))
        current_exists = target.exists()
        if before_exists != current_exists:
            payload["status"] = "stale"
            payload["stale_reason"] = "Target existence changed since the patch was proposed."
            self._write_record(workspace, payload)
            raise ValueError(f"Patch approval `{approval_id}` is stale because the target changed.")

        current_text = target.read_text(encoding="utf-8") if current_exists else ""
        before_text = str(payload.get("before_text", ""))
        if current_text != before_text:
            payload["status"] = "stale"
            payload["stale_reason"] = "Target contents changed since the patch was proposed."
            self._write_record(workspace, payload)
            raise ValueError(f"Patch approval `{approval_id}` is stale because the file contents changed.")

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(str(payload.get("after_text", "")), encoding="utf-8")
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
        path = str(payload.get("path", "(unknown path)")).strip() or "(unknown path)"
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
        path = str(payload.get("path", "")).strip()
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
        if path:
            read_paths.append(path)

        suggested_commands: list[list[str]] = []
        related_test_path = cls._find_related_test_path(workspace, Path(path)) if path else None
        if related_test_path is not None:
            related_test_text = str(related_test_path).replace("\\", "/")
            if related_test_text not in read_paths:
                read_paths.append(related_test_text)
            suggested_commands.append(["python", "-m", "unittest", cls._path_to_module(related_test_path)])

        verification_focus = cls._build_verification_focus(
            path=path,
            source_tool=source_tool,
            reason=reason,
            has_related_tests=related_test_path is not None,
        )

        return {
            "approval_id": approval_id,
            "path": path,
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
    ) -> str:
        fromfile = f"a/{path}" if before_exists else "/dev/null"
        tofile = f"b/{path}"
        diff = difflib.unified_diff(
            before_text.splitlines(keepends=True),
            after_text.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
        rendered = "".join(diff).strip()
        return rendered or "(no textual diff)"

    @staticmethod
    def _build_verification_focus(
        *,
        path: str,
        source_tool: str,
        reason: str,
        has_related_tests: bool,
    ) -> str:
        path_text = path or "the changed file"
        if source_tool == "replace_in_file":
            focus = f"Read {path_text} and confirm the approved replacement landed without disturbing nearby content."
        elif source_tool == "write_file":
            focus = f"Read {path_text} and confirm the approved contents landed as intended."
        else:
            focus = f"Read {path_text} and confirm the approved patch landed cleanly."
        if has_related_tests:
            focus = f"{focus} Run the directly related unit test next."
        if reason:
            focus = f"{focus} Original patch reason: {reason}."
        return focus

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
        return None

    @staticmethod
    def _path_to_module(path: Path) -> str:
        normalized = Path(str(path).replace("\\", "/"))
        return ".".join(normalized.with_suffix("").parts)

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
