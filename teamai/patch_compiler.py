"""
Deterministic patch compiler for teamAI.

Extracted from supervisor.py.  Given a task string and workspace context,
this module attempts to *compile* explicit write requests into concrete
``ToolAction`` patches without invoking the local model at all.  The
compiler is tried early in the routing pipeline; when it produces a
result, the supervisor can skip the full council loop and go directly
to approval-gated application.

The compiler operates on a chain-of-compilers pattern: each
``_compile_*`` function tries one specific write idiom (import insert,
assignment update, anchor-relative insert, etc.).  The first one that
matches wins.  All compilers are pure — they read the task text and
workspace files but never mutate anything.

Usage::

    from teamai.patch_compiler import DeterministicPatchCompiler

    compiler = DeterministicPatchCompiler()
    action = compiler.compile(task=task_text, workspace=workspace)
    if action is not None:
        # skip the model — apply or queue for approval
"""
from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path
from typing import Any, Sequence

from .schemas import RoundRecord, ToolAction, ToolExecutionResult


class DeterministicPatchCompiler:
    """Compiles explicit write tasks into ``ToolAction`` patches
    without invoking a model.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compile(self, *, task: str, workspace: Path) -> ToolAction | None:
        """Try to compile *task* into a deterministic patch.

        Returns ``None`` when the task doesn't match any known idiom.
        """
        bundled = self._compile_python_function_and_unittest_bundle_action(
            task=task, workspace=workspace,
        )
        if bundled is not None:
            return bundled

        target_path = self._extract_primary_file_target(task, workspace)
        if target_path is None:
            return None

        target = (workspace / target_path).resolve()
        try:
            file_text = target.read_text(encoding="utf-8")
        except Exception:
            return None

        for compiler in (
            self._compile_paragraph_insert_action,
            self._compile_assignment_update_action,
            self._compile_import_insert_action,
            self._compile_module_docstring_action,
            self._compile_test_method_insert_action,
            self._compile_replace_all_action,
            self._compile_exact_replace_action,
            self._compile_anchor_insert_action,
            self._compile_append_action,
        ):
            compiled = compiler(task=task, target_path=target_path, file_text=file_text)
            if compiled is not None:
                return compiled
        return None

    def heuristic_write_action(
        self,
        *,
        task: str,
        workspace: Path,
        previous_rounds: list[RoundRecord],
        execution_mode: str,
    ) -> ToolAction | None:
        """Build a write action from a task that didn't match a full compile."""
        from .task_classifier import is_explicit_write_task

        if execution_mode != "workspace_write" or not is_explicit_write_task(task):
            return None

        compiled = self.compile(task=task, workspace=workspace)
        if compiled is not None:
            return compiled

        target_path = self._extract_primary_file_target(task, workspace)
        if target_path is None:
            return None

        contents = self._collect_read_file_outputs(previous_rounds, workspace)
        raw_file_text = contents.get(target_path)
        if not raw_file_text:
            return None

        file_text = self._strip_read_file_line_numbers(raw_file_text)
        sentence = self._extract_task_sentence(task)
        anchor = self._extract_task_anchor(task)
        if not sentence or not anchor:
            return None

        paragraph = self._find_paragraph_starting_with(file_text, anchor)
        if paragraph is None or sentence in paragraph or sentence in file_text:
            return None

        return ToolAction(
            tool="replace_in_file",
            reason="Propose the explicitly requested patch approval for the target file.",
            args={
                "path": target_path,
                "old_text": paragraph,
                "new_text": f"{paragraph}\n\n{sentence}",
                "replace_all": False,
            },
        )

    def action_matches_explicit_write_task(
        self,
        action: ToolAction,
        *,
        task: str,
        workspace: Path,
    ) -> bool:
        """True if *action* is the patch the compiler would produce for *task*."""
        if action.tool not in {"write_file", "replace_in_file"}:
            return False

        expected_action = self.compile(task=task, workspace=workspace)
        if expected_action is not None:
            return self._write_actions_match(action, expected_action, workspace)

        target_path = self._extract_primary_file_target(task, workspace)
        normalized_path = self._normalize_path_arg(action.args.get("path", "."), workspace)
        if target_path is not None and normalized_path != target_path:
            return False

        sentence = self._extract_task_sentence(task)
        if not sentence:
            return True

        if action.tool == "write_file":
            return sentence in str(action.args.get("content", ""))

        new_text = str(action.args.get("new_text", ""))
        return sentence in new_text

    # ------------------------------------------------------------------
    # Individual compilers
    # ------------------------------------------------------------------

    def _compile_python_function_and_unittest_bundle_action(
        self,
        *,
        task: str,
        workspace: Path,
    ) -> ToolAction | None:
        targets = self._extract_file_targets(task, workspace)
        if len(targets) < 2:
            return None

        source_path = next(
            (path for path in targets if path.endswith(".py") and "test" not in Path(path).name.lower()),
            None,
        )
        test_path = next((path for path in targets if path.endswith(".py") and "test" in path.lower()), None)
        if source_path is None or test_path is None:
            return None

        function_name = self._extract_function_update_target(task)
        return_template = self._extract_python_string_normalizer_template(task)
        expectation = self._extract_unittest_io_expectation(task)
        if function_name is None or return_template is None or expectation is None:
            return None

        try:
            source_text = (workspace / source_path).read_text(encoding="utf-8")
            test_text = (workspace / test_path).read_text(encoding="utf-8")
        except Exception:
            return None

        updated_source = self._build_python_function_return_update(
            file_text=source_text,
            function_name=function_name,
            return_template=return_template,
        )
        updated_test = self._build_python_unittest_expectation_update(
            file_text=test_text,
            function_name=function_name,
            input_value=expectation[0],
            output_value=expectation[1],
        )

        changes: list[dict[str, str]] = []
        if updated_source is not None:
            changes.append({"path": source_path, "content": updated_source})
        if updated_test is not None:
            changes.append({"path": test_path, "content": updated_test})
        if not changes:
            return None

        return ToolAction(
            tool="write_file",
            reason="Compile the explicit multi-file code-and-test request into a bundled patch approval.",
            args={"changes": changes},
        )

    def _compile_paragraph_insert_action(
        self, *, task: str, target_path: str, file_text: str,
    ) -> ToolAction | None:
        if "paragraph" not in task.lower():
            return None
        payload = self._extract_task_payload(task)
        anchor = self._extract_task_anchor(task)
        if payload is None or anchor is None:
            return None
        _, inserted_text = payload
        paragraph = self._find_paragraph_starting_with(file_text, anchor)
        if paragraph is None:
            return None
        updated_paragraph = f"{paragraph}\n\n{inserted_text}"
        if updated_paragraph in file_text or inserted_text in paragraph:
            return None
        return ToolAction(
            tool="replace_in_file",
            reason="Compile the explicit paragraph insertion into a deterministic patch approval.",
            args={"path": target_path, "old_text": paragraph, "new_text": updated_paragraph, "replace_all": False},
        )

    def _compile_assignment_update_action(
        self, *, task: str, target_path: str, file_text: str,
    ) -> ToolAction | None:
        assignment = self._extract_assignment_update_values(task)
        if assignment is None:
            return None
        key, raw_value = assignment
        updated_text = self._build_assignment_updated_file_text(
            file_text=file_text, target_path=target_path, key=key, raw_value=raw_value,
        )
        if updated_text is None or updated_text == file_text:
            return None
        return ToolAction(
            tool="write_file",
            reason="Compile the explicit assignment update into a deterministic patch approval.",
            args={"path": target_path, "content": updated_text},
        )

    def _compile_import_insert_action(
        self, *, task: str, target_path: str, file_text: str,
    ) -> ToolAction | None:
        if not target_path.endswith(".py"):
            return None
        import_statement = self._extract_import_statement(task)
        if import_statement is None:
            return None
        updated_text = self._build_python_import_inserted_text(
            file_text=file_text, import_statement=import_statement,
        )
        if updated_text is None or updated_text == file_text:
            return None
        return ToolAction(
            tool="write_file",
            reason="Compile the explicit import insertion into a deterministic patch approval.",
            args={"path": target_path, "content": updated_text},
        )

    def _compile_test_method_insert_action(
        self, *, task: str, target_path: str, file_text: str,
    ) -> ToolAction | None:
        if not target_path.endswith(".py"):
            return None
        class_insert = self._extract_class_block_insert_values(task)
        if class_insert is None:
            return None
        class_name, block_text = class_insert
        updated_text = self._build_class_block_inserted_text(
            file_text=file_text, class_name=class_name, block_text=block_text,
        )
        if updated_text is None or updated_text == file_text:
            return None
        return ToolAction(
            tool="write_file",
            reason="Compile the explicit test-class insertion into a deterministic patch approval.",
            args={"path": target_path, "content": updated_text},
        )

    def _compile_module_docstring_action(
        self, *, task: str, target_path: str, file_text: str,
    ) -> ToolAction | None:
        if not target_path.endswith(".py"):
            return None
        lowered = task.lower()
        if "docstring" not in lowered:
            return None
        if any(scope in lowered for scope in (
            "function docstring", "method docstring", "class docstring",
            "docstring to function", "docstring to method", "docstring to class",
            "docstring for function", "docstring for method", "docstring for class",
        )):
            return None
        try:
            tree = ast.parse(file_text)
        except SyntaxError:
            return None
        if ast.get_docstring(tree) is not None:
            return None
        content = self._extract_docstring_content(task)
        if content is None:
            stem = Path(target_path).with_suffix("").as_posix().replace("/", ".")
            content = f"{stem} module."
        if '"""' in content:
            return None
        updated_text = self._build_module_docstring_inserted_text(
            file_text=file_text, content=content,
        )
        if updated_text is None or updated_text == file_text:
            return None
        return ToolAction(
            tool="write_file",
            reason="Compile the explicit module-docstring insertion into a deterministic patch approval.",
            args={"path": target_path, "content": updated_text},
        )

    def _compile_exact_replace_action(
        self, *, task: str, target_path: str, file_text: str,
    ) -> ToolAction | None:
        replace_values = self._extract_task_replace_values(task)
        if replace_values is None:
            return None
        old_text, new_text, replace_all = replace_values
        if replace_all or not old_text or old_text == new_text or old_text not in file_text:
            return None
        return ToolAction(
            tool="replace_in_file",
            reason="Compile the explicit replace request into a deterministic patch approval.",
            args={"path": target_path, "old_text": old_text, "new_text": new_text, "replace_all": False},
        )

    def _compile_replace_all_action(
        self, *, task: str, target_path: str, file_text: str,
    ) -> ToolAction | None:
        replace_values = self._extract_task_replace_values(task)
        if replace_values is None:
            return None
        old_text, new_text, replace_all = replace_values
        if not replace_all or not old_text or old_text == new_text:
            return None
        if file_text.count(old_text) < 1:
            return None
        return ToolAction(
            tool="replace_in_file",
            reason="Compile the explicit replace-all request into a deterministic patch approval.",
            args={"path": target_path, "old_text": old_text, "new_text": new_text, "replace_all": True},
        )

    def _compile_anchor_insert_action(
        self, *, task: str, target_path: str, file_text: str,
    ) -> ToolAction | None:
        anchor_insert = self._extract_anchor_insert_values(task)
        if anchor_insert is None:
            return None
        kind, inserted_text, position, anchor = anchor_insert
        if anchor not in file_text:
            return None
        delimiter = self._insertion_delimiter(kind, inserted_text)
        if position == "before":
            replacement = f"{inserted_text}{delimiter}{anchor}"
        else:
            replacement = f"{anchor}{delimiter}{inserted_text}"
        if replacement in file_text:
            return None
        return ToolAction(
            tool="replace_in_file",
            reason="Compile the explicit anchored insertion into a deterministic patch approval.",
            args={"path": target_path, "old_text": anchor, "new_text": replacement, "replace_all": False},
        )

    def _compile_append_action(
        self, *, task: str, target_path: str, file_text: str,
    ) -> ToolAction | None:
        append_values = self._extract_append_values(task)
        if append_values is None:
            return None
        kind, appended_text = append_values
        updated_text = self._build_appended_file_text(file_text=file_text, appended_text=appended_text, kind=kind)
        if updated_text is None or updated_text == file_text:
            return None
        return ToolAction(
            tool="write_file",
            reason="Compile the explicit append request into a deterministic patch approval.",
            args={"path": target_path, "content": updated_text},
        )

    # ------------------------------------------------------------------
    # Action matching
    # ------------------------------------------------------------------

    def _write_actions_match(
        self,
        action: ToolAction,
        expected_action: ToolAction,
        workspace: Path,
    ) -> bool:
        if action.tool != expected_action.tool:
            return False
        actual_path = self._normalize_path_arg(action.args.get("path", "."), workspace)
        expected_path = self._normalize_path_arg(expected_action.args.get("path", "."), workspace)
        if actual_path != expected_path:
            return False
        if action.tool == "write_file":
            expected_bundle = self._normalize_write_bundle(expected_action.args, workspace)
            actual_bundle = self._normalize_write_bundle(action.args, workspace)
            if expected_bundle is not None or actual_bundle is not None:
                return actual_bundle == expected_bundle
            return str(action.args.get("content", "")) == str(expected_action.args.get("content", ""))
        return (
            str(action.args.get("old_text", "")) == str(expected_action.args.get("old_text", ""))
            and str(action.args.get("new_text", "")) == str(expected_action.args.get("new_text", ""))
            and bool(action.args.get("replace_all", False)) == bool(expected_action.args.get("replace_all", False))
        )

    def _normalize_write_bundle(
        self, args: dict[str, object], workspace: Path,
    ) -> list[tuple[str, str]] | None:
        raw_changes = args.get("changes")
        if not isinstance(raw_changes, list):
            return None
        normalized: list[tuple[str, str]] = []
        for entry in raw_changes:
            if not isinstance(entry, dict):
                continue
            path = self._normalize_path_arg(entry.get("path", "."), workspace)
            normalized.append((path, str(entry.get("content", ""))))
        return normalized

    # ------------------------------------------------------------------
    # Path extraction
    # ------------------------------------------------------------------

    def _extract_file_targets(self, task: str, workspace: Path) -> list[str]:
        targets: list[str] = []
        for candidate in self._extract_candidate_paths(task):
            normalized = self._normalize_path_arg(candidate, workspace)
            resolved = (workspace / normalized).resolve()
            if resolved.exists() and resolved.is_file() and normalized not in targets:
                targets.append(normalized)
        return targets

    def _extract_primary_file_target(self, task: str, workspace: Path) -> str | None:
        targets = self._extract_file_targets(task, workspace)
        return targets[0] if targets else None

    @staticmethod
    def _extract_candidate_paths(text: str) -> list[str]:
        patterns = [
            re.compile(r"`([^`]+)`"),
            re.compile(r"'([^']+)'"),
            re.compile(r'"([^"]+)"'),
        ]
        candidates: list[str] = []
        for pattern in patterns:
            for match in pattern.finditer(text):
                value = match.group(1).strip()
                if not value:
                    continue
                if "/" in value or "." in value:
                    candidates.append(value)

        unquoted_path_pattern = re.compile(
            r"(?<![`'\"\w])"
            r"((?:\.[A-Za-z0-9_-]+|[A-Za-z0-9_-]+)(?:/[A-Za-z0-9._-]+)+/?|(?:\.[A-Za-z0-9_-]+|[A-Za-z0-9_-]+)\.[A-Za-z0-9_-]+)"
        )
        for match in unquoted_path_pattern.finditer(text):
            candidates.append(match.group(1).strip())

        for common in [
            "README.md", "pyproject.toml", "setup.py", "PROJECT_MEMORY.md",
            ".env", ".env.example", "teamai/", "teamai/model_backend.py",
            "teamai/supervisor.py", "teamai/tools.py", "teamai/api.py",
            "teamai/cli.py", "tests/",
        ]:
            if common in text:
                candidates.append(common)

        ordered: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            normalized = candidate.rstrip(",.:;")
            if normalized not in seen:
                seen.add(normalized)
                ordered.append(normalized)
        return ordered

    @staticmethod
    def _normalize_path_arg(candidate: object, workspace: Path) -> str:
        raw = str(candidate or ".").strip()
        if raw not in {".", "/"}:
            raw = raw.rstrip("/") or "."
        path = Path(raw).expanduser()
        try:
            resolved = path.resolve() if path.is_absolute() else (workspace / path).resolve()
            return str(resolved.relative_to(workspace))
        except Exception:
            return raw

    # ------------------------------------------------------------------
    # Task text extractors
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_function_update_target(task: str) -> str | None:
        match = re.search(r"\bupdate\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+so it\b", task, flags=re.IGNORECASE)
        return match.group("name").strip() if match else None

    @staticmethod
    def _extract_python_string_normalizer_template(task: str) -> str | None:
        lowered = task.lower()
        trims = any(m in lowered for m in ("trim whitespace", "trims whitespace", "trim surrounding whitespace"))
        titles = any(
            m in lowered
            for m in ("title-cases each word", "title-case each word", "title cases each word", "title case each word")
        )
        if trims and titles:
            return '" ".join(part.capitalize() for part in {param}.split())'
        return None

    @staticmethod
    def _extract_unittest_io_expectation(task: str) -> tuple[str, str] | None:
        match = re.search(
            r"proves?\s+(?P<input_quote>['\"`])(?P<input>.+?)(?P=input_quote)\s+becomes\s+"
            r"(?P<output_quote>['\"`])(?P<output>.+?)(?P=output_quote)",
            task, flags=re.IGNORECASE,
        )
        return (match.group("input"), match.group("output")) if match else None

    @staticmethod
    def _extract_task_sentence(task: str) -> str | None:
        match = re.search(r"(?:sentence|line|text)\s+['\"]([^'\"]+)['\"]", task, flags=re.IGNORECASE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_task_payload(task: str) -> tuple[str, str] | None:
        fenced_block = DeterministicPatchCompiler._extract_task_fenced_block(task)
        if fenced_block is not None:
            return "block", fenced_block
        match = re.search(
            r"(sentence|line|text)\s+(?P<quote>['\"`])(?P<payload>.+?)(?P=quote)",
            task, flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).lower(), match.group("payload").strip()
        match = re.search(
            r"(?:append|insert|add)\s+(?:the\s+)?(?:exact\s+|literal\s+|verbatim\s+)?"
            r"(?P<quote>['\"`])(?P<payload>.+?)(?P=quote)",
            task, flags=re.IGNORECASE,
        )
        if match:
            return "text", match.group("payload").strip()
        return None

    @staticmethod
    def _extract_task_replace_values(task: str) -> tuple[str, str, bool] | None:
        match = re.search(
            r"replace(?P<all>\s+all(?:\s+occurrences?)?\s+of|\s+every\s+occurrence\s+of)?(?:\s+the\s+(?:text|line|sentence))?\s+"
            r"(?P<old_quote>['\"`])(?P<old>.+?)(?P=old_quote)\s+with\s+"
            r"(?P<new_quote>['\"`])(?P<new>.+?)(?P=new_quote)",
            task, flags=re.IGNORECASE,
        )
        if match:
            return match.group("old").strip(), match.group("new").strip(), bool(match.group("all"))
        return None

    @staticmethod
    def _extract_import_statement(task: str) -> str | None:
        match = re.search(
            r"(?:add|insert)\s+(?:the\s+)?(?:(?:import|statement)\s+)?"
            r"(?P<quote>['\"`])(?P<statement>(?:from\s+[^\n]+?\s+import\s+[^\n]+?|import\s+[^\n]+?))(?P=quote)\s+"
            r"(?:to|into)\b",
            task, flags=re.IGNORECASE,
        )
        return match.group("statement").strip() if match else None

    @staticmethod
    def _extract_assignment_update_values(task: str) -> tuple[str, str] | None:
        match = re.search(
            r"\b(?:set|change|update)\s+(?P<name>[A-Za-z_][\w.-]*)\s*(?:=|to)\s*(?P<value>.+?)\s+(?:in|inside)\s+[\w./-]+\b",
            task, flags=re.IGNORECASE,
        )
        if not match:
            return None
        return match.group("name").strip(), match.group("value").strip().rstrip(".,")

    @staticmethod
    def _extract_anchor_insert_values(task: str) -> tuple[str, str, str, str] | None:
        fenced_block = DeterministicPatchCompiler._extract_task_fenced_block(task)
        if fenced_block is not None:
            match = re.search(
                r"(?:insert|add)\s+(?:the\s+)?(?:following\s+)?block\s+"
                r"(?:immediately\s+)?(?P<position>before|after)\s+(?:the\s+)?"
                r"(?:(?:sentence|line|text|block)\s+)?(?P<anchor_quote>['\"`])(?P<anchor>.+?)(?P=anchor_quote)",
                task, flags=re.IGNORECASE | re.DOTALL,
            )
            if match:
                return "block", fenced_block, match.group("position").lower(), match.group("anchor").strip()

        match = re.search(
            r"(?:insert|add)\s+(?:the\s+)?(?:(?P<kind>sentence|line|text)\s+)?"
            r"(?P<payload_quote>['\"`])(?P<payload>.+?)(?P=payload_quote)\s+"
            r"(?:immediately\s+)?(?P<position>before|after)\s+(?:the\s+)?"
            r"(?:(?:sentence|line|text)\s+)?(?P<anchor_quote>['\"`])(?P<anchor>.+?)(?P=anchor_quote)",
            task, flags=re.IGNORECASE,
        )
        if match:
            kind = (match.group("kind") or "text").lower()
            return kind, match.group("payload").strip(), match.group("position").lower(), match.group("anchor").strip()
        return None

    @staticmethod
    def _extract_class_block_insert_values(task: str) -> tuple[str, str] | None:
        fenced_block = DeterministicPatchCompiler._extract_task_fenced_block(task)
        if fenced_block is None:
            return None
        match = re.search(
            r"(?:add|insert)\s+(?:the\s+)?(?:following\s+)?(?:test|method|block)\s+to\s+class\s+(?P<class_name>[A-Za-z_][A-Za-z0-9_]*)\b",
            task, flags=re.IGNORECASE | re.DOTALL,
        )
        return (match.group("class_name").strip(), fenced_block) if match else None

    @staticmethod
    def _extract_append_values(task: str) -> tuple[str, str] | None:
        fenced_block = DeterministicPatchCompiler._extract_task_fenced_block(task)
        if fenced_block is not None:
            match = re.search(
                r"append\s+(?:the\s+)?(?:following\s+)?block\s+(?:to|at\s+the\s+end\s+of)\b",
                task, flags=re.IGNORECASE | re.DOTALL,
            )
            if match:
                return "block", fenced_block
        match = re.search(
            r"append\s+(?:the\s+)?(?:exact\s+|literal\s+|verbatim\s+)?(?:(?P<kind>sentence|line|text)\s+)?"
            r"(?P<quote>['\"`])(?P<payload>.+?)(?P=quote)\s+"
            r"(?:to|at\s+the\s+end\s+of)\b",
            task, flags=re.IGNORECASE,
        )
        if match:
            return (match.group("kind") or "text").lower(), match.group("payload").strip()
        return None

    @staticmethod
    def _extract_task_fenced_block(task: str) -> str | None:
        match = re.search(r"```(?:[\w.+-]+)?\n(?P<block>[\s\S]+?)```", task, flags=re.MULTILINE)
        return match.group("block").rstrip("\n") if match else None

    @staticmethod
    def _extract_docstring_content(task: str) -> str | None:
        fenced = DeterministicPatchCompiler._extract_task_fenced_block(task)
        if fenced:
            return fenced.strip()
        match = re.search(
            r"docstring\s+(?:saying|that\s+says|with\s+(?:text|content)|reading)\s+"
            r"(?P<quote>['\"`])(?P<content>.+?)(?P=quote)",
            task, flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group("content").strip()
        match = re.search(
            r"(?:saying|that\s+says|reading|with\s+(?:text|content))\s+"
            r"(?P<quote>['\"`])(?P<content>.+?)(?P=quote)",
            task, flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group("content").strip()
        match = re.search(
            r"docstring\s+(?P<quote>['\"`])(?P<content>.+?)(?P=quote)",
            task, flags=re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group("content").strip()
        return None

    @staticmethod
    def _extract_task_anchor(task: str) -> str | None:
        match = re.search(r"(?:starts|begins)\s+with\s+['\"]([^'\"]+)['\"]", task, flags=re.IGNORECASE)
        return match.group(1).strip() if match else None

    # ------------------------------------------------------------------
    # Text assembly helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_python_function_return_update(
        *, file_text: str, function_name: str, return_template: str,
    ) -> str | None:
        lines = file_text.splitlines(keepends=True)
        function_pattern = re.compile(
            rf"^(?P<indent>\s*)def\s+{re.escape(function_name)}\s*\((?P<params>[^)]*)\)\s*(?:->\s*[^:]+)?:\s*$"
        )
        for index, line in enumerate(lines):
            match = function_pattern.match(line.rstrip("\n"))
            if not match:
                continue
            indent = match.group("indent")
            params = match.group("params")
            first_param = params.split(",", 1)[0].strip() if params.strip() else "value"
            param_name = first_param.split(":", 1)[0].split("=", 1)[0].strip() or "value"
            body_start = index + 1
            body_end = len(lines)
            for candidate_index in range(body_start, len(lines)):
                stripped = lines[candidate_index].strip()
                if not stripped:
                    continue
                current_indent = len(lines[candidate_index]) - len(lines[candidate_index].lstrip(" "))
                if current_indent <= len(indent):
                    body_end = candidate_index
                    break
            body_indent = f"{indent}    "
            new_body_line = f"{body_indent}return {return_template.format(param=param_name)}\n"
            existing_body = "".join(lines[body_start:body_end]).strip()
            if existing_body == new_body_line.strip():
                return None
            return "".join([*lines[:body_start], new_body_line, *lines[body_end:]])
        return None

    def _build_python_unittest_expectation_update(
        self, *, file_text: str, function_name: str, input_value: str, output_value: str,
    ) -> str | None:
        assertion = f"self.assertEqual({function_name}({input_value!r}), {output_value!r})"
        if assertion in file_text:
            return None
        class_match = re.search(
            r"^class\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?:.*\.)?TestCase\):\s*$",
            file_text, flags=re.MULTILINE,
        )
        if class_match is None:
            return None
        method_name = f"test_{function_name}_normalizes_whitespace_and_title_case"
        block_text = f"def {method_name}(self) -> None:\n    {assertion}"
        return self._build_class_block_inserted_text(
            file_text=file_text, class_name=class_match.group("name"), block_text=block_text,
        )

    @staticmethod
    def _build_appended_file_text(*, file_text: str, appended_text: str, kind: str) -> str | None:
        if kind == "block":
            normalized_appended = appended_text.rstrip("\n")
            if normalized_appended and normalized_appended in file_text:
                return None
            separator = ""
            if file_text and not file_text.endswith("\n"):
                separator = "\n"
            elif file_text and not file_text.endswith("\n\n"):
                separator = "\n"
            updated = f"{file_text}{separator}{normalized_appended}"
            return updated if updated.endswith("\n") else f"{updated}\n"
        if kind == "line":
            if file_text.rstrip("\n").endswith(appended_text):
                return None
            prefix = "" if not file_text or file_text.endswith("\n") else "\n"
            updated = f"{file_text}{prefix}{appended_text}"
            return updated if updated.endswith("\n") else f"{updated}\n"
        if kind == "sentence":
            if file_text.rstrip().endswith(appended_text):
                return None
            if not file_text:
                return appended_text
            separator = "\n" if file_text.endswith("\n") else " "
            return f"{file_text}{separator}{appended_text}"
        if file_text.endswith(appended_text):
            return None
        return f"{file_text}{appended_text}"

    @staticmethod
    def _build_module_docstring_inserted_text(*, file_text: str, content: str) -> str | None:
        if not content:
            return None
        if "\n" in content:
            body = content.strip("\n").rstrip()
            docstring = f'"""\n{body}\n"""'
        else:
            docstring = f'"""{content}"""'

        lines = file_text.splitlines(keepends=True)
        insert_at = 0
        if insert_at < len(lines) and lines[insert_at].startswith("#!"):
            insert_at += 1
        if insert_at < len(lines) and re.search(r"coding[:=]", lines[insert_at]):
            insert_at += 1

        prefix_lines = lines[:insert_at]
        while prefix_lines and not prefix_lines[-1].strip():
            prefix_lines.pop()

        trailing = lines[len(prefix_lines):]
        suffix_start = 0
        while suffix_start < len(trailing) and not trailing[suffix_start].strip():
            suffix_start += 1
        suffix_lines = trailing[suffix_start:]

        prefix = "".join(prefix_lines)
        if prefix and not prefix.endswith("\n"):
            prefix = f"{prefix}\n"
        suffix = "".join(suffix_lines)

        if suffix:
            return f"{prefix}{docstring}\n\n{suffix}"
        return f"{prefix}{docstring}\n"

    @staticmethod
    def _build_python_import_inserted_text(*, file_text: str, import_statement: str) -> str | None:
        lines = file_text.splitlines(keepends=True)
        normalized_statement = import_statement.strip()
        existing_lines = {line.strip() for line in lines}
        if normalized_statement in existing_lines:
            return None

        insert_at = 0
        if insert_at < len(lines) and lines[insert_at].startswith("#!"):
            insert_at += 1
        if insert_at < len(lines) and re.search(r"coding[:=]", lines[insert_at]):
            insert_at += 1
        while insert_at < len(lines) and not lines[insert_at].strip():
            insert_at += 1
        if insert_at < len(lines):
            stripped = lines[insert_at].lstrip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                quote = stripped[:3]
                insert_at += 1
                if quote not in stripped[3:]:
                    while insert_at < len(lines):
                        if quote in lines[insert_at]:
                            insert_at += 1
                            break
                        insert_at += 1
                while insert_at < len(lines) and not lines[insert_at].strip():
                    insert_at += 1

        import_insert_at: int | None = None
        last_import = -1
        scan_index = insert_at
        while scan_index < len(lines):
            stripped = lines[scan_index].strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                last_import = scan_index
                scan_index += 1
                continue
            if last_import >= 0 and (not stripped or stripped.startswith("#")):
                scan_index += 1
                continue
            break
        if last_import >= 0:
            import_insert_at = last_import + 1

        insertion_index = import_insert_at if import_insert_at is not None else insert_at
        inserted_lines = [f"{normalized_statement}\n"]
        if import_insert_at is None and insertion_index < len(lines) and lines[insertion_index].strip():
            inserted_lines.append("\n")
        return "".join(lines[:insertion_index] + inserted_lines + lines[insertion_index:])

    @staticmethod
    def _build_assignment_updated_file_text(
        *, file_text: str, target_path: str, key: str, raw_value: str,
    ) -> str | None:
        suffix = Path(target_path).suffix.lower()
        is_env_style = target_path.startswith(".env") or suffix == ".env"
        separators = ["="] if is_env_style else ([":"] if suffix in {".yaml", ".yml"} else ["=", ":"])

        lines = file_text.splitlines(keepends=True)
        for index, line in enumerate(lines):
            line_without_newline = line.rstrip("\n")
            for separator in separators:
                pattern = re.compile(
                    rf"^(?P<prefix>\s*{re.escape(key)}\s*{re.escape(separator)}\s*)(?P<value>.*?)(?P<comment>\s+#.*)?$"
                )
                match = pattern.match(line_without_newline)
                if not match:
                    continue
                existing_value = match.group("value").rstrip()
                replacement_value = DeterministicPatchCompiler._normalize_assignment_value(
                    raw_value=raw_value, existing_value=existing_value, separator=separator, target_path=target_path,
                )
                updated_line = f"{match.group('prefix')}{replacement_value}{match.group('comment') or ''}"
                newline = "\n" if line.endswith("\n") else ""
                new_lines = lines[:]
                new_lines[index] = f"{updated_line}{newline}"
                return "".join(new_lines)

        if is_env_style:
            replacement_value = DeterministicPatchCompiler._normalize_assignment_value(
                raw_value=raw_value, existing_value="", separator="=", target_path=target_path,
            )
            appended_line = f"{key}={replacement_value}"
            existing_lines = {line.strip() for line in file_text.splitlines()}
            if appended_line in existing_lines:
                return None
            sep = "" if not file_text or file_text.endswith("\n") else "\n"
            updated_text = f"{file_text}{sep}{appended_line}"
            return updated_text if updated_text.endswith("\n") else f"{updated_text}\n"
        return None

    @staticmethod
    def _normalize_assignment_value(
        *, raw_value: str, existing_value: str, separator: str, target_path: str,
    ) -> str:
        candidate = raw_value.strip()
        if not candidate:
            return candidate
        if (candidate.startswith('"') and candidate.endswith('"')) or (candidate.startswith("'") and candidate.endswith("'")):
            return candidate
        if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", candidate):
            return candidate
        lowered = candidate.lower()
        if lowered in {"true", "false", "none", "null"}:
            if separator == "=" and target_path.endswith(".py"):
                return lowered.capitalize() if lowered in {"true", "false", "none"} else "None"
            return lowered
        if existing_value[:1] in {'"', "'"} and existing_value[-1:] == existing_value[:1]:
            quote = existing_value[:1]
            return f"{quote}{candidate}{quote}"
        return candidate

    @staticmethod
    def _build_class_block_inserted_text(
        *, file_text: str, class_name: str, block_text: str,
    ) -> str | None:
        lines = file_text.splitlines(keepends=True)
        class_pattern = re.compile(rf"^(?P<indent>\s*)class\s+{re.escape(class_name)}\b.*:\s*$")
        class_index = -1
        class_indent = ""
        for index, line in enumerate(lines):
            match = class_pattern.match(line.rstrip("\n"))
            if match:
                class_index = index
                class_indent = match.group("indent")
                break
        if class_index < 0:
            return None

        class_end = len(lines)
        for index in range(class_index + 1, len(lines)):
            stripped = lines[index].strip()
            if not stripped:
                continue
            current_indent = len(lines[index]) - len(lines[index].lstrip(" "))
            if current_indent <= len(class_indent):
                class_end = index
                break

        dedented_block = textwrap.dedent(block_text).strip("\n")
        if not dedented_block:
            return None
        body_indent = f"{class_indent}    "
        normalized_lines = []
        for line in dedented_block.splitlines():
            if line.strip():
                normalized_lines.append(f"{body_indent}{line.rstrip()}\n")
            else:
                normalized_lines.append("\n")
        normalized_block = "".join(normalized_lines).rstrip("\n")
        class_slice = "".join(lines[class_index:class_end])
        if normalized_block in class_slice:
            return None
        prefix = "".join(lines[:class_end])
        suffix = "".join(lines[class_end:])
        separator_before = ""
        if prefix and not prefix.endswith("\n\n"):
            separator_before = "\n" if prefix.endswith("\n") else "\n\n"
        separator_after = ""
        if suffix and not suffix.startswith("\n"):
            separator_after = "\n"
        return f"{prefix}{separator_before}{normalized_block}\n{separator_after}{suffix}"

    # ------------------------------------------------------------------
    # Misc text helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_paragraph_starting_with(text: str, anchor: str) -> str | None:
        normalized_anchor = DeterministicPatchCompiler._normalize_paragraph_anchor(anchor)
        for paragraph in re.split(r"\n\s*\n", text):
            candidate = paragraph.strip()
            if DeterministicPatchCompiler._normalize_paragraph_anchor(candidate).startswith(normalized_anchor):
                return candidate
        return None

    @staticmethod
    def _normalize_paragraph_anchor(text: str) -> str:
        return re.sub(r"\s+", " ", text.replace("`", " ").strip().lower()).strip()

    @staticmethod
    def _strip_read_file_line_numbers(text: str) -> str:
        return "\n".join(re.sub(r"^\d{4}:\s?", "", line) for line in text.splitlines())

    @staticmethod
    def _insertion_delimiter(kind: str, inserted_text: str) -> str:
        if "\n" in inserted_text or kind == "block":
            return "\n"
        if kind == "line":
            return "\n"
        if kind == "sentence":
            return " "
        return ""

    def _collect_read_file_outputs(
        self, rounds: list[RoundRecord], workspace: Path,
    ) -> dict[str, str]:
        outputs: dict[str, str] = {}
        for record in rounds:
            for action, result in zip(record.planner.actions, record.tool_results):
                if action.tool != "read_file" or not result.success:
                    continue
                path = self._normalize_path_arg(action.args.get("path", "."), workspace)
                outputs[path] = result.output
        return outputs
