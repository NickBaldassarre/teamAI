from __future__ import annotations

import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .approvals import PatchApprovalStore
from .patch_utils import PatchTarget, extract_patch_targets
from .sandbox import Sandbox, SandboxCommandResult


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    log_output: str
    patch_returncode: int
    test_returncode: int | None
    commands_run: tuple[str, ...] = field(default_factory=tuple)


def verify_patch(patch_file: Path, sandbox: Sandbox) -> VerificationResult:
    patch_path = patch_file.resolve()
    patch_text = patch_path.read_text(encoding="utf-8")
    patch_result = sandbox.run(f"patch -p1 < {shlex.quote(str(patch_path))}")
    if patch_result.returncode != 0:
        return VerificationResult(
            success=False,
            log_output=_format_verification_log(patch_result, []),
            patch_returncode=patch_result.returncode,
            test_returncode=None,
        )

    patch_targets = extract_patch_targets(patch_text)
    verification_commands = _build_verification_commands(sandbox.path, patch_targets)
    if not verification_commands:
        return VerificationResult(
            success=False,
            log_output=_format_verification_log(
                patch_result,
                [
                    (
                        "Verification Plan",
                        SandboxCommandResult(
                            command="(none)",
                            cwd=sandbox.path,
                            returncode=1,
                            stdout="",
                            stderr="No verification command could be inferred for this repository.",
                        ),
                    )
                ],
            ),
            patch_returncode=patch_result.returncode,
            test_returncode=None,
        )

    command_results: list[tuple[str, SandboxCommandResult]] = []
    commands_run: list[str] = []
    final_test_returncode: int | None = None
    success = True

    for index, command in enumerate(verification_commands):
        result = sandbox.run(command)
        title = "Test Run" if index == 0 else "Additional Verification"
        command_results.append((title, result))
        commands_run.append(" ".join(str(part) for part in command))
        final_test_returncode = result.returncode
        if result.returncode != 0:
            success = False
            break

    return VerificationResult(
        success=success,
        log_output=_format_verification_log(patch_result, command_results),
        patch_returncode=patch_result.returncode,
        test_returncode=final_test_returncode,
        commands_run=tuple(commands_run),
    )


def _build_verification_commands(
    workspace: Path,
    patch_targets: list[PatchTarget],
) -> list[list[str]]:
    python_command = _preferred_python_command(workspace)
    if python_command is None:
        return []

    changed_paths = _changed_paths_from_patch_targets(patch_targets)
    commands: list[list[str]] = []
    seen_commands: set[tuple[str, ...]] = set()

    for changed_path in changed_paths:
        related_test_path = PatchApprovalStore.related_test_path_for(workspace, changed_path)
        if related_test_path is None:
            continue
        command = [
            *python_command,
            "-m",
            "unittest",
            PatchApprovalStore.path_to_module(related_test_path),
        ]
        key = tuple(command)
        if key in seen_commands:
            continue
        seen_commands.add(key)
        commands.append(command)

    broad_command = _repo_level_verification_command(workspace, python_command, changed_paths)
    if broad_command is not None and tuple(broad_command) not in seen_commands:
        commands.append(broad_command)

    return commands


def _preferred_python_command(workspace: Path) -> list[str] | None:
    for candidate in (
        workspace / ".venv" / "bin" / "python",
        workspace / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.exists():
            return [str(candidate)]

    if sys.executable:
        return [sys.executable]
    return None


def _changed_paths_from_patch_targets(patch_targets: list[PatchTarget]) -> list[Path]:
    changed: list[Path] = []
    seen: set[str] = set()
    for target in patch_targets:
        candidate = target.after_path or target.before_path
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        changed.append(Path(candidate))
    return changed


def _repo_level_verification_command(
    workspace: Path,
    python_command: list[str],
    changed_paths: list[Path],
) -> list[str] | None:
    tests_dir = workspace / "tests"
    if tests_dir.exists() and tests_dir.is_dir():
        if _repo_prefers_pytest(workspace):
            return [*python_command, "-m", "pytest"]
        return [*python_command, "-m", "unittest", "discover", "-s", "tests"]

    changed_python_files = [
        path.as_posix()
        for path in changed_paths
        if path.suffix == ".py" and (workspace / path).exists()
    ]
    if changed_python_files:
        return [*python_command, "-m", "py_compile", *changed_python_files]
    return None


def _repo_prefers_pytest(workspace: Path) -> bool:
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


def _format_verification_log(
    patch_result: SandboxCommandResult,
    verification_results: list[tuple[str, SandboxCommandResult]],
) -> str:
    sections = [_format_command_log("Patch Apply", patch_result)]
    sections.extend(_format_command_log(title, result) for title, result in verification_results)
    return "\n\n".join(sections).strip()


def _format_command_log(title: str, result: SandboxCommandResult) -> str:
    stdout = result.stdout.rstrip() or "<empty>"
    stderr = result.stderr.rstrip() or "<empty>"
    return "\n".join(
        [
            f"== {title} ==",
            f"$ {result.command}",
            f"cwd: {result.cwd}",
            f"exit_code: {result.returncode}",
            "[stdout]",
            stdout,
            "[stderr]",
            stderr,
        ]
    )
