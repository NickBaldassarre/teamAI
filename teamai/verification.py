from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from pathlib import Path

from .autonomy import build_check_commands
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
    # `-E` removes files whose post-patch contents are empty, which keeps
    # delete hunks aligned with the state we'll later apply to the workspace.
    patch_result = sandbox.run(f"patch -p1 -E < {shlex.quote(str(patch_path))}")
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
    changed_paths = _changed_paths_from_patch_targets(patch_targets)
    return build_check_commands(workspace=workspace, changed_paths=[path.as_posix() for path in changed_paths])


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
