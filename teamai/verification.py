from __future__ import annotations

import shlex
from dataclasses import dataclass
from pathlib import Path

from .sandbox import Sandbox, SandboxCommandResult


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    log_output: str
    patch_returncode: int
    test_returncode: int | None


def verify_patch(patch_file: Path, sandbox: Sandbox) -> VerificationResult:
    patch_path = patch_file.resolve()
    patch_result = sandbox.run(f"patch -p1 < {shlex.quote(str(patch_path))}")
    if patch_result.returncode != 0:
        return VerificationResult(
            success=False,
            log_output=_format_verification_log(patch_result, None),
            patch_returncode=patch_result.returncode,
            test_returncode=None,
        )

    test_result = sandbox.run(["./.venv/bin/python", "-m", "unittest", "discover", "-s", "tests"])
    return VerificationResult(
        success=test_result.returncode == 0,
        log_output=_format_verification_log(patch_result, test_result),
        patch_returncode=patch_result.returncode,
        test_returncode=test_result.returncode,
    )


def _format_verification_log(
    patch_result: SandboxCommandResult,
    test_result: SandboxCommandResult | None,
) -> str:
    sections = [_format_command_log("Patch Apply", patch_result)]
    if test_result is not None:
        sections.append(_format_command_log("Test Run", test_result))
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
