from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class SandboxCommandResult:
    command: str
    cwd: Path
    returncode: int
    stdout: str
    stderr: str


class Sandbox:
    def __init__(self, project_root: Path, *, symlink_venv: bool = True) -> None:
        self._project_root = project_root.resolve()
        self._symlink_venv = symlink_venv
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._path: Path | None = None

    @property
    def path(self) -> Path:
        if self._path is None:
            raise RuntimeError("Sandbox is not active.")
        return self._path

    def __enter__(self) -> Sandbox:
        self._temp_dir = tempfile.TemporaryDirectory(prefix="teamai-sandbox-")
        self._path = Path(self._temp_dir.name) / self._project_root.name
        self._path.mkdir(parents=True, exist_ok=False)
        self._populate(self._path)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        temp_dir = self._temp_dir
        self._temp_dir = None
        self._path = None
        if temp_dir is not None:
            temp_dir.cleanup()

    def run(
        self,
        command: Sequence[str] | str,
        *,
        env: Mapping[str, str] | None = None,
        input_text: str | None = None,
        shell: bool | None = None,
        timeout_seconds: float | None = None,
    ) -> SandboxCommandResult:
        active_path = self.path
        use_shell = isinstance(command, str) if shell is None else shell
        merged_env = os.environ.copy()
        if env:
            merged_env.update({key: str(value) for key, value in env.items()})

        completed = subprocess.run(
            command,
            cwd=str(active_path),
            env=merged_env,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
            shell=use_shell,
            executable="/bin/zsh" if use_shell else None,
        )
        command_text = command if isinstance(command, str) else " ".join(str(part) for part in command)
        return SandboxCommandResult(
            command=command_text,
            cwd=active_path,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )

    def _populate(self, destination: Path) -> None:
        for source in self._project_root.iterdir():
            if source.name == ".git":
                continue
            target = destination / source.name
            if source.name == ".venv" and self._symlink_venv and source.exists():
                target.symlink_to(source.resolve(), target_is_directory=True)
                continue
            self._copy_entry(source, target)

    def _copy_entry(self, source: Path, target: Path) -> None:
        if source.is_symlink():
            target.symlink_to(os.readlink(source), target_is_directory=source.is_dir())
            return
        if source.is_dir():
            shutil.copytree(source, target, symlinks=True)
            return
        shutil.copy2(source, target)
