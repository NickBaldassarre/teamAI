from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, Field

from .config import Settings


RuntimeProbeMode = Literal["import", "generate"]
RuntimeProbeStatus = Literal["healthy", "unavailable", "unknown"]
RuntimePythonSource = Literal["env_override", "project_venv", "active_python"]

DEFAULT_RUNTIME_PROBE_MAX_TOKENS = 12
DEFAULT_RUNTIME_PROBE_TIMEOUT_SECONDS = 180
DEFAULT_RUNTIME_PROBE_PROMPT = "Reply with the single word OK."

_RUNTIME_PROBE_SCRIPT = """
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


request = json.loads(os.environ["TEAMAI_RUNTIME_PROBE_REQUEST"])
payload = {
    "status": "unknown",
    "reason": "not_started",
    "summary": "Runtime probe did not complete.",
    "checked_at": datetime.now(timezone.utc).isoformat(),
    "probe_mode": request.get("probe_mode", "generate"),
    "python_executable": sys.executable,
    "model_id": request.get("model_id", ""),
    "model_revision": request.get("model_revision"),
    "warnings": [],
    "details": {},
}

try:
    import mlx_vlm  # noqa: F401
except Exception as exc:
    payload.update(
        status="unavailable",
        reason="mlx_import_failed",
        summary=f"MLX import failed: {exc}",
    )
    print(json.dumps(payload))
    raise SystemExit(0)

try:
    import mlx.core as mx

    payload["details"]["default_device"] = str(mx.default_device())
except Exception as exc:
    payload["details"]["default_device_error"] = str(exc)

if payload["probe_mode"] == "import":
    payload.update(
        status="healthy",
        reason="mlx_import_ok",
        summary="MLX import preflight passed.",
    )
    print(json.dumps(payload))
    raise SystemExit(0)

try:
    from teamai.config import Settings
    from teamai.model_backend import MLXModelBackend

    settings = Settings(
        model_id=request.get("model_id", ""),
        model_revision=request.get("model_revision"),
        force_download=bool(request.get("force_download", False)),
        trust_remote_code=bool(request.get("trust_remote_code", False)),
        enable_thinking=bool(request.get("enable_thinking", False)),
        workspace_root=Path(request.get("workspace_root", ".")).resolve(),
        max_rounds=1,
        max_actions_per_round=1,
        max_tokens_per_turn=max(32, int(request.get("max_tokens", 12))),
        temperature=0.0,
        allow_shell=False,
        allow_writes=False,
        command_timeout_seconds=30,
        max_file_bytes=4096,
        max_command_output_chars=4096,
        host="127.0.0.1",
        port=8000,
    )
    backend = MLXModelBackend(settings)
    response = backend.generate_messages(
        messages=[
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": request.get("prompt", "Reply with the single word OK.")},
        ],
        max_tokens=max(1, int(request.get("max_tokens", 12))),
        temperature=0.0,
        enable_thinking=False,
    )
    payload["details"].update(
        {
            "prompt_tokens": response.prompt_tokens,
            "generation_tokens": response.generation_tokens,
            "total_tokens": response.total_tokens,
            "peak_memory_gb": response.peak_memory_gb,
            "output_preview": response.text[:120],
        }
    )
    payload.update(
        status="healthy",
        reason="mlx_generate_ok",
        summary="MLX model warmup generation succeeded.",
    )
except Exception as exc:
    payload.update(
        status="unavailable",
        reason="mlx_generate_failed",
        summary=f"MLX model warmup generation failed: {exc}",
    )

print(json.dumps(payload))
""".strip()


class RuntimePythonSelection(BaseModel):
    active_python: str
    selected_python: str
    source: RuntimePythonSource
    using_selected_python: bool
    project_root: str
    summary: str


class RuntimeProbeReport(BaseModel):
    status: RuntimeProbeStatus
    reason: str
    summary: str
    checked_at: datetime
    probe_mode: RuntimeProbeMode
    python_executable: str
    model_id: str
    model_revision: str | None = None
    warnings: list[str] = Field(default_factory=list)
    details: dict[str, object] = Field(default_factory=dict)


class RuntimeDoctorReport(BaseModel):
    selection: RuntimePythonSelection
    probe: RuntimeProbeReport
    suggested_run_command: str
    suggested_eval_command: str
    suggested_bridge_command: str


RuntimeSubprocessRunner = Callable[
    [list[str], dict[str, str], Path, float | None],
    subprocess.CompletedProcess[str],
]


def select_runtime_python(
    project_root: Path,
    *,
    current_python: Path | None = None,
) -> RuntimePythonSelection:
    project_root = project_root.resolve()
    active_python = (current_python or Path(sys.executable)).expanduser().resolve()

    override_raw = os.getenv("TEAMAI_PYTHON_EXECUTABLE", "").strip()
    if override_raw:
        override = Path(override_raw).expanduser()
        selected = override if override.is_absolute() else (project_root / override)
        return RuntimePythonSelection(
            active_python=str(active_python),
            selected_python=str(selected),
            source="env_override",
            using_selected_python=selected.resolve() == active_python,
            project_root=str(project_root),
            summary="Using `TEAMAI_PYTHON_EXECUTABLE` as the selected local runtime.",
        )

    for candidate in (
        project_root / ".venv" / "bin" / "python",
        project_root / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.exists():
            selected = candidate
            return RuntimePythonSelection(
                active_python=str(active_python),
                selected_python=str(selected),
                source="project_venv",
                using_selected_python=selected.resolve() == active_python,
                project_root=str(project_root),
                summary="Using the project-local virtualenv Python as the selected local runtime.",
            )

    return RuntimePythonSelection(
        active_python=str(active_python),
        selected_python=str(active_python),
        source="active_python",
        using_selected_python=True,
        project_root=str(project_root),
        summary="Using the active interpreter because no project-local virtualenv override was found.",
    )


def run_runtime_probe(
    *,
    settings: Settings,
    project_root: Path,
    python_executable: Path,
    subprocess_runner: RuntimeSubprocessRunner,
    timeout_seconds: int = DEFAULT_RUNTIME_PROBE_TIMEOUT_SECONDS,
    probe_mode: RuntimeProbeMode = "generate",
    max_tokens: int = DEFAULT_RUNTIME_PROBE_MAX_TOKENS,
    prompt: str = DEFAULT_RUNTIME_PROBE_PROMPT,
) -> RuntimeProbeReport:
    checked_at = datetime.now(timezone.utc)
    command = [str(python_executable), "-c", _RUNTIME_PROBE_SCRIPT]
    env = dict(os.environ)
    env["TEAMAI_RUNTIME_PROBE_REQUEST"] = json.dumps(
        {
            "probe_mode": probe_mode,
            "model_id": settings.model_id,
            "model_revision": settings.model_revision,
            "force_download": settings.force_download,
            "trust_remote_code": settings.trust_remote_code,
            "enable_thinking": settings.enable_thinking,
            "workspace_root": str(settings.workspace_root),
            "max_tokens": max_tokens,
            "prompt": prompt,
        }
    )
    try:
        completed = subprocess_runner(command, env, project_root.resolve(), timeout_seconds)
    except subprocess.TimeoutExpired:
        return RuntimeProbeReport(
            status="unknown",
            reason="runtime_probe_timeout",
            summary="Runtime probe timed out before it could finish.",
            checked_at=checked_at,
            probe_mode=probe_mode,
            python_executable=str(python_executable),
            model_id=settings.model_id,
            model_revision=settings.model_revision,
            warnings=["Runtime probe timed out; local MLX health remains unresolved."],
        )

    stdout = completed.stdout.strip()
    if completed.returncode != 0:
        return RuntimeProbeReport(
            status="unknown",
            reason="runtime_probe_subprocess_failed",
            summary="Runtime probe subprocess did not complete cleanly.",
            checked_at=checked_at,
            probe_mode=probe_mode,
            python_executable=str(python_executable),
            model_id=settings.model_id,
            model_revision=settings.model_revision,
            warnings=[_tail_text(completed.stderr or completed.stdout or "Runtime probe subprocess failed.")],
        )

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError:
        return RuntimeProbeReport(
            status="unknown",
            reason="runtime_probe_invalid_payload",
            summary="Runtime probe returned an unreadable payload.",
            checked_at=checked_at,
            probe_mode=probe_mode,
            python_executable=str(python_executable),
            model_id=settings.model_id,
            model_revision=settings.model_revision,
            warnings=[_tail_text(stdout or completed.stderr or "Runtime probe payload was not valid JSON.")],
        )

    if isinstance(payload, dict):
        payload.setdefault("checked_at", checked_at.isoformat())
        payload.setdefault("probe_mode", probe_mode)
        payload.setdefault("python_executable", str(python_executable))
        payload.setdefault("model_id", settings.model_id)
        payload.setdefault("model_revision", settings.model_revision)
        payload.setdefault("warnings", [])
        payload.setdefault("details", {})

    try:
        return RuntimeProbeReport.model_validate(payload)
    except Exception:
        return RuntimeProbeReport(
            status="unknown",
            reason="runtime_probe_invalid_schema",
            summary="Runtime probe returned an unexpected payload shape.",
            checked_at=checked_at,
            probe_mode=probe_mode,
            python_executable=str(python_executable),
            model_id=settings.model_id,
            model_revision=settings.model_revision,
            warnings=[_tail_text(stdout)],
        )


def run_runtime_doctor(
    *,
    settings: Settings,
    project_root: Path,
    current_python: Path | None,
    subprocess_runner: RuntimeSubprocessRunner,
    probe_mode: RuntimeProbeMode = "generate",
    timeout_seconds: int = DEFAULT_RUNTIME_PROBE_TIMEOUT_SECONDS,
    max_tokens: int = DEFAULT_RUNTIME_PROBE_MAX_TOKENS,
) -> RuntimeDoctorReport:
    selection = select_runtime_python(project_root, current_python=current_python)
    probe = run_runtime_probe(
        settings=settings,
        project_root=project_root,
        python_executable=Path(selection.selected_python),
        subprocess_runner=subprocess_runner,
        timeout_seconds=timeout_seconds,
        probe_mode=probe_mode,
        max_tokens=max_tokens,
    )
    selected_python = Path(selection.selected_python)
    command_prefix = _render_command_prefix(selected_python, project_root)
    return RuntimeDoctorReport(
        selection=selection,
        probe=probe,
        suggested_run_command=f"{command_prefix} run \"Inspect this repository and identify the next engineering tasks.\" --workspace .",
        suggested_eval_command=(
            f"{command_prefix} eval --suite-file evals/teamai_smoke.json "
            "--allow-write-cases --runner-mode terminal_bridge"
        ),
        suggested_bridge_command=(
            f"{command_prefix} bridge-launch "
            "\"Inspect this repository and identify the next engineering tasks.\" --workspace ."
        ),
    )


def default_runtime_subprocess_runner(
    command: list[str],
    env: dict[str, str],
    cwd: Path,
    timeout_seconds: float | None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )


def render_runtime_doctor_markdown(report: RuntimeDoctorReport) -> str:
    lines = [
        "# teamAI Doctor",
        "",
        f"- Active Python: {report.selection.active_python}",
        f"- Selected Python: {report.selection.selected_python}",
        f"- Runtime source: {report.selection.source}",
        f"- Runtime summary: {report.selection.summary}",
        f"- Probe mode: {report.probe.probe_mode}",
        f"- Probe status: {report.probe.status} ({report.probe.reason})",
        f"- Probe summary: {report.probe.summary}",
        f"- Model: {report.probe.model_id}",
    ]
    default_device = str(report.probe.details.get("default_device", "")).strip()
    if default_device:
        lines.append(f"- Default device: {default_device}")
    if report.probe.warnings:
        lines.append("- Warnings:")
        lines.extend(f"  - {warning}" for warning in report.probe.warnings)
    lines.extend(
        [
            "",
            "## Suggested Commands",
            "",
            f"- Run: `{report.suggested_run_command}`",
            f"- Live eval: `{report.suggested_eval_command}`",
            f"- Bridge launch: `{report.suggested_bridge_command}`",
        ]
    )
    return "\n".join(lines)


def _render_command_prefix(selected_python: Path, project_root: Path) -> str:
    try:
        relative = selected_python.resolve().relative_to(project_root.resolve())
        python_literal = f"./{relative.as_posix()}"
    except Exception:
        python_literal = str(selected_python)
    return f"{python_literal} -m teamai"


def _tail_text(text: str, *, max_chars: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return "..." + compact[-max_chars:]
