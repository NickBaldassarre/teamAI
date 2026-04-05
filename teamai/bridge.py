from __future__ import annotations

import json
import shlex
import stat
import subprocess
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import ConfigError, Settings


DEFAULT_HANDOFF_FILE_NAME = "codex-handoff.json"
DEFAULT_STATUS_FILE_NAME = "codex-bridge-status.json"
DEFAULT_LOG_FILE_NAME = "codex-bridge.log"
DEFAULT_SCRIPT_FILE_NAME = "codex-bridge-launch.sh"
MEMORY_PROFILE_SPECS: dict[str, tuple[int, int, int, float, str]] = {
    "inspection": (
        3,
        2,
        160,
        0.15,
        "Applied light inspection guardrails to keep repository reconnaissance within the local model's memory envelope.",
    ),
    "light_recon": (
        2,
        2,
        128,
        0.1,
        "Applied light reconnaissance guardrails for a broad task to reduce MLX memory pressure.",
    ),
    "light_write": (
        2,
        2,
        128,
        0.1,
        "Applied light write guardrails for a write-capable bridge run to reduce MLX memory pressure.",
    ),
    "emergency_inspection": (
        1,
        1,
        96,
        0.0,
        "Applied emergency inspection guardrails after a memory-pressure failure.",
    ),
    "emergency_recon": (
        1,
        1,
        96,
        0.0,
        "Applied emergency reconnaissance guardrails after a memory-pressure failure.",
    ),
    "emergency_write": (
        1,
        1,
        96,
        0.0,
        "Applied emergency write guardrails after a memory-pressure failure.",
    ),
    "emergency_default": (
        1,
        1,
        96,
        0.0,
        "Applied an emergency bridge profile after a memory-pressure failure.",
    ),
}


@dataclass(frozen=True)
class BridgeArtifacts:
    handoff_file: Path
    status_file: Path
    log_file: Path
    script_file: Path

    @property
    def state_dir(self) -> Path:
        return self.status_file.parent


@dataclass(frozen=True)
class BridgeLaunchConfig:
    task: str
    project_root: Path
    python_executable: Path
    workspace: str | None
    max_rounds: int | None
    max_actions: int | None
    max_tokens: int | None
    temperature: float | None
    execution_mode: str
    inject_write_env: bool
    terminal_app: str
    artifacts: BridgeArtifacts
    memory_profile: str = "default"
    guardrail_notes: tuple[str, ...] = ()
    bridge_run_id: str = field(default_factory=lambda: uuid4().hex[:12])


class BridgePreflightError(RuntimeError):
    def __init__(self, message: str, payload: dict[str, object]) -> None:
        super().__init__(message)
        self.payload = payload


def default_bridge_artifacts(project_root: Path) -> BridgeArtifacts:
    state_dir = project_root / ".teamai"
    return BridgeArtifacts(
        handoff_file=state_dir / DEFAULT_HANDOFF_FILE_NAME,
        status_file=state_dir / DEFAULT_STATUS_FILE_NAME,
        log_file=state_dir / DEFAULT_LOG_FILE_NAME,
        script_file=state_dir / DEFAULT_SCRIPT_FILE_NAME,
    )


def launch_bridge(config: BridgeLaunchConfig, *, dry_run: bool = False) -> dict[str, object]:
    config = prepare_bridge_config(config)
    config.project_root.mkdir(parents=True, exist_ok=True)
    config.artifacts.state_dir.mkdir(parents=True, exist_ok=True)
    _clear_previous_bridge_outputs(config.artifacts)
    _preflight_bridge_launch(config)
    script_text = render_bridge_script(config)
    config.artifacts.script_file.write_text(script_text, encoding="utf-8")
    current_mode = config.artifacts.script_file.stat().st_mode
    config.artifacts.script_file.chmod(current_mode | stat.S_IXUSR)

    queued_status = _status_payload(
        state="queued",
        config=config,
        event_at=datetime.now(timezone.utc),
        exit_code=None,
        error=None,
    )
    _write_status(config.artifacts.status_file, queued_status)

    if dry_run:
        return queued_status

    if sys.platform != "darwin":
        raise RuntimeError("Bridge launch currently supports macOS Terminal.app only.")

    script_path_literal = _apple_script_string(str(config.artifacts.script_file))
    try:
        subprocess.run(
            [
                "osascript",
                "-e",
                f'tell application "{config.terminal_app}"',
                "-e",
                "activate",
                "-e",
                f'do script "/bin/zsh " & quoted form of {script_path_literal}',
                "-e",
                "end tell",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        failed_status = _status_payload(
            state="launch_failed",
            config=config,
            event_at=datetime.now(timezone.utc),
            exit_code=exc.returncode,
            error=(exc.stderr or exc.stdout or str(exc)).strip() or "Unknown AppleScript failure.",
        )
        _write_status(config.artifacts.status_file, failed_status)
        raise RuntimeError(f"Failed to launch Terminal bridge: {failed_status['error']}") from exc

    launched_status = _status_payload(
        state="launched",
        config=config,
        event_at=datetime.now(timezone.utc),
        exit_code=None,
        error=None,
    )
    _write_status(config.artifacts.status_file, launched_status)
    return launched_status


def _clear_previous_bridge_outputs(artifacts: BridgeArtifacts) -> None:
    for path in (artifacts.handoff_file, artifacts.log_file):
        try:
            if path.exists() or path.is_symlink():
                path.unlink()
        except IsADirectoryError:
            continue


def _preflight_bridge_launch(config: BridgeLaunchConfig) -> None:
    if config.execution_mode != "workspace_write":
        return

    try:
        settings = Settings.from_env()
    except ConfigError as exc:
        payload = _status_payload(
            state="preflight_failed",
            config=config,
            event_at=datetime.now(timezone.utc),
            exit_code=None,
            error=f"Bridge preflight could not load settings: {exc}",
        )
        _write_status(config.artifacts.status_file, payload)
        raise BridgePreflightError(str(payload["error"]), payload) from exc

    if settings.allow_writes or config.inject_write_env:
        return

    payload = _status_payload(
        state="preflight_failed",
        config=config,
        event_at=datetime.now(timezone.utc),
        exit_code=None,
        error=(
            "Bridge launch refused: `workspace_write` was requested, but `TEAMAI_ALLOW_WRITES` is false "
            "in local configuration. Enable writes first, rerun the bridge in `read_only` mode, "
            "or explicitly pass `--inject-write-env` for a temporary write-enabled bridge run."
        ),
    )
    _write_status(config.artifacts.status_file, payload)
    raise BridgePreflightError(str(payload["error"]), payload)


def _apply_memory_guardrails(config: BridgeLaunchConfig) -> BridgeLaunchConfig:
    profile = _classify_memory_profile(task=config.task, execution_mode=config.execution_mode)
    return _apply_memory_profile(config, profile)


def prepare_bridge_config(config: BridgeLaunchConfig) -> BridgeLaunchConfig:
    return _apply_memory_guardrails(config)


def _classify_memory_profile(*, task: str, execution_mode: str) -> str:
    lowered = task.lower()
    if execution_mode == "workspace_write":
        return "light_write"
    if _is_repository_inspection_task(lowered):
        return "inspection"
    if _is_broad_bridge_task(lowered):
        return "light_recon"
    return "default"


def _apply_memory_profile(config: BridgeLaunchConfig, profile: str) -> BridgeLaunchConfig:
    if profile == "default":
        return replace(config, memory_profile=profile)

    spec = MEMORY_PROFILE_SPECS.get(profile)
    if spec is None:
        return replace(config, memory_profile=profile)

    preferred_rounds, preferred_actions, preferred_tokens, preferred_temperature, note = spec
    max_rounds = _guardrailed_int(config.max_rounds, preferred=preferred_rounds)
    max_actions = _guardrailed_int(config.max_actions, preferred=preferred_actions)
    max_tokens = _guardrailed_int(config.max_tokens, preferred=preferred_tokens)
    temperature = _guardrailed_float(config.temperature, preferred=preferred_temperature)
    notes = [
        note,
        f"Effective limits: max_rounds={max_rounds}, max_actions={max_actions}, max_tokens={max_tokens}, temperature={temperature}.",
    ]
    return replace(
        config,
        max_rounds=max_rounds,
        max_actions=max_actions,
        max_tokens=max_tokens,
        temperature=temperature,
        memory_profile=profile,
        guardrail_notes=tuple(notes),
    )


def _build_memory_retry_config(config: BridgeLaunchConfig) -> BridgeLaunchConfig | None:
    retry_profile = _retry_memory_profile(config.memory_profile)
    if retry_profile is None:
        return None
    return _apply_memory_profile(
        replace(config, guardrail_notes=(), memory_profile=retry_profile),
        retry_profile,
    )


def _retry_memory_profile(memory_profile: str) -> str | None:
    mapping = {
        "default": "emergency_default",
        "inspection": "emergency_inspection",
        "light_recon": "emergency_recon",
        "light_write": "emergency_write",
    }
    return mapping.get(memory_profile)


def _is_repository_inspection_task(lowered_task: str) -> bool:
    inspection_markers = ["inspect", "review", "analyze", "understand", "summarize"]
    repo_markers = ["repository", "repo", "codebase", "project", "workspace"]
    task_markers = ["engineering task", "implementation step", "next step", "project state"]
    return (
        any(marker in lowered_task for marker in inspection_markers)
        and (any(marker in lowered_task for marker in repo_markers) or any(marker in lowered_task for marker in task_markers))
    )


def _is_broad_bridge_task(lowered_task: str) -> bool:
    explicit_write_markers = [
        "replace the text",
        "append",
        "insert",
        "update ",
        "set ",
        "write ",
    ]
    if any(marker in lowered_task for marker in explicit_write_markers):
        return False
    broad_markers = [
        "implement",
        "improve",
        "harden",
        "optimize",
        "refactor",
        "stabilize",
        "upgrade",
        "extend",
        "build",
        "create",
        "debug",
        "fix",
    ]
    return any(marker in lowered_task for marker in broad_markers)


def _guardrailed_int(current: int | None, *, preferred: int) -> int:
    if current is None:
        return preferred
    return min(current, preferred)


def _guardrailed_float(current: float | None, *, preferred: float) -> float:
    if current is None:
        return preferred
    return min(current, preferred)


def render_bridge_script(config: BridgeLaunchConfig) -> str:
    run_command = _render_run_command(config)
    retry_config = _build_memory_retry_config(config)
    retry_command = _render_run_command(retry_config) if retry_config is not None else ""
    python_exec = shlex.quote(str(config.python_executable))
    project_root = shlex.quote(str(config.project_root))
    log_file = shlex.quote(str(config.artifacts.log_file))
    metadata = {
        "bridge_run_id": config.bridge_run_id,
        "task": config.task,
        "workspace": config.workspace,
        "execution_mode": config.execution_mode,
        "inject_write_env": config.inject_write_env,
        "memory_profile": config.memory_profile,
        "guardrail_notes": list(config.guardrail_notes),
        "retry_on_memory_pressure": retry_config is not None,
        "retry_profile": retry_config.memory_profile if retry_config is not None else None,
        "handoff_file": str(config.artifacts.handoff_file),
        "status_file": str(config.artifacts.status_file),
        "log_file": str(config.artifacts.log_file),
        "script_file": str(config.artifacts.script_file),
        "command": run_command,
        "retry_command": retry_command,
    }
    metadata_literal = json.dumps(metadata, ensure_ascii=True)

    return f"""#!/bin/zsh
set -u

cd {project_root}

TEAMAI_BRIDGE_METADATA={shlex.quote(metadata_literal)} {python_exec} - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

metadata = json.loads(os.environ["TEAMAI_BRIDGE_METADATA"])
status = metadata | {{
    "state": "running",
    "event_at": datetime.now(timezone.utc).isoformat(),
    "exit_code": None,
    "error": None,
}}
Path(metadata["status_file"]).write_text(json.dumps(status, indent=2) + "\\n", encoding="utf-8")
PY

{run_command} > {log_file} 2>&1
exit_code=$?
first_exit_code=$exit_code
retry_attempted=0
retry_recovered=0
retry_profile=""
memory_pressure=$(TEAMAI_BRIDGE_METADATA={shlex.quote(metadata_literal)} {python_exec} - <<'PY'
import json
import os
from pathlib import Path

metadata = json.loads(os.environ["TEAMAI_BRIDGE_METADATA"])
log_path = Path(metadata["log_file"])
log_text = ""
if log_path.exists():
    try:
        log_text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        log_text = ""
memory_pressure = "Insufficient Memory" in log_text or "kIOGPUCommandBufferCallbackErrorOutOfMemory" in log_text
print("1" if memory_pressure else "0")
PY
)

if [ "$exit_code" -ne 0 ] && [ "$memory_pressure" = "1" ] && [ -n "{retry_command}" ]; then
  retry_attempted=1
  retry_profile={shlex.quote(retry_config.memory_profile if retry_config is not None else "")}
  TEAMAI_BRIDGE_METADATA={shlex.quote(metadata_literal)} TEAMAI_BRIDGE_RETRY_PROFILE="$retry_profile" {python_exec} - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

metadata = json.loads(os.environ["TEAMAI_BRIDGE_METADATA"])
retry_profile = os.environ["TEAMAI_BRIDGE_RETRY_PROFILE"]
status = metadata | {{
    "state": "retrying_after_memory_pressure",
    "event_at": datetime.now(timezone.utc).isoformat(),
    "exit_code": None,
    "error": None,
    "retry_attempted": True,
    "retry_profile": retry_profile,
    "memory_pressure": True,
}}
Path(metadata["status_file"]).write_text(json.dumps(status, indent=2) + "\\n", encoding="utf-8")
PY

  printf '\\n[teamai-bridge] Memory pressure detected. Retrying with %s profile.\\n' "$retry_profile" >> {log_file}
  {retry_command} >> {log_file} 2>&1
  retry_exit_code=$?
  exit_code=$retry_exit_code
  if [ "$retry_exit_code" -eq 0 ]; then
    retry_recovered=1
  fi
fi

TEAMAI_BRIDGE_METADATA={shlex.quote(metadata_literal)} TEAMAI_BRIDGE_EXIT_CODE="$exit_code" TEAMAI_BRIDGE_FIRST_EXIT_CODE="$first_exit_code" TEAMAI_BRIDGE_RETRY_ATTEMPTED="$retry_attempted" TEAMAI_BRIDGE_RETRY_RECOVERED="$retry_recovered" TEAMAI_BRIDGE_RETRY_PROFILE="$retry_profile" TEAMAI_BRIDGE_MEMORY_PRESSURE="$memory_pressure" {python_exec} - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

metadata = json.loads(os.environ["TEAMAI_BRIDGE_METADATA"])
exit_code = int(os.environ["TEAMAI_BRIDGE_EXIT_CODE"])
first_exit_code = int(os.environ.get("TEAMAI_BRIDGE_FIRST_EXIT_CODE", str(exit_code)))
retry_attempted = os.environ.get("TEAMAI_BRIDGE_RETRY_ATTEMPTED", "0") == "1"
retry_recovered = os.environ.get("TEAMAI_BRIDGE_RETRY_RECOVERED", "0") == "1"
retry_profile = os.environ.get("TEAMAI_BRIDGE_RETRY_PROFILE") or None
memory_pressure = os.environ.get("TEAMAI_BRIDGE_MEMORY_PRESSURE", "0") == "1"
handoff_exists = Path(metadata["handoff_file"]).exists()
state = "completed" if exit_code == 0 and handoff_exists else "failed"
status = metadata | {{
    "state": state,
    "event_at": datetime.now(timezone.utc).isoformat(),
    "exit_code": exit_code,
    "first_exit_code": first_exit_code,
    "error": (
        None
        if state == "completed"
        else (
            (
                "Bridge run failed with local MLX memory pressure even after an automatic lighter retry; inspect the log file."
                if retry_attempted and memory_pressure
                else "Bridge run failed with local MLX memory pressure; inspect the log file and consider an even lighter bridge profile."
            )
            if memory_pressure
            else "Bridge run failed; inspect the log file."
        )
    ),
    "handoff_exists": handoff_exists,
    "memory_pressure": memory_pressure,
    "retry_attempted": retry_attempted,
    "retry_recovered": retry_recovered,
    "retry_profile": retry_profile,
}}
Path(metadata["status_file"]).write_text(json.dumps(status, indent=2) + "\\n", encoding="utf-8")
PY

exit "$exit_code"
"""


def load_bridge_status(artifacts: BridgeArtifacts) -> dict[str, object]:
    if not artifacts.status_file.exists():
        return {
            "state": "missing",
            "bridge_run_id": None,
            "status_file": str(artifacts.status_file),
            "handoff_file": str(artifacts.handoff_file),
            "log_file": str(artifacts.log_file),
            "script_file": str(artifacts.script_file),
            "handoff_exists": artifacts.handoff_file.exists(),
            "log_exists": artifacts.log_file.exists(),
        }

    try:
        payload = json.loads(artifacts.status_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        payload = {
            "state": "invalid",
            "error": "Status file is not valid JSON.",
        }

    if not isinstance(payload, dict):
        payload = {
            "state": "invalid",
            "error": "Status payload is not a JSON object.",
        }

    payload.setdefault("status_file", str(artifacts.status_file))
    payload.setdefault("bridge_run_id", None)
    payload.setdefault("handoff_file", str(artifacts.handoff_file))
    payload.setdefault("log_file", str(artifacts.log_file))
    payload.setdefault("script_file", str(artifacts.script_file))
    payload["handoff_exists"] = artifacts.handoff_file.exists()
    payload["log_exists"] = artifacts.log_file.exists()
    return payload


def _build_run_command(config: BridgeLaunchConfig) -> list[str]:
    return build_run_command(config)


def build_run_command(
    config: BridgeLaunchConfig,
    *,
    output_format: str = "handoff_json",
    output_file: Path | None = None,
) -> list[str]:
    command = [
        str(config.python_executable),
        "-u",
        "-m",
        "teamai",
        "run",
        config.task,
        "--output-format",
        output_format,
        "--output-file",
        str(output_file or config.artifacts.handoff_file),
        "--execution-mode",
        config.execution_mode,
    ]
    if config.workspace is not None:
        command.extend(["--workspace", config.workspace])
    if config.max_rounds is not None:
        command.extend(["--max-rounds", str(config.max_rounds)])
    if config.max_actions is not None:
        command.extend(["--max-actions", str(config.max_actions)])
    if config.max_tokens is not None:
        command.extend(["--max-tokens", str(config.max_tokens)])
    if config.temperature is not None:
        command.extend(["--temperature", str(config.temperature)])
    return command


def _render_run_command(config: BridgeLaunchConfig | None) -> str:
    if config is None:
        return ""
    command = shlex.join(_build_run_command(config))
    if config.inject_write_env and config.execution_mode == "workspace_write":
        return f"TEAMAI_ALLOW_WRITES=true {command}"
    return command


def _status_payload(
    *,
    state: str,
    config: BridgeLaunchConfig,
    event_at: datetime,
    exit_code: int | None,
    error: str | None,
) -> dict[str, object]:
    return {
        "state": state,
        "event_at": event_at.isoformat(),
        "bridge_run_id": config.bridge_run_id,
        "task": config.task,
        "workspace": config.workspace,
        "execution_mode": config.execution_mode,
        "inject_write_env": config.inject_write_env,
        "memory_profile": config.memory_profile,
        "guardrail_notes": list(config.guardrail_notes),
        "retry_on_memory_pressure": _build_memory_retry_config(config) is not None,
        "retry_profile": (
            _build_memory_retry_config(config).memory_profile
            if _build_memory_retry_config(config) is not None
            else None
        ),
        "handoff_file": str(config.artifacts.handoff_file),
        "status_file": str(config.artifacts.status_file),
        "log_file": str(config.artifacts.log_file),
        "script_file": str(config.artifacts.script_file),
        "command": _render_run_command(config),
        "exit_code": exit_code,
        "error": error,
    }


def _write_status(status_file: Path, payload: dict[str, object]) -> None:
    status_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _apple_script_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'
