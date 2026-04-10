"""
Persistent background daemon for teamAI.

Wraps the FastAPI service in a long-lived subprocess so the MLX model stays
loaded between tasks, avoiding repeated Metal initialization and cold-start
overhead. The daemon exposes the same HTTP API as `teamai serve`.

Usage:
    teamai daemon start [--port 8000] [--workspace .]
    teamai daemon stop
    teamai daemon status
    teamai daemon submit "Inspect this repo" [--workspace .]
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

GLOBAL_STATE_DIR = Path("~/.teamai")
DAEMON_PID_FILE = "daemon.pid"
DAEMON_LOG_FILE = "daemon.log"
DEFAULT_DAEMON_PORT = 8000
DEFAULT_DAEMON_HOST = "127.0.0.1"
_STARTUP_POLL_SECONDS = 3.0
_STARTUP_POLL_INTERVAL = 0.25
_SHUTDOWN_POLL_SECONDS = 4.0
_SHUTDOWN_POLL_INTERVAL = 0.5


def _global_state_dir() -> Path:
    return GLOBAL_STATE_DIR.expanduser()


def _pid_path() -> Path:
    return _global_state_dir() / DAEMON_PID_FILE


def _log_path() -> Path:
    return _global_state_dir() / DAEMON_LOG_FILE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def start_daemon(
    *,
    host: str = DEFAULT_DAEMON_HOST,
    port: int = DEFAULT_DAEMON_PORT,
    workspace: str | None = None,
    python_executable: str | None = None,
    env_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Start the teamAI daemon in the background.

    Returns a dict describing the outcome with keys:
        status: "started" | "already_running" | "start_failed"
        pid: int
        host: str
        port: int
        log_file: str
    """
    pid_file = _pid_path()
    if pid_file.exists():
        existing_pid = _read_pid(pid_file)
        if existing_pid and _pid_is_alive(existing_pid):
            return {
                "status": "already_running",
                "pid": existing_pid,
                "host": host,
                "port": port,
                "log_file": str(_log_path()),
            }

    _global_state_dir().mkdir(parents=True, exist_ok=True)
    log_file = _log_path()
    python = python_executable or sys.executable

    env = os.environ.copy()
    if workspace:
        env["TEAMAI_WORKSPACE_ROOT"] = workspace
    if env_overrides:
        env.update(env_overrides)
    env["TEAMAI_HOST"] = host
    env["TEAMAI_PORT"] = str(port)

    with log_file.open("w", encoding="utf-8") as log_handle:
        proc = subprocess.Popen(
            [python, "-m", "teamai", "serve", "--host", host, "--port", str(port)],
            stdout=log_handle,
            stderr=log_handle,
            env=env,
            start_new_session=True,
        )

    pid_file.write_text(str(proc.pid), encoding="utf-8")

    # Poll until the healthz endpoint responds or timeout
    deadline = time.monotonic() + _STARTUP_POLL_SECONDS
    started = False
    while time.monotonic() < deadline:
        time.sleep(_STARTUP_POLL_INTERVAL)
        if not _pid_is_alive(proc.pid):
            break
        health = _probe_health(port)
        if health.get("status") not in (None, "unreachable"):
            started = True
            break

    if not started and not _pid_is_alive(proc.pid):
        pid_file.unlink(missing_ok=True)
        return {
            "status": "start_failed",
            "pid": proc.pid,
            "host": host,
            "port": port,
            "log_file": str(log_file),
        }

    return {
        "status": "started" if started else "starting",
        "pid": proc.pid,
        "host": host,
        "port": port,
        "log_file": str(log_file),
    }


def stop_daemon() -> dict[str, Any]:
    """Send SIGTERM to the running daemon and clean up the PID file."""
    pid_file = _pid_path()
    if not pid_file.exists():
        return {"status": "not_running"}

    pid = _read_pid(pid_file)
    if not pid:
        pid_file.unlink(missing_ok=True)
        return {"status": "no_pid_found"}

    if not _pid_is_alive(pid):
        pid_file.unlink(missing_ok=True)
        return {"status": "already_stopped", "pid": pid}

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pid_file.unlink(missing_ok=True)
        return {"status": "already_stopped", "pid": pid}

    deadline = time.monotonic() + _SHUTDOWN_POLL_SECONDS
    while time.monotonic() < deadline:
        time.sleep(_SHUTDOWN_POLL_INTERVAL)
        if not _pid_is_alive(pid):
            break

    pid_file.unlink(missing_ok=True)
    return {"status": "stopped", "pid": pid}


def daemon_status(*, port: int = DEFAULT_DAEMON_PORT) -> dict[str, Any]:
    """Return the current daemon status."""
    pid_file = _pid_path()
    if not pid_file.exists():
        return {"status": "not_running"}

    pid = _read_pid(pid_file)
    if not pid or not _pid_is_alive(pid):
        return {"status": "not_running", "stale_pid_file": True, "pid_file": str(pid_file)}

    health = _probe_health(port)
    return {
        "status": "running",
        "pid": pid,
        "host": DEFAULT_DAEMON_HOST,
        "port": port,
        "health": health,
        "log_file": str(_log_path()),
    }


def submit_task_to_daemon(
    task: str,
    *,
    workspace: str | None = None,
    port: int = DEFAULT_DAEMON_PORT,
    execution_mode: str = "read_only",
    max_rounds: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> dict[str, Any]:
    """Submit a task to the running daemon as a background job.

    Returns the job response from the daemon API, or an error dict.
    """
    import urllib.error
    import urllib.request

    payload: dict[str, Any] = {
        "task": task,
        "execution_mode": execution_mode,
    }
    if workspace:
        payload["workspace_path"] = workspace
    if max_rounds is not None:
        payload["max_rounds"] = max_rounds
    if max_tokens is not None:
        payload["max_tokens_per_turn"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature

    body = json.dumps(payload).encode("utf-8")
    url = f"http://127.0.0.1:{port}/v1/jobs"
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return {"status": "error", "error": str(exc), "hint": "Is the daemon running? Try: teamai daemon status"}


def get_daemon_job(job_id: str, *, port: int = DEFAULT_DAEMON_PORT) -> dict[str, Any]:
    """Fetch the status and result of a previously submitted job."""
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{port}/v1/jobs/{job_id}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return {"status": "error", "error": str(exc)}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _read_pid(pid_file: Path) -> int | None:
    try:
        raw = pid_file.read_text(encoding="utf-8").strip()
        return int(raw) if raw.isdigit() else None
    except (OSError, ValueError):
        return None


def _pid_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _probe_health(port: int) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    try:
        url = f"http://127.0.0.1:{port}/healthz"
        with urllib.request.urlopen(url, timeout=2) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        return {"status": "unreachable", "error": str(exc)}
