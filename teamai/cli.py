from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable

from .schemas import RunEvent, RunResult


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local-first closed-loop orchestration with MLX.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one local closed-loop task.")
    run_parser.add_argument("task", help="Task for the local LLM council and agents.")
    run_parser.add_argument("--workspace", default=None, help="Workspace path inside TEAMAI_WORKSPACE_ROOT.")
    run_parser.add_argument("--max-rounds", type=int, default=None)
    run_parser.add_argument("--max-actions", type=int, default=None)
    run_parser.add_argument("--max-tokens", type=int, default=None)
    run_parser.add_argument("--temperature", type=float, default=None)
    run_parser.add_argument(
        "--output-format",
        choices=["full_json", "handoff_json", "handoff_markdown"],
        default="full_json",
        help="How to render the final result.",
    )
    run_parser.add_argument(
        "--output-file",
        default=None,
        help="Optional path to also write the rendered output.",
    )
    run_parser.add_argument(
        "--execution-mode",
        choices=["read_only", "workspace_write"],
        default="read_only",
    )
    run_parser.add_argument(
        "--stream-format",
        choices=["text", "jsonl"],
        default="text",
        help="How to render progress events on stderr while the run is active.",
    )
    run_parser.add_argument(
        "--event-log-file",
        default=None,
        help="Optional path to also write structured progress events as JSONL.",
    )

    eval_parser = subparsers.add_parser(
        "eval",
        help="Run an agentic evaluation suite against the current supervisor.",
    )
    eval_parser.add_argument(
        "--suite-file",
        required=True,
        help="Path to a JSON evaluation suite definition.",
    )
    eval_parser.add_argument(
        "--workspace",
        default=None,
        help="Optional workspace override applied to cases that do not declare one.",
    )
    eval_parser.add_argument(
        "--allow-write-cases",
        action="store_true",
        help="Temporarily allow workspace_write eval cases without changing TEAMAI_ALLOW_WRITES in .env.",
    )
    eval_parser.add_argument(
        "--runner-mode",
        choices=["isolated_subprocess", "in_process", "terminal_bridge"],
        default="isolated_subprocess",
        help=(
            "How to execute eval cases. `isolated_subprocess` runs each case in its own guarded `teamai run` "
            "process with a timeout; `terminal_bridge` launches each case through the macOS Terminal bridge; "
            "`in_process` keeps the older single-process behavior."
        ),
    )
    eval_parser.add_argument(
        "--per-case-timeout-seconds",
        type=int,
        default=180,
        help="Timeout for each isolated eval case. Ignored in `in_process` mode.",
    )
    eval_parser.add_argument(
        "--terminal-app",
        default="Terminal",
        help="Terminal app name used when `--runner-mode terminal_bridge` is selected.",
    )
    eval_parser.add_argument(
        "--output-format",
        choices=["full_json", "summary_markdown"],
        default="full_json",
        help="How to render the evaluation report.",
    )
    eval_parser.add_argument(
        "--output-file",
        default=None,
        help="Optional path to also write the rendered evaluation report.",
    )

    serve_parser = subparsers.add_parser("serve", help="Run the local FastAPI service.")
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)
    serve_parser.add_argument("--reload", action="store_true")

    bridge_launch_parser = subparsers.add_parser(
        "bridge-launch",
        help="Launch a local Terminal.app run that writes a Codex handoff artifact.",
    )
    bridge_launch_parser.add_argument("task", help="Task for the local model to run in Terminal.app.")
    bridge_launch_parser.add_argument("--workspace", default=".", help="Workspace path passed through to `teamai run`.")
    bridge_launch_parser.add_argument("--max-rounds", type=int, default=None)
    bridge_launch_parser.add_argument("--max-actions", type=int, default=None)
    bridge_launch_parser.add_argument("--max-tokens", type=int, default=None)
    bridge_launch_parser.add_argument("--temperature", type=float, default=None)
    bridge_launch_parser.add_argument(
        "--execution-mode",
        choices=["read_only", "workspace_write"],
        default="read_only",
    )
    bridge_launch_parser.add_argument(
        "--inject-write-env",
        action="store_true",
        help=(
            "Temporarily set TEAMAI_ALLOW_WRITES=true for the spawned Terminal run only. "
            "Intended for explicitly approved workspace_write bridge runs."
        ),
    )
    bridge_launch_parser.add_argument("--handoff-file", default=".teamai/codex-handoff.json")
    bridge_launch_parser.add_argument("--status-file", default=".teamai/codex-bridge-status.json")
    bridge_launch_parser.add_argument("--log-file", default=".teamai/codex-bridge.log")
    bridge_launch_parser.add_argument("--script-file", default=".teamai/codex-bridge-launch.sh")
    bridge_launch_parser.add_argument("--terminal-app", default="Terminal")
    bridge_launch_parser.add_argument("--dry-run", action="store_true")

    bridge_status_parser = subparsers.add_parser(
        "bridge-status",
        help="Read the latest bridge status and whether the handoff artifact exists.",
    )
    bridge_status_parser.add_argument("--handoff-file", default=".teamai/codex-handoff.json")
    bridge_status_parser.add_argument("--status-file", default=".teamai/codex-bridge-status.json")
    bridge_status_parser.add_argument("--log-file", default=".teamai/codex-bridge.log")
    bridge_status_parser.add_argument("--script-file", default=".teamai/codex-bridge-launch.sh")

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Probe the selected local MLX/Gemma runtime and print the supported launch path.",
    )
    doctor_parser.add_argument(
        "--probe-mode",
        choices=["import", "generate"],
        default="generate",
        help="`generate` performs a real model warmup generation; `import` only checks MLX imports.",
    )
    doctor_parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=180,
        help="Timeout for the runtime probe subprocess.",
    )
    doctor_parser.add_argument(
        "--max-probe-tokens",
        type=int,
        default=12,
        help="Max tokens used for the model warmup generation probe.",
    )
    doctor_parser.add_argument(
        "--output-format",
        choices=["full_json", "summary_markdown"],
        default="summary_markdown",
        help="How to render the doctor report.",
    )
    doctor_parser.add_argument(
        "--output-file",
        default=None,
        help="Optional path to also write the rendered doctor report.",
    )

    execute_handoff_parser = subparsers.add_parser(
        "execute-handoff",
        help="Send the local semantic skeleton to the cloud Codex model, verify the returned patch in a sandbox, and save the patch for review.",
    )
    execute_handoff_parser.add_argument(
        "--payload-file",
        default=".teamai/codex_payload.json",
        help="Path to the local semantic skeleton payload JSON.",
    )
    execute_handoff_parser.add_argument(
        "--patch-file",
        default=".teamai/codex_solution.patch",
        help="Path where the returned patch should be written.",
    )
    execute_handoff_parser.add_argument(
        "--engine",
        choices=["codex", "gemini"],
        default="codex",
        help="Which cloud execution engine to use for the handoff.",
    )
    execute_handoff_parser.add_argument(
        "--model",
        default=None,
        help="Optional cloud model override. Defaults to TEAMAI_CODEX_MODEL or gpt-5.4.",
    )

    approvals_parser = subparsers.add_parser(
        "approvals",
        help="List, inspect, apply, reject, or prune patch approvals.",
    )
    approvals_subparsers = approvals_parser.add_subparsers(dest="approvals_command", required=True)

    approvals_list_parser = approvals_subparsers.add_parser("list", help="List patch approvals.")
    approvals_list_parser.add_argument("--workspace", default=None, help="Workspace path inside TEAMAI_WORKSPACE_ROOT.")
    approvals_list_parser.add_argument("--all", action="store_true", help="Include applied or stale approvals too.")

    approvals_show_parser = approvals_subparsers.add_parser("show", help="Show one patch approval.")
    approvals_show_parser.add_argument("--workspace", default=None, help="Workspace path inside TEAMAI_WORKSPACE_ROOT.")
    approvals_show_parser.add_argument("approval_id")

    approvals_apply_parser = approvals_subparsers.add_parser("apply", help="Apply a pending patch approval.")
    approvals_apply_parser.add_argument("--workspace", default=None, help="Workspace path inside TEAMAI_WORKSPACE_ROOT.")
    approvals_apply_parser.add_argument("approval_id")
    approvals_apply_parser.add_argument(
        "--continue",
        dest="continue_run",
        action="store_true",
        help="After applying the patch, immediately resume the originating task.",
    )
    approvals_apply_parser.add_argument("--max-rounds", type=int, default=None)
    approvals_apply_parser.add_argument("--max-actions", type=int, default=None)
    approvals_apply_parser.add_argument("--max-tokens", type=int, default=None)
    approvals_apply_parser.add_argument("--temperature", type=float, default=None)

    approvals_reject_parser = approvals_subparsers.add_parser("reject", help="Reject a pending patch approval.")
    approvals_reject_parser.add_argument("--workspace", default=None, help="Workspace path inside TEAMAI_WORKSPACE_ROOT.")
    approvals_reject_parser.add_argument("approval_id")
    approvals_reject_parser.add_argument(
        "--reason",
        default="",
        help="Optional reason to record with the rejection.",
    )

    approvals_prune_stale_parser = approvals_subparsers.add_parser(
        "prune-stale",
        help="Delete stale patch approval artifacts.",
    )
    approvals_prune_stale_parser.add_argument(
        "--workspace",
        default=None,
        help="Workspace path inside TEAMAI_WORKSPACE_ROOT.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        print("[teamai] Booting run command", file=sys.stderr, flush=True)
        from .config import Settings
        from .handoff import build_handoff_packet, render_handoff_markdown
        from .schemas import RunRequest
        from .supervisor import ClosedLoopSupervisor

        settings = Settings.from_env()
        request = RunRequest(
            task=args.task,
            workspace_path=args.workspace,
            max_rounds=args.max_rounds,
            max_actions_per_round=args.max_actions,
            max_tokens_per_turn=args.max_tokens,
            temperature=args.temperature,
            execution_mode=args.execution_mode,
        )
        progress_callback, event_callback, close_event_stream = _build_run_stream_handlers(
            project_root=Path.cwd().resolve(),
            stream_format=args.stream_format,
            event_log_file=args.event_log_file,
        )
        try:
            result = ClosedLoopSupervisor(settings).run(
                request,
                progress_callback=progress_callback,
                event_callback=event_callback,
            )
        finally:
            close_event_stream()
        if args.output_format == "full_json":
            rendered_output = json.dumps(result.model_dump(mode="json"), indent=2)
        else:
            handoff = build_handoff_packet(task=args.task, result=result)
            if args.output_format == "handoff_json":
                rendered_output = json.dumps(handoff.model_dump(mode="json"), indent=2)
            else:
                rendered_output = render_handoff_markdown(handoff)

        _write_cli_output(rendered_output=rendered_output, output_file=args.output_file)
        payload_path = _write_codex_payload_artifact(result)
        if payload_path is not None:
            print(
                f"[teamai] Wrote semantic skeleton to {payload_path}",
                file=sys.stderr,
                flush=True,
            )

        print(rendered_output)
        return 0

    if args.command == "eval":
        from .config import Settings
        from .evals import load_eval_suite, render_eval_markdown, run_eval_suite
        from .runtime import select_runtime_python

        settings = Settings.from_env()
        project_root = Path.cwd().resolve()
        selection = select_runtime_python(project_root, current_python=Path(sys.executable))
        if not selection.using_selected_python:
            print(
                f"[teamai] Selected local runtime {selection.selected_python} ({selection.source})",
                file=sys.stderr,
                flush=True,
            )
        suite = load_eval_suite(_resolve_cli_path(project_root, args.suite_file))
        report = run_eval_suite(
            settings=settings,
            suite=suite,
            workspace_override=args.workspace,
            allow_write_cases=args.allow_write_cases,
            runner_mode=args.runner_mode,
            per_case_timeout_seconds=args.per_case_timeout_seconds,
            project_root=project_root,
            python_executable=Path(selection.selected_python),
            terminal_app=args.terminal_app,
        )
        if args.output_format == "summary_markdown":
            rendered_output = render_eval_markdown(report)
        else:
            rendered_output = json.dumps(report.model_dump(mode="json"), indent=2)

        _write_cli_output(rendered_output=rendered_output, output_file=args.output_file)
        print(rendered_output)
        return 0 if report.metrics.failed_cases == 0 else 1

    if args.command == "serve":
        from .api import create_app
        from .config import Settings
        import uvicorn

        settings = Settings.from_env()
        uvicorn.run(
            create_app(settings),
            host=args.host or settings.host,
            port=args.port or settings.port,
            reload=args.reload,
        )
        return 0

    if args.command == "bridge-launch":
        from .bridge import BridgeArtifacts, BridgeLaunchConfig, BridgePreflightError, launch_bridge
        from .runtime import select_runtime_python

        project_root = Path.cwd().resolve()
        selection = select_runtime_python(project_root, current_python=Path(sys.executable))
        if not selection.using_selected_python:
            print(
                f"[teamai] Selected local runtime {selection.selected_python} ({selection.source})",
                file=sys.stderr,
                flush=True,
            )
        artifacts = BridgeArtifacts(
            handoff_file=_resolve_cli_path(project_root, args.handoff_file),
            status_file=_resolve_cli_path(project_root, args.status_file),
            log_file=_resolve_cli_path(project_root, args.log_file),
            script_file=_resolve_cli_path(project_root, args.script_file),
        )
        config = BridgeLaunchConfig(
            task=args.task,
            project_root=project_root,
            python_executable=Path(selection.selected_python),
            workspace=args.workspace,
            max_rounds=args.max_rounds,
            max_actions=args.max_actions,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            execution_mode=args.execution_mode,
            inject_write_env=args.inject_write_env,
            terminal_app=args.terminal_app,
            artifacts=artifacts,
        )
        try:
            payload = launch_bridge(config, dry_run=args.dry_run)
        except BridgePreflightError as exc:
            print(json.dumps(exc.payload, indent=2))
            return 1
        except RuntimeError as exc:
            print(json.dumps({"state": "launch_failed", "error": str(exc)}, indent=2))
            return 1

        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "bridge-status":
        from .bridge import BridgeArtifacts, load_bridge_status

        project_root = Path.cwd().resolve()
        artifacts = BridgeArtifacts(
            handoff_file=_resolve_cli_path(project_root, args.handoff_file),
            status_file=_resolve_cli_path(project_root, args.status_file),
            log_file=_resolve_cli_path(project_root, args.log_file),
            script_file=_resolve_cli_path(project_root, args.script_file),
        )
        print(json.dumps(load_bridge_status(artifacts), indent=2))
        return 0

    if args.command == "doctor":
        from .config import Settings
        from .runtime import (
            default_runtime_subprocess_runner,
            render_runtime_doctor_markdown,
            run_runtime_doctor,
        )

        project_root = Path.cwd().resolve()
        settings = Settings.from_env()
        report = run_runtime_doctor(
            settings=settings,
            project_root=project_root,
            current_python=Path(sys.executable),
            subprocess_runner=default_runtime_subprocess_runner,
            probe_mode=args.probe_mode,
            timeout_seconds=args.timeout_seconds,
            max_tokens=args.max_probe_tokens,
        )
        if args.output_format == "summary_markdown":
            rendered_output = render_runtime_doctor_markdown(report)
        else:
            rendered_output = json.dumps(report.model_dump(mode="json"), indent=2)
        _write_cli_output(rendered_output=rendered_output, output_file=args.output_file)
        print(rendered_output)
        return 0 if report.probe.status == "healthy" else 1

    if args.command == "execute-handoff":
        from .verification import Sandbox, verify_patch

        project_root = Path.cwd().resolve()
        
        if args.engine == "gemini":
            from .integrations.gemini_bridge import execute_gemini_handoff
            try:
                execute_gemini_handoff(
                    project_root=project_root,
                    payload_file=args.payload_file,
                    patch_file=args.patch_file,
                    model=args.model or "gemini-2.5-pro",
                )
            except (OSError, RuntimeError, ValueError) as exc:
                print(json.dumps({"error": str(exc)}, indent=2))
                return 1
        else:
            from .integrations.codex_bridge import execute_codex_handoff
            try:
                execute_codex_handoff(
                    project_root=project_root,
                    payload_file=args.payload_file,
                    patch_file=args.patch_file,
                    model=args.model or "gpt-5.4",
                )
            except (OSError, RuntimeError, ValueError) as exc:
                print(json.dumps({"error": str(exc)}, indent=2))
                return 1

        with Sandbox(project_root) as sandbox:
            verification_result = verify_patch(Path(args.patch_file), sandbox)

        if verification_result.success:
            print("Patch verified successfully in sandbox. Ready for human review.")
            return 0

        print("Sandbox verification failed.")
        log_path = project_root / ".teamai" / "failure_context.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(verification_result.log_output, encoding="utf-8")
        return 1

    if args.command == "approvals":
        from .approvals import PatchApprovalStore
        from .config import ConfigError, Settings

        settings = Settings.from_env()
        try:
            workspace = settings.resolve_workspace(args.workspace)
        except ConfigError as exc:
            print(json.dumps({"error": str(exc)}, indent=2))
            return 1

        store = PatchApprovalStore()
        try:
            if args.approvals_command == "list":
                payload = [
                    store.summarize(record)
                    for record in store.list(workspace=workspace, include_all=args.all)
                ]
                print(json.dumps(payload, indent=2))
                return 0

            if args.approvals_command == "show":
                payload = store.summarize(
                    store.get(workspace=workspace, approval_id=args.approval_id),
                    include_diff=True,
                )
                print(json.dumps(payload, indent=2))
                return 0

            if args.approvals_command == "apply":
                applied = store.apply(workspace=workspace, approval_id=args.approval_id)
                payload = store.summarize(applied, include_diff=True)
                if not args.continue_run:
                    print(json.dumps(payload, indent=2))
                    return 0

                from .schemas import RunRequest
                from .supervisor import ClosedLoopSupervisor

                continuation_context = store.build_continuation_context(applied, workspace=workspace)
                continuation_task = store.build_continuation_task(
                    applied,
                    continuation_context=continuation_context,
                )
                continuation_mode = store.continuation_execution_mode(applied)
                continuation_settings = settings
                if continuation_mode == "workspace_write" and not continuation_settings.allow_writes:
                    continuation_settings = replace(continuation_settings, allow_writes=True)

                continuation_result = ClosedLoopSupervisor(continuation_settings).run(
                    RunRequest(
                        task=continuation_task,
                        workspace_path=args.workspace,
                        max_rounds=args.max_rounds,
                        max_actions_per_round=args.max_actions,
                        max_tokens_per_turn=args.max_tokens,
                        temperature=args.temperature,
                        execution_mode=continuation_mode,
                        continuation_context=continuation_context,
                    ),
                    progress_callback=lambda message: print(f"[teamai] {message}", file=sys.stderr, flush=True),
                )
                print(
                    json.dumps(
                        {
                            "approval": payload,
                            "continuation_task": continuation_task,
                            "continuation_context": continuation_context,
                            "continuation": continuation_result.model_dump(mode="json"),
                        },
                        indent=2,
                    )
                )
                return 0

            if args.approvals_command == "reject":
                payload = store.summarize(
                    store.reject(
                        workspace=workspace,
                        approval_id=args.approval_id,
                        reason=args.reason,
                    ),
                    include_diff=True,
                )
                print(json.dumps(payload, indent=2))
                return 0

            if args.approvals_command == "prune-stale":
                pruned = store.prune_stale(workspace=workspace)
                payload = {
                    "pruned": [store.summarize(record) for record in pruned],
                    "count": len(pruned),
                }
                print(json.dumps(payload, indent=2))
                return 0
        except (KeyError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, indent=2))
            return 1

    parser.error("Unknown command")
    return 2


def _resolve_cli_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else project_root / path


def _write_cli_output(*, rendered_output: str, output_file: str | None) -> None:
    if not output_file:
        return

    output_path = Path(output_file).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered_output + ("\n" if not rendered_output.endswith("\n") else ""), encoding="utf-8")


def _write_codex_payload_artifact(result: RunResult) -> Path | None:
    if result.codex_payload is None:
        return None

    workspace = Path(result.workspace).expanduser()
    payload_path = workspace / ".teamai" / "codex_payload.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(result.codex_payload.model_dump(mode="json"), indent=2)
    payload_path.write_text(rendered + "\n", encoding="utf-8")
    return payload_path


def _build_run_stream_handlers(
    *,
    project_root: Path,
    stream_format: str,
    event_log_file: str | None,
) -> tuple[
    Callable[[str], None] | None,
    Callable[[RunEvent], None] | None,
    Callable[[], None],
]:
    event_log_handle = None
    if event_log_file:
        event_log_path = _resolve_cli_path(project_root, event_log_file)
        event_log_path.parent.mkdir(parents=True, exist_ok=True)
        event_log_handle = event_log_path.open("w", encoding="utf-8")

    def close_stream() -> None:
        if event_log_handle is not None:
            event_log_handle.close()

    def on_progress(message: str) -> None:
        if stream_format == "text":
            print(f"[teamai] {message}", file=sys.stderr, flush=True)

    def on_event(event: RunEvent) -> None:
        payload = json.dumps(event.model_dump(mode="json"))
        if stream_format == "jsonl":
            print(payload, file=sys.stderr, flush=True)
        if event_log_handle is not None:
            event_log_handle.write(payload + "\n")
            event_log_handle.flush()

    return on_progress, on_event, close_stream


if __name__ == "__main__":
    raise SystemExit(main())
