from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .autonomy import derive_write_policy
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
        "--write-policy",
        choices=["read_only", "propose_only", "auto_apply_low_risk", "auto_apply_scoped", "full_auto"],
        default=None,
        help=(
            "Explicit write policy. If omitted, CLI `workspace_write` runs default to `auto_apply_low_risk` "
            "while `read_only` runs stay read-only."
        ),
    )
    run_parser.add_argument(
        "--auto-commit",
        action="store_true",
        help="After a successful autonomous write run, create a git commit when the workspace is a git repo.",
    )
    run_parser.add_argument(
        "--auto-push",
        action="store_true",
        help=(
            "After a successful autonomous write run, push the review branch to the configured remote. "
            "Requires TEAMAI_ALLOW_GIT_PUSH=true and never pushes to a protected default branch."
        ),
    )
    run_parser.add_argument(
        "--push-branch",
        default=None,
        help="Optional feature-branch override used with --auto-push. Protected default branches are blocked.",
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
    run_parser.add_argument(
        "--follow-up",
        action="store_true",
        help=(
            "After a completed run, automatically run the next tasks extracted from the final answer. "
            "Chains runs until no next tasks remain or --follow-up-depth is exhausted."
        ),
    )
    run_parser.add_argument(
        "--follow-up-depth",
        type=int,
        default=1,
        help="Maximum number of follow-up tasks to chain automatically (default: 1).",
    )
    run_parser.add_argument(
        "--auto-execute-handoff",
        action="store_true",
        help=(
            "If the run produces a handoff payload for a broad task, immediately execute the selected handoff engine, "
            "verify the returned patch in a sandbox, and create a pending approval."
        ),
    )
    run_parser.add_argument(
        "--handoff-engine",
        choices=["codex", "gemini", "grok", "local"],
        default="codex",
        help="Execution engine used with `--auto-execute-handoff`.",
    )
    run_parser.add_argument(
        "--handoff-model",
        default=None,
        help="Optional model override used with `--auto-execute-handoff`.",
    )
    run_parser.add_argument(
        "--handoff-patch-file",
        default=".teamai/codex_solution.patch",
        help="Patch output path used with `--auto-execute-handoff`.",
    )
    run_parser.add_argument(
        "--autopilot-cycles",
        type=int,
        default=0,
        help=(
            "Run a bounded autonomous loop for broad tasks: execute a verified handoff, "
            "auto-apply the verified approval, and continue the task for up to this many cycles."
        ),
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

    eval_artifacts_parser = subparsers.add_parser(
        "eval-artifacts-report",
        help="Regenerate a human-readable eval report from captured status and stdout artifacts.",
    )
    eval_artifacts_parser.add_argument("--status-file", required=True)
    eval_artifacts_parser.add_argument("--stdout-file", required=True)
    eval_artifacts_parser.add_argument("--output-file", default=None)

    serve_parser = subparsers.add_parser("serve", help="Run the local FastAPI service.")
    serve_parser.add_argument("--host", default=None)
    serve_parser.add_argument("--port", type=int, default=None)
    serve_parser.add_argument("--reload", action="store_true")

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Run the local service and open the browser control dashboard.",
    )
    dashboard_parser.add_argument("--host", default=None)
    dashboard_parser.add_argument("--port", type=int, default=None)
    dashboard_parser.add_argument("--reload", action="store_true")
    dashboard_parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Start the dashboard server without launching a browser tab.",
    )

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
        help="Send the local semantic skeleton to the selected handoff engine, verify the returned patch in a sandbox, and save the patch for review.",
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
        choices=["codex", "gemini", "grok", "local"],
        default="codex",
        help="Which execution engine to use for the handoff.",
    )
    execute_handoff_parser.add_argument(
        "--model",
        default=None,
        help="Optional model override for the selected handoff engine.",
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

    # ------------------------------------------------------------------ next
    next_parser = subparsers.add_parser(
        "next",
        help="Run the next queued task from the most recent run's next_tasks list.",
    )
    next_parser.add_argument("--workspace", default=None, help="Workspace path inside TEAMAI_WORKSPACE_ROOT.")
    next_parser.add_argument("--max-rounds", type=int, default=None)
    next_parser.add_argument("--max-actions", type=int, default=None)
    next_parser.add_argument("--max-tokens", type=int, default=None)
    next_parser.add_argument("--temperature", type=float, default=None)
    next_parser.add_argument(
        "--execution-mode",
        choices=["read_only", "workspace_write"],
        default="read_only",
    )
    next_parser.add_argument(
        "--depth",
        type=int,
        default=1,
        help="How many next_tasks to chain in sequence (default: 1).",
    )
    next_parser.add_argument(
        "--output-format",
        choices=["full_json", "handoff_json", "handoff_markdown"],
        default="full_json",
    )

    # ------------------------------------------------------------------ daemon
    daemon_parser = subparsers.add_parser(
        "daemon",
        help="Manage the persistent teamAI background service.",
    )
    daemon_subparsers = daemon_parser.add_subparsers(dest="daemon_command", required=True)

    daemon_start = daemon_subparsers.add_parser("start", help="Start the daemon.")
    daemon_start.add_argument("--port", type=int, default=8000)
    daemon_start.add_argument("--workspace", default=None, help="Workspace root for the daemon.")
    daemon_start.add_argument("--host", default="127.0.0.1")

    daemon_subparsers.add_parser("stop", help="Stop the running daemon.")

    daemon_status_p = daemon_subparsers.add_parser("status", help="Show daemon status.")
    daemon_status_p.add_argument("--port", type=int, default=8000)

    daemon_submit = daemon_subparsers.add_parser("submit", help="Submit a task to the running daemon.")
    daemon_submit.add_argument("task", help="Task description.")
    daemon_submit.add_argument("--workspace", default=None)
    daemon_submit.add_argument("--port", type=int, default=8000)
    daemon_submit.add_argument("--execution-mode", choices=["read_only", "workspace_write"], default="read_only")
    daemon_submit.add_argument("--max-rounds", type=int, default=None)

    daemon_job = daemon_subparsers.add_parser("job", help="Fetch a submitted job's status and result.")
    daemon_job.add_argument("job_id", help="Job ID returned by `daemon submit`.")
    daemon_job.add_argument("--port", type=int, default=8000)

    daemon_install = daemon_subparsers.add_parser("install", help="Install as a launchd service (survives reboots).")
    daemon_install.add_argument("--port", type=int, default=8000)
    daemon_install.add_argument("--host", default="127.0.0.1")
    daemon_install.add_argument("--workspace", default=None, help="Workspace root for the daemon.")

    daemon_subparsers.add_parser("uninstall", help="Remove the launchd service.")

    daemon_sched = daemon_subparsers.add_parser("schedule", help="Manage recurring scheduled tasks.")
    sched_sub = daemon_sched.add_subparsers(dest="schedule_command", required=True)

    sched_add = sched_sub.add_parser("add", help="Add a recurring task schedule.")
    sched_add.add_argument("task", help="Task description to run on schedule.")
    sched_add.add_argument("--cron", required=True, help='Cron expression (5-field), e.g. "0 3 * * *".')
    sched_add.add_argument("--id", dest="schedule_id", default=None, help="Custom schedule ID.")
    sched_add.add_argument("--execution-mode", choices=["read_only", "workspace_write"], default="read_only")
    sched_add.add_argument("--workspace", default=None)
    sched_add.add_argument("--max-rounds", type=int, default=None)

    sched_sub.add_parser("list", help="List all scheduled tasks.")

    sched_rm = sched_sub.add_parser("remove", help="Remove a scheduled task.")
    sched_rm.add_argument("schedule_id", help="ID of the schedule to remove.")

    # ------------------------------------------------------------------ team
    team_parser = subparsers.add_parser(
        "team",
        help="Decompose a goal into subtasks and execute with multiple agents.",
    )
    team_subparsers = team_parser.add_subparsers(dest="team_command", required=True)

    team_run = team_subparsers.add_parser("run", help="Decompose and execute a goal with a team of agents.")
    team_run.add_argument("goal", help="High-level goal to accomplish.")
    team_run.add_argument("--workspace", default=None)
    team_run.add_argument("--max-rounds", type=int, default=None)
    team_run.add_argument("--execution-mode", choices=["read_only", "workspace_write"], default="read_only")
    team_run.add_argument("--max-tasks", type=int, default=12, help="Maximum subtasks to decompose into.")

    team_plan = team_subparsers.add_parser("plan", help="Decompose a goal into subtasks without executing.")
    team_plan.add_argument("goal", help="High-level goal to decompose.")
    team_plan.add_argument("--workspace", default=None)
    team_plan.add_argument("--max-tasks", type=int, default=12)

    # ------------------------------------------------------------------ agents
    agents_parser = subparsers.add_parser(
        "agents",
        help="Inspect the agent registry (agents.yaml).",
    )
    agents_subparsers = agents_parser.add_subparsers(dest="agents_command", required=True)

    agents_list = agents_subparsers.add_parser("list", help="List all registered agents.")
    agents_list.add_argument(
        "--format",
        choices=["table", "json"],
        default="table",
        help="Output format.",
    )

    agents_pick = agents_subparsers.add_parser("pick", help="Pick the best agent for a capability.")
    agents_pick.add_argument("capability", help="Task route capability (e.g. codex_handoff, multi_agent_loop).")
    agents_pick.add_argument("--prefer-local", action="store_true")
    agents_pick.add_argument("--no-env-check", action="store_true", help="Skip env var availability check.")

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
        requested_write_policy = args.write_policy
        if requested_write_policy is None and args.execution_mode == "workspace_write":
            requested_write_policy = "auto_apply_low_risk"
        request = RunRequest(
            task=args.task,
            workspace_path=args.workspace,
            max_rounds=args.max_rounds,
            max_actions_per_round=args.max_actions,
            max_tokens_per_turn=args.max_tokens,
            temperature=args.temperature,
            execution_mode=args.execution_mode,
            write_policy=derive_write_policy(
                execution_mode=args.execution_mode,
                requested_policy=requested_write_policy,
            ),
            auto_commit=bool(args.auto_commit),
            auto_push=bool(args.auto_push),
            push_remote="origin",
            push_branch_name=args.push_branch,
            handoff_engine=args.handoff_engine,
            handoff_model=args.handoff_model,
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
        handoff_packet = None
        if args.output_format != "full_json":
            handoff_packet = build_handoff_packet(task=args.task, result=result)
        payload_path = _write_codex_payload_artifact(result)
        if payload_path is not None:
            print(
                f"[teamai] Wrote semantic skeleton to {payload_path}",
                file=sys.stderr,
                flush=True,
            )

        auto_handoff: dict[str, Any] | None = None
        autopilot: dict[str, Any] | None = None
        auto_handoff_executed = False
        exit_code = 0
        autopilot_cycles = max(int(getattr(args, "autopilot_cycles", 0) or 0), 0)
        if autopilot_cycles > 0:
            print(
                f"[teamai] Autopilot enabled for up to {autopilot_cycles} cycle(s)",
                file=sys.stderr,
                flush=True,
            )
            autopilot = _run_autopilot_workflow(
                project_root=Path.cwd().resolve(),
                settings=settings,
                initial_result=result,
                initial_payload_path=payload_path,
                workspace_path=args.workspace,
                max_rounds=args.max_rounds,
                max_actions=args.max_actions,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                handoff_engine=args.handoff_engine,
                handoff_model=args.handoff_model,
                handoff_patch_file=args.handoff_patch_file,
                max_cycles=autopilot_cycles,
            )
            exit_code = 0 if bool(autopilot.get("success", False)) else 1
        elif getattr(args, "auto_execute_handoff", False):
            if payload_path is None:
                print(
                    "[teamai] Auto handoff requested, but this run did not produce a handoff payload.",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"[teamai] Auto handoff: executing verified {args.handoff_engine} handoff",
                    file=sys.stderr,
                    flush=True,
                )
                try:
                    auto_handoff = _execute_verified_handoff_workflow(
                        project_root=Path.cwd().resolve(),
                        engine=args.handoff_engine,
                        payload_file=payload_path,
                        patch_file=args.handoff_patch_file,
                        model=args.handoff_model,
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    auto_handoff = {
                        "engine": args.handoff_engine,
                        "model": args.handoff_model,
                        "payload_path": str(payload_path),
                        "patch_path": str(_resolve_cli_path(Path.cwd().resolve(), args.handoff_patch_file)),
                        "verification": {
                            "success": False,
                            "patch_returncode": None,
                            "test_returncode": None,
                            "commands_run": [],
                        },
                        "approval": None,
                        "approval_error": None,
                        "failure_context_path": None,
                        "summary": f"Auto handoff failed: {exc}",
                        "success": False,
                        "error": str(exc),
                    }
                auto_handoff_executed = True
                exit_code = 0 if bool(auto_handoff.get("success", False)) else 1

        if args.output_format == "full_json":
            if autopilot is not None:
                rendered_output = json.dumps(
                    {
                        "run": result.model_dump(mode="json"),
                        "autopilot": autopilot,
                    },
                    indent=2,
                )
            elif auto_handoff is None:
                rendered_output = json.dumps(result.model_dump(mode="json"), indent=2)
            else:
                rendered_output = json.dumps(
                    {
                        "run": result.model_dump(mode="json"),
                        "auto_handoff": auto_handoff,
                    },
                    indent=2,
                )
        elif args.output_format == "handoff_json":
            assert handoff_packet is not None
            if autopilot is not None:
                rendered_output = json.dumps(
                    {
                        "handoff_packet": handoff_packet.model_dump(mode="json"),
                        "autopilot": autopilot,
                    },
                    indent=2,
                )
            elif auto_handoff is None:
                rendered_output = json.dumps(handoff_packet.model_dump(mode="json"), indent=2)
            else:
                rendered_output = json.dumps(
                    {
                        "handoff_packet": handoff_packet.model_dump(mode="json"),
                        "auto_handoff": auto_handoff,
                    },
                    indent=2,
                )
        else:
            assert handoff_packet is not None
            rendered_output = render_handoff_markdown(handoff_packet)
            if autopilot is not None:
                rendered_output = (
                    f"{rendered_output}\n\n## Autopilot\n{autopilot['summary']}"
                )
            elif auto_handoff is not None:
                rendered_output = (
                    f"{rendered_output}\n\n## Verified Handoff Execution\n{auto_handoff['summary']}"
                )

        _write_cli_output(rendered_output=rendered_output, output_file=args.output_file)

        print(rendered_output)

        # --follow-up: chain through next_tasks from the completed run
        if getattr(args, "follow_up", False) and autopilot is not None:
            print(
                "[teamai] Skipping --follow-up because autopilot already continued the task.",
                file=sys.stderr,
                flush=True,
            )
        elif getattr(args, "follow_up", False) and auto_handoff_executed:
            print(
                "[teamai] Skipping --follow-up because the run already continued into a verified handoff.",
                file=sys.stderr,
                flush=True,
            )
        elif getattr(args, "follow_up", False) and result.status == "completed":
            from .memory import WorkspaceMemoryStore
            depth = getattr(args, "follow_up_depth", 1)
            _, next_tasks = WorkspaceMemoryStore._extract_summary_and_tasks(result.final_answer)
            chained = 0
            for follow_task in next_tasks:
                if chained >= depth:
                    break
                print(
                    f"[teamai] follow-up {chained + 1}/{depth}: {follow_task!r}",
                    file=sys.stderr,
                    flush=True,
                )
                follow_request = RunRequest(
                    task=follow_task,
                    workspace_path=args.workspace,
                    max_rounds=args.max_rounds,
                    max_actions_per_round=args.max_actions,
                    max_tokens_per_turn=args.max_tokens,
                    temperature=args.temperature,
                    execution_mode=args.execution_mode,
                )
                follow_progress, follow_events, follow_close = _build_run_stream_handlers(
                    project_root=Path.cwd().resolve(),
                    stream_format=args.stream_format,
                    event_log_file=args.event_log_file,
                )
                try:
                    follow_result = ClosedLoopSupervisor(settings).run(
                        follow_request,
                        progress_callback=follow_progress,
                        event_callback=follow_events,
                    )
                finally:
                    follow_close()
                follow_output = json.dumps(follow_result.model_dump(mode="json"), indent=2)
                print(follow_output)
                chained += 1
                if follow_result.status != "completed":
                    break

        return exit_code

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

    if args.command == "eval-artifacts-report":
        from .evals import render_eval_artifact_markdown

        project_root = Path.cwd().resolve()
        status_path = _resolve_cli_path(project_root, args.status_file)
        stdout_path = _resolve_cli_path(project_root, args.stdout_file)
        status_payload = json.loads(status_path.read_text(encoding="utf-8"))
        stdout_text = stdout_path.read_text(encoding="utf-8") if stdout_path.exists() else ""
        rendered_output = render_eval_artifact_markdown(
            status_payload=status_payload if isinstance(status_payload, dict) else {},
            stdout_text=stdout_text,
        )
        _write_cli_output(rendered_output=rendered_output, output_file=args.output_file)
        print(rendered_output)
        return 0

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

    if args.command == "dashboard":
        from .api import create_app
        from .config import Settings
        import uvicorn

        settings = Settings.from_env()
        host = args.host or settings.host
        port = args.port or settings.port
        dashboard_url = _dashboard_url(host, port)
        print(
            f"[teamai] Dashboard available at {dashboard_url}",
            file=sys.stderr,
            flush=True,
        )
        if not args.no_open_browser:
            _schedule_dashboard_open(dashboard_url)
        uvicorn.run(
            create_app(settings),
            host=host,
            port=port,
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
        project_root = Path.cwd().resolve()
        try:
            execution = _execute_verified_handoff_workflow(
                project_root=project_root,
                engine=args.engine,
                payload_file=args.payload_file,
                patch_file=args.patch_file,
                model=args.model,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            print(json.dumps({"error": str(exc)}, indent=2))
            return 1

        print(str(execution["summary"]))
        return 0 if bool(execution["success"]) else 1

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
                if not args.continue_run:
                    applied = store.apply(workspace=workspace, approval_id=args.approval_id)
                    payload = store.summarize(applied, include_diff=True)
                    print(json.dumps(payload, indent=2))
                    return 0

                continued = _apply_approval_and_continue(
                    settings=settings,
                    workspace=workspace,
                    workspace_path=args.workspace,
                    approval_id=args.approval_id,
                    max_rounds=args.max_rounds,
                    max_actions=args.max_actions,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    progress_callback=lambda message: print(f"[teamai] {message}", file=sys.stderr, flush=True),
                )
                print(
                    json.dumps(
                        {
                            "approval": continued["approval_summary"],
                            "continuation_task": continued["continuation_task"],
                            "continuation_context": continued["continuation_context"],
                            "continuation": continued["continuation_result"].model_dump(mode="json"),
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

    if args.command == "next":
        from .config import Settings
        from .memory import WorkspaceMemoryStore
        from .schemas import RunRequest
        from .supervisor import ClosedLoopSupervisor

        settings = Settings.from_env()
        project_root = Path.cwd().resolve()
        try:
            workspace = settings.resolve_workspace(args.workspace)
        except Exception as exc:
            print(json.dumps({"error": str(exc)}, indent=2))
            return 1

        records = WorkspaceMemoryStore()._load_history_records(workspace)
        # Find the most recent run with next_tasks
        next_tasks: list[str] = []
        for record in reversed(records):
            raw = record.get("next_tasks", [])
            if isinstance(raw, list) and raw:
                next_tasks = [str(t) for t in raw if str(t).strip()]
                break

        if not next_tasks:
            print(json.dumps({"status": "nothing_queued", "message": "No next_tasks found in recent run history."}))
            return 0

        depth = getattr(args, "depth", 1)
        ran = 0
        for task in next_tasks:
            if ran >= depth:
                break
            print(f"[teamai] next task {ran + 1}/{depth}: {task!r}", file=sys.stderr, flush=True)
            request = RunRequest(
                task=task,
                workspace_path=args.workspace,
                max_rounds=args.max_rounds,
                max_actions_per_round=args.max_actions,
                max_tokens_per_turn=args.max_tokens,
                temperature=args.temperature,
                execution_mode=args.execution_mode,
            )
            result = ClosedLoopSupervisor(settings).run(
                request,
                progress_callback=lambda msg: print(f"[teamai] {msg}", file=sys.stderr, flush=True),
            )
            print(json.dumps(result.model_dump(mode="json"), indent=2))
            ran += 1
            if result.status != "completed":
                break
        return 0

    if args.command == "daemon":
        from .daemon import (
            daemon_status,
            get_daemon_job,
            start_daemon,
            stop_daemon,
            submit_task_to_daemon,
        )
        from .runtime import select_runtime_python

        if args.daemon_command == "start":
            project_root = Path.cwd().resolve()
            selection = select_runtime_python(project_root, current_python=Path(sys.executable))
            result_payload = start_daemon(
                host=getattr(args, "host", "127.0.0.1"),
                port=getattr(args, "port", 8000),
                workspace=getattr(args, "workspace", None),
                python_executable=selection.selected_python,
            )
            print(json.dumps(result_payload, indent=2))
            return 0 if result_payload.get("status") in ("started", "already_running", "starting") else 1

        if args.daemon_command == "stop":
            print(json.dumps(stop_daemon(), indent=2))
            return 0

        if args.daemon_command == "status":
            print(json.dumps(daemon_status(port=getattr(args, "port", 8000)), indent=2))
            return 0

        if args.daemon_command == "submit":
            result_payload = submit_task_to_daemon(
                args.task,
                workspace=getattr(args, "workspace", None),
                port=getattr(args, "port", 8000),
                execution_mode=getattr(args, "execution_mode", "read_only"),
                max_rounds=getattr(args, "max_rounds", None),
            )
            print(json.dumps(result_payload, indent=2))
            return 0 if result_payload.get("status") != "error" else 1

        if args.daemon_command == "job":
            print(json.dumps(get_daemon_job(args.job_id, port=getattr(args, "port", 8000)), indent=2))
            return 0

        if args.daemon_command == "install":
            from .daemon import install_launchd

            project_root = Path.cwd().resolve()
            selection = select_runtime_python(project_root, current_python=Path(sys.executable))
            result_payload = install_launchd(
                host=getattr(args, "host", "127.0.0.1"),
                port=getattr(args, "port", 8000),
                workspace=getattr(args, "workspace", None),
                python_executable=selection.selected_python,
            )
            print(json.dumps(result_payload, indent=2))
            return 0 if result_payload.get("status") != "error" else 1

        if args.daemon_command == "uninstall":
            from .daemon import uninstall_launchd

            print(json.dumps(uninstall_launchd(), indent=2))
            return 0

        if args.daemon_command == "schedule":
            from .scheduler import add_schedule, load_schedules, remove_schedule

            if args.schedule_command == "add":
                try:
                    entry = add_schedule(
                        task=args.task,
                        cron=args.cron,
                        schedule_id=getattr(args, "schedule_id", None),
                        execution_mode=getattr(args, "execution_mode", "read_only"),
                        workspace=getattr(args, "workspace", None),
                        max_rounds=getattr(args, "max_rounds", None),
                    )
                    print(json.dumps(entry.to_dict(), indent=2))
                except ValueError as exc:
                    print(json.dumps({"error": str(exc)}, indent=2))
                    return 1
                return 0

            if args.schedule_command == "list":
                entries = load_schedules()
                print(json.dumps([e.to_dict() for e in entries], indent=2))
                return 0

            if args.schedule_command == "remove":
                removed = remove_schedule(args.schedule_id)
                if removed:
                    print(json.dumps({"status": "removed", "id": args.schedule_id}, indent=2))
                else:
                    print(json.dumps({"error": f"Schedule not found: {args.schedule_id!r}"}, indent=2))
                    return 1
                return 0

    if args.command == "team":
        from .config import Settings
        from .spawn import AgentTeam

        settings = Settings.from_env()

        def _team_progress(msg: str) -> None:
            print(f"[teamai:team] {msg}", file=sys.stderr, flush=True)

        team = AgentTeam(
            settings=settings,
            progress_callback=_team_progress,
            max_tasks=getattr(args, "max_tasks", 12),
        )

        if args.team_command == "plan":
            plan = team.decompose(args.goal, workspace_path=getattr(args, "workspace", None))
            print(json.dumps(plan.model_dump(mode="json"), indent=2))
            return 0

        if args.team_command == "run":
            plan = team.run(
                args.goal,
                workspace_path=getattr(args, "workspace", None),
            )
            print(json.dumps(plan.model_dump(mode="json"), indent=2))
            return 0 if plan.status == "completed" else 1

    if args.command == "agents":
        from .agent_registry import AgentRegistry

        registry = AgentRegistry.load()

        if args.agents_command == "list":
            if getattr(args, "format", "table") == "json":
                payload = [
                    {
                        "id": a.id,
                        "name": a.name,
                        "type": a.type,
                        "model_id": a.model_id,
                        "capabilities": a.capabilities,
                        "cost": a.cost,
                        "latency": a.latency,
                        "env_satisfied": a.env_satisfied(),
                        "requires_env": a.requires_env,
                        "notes": a.notes,
                    }
                    for a in registry.agents
                ]
                print(json.dumps(payload, indent=2))
            else:
                print(registry.render_table())
            return 0

        if args.agents_command == "pick":
            agent = registry.pick_best(
                args.capability,
                prefer_local=getattr(args, "prefer_local", False),
                env_check=not getattr(args, "no_env_check", False),
            )
            if agent is None:
                print(json.dumps({"error": f"No agent found for capability: {args.capability!r}"}))
                return 1
            print(json.dumps({
                "id": agent.id,
                "name": agent.name,
                "type": agent.type,
                "model_id": agent.model_id,
                "cost": agent.cost,
                "latency": agent.latency,
            }, indent=2))
            return 0

    parser.error("Unknown command")
    return 2


def _resolve_cli_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else project_root / path


def _dashboard_url(host: str, port: int) -> str:
    browser_host = host.strip() or "127.0.0.1"
    if browser_host in {"0.0.0.0", "::"}:
        browser_host = "127.0.0.1"
    return f"http://{browser_host}:{port}/dashboard"


def _schedule_dashboard_open(url: str) -> None:
    import threading
    import webbrowser

    timer = threading.Timer(0.8, lambda: webbrowser.open(url))
    timer.daemon = True
    timer.start()


def _write_cli_output(*, rendered_output: str, output_file: str | None) -> None:
    if not output_file:
        return

    output_path = Path(output_file).expanduser()
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered_output + ("\n" if not rendered_output.endswith("\n") else ""), encoding="utf-8")


def _apply_approval_and_continue(
    *,
    settings: Any,
    workspace: Path,
    workspace_path: str | None,
    approval_id: str,
    max_rounds: int | None,
    max_actions: int | None,
    max_tokens: int | None,
    temperature: float | None,
    progress_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    from .approvals import PatchApprovalStore
    from .schemas import RunRequest
    from .supervisor import ClosedLoopSupervisor

    store = PatchApprovalStore()
    applied = store.apply(workspace=workspace, approval_id=approval_id)
    approval_summary = store.summarize(applied, include_diff=True)
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
            workspace_path=workspace_path,
            max_rounds=max_rounds,
            max_actions_per_round=max_actions,
            max_tokens_per_turn=max_tokens,
            temperature=temperature,
            execution_mode=continuation_mode,
            continuation_context=continuation_context,
        ),
        progress_callback=progress_callback,
    )
    return {
        "approval": applied,
        "approval_summary": approval_summary,
        "continuation_task": continuation_task,
        "continuation_context": continuation_context,
        "continuation_result": continuation_result,
    }


def _execute_verified_handoff_workflow(
    *,
    project_root: Path,
    engine: str,
    payload_file: str | Path,
    patch_file: str | Path,
    model: str | None = None,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    from .integrations import get_bridge

    bridge = get_bridge(engine)
    verified_result = bridge.execute_verified(
        project_root=project_root,
        payload_file=payload_file,
        patch_file=patch_file,
        model=model,
    )
    payload_path = verified_result.execution.payload_file
    patch_path = verified_result.execution.patch_file
    model_name = verified_result.execution.model
    patch_text = verified_result.execution.patch_text
    verification_result = verified_result.verification
    failure_context_path = verified_result.failure_context_file
    approval_payload = getattr(verified_result, "approval", None)
    approval_error = getattr(verified_result, "approval_error", None)

    failure_context_path = _sync_failure_context_log(
        verification_result=verification_result,
        failure_context_path=failure_context_path,
    )
    rendered_summary = _render_execute_handoff_summary(
        engine=engine,
        model=model_name,
        payload_path=payload_path,
        patch_path=patch_path,
        patch_text=patch_text,
        verification_result=verification_result,
        failure_context_path=failure_context_path,
        approval_payload=approval_payload,
        approval_error=approval_error,
    )
    return {
        "engine": engine,
        "model": model_name,
        "payload_path": str(payload_path),
        "patch_path": str(patch_path),
        "verification": {
            "success": bool(getattr(verification_result, "success", False)),
            "patch_returncode": getattr(verification_result, "patch_returncode", None),
            "test_returncode": getattr(verification_result, "test_returncode", None),
            "commands_run": list(getattr(verification_result, "commands_run", ()) or ()),
        },
        "approval": approval_payload,
        "approval_error": approval_error,
        "failure_context_path": None if failure_context_path is None else str(failure_context_path),
        "summary": rendered_summary,
        "success": bool(getattr(verification_result, "success", False)),
    }


def _run_autopilot_workflow(
    *,
    project_root: Path,
    settings: Any,
    initial_result: RunResult,
    initial_payload_path: Path | None,
    workspace_path: str | None,
    max_rounds: int | None,
    max_actions: int | None,
    max_tokens: int | None,
    temperature: float | None,
    handoff_engine: str,
    handoff_model: str | None,
    handoff_patch_file: str | Path,
    max_cycles: int,
) -> dict[str, Any]:
    cycles: list[dict[str, Any]] = []
    current_result = initial_result
    payload_path = initial_payload_path

    if current_result.status == "failed":
        autopilot = {
            "requested_cycles": max_cycles,
            "completed_cycles": 0,
            "status": "failed",
            "stop_reason": "initial_run_failed",
            "cycles": cycles,
            "final_result": current_result.model_dump(mode="json"),
            "success": False,
        }
        autopilot["summary"] = _render_autopilot_summary(autopilot)
        return autopilot

    final_status = "stopped"
    stop_reason = "max_autopilot_cycles_reached"
    success = False

    def emit(message: str) -> None:
        print(f"[teamai][autopilot] {message}", file=sys.stderr, flush=True)

    for cycle_index in range(1, max_cycles + 1):
        if payload_path is None:
            payload_path = _write_codex_payload_artifact(current_result)
            if payload_path is not None:
                emit(f"Wrote semantic skeleton to {payload_path}")

        if payload_path is None:
            if current_result.status == "completed":
                final_status = "completed"
                stop_reason = "completed_without_handoff"
                success = True
            elif current_result.stop_reason == "approval_required":
                final_status = "stopped"
                stop_reason = "manual_approval_required"
            else:
                final_status = current_result.status
                stop_reason = current_result.stop_reason
            break

        emit(f"Cycle {cycle_index}/{max_cycles}: executing verified {handoff_engine} handoff")
        cycle_record: dict[str, Any] = {
            "cycle": cycle_index,
            "source_result": {
                "status": current_result.status,
                "task_route": current_result.task_route,
                "stop_reason": current_result.stop_reason,
            },
        }
        try:
            handoff = _execute_verified_handoff_workflow(
                project_root=project_root,
                engine=handoff_engine,
                payload_file=payload_path,
                patch_file=handoff_patch_file,
                model=handoff_model,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            handoff = {
                "engine": handoff_engine,
                "model": handoff_model,
                "payload_path": str(payload_path),
                "patch_path": str(_resolve_cli_path(project_root, str(handoff_patch_file))),
                "verification": {
                    "success": False,
                    "patch_returncode": None,
                    "test_returncode": None,
                    "commands_run": [],
                },
                "approval": None,
                "approval_error": None,
                "failure_context_path": None,
                "summary": f"Autopilot handoff failed: {exc}",
                "success": False,
                "error": str(exc),
            }
        cycle_record["handoff"] = handoff

        if not bool(handoff.get("success", False)):
            cycles.append(cycle_record)
            final_status = "failed"
            stop_reason = "verified_handoff_failed"
            break

        approval_payload = handoff.get("approval")
        approval_id = ""
        if isinstance(approval_payload, dict):
            approval_id = str(approval_payload.get("approval_id", "")).strip()
        if not approval_id:
            cycles.append(cycle_record)
            final_status = "failed"
            stop_reason = "verified_handoff_missing_approval"
            break

        workspace = Path(current_result.workspace).expanduser()
        emit(f"Cycle {cycle_index}/{max_cycles}: applying verified approval {approval_id} and continuing")
        try:
            continuation = _apply_approval_and_continue(
                settings=settings,
                workspace=workspace,
                workspace_path=workspace_path or str(workspace),
                approval_id=approval_id,
                max_rounds=max_rounds,
                max_actions=max_actions,
                max_tokens=max_tokens,
                temperature=temperature,
                progress_callback=emit,
            )
        except (KeyError, ValueError, RuntimeError) as exc:
            cycle_record["approval_apply_error"] = str(exc)
            cycles.append(cycle_record)
            final_status = "failed"
            stop_reason = "approval_apply_failed"
            break

        continuation_result = continuation["continuation_result"]
        cycle_record["approval"] = continuation["approval_summary"]
        cycle_record["continuation_task"] = continuation["continuation_task"]
        cycle_record["continuation_context"] = continuation["continuation_context"]
        cycle_record["continuation"] = continuation_result.model_dump(mode="json")
        cycles.append(cycle_record)

        current_result = continuation_result
        payload_path = None

        if current_result.status == "completed" and current_result.codex_payload is None:
            final_status = "completed"
            stop_reason = "task_completed"
            success = True
            break
        if current_result.status == "failed":
            final_status = "failed"
            stop_reason = f"continuation_{current_result.stop_reason}"
            break
        if current_result.stop_reason == "approval_required":
            final_status = "stopped"
            stop_reason = "manual_approval_required"
            break
    else:
        final_status = "stopped"
        stop_reason = "max_autopilot_cycles_reached"

    autopilot = {
        "requested_cycles": max_cycles,
        "completed_cycles": len(cycles),
        "status": final_status,
        "stop_reason": stop_reason,
        "cycles": cycles,
        "final_result": current_result.model_dump(mode="json"),
        "success": success,
    }
    autopilot["summary"] = _render_autopilot_summary(autopilot)
    return autopilot


def _render_autopilot_summary(payload: dict[str, Any]) -> str:
    lines = [
        "Autopilot summary",
        f"- Status: {payload.get('status', 'unknown')} ({payload.get('stop_reason', 'unknown')})",
        f"- Requested cycles: {payload.get('requested_cycles', 0)}",
        f"- Completed cycles: {payload.get('completed_cycles', 0)}",
    ]
    final_result = payload.get("final_result")
    if isinstance(final_result, dict):
        lines.append(
            f"- Final result: {final_result.get('status', 'unknown')} ({final_result.get('stop_reason', 'unknown')})"
        )

    cycles = payload.get("cycles", [])
    if isinstance(cycles, list) and cycles:
        for cycle in cycles:
            if not isinstance(cycle, dict):
                continue
            cycle_number = cycle.get("cycle", "?")
            handoff = cycle.get("handoff", {})
            approval = cycle.get("approval", {})
            continuation = cycle.get("continuation", {})
            approval_id = ""
            if isinstance(approval, dict):
                approval_id = str(approval.get("approval_id", "")).strip()
            handoff_status = "passed" if isinstance(handoff, dict) and handoff.get("success") else "failed"
            continuation_status = ""
            if isinstance(continuation, dict):
                continuation_status = (
                    f" -> continuation {continuation.get('status', 'unknown')} ({continuation.get('stop_reason', 'unknown')})"
                )
            approval_text = f" -> approval {approval_id} applied" if approval_id else ""
            lines.append(
                f"- Cycle {cycle_number}: handoff {handoff_status}{approval_text}{continuation_status}"
            )
    else:
        lines.append("- No autopilot cycles ran.")

    return "\n".join(lines)


def _render_execute_handoff_summary(
    *,
    engine: str,
    model: str,
    payload_path: Path,
    patch_path: Path,
    patch_text: str,
    verification_result: object,
    failure_context_path: Path | None,
    approval_payload: dict[str, object] | None = None,
    approval_error: str | None = None,
) -> str:
    success = bool(getattr(verification_result, "success", False))
    patch_returncode = getattr(verification_result, "patch_returncode", None)
    test_returncode = getattr(verification_result, "test_returncode", None)
    patch_lines = len(patch_text.splitlines()) if patch_text.strip() else 0
    patch_files = _count_patch_files(patch_text)

    lines = [
        "Handoff execution summary",
        f"- Engine: {engine}",
        f"- Model: {model}",
        f"- Payload: {payload_path}",
        f"- Patch: {patch_path}",
        f"- Patch files: {patch_files}",
        f"- Patch lines: {patch_lines}",
        f"- Sandbox verification: {'passed' if success else 'failed'}",
        f"- Verification detail: {_describe_verification_outcome(patch_returncode=patch_returncode, test_returncode=test_returncode)}",
    ]
    if patch_returncode is not None:
        lines.append(f"- Patch apply exit code: {patch_returncode}")
    if test_returncode is None:
        lines.append("- Test run: not executed")
    else:
        lines.append(f"- Test exit code: {test_returncode}")

    if approval_payload is not None:
        approval_id = str(approval_payload.get("approval_id", "")).strip()
        change_count = int(approval_payload.get("change_count", 1) or 1)
        lines.append(f"- Approval: {approval_id}")
        lines.append(f"- Approval scope: {change_count} file(s)")
    elif approval_error:
        lines.append(f"- Approval creation: failed ({approval_error})")

    if failure_context_path is not None:
        lines.append(f"- Failure log: {failure_context_path}")
    lines.append(
        f"- Next step: {_render_execute_handoff_next_step(success=success, patch_returncode=patch_returncode, test_returncode=test_returncode, approval_payload=approval_payload, approval_error=approval_error)}"
    )
    return "\n".join(lines)


def _sync_failure_context_log(*, verification_result: object, failure_context_path: Path) -> Path | None:
    success = bool(getattr(verification_result, "success", False))
    log_output = str(getattr(verification_result, "log_output", "") or "").strip()

    if success:
        if failure_context_path.exists():
            failure_context_path.unlink()
        return None

    failure_context_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_log = log_output or _render_missing_verification_log(
        patch_returncode=getattr(verification_result, "patch_returncode", None),
        test_returncode=getattr(verification_result, "test_returncode", None),
    )
    failure_context_path.write_text(rendered_log + ("\n" if not rendered_log.endswith("\n") else ""), encoding="utf-8")
    return failure_context_path


def _render_missing_verification_log(*, patch_returncode: int | None, test_returncode: int | None) -> str:
    lines = [
        "Verification failed without captured sandbox output.",
        f"patch_returncode={patch_returncode}",
        f"test_returncode={test_returncode}",
    ]
    return "\n".join(lines)


def _count_patch_files(patch_text: str) -> int:
    diff_headers = re.findall(r"^diff --git a/(.+?) b/", patch_text, flags=re.MULTILINE)
    if diff_headers:
        return len(dict.fromkeys(diff_headers))

    file_headers = re.findall(r"^\+\+\+ b/(.+)$", patch_text, flags=re.MULTILINE)
    return len(dict.fromkeys(file_headers))


def _describe_verification_outcome(*, patch_returncode: int | None, test_returncode: int | None) -> str:
    if patch_returncode not in (None, 0):
        return "patch apply failed before tests ran"
    if test_returncode is None:
        return "patch applied, but no sandbox test run was recorded"
    if test_returncode == 0:
        return "patch applied and sandbox tests passed"
    return "patch applied, but sandbox tests failed"


def _render_execute_handoff_next_step(
    *,
    success: bool,
    patch_returncode: int | None,
    test_returncode: int | None,
    approval_payload: dict[str, object] | None = None,
    approval_error: str | None = None,
) -> str:
    if success and approval_payload is not None:
        approval_id = str(approval_payload.get("approval_id", "")).strip() or "<approval_id>"
        return (
            f"review the verified patch with `teamai approvals show {approval_id}` "
            f"and apply it with `teamai approvals apply {approval_id}`."
        )
    if success and approval_error:
        return "patch verified successfully, but approval creation failed. Inspect the patch manually before applying it."
    if success:
        return "patch verified successfully in sandbox. Ready for human review."
    if patch_returncode not in (None, 0):
        return "inspect the returned diff and patch-apply errors before retrying."
    if test_returncode not in (None, 0):
        return "inspect the returned diff and sandbox test failures before retrying."
    return "inspect the patch and failure log before retrying."


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
