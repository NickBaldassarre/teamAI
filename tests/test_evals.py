from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from teamai.config import Settings
from teamai.evals import EvalSuite, load_eval_suite, run_eval_suite
from teamai.schemas import PlannerTurn, RoundRecord, RunRequest, RunResult, ToolExecutionResult, VerifierVerdict


def _build_result(
    *,
    workspace: Path,
    status: str = "completed",
    task_route: str = "multi_agent_loop",
    stop_reason: str = "inspection_synthesized",
    final_answer: str = "Next engineering tasks: strengthen evals.",
    warnings: list[str] | None = None,
    tool_results: list[ToolExecutionResult] | None = None,
) -> RunResult:
    timestamp = datetime.now(timezone.utc)
    return RunResult(
        status=status,  # type: ignore[arg-type]
        model_id="dummy",
        workspace=str(workspace),
        execution_mode="read_only",
        task_route=task_route,
        stop_reason=stop_reason,
        final_answer=final_answer,
        transcript="",
        rounds=[
            RoundRecord(
                round_number=1,
                strategist="inspect",
                critic="verify",
                planner=PlannerTurn(summary="plan", should_stop=False, final_answer=None, actions=[]),
                tool_results=tool_results or [],
                verifier=VerifierVerdict(done=False, confidence=0.5, summary="summary", next_focus="focus"),
            )
        ],
        warnings=warnings or [],
        started_at=timestamp,
        completed_at=timestamp,
    )


class EvalHarnessTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)
        self.settings = Settings(
            model_id="dummy",
            model_revision=None,
            force_download=False,
            trust_remote_code=False,
            enable_thinking=False,
            workspace_root=self.workspace,
            max_rounds=2,
            max_actions_per_round=2,
            max_tokens_per_turn=64,
            temperature=0.3,
            allow_shell=False,
            allow_writes=False,
            command_timeout_seconds=5,
            max_file_bytes=10_000,
            max_command_output_chars=10_000,
            host="127.0.0.1",
            port=8000,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_eval_suite_accepts_top_level_list(self) -> None:
        suite_path = self.workspace / "suite.json"
        suite_path.write_text(
            '[{"case_id":"inspection","task":"Inspect repo.","expectations":{"allowed_stop_reasons":["inspection_synthesized"]}}]\n',
            encoding="utf-8",
        )

        suite = load_eval_suite(suite_path)

        self.assertEqual(suite.name, "eval_suite")
        self.assertEqual(len(suite.cases), 1)
        self.assertEqual(suite.cases[0].case_id, "inspection")

    def test_run_eval_suite_scores_metrics_and_restores_setup_files(self) -> None:
        fixture = self.workspace / ".teamai" / "fixture.md"
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text("original\n", encoding="utf-8")

        approval_tool_result = ToolExecutionResult(
            tool="write_file",
            success=True,
            output="Created pending patch approval abc123.",
            metadata={"approval_id": "abc123"},
        )
        verification_tool_result = ToolExecutionResult(
            tool="run_command",
            success=True,
            output="ok",
            metadata={"command": ["python", "-m", "unittest", "tests.test_memory"], "returncode": 0},
        )

        responses = {
            "inspection": _build_result(workspace=self.workspace),
            "patch": _build_result(
                workspace=self.workspace,
                status="stopped",
                task_route="deterministic_patch",
                stop_reason="approval_required",
                final_answer="Approval required.",
                tool_results=[approval_tool_result, verification_tool_result],
            ),
        }

        def runner(request: RunRequest, settings: Settings) -> RunResult:
            return responses[request.task]

        suite = EvalSuite.model_validate(
            {
                "name": "smoke",
                "cleanup_approvals": False,
                "cases": [
                    {
                        "case_id": "inspection",
                        "task": "inspection",
                        "setup_files": [{"path": ".teamai/fixture.md", "content": "eval\n"}],
                        "expectations": {
                            "allowed_stop_reasons": ["inspection_synthesized"],
                            "local_completion": True,
                            "handoff": False,
                        },
                    },
                    {
                        "case_id": "patch",
                        "task": "patch",
                        "expectations": {
                            "allowed_task_routes": ["deterministic_patch"],
                            "approval_required": True,
                            "verification_success": True,
                        },
                    },
                ],
            }
        )

        report = run_eval_suite(settings=self.settings, suite=suite, runner=runner)

        self.assertEqual(report.metrics.total_cases, 2)
        self.assertEqual(report.metrics.passed_cases, 2)
        self.assertEqual(report.metrics.failed_cases, 0)
        self.assertAlmostEqual(report.metrics.local_completion_rate, 0.5)
        self.assertAlmostEqual(report.metrics.approval_rate, 0.5)
        self.assertAlmostEqual(report.metrics.verification_attempt_rate, 0.5)
        self.assertAlmostEqual(report.metrics.verification_success_rate, 1.0)
        self.assertEqual(fixture.read_text(encoding="utf-8"), "original\n")

        history_path = self.workspace / ".teamai" / "run-history.jsonl"
        self.assertTrue(history_path.exists())
        records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]
        latest = records[-1]
        self.assertEqual(latest["source"], "eval_suite")
        self.assertEqual(latest["task_route"], "eval_feedback")
        self.assertEqual(latest["task"], "Eval suite: smoke")
        self.assertIn("Eval suite `smoke` completed", latest["summary"])
        self.assertTrue(latest["improvement_notes"])

    def test_run_eval_suite_isolated_subprocess_uses_guardrailed_case_runs(self) -> None:
        captured: dict[str, object] = {}

        def subprocess_runner(
            command: list[str],
            env: dict[str, str],
            cwd: Path,
            timeout_seconds: float | None,
        ) -> subprocess.CompletedProcess[str]:
            if "-c" in command:
                captured["preflight_command"] = command
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "status": "healthy",
                            "reason": "mlx_import_ok",
                            "summary": "MLX import preflight passed.",
                        }
                    ),
                    stderr="",
                )
            captured["command"] = command
            captured["env"] = env
            captured["cwd"] = cwd
            captured["timeout_seconds"] = timeout_seconds
            result_path = Path(command[command.index("--output-file") + 1])
            result = _build_result(
                workspace=self.workspace,
                status="completed",
                task_route="codex_handoff",
                stop_reason="codex_handoff_synthesized",
                final_answer="Implement teamai/evals.py improvements.",
            )
            result_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        suite = EvalSuite.model_validate(
            {
                "name": "isolated",
                "cases": [
                    {
                        "case_id": "broad",
                        "task": "Improve the eval harness and benchmark routing behavior.",
                        "expectations": {
                            "allowed_task_routes": ["codex_handoff"],
                            "handoff": True,
                            "handoff_completed": True,
                        },
                    }
                ],
            }
        )

        report = run_eval_suite(
            settings=self.settings,
            suite=suite,
            runner_mode="isolated_subprocess",
            per_case_timeout_seconds=45,
            project_root=self.workspace,
            python_executable=Path("/tmp/fake-python"),
            subprocess_runner=subprocess_runner,
        )

        self.assertEqual(report.metrics.total_cases, 1)
        self.assertEqual(report.metrics.passed_cases, 1)
        self.assertEqual(report.runtime_health.status, "healthy")
        self.assertEqual(report.cases[0].runner_mode, "isolated_subprocess")
        self.assertEqual(report.cases[0].memory_profile, "light_recon")
        self.assertEqual(report.cases[0].failure_classification, "passed")
        self.assertTrue(report.cases[0].guardrail_notes)
        self.assertEqual(Path(captured["cwd"]).resolve(), self.workspace.resolve())
        self.assertEqual(captured["timeout_seconds"], 45)
        command = captured["command"]
        self.assertIsInstance(command, list)
        assert isinstance(command, list)
        self.assertEqual(Path(command[0]).resolve(), Path("/tmp/fake-python").resolve())
        self.assertEqual(command[1:5], ["-u", "-m", "teamai", "run"])
        self.assertIn("--max-rounds", command)
        self.assertIn("2", command)
        self.assertIn("--max-actions", command)
        self.assertIn("--max-tokens", command)
        self.assertIn("128", command)
        self.assertIn("--temperature", command)
        self.assertIn("0.1", command)
        env = captured["env"]
        self.assertIsInstance(env, dict)
        assert isinstance(env, dict)
        self.assertNotEqual(env.get("TEAMAI_ALLOW_WRITES"), "true")

    def test_run_eval_suite_isolated_subprocess_times_out_one_case_and_continues(self) -> None:
        calls: list[str] = []

        def subprocess_runner(
            command: list[str],
            env: dict[str, str],
            cwd: Path,
            timeout_seconds: float | None,
        ) -> subprocess.CompletedProcess[str]:
            if "-c" in command:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "status": "healthy",
                            "reason": "mlx_import_ok",
                            "summary": "MLX import preflight passed.",
                        }
                    ),
                    stderr="",
                )
            task = command[5]
            calls.append(task)
            if task == "timeout-case":
                raise subprocess.TimeoutExpired(cmd=command, timeout=timeout_seconds or 0)
            result_path = Path(command[command.index("--output-file") + 1])
            result = _build_result(workspace=self.workspace)
            result_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        suite = EvalSuite.model_validate(
            {
                "name": "timeouts",
                "cases": [
                    {
                        "case_id": "timeout",
                        "task": "timeout-case",
                    },
                    {
                        "case_id": "inspection",
                        "task": "inspection-case",
                        "expectations": {"allowed_stop_reasons": ["inspection_synthesized"]},
                    },
                ],
            }
        )

        report = run_eval_suite(
            settings=self.settings,
            suite=suite,
            runner_mode="isolated_subprocess",
            per_case_timeout_seconds=12,
            project_root=self.workspace,
            python_executable=Path("/tmp/fake-python"),
            subprocess_runner=subprocess_runner,
        )

        self.assertEqual(calls, ["timeout-case", "inspection-case"])
        self.assertEqual(report.metrics.total_cases, 2)
        self.assertEqual(report.metrics.passed_cases, 1)
        self.assertEqual(report.metrics.failed_cases, 1)
        timeout_case = report.cases[0]
        self.assertFalse(timeout_case.passed)
        self.assertEqual(timeout_case.stop_reason, "case_timeout")
        self.assertEqual(timeout_case.runner_mode, "isolated_subprocess")
        self.assertEqual(timeout_case.failure_classification, "case_timeout")
        self.assertIn("timed out", timeout_case.failures[0].lower())
        self.assertEqual(report.cases[1].case_id, "inspection")
        self.assertTrue(report.cases[1].passed)

    def test_run_eval_suite_isolated_subprocess_injects_write_env_for_allowed_write_case(self) -> None:
        captured_env: dict[str, str] = {}

        def subprocess_runner(
            command: list[str],
            env: dict[str, str],
            cwd: Path,
            timeout_seconds: float | None,
        ) -> subprocess.CompletedProcess[str]:
            if "-c" in command:
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=json.dumps(
                        {
                            "status": "healthy",
                            "reason": "mlx_import_ok",
                            "summary": "MLX import preflight passed.",
                        }
                    ),
                    stderr="",
                )
            captured_env.update(env)
            result_path = Path(command[command.index("--output-file") + 1])
            result = _build_result(
                workspace=self.workspace,
                status="stopped",
                task_route="deterministic_patch",
                stop_reason="approval_required",
                final_answer="Approval required.",
            )
            result_path.write_text(json.dumps(result.model_dump(mode="json"), indent=2), encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        suite = EvalSuite.model_validate(
            {
                "name": "write-env",
                "cases": [
                    {
                        "case_id": "patch",
                        "task": "Append the exact line 'x' to scratch.md.",
                        "execution_mode": "workspace_write",
                        "expectations": {
                            "allowed_task_routes": ["deterministic_patch"],
                            "approval_required": True,
                        },
                    }
                ],
            }
        )

        report = run_eval_suite(
            settings=self.settings,
            suite=suite,
            allow_write_cases=True,
            runner_mode="isolated_subprocess",
            project_root=self.workspace,
            python_executable=Path("/tmp/fake-python"),
            subprocess_runner=subprocess_runner,
        )

        self.assertEqual(report.metrics.passed_cases, 1)
        self.assertEqual(captured_env.get("TEAMAI_ALLOW_WRITES"), "true")

    def test_run_eval_suite_classifies_model_backend_errors_as_infra_runtime_failures(self) -> None:
        def runner(request: RunRequest, settings: Settings) -> RunResult:
            del request, settings
            return _build_result(
                workspace=self.workspace,
                status="failed",
                task_route="repository_inspection",
                stop_reason="model_backend_error",
                final_answer="The local model backend failed before the loop could finish.",
                warnings=["Failed to import MLX runtime. This usually means MLX is not installed correctly or Metal initialization failed on this machine."],
            )

        suite = EvalSuite.model_validate(
            {
                "name": "infra",
                "cases": [
                    {
                        "case_id": "inspection",
                        "task": "Inspect repo.",
                        "expectations": {
                            "allowed_stop_reasons": ["inspection_synthesized"],
                        },
                    }
                ],
            }
        )

        report = run_eval_suite(settings=self.settings, suite=suite, runner=runner)

        self.assertEqual(report.runtime_health.status, "unknown")
        self.assertEqual(report.metrics.failed_cases, 1)
        self.assertEqual(report.metrics.infra_failure_cases, 1)
        self.assertEqual(report.metrics.agent_failure_cases, 0)
        self.assertEqual(report.metrics.actionable_cases, 0)
        self.assertEqual(report.cases[0].failure_classification, "infra_runtime")
        self.assertIn("not scored as an agent-behavior failure", report.cases[0].failures[0].lower())


if __name__ == "__main__":
    unittest.main()
