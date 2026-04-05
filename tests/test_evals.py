from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
