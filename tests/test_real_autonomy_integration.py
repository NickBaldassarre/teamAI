from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from teamai.config import Settings
from teamai.schemas import RunRequest
from teamai.supervisor import ClosedLoopSupervisor


def _create_real_autonomy_fixture(workspace: Path) -> None:
    (workspace / "calc.py").write_text(
        "def answer() -> int:\n    return 0\n",
        encoding="utf-8",
    )
    (workspace / "tests").mkdir(exist_ok=True)
    (workspace / "tests" / "test_calc.py").write_text(
        "import unittest\n"
        "from calc import answer\n\n\n"
        "class CalcTest(unittest.TestCase):\n"
        "    def test_answer(self) -> None:\n"
        "        self.assertEqual(answer(), 42)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init"], cwd=workspace, capture_output=True, text=True, check=True)
    subprocess.run(["git", "config", "user.email", "teamai@example.com"], cwd=workspace, capture_output=True, text=True, check=True)
    subprocess.run(["git", "config", "user.name", "teamAI"], cwd=workspace, capture_output=True, text=True, check=True)
    subprocess.run(["git", "add", "calc.py", "tests/test_calc.py"], cwd=workspace, capture_output=True, text=True, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=workspace, capture_output=True, text=True, check=True)


class RealModelAutonomyIntegrationTest(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("TEAMAI_RUN_REAL_MLX_TESTS") == "1",
        "Set TEAMAI_RUN_REAL_MLX_TESTS=1 to run the real end-to-end autonomy integration test.",
    )
    def test_real_model_autonomous_loop_can_fix_bug_and_commit(self) -> None:
        with tempfile.TemporaryDirectory(prefix="teamai-real-autonomy-") as temp_dir:
            workspace = Path(temp_dir)
            _create_real_autonomy_fixture(workspace)

            base_settings = Settings.from_env()
            model_override = os.getenv("TEAMAI_REAL_AUTONOMY_MODEL", "").strip()
            settings = replace(
                base_settings,
                model_id=model_override or base_settings.model_id,
                workspace_root=workspace,
                allow_writes=True,
                max_rounds=max(base_settings.max_rounds, 4),
                max_actions_per_round=max(base_settings.max_actions_per_round, 2),
                max_tokens_per_turn=max(base_settings.max_tokens_per_turn, 256),
                model_router=False,
            )

            result = ClosedLoopSupervisor(settings).run(
                RunRequest(
                    task=(
                        "Fix calc.py so answer() returns 42 and the tests pass. "
                        "Only change the minimum necessary code and create a commit if successful."
                    ),
                    workspace_path=".",
                    execution_mode="workspace_write",
                    write_policy="auto_apply_low_risk",
                    auto_commit=True,
                    max_repair_attempts=2,
                )
            )

            diagnostic = "\n".join(
                [
                    f"status={result.status}",
                    f"stop_reason={result.stop_reason}",
                    f"final_answer={result.final_answer}",
                    f"warnings={result.warnings}",
                    f"commit_metadata={result.commit_metadata}",
                ]
            )

            self.assertEqual(result.status, "completed", diagnostic)
            self.assertEqual((workspace / "calc.py").read_text(encoding="utf-8"), "def answer() -> int:\n    return 42\n", diagnostic)
            self.assertTrue(result.commit_metadata, diagnostic)
            assert result.run_state is not None
            self.assertTrue(result.run_state.files_changed, diagnostic)
            self.assertTrue(result.run_state.routing_trace, diagnostic)
            latest_log = subprocess.run(
                ["git", "log", "--oneline", "-1"],
                cwd=workspace,
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            self.assertIn("teamai:", latest_log, diagnostic)


if __name__ == "__main__":
    unittest.main()
