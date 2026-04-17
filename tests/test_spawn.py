from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from teamai.schemas import RunResult
from teamai.spawn import AgentSpawner, AgentTeam


class _TemplateSupervisor:
    def __init__(self) -> None:
        self.clone_count = 0

    def isolated_copy(self):  # noqa: ANN202
        self.clone_count += 1
        return _IsolatedSupervisor(self.clone_count)

    def run(self, request, progress_callback=None, event_callback=None):  # noqa: ANN001
        raise AssertionError("template supervisor should not execute runs directly")


class _IsolatedSupervisor:
    def __init__(self, clone_id: int) -> None:
        self._clone_id = clone_id

    def run(self, request, progress_callback=None, event_callback=None):  # noqa: ANN001
        return RunResult(
            status="completed",
            model_id=f"child-{self._clone_id}",
            workspace=request.workspace_path or ".",
            execution_mode=request.execution_mode,
            stop_reason="verifier_declared_complete",
            final_answer=f"child-{self._clone_id}",
            transcript="spawn transcript",
            warnings=[],
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )


class _BadDecompositionSupervisor:
    def generate_raw(self, *, system, user, max_tokens=512, temperature=0.3):  # noqa: ANN001
        return "not valid json"


class AgentSpawnerIsolationTest(unittest.TestCase):
    def test_spawn_sync_uses_isolated_supervisor_copy_per_call(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            template = _TemplateSupervisor()
            spawner = AgentSpawner(supervisor=template)

            first = spawner.spawn_sync("Inspect the repo.", workspace_path=str(workspace))
            second = spawner.spawn_sync("Inspect it again.", workspace_path=str(workspace))

            self.assertEqual(first.final_answer, "child-1")
            self.assertEqual(second.final_answer, "child-2")
            self.assertEqual(template.clone_count, 2)


class AgentTeamDecompositionFallbackTest(unittest.TestCase):
    def test_invalid_decomposition_falls_back_to_single_task(self) -> None:
        team = AgentTeam(supervisor=_BadDecompositionSupervisor())

        plan = team.decompose("Implement the feature.")

        self.assertEqual(len(plan.tasks), 1)
        self.assertEqual(plan.tasks[0].spawn_id, "t1")
        self.assertEqual(plan.tasks[0].task, "Implement the feature.")
        self.assertEqual(plan.tasks[0].status, "pending")


if __name__ == "__main__":
    unittest.main()
