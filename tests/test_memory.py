from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from teamai.config import Settings
from teamai.memory import WorkspaceMemoryStore
from teamai.model_backend import ModelResponse
from teamai.schemas import PlannerTurn, RoundRecord, RunRequest, ToolExecutionResult, VerifierVerdict
from teamai.supervisor import ClosedLoopSupervisor


class FakeBackend:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses[:]
        self.user_prompts: list[str] = []

    @property
    def model_loaded(self) -> bool:
        return True

    def generate_messages(
        self,
        *,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float,
        enable_thinking: bool | None = None,
    ) -> ModelResponse:
        del max_tokens, temperature, enable_thinking
        self.user_prompts.append(messages[-1]["content"])
        if not self._responses:
            raise AssertionError("No fake responses left for backend.")
        text = self._responses.pop(0)
        return ModelResponse(
            text=text,
            prompt_tokens=1,
            generation_tokens=1,
            total_tokens=2,
            prompt_tps=1.0,
            generation_tps=1.0,
            peak_memory_gb=0.0,
        )


class WorkspaceMemoryStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_persist_run_writes_history_and_memory_files(self) -> None:
        store = WorkspaceMemoryStore()
        completed_at = datetime(2026, 4, 4, 20, 45, tzinfo=timezone.utc)

        store.persist_run(
            workspace=self.workspace,
            task="Inspect this repository and identify the next engineering tasks.",
            status="completed",
            stop_reason="inspection_synthesized",
            final_answer=(
                "Current state: The repo is stable and ready for the next milestone.\n\n"
                "Next engineering tasks:\n"
                "- Add persistent run history and memory.\n"
                "- Add streaming event output.\n"
            ),
            warnings=["minor warning"],
            completed_at=completed_at,
            model_id="dummy-model",
            task_route="codex_handoff",
        )

        state_dir = self.workspace / ".teamai"
        history_path = state_dir / "run-history.jsonl"
        memory_path = state_dir / "memory.md"

        self.assertTrue(history_path.exists())
        self.assertTrue(memory_path.exists())

        records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["status"], "completed")
        self.assertEqual(records[0]["task_route"], "codex_handoff")
        self.assertEqual(records[0]["next_tasks"], ["Add persistent run history and memory.", "Add streaming event output."])
        self.assertIn("improvement_notes", records[0])
        self.assertIn("Codex handoff", " ".join(records[0]["improvement_notes"]))

        memory_text = memory_path.read_text(encoding="utf-8")
        self.assertIn("# Workspace Memory", memory_text)
        self.assertIn("## Local Improvement Notes", memory_text)
        self.assertIn("## Open Tasks", memory_text)
        self.assertIn("Add persistent run history and memory.", memory_text)
        self.assertIn("Codex handoff", memory_text)

    def test_load_snapshot_returns_recent_runs_summary(self) -> None:
        store = WorkspaceMemoryStore()
        store.persist_run(
            workspace=self.workspace,
            task="Task 1",
            status="completed",
            stop_reason="codex_handoff_synthesized",
            final_answer=(
                "Current state: Summary 1.\n\n"
                "Next engineering tasks:\n"
                "- Follow-up task 1.\n"
            ),
            warnings=[],
            completed_at=datetime(2026, 4, 4, 21, 0, tzinfo=timezone.utc),
            model_id="dummy-model",
            task_route="codex_handoff",
        )
        store.persist_run(
            workspace=self.workspace,
            task="Task 2",
            status="stopped",
            stop_reason="approval_required",
            final_answer=(
                "Current state: Summary 2.\n\n"
                "Next engineering tasks:\n"
                "- Follow-up task 2.\n"
            ),
            warnings=[],
            completed_at=datetime(2026, 4, 4, 21, 1, tzinfo=timezone.utc),
            model_id="dummy-model",
            task_route="deterministic_patch",
            execution_mode="workspace_write",
        )

        snapshot = store.load_snapshot(self.workspace)

        self.assertIn("# Workspace Memory", snapshot.memory_text)
        self.assertIn("Task 2", snapshot.memory_text)
        self.assertIn("route=codex_handoff", snapshot.recent_runs_text)
        self.assertIn("route=deterministic_patch", snapshot.recent_runs_text)
        self.assertIn("Summary 2", snapshot.recent_runs_text)
        self.assertIn("Latest route outcome: deterministic_patch -> approval_required.", snapshot.improvement_notes_text)
        self.assertIn("Codex handoff", snapshot.improvement_notes_text)

    def test_improvement_notes_prioritize_recurring_high_value_behaviors(self) -> None:
        store = WorkspaceMemoryStore()
        store.persist_run(
            workspace=self.workspace,
            task="Improve the API streaming design.",
            status="completed",
            stop_reason="codex_handoff_synthesized",
            final_answer="Current state: Recon complete.",
            warnings=[],
            completed_at=datetime(2026, 4, 4, 21, 0, tzinfo=timezone.utc),
            model_id="dummy-model",
            task_route="codex_handoff",
        )
        store.persist_run(
            workspace=self.workspace,
            task="Harden a broad implementation plan.",
            status="completed",
            stop_reason="codex_handoff_synthesized",
            final_answer="Current state: Recon complete again.",
            warnings=[],
            completed_at=datetime(2026, 4, 4, 21, 1, tzinfo=timezone.utc),
            model_id="dummy-model",
            task_route="codex_handoff",
        )
        store.persist_run(
            workspace=self.workspace,
            task="Continue a stalled local loop.",
            status="stopped",
            stop_reason="max_rounds_reached",
            final_answer="Current state: The run stalled.",
            warnings=["Planner had no novel actions; used heuristic fallback action synthesis."],
            completed_at=datetime(2026, 4, 4, 21, 2, tzinfo=timezone.utc),
            model_id="dummy-model",
            task_route="multi_agent_loop",
        )

        snapshot = store.load_snapshot(self.workspace)
        note_lines = [line for line in snapshot.improvement_notes_text.splitlines() if line.startswith("- ")]

        self.assertGreaterEqual(len(note_lines), 1)
        self.assertIn("Codex handoff", note_lines[0])

    def test_task_aware_snapshot_prioritizes_repository_inspection_lessons(self) -> None:
        store = WorkspaceMemoryStore()
        store.persist_run(
            workspace=self.workspace,
            task="Implement a broad feature across the CLI and API.",
            status="completed",
            stop_reason="codex_handoff_synthesized",
            final_answer="Current state: Broad recon complete.",
            warnings=[],
            completed_at=datetime(2026, 4, 4, 21, 0, tzinfo=timezone.utc),
            model_id="dummy-model",
            task_route="codex_handoff",
        )
        store.persist_run(
            workspace=self.workspace,
            task="Inspect this repository and identify the next engineering tasks.",
            status="completed",
            stop_reason="inspection_synthesized",
            final_answer="Current state: Inspection complete.",
            warnings=[],
            completed_at=datetime(2026, 4, 4, 21, 1, tzinfo=timezone.utc),
            model_id="dummy-model",
            task_route="repository_inspection",
        )

        snapshot = store.load_snapshot(
            self.workspace,
            task="Inspect this repository and identify the next engineering tasks.",
            task_route="repository_inspection",
        )

        note_lines = [line for line in snapshot.improvement_notes_text.splitlines() if line.startswith("- ")]
        self.assertIn("inspection lessons first", snapshot.improvement_notes_text)
        self.assertTrue(note_lines)
        self.assertIn("repository inspection", note_lines[0].lower())

    def test_task_aware_snapshot_prioritizes_patch_and_verification_for_continuation(self) -> None:
        store = WorkspaceMemoryStore()
        verification_round = RoundRecord(
            round_number=0,
            strategist="probe",
            critic="probe",
            planner=PlannerTurn(summary="run a focused unittest", should_stop=False, final_answer=None, actions=[]),
            tool_results=[
                ToolExecutionResult(
                    tool="run_command",
                    success=True,
                    output="ok",
                    metadata={"command": "python -m unittest tests.test_memory"},
                )
            ],
            verifier=VerifierVerdict(done=False, confidence=0.7, summary="verification focused", next_focus="continue"),
        )
        store.persist_run(
            workspace=self.workspace,
            task="Use workspace_write mode. Replace the text 'a' with 'b' in demo.txt.",
            status="stopped",
            stop_reason="approval_required",
            final_answer="Current state: Patch is pending approval.",
            warnings=[],
            completed_at=datetime(2026, 4, 4, 21, 2, tzinfo=timezone.utc),
            model_id="dummy-model",
            task_route="deterministic_patch",
            execution_mode="workspace_write",
            rounds=[verification_round],
        )
        store.persist_run(
            workspace=self.workspace,
            task="Implement a broad feature.",
            status="completed",
            stop_reason="codex_handoff_synthesized",
            final_answer="Current state: Recon complete.",
            warnings=[],
            completed_at=datetime(2026, 4, 4, 21, 3, tzinfo=timezone.utc),
            model_id="dummy-model",
            task_route="codex_handoff",
        )

        snapshot = store.load_snapshot(
            self.workspace,
            task="Continue the previously approved write task and verify it.",
            task_route="deterministic_patch",
            continuation_context={"approval_id": "demo", "verification_focus": "Verify the changed file."},
        )

        note_lines = [line for line in snapshot.improvement_notes_text.splitlines() if line.startswith("- ")]
        self.assertIn("patch and approval lessons first", snapshot.improvement_notes_text)
        self.assertIn("verification and continuation lessons first", snapshot.improvement_notes_text)
        self.assertTrue(note_lines)
        self.assertTrue(
            "approval_required" in note_lines[0].lower()
            or "approved patch" in note_lines[0].lower()
            or "deterministic patch route" in note_lines[0].lower()
        )

    def test_stale_specialized_notes_prune_when_newer_route_patterns_dominate(self) -> None:
        store = WorkspaceMemoryStore()
        store.persist_run(
            workspace=self.workspace,
            task="Implement a broad CLI change.",
            status="completed",
            stop_reason="codex_handoff_synthesized",
            final_answer="Current state: Broad recon complete.",
            warnings=[],
            completed_at=datetime(2026, 4, 4, 21, 0, tzinfo=timezone.utc),
            model_id="dummy-model",
            task_route="codex_handoff",
        )
        store.persist_run(
            workspace=self.workspace,
            task="Harden a broad API plan.",
            status="completed",
            stop_reason="codex_handoff_synthesized",
            final_answer="Current state: Broad recon complete again.",
            warnings=[],
            completed_at=datetime(2026, 4, 4, 21, 1, tzinfo=timezone.utc),
            model_id="dummy-model",
            task_route="codex_handoff",
        )
        store.persist_run(
            workspace=self.workspace,
            task="Use workspace_write mode. Replace the text 'a' with 'b' in demo.txt.",
            status="stopped",
            stop_reason="approval_required",
            final_answer="Current state: Patch is pending approval.",
            warnings=[],
            completed_at=datetime(2026, 4, 4, 21, 2, tzinfo=timezone.utc),
            model_id="dummy-model",
            task_route="deterministic_patch",
            execution_mode="workspace_write",
        )
        store.persist_run(
            workspace=self.workspace,
            task="Use workspace_write mode. Replace the text 'b' with 'c' in demo.txt.",
            status="stopped",
            stop_reason="approval_required",
            final_answer="Current state: Newer patch is pending approval.",
            warnings=[],
            completed_at=datetime(2026, 4, 4, 21, 3, tzinfo=timezone.utc),
            model_id="dummy-model",
            task_route="deterministic_patch",
            execution_mode="workspace_write",
        )

        snapshot = store.load_snapshot(
            self.workspace,
            task="Continue the previously approved write task and verify it.",
            task_route="deterministic_patch",
            continuation_context={"approval_id": "demo", "verification_focus": "Verify the changed file."},
        )

        note_lines = [line for line in snapshot.improvement_notes_text.splitlines() if line.startswith("- ")]
        rendered_notes = "\n".join(note_lines)
        self.assertIn("approval_required", rendered_notes)
        self.assertNotIn("Codex handoff", rendered_notes)


class SupervisorMemoryIntegrationTest(unittest.TestCase):
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
            max_rounds=1,
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

    def test_second_run_includes_persisted_memory_in_prompt(self) -> None:
        first_backend = FakeBackend(
            [
                "Inspect the repo.",
                "That is enough context.",
                (
                    '{"summary":"Complete.","should_stop":true,'
                    '"final_answer":"Current state: The repo has a working closed loop.\\n\\n'
                    'Next engineering tasks:\\n'
                    '- Add persistent run history and memory.\\n'
                    '- Add streaming event output.\\n",'
                    '"actions":[]}'
                ),
                '{"done":false,"confidence":0.2,"summary":"Planner already completed.","next_focus":"None."}',
            ]
        )

        ClosedLoopSupervisor(self.settings, backend=first_backend).run(
            RunRequest(task="Inspect this repository and identify the next engineering tasks.", workspace_path="."),
        )

        state_dir = self.workspace / ".teamai"
        self.assertTrue((state_dir / "run-history.jsonl").exists())
        self.assertTrue((state_dir / "memory.md").exists())

        second_backend = FakeBackend(
            [
                "Continue from the saved context.",
                "Proceed carefully.",
                (
                    '{"summary":"Complete again.","should_stop":true,'
                    '"final_answer":"Current state: Follow-up complete.","actions":[]}'
                ),
                '{"done":false,"confidence":0.2,"summary":"Planner already completed.","next_focus":"None."}',
            ]
        )

        ClosedLoopSupervisor(self.settings, backend=second_backend).run(
            RunRequest(task="Continue the previous engineering work.", workspace_path="."),
        )

        first_prompt = second_backend.user_prompts[0]
        self.assertIn("Persistent workspace memory:", first_prompt)
        self.assertIn("Recent persisted runs:", first_prompt)
        self.assertIn("Local improvement notes:", first_prompt)
        self.assertIn("Add persistent run history and memory.", first_prompt)
        self.assertIn("Inspect this repository and identify the next engineering tasks.", first_prompt)

    def test_second_run_includes_recent_improvement_notes_in_prompt(self) -> None:
        store = WorkspaceMemoryStore()
        store.persist_run(
            workspace=self.workspace,
            task="Improve the CLI and API streaming flow.",
            status="completed",
            stop_reason="codex_handoff_synthesized",
            final_answer=(
                "Current state: Broad implementation work needs a narrower execution plan.\n\n"
                "Next engineering tasks:\n"
                "- Stream supervisor progress through the CLI and API.\n"
            ),
            warnings=[],
            completed_at=datetime(2026, 4, 4, 21, 5, tzinfo=timezone.utc),
            model_id="dummy-model",
            task_route="codex_handoff",
        )

        backend = FakeBackend(
            [
                "Use the saved learning notes.",
                "Stay focused.",
                (
                    '{"summary":"Complete.","should_stop":true,'
                    '"final_answer":"Current state: Follow-up complete.","actions":[]}'
                ),
                '{"done":false,"confidence":0.2,"summary":"Planner already completed.","next_focus":"None."}',
            ]
        )

        ClosedLoopSupervisor(self.settings, backend=backend).run(
            RunRequest(task="Continue from the saved broad-task context.", workspace_path="."),
        )

        first_prompt = backend.user_prompts[0]
        self.assertIn("Local improvement notes:", first_prompt)
        self.assertIn("read-only reconnaissance plus a Codex handoff", first_prompt)


if __name__ == "__main__":
    unittest.main()
