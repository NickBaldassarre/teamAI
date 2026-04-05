from __future__ import annotations

from datetime import datetime, timezone
import unittest

from teamai.handoff import build_handoff_packet, render_handoff_markdown
from teamai.schemas import (
    PlannerTurn,
    RoundRecord,
    RunResult,
    ToolAction,
    ToolExecutionResult,
    VerifierVerdict,
)


class HandoffPacketTest(unittest.TestCase):
    def test_build_handoff_packet_extracts_tasks_and_paths(self) -> None:
        result = RunResult(
            status="completed",
            model_id="dummy-model",
            workspace="/tmp/demo-workspace",
            execution_mode="read_only",
            stop_reason="inspection_synthesized",
            final_answer=(
                "Current state: The repo is already packaged and the core loop is implemented.\n\n"
                "Next engineering tasks:\n"
                "- Add persistent run history and memory.\n"
                "- Add streaming event output.\n"
            ),
            transcript="demo transcript",
            rounds=[
                RoundRecord(
                    round_number=1,
                    strategist="inspect",
                    critic="verify",
                    planner=PlannerTurn(
                        summary="read key files",
                        should_stop=False,
                        final_answer=None,
                        actions=[
                            ToolAction(tool="read_file", reason="inspect", args={"path": "README.md"}),
                            ToolAction(tool="read_file", reason="inspect", args={"path": "teamai/supervisor.py"}),
                        ],
                    ),
                    tool_results=[
                        ToolExecutionResult(
                            tool="read_file",
                            success=True,
                            output="readme",
                            metadata={"path": "/tmp/demo-workspace/README.md"},
                        ),
                        ToolExecutionResult(
                            tool="read_file",
                            success=True,
                            output="supervisor",
                            metadata={"path": "/tmp/demo-workspace/teamai/supervisor.py"},
                        ),
                    ],
                    verifier=VerifierVerdict(
                        done=False,
                        confidence=0.6,
                        summary="enough evidence",
                        next_focus="synthesize",
                    ),
                )
            ],
            warnings=["minor warning"],
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        packet = build_handoff_packet(
            task="Inspect this repository and identify the next engineering tasks.",
            result=result,
        )

        self.assertIn("Current state", packet.summary)
        self.assertEqual(packet.task_route, "multi_agent_loop")
        self.assertEqual(
            packet.next_tasks,
            ["Add persistent run history and memory.", "Add streaming event output."],
        )
        self.assertEqual(packet.primary_task, "Add persistent run history and memory.")
        self.assertEqual(packet.key_paths, ["README.md", "teamai/supervisor.py"])
        self.assertIn("Read README.md.", packet.evidence)
        self.assertIn("enough evidence", " ".join(packet.evidence))
        self.assertEqual(packet.open_questions, ["synthesize"])
        self.assertIn("Prioritize this next task first", packet.suggested_codex_prompt)

    def test_render_handoff_markdown_includes_sections(self) -> None:
        result = RunResult(
            status="completed",
            model_id="dummy-model",
            workspace="/tmp/demo-workspace",
            execution_mode="read_only",
            stop_reason="inspection_synthesized",
            final_answer="Current state: Ready.\n\nNext engineering tasks:\n- Add memory.\n",
            transcript="demo transcript",
            rounds=[
                RoundRecord(
                    round_number=1,
                    strategist="inspect",
                    critic="verify",
                    planner=PlannerTurn(
                        summary="read docs",
                        should_stop=False,
                        final_answer=None,
                        actions=[ToolAction(tool="read_file", reason="inspect", args={"path": "README.md"})],
                    ),
                    tool_results=[
                        ToolExecutionResult(
                            tool="read_file",
                            success=True,
                            output="readme",
                            metadata={"path": "/tmp/demo-workspace/README.md"},
                        )
                    ],
                    verifier=VerifierVerdict(
                        done=False,
                        confidence=0.4,
                        summary="enough evidence",
                        next_focus="implement memory",
                    ),
                )
            ],
            warnings=[],
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )
        packet = build_handoff_packet(task="Inspect repo.", result=result)

        rendered = render_handoff_markdown(packet)

        self.assertIn("# Local Model Handoff", rendered)
        self.assertIn("Route: multi_agent_loop", rendered)
        self.assertIn("## Primary Task", rendered)
        self.assertIn("## Next Tasks", rendered)
        self.assertIn("## Evidence", rendered)
        self.assertIn("## Suggested Codex Prompt", rendered)

    def test_build_handoff_packet_filters_open_questions_for_implemented_features(self) -> None:
        result = RunResult(
            status="completed",
            model_id="dummy-model",
            workspace="/tmp/demo-workspace",
            execution_mode="read_only",
            stop_reason="inspection_synthesized",
            final_answer=(
                "Current state: The repo is packaged. Persistent run history and cross-run workspace memory are already implemented through `.teamai/` state files and prompt injection.\n\n"
                "Next engineering tasks:\n"
                "- Replace the coarse write path with patch-oriented editing tools and approval checkpoints before destructive changes.\n"
                "- Add streaming event output.\n"
            ),
            transcript="demo transcript",
            rounds=[
                RoundRecord(
                    round_number=1,
                    strategist="inspect",
                    critic="verify",
                    planner=PlannerTurn(
                        summary="read memory",
                        should_stop=False,
                        final_answer=None,
                        actions=[ToolAction(tool="read_file", reason="inspect", args={"path": "teamai/memory.py"})],
                    ),
                    tool_results=[
                        ToolExecutionResult(
                            tool="read_file",
                            success=True,
                            output="memory implementation",
                            metadata={"path": "/tmp/demo-workspace/teamai/memory.py"},
                        )
                    ],
                    verifier=VerifierVerdict(
                        done=False,
                        confidence=0.5,
                        summary="memory exists",
                        next_focus=(
                            "Read `teamai/memory.py` to find the persistence methods (`persist_run`, `_load_history_records`) "
                            "and analyze their scalability."
                        ),
                    ),
                ),
                RoundRecord(
                    round_number=2,
                    strategist="inspect",
                    critic="verify",
                    planner=PlannerTurn(
                        summary="trace handoff shaping",
                        should_stop=False,
                        final_answer=None,
                        actions=[ToolAction(tool="read_file", reason="inspect", args={"path": "teamai/handoff.py"})],
                    ),
                    tool_results=[
                        ToolExecutionResult(
                            tool="read_file",
                            success=True,
                            output="handoff shaping",
                            metadata={"path": "/tmp/demo-workspace/teamai/handoff.py"},
                        )
                    ],
                    verifier=VerifierVerdict(
                        done=False,
                        confidence=0.6,
                        summary="need the next task",
                        next_focus=(
                            "Investigate how `_extract_summary_and_tasks` should support the patch-oriented editing goal "
                            "without losing the next_tasks list."
                        ),
                    ),
                ),
            ],
            warnings=[],
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        packet = build_handoff_packet(
            task="Inspect this repository and identify the next engineering tasks.",
            result=result,
        )

        self.assertEqual(
            packet.primary_task,
            "Replace the coarse write path with patch-oriented editing tools and approval checkpoints before destructive changes.",
        )
        self.assertEqual(
            packet.open_questions,
            [
                "Investigate how `_extract_summary_and_tasks` should support the patch-oriented editing goal without losing the next_tasks list."
            ],
        )
        self.assertIn("Verify this first if still unresolved: Investigate how `_extract_summary_and_tasks`", packet.suggested_codex_prompt)
        self.assertNotIn("persist_run", " ".join(packet.open_questions))

    def test_build_handoff_packet_emphasizes_codex_lead_route(self) -> None:
        result = RunResult(
            status="completed",
            model_id="dummy-model",
            workspace="/tmp/demo-workspace",
            execution_mode="read_only",
            task_route="codex_handoff",
            stop_reason="codex_handoff_synthesized",
            final_answer=(
                "Current state: The local run gathered reconnaissance.\n\n"
                "Next engineering tasks:\n"
                "- Inspect teamai/supervisor.py and teamai/bridge.py.\n"
            ),
            transcript="demo transcript",
            rounds=[],
            warnings=[],
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        packet = build_handoff_packet(
            task="Implement better task routing across the supervisor and bridge.",
            result=result,
        )

        self.assertEqual(packet.task_route, "codex_handoff")
        self.assertIn("Codex-lead implementation task", packet.suggested_codex_prompt)

    def test_build_handoff_packet_prioritizes_specific_self_improvement_paths(self) -> None:
        result = RunResult(
            status="completed",
            model_id="dummy-model",
            workspace="/tmp/demo-workspace",
            execution_mode="read_only",
            task_route="codex_handoff",
            stop_reason="codex_handoff_synthesized",
            final_answer=(
                "Current state: The local run gathered reconnaissance.\n\n"
                "Next engineering tasks:\n"
                "- Inspect teamai/memory.py and tests/test_memory.py.\n"
            ),
            transcript="demo transcript",
            rounds=[
                RoundRecord(
                    round_number=1,
                    strategist="inspect",
                    critic="verify",
                    planner=PlannerTurn(
                        summary="map the workspace",
                        should_stop=False,
                        final_answer=None,
                        actions=[
                            ToolAction(tool="list_files", reason="inspect", args={"path": "."}),
                            ToolAction(tool="read_file", reason="inspect", args={"path": "teamai/supervisor.py"}),
                            ToolAction(tool="read_file", reason="inspect", args={"path": "teamai/memory.py"}),
                            ToolAction(tool="read_file", reason="inspect", args={"path": "tests/test_memory.py"}),
                        ],
                    ),
                    tool_results=[
                        ToolExecutionResult(
                            tool="list_files",
                            success=True,
                            output="teamai/\ntests/\n",
                            metadata={"path": "/tmp/demo-workspace"},
                        ),
                        ToolExecutionResult(
                            tool="read_file",
                            success=True,
                            output="supervisor",
                            metadata={"path": "/tmp/demo-workspace/teamai/supervisor.py"},
                        ),
                        ToolExecutionResult(
                            tool="read_file",
                            success=True,
                            output="memory",
                            metadata={"path": "/tmp/demo-workspace/teamai/memory.py"},
                        ),
                        ToolExecutionResult(
                            tool="read_file",
                            success=True,
                            output="memory test",
                            metadata={"path": "/tmp/demo-workspace/tests/test_memory.py"},
                        ),
                    ],
                    verifier=VerifierVerdict(
                        done=False,
                        confidence=0.5,
                        summary="enough evidence",
                        next_focus="Inspect teamai/memory.py before changing the learned-note ranking.",
                    ),
                )
            ],
            warnings=[],
            started_at=datetime.now(timezone.utc),
            completed_at=datetime.now(timezone.utc),
        )

        packet = build_handoff_packet(
            task=(
                "Improve self-improvement reconnaissance so learned-note, decay, pruning, and memory work "
                "prioritize teamai/memory.py and tests/test_memory.py before generic control-flow files."
            ),
            result=result,
        )

        self.assertEqual(packet.key_paths[:2], ["teamai/memory.py", "tests/test_memory.py"])
        self.assertNotIn(".", packet.key_paths)


if __name__ == "__main__":
    unittest.main()
