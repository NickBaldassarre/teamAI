from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from teamai.config import Settings
from teamai.model_backend import ModelResponse
from teamai.schemas import PlannerTurn, RoundRecord, RunEvent, RunRequest, ToolAction, ToolExecutionResult, VerifierVerdict
from teamai.supervisor import ClosedLoopSupervisor


class FakeBackend:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses[:]

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


class SupervisorStructuredOutputTest(unittest.TestCase):
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

    def test_planner_repairs_prose_into_valid_actions(self) -> None:
        (self.workspace / "README.md").write_text("# demo\n", encoding="utf-8")
        backend = FakeBackend(
            [
                "We should inspect README.md first and then check the core package.",
                (
                    '{"summary":"Read the README to gather project context.",'
                    '"should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"read_file","reason":"Read the main docs first.",'
                    '"args":{"path":"README.md","start_line":1,"end_line":80}}]}'
                ),
            ]
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=backend)
        warnings: list[str] = []

        planner = supervisor._plan(  # noqa: SLF001 - exercising structured repair path directly
            task="Inspect this repository and identify the next engineering tasks.",
            user_prompt="Plan the next action.",
            workspace=self.workspace,
            previous_rounds=[],
            execution_mode="read_only",
            max_actions=2,
            max_tokens=128,
            temperature=0.3,
            warnings=warnings,
        )

        self.assertEqual(planner.actions[0].tool, "read_file")
        self.assertIn("required repair", warnings[0])

    def test_planner_heuristic_fallback_uses_readme(self) -> None:
        (self.workspace / "README.md").write_text("# demo\n", encoding="utf-8")
        backend = FakeBackend(
            [
                "Read README.md next.",
                "Still not JSON.",
            ]
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=backend)
        warnings: list[str] = []

        planner = supervisor._plan(  # noqa: SLF001
            task="Inspect this repository and identify the next engineering tasks.",
            user_prompt="The critic says to inspect README.md next.",
            workspace=self.workspace,
            previous_rounds=[],
            execution_mode="read_only",
            max_actions=2,
            max_tokens=128,
            temperature=0.3,
            warnings=warnings,
        )

        self.assertEqual(planner.actions[0].tool, "read_file")
        self.assertEqual(planner.actions[0].args["path"], "README.md")
        self.assertTrue(any("heuristic fallback" in warning.lower() for warning in warnings))

    def test_extract_candidate_paths_includes_unquoted_hidden_file_path(self) -> None:
        candidates = ClosedLoopSupervisor._extract_candidate_paths(  # noqa: SLF001
            "Replace the text 'Old note.' with 'New note.' in .teamai/compiler-demo.md."
        )

        self.assertIn(".teamai/compiler-demo.md", candidates)

    def test_planner_replaces_repeated_listing_with_novel_action(self) -> None:
        (self.workspace / "README.md").write_text("# demo\n", encoding="utf-8")
        backend = FakeBackend(
            [
                (
                    '{"summary":"List the root again.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"list_files","reason":"Repeat the root listing.","args":{"path":"."}}]}'
                ),
            ]
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=backend)
        previous_rounds = [
            RoundRecord(
                round_number=1,
                strategist="list root",
                critic="fine",
                planner=PlannerTurn(
                    summary="Listed root once.",
                    should_stop=False,
                    final_answer=None,
                    actions=[ToolAction(tool="list_files", reason="inspect", args={"path": "."})],
                ),
                tool_results=[
                    ToolExecutionResult(
                        tool="list_files",
                        success=True,
                        output="README.md",
                        error=None,
                        metadata={},
                    )
                ],
                verifier=VerifierVerdict(
                    done=False,
                    confidence=0.1,
                    summary="Need to read README next.",
                    next_focus="Read README.md",
                ),
            )
        ]
        warnings: list[str] = []

        planner = supervisor._plan(  # noqa: SLF001
            task="Inspect this repository and identify the next engineering tasks.",
            user_prompt="The next step should inspect README.md rather than relist the root.",
            workspace=self.workspace,
            previous_rounds=previous_rounds,
            execution_mode="read_only",
            max_actions=2,
            max_tokens=128,
            temperature=0.3,
            warnings=warnings,
        )

        self.assertEqual(planner.actions[0].tool, "read_file")
        self.assertEqual(planner.actions[0].args["path"], "README.md")
        self.assertTrue(any("skipping repeated successful action" in warning.lower() for warning in warnings))

    def test_invalid_missing_path_falls_back_to_real_priority_file(self) -> None:
        (self.workspace / "teamai").mkdir()
        (self.workspace / "teamai" / "config.py").write_text("settings = 1\n", encoding="utf-8")
        (self.workspace / "teamai" / "cli.py").write_text("def main():\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "supervisor.py").write_text("class X:\n    pass\n", encoding="utf-8")
        backend = FakeBackend(
            [
                (
                    '{"summary":"Read the main entrypoint.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"read_file","reason":"Inspect the entrypoint.","args":{"path":"teamai/main.py"}}]}'
                ),
            ]
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=backend)
        previous_rounds = [
            RoundRecord(
                round_number=1,
                strategist="list package and read config",
                critic="then inspect execution flow",
                planner=PlannerTurn(
                    summary="Listed package and read config first.",
                    should_stop=False,
                    final_answer=None,
                    actions=[
                        ToolAction(tool="list_files", reason="inspect", args={"path": "teamai"}),
                        ToolAction(tool="read_file", reason="inspect", args={"path": "teamai/config.py"}),
                    ],
                ),
                tool_results=[
                    ToolExecutionResult(
                        tool="list_files",
                        success=True,
                        output="teamai/cli.py\nteamai/config.py\nteamai/supervisor.py",
                        error=None,
                        metadata={},
                    ),
                    ToolExecutionResult(
                        tool="read_file",
                        success=True,
                        output="settings",
                        error=None,
                        metadata={},
                    )
                ],
                verifier=VerifierVerdict(
                    done=False,
                    confidence=0.2,
                    summary="Need to inspect execution flow next.",
                    next_focus="Read teamai/cli.py or teamai/supervisor.py",
                ),
            )
        ]
        warnings: list[str] = []

        planner = supervisor._plan(  # noqa: SLF001
            task="Inspect this repository and identify the next engineering tasks.",
            user_prompt="Inspect the next real entrypoint after config.py.",
            workspace=self.workspace,
            previous_rounds=previous_rounds,
            execution_mode="read_only",
            max_actions=2,
            max_tokens=128,
            temperature=0.3,
            warnings=warnings,
        )

        self.assertEqual(planner.actions[0].tool, "read_file")
        self.assertEqual(planner.actions[0].args["path"], "teamai/cli.py")
        self.assertTrue(any("invalid action target" in warning.lower() for warning in warnings))

    def test_missing_required_search_pattern_falls_back_to_real_action(self) -> None:
        (self.workspace / "README.md").write_text("# demo\n", encoding="utf-8")
        backend = FakeBackend(
            [
                (
                    '{"summary":"Search the repo.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"search_text","reason":"Search for build configuration issues.","args":{}}]}'
                ),
            ]
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=backend)
        warnings: list[str] = []

        planner = supervisor._plan(  # noqa: SLF001
            task="Inspect this repository and identify the next engineering tasks.",
            user_prompt="Compare the build configuration files.",
            workspace=self.workspace,
            previous_rounds=[],
            execution_mode="read_only",
            max_actions=2,
            max_tokens=128,
            temperature=0.3,
            warnings=warnings,
        )

        self.assertEqual(planner.actions[0].tool, "read_file")
        self.assertEqual(planner.actions[0].args["path"], "README.md")
        self.assertTrue(any("invalid action arguments" in warning.lower() for warning in warnings))

    def test_incompatible_directory_read_falls_back_to_directory_listing(self) -> None:
        (self.workspace / "teamai").mkdir()
        (self.workspace / "teamai" / "config.py").write_text("settings = 1\n", encoding="utf-8")
        backend = FakeBackend(
            [
                (
                    '{"summary":"Inspect the package.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"read_file","reason":"Read the package root.","args":{"path":"teamai"}}]}'
                ),
            ]
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=backend)
        warnings: list[str] = []

        planner = supervisor._plan(  # noqa: SLF001
            task="Inspect this repository and identify the next engineering tasks.",
            user_prompt="Inspect the teamai package next.",
            workspace=self.workspace,
            previous_rounds=[],
            execution_mode="read_only",
            max_actions=2,
            max_tokens=128,
            temperature=0.3,
            warnings=warnings,
        )

        self.assertEqual(planner.actions[0].tool, "list_files")
        self.assertEqual(planner.actions[0].args["path"], "teamai")
        self.assertTrue(any("incompatible action target" in warning.lower() for warning in warnings))

    def test_repository_inspection_plan_batches_additional_reads(self) -> None:
        (self.workspace / "README.md").write_text("# demo\n", encoding="utf-8")
        (self.workspace / "teamai").mkdir()
        (self.workspace / "teamai" / "config.py").write_text("settings = 1\n", encoding="utf-8")
        backend = FakeBackend(
            [
                (
                    '{"summary":"List the root.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"list_files","reason":"Inspect the root.","args":{"path":"."}}]}'
                ),
            ]
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=backend)
        warnings: list[str] = []

        planner = supervisor._plan(  # noqa: SLF001
            task="Inspect this repository and identify the next engineering tasks.",
            user_prompt="Start by inspecting the repository structure.",
            workspace=self.workspace,
            previous_rounds=[],
            execution_mode="read_only",
            max_actions=3,
            max_tokens=128,
            temperature=0.3,
            warnings=warnings,
        )

        signatures = [(action.tool, action.args.get("path")) for action in planner.actions]
        self.assertIn(("list_files", "."), signatures)
        self.assertIn(("read_file", "README.md"), signatures)
        self.assertIn(("list_files", "teamai"), signatures)
        self.assertTrue(any("expanded plan" in warning.lower() for warning in warnings))

    def test_repository_inspection_plan_prioritizes_runtime_anchor_after_planned_config_read(self) -> None:
        (self.workspace / "README.md").write_text("# teamAI\n", encoding="utf-8")
        (self.workspace / "pyproject.toml").write_text('[project]\nname = "teamai"\n', encoding="utf-8")
        (self.workspace / "teamai").mkdir()
        (self.workspace / "teamai" / "config.py").write_text("class Settings:\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "cli.py").write_text("def main() -> None:\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "supervisor.py").write_text("class ClosedLoopSupervisor:\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "api.py").write_text("def create_app() -> None:\n    pass\n", encoding="utf-8")

        previous_rounds = [
            RoundRecord(
                round_number=1,
                strategist="Map the repo.",
                critic="Read the docs first.",
                planner=PlannerTurn(
                    summary="Inspect the root and README.",
                    actions=[
                        ToolAction(tool="list_files", args={"path": "."}),
                        ToolAction(tool="read_file", args={"path": "README.md"}),
                    ],
                ),
                tool_results=[
                    ToolExecutionResult(tool="list_files", success=True, metadata={"path": "."}),
                    ToolExecutionResult(
                        tool="read_file",
                        success=True,
                        metadata={"path": "README.md"},
                        output="# teamAI\n",
                    ),
                ],
                verifier=VerifierVerdict(
                    done=False,
                    confidence=0.1,
                    summary="Need the runtime entrypoints next.",
                    next_focus="Read the config and runtime anchors.",
                ),
            )
        ]
        backend = FakeBackend(
            [
                (
                    '{"summary":"Inspect the runtime config.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"read_file","reason":"Start with the runtime settings.","args":{"path":"teamai/config.py"}}]}'
                ),
            ]
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=backend)
        warnings: list[str] = []

        planner = supervisor._plan(  # noqa: SLF001
            task="Inspect this repository and identify the next engineering tasks.",
            user_prompt="Read the runtime settings and the most relevant entrypoint next.",
            workspace=self.workspace,
            previous_rounds=previous_rounds,
            execution_mode="read_only",
            max_actions=2,
            max_tokens=128,
            temperature=0.3,
            warnings=warnings,
        )

        signatures = [(action.tool, action.args.get("path")) for action in planner.actions]
        self.assertIn(("read_file", "teamai/config.py"), signatures)
        self.assertNotIn(("read_file", "pyproject.toml"), signatures)
        runtime_anchor_paths = {
            path
            for tool, path in signatures
            if tool == "read_file" and path in {"teamai/cli.py", "teamai/supervisor.py", "teamai/api.py"}
        }
        self.assertTrue(runtime_anchor_paths)

    def test_repository_inspection_run_auto_completes_after_core_reads(self) -> None:
        (self.workspace / "README.md").write_text(
            "# teamAI\npersistent memory\naction approval checkpoints\npatch-oriented editing tools\nstreaming event output\n",
            encoding="utf-8",
        )
        (self.workspace / "teamai").mkdir()
        (self.workspace / "teamai" / "__init__.py").write_text("", encoding="utf-8")
        (self.workspace / "teamai" / "config.py").write_text("class Settings:\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "cli.py").write_text("def main():\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "supervisor.py").write_text("class ClosedLoopSupervisor:\n    pass\n", encoding="utf-8")

        backend = FakeBackend(
            [
                "List the root first.",
                "That establishes the repo shape.",
                '{"summary":"List the root.","should_stop":false,"final_answer":null,"actions":[{"tool":"list_files","reason":"Inspect the root.","args":{"path":"."}}]}',
                '{"done":false,"confidence":0.1,"summary":"Need core docs and package structure.","next_focus":"Read the README and inspect the package."}',
                "Read the core configuration next.",
                "That should be enough to understand the runtime shape.",
                '{"summary":"Read config.","should_stop":false,"final_answer":null,"actions":[{"tool":"read_file","reason":"Inspect runtime settings.","args":{"path":"teamai/config.py"}}]}',
                '{"done":false,"confidence":0.2,"summary":"Enough evidence gathered for synthesis soon.","next_focus":"Summarize the engineering tasks."}',
            ]
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=backend)

        result = supervisor.run(
            RunRequest(
                task="Inspect this repository and identify the next engineering tasks.",
                workspace_path=".",
                max_actions_per_round=3,
            ),
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.stop_reason, "inspection_synthesized")
        self.assertIn("Next engineering tasks", result.final_answer)
        self.assertIn("memory", result.final_answer.lower())
        self.assertIn("patch-oriented", result.final_answer.lower())

    def test_repository_inspection_run_partially_synthesizes_after_max_rounds(self) -> None:
        (self.workspace / "README.md").write_text(
            "# teamAI\nlocal-first closed-loop orchestration\nstreaming event output\n",
            encoding="utf-8",
        )
        (self.workspace / "pyproject.toml").write_text(
            '[project]\nname = "teamai"\ndependencies = ["mlx-vlm>=0.4.4"]\n',
            encoding="utf-8",
        )
        (self.workspace / "teamai").mkdir()
        (self.workspace / "teamai" / "supervisor.py").write_text(
            "class ClosedLoopSupervisor:\n    pass\n",
            encoding="utf-8",
        )

        backend = FakeBackend(
            [
                "Read the supervisor first.",
                "That should anchor the runtime flow.",
                (
                    '{"summary":"Inspect supervisor.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"read_file","reason":"Inspect the core loop.","args":{"path":"teamai/supervisor.py"}}]}'
                ),
                '{"done":false,"confidence":0.2,"summary":"Enough repo context collected, but not enough time to converge.","next_focus":"Summarize the best next engineering tasks."}',
            ]
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=backend)

        result = supervisor.run(
            RunRequest(
                task="Inspect this repository and identify the next engineering tasks.",
                workspace_path=".",
                max_rounds=1,
                max_actions_per_round=3,
            ),
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.stop_reason, "inspection_synthesized")
        self.assertIn("Next engineering tasks", result.final_answer)
        self.assertIn("streaming event output", result.final_answer.lower())
        self.assertTrue(any("partial synthesis" in warning.lower() for warning in result.warnings))

    def test_repository_inspection_run_partially_synthesizes_on_penultimate_round(self) -> None:
        (self.workspace / "README.md").write_text(
            "# teamAI\nlocal-first closed-loop orchestration\nstreaming event output\n",
            encoding="utf-8",
        )
        (self.workspace / "teamai").mkdir()
        (self.workspace / "teamai" / "config.py").write_text("class Settings:\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "cli.py").write_text("def main():\n    pass\n", encoding="utf-8")

        backend = FakeBackend(
            [
                "List the root first.",
                "That anchors the repo shape.",
                '{"summary":"Inspect the root.","should_stop":false,"final_answer":null,"actions":[{"tool":"list_files","reason":"Inspect the workspace root.","args":{"path":"."}}]}',
                '{"done":false,"confidence":0.1,"summary":"Need one runtime file before summarizing.","next_focus":"Read the runtime config and a primary entrypoint."}',
                "Read the runtime settings next.",
                "That should be enough to summarize the next engineering tasks.",
                '{"summary":"Inspect runtime settings.","should_stop":false,"final_answer":null,"actions":[{"tool":"read_file","reason":"Inspect runtime settings.","args":{"path":"teamai/config.py"}}]}',
                '{"done":false,"confidence":0.2,"summary":"Enough context gathered for a partial repository summary.","next_focus":"Summarize the next engineering tasks."}',
            ]
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=backend)

        result = supervisor.run(
            RunRequest(
                task="Inspect this repository and identify the next engineering tasks.",
                workspace_path=".",
                max_rounds=3,
                max_actions_per_round=2,
            ),
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.stop_reason, "inspection_synthesized")
        self.assertIn("Next engineering tasks", result.final_answer)
        self.assertIn("Runtime settings are centralized", result.final_answer)

    def test_repository_inspection_run_can_complete_without_model_calls(self) -> None:
        (self.workspace / "README.md").write_text(
            "# teamAI\nlocal-first closed-loop orchestration\nstreaming event output\n",
            encoding="utf-8",
        )
        (self.workspace / "pyproject.toml").write_text(
            '[project]\nname = "teamai"\ndependencies = ["mlx-vlm>=0.4.4"]\n',
            encoding="utf-8",
        )
        (self.workspace / "teamai").mkdir()
        (self.workspace / "teamai" / "config.py").write_text("class Settings:\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "cli.py").write_text("def main():\n    pass\n", encoding="utf-8")

        result = ClosedLoopSupervisor(self.settings, backend=FakeBackend([])).run(
            RunRequest(
                task="Inspect this repository and identify the next engineering tasks.",
                workspace_path=".",
                max_rounds=3,
                max_actions_per_round=2,
            ),
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.stop_reason, "inspection_synthesized")
        self.assertIn("Next engineering tasks", result.final_answer)
        self.assertIn("Runtime settings are centralized", result.final_answer)

    def test_workspace_write_run_stops_for_pending_patch_approval(self) -> None:
        (self.workspace / "README.md").write_text("# demo\n", encoding="utf-8")
        writable_settings = Settings(
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
            allow_writes=True,
            command_timeout_seconds=5,
            max_file_bytes=10_000,
            max_command_output_chars=10_000,
            host="127.0.0.1",
            port=8000,
        )
        backend = FakeBackend(
            [
                "Update the README.",
                "Use the write tool carefully.",
                (
                    '{"summary":"Prepare a README patch.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"write_file","reason":"Propose a README update.","args":{"path":"README.md","content":"# updated\\n"}}]}'
                ),
            ]
        )

        result = ClosedLoopSupervisor(writable_settings, backend=backend).run(
            RunRequest(
                task="Update the README with a patch-oriented approval flow.",
                workspace_path=".",
                execution_mode="workspace_write",
            ),
        )

        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stop_reason, "approval_required")
        self.assertIn("teamai approvals show", result.final_answer)
        self.assertIn("No file changes were applied yet.", result.final_answer)
        self.assertEqual((self.workspace / "README.md").read_text(encoding="utf-8"), "# demo\n")
        self.assertEqual(result.rounds[0].verifier.summary, "Patch approval is required before the proposed file changes can be applied.")

    def test_synthesis_suppresses_implemented_memory_feature(self) -> None:
        (self.workspace / "README.md").write_text(
            "# teamAI\npersistent memory\npatch-oriented editing tools\nstreaming event output\n",
            encoding="utf-8",
        )
        (self.workspace / "pyproject.toml").write_text(
            '[project]\nname = "teamai"\ndependencies = ["mlx-vlm>=0.4.4"]\n',
            encoding="utf-8",
        )
        (self.workspace / "teamai").mkdir()
        (self.workspace / "teamai" / "memory.py").write_text(
            "RUN_HISTORY_FILE_NAME = 'run-history.jsonl'\n"
            "MEMORY_FILE_NAME = 'memory.md'\n"
            "class WorkspaceMemoryStore:\n"
            "    pass\n",
            encoding="utf-8",
        )
        (self.workspace / "teamai" / "prompts.py").write_text(
            "Persistent workspace memory:\nRecent persisted runs:\n",
            encoding="utf-8",
        )
        (self.workspace / "teamai" / "supervisor.py").write_text(
            "from .memory import WorkspaceMemoryStore\n"
            "class ClosedLoopSupervisor:\n"
            "    def load(self):\n"
            "        self.store = WorkspaceMemoryStore()\n"
            "        self.store.load_snapshot(None)\n"
            "        self.store.persist_run(None)\n",
            encoding="utf-8",
        )

        rounds = [
            RoundRecord(
                round_number=1,
                strategist="inspect",
                critic="verify",
                planner=PlannerTurn(
                    summary="read core docs",
                    should_stop=False,
                    final_answer=None,
                    actions=[
                        ToolAction(tool="read_file", reason="inspect", args={"path": "README.md"}),
                        ToolAction(tool="read_file", reason="inspect", args={"path": "pyproject.toml"}),
                        ToolAction(tool="list_files", reason="inspect", args={"path": "teamai"}),
                        ToolAction(tool="read_file", reason="inspect", args={"path": "teamai/supervisor.py"}),
                    ],
                ),
                tool_results=[
                    ToolExecutionResult(
                        tool="read_file",
                        success=True,
                        output="persistent memory\npatch-oriented editing tools\nstreaming event output",
                        metadata={},
                    ),
                    ToolExecutionResult(tool="read_file", success=True, output='[project]\nname = "teamai"', metadata={}),
                    ToolExecutionResult(tool="list_files", success=True, output="teamai/memory.py", metadata={}),
                    ToolExecutionResult(tool="read_file", success=True, output="WorkspaceMemoryStore", metadata={}),
                ],
                verifier=VerifierVerdict(
                    done=False,
                    confidence=0.4,
                    summary="Enough context for synthesis.",
                    next_focus="Rank the next engineering tasks.",
                ),
            )
        ]

        supervisor = ClosedLoopSupervisor(self.settings, backend=FakeBackend([]))
        synthesized = supervisor._maybe_synthesize_repository_answer(  # noqa: SLF001
            task="Inspect this repository and identify the next engineering tasks.",
            rounds=rounds,
            workspace=self.workspace,
            allow_partial=True,
        )

        assert synthesized is not None
        self.assertIn("Persistent run history and cross-run workspace memory are already implemented", synthesized)
        self.assertNotIn("Add persistent run history and memory", synthesized)
        self.assertIn("patch-oriented editing tools", synthesized)

    def test_heuristic_fallback_ignores_malformed_long_path_blob(self) -> None:
        (self.workspace / "README.md").write_text("# demo\n", encoding="utf-8")
        supervisor = ClosedLoopSupervisor(self.settings, backend=FakeBackend([]))

        planner = supervisor._heuristic_plan_from_context(  # noqa: SLF001
            task="Inspect this repository and identify the next engineering tasks.",
            raw_response=(
                '"/Users/home/Documents/teamAI/, and the Critic agreed but cautioned that the default '
                'settings must be validated first before proceeding with deep dives into configuration.", '
                'and then inspect "README.md"'
            ),
            user_prompt="Read README.md next.",
            workspace=self.workspace,
            previous_rounds=[],
            max_actions=2,
        )

        self.assertEqual(planner.actions[0].tool, "read_file")
        self.assertEqual(planner.actions[0].args["path"], "README.md")

    def test_explicit_workspace_write_task_is_not_treated_as_repository_inspection(self) -> None:
        task = (
            "Use workspace_write mode. Read README.md, then create exactly one replace_in_file patch approval "
            "that inserts the sentence 'Pending approvals may become stale if the target file changes before approval.' "
            "immediately after the paragraph that starts with 'In workspace_write mode'. Do not inspect unrelated files."
        )

        self.assertFalse(ClosedLoopSupervisor._is_repository_inspection_task(task))  # noqa: SLF001

    def test_heuristic_write_fallback_compiles_exact_replace_from_task(self) -> None:
        (self.workspace / "README.md").write_text(
            "# teamAI\n\nOld note.\n",
            encoding="utf-8",
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=FakeBackend([]))

        planner = supervisor._heuristic_plan_from_context(  # noqa: SLF001
            task="Use workspace_write mode. Replace the text 'Old note.' with 'New note.' in README.md.",
            raw_response="No valid planner JSON.",
            user_prompt="Compile the requested patch.",
            workspace=self.workspace,
            previous_rounds=[],
            max_actions=1,
            execution_mode="workspace_write",
        )

        self.assertEqual(planner.actions[0].tool, "replace_in_file")
        self.assertEqual(planner.actions[0].args["path"], "README.md")
        self.assertEqual(planner.actions[0].args["old_text"], "Old note.")
        self.assertEqual(planner.actions[0].args["new_text"], "New note.")

    def test_heuristic_write_fallback_compiles_replace_all_from_task(self) -> None:
        (self.workspace / "README.md").write_text(
            "repeat\nrepeat\n",
            encoding="utf-8",
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=FakeBackend([]))

        planner = supervisor._heuristic_plan_from_context(  # noqa: SLF001
            task="Use workspace_write mode. Replace all occurrences of 'repeat' with 'done' in README.md.",
            raw_response="No valid planner JSON.",
            user_prompt="Compile the requested patch.",
            workspace=self.workspace,
            previous_rounds=[],
            max_actions=1,
            execution_mode="workspace_write",
        )

        self.assertEqual(planner.actions[0].tool, "replace_in_file")
        self.assertEqual(planner.actions[0].args["path"], "README.md")
        self.assertEqual(planner.actions[0].args["old_text"], "repeat")
        self.assertEqual(planner.actions[0].args["new_text"], "done")
        self.assertEqual(planner.actions[0].args["replace_all"], True)

    def test_heuristic_write_fallback_compiles_insert_before_anchor_from_task(self) -> None:
        (self.workspace / "README.md").write_text(
            "# teamAI\n\nAnchor line\n",
            encoding="utf-8",
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=FakeBackend([]))

        planner = supervisor._heuristic_plan_from_context(  # noqa: SLF001
            task="Use workspace_write mode. Insert the line 'Inserted line' before the line 'Anchor line' in README.md.",
            raw_response="No valid planner JSON.",
            user_prompt="Compile the requested patch.",
            workspace=self.workspace,
            previous_rounds=[],
            max_actions=1,
            execution_mode="workspace_write",
        )

        self.assertEqual(planner.actions[0].tool, "replace_in_file")
        self.assertEqual(planner.actions[0].args["path"], "README.md")
        self.assertEqual(planner.actions[0].args["old_text"], "Anchor line")
        self.assertEqual(planner.actions[0].args["new_text"], "Inserted line\nAnchor line")

    def test_heuristic_write_fallback_compiles_fenced_block_before_anchor(self) -> None:
        test_file = self.workspace / "tests" / "test_sample.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(
            "import unittest\n\nif __name__ == \"__main__\":\n    unittest.main()\n",
            encoding="utf-8",
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=FakeBackend([]))

        planner = supervisor._heuristic_plan_from_context(  # noqa: SLF001
            task=(
                "Use workspace_write mode. Insert the following block before 'if __name__ == \"__main__\":' "
                "in tests/test_sample.py.\n"
                "```python\n"
                "class SampleTest(unittest.TestCase):\n"
                "    def test_truth(self) -> None:\n"
                "        self.assertTrue(True)\n"
                "```\n"
            ),
            raw_response="No valid planner JSON.",
            user_prompt="Compile the requested patch.",
            workspace=self.workspace,
            previous_rounds=[],
            max_actions=1,
            execution_mode="workspace_write",
        )

        self.assertEqual(planner.actions[0].tool, "replace_in_file")
        self.assertEqual(planner.actions[0].args["path"], "tests/test_sample.py")
        self.assertEqual(planner.actions[0].args["old_text"], 'if __name__ == "__main__":')
        self.assertIn("class SampleTest(unittest.TestCase):", planner.actions[0].args["new_text"])
        self.assertIn('if __name__ == "__main__":', planner.actions[0].args["new_text"])

    def test_heuristic_write_fallback_compiles_append_line_from_task(self) -> None:
        (self.workspace / "README.md").write_text(
            "# teamAI\n",
            encoding="utf-8",
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=FakeBackend([]))

        planner = supervisor._heuristic_plan_from_context(  # noqa: SLF001
            task="Use workspace_write mode. Append the line 'Final reminder' to README.md.",
            raw_response="No valid planner JSON.",
            user_prompt="Compile the requested patch.",
            workspace=self.workspace,
            previous_rounds=[],
            max_actions=1,
            execution_mode="workspace_write",
        )

        self.assertEqual(planner.actions[0].tool, "write_file")
        self.assertEqual(planner.actions[0].args["path"], "README.md")
        self.assertEqual(planner.actions[0].args["content"], "# teamAI\nFinal reminder\n")

    def test_heuristic_write_fallback_compiles_exact_append_line_from_task(self) -> None:
        hidden_target = self.workspace / ".teamai" / "eval-fixtures" / "scratch.md"
        hidden_target.parent.mkdir(parents=True, exist_ok=True)
        hidden_target.write_text(
            "Seed line.\n",
            encoding="utf-8",
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=FakeBackend([]))

        planner = supervisor._heuristic_plan_from_context(  # noqa: SLF001
            task="Append the exact line 'Eval harness scratch line.' to .teamai/eval-fixtures/scratch.md.",
            raw_response="No valid planner JSON.",
            user_prompt="Compile the requested patch.",
            workspace=self.workspace,
            previous_rounds=[],
            max_actions=1,
            execution_mode="workspace_write",
        )

        self.assertEqual(planner.actions[0].tool, "write_file")
        self.assertEqual(planner.actions[0].args["path"], ".teamai/eval-fixtures/scratch.md")
        self.assertEqual(planner.actions[0].args["content"], "Seed line.\nEval harness scratch line.\n")

    def test_heuristic_write_fallback_compiles_append_fenced_block_to_test_file(self) -> None:
        test_file = self.workspace / "tests" / "test_sample.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(
            "import unittest\n",
            encoding="utf-8",
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=FakeBackend([]))

        planner = supervisor._heuristic_plan_from_context(  # noqa: SLF001
            task=(
                "Use workspace_write mode. Append the following block to tests/test_sample.py.\n"
                "```python\n"
                "class SampleTest(unittest.TestCase):\n"
                "    def test_truth(self) -> None:\n"
                "        self.assertTrue(True)\n"
                "```\n"
            ),
            raw_response="No valid planner JSON.",
            user_prompt="Compile the requested patch.",
            workspace=self.workspace,
            previous_rounds=[],
            max_actions=1,
            execution_mode="workspace_write",
        )

        self.assertEqual(planner.actions[0].tool, "write_file")
        self.assertEqual(planner.actions[0].args["path"], "tests/test_sample.py")
        self.assertIn("class SampleTest(unittest.TestCase):", planner.actions[0].args["content"])

    def test_heuristic_write_fallback_compiles_import_insertion(self) -> None:
        api_file = self.workspace / "teamai" / "api.py"
        api_file.parent.mkdir(parents=True, exist_ok=True)
        api_file.write_text(
            "from fastapi import FastAPI\n\napp = FastAPI()\n",
            encoding="utf-8",
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=FakeBackend([]))

        planner = supervisor._heuristic_plan_from_context(  # noqa: SLF001
            task="Use workspace_write mode. Add the import 'import threading' to teamai/api.py.",
            raw_response="No valid planner JSON.",
            user_prompt="Compile the requested patch.",
            workspace=self.workspace,
            previous_rounds=[],
            max_actions=1,
            execution_mode="workspace_write",
        )

        self.assertEqual(planner.actions[0].tool, "write_file")
        self.assertEqual(planner.actions[0].args["path"], "teamai/api.py")
        self.assertIn("import threading", planner.actions[0].args["content"])
        self.assertIn("from fastapi import FastAPI", planner.actions[0].args["content"])

    def test_heuristic_write_fallback_compiles_assignment_update_in_env_file(self) -> None:
        env_file = self.workspace / ".env"
        env_file.write_text(
            "TEAMAI_ALLOW_WRITES=false\nTEAMAI_MAX_ROUNDS=4\n",
            encoding="utf-8",
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=FakeBackend([]))

        planner = supervisor._heuristic_plan_from_context(  # noqa: SLF001
            task="Use workspace_write mode. Set TEAMAI_ALLOW_WRITES to true in .env.",
            raw_response="No valid planner JSON.",
            user_prompt="Compile the requested patch.",
            workspace=self.workspace,
            previous_rounds=[],
            max_actions=1,
            execution_mode="workspace_write",
        )

        self.assertEqual(planner.actions[0].tool, "write_file")
        self.assertEqual(planner.actions[0].args["path"], ".env")
        self.assertIn("TEAMAI_ALLOW_WRITES=true", planner.actions[0].args["content"])

    def test_heuristic_write_fallback_compiles_test_method_insertion_into_class(self) -> None:
        test_file = self.workspace / "tests" / "test_sample.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(
            "import unittest\n\nclass SampleTest(unittest.TestCase):\n    def test_existing(self) -> None:\n        self.assertTrue(True)\n\nif __name__ == \"__main__\":\n    unittest.main()\n",
            encoding="utf-8",
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=FakeBackend([]))

        planner = supervisor._heuristic_plan_from_context(  # noqa: SLF001
            task=(
                "Use workspace_write mode. Add the following test to class SampleTest in tests/test_sample.py.\n"
                "```python\n"
                "def test_new_case(self) -> None:\n"
                "    self.assertEqual(1 + 1, 2)\n"
                "```\n"
            ),
            raw_response="No valid planner JSON.",
            user_prompt="Compile the requested patch.",
            workspace=self.workspace,
            previous_rounds=[],
            max_actions=1,
            execution_mode="workspace_write",
        )

        self.assertEqual(planner.actions[0].tool, "write_file")
        self.assertEqual(planner.actions[0].args["path"], "tests/test_sample.py")
        self.assertIn("def test_new_case(self) -> None:", planner.actions[0].args["content"])
        self.assertIn('if __name__ == "__main__":', planner.actions[0].args["content"])
        self.assertLess(
            planner.actions[0].args["content"].index("def test_new_case(self) -> None:"),
            planner.actions[0].args["content"].index('if __name__ == "__main__":'),
        )

    def test_workspace_write_run_compiles_import_insertion_without_prior_read(self) -> None:
        api_file = self.workspace / "teamai" / "api.py"
        api_file.parent.mkdir(parents=True, exist_ok=True)
        api_file.write_text(
            "from fastapi import FastAPI\n\napp = FastAPI()\n",
            encoding="utf-8",
        )
        writable_settings = Settings(
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
            allow_writes=True,
            command_timeout_seconds=5,
            max_file_bytes=10_000,
            max_command_output_chars=10_000,
            host="127.0.0.1",
            port=8000,
        )

        result = ClosedLoopSupervisor(writable_settings, backend=FakeBackend([])).run(
            RunRequest(
                task="Use workspace_write mode. Add the import 'import threading' to teamai/api.py.",
                workspace_path=".",
                execution_mode="workspace_write",
            ),
        )

        self.assertEqual(result.task_route, "deterministic_patch")
        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stop_reason, "approval_required")
        approval_files = sorted((self.workspace / ".teamai" / "approvals").glob("*.json"))
        self.assertEqual(len(approval_files), 1)
        self.assertIn("import threading", approval_files[0].read_text(encoding="utf-8"))

    def test_workspace_write_run_compiles_exact_replace_without_prior_read(self) -> None:
        (self.workspace / "README.md").write_text(
            "# teamAI\n\nOld note.\n",
            encoding="utf-8",
        )
        writable_settings = Settings(
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
            allow_writes=True,
            command_timeout_seconds=5,
            max_file_bytes=10_000,
            max_command_output_chars=10_000,
            host="127.0.0.1",
            port=8000,
        )
        backend = FakeBackend(
            [
                "Apply the exact note update.",
                "A single patch approval is sufficient here.",
                '{"summary":"Need the concrete patch.","should_stop":false,"final_answer":null,"actions":[]}',
            ]
        )

        result = ClosedLoopSupervisor(writable_settings, backend=backend).run(
            RunRequest(
                task=(
                    "Use workspace_write mode. Replace the text 'Old note.' with 'New note.' in README.md. "
                    "Stop as soon as the patch approval is created."
                ),
                workspace_path=".",
                execution_mode="workspace_write",
                max_rounds=1,
                max_actions_per_round=2,
            ),
        )

        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stop_reason, "approval_required")
        self.assertEqual(result.task_route, "deterministic_patch")
        approval_files = sorted((self.workspace / ".teamai" / "approvals").glob("*.json"))
        self.assertEqual(len(approval_files), 1)
        approval_payload = approval_files[0].read_text(encoding="utf-8")
        self.assertIn('"source_tool": "replace_in_file"', approval_payload)
        self.assertIn("New note.", approval_payload)

    def test_workspace_write_run_routes_compiler_safe_patch_without_model_calls(self) -> None:
        (self.workspace / "README.md").write_text(
            "# teamAI\n\nOld note.\n",
            encoding="utf-8",
        )
        writable_settings = Settings(
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
            allow_writes=True,
            command_timeout_seconds=5,
            max_file_bytes=10_000,
            max_command_output_chars=10_000,
            host="127.0.0.1",
            port=8000,
        )

        result = ClosedLoopSupervisor(writable_settings, backend=FakeBackend([])).run(
            RunRequest(
                task="Use workspace_write mode. Replace the text 'Old note.' with 'New note.' in README.md.",
                workspace_path=".",
                execution_mode="workspace_write",
            ),
        )

        self.assertEqual(result.task_route, "deterministic_patch")
        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stop_reason, "approval_required")
        self.assertEqual(len(result.rounds), 1)
        self.assertIn("deterministic", result.rounds[0].planner.summary.lower())
        approval_files = sorted((self.workspace / ".teamai" / "approvals").glob("*.json"))
        self.assertEqual(len(approval_files), 1)
        self.assertIn("New note.", approval_files[0].read_text(encoding="utf-8"))

    def test_workspace_write_run_compiles_exact_replace_for_hidden_file_path(self) -> None:
        hidden_target = self.workspace / ".teamai" / "compiler-demo.md"
        hidden_target.parent.mkdir(parents=True, exist_ok=True)
        hidden_target.write_text(
            "Compiler demo.\n\nOriginal note.\n",
            encoding="utf-8",
        )
        writable_settings = Settings(
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
            allow_writes=True,
            command_timeout_seconds=5,
            max_file_bytes=10_000,
            max_command_output_chars=10_000,
            host="127.0.0.1",
            port=8000,
        )
        backend = FakeBackend(
            [
                "Apply the exact note update.",
                "A single patch approval is sufficient here.",
                '{"summary":"Need the concrete patch.","should_stop":false,"final_answer":null,"actions":[]}',
            ]
        )

        result = ClosedLoopSupervisor(writable_settings, backend=backend).run(
            RunRequest(
                task=(
                    "Use workspace_write mode. Replace the text 'Original note.' with "
                    "'Updated note from the deterministic compiler.' in .teamai/compiler-demo.md. "
                    "Stop as soon as the patch approval is created."
                ),
                workspace_path=".",
                execution_mode="workspace_write",
                max_rounds=1,
                max_actions_per_round=2,
            ),
        )

        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stop_reason, "approval_required")
        approval_files = sorted((self.workspace / ".teamai" / "approvals").glob("*.json"))
        self.assertEqual(len(approval_files), 1)
        approval_payload = approval_files[0].read_text(encoding="utf-8")
        self.assertIn(".teamai/compiler-demo.md", approval_payload)
        self.assertIn("Updated note from the deterministic compiler.", approval_payload)

    def test_workspace_write_run_compiles_exact_append_line_for_smoke_fixture(self) -> None:
        hidden_target = self.workspace / ".teamai" / "eval-fixtures" / "scratch.md"
        hidden_target.parent.mkdir(parents=True, exist_ok=True)
        hidden_target.write_text(
            "Seed line.\n",
            encoding="utf-8",
        )
        writable_settings = Settings(
            model_id="dummy",
            model_revision=None,
            force_download=False,
            trust_remote_code=False,
            enable_thinking=False,
            workspace_root=self.workspace,
            max_rounds=3,
            max_actions_per_round=2,
            max_tokens_per_turn=128,
            temperature=0.3,
            allow_shell=False,
            allow_writes=True,
            command_timeout_seconds=5,
            max_file_bytes=10_000,
            max_command_output_chars=10_000,
            host="127.0.0.1",
            port=8000,
        )

        result = ClosedLoopSupervisor(writable_settings, backend=FakeBackend([])).run(
            RunRequest(
                task="Append the exact line 'Eval harness scratch line.' to .teamai/eval-fixtures/scratch.md.",
                workspace_path=".",
                execution_mode="workspace_write",
                max_rounds=3,
                max_actions_per_round=2,
                max_tokens_per_turn=128,
            ),
        )

        self.assertEqual(result.task_route, "deterministic_patch")
        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stop_reason, "approval_required")
        self.assertEqual(len(result.rounds), 1)
        approval_files = sorted((self.workspace / ".teamai" / "approvals").glob("*.json"))
        self.assertEqual(len(approval_files), 1)
        approval_payload = approval_files[0].read_text(encoding="utf-8")
        self.assertIn(".teamai/eval-fixtures/scratch.md", approval_payload)
        self.assertIn("Eval harness scratch line.", approval_payload)

    def test_workspace_write_run_compiles_fenced_test_block_without_prior_read(self) -> None:
        test_file = self.workspace / "tests" / "test_sample.py"
        test_file.parent.mkdir(parents=True, exist_ok=True)
        test_file.write_text(
            "import unittest\n\nif __name__ == \"__main__\":\n    unittest.main()\n",
            encoding="utf-8",
        )
        writable_settings = Settings(
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
            allow_writes=True,
            command_timeout_seconds=5,
            max_file_bytes=10_000,
            max_command_output_chars=10_000,
            host="127.0.0.1",
            port=8000,
        )
        backend = FakeBackend(
            [
                "Add the requested test block.",
                "A single deterministic patch approval is enough.",
                '{"summary":"Need the concrete patch.","should_stop":false,"final_answer":null,"actions":[]}',
            ]
        )

        result = ClosedLoopSupervisor(writable_settings, backend=backend).run(
            RunRequest(
                task=(
                    "Use workspace_write mode. Insert the following block before 'if __name__ == \"__main__\":' "
                    "in tests/test_sample.py.\n"
                    "```python\n"
                    "class SampleTest(unittest.TestCase):\n"
                    "    def test_truth(self) -> None:\n"
                    "        self.assertTrue(True)\n"
                    "```\n"
                    "Stop as soon as the patch approval is created."
                ),
                workspace_path=".",
                execution_mode="workspace_write",
                max_rounds=1,
                max_actions_per_round=2,
            ),
        )

        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stop_reason, "approval_required")
        approval_files = sorted((self.workspace / ".teamai" / "approvals").glob("*.json"))
        self.assertEqual(len(approval_files), 1)
        approval_payload = approval_files[0].read_text(encoding="utf-8")
        self.assertIn("tests/test_sample.py", approval_payload)
        self.assertIn("class SampleTest(unittest.TestCase):", approval_payload)

    def test_workspace_write_run_uses_heuristic_replace_in_file_after_reading_target(self) -> None:
        (self.workspace / "README.md").write_text(
            "# teamAI\n\n"
            "In `workspace_write` mode, write actions no longer mutate files immediately. They create pending patch approvals under `.teamai/approvals/`, stop the run with `approval_required`, and tell you how to review or apply the patch.\n\n"
            "List pending approvals:\n",
            encoding="utf-8",
        )
        writable_settings = Settings(
            model_id="dummy",
            model_revision=None,
            force_download=False,
            trust_remote_code=False,
            enable_thinking=False,
            workspace_root=self.workspace,
            max_rounds=2,
            max_actions_per_round=3,
            max_tokens_per_turn=64,
            temperature=0.3,
            allow_shell=False,
            allow_writes=True,
            command_timeout_seconds=5,
            max_file_bytes=10_000,
            max_command_output_chars=10_000,
            host="127.0.0.1",
            port=8000,
        )
        backend = FakeBackend(
            [
                "Read README.md first.",
                "That anchors the requested patch.",
                (
                    '{"summary":"Read the target doc first.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"read_file","reason":"Need the target paragraph before proposing a patch.","args":{"path":"README.md","start_line":1,"end_line":120}}]}'
                ),
                '{"done":false,"confidence":0.2,"summary":"The README target paragraph is now available.","next_focus":"Create the requested patch approval."}',
                "Create the patch approval now.",
                "Use replace_in_file against the README paragraph.",
                '{"summary":"The patch approval is ready.","should_stop":true,"final_answer":"A replace_in_file patch approval has been created.","actions":[]}',
            ]
        )

        result = ClosedLoopSupervisor(writable_settings, backend=backend).run(
            RunRequest(
                task=(
                    "Use workspace_write mode. Read README.md, then create exactly one replace_in_file patch approval "
                    "that inserts the sentence 'Pending approvals may become stale if the target file changes before approval.' "
                    "immediately after the paragraph that starts with 'In workspace_write mode'. Do not inspect unrelated files. "
                    "Stop as soon as the patch approval is created."
                ),
                workspace_path=".",
                execution_mode="workspace_write",
                max_rounds=2,
                max_actions_per_round=3,
            ),
        )

        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stop_reason, "approval_required")
        self.assertIn("teamai approvals show", result.final_answer)
        self.assertEqual(result.task_route, "deterministic_patch")

    def test_broad_workspace_write_task_routes_to_codex_handoff_reconnaissance(self) -> None:
        (self.workspace / "README.md").write_text("# teamAI\n", encoding="utf-8")
        (self.workspace / "pyproject.toml").write_text('[project]\nname = "teamai"\n', encoding="utf-8")
        (self.workspace / "teamai").mkdir()
        (self.workspace / "teamai" / "supervisor.py").write_text("class ClosedLoopSupervisor:\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "cli.py").write_text("def main() -> None:\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "api.py").write_text("def create_app() -> None:\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "jobs.py").write_text("def run_job() -> None:\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "bridge.py").write_text("def launch_bridge() -> None:\n    pass\n", encoding="utf-8")
        writable_settings = Settings(
            model_id="dummy",
            model_revision=None,
            force_download=False,
            trust_remote_code=False,
            enable_thinking=False,
            workspace_root=self.workspace,
            max_rounds=2,
            max_actions_per_round=3,
            max_tokens_per_turn=64,
            temperature=0.3,
            allow_shell=False,
            allow_writes=True,
            command_timeout_seconds=5,
            max_file_bytes=10_000,
            max_command_output_chars=10_000,
            host="127.0.0.1",
            port=8000,
        )
        backend = FakeBackend(
            [
                "Inspect the main orchestration files first.",
                "Gather enough context for a Codex handoff before implementation.",
                (
                    '{"summary":"Map the workspace first.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"list_files","reason":"Inspect the root.","args":{"path":"."}}]}'
                ),
                '{"done":false,"confidence":0.3,"summary":"Enough context for handoff.","next_focus":"Inspect `teamai/supervisor.py` and `teamai/bridge.py` before implementing better task routing."}',
            ]
        )

        result = ClosedLoopSupervisor(writable_settings, backend=backend).run(
            RunRequest(
                task="Implement better task routing across the supervisor and bridge.",
                workspace_path=".",
                execution_mode="workspace_write",
                max_rounds=2,
                max_actions_per_round=3,
            ),
        )

        self.assertEqual(result.task_route, "codex_handoff")
        self.assertEqual(result.execution_mode, "read_only")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.stop_reason, "codex_handoff_synthesized")
        self.assertIn("Current state: The local run treated this as a broad coding task", result.final_answer)
        self.assertIn("teamai/cli.py", result.final_answer)
        self.assertIn("teamai/supervisor.py", result.final_answer)
        self.assertTrue(any("codex handoff" in warning.lower() for warning in result.warnings))

    def test_improve_task_routes_to_codex_handoff_reconnaissance(self) -> None:
        (self.workspace / "README.md").write_text("# teamAI\n", encoding="utf-8")
        (self.workspace / "pyproject.toml").write_text('[project]\nname = "teamai"\n', encoding="utf-8")
        (self.workspace / "teamai").mkdir()
        (self.workspace / "teamai" / "supervisor.py").write_text("class ClosedLoopSupervisor:\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "cli.py").write_text("def main() -> None:\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "api.py").write_text("def create_app() -> None:\n    pass\n", encoding="utf-8")
        backend = FakeBackend(
            [
                "Map the workspace first.",
                "Gather enough context for a Codex handoff before implementation.",
                (
                    '{"summary":"Map the workspace first.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"list_files","reason":"Inspect the root.","args":{"path":"."}}]}'
                ),
                '{"done":false,"confidence":0.3,"summary":"Enough context for handoff.","next_focus":"Inspect `teamai/supervisor.py` and `teamai/cli.py` before improving convergence."}',
            ]
        )

        result = ClosedLoopSupervisor(self.settings, backend=backend).run(
            RunRequest(
                task="Improve supervisor convergence and completion criteria.",
                workspace_path=".",
                max_rounds=1,
                max_actions_per_round=3,
            ),
        )

        self.assertEqual(result.task_route, "codex_handoff")
        self.assertEqual(result.stop_reason, "codex_handoff_synthesized")
        self.assertIn("Current state: The local run treated this as a broad coding task", result.final_answer)

    def test_explicit_write_loop_reroutes_after_repeated_low_confidence_drift(self) -> None:
        (self.workspace / "README.md").write_text(
            "# teamAI\n\nPatch approvals require explicit review.\n",
            encoding="utf-8",
        )
        writable_settings = Settings(
            model_id="dummy",
            model_revision=None,
            force_download=False,
            trust_remote_code=False,
            enable_thinking=False,
            workspace_root=self.workspace,
            max_rounds=4,
            max_actions_per_round=2,
            max_tokens_per_turn=64,
            temperature=0.3,
            allow_shell=False,
            allow_writes=True,
            command_timeout_seconds=5,
            max_file_bytes=10_000,
            max_command_output_chars=10_000,
            host="127.0.0.1",
            port=8000,
        )
        backend = FakeBackend(
            [
                "Read README.md first.",
                "Inspect the target paragraph before proposing any patch.",
                (
                    '{"summary":"Read the target file first.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"read_file","reason":"Inspect README.md first.","args":{"path":"README.md","start_line":1,"end_line":80}}]}'
                ),
                '{"done":false,"confidence":0.2,"summary":"Need a concrete patch.","next_focus":"Produce a concrete patch for README.md."}',
                "The target still needs a more concrete patch plan.",
                "Do not guess at the patch yet.",
                '{"summary":"Still clarifying the change.","should_stop":false,"final_answer":null,"actions":[]}',
                '{"done":false,"confidence":0.1,"summary":"Still no concrete patch.","next_focus":"Produce a concrete patch for README.md."}',
            ]
        )

        result = ClosedLoopSupervisor(writable_settings, backend=backend).run(
            RunRequest(
                task="Use workspace_write mode. Update README.md to clarify the patch approval caveat in the approval section.",
                workspace_path=".",
                execution_mode="workspace_write",
                max_rounds=4,
                max_actions_per_round=2,
            ),
        )

        self.assertEqual(result.task_route, "codex_handoff")
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.stop_reason, "local_drift_rerouted")
        self.assertIn("Current state: The local run started locally, gathered partial evidence", result.final_answer)
        self.assertIn("README.md", result.final_answer)
        self.assertTrue(any("drifted" in warning.lower() for warning in result.warnings))

    def test_priority_candidates_prioritize_cli_and_api_for_streaming_handoff(self) -> None:
        (self.workspace / "README.md").write_text("# teamAI\n", encoding="utf-8")
        (self.workspace / "pyproject.toml").write_text('[project]\nname = "teamai"\n', encoding="utf-8")
        (self.workspace / "teamai").mkdir()
        (self.workspace / "teamai" / "cli.py").write_text("def main() -> None:\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "api.py").write_text("def create_app() -> None:\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "jobs.py").write_text("def run_job() -> None:\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "supervisor.py").write_text("class ClosedLoopSupervisor:\n    pass\n", encoding="utf-8")

        supervisor = ClosedLoopSupervisor(self.settings, backend=FakeBackend([]))
        candidates = supervisor._priority_candidates(  # noqa: SLF001
            [],
            self.workspace,
            task="Implement streaming event output across the CLI and API.",
            task_route="codex_handoff",
        )

        self.assertIn("teamai/cli.py", candidates[:3])
        self.assertIn("teamai/api.py", candidates[:3])
        self.assertLess(candidates.index("teamai/cli.py"), candidates.index("README.md"))
        self.assertLess(candidates.index("teamai/api.py"), candidates.index("pyproject.toml"))

    def test_streaming_handoff_reads_cli_and_api_before_generic_docs(self) -> None:
        (self.workspace / "README.md").write_text("# teamAI\n", encoding="utf-8")
        (self.workspace / "pyproject.toml").write_text('[project]\nname = "teamai"\n', encoding="utf-8")
        (self.workspace / "teamai").mkdir()
        (self.workspace / "teamai" / "cli.py").write_text("def main() -> None:\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "api.py").write_text("def create_app() -> None:\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "jobs.py").write_text("def run_job() -> None:\n    pass\n", encoding="utf-8")
        backend = FakeBackend(
            [
                "Start with a quick workspace map.",
                "Then inspect the implementation entrypoints.",
                (
                    '{"summary":"Map the workspace first.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"list_files","reason":"Inspect the root.","args":{"path":"."}}]}'
                ),
                '{"done":false,"confidence":0.3,"summary":"Enough context for a Codex handoff.","next_focus":"Inspect `teamai/cli.py` and `teamai/api.py` before implementing streaming output."}',
            ]
        )

        result = ClosedLoopSupervisor(self.settings, backend=backend).run(
            RunRequest(
                task="Implement streaming event output across the CLI and API.",
                workspace_path=".",
                max_rounds=1,
                max_actions_per_round=3,
            ),
        )

        self.assertEqual(result.task_route, "codex_handoff")
        self.assertEqual(result.stop_reason, "codex_handoff_synthesized")
        self.assertIn("teamai/cli.py", result.final_answer)
        self.assertIn("teamai/api.py", result.final_answer)

    def test_priority_candidates_prioritize_memory_files_for_self_improvement_handoff(self) -> None:
        (self.workspace / "README.md").write_text("# teamAI\n", encoding="utf-8")
        (self.workspace / "pyproject.toml").write_text('[project]\nname = "teamai"\n', encoding="utf-8")
        (self.workspace / "teamai").mkdir()
        (self.workspace / "teamai" / "memory.py").write_text("class WorkspaceMemoryStore:\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "prompts.py").write_text("def build_round_context():\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "bridge.py").write_text("def launch_bridge():\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "handoff.py").write_text("def build_handoff_packet():\n    pass\n", encoding="utf-8")
        (self.workspace / "teamai" / "supervisor.py").write_text("class ClosedLoopSupervisor:\n    pass\n", encoding="utf-8")
        (self.workspace / "tests").mkdir()
        (self.workspace / "tests" / "test_memory.py").write_text("def test_memory():\n    pass\n", encoding="utf-8")

        supervisor = ClosedLoopSupervisor(self.settings, backend=FakeBackend([]))
        candidates = supervisor._priority_candidates(  # noqa: SLF001
            [],
            self.workspace,
            task=(
                "Improve self-improvement reconnaissance so learned-note, decay, pruning, and memory work "
                "prioritize teamai/memory.py and tests/test_memory.py before generic control-flow files."
            ),
            task_route="codex_handoff",
        )

        self.assertIn("teamai/memory.py", candidates[:3])
        self.assertIn("tests/test_memory.py", candidates[:4])
        self.assertLess(candidates.index("teamai/memory.py"), candidates.index("teamai/supervisor.py"))
        self.assertLess(candidates.index("tests/test_memory.py"), candidates.index("README.md"))

    def test_workspace_write_run_overrides_incorrect_model_replace_action(self) -> None:
        (self.workspace / "README.md").write_text(
            "# teamAI\n\n"
            "In `workspace_write` mode, write actions no longer mutate files immediately. They create pending patch approvals under `.teamai/approvals/`, stop the run with `approval_required`, and tell you how to review or apply the patch.\n\n"
            "List pending approvals:\n",
            encoding="utf-8",
        )
        writable_settings = Settings(
            model_id="dummy",
            model_revision=None,
            force_download=False,
            trust_remote_code=False,
            enable_thinking=False,
            workspace_root=self.workspace,
            max_rounds=2,
            max_actions_per_round=3,
            max_tokens_per_turn=64,
            temperature=0.3,
            allow_shell=False,
            allow_writes=True,
            command_timeout_seconds=5,
            max_file_bytes=10_000,
            max_command_output_chars=10_000,
            host="127.0.0.1",
            port=8000,
        )
        backend = FakeBackend(
            [
                "Read README.md first.",
                "That anchors the requested patch.",
                (
                    '{"summary":"Read the target doc first.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"read_file","reason":"Need the target paragraph before proposing a patch.","args":{"path":"README.md","start_line":1,"end_line":120}}]}'
                ),
                '{"done":false,"confidence":0.2,"summary":"The README target paragraph is now available.","next_focus":"Create the requested patch approval."}',
                "Create the patch approval now.",
                "Use replace_in_file against the README paragraph.",
                (
                    '{"summary":"Patch ready.","should_stop":false,"final_answer":null,"actions":['
                    '{"tool":"replace_in_file","reason":"Attempt the patch.","args":{"path":"README.md","old_text":"In `workspace_write` mode, write actions no longer mutate files immediately. They create pending patch approvals under `.teamai/approvals/`, stop the run with `approval_required`, and tell you how to review or apply the patch.","new_text":"In `workspace_write` mode, write actions","replace_all":false}}]}'
                ),
            ]
        )

        result = ClosedLoopSupervisor(writable_settings, backend=backend).run(
            RunRequest(
                task=(
                    "Use workspace_write mode. Read README.md, then create exactly one replace_in_file patch approval "
                    "that inserts the sentence 'Pending approvals may become stale if the target file changes before approval.' "
                    "immediately after the paragraph that starts with 'In workspace_write mode'. Do not inspect unrelated files. "
                    "Stop as soon as the patch approval is created."
                ),
                workspace_path=".",
                execution_mode="workspace_write",
                max_rounds=2,
                max_actions_per_round=3,
            ),
        )

        self.assertEqual(result.status, "stopped")
        self.assertEqual(result.stop_reason, "approval_required")
        approval_files = sorted((self.workspace / ".teamai" / "approvals").glob("*.json"))
        self.assertEqual(len(approval_files), 1)
        approval_payload = approval_files[0].read_text(encoding="utf-8")
        self.assertIn("Pending approvals may become stale if the target file changes before approval.", approval_payload)
        self.assertNotIn('"new_text":"In `workspace_write` mode, write actions"', approval_payload)

    def test_run_emits_progress_messages(self) -> None:
        backend = FakeBackend(
            [
                "Inspect the README first.",
                "That is a reasonable first move.",
                '{"summary":"No actions yet.","should_stop":false,"final_answer":null,"actions":[]}',
                '{"done":false,"confidence":0.1,"summary":"Still gathering context.","next_focus":"Inspect core files."}',
            ]
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=backend)
        progress: list[str] = []

        result = supervisor.run(
            RunRequest(task="Inspect the repo.", workspace_path=".", max_rounds=1),
            progress_callback=progress.append,
        )

        self.assertEqual(result.status, "stopped")
        self.assertTrue(any("Round 1/1: strategist" in item for item in progress))
        self.assertTrue(any("Round 1/1: planner" in item for item in progress))
        self.assertTrue(any("Stopped: max_rounds_reached" in item for item in progress))

    def test_run_emits_structured_events(self) -> None:
        backend = FakeBackend(
            [
                "Inspect the README first.",
                "That is a reasonable first move.",
                '{"summary":"No actions yet.","should_stop":false,"final_answer":null,"actions":[]}',
                '{"done":false,"confidence":0.1,"summary":"Still gathering context.","next_focus":"Inspect core files."}',
            ]
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=backend)
        events: list[RunEvent] = []

        result = supervisor.run(
            RunRequest(task="Inspect the repo.", workspace_path=".", max_rounds=1),
            event_callback=events.append,
        )

        self.assertEqual(result.status, "stopped")
        self.assertGreaterEqual(len(events), 5)
        self.assertEqual(events[0].kind, "run_started")
        self.assertEqual(events[1].kind, "task_route_selected")
        self.assertTrue(any(event.kind == "round_stage" and event.stage == "strategist" for event in events))
        self.assertEqual(events[-1].kind, "run_stopped")
        self.assertTrue(events[-1].terminal)

    def test_continuation_context_adds_scoped_verification_probe_before_resuming(self) -> None:
        (self.workspace / "README.md").write_text("# demo\npatched line\n", encoding="utf-8")
        backend = FakeBackend(
            [
                "Verify the patch and finish the remaining task.",
                "Focus on the verified README contents before proposing more work.",
                '{"summary":"No additional changes are needed.","should_stop":false,"final_answer":null,"actions":[]}',
                '{"done":false,"confidence":0.1,"summary":"Verification is complete.","next_focus":"Finish the remaining task."}',
            ]
        )
        supervisor = ClosedLoopSupervisor(self.settings, backend=backend)

        result = supervisor.run(
            RunRequest(
                task="Continue the approved change.",
                workspace_path=".",
                max_rounds=1,
                continuation_context={
                    "approval_id": "abc123",
                    "path": "README.md",
                    "source_tool": "replace_in_file",
                    "verification_focus": "Read README.md and confirm the approved replacement landed.",
                    "suggested_read_paths": ["README.md"],
                    "suggested_commands": [],
                },
            )
        )

        self.assertEqual(result.rounds[0].round_number, 0)
        self.assertEqual(result.rounds[0].planner.summary, "Ran deterministic post-approval verification before continuing.")
        self.assertEqual(result.rounds[0].tool_results[0].tool, "read_file")
        self.assertIn("patched line", result.rounds[0].tool_results[0].output)
        self.assertEqual(result.task_route, "multi_agent_loop")


if __name__ == "__main__":
    unittest.main()
