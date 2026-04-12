from __future__ import annotations

import unittest
from pathlib import Path

from teamai.agent_registry import AgentRegistry, RouteHealth, TaskSignature


class AgentRegistryRoutingTest(unittest.TestCase):
    def test_pick_best_routes_deterministic_candidate_to_deterministic_agent(self) -> None:
        registry = AgentRegistry.load(search_paths=[Path("/Users/home/Documents/teamAI/agents.yaml")])
        signature = TaskSignature(
            task="Append one line to README.md.",
            execution_mode="workspace_write",
            deterministic_candidate=True,
            explicit_write=True,
            route_health={
                "deterministic_patch": RouteHealth(capability="deterministic_patch", recent_runs=3, success_rate=1.0),
                "explicit_write_loop": RouteHealth(
                    capability="explicit_write_loop",
                    recent_runs=3,
                    success_rate=0.5,
                    repair_rate=0.5,
                    broken=True,
                ),
            },
        )

        decision = registry.pick_best(task_signature=signature, prefer_local=True, env_check=False)

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.capability, "deterministic_patch")
        self.assertEqual(decision.agent.id, "deterministic_patcher")

    def test_pick_best_demotes_broken_route_with_high_repair_rate(self) -> None:
        registry = AgentRegistry.load(search_paths=[Path("/Users/home/Documents/teamAI/agents.yaml")])
        signature = TaskSignature(
            task="Investigate this repo and summarize next steps.",
            execution_mode="read_only",
            repository_inspection=True,
            route_health={
                "repository_inspection": RouteHealth(
                    capability="repository_inspection",
                    recent_runs=4,
                    success_rate=0.75,
                    repair_rate=0.6,
                    broken=True,
                ),
                "multi_agent_loop": RouteHealth(
                    capability="multi_agent_loop",
                    recent_runs=4,
                    success_rate=1.0,
                    repair_rate=0.0,
                    broken=False,
                ),
            },
        )

        decision = registry.pick_best(task_signature=signature, prefer_local=True, env_check=False)

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertNotEqual(decision.capability, "repository_inspection")

    def test_pick_best_never_routes_to_deterministic_patch_without_a_compiler_match(self) -> None:
        registry = AgentRegistry.load(search_paths=[Path("/Users/home/Documents/teamAI/agents.yaml")])
        signature = TaskSignature(
            task="Update app.py and tests/test_app.py so the feature works end-to-end.",
            execution_mode="workspace_write",
            explicit_write=True,
            route_health={
                "deterministic_patch": RouteHealth(capability="deterministic_patch", recent_runs=8, success_rate=1.0),
                "explicit_write_loop": RouteHealth(
                    capability="explicit_write_loop",
                    recent_runs=2,
                    success_rate=0.0,
                    repair_rate=0.3,
                    broken=False,
                ),
            },
        )

        decision = registry.pick_best(task_signature=signature, prefer_local=True, env_check=False)

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertNotEqual(decision.capability, "deterministic_patch")

    def test_pick_best_uses_policy_memory_and_complexity_to_avoid_undersized_route(self) -> None:
        registry = AgentRegistry.load(search_paths=[Path("/Users/home/Documents/teamAI/agents.yaml")])
        signature = TaskSignature(
            task="Refactor routing across teamai/supervisor.py and teamai/agent_registry.py with test updates.",
            execution_mode="workspace_write",
            complexity="high",
            explicit_write=True,
            has_tests=True,
            policy_scores={
                "explicit_write_loop": 6.0,
                "codex_handoff": 8.0,
                "multi_agent_loop": -4.0,
            },
            route_health={
                "explicit_write_loop": RouteHealth(
                    capability="explicit_write_loop",
                    recent_runs=4,
                    success_rate=0.5,
                    average_retries=2.0,
                    verifier_disagreement_rate=0.5,
                ),
                "multi_agent_loop": RouteHealth(
                    capability="multi_agent_loop",
                    recent_runs=4,
                    success_rate=1.0,
                    average_retries=0.0,
                    verifier_disagreement_rate=0.0,
                ),
            },
        )

        decision = registry.pick_best(task_signature=signature, prefer_local=True, env_check=False)

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertNotEqual(decision.agent.id, "local_gemma")

    def test_pick_best_uses_agent_policy_memory_to_prefer_stronger_successful_model(self) -> None:
        registry = AgentRegistry.load(search_paths=[Path("/Users/home/Documents/teamAI/agents.yaml")])
        signature = TaskSignature(
            task="Fix routing across teamai/supervisor.py and tests/test_supervisor.py after repeated local failures.",
            execution_mode="workspace_write",
            complexity="high",
            explicit_write=True,
            has_tests=True,
            agent_policy_scores={
                "mlx-community/gemma-4-2b-it-4bit": -3.0,
                "mlx-community/gemma-4-12b-it-4bit": 6.5,
            },
            route_health={
                "explicit_write_loop": RouteHealth(
                    capability="explicit_write_loop",
                    recent_runs=4,
                    success_rate=0.75,
                    average_retries=0.5,
                )
            },
        )

        decision = registry.pick_best(task_signature=signature, prefer_local=True, env_check=False)

        self.assertIsNotNone(decision)
        assert decision is not None
        self.assertEqual(decision.agent.id, "local_gemma_large")


if __name__ == "__main__":
    unittest.main()
