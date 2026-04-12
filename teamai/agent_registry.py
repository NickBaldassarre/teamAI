"""
Agent registry for teamAI.

Loads agent capability profiles from agents.yaml (searched at the project root
and ~/.teamai/agents.yaml as a user-level override). The registry owns both
simple capability lookup and task-signature-based routing decisions so the
supervisor does not need its own hardcoded route ladder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_COST_ORDER = {"free": 0, "cheap": 1, "moderate": 2, "expensive": 3}
_DEFAULT_REGISTRY_SEARCH = [
    Path("agents.yaml"),
    Path("~/.teamai/agents.yaml"),
]
_LOCAL_AGENT_TYPES = {"deterministic", "local_mlx"}


@dataclass(frozen=True)
class AgentEntry:
    id: str
    name: str
    type: str
    model_id: str
    capabilities: list[str]
    max_context_tokens: int
    cost: str
    latency: str
    requires_env: list[str]
    notes: str = ""

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def env_satisfied(self) -> bool:
        import os

        return all(os.environ.get(var, "").strip() for var in self.requires_env)

    @property
    def cost_rank(self) -> int:
        return _COST_ORDER.get(self.cost, 99)

    @property
    def is_local(self) -> bool:
        return self.type in _LOCAL_AGENT_TYPES


@dataclass(frozen=True)
class RouteHealth:
    capability: str
    recent_runs: int = 0
    success_rate: float = 0.0
    repair_rate: float = 0.0
    average_retries: float = 0.0
    verifier_disagreement_rate: float = 0.0
    broken: bool = False
    memory_pressure: bool = False


@dataclass(frozen=True)
class TaskSignature:
    task: str
    execution_mode: str
    complexity: str = "low"
    continuation: bool = False
    deterministic_candidate: bool = False
    explicit_write: bool = False
    repository_inspection: bool = False
    broad_coding: bool = False
    memory_pressure: bool = False
    expected_files_touched: int = 0
    has_tests: bool = False
    policy_scores: dict[str, float] = field(default_factory=dict)
    agent_policy_scores: dict[str, float] = field(default_factory=dict)
    route_health: dict[str, RouteHealth] = field(default_factory=dict)

    def flag_value(self, key: str) -> bool:
        normalized = key.strip().lower()
        flags = {
            "continuation": self.continuation,
            "continuation_context": self.continuation,
            "deterministic_candidate": self.deterministic_candidate,
            "explicit_write": self.explicit_write,
            "repository_inspection": self.repository_inspection,
            "broad_coding": self.broad_coding,
            "memory_pressure": self.memory_pressure,
            "complexity_low": self.complexity == "low",
            "complexity_medium": self.complexity == "medium",
            "complexity_high": self.complexity == "high",
            "has_tests": self.has_tests,
            "workspace_write": self.execution_mode == "workspace_write",
            "read_only": self.execution_mode == "read_only",
        }
        return bool(flags.get(normalized, False))


@dataclass(frozen=True)
class TaskProfile:
    capability: str
    base_score: int = 0
    score_if: dict[str, int] = field(default_factory=dict)


@dataclass
class RoutingConfig:
    default_local: str = "local_gemma"
    default_cloud: str = "codex"
    require_all_capabilities: bool = False
    prefer_lower_cost: bool = True
    recent_success_window: int = 8
    success_rate_weight: int = 8
    repair_rate_weight: int = 12
    retry_weight: int = 6
    verifier_disagreement_weight: int = 8
    broken_repair_rate: float = 0.35
    broken_route_penalty: int = 18
    memory_pressure_penalty: int = 6
    local_preference_bonus: int = 2
    task_profiles: dict[str, TaskProfile] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingDecision:
    agent: AgentEntry
    capability: str
    score: float
    reasons: tuple[str, ...] = ()
    health: RouteHealth = field(default_factory=lambda: RouteHealth(capability="multi_agent_loop"))


class AgentRegistry:
    """Registry of available agents loaded from agents.yaml."""

    def __init__(self, agents: list[AgentEntry], routing: RoutingConfig) -> None:
        self._agents = agents
        self._routing = routing
        self._by_id: dict[str, AgentEntry] = {a.id: a for a in agents}

    @classmethod
    def load(cls, search_paths: list[Path] | None = None) -> "AgentRegistry":
        paths = search_paths or [p.expanduser() for p in _DEFAULT_REGISTRY_SEARCH]
        cwd = Path.cwd()
        resolved = [p if p.is_absolute() else cwd / p for p in paths]

        for path in resolved:
            if path.exists():
                return cls._from_yaml(path)

        return cls._empty()

    @classmethod
    def _from_yaml(cls, path: Path) -> "AgentRegistry":
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError:
            return cls._empty()

        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        agents = [cls._parse_agent(entry) for entry in (raw.get("agents") or [])]
        routing = cls._parse_routing(raw.get("routing") or {})
        return cls(agents=[a for a in agents if a is not None], routing=routing)  # type: ignore[misc]

    @classmethod
    def _empty(cls) -> "AgentRegistry":
        return cls(agents=[], routing=RoutingConfig(task_profiles=_default_task_profiles()))

    @staticmethod
    def _parse_agent(raw: dict[str, Any]) -> AgentEntry | None:
        try:
            return AgentEntry(
                id=str(raw["id"]),
                name=str(raw.get("name", raw["id"])),
                type=str(raw.get("type", "unknown")),
                model_id=str(raw.get("model_id", "")),
                capabilities=[str(c) for c in (raw.get("capabilities") or [])],
                max_context_tokens=int(raw.get("max_context_tokens", 0)),
                cost=str(raw.get("cost", "unknown")),
                latency=str(raw.get("latency", "unknown")),
                requires_env=[str(e) for e in (raw.get("requires_env") or [])],
                notes=str(raw.get("notes", "") or "").strip(),
            )
        except (KeyError, TypeError, ValueError):
            return None

    @staticmethod
    def _parse_routing(raw: dict[str, Any]) -> RoutingConfig:
        raw_profiles = raw.get("task_profiles") or {}
        profiles = _default_task_profiles()
        for capability, entry in raw_profiles.items():
            if not isinstance(entry, dict):
                continue
            profiles[str(capability)] = TaskProfile(
                capability=str(capability),
                base_score=int(entry.get("base_score", profiles.get(str(capability), TaskProfile(str(capability))).base_score)),
                score_if={
                    str(key): int(value)
                    for key, value in (entry.get("score_if") or {}).items()
                    if isinstance(value, (int, float))
                },
            )

        return RoutingConfig(
            default_local=str(raw.get("default_local", "local_gemma")),
            default_cloud=str(raw.get("default_cloud", "codex")),
            require_all_capabilities=bool(raw.get("require_all_capabilities", False)),
            prefer_lower_cost=bool(raw.get("prefer_lower_cost", True)),
            recent_success_window=int(raw.get("recent_success_window", 8) or 8),
            success_rate_weight=int(raw.get("success_rate_weight", 8) or 8),
            repair_rate_weight=int(raw.get("repair_rate_weight", 12) or 12),
            retry_weight=int(raw.get("retry_weight", 6) or 6),
            verifier_disagreement_weight=int(raw.get("verifier_disagreement_weight", 8) or 8),
            broken_repair_rate=float(raw.get("broken_repair_rate", 0.35) or 0.35),
            broken_route_penalty=int(raw.get("broken_route_penalty", 18) or 18),
            memory_pressure_penalty=int(raw.get("memory_pressure_penalty", 6) or 6),
            local_preference_bonus=int(raw.get("local_preference_bonus", 2) or 2),
            task_profiles=profiles,
        )

    @property
    def agents(self) -> list[AgentEntry]:
        return list(self._agents)

    @property
    def routing(self) -> RoutingConfig:
        return self._routing

    def get(self, agent_id: str) -> AgentEntry | None:
        return self._by_id.get(agent_id)

    def pick_best(
        self,
        capability: str | None = None,
        *,
        task_signature: TaskSignature | None = None,
        prefer_local: bool = False,
        env_check: bool = True,
    ) -> AgentEntry | RoutingDecision | None:
        if task_signature is not None:
            return self._pick_best_for_task(
                task_signature=task_signature,
                prefer_local=prefer_local,
                env_check=env_check,
            )
        if capability is None:
            return None

        candidates = [a for a in self._agents if a.supports(capability)]
        if env_check:
            candidates = [a for a in candidates if a.env_satisfied()]
        if not candidates:
            return None

        def sort_key(agent: AgentEntry) -> tuple[int, int]:
            local_rank = 0 if (prefer_local and agent.is_local) else 1
            cost_rank = agent.cost_rank if self._routing.prefer_lower_cost else 0
            return (local_rank, cost_rank)

        return sorted(candidates, key=sort_key)[0]

    def pick_local(self) -> AgentEntry | None:
        return self._by_id.get(self._routing.default_local)

    def pick_cloud(self, *, env_check: bool = True) -> AgentEntry | None:
        agent = self._by_id.get(self._routing.default_cloud)
        if agent and env_check and not agent.env_satisfied():
            return None
        return agent

    def capable_of(self, capability: str, *, env_check: bool = True) -> list[AgentEntry]:
        result = [a for a in self._agents if a.supports(capability)]
        if env_check:
            result = [a for a in result if a.env_satisfied()]
        return result

    def stronger_local_candidates(
        self,
        *,
        current_model_id: str,
        capabilities: list[str],
        env_check: bool = True,
    ) -> list[AgentEntry]:
        current = next((agent for agent in self._agents if agent.model_id == current_model_id), None)
        current_context = current.max_context_tokens if current is not None else 0
        matches = [
            agent
            for agent in self._agents
            if agent.is_local
            and agent.model_id != current_model_id
            and any(capability in agent.capabilities for capability in capabilities)
        ]
        if env_check:
            matches = [agent for agent in matches if agent.env_satisfied()]
        matches.sort(
            key=lambda agent: (
                agent.max_context_tokens > current_context,
                agent.max_context_tokens,
                -agent.cost_rank,
            ),
            reverse=True,
        )
        return matches

    def render_table(self) -> str:
        if not self._agents:
            return "No agents registered. Create agents.yaml in your project root."

        col_widths = [
            max(len("ID"), max(len(agent.id) for agent in self._agents)),
            max(len("Type"), max(len(agent.type) for agent in self._agents)),
            max(len("Cost"), max(len(agent.cost) for agent in self._agents)),
            max(len("Latency"), max(len(agent.latency) for agent in self._agents)),
            max(len("Capabilities"), max(len(", ".join(agent.capabilities)) for agent in self._agents)),
        ]
        header = (
            f"{'ID':<{col_widths[0]}}  "
            f"{'Type':<{col_widths[1]}}  "
            f"{'Cost':<{col_widths[2]}}  "
            f"{'Latency':<{col_widths[3]}}  "
            f"Capabilities"
        )
        separator = "-" * (sum(col_widths) + 8 + 14)
        rows = [header, separator]
        for agent in self._agents:
            caps = ", ".join(agent.capabilities)
            env_warning = " [env missing]" if agent.requires_env and not agent.env_satisfied() else ""
            rows.append(
                f"{agent.id:<{col_widths[0]}}  "
                f"{agent.type:<{col_widths[1]}}  "
                f"{agent.cost:<{col_widths[2]}}  "
                f"{agent.latency:<{col_widths[3]}}  "
                f"{caps}{env_warning}"
            )
        return "\n".join(rows)

    def _pick_best_for_task(
        self,
        *,
        task_signature: TaskSignature,
        prefer_local: bool,
        env_check: bool,
    ) -> RoutingDecision | None:
        candidates: list[RoutingDecision] = []

        for agent in self._agents:
            if env_check and not agent.env_satisfied():
                continue
            for capability in agent.capabilities:
                if capability == "deterministic_patch" and not task_signature.deterministic_candidate:
                    continue
                profile = self._routing.task_profiles.get(capability, TaskProfile(capability=capability))
                health = task_signature.route_health.get(capability, RouteHealth(capability=capability))
                score, reasons = self._score_candidate(
                    agent=agent,
                    capability=capability,
                    profile=profile,
                    task_signature=task_signature,
                    health=health,
                    prefer_local=prefer_local,
                )
                candidates.append(
                    RoutingDecision(
                        agent=agent,
                        capability=capability,
                        score=score,
                        reasons=tuple(reasons),
                        health=health,
                    )
                )

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda decision: (
                decision.score,
                1 if decision.agent.is_local else 0,
                -decision.agent.cost_rank,
                decision.agent.max_context_tokens,
            ),
        )

    def _score_candidate(
        self,
        *,
        agent: AgentEntry,
        capability: str,
        profile: TaskProfile,
        task_signature: TaskSignature,
        health: RouteHealth,
        prefer_local: bool,
    ) -> tuple[float, list[str]]:
        score = float(profile.base_score)
        reasons: list[str] = []

        for flag, weight in profile.score_if.items():
            if task_signature.flag_value(flag):
                score += weight
                reasons.append(f"{flag}={weight:+d}")

        if prefer_local and agent.is_local:
            score += self._routing.local_preference_bonus
            reasons.append(f"prefer_local={self._routing.local_preference_bonus:+d}")

        if self._routing.prefer_lower_cost:
            cost_penalty = float(agent.cost_rank)
            score -= cost_penalty
            reasons.append(f"cost={-cost_penalty:+.1f}")

        if health.recent_runs:
            success_bonus = round(health.success_rate * self._routing.success_rate_weight, 2)
            score += success_bonus
            reasons.append(f"success_rate={success_bonus:+.2f}")

            repair_penalty = round(health.repair_rate * self._routing.repair_rate_weight, 2)
            if repair_penalty:
                score -= repair_penalty
                reasons.append(f"repair_rate={-repair_penalty:+.2f}")

            retry_penalty = round(health.average_retries * self._routing.retry_weight, 2)
            if retry_penalty:
                score -= retry_penalty
                reasons.append(f"average_retries={-retry_penalty:+.2f}")

            disagreement_penalty = round(
                health.verifier_disagreement_rate * self._routing.verifier_disagreement_weight,
                2,
            )
            if disagreement_penalty:
                score -= disagreement_penalty
                reasons.append(f"verifier_disagreement={-disagreement_penalty:+.2f}")

        if health.broken:
            score -= self._routing.broken_route_penalty
            reasons.append(f"broken_route={-self._routing.broken_route_penalty:+d}")

        if task_signature.memory_pressure and agent.type == "local_mlx":
            score -= self._routing.memory_pressure_penalty
            reasons.append(f"memory_pressure={-self._routing.memory_pressure_penalty:+d}")

        policy_adjustment = float(task_signature.policy_scores.get(capability, 0.0))
        if policy_adjustment:
            score += policy_adjustment
            reasons.append(f"policy_memory={policy_adjustment:+.2f}")

        agent_policy_adjustment = float(
            task_signature.agent_policy_scores.get(agent.id, task_signature.agent_policy_scores.get(agent.model_id, 0.0))
        )
        if agent_policy_adjustment:
            score += agent_policy_adjustment
            reasons.append(f"agent_policy_memory={agent_policy_adjustment:+.2f}")

        if task_signature.complexity == "high":
            if not agent.is_local and capability in {"verified_handoff_execution", "codex_handoff"}:
                score += 6
                reasons.append("high_complexity_cloud=+6")
            if agent.is_local and agent.max_context_tokens < 4096:
                score -= 6
                reasons.append("high_complexity_small_context=-6")
        elif task_signature.complexity == "medium":
            if agent.is_local and agent.max_context_tokens >= 4096:
                score += 3
                reasons.append("medium_complexity_large_context=+3")

        return score, reasons


def _default_task_profiles() -> dict[str, TaskProfile]:
    return {
        "deterministic_patch": TaskProfile(
            capability="deterministic_patch",
            base_score=0,
            score_if={
                "deterministic_candidate": 18,
                "workspace_write": 4,
                "explicit_write": 5,
                "complexity_low": 6,
                "complexity_medium": -4,
                "complexity_high": -10,
                "continuation_context": -4,
                "repository_inspection": -12,
                "broad_coding": -14,
            },
        ),
        "repository_inspection": TaskProfile(
            capability="repository_inspection",
            base_score=0,
            score_if={
                "repository_inspection": 18,
                "read_only": 2,
                "complexity_low": 2,
                "complexity_medium": -2,
                "complexity_high": -8,
                "explicit_write": -10,
                "deterministic_candidate": -10,
                "broad_coding": -6,
            },
        ),
        "explicit_write_loop": TaskProfile(
            capability="explicit_write_loop",
            base_score=0,
            score_if={
                "explicit_write": 16,
                "workspace_write": 4,
                "complexity_medium": 6,
                "complexity_high": -4,
                "deterministic_candidate": -10,
                "repository_inspection": -10,
                "broad_coding": -4,
            },
        ),
        "codex_handoff": TaskProfile(
            capability="codex_handoff",
            base_score=0,
            score_if={
                "broad_coding": 18,
                "workspace_write": 2,
                "complexity_high": 8,
                "repository_inspection": -8,
                "deterministic_candidate": -12,
            },
        ),
        "multi_agent_loop": TaskProfile(
            capability="multi_agent_loop",
            base_score=1,
            score_if={
                "continuation_context": 8,
                "complexity_medium": 4,
                "complexity_high": -6,
                "broad_coding": -4,
            },
        ),
    }
