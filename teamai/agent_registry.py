"""
Agent registry for teamAI.

Loads agent capability profiles from agents.yaml (searched at the project root
and ~/.teamai/agents.yaml as a user-level override). Provides capability-based
agent selection for the orchestrator.

Example:
    registry = AgentRegistry.load()
    agent = registry.pick_best(capability="codex_handoff")
    print(agent.id, agent.model_id)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_COST_ORDER = {"free": 0, "cheap": 1, "moderate": 2, "expensive": 3}
_DEFAULT_REGISTRY_SEARCH = [
    Path("agents.yaml"),                         # repo root (cwd)
    Path("~/.teamai/agents.yaml"),               # user-level override
]


@dataclass
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
        """Return True if all required environment variables are set."""
        import os
        return all(os.environ.get(var, "").strip() for var in self.requires_env)

    @property
    def cost_rank(self) -> int:
        return _COST_ORDER.get(self.cost, 99)


@dataclass
class RoutingConfig:
    default_local: str = "local_gemma"
    default_cloud: str = "codex"
    require_all_capabilities: bool = False
    prefer_lower_cost: bool = True


class AgentRegistry:
    """Registry of available agents loaded from agents.yaml."""

    def __init__(self, agents: list[AgentEntry], routing: RoutingConfig) -> None:
        self._agents = agents
        self._routing = routing
        self._by_id: dict[str, AgentEntry] = {a.id: a for a in agents}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, search_paths: list[Path] | None = None) -> "AgentRegistry":
        """Load the registry from the first agents.yaml found in search_paths."""
        paths = search_paths or [p.expanduser() for p in _DEFAULT_REGISTRY_SEARCH]
        # Also try cwd-relative lookup
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
            # PyYAML not available; return empty registry
            return cls._empty()

        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        agents = [cls._parse_agent(entry) for entry in (raw.get("agents") or [])]
        routing = cls._parse_routing(raw.get("routing") or {})
        return cls(agents=[a for a in agents if a is not None], routing=routing)  # type: ignore[misc]

    @classmethod
    def _empty(cls) -> "AgentRegistry":
        return cls(agents=[], routing=RoutingConfig())

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
        return RoutingConfig(
            default_local=str(raw.get("default_local", "local_gemma")),
            default_cloud=str(raw.get("default_cloud", "codex")),
            require_all_capabilities=bool(raw.get("require_all_capabilities", False)),
            prefer_lower_cost=bool(raw.get("prefer_lower_cost", True)),
        )

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    @property
    def agents(self) -> list[AgentEntry]:
        return list(self._agents)

    def get(self, agent_id: str) -> AgentEntry | None:
        return self._by_id.get(agent_id)

    def pick_best(
        self,
        capability: str,
        *,
        prefer_local: bool = False,
        env_check: bool = True,
    ) -> AgentEntry | None:
        """Return the best available agent for the given capability.

        Selection criteria (in order):
        1. Must support the requested capability
        2. If env_check=True, required env vars must be set
        3. If prefer_local=True, local agents ranked first
        4. If prefer_lower_cost=True (registry default), lower cost ranked higher
        """
        candidates = [a for a in self._agents if a.supports(capability)]
        if env_check:
            candidates = [a for a in candidates if a.env_satisfied()]
        if not candidates:
            return None

        def sort_key(a: AgentEntry) -> tuple[int, int]:
            local_rank = 0 if (prefer_local and a.type == "local_mlx") else 1
            cost_rank = a.cost_rank if self._routing.prefer_lower_cost else 0
            return (local_rank, cost_rank)

        return sorted(candidates, key=sort_key)[0]

    def pick_local(self) -> AgentEntry | None:
        """Return the default local agent."""
        return self._by_id.get(self._routing.default_local)

    def pick_cloud(self, *, env_check: bool = True) -> AgentEntry | None:
        """Return the default cloud agent (if env vars are satisfied)."""
        agent = self._by_id.get(self._routing.default_cloud)
        if agent and env_check and not agent.env_satisfied():
            return None
        return agent

    def capable_of(self, capability: str, *, env_check: bool = True) -> list[AgentEntry]:
        """Return all agents capable of handling the given task route."""
        result = [a for a in self._agents if a.supports(capability)]
        if env_check:
            result = [a for a in result if a.env_satisfied()]
        return result

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render_table(self) -> str:
        if not self._agents:
            return "No agents registered. Create agents.yaml in your project root."

        col_widths = [
            max(len("ID"), max(len(a.id) for a in self._agents)),
            max(len("Type"), max(len(a.type) for a in self._agents)),
            max(len("Cost"), max(len(a.cost) for a in self._agents)),
            max(len("Latency"), max(len(a.latency) for a in self._agents)),
            max(len("Capabilities"), max(len(", ".join(a.capabilities)) for a in self._agents)),
        ]
        header = (
            f"{'ID':<{col_widths[0]}}  "
            f"{'Type':<{col_widths[1]}}  "
            f"{'Cost':<{col_widths[2]}}  "
            f"{'Latency':<{col_widths[3]}}  "
            f"Capabilities"
        )
        sep = "-" * (sum(col_widths) + 8 + 14)
        rows = [header, sep]
        for a in self._agents:
            caps = ", ".join(a.capabilities)
            env_warning = " [env missing]" if a.requires_env and not a.env_satisfied() else ""
            rows.append(
                f"{a.id:<{col_widths[0]}}  "
                f"{a.type:<{col_widths[1]}}  "
                f"{a.cost:<{col_widths[2]}}  "
                f"{a.latency:<{col_widths[3]}}  "
                f"{caps}{env_warning}"
            )
        return "\n".join(rows)
