"""
Agent Definitions — configurable agent archetypes with capabilities.

Each agent has:
    - A role (navigator, manipulator, researcher, coordinator)
    - A capability set (which actions it can perform)
    - A knowledge set (what domain information it has access to)
    - A state (current position, inventory, observations)

An AgentTeam is a collection of agents assigned to a task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import torch
import torch.nn as nn


# ─── Action Vocabulary ─────────────────────────────────────────────────────────

# Shared action vocabulary for all environments
ACTION_VOCAB = {
    # Navigation
    "navigate": 0, "look": 1, "examine": 2, "open": 3, "close": 4,
    "go_north": 5, "go_south": 6, "go_east": 7, "go_west": 8,
    # Manipulation
    "pick_up": 10, "put_down": 11, "use": 12, "combine": 13,
    "heat": 14, "cool": 15, "clean": 16, "slice": 17, "toggle": 18,
    # Web / Research
    "search": 20, "read": 21, "click": 22, "type": 23, "scroll": 24,
    "bookmark": 25, "submit": 26, "navigate_url": 27,
    # Communication
    "say": 30, "ask": 31, "report": 32, "confirm": 33, "request": 34,
    "delegate": 35, "acknowledge": 36,
    # Meta / Coordination
    "plan": 40, "verify": 41, "summarize": 42, "wait": 43,
    "decompose": 44, "assign": 45, "monitor": 46,
    # Special
    "nop": 100, "done": 101, "fail": 102,
}

VOCAB_SIZE = 128  # Padded vocab size


# ─── Agent Dataclass ───────────────────────────────────────────────────────────

@dataclass
class AgentConfig:
    """Configuration for a single agent."""
    name: str
    role: str
    capabilities: Set[str]
    knowledge_domains: Set[str]
    description: str = ""

    def capability_vector(self) -> torch.Tensor:
        """Returns a binary capability vector."""
        vec = torch.zeros(VOCAB_SIZE)
        for action_name in self.capabilities:
            if action_name in ACTION_VOCAB:
                vec[ACTION_VOCAB[action_name]] = 1.0
        return vec


@dataclass
class AgentState:
    """Mutable state of an agent during task execution."""
    agent_id: int
    position: str = "start"
    inventory: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    energy: float = 1.0  # Remaining action budget (normalized)

    def to_tensor(self, dim: int = 128) -> torch.Tensor:
        """
        Encode state as a fixed-size tensor.

        This is a simplified encoding — in practice you'd use a more
        sophisticated state encoder (e.g., text embedding of observations).
        """
        # For now: one-hot position + inventory count + energy + action count
        vec = torch.zeros(dim)
        vec[0] = hash(self.position) % 64 / 64.0  # Hashed position
        vec[1] = len(self.inventory) / 10.0
        vec[2] = len(self.observations) / 50.0
        vec[3] = len(self.actions_taken) / 50.0
        vec[4] = self.energy
        return vec


# ─── Predefined Archetypes ────────────────────────────────────────────────────

AGENT_ARCHETYPES = {
    "navigator": AgentConfig(
        name="Navigator",
        role="navigator",
        description="Specializes in spatial reasoning and navigation",
        capabilities={
            "navigate", "look", "examine", "open", "close",
            "go_north", "go_south", "go_east", "go_west",
            "say", "report", "acknowledge",
        },
        knowledge_domains={"spatial", "layout"},
    ),
    "manipulator": AgentConfig(
        name="Manipulator",
        role="manipulator",
        description="Specializes in object interaction and tool use",
        capabilities={
            "pick_up", "put_down", "use", "combine",
            "heat", "cool", "clean", "slice", "toggle",
            "look", "examine",
            "say", "report", "confirm", "acknowledge",
        },
        knowledge_domains={"objects", "tools", "recipes"},
    ),
    "researcher": AgentConfig(
        name="Researcher",
        role="researcher",
        description="Specializes in information lookup and web tasks",
        capabilities={
            "search", "read", "click", "type", "scroll",
            "bookmark", "submit", "navigate_url",
            "say", "report", "acknowledge",
        },
        knowledge_domains={"web", "documents", "databases"},
    ),
    "coordinator": AgentConfig(
        name="Coordinator",
        role="coordinator",
        description="Manages task decomposition and delegation",
        capabilities={
            "plan", "verify", "summarize", "decompose",
            "assign", "monitor", "delegate",
            "say", "ask", "report", "confirm", "request", "acknowledge",
        },
        knowledge_domains={"meta", "task_structure"},
    ),
}


# ─── Agent and Team ────────────────────────────────────────────────────────────

class Agent:
    """A single agent with config and mutable state."""

    def __init__(self, config: AgentConfig, agent_id: int = 0):
        self.config = config
        self.state = AgentState(agent_id=agent_id)
        self.agent_id = agent_id

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def role(self) -> str:
        return self.config.role

    def can_do(self, action: str) -> bool:
        """Check if this agent can perform the given action."""
        return action in self.config.capabilities

    def capability_vector(self) -> torch.Tensor:
        return self.config.capability_vector()

    def state_tensor(self, dim: int = 128) -> torch.Tensor:
        return self.state.to_tensor(dim)

    def reset(self):
        """Reset agent state for a new episode."""
        self.state = AgentState(agent_id=self.agent_id)

    def __repr__(self) -> str:
        return f"Agent(name={self.name}, role={self.role}, id={self.agent_id})"


class AgentTeam:
    """
    A team of agents assembled for a task.

    Provides batch-ready tensors for capabilities and states.
    """

    def __init__(self, agents: List[Agent], max_agents: int = 4):
        self.agents = agents
        self.max_agents = max_agents
        assert len(agents) <= max_agents, f"Too many agents: {len(agents)} > {max_agents}"

    @classmethod
    def from_archetypes(
        cls,
        archetype_names: List[str],
        max_agents: int = 4,
    ) -> "AgentTeam":
        """Create a team from archetype names."""
        agents = []
        for i, name in enumerate(archetype_names):
            if name not in AGENT_ARCHETYPES:
                raise ValueError(f"Unknown archetype: {name}. Choose from {list(AGENT_ARCHETYPES.keys())}")
            config = AGENT_ARCHETYPES[name]
            agents.append(Agent(config, agent_id=i))
        return cls(agents, max_agents)

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    def capabilities_tensor(self) -> torch.Tensor:
        """
        Returns:
            (max_agents, VOCAB_SIZE) — padded capability matrix.
        """
        caps = torch.zeros(self.max_agents, VOCAB_SIZE)
        for i, agent in enumerate(self.agents):
            caps[i] = agent.capability_vector()
        return caps

    def states_tensor(self, state_dim: int = 128) -> torch.Tensor:
        """
        Returns:
            (max_agents, state_dim) — padded state matrix.
        """
        states = torch.zeros(self.max_agents, state_dim)
        for i, agent in enumerate(self.agents):
            states[i] = agent.state_tensor(state_dim)
        return states

    def num_agents_tensor(self) -> torch.Tensor:
        """Returns scalar tensor of actual agent count."""
        return torch.tensor(self.num_agents, dtype=torch.long)

    def reset_all(self):
        """Reset all agents."""
        for agent in self.agents:
            agent.reset()

    def get_agent(self, idx: int) -> Agent:
        return self.agents[idx]

    def __len__(self) -> int:
        return self.num_agents

    def __repr__(self) -> str:
        agent_strs = [f"  {a}" for a in self.agents]
        return f"AgentTeam(n={self.num_agents}):\n" + "\n".join(agent_strs)
