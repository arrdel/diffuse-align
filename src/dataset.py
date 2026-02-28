"""
Dataset — loading, processing, and batching multi-agent trajectories.

A trajectory consists of:
    - task_spec: TaskSpec with text description
    - agents: List of agent configs
    - plan: Sequence of (agent_id, action, args) tuples
    - dialogue: Corresponding natural language utterances
    - outcome: {success, failure, partial}

Trajectories are collected by running LLM agents (GPT-4o) in environments,
then stored as JSON. This module loads them and converts to tensors for
diffusion model training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from .agents import ACTION_VOCAB, VOCAB_SIZE


# ─── Trajectory Data Structures ───────────────────────────────────────────────

class TrajectoryStep:
    """A single step in a multi-agent trajectory."""

    def __init__(
        self,
        step_idx: int,
        agent_id: int,
        action: str,
        args: str = "",
        utterance: str = "",
        observation: str = "",
    ):
        self.step_idx = step_idx
        self.agent_id = agent_id
        self.action = action
        self.args = args
        self.utterance = utterance
        self.observation = observation

    def action_id(self) -> int:
        """Map action string to vocabulary ID."""
        return ACTION_VOCAB.get(self.action, ACTION_VOCAB.get("nop", 100))

    def to_dict(self) -> dict:
        return {
            "step": self.step_idx,
            "agent_id": self.agent_id,
            "action": self.action,
            "args": self.args,
            "utterance": self.utterance,
            "observation": self.observation,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrajectoryStep":
        return cls(
            step_idx=d["step"],
            agent_id=d["agent_id"],
            action=d["action"],
            args=d.get("args", ""),
            utterance=d.get("utterance", ""),
            observation=d.get("observation", ""),
        )


class Trajectory:
    """A complete multi-agent trajectory for one task episode."""

    def __init__(
        self,
        task_id: str,
        task_description: str,
        num_agents: int,
        agent_roles: List[str],
        steps: List[TrajectoryStep],
        success: bool,
        metadata: Optional[Dict] = None,
    ):
        self.task_id = task_id
        self.task_description = task_description
        self.num_agents = num_agents
        self.agent_roles = agent_roles
        self.steps = steps
        self.success = success
        self.metadata = metadata or {}

    @property
    def plan_length(self) -> int:
        """Number of steps in the trajectory."""
        return len(self.steps)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "task_description": self.task_description,
            "num_agents": self.num_agents,
            "agent_roles": self.agent_roles,
            "steps": [s.to_dict() for s in self.steps],
            "success": self.success,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        return cls(
            task_id=d["task_id"],
            task_description=d["task_description"],
            num_agents=d["num_agents"],
            agent_roles=d["agent_roles"],
            steps=[TrajectoryStep.from_dict(s) for s in d["steps"]],
            success=d["success"],
            metadata=d.get("metadata"),
        )

    def to_plan_tensor(
        self,
        max_agents: int = 4,
        max_steps: int = 32,
        plan_dim: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert trajectory to tensored plan representation.

        Returns:
            plan_actions: (max_agents, max_steps) — action IDs per agent per step.
            validity_mask: (max_agents, max_steps) — 1 where valid, 0 for padding.
            agent_step_map: (max_agents, max_steps) — step index in original traj.
        """
        plan_actions = torch.full((max_agents, max_steps), ACTION_VOCAB["nop"], dtype=torch.long)
        validity_mask = torch.zeros(max_agents, max_steps)
        agent_step_map = torch.zeros(max_agents, max_steps, dtype=torch.long)

        # Track per-agent step counters
        agent_step_counters = [0] * max_agents

        for step in self.steps:
            a = step.agent_id
            if a >= max_agents:
                continue
            s = agent_step_counters[a]
            if s >= max_steps:
                continue

            plan_actions[a, s] = step.action_id()
            validity_mask[a, s] = 1.0
            agent_step_map[a, s] = step.step_idx
            agent_step_counters[a] += 1

        return plan_actions, validity_mask, agent_step_map


# ─── Dataset ──────────────────────────────────────────────────────────────────

class MultiAgentTrajectoryDataset(Dataset):
    """
    PyTorch Dataset for multi-agent trajectories.

    Loads trajectories from JSON files and converts them to tensor batches
    suitable for diffusion model training.
    """

    def __init__(
        self,
        data_dir: str,
        max_agents: int = 4,
        max_steps: int = 32,
        plan_dim: int = 512,
        action_embed_dim: int = 256,
        success_only: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.max_agents = max_agents
        self.max_steps = max_steps
        self.plan_dim = plan_dim
        self.action_embed_dim = action_embed_dim

        # Action embedding layer (shared, learned during training)
        self.action_embedding = torch.nn.Embedding(VOCAB_SIZE, action_embed_dim)

        # Load trajectories
        self.trajectories = self._load_trajectories(success_only)

    def _load_trajectories(self, success_only: bool) -> List[Trajectory]:
        """Load all trajectory JSON files from data directory."""
        trajectories = []

        if not self.data_dir.exists():
            # Return empty dataset with synthetic placeholder
            print(f"Warning: {self.data_dir} not found. Using synthetic data.")
            return self._generate_synthetic(100)

        for json_file in sorted(self.data_dir.glob("*.json")):
            with open(json_file) as f:
                data = json.load(f)

            if isinstance(data, list):
                for item in data:
                    traj = Trajectory.from_dict(item)
                    if not success_only or traj.success:
                        trajectories.append(traj)
            else:
                traj = Trajectory.from_dict(data)
                if not success_only or traj.success:
                    trajectories.append(traj)

        if not trajectories:
            print(f"Warning: No trajectories found in {self.data_dir}. Using synthetic data.")
            return self._generate_synthetic(100)

        return trajectories

    def _generate_synthetic(self, n: int = 100) -> List[Trajectory]:
        """Generate synthetic trajectories for development."""
        import random
        random.seed(42)

        trajectories = []
        action_names = list(ACTION_VOCAB.keys())

        tasks = [
            "Find the red mug and put it on the kitchen counter",
            "Clean the living room and organize books on the shelf",
            "Cook pasta in the kitchen — boil water, add pasta, drain",
            "Search the web for flight prices and book the cheapest one",
            "Navigate to the bedroom, find the phone charger, bring it to the office",
        ]

        for i in range(n):
            num_agents = random.choice([2, 3])
            roles = random.sample(["navigator", "manipulator", "researcher", "coordinator"], num_agents)
            task = random.choice(tasks)
            num_steps = random.randint(4, 20)

            steps = []
            for s in range(num_steps):
                agent_id = random.randint(0, num_agents - 1)
                action = random.choice(action_names[:20])  # Stick to common actions
                steps.append(TrajectoryStep(
                    step_idx=s,
                    agent_id=agent_id,
                    action=action,
                    utterance=f"Agent {agent_id} performs {action}",
                ))

            trajectories.append(Trajectory(
                task_id=f"synthetic_{i:04d}",
                task_description=task,
                num_agents=num_agents,
                agent_roles=roles,
                steps=steps,
                success=random.random() > 0.3,
            ))

        return trajectories

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj = self.trajectories[idx]

        # Convert to plan tensor
        plan_actions, validity_mask, _ = traj.to_plan_tensor(
            max_agents=self.max_agents,
            max_steps=self.max_steps,
        )

        # Embed actions → continuous plan representation
        # .detach() so the tensor is a leaf — required for multi-worker DataLoader
        plan_embedded = self.action_embedding(plan_actions).detach()  # (agents, steps, embed_dim)

        # Pad to plan_dim if needed
        if self.action_embed_dim < self.plan_dim:
            padding = torch.zeros(
                self.max_agents, self.max_steps,
                self.plan_dim - self.action_embed_dim,
            )
            plan_embedded = torch.cat([plan_embedded, padding], dim=-1)

        return {
            "plan": plan_embedded,                              # (agents, steps, plan_dim)
            "plan_actions": plan_actions,                       # (agents, steps) action IDs
            "validity_mask": validity_mask,                     # (agents, steps) binary
            "task_description": traj.task_description,          # string
            "num_agents": torch.tensor(traj.num_agents),        # scalar
            "plan_length": torch.tensor(traj.plan_length),      # scalar
            "success": torch.tensor(float(traj.success)),       # scalar
        }


def collate_trajectories(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for trajectory batches."""
    return {
        "plan": torch.stack([b["plan"] for b in batch]),
        "plan_actions": torch.stack([b["plan_actions"] for b in batch]),
        "validity_mask": torch.stack([b["validity_mask"] for b in batch]),
        "task_descriptions": [b["task_description"] for b in batch],
        "num_agents": torch.stack([b["num_agents"] for b in batch]),
        "plan_lengths": torch.stack([b["plan_length"] for b in batch]),
        "success": torch.stack([b["success"] for b in batch]),
    }


def get_dataloader(
    data_dir: str,
    batch_size: int = 64,
    max_agents: int = 4,
    max_steps: int = 32,
    plan_dim: int = 512,
    shuffle: bool = True,
    num_workers: int = 4,
) -> DataLoader:
    """Create a DataLoader for multi-agent trajectories."""
    dataset = MultiAgentTrajectoryDataset(
        data_dir=data_dir,
        max_agents=max_agents,
        max_steps=max_steps,
        plan_dim=plan_dim,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_trajectories,
        pin_memory=True,
    )
