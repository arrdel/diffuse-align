"""
Multi-Agent Environment Wrapper — provides a unified interface for
different task environments (ALFWorld, WebArena, Cooking).

The environment exposes:
    - A task specification (natural language)
    - Agent observations (what each agent can see)
    - An action execution interface
    - Success/failure signals

This is used both for collecting training trajectories (via LLM rollouts)
and for evaluating generated plans.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class Observation:
    """What a single agent observes at a timestep."""
    agent_id: int
    text: str
    structured: Optional[Dict[str, Any]] = None
    reward: float = 0.0
    done: bool = False


@dataclass
class TaskSpec:
    """A multi-agent task specification."""
    task_id: str
    description: str
    goal: str
    num_required_agents: int
    difficulty: str = "moderate"  # "simple", "moderate", "complex"
    optimal_steps: int = 10
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class StepResult:
    """Result of executing one step in the environment."""
    observations: Dict[int, Observation]  # agent_id → observation
    rewards: Dict[int, float]
    done: bool
    info: Dict[str, Any]


class MultiAgentEnvironment(ABC):
    """Abstract base class for multi-agent task environments."""

    @abstractmethod
    def reset(self, task: Optional[TaskSpec] = None) -> Tuple[TaskSpec, Dict[int, Observation]]:
        """Reset environment, optionally with a specific task."""
        ...

    @abstractmethod
    def step(self, actions: Dict[int, str]) -> StepResult:
        """
        Execute actions for each agent.

        Args:
            actions: Dict mapping agent_id → action string.
        Returns:
            StepResult with observations, rewards, done flag, info.
        """
        ...

    @abstractmethod
    def get_valid_actions(self, agent_id: int) -> List[str]:
        """Get valid actions for an agent in current state."""
        ...

    @abstractmethod
    def is_success(self) -> bool:
        """Check if the task was completed successfully."""
        ...

    @abstractmethod
    def get_metrics(self) -> Dict[str, float]:
        """Get evaluation metrics for the current episode."""
        ...


class SimulatedMultiAgentEnv(MultiAgentEnvironment):
    """
    A simulated multi-agent environment for development and testing.

    Simulates household tasks where agents must coordinate to find and
    manipulate objects. Used when real environment backends (ALFWorld, etc.)
    are not available.
    """

    def __init__(
        self,
        num_rooms: int = 5,
        num_objects: int = 10,
        max_steps: int = 50,
        seed: int = 42,
    ):
        self.num_rooms = num_rooms
        self.num_objects = num_objects
        self.max_steps = max_steps

        self._rng = torch.Generator().manual_seed(seed)
        self._step_count = 0
        self._done = False
        self._success = False
        self._task = None

        # Simulated world state
        self._rooms = [f"room_{i}" for i in range(num_rooms)]
        self._objects = {}
        self._agent_positions = {}
        self._agent_inventories = {}
        self._goal_achieved = {}

    def reset(self, task: Optional[TaskSpec] = None) -> Tuple[TaskSpec, Dict[int, Observation]]:
        self._step_count = 0
        self._done = False
        self._success = False

        if task is None:
            task = TaskSpec(
                task_id="sim_001",
                description="Find the red mug in the house and place it on the kitchen counter.",
                goal="red_mug on kitchen_counter",
                num_required_agents=2,
                difficulty="moderate",
                optimal_steps=8,
            )

        self._task = task

        # Initialize agents in random rooms
        num_agents = task.num_required_agents
        for i in range(num_agents):
            idx = torch.randint(0, self.num_rooms, (1,), generator=self._rng).item()
            self._agent_positions[i] = self._rooms[idx]
            self._agent_inventories[i] = []

        # Place target object
        target_room_idx = torch.randint(0, self.num_rooms, (1,), generator=self._rng).item()
        self._objects["red_mug"] = {"location": self._rooms[target_room_idx], "held_by": None}

        # Initial observations
        observations = {}
        for agent_id in range(num_agents):
            obs_text = f"You are in {self._agent_positions[agent_id]}. "
            obs_text += f"Task: {task.description}"
            observations[agent_id] = Observation(
                agent_id=agent_id,
                text=obs_text,
            )

        return task, observations

    def step(self, actions: Dict[int, str]) -> StepResult:
        self._step_count += 1
        observations = {}
        rewards = {}

        for agent_id, action in actions.items():
            # Lazily initialize agents not created during reset
            if agent_id not in self._agent_positions:
                idx = torch.randint(0, self.num_rooms, (1,), generator=self._rng).item()
                self._agent_positions[agent_id] = self._rooms[idx]
                self._agent_inventories[agent_id] = []

            obs_text = f"Step {self._step_count}: "
            reward = -0.01  # Small step penalty

            if action.startswith("navigate"):
                target = action.split("(")[-1].rstrip(")")
                if target in self._rooms:
                    self._agent_positions[agent_id] = target
                    obs_text += f"Moved to {target}."
                else:
                    obs_text += f"Cannot navigate to {target}."
                    reward -= 0.1

            elif action.startswith("look"):
                pos = self._agent_positions[agent_id]
                visible_objects = [
                    name for name, obj in self._objects.items()
                    if obj["location"] == pos and obj["held_by"] is None
                ]
                if visible_objects:
                    obs_text += f"You see: {', '.join(visible_objects)}"
                else:
                    obs_text += "Nothing notable here."

            elif action.startswith("pick_up"):
                target = action.split("(")[-1].rstrip(")")
                pos = self._agent_positions[agent_id]
                if target in self._objects and self._objects[target]["location"] == pos:
                    self._objects[target]["held_by"] = agent_id
                    self._objects[target]["location"] = None
                    self._agent_inventories[agent_id].append(target)
                    obs_text += f"Picked up {target}."
                    reward += 0.1
                else:
                    obs_text += f"Cannot pick up {target} here."

            elif action.startswith("put_down"):
                target = action.split("(")[-1].rstrip(")")
                if target in self._agent_inventories.get(agent_id, []):
                    pos = self._agent_positions[agent_id]
                    self._objects[target]["held_by"] = None
                    self._objects[target]["location"] = pos
                    self._agent_inventories[agent_id].remove(target)
                    obs_text += f"Put down {target} at {pos}."

                    # Check goal
                    if self._task and self._task.goal == f"{target} on {pos}":
                        self._success = True
                        self._done = True
                        reward += 1.0
                else:
                    obs_text += f"You don't have {target}."

            elif action.startswith("say") or action.startswith("report"):
                message = action.split("(", 1)[-1].rstrip(")")
                obs_text += f"Said: '{message}'"

            elif action == "nop" or action == "wait":
                obs_text += "Waiting."

            elif action == "done":
                self._done = True
                obs_text += "Task declared complete."
            else:
                obs_text += f"Unknown action: {action}"
                reward -= 0.05

            observations[agent_id] = Observation(
                agent_id=agent_id,
                text=obs_text,
                reward=reward,
                done=self._done,
            )
            rewards[agent_id] = reward

        if self._step_count >= self.max_steps:
            self._done = True

        return StepResult(
            observations=observations,
            rewards=rewards,
            done=self._done,
            info={
                "step": self._step_count,
                "success": self._success,
                "agent_positions": dict(self._agent_positions),
            },
        )

    def get_valid_actions(self, agent_id: int) -> List[str]:
        pos = self._agent_positions.get(agent_id, self._rooms[0])
        actions = ["nop", "wait", "done"]

        # Navigation
        for room in self._rooms:
            if room != pos:
                actions.append(f"navigate({room})")

        # Look
        actions.append("look()")

        # Pick up visible objects
        for name, obj in self._objects.items():
            if obj["location"] == pos and obj["held_by"] is None:
                actions.append(f"pick_up({name})")

        # Put down held objects
        for obj_name in self._agent_inventories.get(agent_id, []):
            actions.append(f"put_down({obj_name})")

        # Communication
        actions.extend(["say(status)", "report(findings)"])

        return actions

    def is_success(self) -> bool:
        return self._success

    def get_metrics(self) -> Dict[str, float]:
        optimal = self._task.optimal_steps if self._task else 10
        return {
            "task_success": float(self._success),
            "steps_taken": self._step_count,
            "optimal_steps": optimal,
            "action_efficiency": optimal / max(self._step_count, 1),
            "step_ratio": self._step_count / max(optimal, 1),
        }
