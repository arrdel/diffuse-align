"""
Trajectory generation script — collects multi-agent dialogue trajectories
by running LLM agents in task environments.

These trajectories form the training data for the DiffuseAlign diffusion model.

Usage:
    python scripts/generate_trajectories.py \
        --env simulated \
        --num_episodes 1000 \
        --num_agents 2 \
        --output data/simulated_multi/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import SimulatedMultiAgentEnv, TaskSpec
from src.agents import AgentTeam, AGENT_ARCHETYPES
from src.dataset import Trajectory, TrajectoryStep
from src.utils import set_seed, save_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="simulated")
    parser.add_argument("--num_episodes", type=int, default=1000)
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--output", type=str, default="data/simulated_multi")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def collect_random_trajectories(
    env: SimulatedMultiAgentEnv,
    team: AgentTeam,
    num_episodes: int,
    max_steps: int,
) -> list:
    """Collect trajectories using random actions (baseline data)."""
    import random

    trajectories = []

    tasks = [
        TaskSpec("t001", "Find the red mug and place it on the kitchen counter.",
                 "red_mug on room_2", 2, "simple", 5),
        TaskSpec("t002", "Find the blue book and bring it to the study.",
                 "blue_book on room_3", 2, "simple", 6),
        TaskSpec("t003", "Clean the kitchen and organize the pantry.",
                 "kitchen_clean", 2, "moderate", 10),
        TaskSpec("t004", "Cook dinner: gather ingredients and prepare the meal.",
                 "dinner_ready", 2, "complex", 18),
        TaskSpec("t005", "Find all scattered toys and put them in the toy box.",
                 "toys_collected", 2, "moderate", 12),
    ]

    for ep in tqdm(range(num_episodes), desc="Collecting trajectories"):
        task = tasks[ep % len(tasks)]
        task_spec, _ = env.reset(task)
        team.reset_all()

        steps = []
        for t in range(max_steps):
            actions = {}
            for agent_id in range(team.num_agents):
                valid = env.get_valid_actions(agent_id)
                action = random.choice(valid) if valid else "nop"
                actions[agent_id] = action

                steps.append(TrajectoryStep(
                    step_idx=t,
                    agent_id=agent_id,
                    action=action.split("(")[0],
                    args=action.split("(")[-1].rstrip(")") if "(" in action else "",
                    utterance=f"Agent {agent_id}: {action}",
                ))

            result = env.step(actions)
            if result.done:
                break

        metrics = env.get_metrics()

        traj = Trajectory(
            task_id=f"{task.task_id}_ep{ep:04d}",
            task_description=task.description,
            num_agents=team.num_agents,
            agent_roles=[a.role for a in team.agents],
            steps=steps,
            success=metrics["task_success"] > 0,
            metadata={
                "optimal_steps": task.optimal_steps,
                "actual_steps": int(metrics["steps_taken"]),
                "difficulty": task.difficulty,
            },
        )
        trajectories.append(traj)

    return trajectories


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup
    env = SimulatedMultiAgentEnv(max_steps=args.max_steps, seed=args.seed)
    archetypes = list(AGENT_ARCHETYPES.keys())[:args.num_agents]
    team = AgentTeam.from_archetypes(archetypes)

    print(f"Environment: {args.env}")
    print(f"Team: {team}")
    print(f"Collecting {args.num_episodes} trajectories...")

    # Collect
    trajectories = collect_random_trajectories(
        env, team, args.num_episodes, args.max_steps
    )

    # Save
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as batched JSON files (1000 per file)
    batch_size = 1000
    for i in range(0, len(trajectories), batch_size):
        batch = trajectories[i:i + batch_size]
        data = [t.to_dict() for t in batch]
        save_json(data, str(output_dir / f"batch_{i // batch_size:04d}.json"))

    # Stats
    success_count = sum(1 for t in trajectories if t.success)
    avg_length = sum(t.plan_length for t in trajectories) / len(trajectories)

    print(f"\nCollection complete!")
    print(f"  Total trajectories: {len(trajectories)}")
    print(f"  Success rate: {success_count / len(trajectories):.1%}")
    print(f"  Avg plan length: {avg_length:.1f}")
    print(f"  Saved to: {output_dir}")


if __name__ == "__main__":
    main()
