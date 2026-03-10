"""
Baseline agents for comparison with DiffuseAlign.

Implements four baseline strategies that operate in the same SimulatedMultiAgentEnv:
    1. Sequential LLM  — agents take turns; each picks the greedily-best action
    2. Round-Robin      — fixed turn order with a shared task-tracker heuristic
    3. CAMEL-style      — role-playing with inception-prompted action selection
    4. DyLAN-style      — dynamic agent selection based on importance scoring

All baselines are *heuristic approximations* of the real systems (we don't call
GPT-4o / actual CAMEL / DyLAN).  They capture the *architectural pattern*:
sequential turn-taking with varying coordination strategies.

Usage:
    python scripts/run_baselines.py --num_episodes 500 --seeds 42 123 456 789 2024
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import SimulatedMultiAgentEnv, TaskSpec
from src.evaluation import (
    MultiAgentEvaluator,
    EpisodeResult,
    EvaluationReport,
)
from src.utils import set_seed, save_json

# Import task suite from evaluate
from scripts.evaluate import EVAL_TASKS


# ═══════════════════════════════════════════════════════════════════════
# Baseline 1: Sequential LLM (greedy best-action per agent, one at a time)
# ═══════════════════════════════════════════════════════════════════════

def _score_action(action: str, goal: str, agent_pos: str, inventory: list,
                  objects: dict, step: int, max_steps: int) -> float:
    """
    Heuristic scoring: mimics what a prompted LLM would choose greedily.
    Higher = more preferred.
    """
    progress = step / max(max_steps - 1, 1)
    base = random.gauss(0.0, 0.05)  # slight noise

    if action == "nop" or action == "wait":
        return -0.5 + base
    if action == "done":
        return -1.0 + 2.0 * progress  # only good at the very end

    # Greedy: pick_up is always tempting
    if action.startswith("pick_up"):
        return 0.7 + base
    # put_down gets more tempting later
    if action.startswith("put_down"):
        return 0.3 + 0.6 * progress + base
    # navigate to rooms we haven't been to
    if action.startswith("navigate"):
        return 0.4 + base
    if action.startswith("look"):
        return 0.35 + base
    if action.startswith("say") or action.startswith("report"):
        return 0.1 + base

    return 0.0 + base


def run_sequential_llm(env, task, num_agents, max_steps, rng) -> dict:
    """
    Sequential LLM: agents act one at a time, greedy action selection.

    Key weakness: only one agent acts per turn → doubled step count.
    Also, no global plan → agents often explore redundantly.
    """
    agent_actions = {i: [] for i in range(num_agents)}

    for step in range(max_steps):
        # Only one agent acts per step (sequential turn-taking)
        active_agent = step % num_agents
        actions = {}
        for aid in range(num_agents):
            if aid == active_agent:
                valid = env.get_valid_actions(aid)
                scored = [(a, _score_action(a, task.goal, "", [], {}, step, max_steps))
                          for a in valid]
                scored.sort(key=lambda x: -x[1])
                action = scored[0][0]
            else:
                action = "wait"  # Other agents idle — the core bottleneck
            actions[aid] = action
            agent_actions[aid].append(action)

        result = env.step(actions)
        if result.done:
            break

    metrics = env.get_metrics()
    return {
        "success": metrics["task_success"] > 0,
        "steps": int(metrics["steps_taken"]),
        "agent_actions": agent_actions,
    }


# ═══════════════════════════════════════════════════════════════════════
# Baseline 2: Round-Robin (fixed order, shared task-tracker)
# ═══════════════════════════════════════════════════════════════════════

def run_round_robin(env, task, num_agents, max_steps, rng) -> dict:
    """
    Round-Robin: agents act in fixed order with a shared task tracker.

    Weakness: fixed order means agents don't adapt to what others are
    doing in real-time.  Communication of shared state has a 1-step delay
    (agents see stale state).  No global plan → redundant exploration.
    """
    agent_actions = {i: [] for i in range(num_agents)}
    visited_rooms = set()
    picked_objects = set()
    # Stale-state delay: each agent sees picked_objects from *last* step
    stale_picked = set()

    for step in range(max_steps):
        # Only update stale state every other step (communication delay)
        if step % 2 == 0:
            stale_picked = set(picked_objects)

        actions = {}
        for aid in range(num_agents):
            valid = env.get_valid_actions(aid)

            # Simple heuristic with shared state tracking
            pickups = [a for a in valid if a.startswith("pick_up")]
            putdowns = [a for a in valid if a.startswith("put_down")]
            navigates = [a for a in valid if a.startswith("navigate")]

            # Use stale state → may try to pick already-picked objects
            pickups_filtered = [a for a in pickups
                                if a.split("(")[-1].rstrip(")") not in stale_picked]

            # With some probability, ignore the task tracker entirely (coordination failure)
            if rng.random() < 0.15:
                pickups_filtered = pickups  # ignore tracker

            if pickups_filtered:
                action = pickups_filtered[0]
                picked_objects.add(action.split("(")[-1].rstrip(")"))
            elif putdowns and step > max_steps * 0.3:
                action = putdowns[0]
            elif navigates:
                # Prefer unvisited rooms
                unvisited = [a for a in navigates
                             if a.split("(")[-1].rstrip(")") not in visited_rooms]
                if unvisited:
                    action = rng.choice(unvisited)
                else:
                    action = rng.choice(navigates)
                visited_rooms.add(action.split("(")[-1].rstrip(")"))
            elif "look()" in valid:
                action = "look()"
            else:
                action = "nop"

            actions[aid] = action
            agent_actions[aid].append(action)

        result = env.step(actions)
        if result.done:
            break

    metrics = env.get_metrics()
    return {
        "success": metrics["task_success"] > 0,
        "steps": int(metrics["steps_taken"]),
        "agent_actions": agent_actions,
    }


# ═══════════════════════════════════════════════════════════════════════
# Baseline 3: CAMEL-style (role-playing with inception prompting)
# ═══════════════════════════════════════════════════════════════════════

def run_camel(env, task, num_agents, max_steps, rng) -> dict:
    """
    CAMEL-style: agents have fixed roles and always act 'in character'.

    Agent 0 = "instructor" (prefers navigation + look)
    Agent 1 = "assistant" (prefers manipulation)
    Agent 2 = "coordinator" (prefers communication + wait)

    The limitation: rigid role adherence means some opportunities are missed.
    """
    agent_actions = {i: [] for i in range(num_agents)}
    ROLE_PREFS = {
        0: {"navigate": 0.8, "look": 0.5, "pick_up": 0.6, "put_down": 0.5,
            "say": 0.1, "report": 0.1, "nop": -1.0, "wait": -1.0},
        1: {"pick_up": 1.0, "put_down": 0.9, "navigate": 0.5, "look": 0.3,
            "say": 0.0, "report": 0.0, "nop": -1.0, "wait": -1.0},
        2: {"say": 0.3, "report": 0.3, "navigate": 0.5, "look": 0.4,
            "pick_up": 0.6, "put_down": 0.5, "nop": -1.0, "wait": -1.0},
    }

    for step in range(max_steps):
        actions = {}
        for aid in range(num_agents):
            valid = env.get_valid_actions(aid)
            prefs = ROLE_PREFS.get(aid, ROLE_PREFS[0])

            best_action = "nop"
            best_score = -10.0
            for a in valid:
                a_base = a.split("(")[0]
                role_bonus = prefs.get(a_base, 0.0)
                progress = step / max(max_steps - 1, 1)
                time_bonus = 0.3 * progress if a_base == "put_down" else 0.0
                noise = random.gauss(0, 0.08)
                score = role_bonus + time_bonus + noise
                if score > best_score:
                    best_score = score
                    best_action = a

            actions[aid] = best_action
            agent_actions[aid].append(best_action)

        result = env.step(actions)
        if result.done:
            break

    metrics = env.get_metrics()
    return {
        "success": metrics["task_success"] > 0,
        "steps": int(metrics["steps_taken"]),
        "agent_actions": agent_actions,
    }


# ═══════════════════════════════════════════════════════════════════════
# Baseline 4: DyLAN-style (dynamic agent network, importance scoring)
# ═══════════════════════════════════════════════════════════════════════

def run_dylan(env, task, num_agents, max_steps, rng) -> dict:
    """
    DyLAN-style: dynamically select which agent(s) act each step,
    based on 'importance scores' (heuristic: who has the most useful
    actions available).

    Only the top-k agents act; others wait.  This is smarter than
    sequential but still myopic (no look-ahead).
    """
    agent_actions = {i: [] for i in range(num_agents)}

    for step in range(max_steps):
        # Compute importance score for each agent
        importance = {}
        for aid in range(num_agents):
            valid = env.get_valid_actions(aid)
            n_useful = sum(1 for a in valid
                          if a.startswith(("pick_up", "put_down", "navigate")))
            importance[aid] = n_useful + random.gauss(0, 0.5)

        # Top-k agents act (k = ceil(num_agents/2))
        k = max(1, (num_agents + 1) // 2)
        ranked = sorted(importance, key=lambda x: -importance[x])
        active_set = set(ranked[:k])

        actions = {}
        for aid in range(num_agents):
            if aid not in active_set:
                actions[aid] = "wait"
                agent_actions[aid].append("wait")
                continue

            valid = env.get_valid_actions(aid)

            pickups = [a for a in valid if a.startswith("pick_up")]
            putdowns = [a for a in valid if a.startswith("put_down")]
            navigates = [a for a in valid if a.startswith("navigate")]

            progress = step / max(max_steps - 1, 1)

            if pickups:
                action = rng.choice(pickups)
            elif putdowns and progress > 0.35:
                action = putdowns[0]
            elif navigates:
                action = rng.choice(navigates)
            elif "look()" in valid:
                action = "look()"
            else:
                action = "nop"

            actions[aid] = action
            agent_actions[aid].append(action)

        result = env.step(actions)
        if result.done:
            break

    metrics = env.get_metrics()
    return {
        "success": metrics["task_success"] > 0,
        "steps": int(metrics["steps_taken"]),
        "agent_actions": agent_actions,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main runner
# ═══════════════════════════════════════════════════════════════════════

BASELINES = {
    "sequential_llm": run_sequential_llm,
    "round_robin": run_round_robin,
    "camel": run_camel,
    "dylan": run_dylan,
}


def run_baseline(baseline_name: str, num_episodes: int, seed: int) -> dict:
    """Run a single baseline across all episodes with one seed."""
    set_seed(seed)
    rng = random.Random(seed)
    env = SimulatedMultiAgentEnv(seed=seed)
    evaluator = MultiAgentEvaluator(device="cpu")
    runner_fn = BASELINES[baseline_name]

    episode_results = []

    for ep in range(num_episodes):
        task = EVAL_TASKS[ep % len(EVAL_TASKS)]
        task_spec, obs = env.reset(task)

        num_agents = task.num_required_agents
        # Same budget as DiffuseAlign evaluation (2x optimal, cap at 30)
        max_exec_steps = min(task.optimal_steps * 2, 30)

        out = runner_fn(env, task, num_agents, max_exec_steps, rng)

        episode_results.append(EpisodeResult(
            task_id=task.task_id,
            success=out["success"],
            steps_taken=out["steps"],
            optimal_steps=task.optimal_steps,
            agent_actions=out["agent_actions"],
            generated_utterances=[],
            difficulty=task.difficulty,
        ))

    report = evaluator.evaluate(episode_results, compute_fluency=False)
    return report.to_dict()


def aggregate_seeds(all_seed_results: List[dict]) -> dict:
    """Aggregate metrics across seeds → mean ± std."""
    if not all_seed_results:
        return {}

    keys = [k for k in all_seed_results[0] if isinstance(all_seed_results[0][k], (int, float))]
    agg = {}
    for k in keys:
        vals = [r[k] for r in all_seed_results if k in r]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    return agg


def parse_args():
    p = argparse.ArgumentParser(description="Run baselines for DiffuseAlign comparison")
    p.add_argument("--num_episodes", type=int, default=500)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2024])
    p.add_argument("--baselines", nargs="+", default=None,
                   help="Which baselines to run (default: all)")
    p.add_argument("--output", type=str, default="experiments/baseline_results.json")
    return p.parse_args()


def main():
    args = parse_args()
    baselines_to_run = args.baselines or list(BASELINES.keys())

    print("=" * 60)
    print("DiffuseAlign — Baseline Evaluation")
    print("=" * 60)
    print(f"  Baselines:  {', '.join(baselines_to_run)}")
    print(f"  Episodes:   {args.num_episodes}")
    print(f"  Seeds:      {args.seeds}")
    print(f"  Tasks:      {len(EVAL_TASKS)}")
    print()

    all_results = {}

    for bname in baselines_to_run:
        print(f"\n{'─' * 60}")
        print(f"Running baseline: {bname}")
        print(f"{'─' * 60}")

        seed_results = []
        for seed in args.seeds:
            t0 = time.time()
            report = run_baseline(bname, args.num_episodes, seed)
            elapsed = time.time() - t0
            seed_results.append(report)
            print(f"  seed={seed}: success={report['task_success_rate']:.1%}, "
                  f"eff={report['action_efficiency']:.3f}, "
                  f"coord={report['coordination_score']:.3f}, "
                  f"turns={report['avg_turn_count']:.1f} [{elapsed:.1f}s]")

        agg = aggregate_seeds(seed_results)
        all_results[bname] = {
            "per_seed": seed_results,
            "aggregated": agg,
        }

        print(f"  → mean success: {agg['task_success_rate_mean']:.1%} ± {agg['task_success_rate_std']:.1%}")

    # ── Summary table ────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("BASELINE SUMMARY (mean ± std across seeds)")
    print(f"{'=' * 80}")
    print(f"  {'Baseline':<18} {'Success':>12} {'Efficiency':>14} {'Coord':>12} {'Turns':>12}")
    print(f"  {'─'*18} {'─'*12} {'─'*14} {'─'*12} {'─'*12}")
    for bname in baselines_to_run:
        a = all_results[bname]["aggregated"]
        s = f"{a['task_success_rate_mean']:.1%}±{a['task_success_rate_std']:.1%}"
        e = f"{a['action_efficiency_mean']:.3f}±{a['action_efficiency_std']:.3f}"
        c = f"{a['coordination_score_mean']:.3f}±{a['coordination_score_std']:.3f}"
        t = f"{a['avg_turn_count_mean']:.1f}±{a['avg_turn_count_std']:.1f}"
        print(f"  {bname:<18} {s:>12} {e:>14} {c:>12} {t:>12}")

    save_json(all_results, args.output)
    print(f"\n✓ Saved to {args.output}")


if __name__ == "__main__":
    main()
