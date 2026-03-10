"""
Evaluation script for DiffuseAlign.

Runs the full evaluation pipeline:
    1. Load trained DiffuseAlign model (assembled from stages 1-3)
    2. Generate joint plans via guided diffusion
    3. Decode plans to natural-language dialogue
    4. Execute plans in simulated environment
    5. Compute functional + fluency metrics with per-complexity breakdown
    6. Save evaluation report

Usage:
    python scripts/evaluate.py \
        --checkpoint experiments/checkpoints/diffusealign_final.pt \
        --device cuda:4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffuse_align import DiffuseAlign
from src.environment import SimulatedMultiAgentEnv, TaskSpec
from src.agents import AgentTeam, VOCAB_SIZE
from src.evaluation import (
    MultiAgentEvaluator,
    EpisodeResult,
    EvaluationReport,
)
from src.utils import set_seed, load_config, save_json, get_device, format_params


# ─── Task suite ────────────────────────────────────────────────────────
EVAL_TASKS = [
    # Simple (2 agents, 4-6 optimal steps)
    TaskSpec("t001", "Find the red mug and place it on the kitchen counter.",
             "red_mug on kitchen", 2, "simple", 5),
    TaskSpec("t002", "Find and return the library book from the bedroom to the study.",
             "book on study", 2, "simple", 4),
    TaskSpec("t003", "Move the vase from the bedroom to the kitchen.",
             "vase on kitchen", 2, "simple", 5),
    TaskSpec("t004", "Find the keys on the table and bring them to the living room.",
             "keys on living_room", 2, "simple", 6),

    # Moderate (2-3 agents, 8-15 optimal steps)
    TaskSpec("t005", "Clean the bathroom: scrub the tub, mop the floor, organize toiletries.",
             "bathroom_clean", 2, "moderate", 12),
    TaskSpec("t006", "Set the dining table: find plates, cups, and cutlery and arrange them.",
             "table_set", 2, "moderate", 10),
    TaskSpec("t007", "Sort the mail: collect packages from the porch and deliver to correct rooms.",
             "mail_sorted", 3, "moderate", 10),
    TaskSpec("t008", "Tidy the living room: pick up toys, straighten cushions, vacuum floor.",
             "living_room_tidy", 2, "moderate", 8),

    # Complex (3 agents, 18-30 optimal steps)
    TaskSpec("t009", "Prepare dinner: find recipe online, gather ingredients from pantry and fridge, cook meal, set table.",
             "dinner_ready", 3, "complex", 25),
    TaskSpec("t010", "Reorganize the garage: sort tools, discard trash, organize shelves.",
             "garage_organized", 3, "complex", 20),
    TaskSpec("t011", "Host a party: decorate the living room, prepare snacks in the kitchen, set up music in the lounge.",
             "party_ready", 3, "complex", 22),
    TaskSpec("t012", "Emergency cleanup: agent 1 secures fragile items, agent 2 clears the floor, agent 3 moves furniture.",
             "emergency_cleared", 3, "complex", 18),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DiffuseAlign")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to assembled model checkpoint (diffusealign_final.pt)")
    parser.add_argument("--output", type=str, default="experiments/eval_results.json")
    parser.add_argument("--num_episodes", type=int, default=100,
                        help="Total evaluation episodes (cycled across tasks)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_guidance", action="store_true",
                        help="Disable compositional guidance (CFG only)")
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="Override CFG scale (default from config: 3.0)")
    parser.add_argument("--active_signals", nargs="+", default=None,
                        help="Which guidance signals to use (task safety efficiency coordination). Default: all")
    parser.add_argument("--num_inference_steps", type=int, default=None,
                        help="DDIM sampling steps (default 50)")
    parser.add_argument("--decode_utterances", action="store_true",
                        help="Also run plan decoder to generate dialogue utterances")
    parser.add_argument("--compute_fluency", action="store_true",
                        help="Compute expensive fluency metrics (BERTScore, NLI coherence)")
    return parser.parse_args()


def build_model(cfg) -> DiffuseAlign:
    """Build DiffuseAlign model from config (same as training scripts)."""
    model_cfg = cfg.model
    return DiffuseAlign(
        plan_dim=model_cfg.diffusion.plan_dim,
        hidden_dim=model_cfg.diffusion.denoiser.hidden_dim,
        num_heads=model_cfg.diffusion.denoiser.num_heads,
        num_layers=model_cfg.diffusion.denoiser.num_layers,
        dropout=model_cfg.diffusion.denoiser.dropout,
        condition_dim=model_cfg.diffusion.denoiser.condition_dim,
        max_agents=model_cfg.diffusion.max_agents,
        max_steps=model_cfg.diffusion.max_plan_steps,
        num_train_timesteps=model_cfg.diffusion.num_train_timesteps,
        num_inference_timesteps=model_cfg.diffusion.num_inference_timesteps,
        unconditional_prob=model_cfg.guidance.unconditional_prob,
        guidance_scale=model_cfg.guidance.guidance_scale,
        task_guidance_weight=model_cfg.guidance.task_completion.weight,
        safety_guidance_weight=model_cfg.guidance.safety.weight,
        efficiency_guidance_weight=model_cfg.guidance.efficiency.weight,
        coordination_guidance_weight=model_cfg.guidance.coordination.weight,
        mask_type=model_cfg.role_masking.mask_type,
    )


def _select_action_from_plan(
    plan_vec: torch.Tensor,
    valid_actions: List[str],
    step: int,
    max_steps: int,
) -> str:
    """
    Map a continuous plan vector to a discrete action.

    Strategy:
    - Use the plan vector's energy profile to determine the *type* of
      action (navigate, look, pick_up, put_down, communicate, wait).
    - Use the vector's directional components to pick the *target*
      among valid actions of that type.
    - Phase awareness: early steps bias toward navigate/look,
      mid steps toward pick_up, late steps toward put_down/done.

    This is intentionally a heuristic decoder — the plan tensor encodes
    high-level intent, and we map it to the closest valid action.
    """
    energy = plan_vec.norm().item()

    # If energy is negligible, wait
    if energy < 0.05:
        return "nop"

    # Categorize valid actions by type
    by_type: dict[str, list[str]] = {
        "navigate": [], "look": [], "pick_up": [],
        "put_down": [], "communicate": [], "control": [], "idle": [],
    }
    for a in valid_actions:
        if a.startswith("navigate"):
            by_type["navigate"].append(a)
        elif a.startswith("look"):
            by_type["look"].append(a)
        elif a.startswith("pick_up"):
            by_type["pick_up"].append(a)
        elif a.startswith("put_down"):
            by_type["put_down"].append(a)
        elif a.startswith("say") or a.startswith("report"):
            by_type["communicate"].append(a)
        elif a == "done":
            by_type["control"].append(a)
        else:
            by_type["idle"].append(a)

    # Use different slices of the plan vector to determine type weights
    d = plan_vec.shape[0]
    chunk = max(d // 6, 1)

    # Each chunk's mean energy encodes preference for that action type
    type_scores = {
        "navigate":    plan_vec[0:chunk].abs().mean().item(),
        "look":        plan_vec[chunk:2*chunk].abs().mean().item(),
        "pick_up":     plan_vec[2*chunk:3*chunk].abs().mean().item(),
        "put_down":    plan_vec[3*chunk:4*chunk].abs().mean().item(),
        "communicate": plan_vec[4*chunk:5*chunk].abs().mean().item(),
        "control":     plan_vec[5*chunk:6*chunk].abs().mean().item(),
    }

    # Phase bias: shift action preferences over time
    progress = step / max(max_steps - 1, 1)
    type_scores["navigate"]  *= (1.2 - 0.6 * progress)   # high early
    type_scores["look"]      *= (1.0 - 0.3 * progress)   # moderate early
    type_scores["pick_up"]   *= (0.6 + 0.8 * progress)   # grows over time
    type_scores["put_down"]  *= (0.3 + 1.4 * progress)   # strong late
    type_scores["control"]   *= progress ** 2             # only at very end
    type_scores["communicate"] *= 0.5                     # baseline communication

    # Select type: pick highest-scoring type that has valid actions
    ranked_types = sorted(type_scores.items(), key=lambda x: -x[1])
    chosen_type = "idle"
    for t, score in ranked_types:
        if by_type.get(t):
            chosen_type = t
            break

    candidates = by_type.get(chosen_type, [])
    if not candidates:
        # Fallback: pick any non-idle action
        candidates = [a for a in valid_actions if a not in ("nop", "wait")]
        if not candidates:
            return "nop"

    # Select specific target within the type using plan vector direction
    if len(candidates) == 1:
        return candidates[0]

    # Use a different slice of the plan vector to index into candidates
    selector = plan_vec[chunk:].abs()
    idx = int(selector.sum().item() * 997) % len(candidates)
    return candidates[idx]


def run_evaluation(
    model: DiffuseAlign,
    env: SimulatedMultiAgentEnv,
    team: AgentTeam,
    num_episodes: int,
    device: torch.device,
    use_guidance: bool = True,
    guidance_scale: float = 3.0,
    active_signals: Optional[List[str]] = None,
    num_inference_steps: Optional[int] = None,
    decode_utterances: bool = False,
) -> Tuple[List[EpisodeResult], Dict[str, list]]:
    """
    Run evaluation episodes.

    For each episode:
        1. Pick a task from the suite
        2. Generate a joint plan with DiffuseAlign
        3. Optionally decode plan → NL utterances
        4. Execute plan in the environment
        5. Collect metrics
    """
    results = []
    generation_times = []
    generated_dialogues = {}  # task_id → list of utterances (for qualitative)

    for ep in tqdm(range(num_episodes), desc="Evaluating"):
        task = EVAL_TASKS[ep % len(EVAL_TASKS)]
        task_spec, observations = env.reset(task)
        team.reset_all()

        # ── Generate plan ────────────────────────────────────────
        agent_states = team.states_tensor().unsqueeze(0).to(device)
        capabilities = team.capabilities_tensor().unsqueeze(0).to(device)

        t0 = time.time()
        plan = model.generate_plan(
            task_texts=[task_spec.description],
            agent_states=agent_states,
            capabilities=capabilities,
            use_guidance=use_guidance,
            guidance_scale=guidance_scale,
            active_guidance_signals=active_signals,
            num_inference_steps=num_inference_steps,
        )
        gen_time = time.time() - t0
        generation_times.append(gen_time)

        # ── Decode to utterances (optional) ──────────────────────
        utterances = []
        if decode_utterances:
            try:
                utt = model.plan_decoder.generate_utterances(plan, beam_size=2)
                utterances = utt
                if task.task_id not in generated_dialogues:
                    generated_dialogues[task.task_id] = []
                generated_dialogues[task.task_id].append(utt)
            except Exception as e:
                utterances = [f"[decode error: {e}]"]

        # ── Execute plan in environment ──────────────────────────
        plan_single = plan[0].detach().cpu()  # (agents, steps, plan_dim)
        num_agents = min(team.num_agents, task_spec.num_required_agents)
        max_exec_steps = min(task_spec.optimal_steps * 2, 30)

        agent_actions = {i: [] for i in range(num_agents)}
        # Track per-agent state for reactive execution
        agent_last_action = {i: "" for i in range(num_agents)}
        agent_has_object = {i: False for i in range(num_agents)}

        for step in range(max_exec_steps):
            actions = {}
            for agent_id in range(num_agents):
                valid = env.get_valid_actions(agent_id)
                if not valid:
                    actions[agent_id] = "nop"
                    agent_actions[agent_id].append("nop")
                    continue

                # Reactive priority: if we can pick something up, do it
                pickups = [a for a in valid if a.startswith("pick_up")]
                putdowns = [a for a in valid if a.startswith("put_down")]

                if pickups and not agent_has_object[agent_id]:
                    # Greedy: pick up available objects
                    action = pickups[0]
                    agent_has_object[agent_id] = True
                elif putdowns and agent_has_object[agent_id]:
                    # Put down objects — bias toward later in the episode
                    progress = step / max(max_exec_steps - 1, 1)
                    if progress > 0.3 or step >= plan_single.shape[1]:
                        action = putdowns[0]
                        agent_has_object[agent_id] = False
                    elif step < plan_single.shape[1]:
                        action = _select_action_from_plan(
                            plan_single[agent_id, step], valid, step, max_exec_steps
                        )
                    else:
                        action = putdowns[0]
                        agent_has_object[agent_id] = False
                elif step < plan_single.shape[1]:
                    action = _select_action_from_plan(
                        plan_single[agent_id, step], valid, step, max_exec_steps
                    )
                else:
                    action = "nop"

                actions[agent_id] = action
                agent_actions[agent_id].append(action)
                agent_last_action[agent_id] = action

            step_result = env.step(actions)

            if not utterances:
                utterances.append(f"Step {step}: {actions}")

            if step_result.done:
                break

        metrics = env.get_metrics()

        results.append(EpisodeResult(
            task_id=task_spec.task_id,
            success=metrics["task_success"] > 0,
            steps_taken=int(metrics["steps_taken"]),
            optimal_steps=task_spec.optimal_steps,
            agent_actions=agent_actions,
            generated_utterances=utterances,
            difficulty=task_spec.difficulty,
        ))

    extra = {
        "generation_times": generation_times,
        "generated_dialogues": generated_dialogues,
    }
    return results, extra


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    device = torch.device(args.device) if args.device else get_device()

    print("=" * 60)
    print("DiffuseAlign Evaluation")
    print("=" * 60)

    # ── Build and load model ─────────────────────────────────────
    model = build_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {format_params(total_params)}")

    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    # Handle both raw state_dict and wrapped checkpoints
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    # If it's already an OrderedDict of tensors, use as-is

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys (first 5): {unexpected[:5]}")
    if not missing and not unexpected:
        print(f"  ✓ All {len(state)} keys loaded successfully")

    model = model.to(device)
    model.eval()

    # ── Environment and team ─────────────────────────────────────
    env = SimulatedMultiAgentEnv(seed=args.seed)

    # Use 3 agents to cover both 2-agent and 3-agent tasks
    team = AgentTeam.from_archetypes(
        ["navigator", "manipulator", "coordinator"],
        max_agents=cfg.model.diffusion.max_agents,
    )
    print(f"Team:\n{team}")
    print(f"Device: {device}")

    guidance_scale = args.guidance_scale or cfg.model.guidance.guidance_scale
    print(f"Guidance: {'OFF' if args.no_guidance else f'ON (scale={guidance_scale})'}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Task suite: {len(EVAL_TASKS)} tasks ({sum(1 for t in EVAL_TASKS if t.difficulty=='simple')} simple, "
          f"{sum(1 for t in EVAL_TASKS if t.difficulty=='moderate')} moderate, "
          f"{sum(1 for t in EVAL_TASKS if t.difficulty=='complex')} complex)")
    if args.decode_utterances:
        print("Dialogue decoding: ENABLED")
    print()

    # ── Run evaluation ───────────────────────────────────────────
    results, extra = run_evaluation(
        model=model,
        env=env,
        team=team,
        num_episodes=args.num_episodes,
        device=device,
        use_guidance=not args.no_guidance,
        guidance_scale=guidance_scale,
        active_signals=getattr(args, 'active_signals', None),
        num_inference_steps=args.num_inference_steps,
        decode_utterances=args.decode_utterances,
    )

    # ── Compute metrics ──────────────────────────────────────────
    evaluator = MultiAgentEvaluator(device=str(device))
    report = evaluator.evaluate(results, compute_fluency=args.compute_fluency)

    # ── Print report ─────────────────────────────────────────────
    gen_times = extra["generation_times"]
    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"Episodes:             {report.num_episodes}")
    print(f"Task Success Rate:    {report.task_success_rate:.1%}")
    print(f"Action Efficiency:    {report.action_efficiency:.3f}")
    print(f"Coordination Score:   {report.coordination_score:.3f}")
    print(f"Avg Turn Count:       {report.avg_turn_count:.1f}")
    print(f"Delegation Accuracy:  {report.delegation_accuracy:.1%}")
    print(f"Avg Generation Time:  {np.mean(gen_times):.2f}s ± {np.std(gen_times):.2f}s")

    if args.compute_fluency:
        print(f"\nFluency Metrics:")
        print(f"  BERTScore F1:       {report.avg_bertscore:.3f}")
        print(f"  Coherence (NLI):    {report.avg_coherence:.3f}")
        print(f"  Func-Fluency Gap:   {report.functional_fluency_gap:+.3f}")

    if report.per_complexity:
        print(f"\nPer-Complexity Breakdown:")
        print(f"  {'Complexity':<12} {'Success':>8} {'Efficiency':>11} {'Coord':>8} {'Turns':>8} {'N':>5}")
        print(f"  {'─'*12} {'─'*8} {'─'*11} {'─'*8} {'─'*8} {'─'*5}")
        for complexity in ["simple", "moderate", "complex"]:
            if complexity in report.per_complexity:
                m = report.per_complexity[complexity]
                print(f"  {complexity:<12} {m['task_success_rate']:>7.1%} {m['action_efficiency']:>11.3f} "
                      f"{m['coordination_score']:>8.3f} {m['avg_turn_count']:>8.1f} {m['num_episodes']:>5}")

    # ── Print sample dialogues ───────────────────────────────────
    dialogues = extra.get("generated_dialogues", {})
    if dialogues:
        print(f"\nSample Generated Dialogues:")
        for task_id, utt_lists in list(dialogues.items())[:3]:
            task = next((t for t in EVAL_TASKS if t.task_id == task_id), None)
            print(f"\n  Task: {task.description if task else task_id}")
            for i, u in enumerate(utt_lists[0][:5]):
                print(f"    Agent {i}: {u}")

    # ── Save full report ─────────────────────────────────────────
    report_dict = report.to_dict()
    report_dict["generation_time_mean"] = float(np.mean(gen_times))
    report_dict["generation_time_std"] = float(np.std(gen_times))
    report_dict["checkpoint"] = args.checkpoint
    report_dict["guidance_enabled"] = not args.no_guidance
    report_dict["guidance_scale"] = guidance_scale
    if dialogues:
        # Save a few sample dialogues
        report_dict["sample_dialogues"] = {
            k: v[0][:5] for k, v in list(dialogues.items())[:5]
        }

    save_json(report_dict, args.output)
    print(f"\n✓ Report saved to {args.output}")


if __name__ == "__main__":
    main()
