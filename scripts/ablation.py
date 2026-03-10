"""
Ablation experiment runner.

Systematically disables components and re-evaluates to measure contribution
of each module. Runs each ablation sequentially with the same model,
toggling guidance signals and CFG scale.

Usage:
    python scripts/ablation.py \
        --checkpoint experiments/checkpoints/diffusealign_final.pt \
        --device cuda:0 --num_episodes 50

    # Run specific ablations only:
    python scripts/ablation.py \
        --checkpoint experiments/checkpoints/diffusealign_final.pt \
        --ablations full no_guidance task_only
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

# Import shared pieces from evaluate
from scripts.evaluate import (
    EVAL_TASKS,
    build_model,
    run_evaluation,
)


# ─── Ablation configurations ─────────────────────────────────────────
ABLATION_CONFIGS = {
    "full": {
        "description": "Full DiffuseAlign (all guidance signals, s=3.0)",
        "use_guidance": True,
        "active_signals": ["task", "safety", "efficiency", "coordination"],
        "guidance_scale": 3.0,
    },
    "no_guidance": {
        "description": "No compositional guidance (unconditional diffusion)",
        "use_guidance": False,
        "active_signals": [],
        "guidance_scale": 1.0,
    },
    "no_coordination": {
        "description": "Remove coordination guidance signal",
        "use_guidance": True,
        "active_signals": ["task", "safety", "efficiency"],
        "guidance_scale": 3.0,
    },
    "no_efficiency": {
        "description": "Remove efficiency guidance signal",
        "use_guidance": True,
        "active_signals": ["task", "safety", "coordination"],
        "guidance_scale": 3.0,
    },
    "no_safety": {
        "description": "Remove safety guidance signal",
        "use_guidance": True,
        "active_signals": ["task", "efficiency", "coordination"],
        "guidance_scale": 3.0,
    },
    "task_only": {
        "description": "Task completion guidance only",
        "use_guidance": True,
        "active_signals": ["task"],
        "guidance_scale": 3.0,
    },
    "low_cfg": {
        "description": "Low classifier-free guidance scale (s=1.5)",
        "use_guidance": True,
        "active_signals": ["task", "safety", "efficiency", "coordination"],
        "guidance_scale": 1.5,
    },
    "high_cfg": {
        "description": "High classifier-free guidance scale (s=7.0)",
        "use_guidance": True,
        "active_signals": ["task", "safety", "efficiency", "coordination"],
        "guidance_scale": 7.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="DiffuseAlign ablation experiments")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=50,
                        help="Episodes per ablation condition")
    parser.add_argument("--output_dir", type=str, default="experiments/ablation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decode_utterances", action="store_true",
                        help="Decode plan to NL dialogue for each ablation")
    parser.add_argument("--compute_fluency", action="store_true",
                        help="Compute BERTScore/NLI coherence (slow)")
    parser.add_argument("--ablations", nargs="+", default=None,
                        help="Specific ablations to run (default: all)")
    return parser.parse_args()


def print_comparison_table(all_results: Dict[str, dict]):
    """Print a formatted comparison table across all ablation conditions."""
    print("\n" + "=" * 90)
    print("ABLATION COMPARISON TABLE")
    print("=" * 90)

    header = (
        f"  {'Condition':<20} {'Success':>8} {'Efficiency':>11} {'Coord':>8} "
        f"{'Deleg':>8} {'Turns':>7} {'GenTime':>8}"
    )
    print(header)
    print("  " + "─" * 86)

    for name, res in all_results.items():
        m = res["metrics"]
        print(
            f"  {name:<20} {m['task_success_rate']:>7.1%} {m['action_efficiency']:>11.3f} "
            f"{m['coordination_score']:>8.3f} {m['delegation_accuracy']:>7.1%} "
            f"{m['avg_turn_count']:>7.1f} {res['gen_time_mean']:>7.2f}s"
        )

    # Also show fluency if available
    has_fluency = any("avg_bertscore" in r["metrics"] and r["metrics"]["avg_bertscore"] > 0
                      for r in all_results.values())
    if has_fluency:
        print("\n  Fluency Metrics:")
        print(f"  {'Condition':<20} {'BERTScore':>10} {'Coherence':>10} {'F-F Gap':>8}")
        print("  " + "─" * 50)
        for name, res in all_results.items():
            m = res["metrics"]
            print(
                f"  {name:<20} {m.get('avg_bertscore', 0):>10.3f} "
                f"{m.get('avg_coherence', 0):>10.3f} "
                f"{m.get('functional_fluency_gap', 0):>+8.3f}"
            )

    # Per-complexity breakdown for full vs no_guidance
    for cond_name in ["full", "no_guidance"]:
        if cond_name in all_results and "per_complexity" in all_results[cond_name]["metrics"]:
            pc = all_results[cond_name]["metrics"]["per_complexity"]
            print(f"\n  Per-Complexity ({cond_name}):")
            print(f"    {'Complexity':<12} {'Success':>8} {'Efficiency':>11} {'Coord':>8} {'N':>5}")
            print(f"    {'─'*12} {'─'*8} {'─'*11} {'─'*8} {'─'*5}")
            for c in ["simple", "moderate", "complex"]:
                if c in pc:
                    cm = pc[c]
                    print(
                        f"    {c:<12} {cm['task_success_rate']:>7.1%} "
                        f"{cm['action_efficiency']:>11.3f} "
                        f"{cm['coordination_score']:>8.3f} {cm['num_episodes']:>5}"
                    )


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    device = torch.device(args.device) if args.device else get_device()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ablations = args.ablations or list(ABLATION_CONFIGS.keys())

    print("=" * 60)
    print("DiffuseAlign — Ablation Experiments")
    print("=" * 60)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Device: {device}")
    print(f"  Episodes/condition: {args.num_episodes}")
    print(f"  Conditions: {len(ablations)} — {', '.join(ablations)}")
    print(f"  Output: {output_dir}")
    print()

    # ── Build and load model (once) ──────────────────────────────
    model = build_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {format_params(total_params)}")

    print(f"Loading checkpoint: {args.checkpoint}")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  Warning: {len(missing)} missing keys")
    if unexpected:
        print(f"  Warning: {len(unexpected)} unexpected keys")
    if not missing and not unexpected:
        print(f"  ✓ All {len(state)} keys loaded successfully")

    model = model.to(device)
    model.eval()

    # ── Environment and team ─────────────────────────────────────
    env = SimulatedMultiAgentEnv(seed=args.seed)
    team = AgentTeam.from_archetypes(
        ["navigator", "manipulator", "coordinator"],
        max_agents=cfg.model.diffusion.max_agents,
    )

    # Evaluator (shared, lazy-loads fluency models on first use)
    evaluator = MultiAgentEvaluator(device=str(device))

    # ── Run each ablation ────────────────────────────────────────
    all_results = {}
    total_time = 0

    for i, abl_name in enumerate(ablations):
        if abl_name not in ABLATION_CONFIGS:
            print(f"⚠️  Unknown ablation: {abl_name}, skipping")
            continue

        abl_cfg = ABLATION_CONFIGS[abl_name]

        print(f"\n{'─' * 60}")
        print(f"[{i+1}/{len(ablations)}] {abl_name}: {abl_cfg['description']}")
        print(f"  Guidance: {'ON' if abl_cfg['use_guidance'] else 'OFF'}, "
              f"signals: {abl_cfg['active_signals'] or '(none)'}, "
              f"scale: {abl_cfg['guidance_scale']}")
        print(f"{'─' * 60}")

        set_seed(args.seed)  # Reset seed per ablation for fair comparison
        env = SimulatedMultiAgentEnv(seed=args.seed)

        t0 = time.time()

        results, extra = run_evaluation(
            model=model,
            env=env,
            team=team,
            num_episodes=args.num_episodes,
            device=device,
            use_guidance=abl_cfg["use_guidance"],
            guidance_scale=abl_cfg["guidance_scale"],
            active_signals=abl_cfg["active_signals"] if abl_cfg["use_guidance"] else None,
            decode_utterances=args.decode_utterances,
        )

        elapsed = time.time() - t0
        total_time += elapsed

        # Compute metrics
        report = evaluator.evaluate(results, compute_fluency=args.compute_fluency)

        gen_times = extra["generation_times"]
        report_dict = report.to_dict()

        # Print summary for this ablation
        print(f"  ✓ Done in {elapsed:.1f}s")
        print(f"    Success: {report.task_success_rate:.1%}, "
              f"Efficiency: {report.action_efficiency:.3f}, "
              f"Coord: {report.coordination_score:.3f}, "
              f"Deleg: {report.delegation_accuracy:.1%}")
        print(f"    Gen time: {np.mean(gen_times):.2f}s ± {np.std(gen_times):.2f}s")

        if args.compute_fluency:
            print(f"    BERTScore: {report.avg_bertscore:.3f}, "
                  f"Coherence: {report.avg_coherence:.3f}")

        # Store results
        abl_result = {
            "ablation": abl_name,
            "description": abl_cfg["description"],
            "config": abl_cfg,
            "metrics": report_dict,
            "gen_time_mean": float(np.mean(gen_times)),
            "gen_time_std": float(np.std(gen_times)),
            "elapsed_s": elapsed,
            "num_episodes": args.num_episodes,
        }

        # Add sample dialogues if decoded
        dialogues = extra.get("generated_dialogues", {})
        if dialogues:
            abl_result["sample_dialogues"] = {
                k: v[0][:5] for k, v in list(dialogues.items())[:3]
            }

        all_results[abl_name] = abl_result
        save_json(abl_result, str(output_dir / f"{abl_name}.json"))

    # ── Summary ──────────────────────────────────────────────────
    print_comparison_table(all_results)

    print(f"\nTotal ablation time: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Save combined results
    summary_path = output_dir / "ablation_summary.json"
    save_json(all_results, str(summary_path))
    print(f"✓ Full results saved to {summary_path}")

    # Compute deltas from full model
    if "full" in all_results:
        full_m = all_results["full"]["metrics"]
        print("\n  Δ from Full Model:")
        print(f"  {'Condition':<20} {'ΔSuccess':>9} {'ΔEfficiency':>12} {'ΔCoord':>8}")
        print("  " + "─" * 50)
        for name, res in all_results.items():
            if name == "full":
                continue
            m = res["metrics"]
            ds = m["task_success_rate"] - full_m["task_success_rate"]
            de = m["action_efficiency"] - full_m["action_efficiency"]
            dc = m["coordination_score"] - full_m["coordination_score"]
            print(f"  {name:<20} {ds:>+8.1%} {de:>+12.3f} {dc:>+8.3f}")


if __name__ == "__main__":
    main()
