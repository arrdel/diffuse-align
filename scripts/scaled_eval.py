"""
Scaled multi-seed evaluation for DiffuseAlign.

Runs evaluation across multiple seeds and aggregates results with
confidence intervals for the paper.  Parallelises across GPUs by
launching one seed per GPU.

Usage (recommended — via the wrapper shell script):
    bash scripts/run_scaled_eval.sh

Or directly:
    python scripts/scaled_eval.py \
        --checkpoint experiments/checkpoints/diffusealign_final.pt \
        --num_episodes 500 \
        --seed 42 \
        --device cuda:0 \
        --decode_utterances \
        --output experiments/scaled/seed_42.json

After all seeds finish, run the aggregator:
    python scripts/scaled_eval.py --aggregate \
        --input_dir experiments/scaled \
        --output experiments/scaled/aggregated.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffuse_align import DiffuseAlign
from src.environment import SimulatedMultiAgentEnv, TaskSpec
from src.agents import AgentTeam, VOCAB_SIZE
from src.evaluation import MultiAgentEvaluator, EpisodeResult
from src.utils import set_seed, load_config, save_json, get_device, format_params

from scripts.evaluate import EVAL_TASKS, build_model, run_evaluation


# ─── Single-seed evaluation ──────────────────────────────────────────

def run_single_seed(args):
    """Run a full evaluation for one seed on one GPU."""
    set_seed(args.seed)
    cfg = load_config(args.config)
    device = torch.device(args.device) if args.device else get_device()

    print(f"[seed={args.seed} device={device}] Loading model...")
    model = build_model(cfg)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    elif isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    env = SimulatedMultiAgentEnv(seed=args.seed)
    team = AgentTeam.from_archetypes(
        ["navigator", "manipulator", "coordinator"],
        max_agents=cfg.model.diffusion.max_agents,
    )

    print(f"[seed={args.seed}] Running {args.num_episodes} episodes...")
    t0 = time.time()

    results, extra = run_evaluation(
        model=model,
        env=env,
        team=team,
        num_episodes=args.num_episodes,
        device=device,
        use_guidance=True,
        guidance_scale=args.guidance_scale,
        decode_utterances=args.decode_utterances,
    )

    elapsed = time.time() - t0

    evaluator = MultiAgentEvaluator(device=str(device))
    report = evaluator.evaluate(results, compute_fluency=False)

    gen_times = extra["generation_times"]
    report_dict = report.to_dict()
    report_dict["seed"] = args.seed
    report_dict["device"] = str(device)
    report_dict["elapsed_s"] = elapsed
    report_dict["generation_time_mean"] = float(np.mean(gen_times))
    report_dict["generation_time_std"] = float(np.std(gen_times))

    # Save sample dialogues
    dialogues = extra.get("generated_dialogues", {})
    if dialogues:
        report_dict["sample_dialogues"] = {
            k: v[0][:5] for k, v in list(dialogues.items())[:5]
        }

    save_json(report_dict, args.output)
    print(f"[seed={args.seed}] ✓ Done in {elapsed:.1f}s — "
          f"success={report.task_success_rate:.1%}, "
          f"eff={report.action_efficiency:.3f}")
    print(f"[seed={args.seed}] Saved → {args.output}")


# ─── Aggregation across seeds ────────────────────────────────────────

def aggregate(args):
    """Aggregate per-seed JSON files into a single result with CIs."""
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("seed_*.json"))

    if not files:
        print(f"No seed_*.json files found in {input_dir}")
        return

    print(f"Aggregating {len(files)} seed results from {input_dir}")

    all_reports = []
    for f in files:
        with open(f) as fh:
            all_reports.append(json.load(fh))

    # Identify numeric keys
    numeric_keys = [k for k in all_reports[0]
                    if isinstance(all_reports[0][k], (int, float))
                    and k not in ("seed",)]

    agg = {"num_seeds": len(all_reports), "seeds": [r["seed"] for r in all_reports]}

    for k in numeric_keys:
        vals = [r[k] for r in all_reports if k in r]
        mean_v = float(np.mean(vals))
        std_v = float(np.std(vals))
        # 95% CI (t-distribution approximation for small n)
        n = len(vals)
        ci_half = 1.96 * std_v / np.sqrt(n) if n > 1 else 0.0
        agg[k] = {
            "mean": mean_v,
            "std": std_v,
            "ci95": ci_half,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": vals,
        }

    # Pretty-print key metrics
    print(f"\n{'=' * 70}")
    print(f"AGGREGATED RESULTS ({len(all_reports)} seeds)")
    print(f"{'=' * 70}")
    for k in ["task_success_rate", "action_efficiency", "coordination_score",
              "avg_turn_count", "delegation_accuracy", "generation_time_mean"]:
        if k in agg:
            m = agg[k]
            print(f"  {k:<28} {m['mean']:.4f} ± {m['std']:.4f}  "
                  f"(95% CI: ±{m['ci95']:.4f})")

    # Per-complexity
    complexities = ["simple", "moderate", "complex"]
    for cx in complexities:
        key = f"{cx}/task_success_rate"
        if key in agg:
            m = agg[key]
            print(f"  {key:<28} {m['mean']:.4f} ± {m['std']:.4f}")

    save_json(agg, args.output)
    print(f"\n✓ Aggregated results → {args.output}")


# ─── CLI ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Scaled evaluation for DiffuseAlign")
    p.add_argument("--aggregate", action="store_true",
                   help="Aggregate per-seed results instead of running eval")

    # Evaluation args
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--checkpoint", type=str,
                   default="experiments/checkpoints/diffusealign_final.pt")
    p.add_argument("--num_episodes", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--guidance_scale", type=float, default=3.0)
    p.add_argument("--decode_utterances", action="store_true")
    p.add_argument("--output", type=str, default="experiments/scaled/seed_42.json")

    # Aggregation args
    p.add_argument("--input_dir", type=str, default="experiments/scaled")

    return p.parse_args()


def main():
    args = parse_args()
    if args.aggregate:
        aggregate(args)
    else:
        run_single_seed(args)


if __name__ == "__main__":
    main()
