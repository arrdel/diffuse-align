"""
Ablation experiment runner.

Systematically disables components and re-evaluates to measure contribution
of each module.

Usage:
    python scripts/ablation.py --config configs/default.yaml --checkpoint experiments/checkpoints/stage1_final.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import set_seed, load_config, save_json, get_device


ABLATION_CONFIGS = {
    "full": {
        "description": "Full DiffuseAlign",
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
        "description": "Remove coordination guidance only",
        "use_guidance": True,
        "active_signals": ["task", "safety", "efficiency"],
        "guidance_scale": 3.0,
    },
    "no_efficiency": {
        "description": "Remove efficiency guidance only",
        "use_guidance": True,
        "active_signals": ["task", "safety", "coordination"],
        "guidance_scale": 3.0,
    },
    "no_safety": {
        "description": "Remove safety guidance only",
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
        "description": "Low classifier-free guidance (s=1.5)",
        "use_guidance": True,
        "active_signals": ["task", "safety", "efficiency", "coordination"],
        "guidance_scale": 1.5,
    },
    "high_cfg": {
        "description": "High classifier-free guidance (s=7.0)",
        "use_guidance": True,
        "active_signals": ["task", "safety", "efficiency", "coordination"],
        "guidance_scale": 7.0,
    },
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_episodes", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="experiments/ablation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablations", nargs="+", default=None,
                        help="Specific ablations to run (default: all)")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = load_config(args.config)
    device = torch.device(args.device) if args.device else get_device()

    ablations = args.ablations or list(ABLATION_CONFIGS.keys())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running {len(ablations)} ablation experiments")
    print(f"Episodes per ablation: {args.num_episodes}")
    print(f"Output: {output_dir}")
    print()

    all_results = {}

    for abl_name in ablations:
        if abl_name not in ABLATION_CONFIGS:
            print(f"Unknown ablation: {abl_name}, skipping")
            continue

        abl_cfg = ABLATION_CONFIGS[abl_name]
        print(f"{'=' * 50}")
        print(f"Ablation: {abl_name}")
        print(f"  {abl_cfg['description']}")
        print(f"  Guidance: {abl_cfg['use_guidance']}, signals: {abl_cfg['active_signals']}")
        print(f"  CFG scale: {abl_cfg['guidance_scale']}")

        # In full implementation: run evaluate.py with these settings
        # For now, save the config
        result = {
            "ablation": abl_name,
            "config": abl_cfg,
            "status": "pending",
            "metrics": {},
        }

        all_results[abl_name] = result
        save_json(result, str(output_dir / f"{abl_name}.json"))

    # Summary
    save_json(all_results, str(output_dir / "ablation_summary.json"))
    print(f"\nAblation configs saved to {output_dir}")
    print("Run evaluate.py with each config to get results.")


if __name__ == "__main__":
    main()
