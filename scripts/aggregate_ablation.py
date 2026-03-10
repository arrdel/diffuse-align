"""
Aggregate ablation results across multiple seeds.

Reads per-seed ablation directories and produces a single summary with
mean ± std and 95% CIs for each condition.

Usage:
    python scripts/aggregate_ablation.py \
        --input_dir experiments/scaled_ablation \
        --output experiments/scaled_ablation/aggregated.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import save_json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str,
                   default="experiments/scaled_ablation")
    p.add_argument("--output", type=str,
                   default="experiments/scaled_ablation/aggregated.json")
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)

    # Find all seed directories
    seed_dirs = sorted([d for d in input_dir.iterdir()
                        if d.is_dir() and d.name.startswith("seed_")])

    if not seed_dirs:
        print(f"No seed directories found in {input_dir}")
        return

    print(f"Found {len(seed_dirs)} seed directories: "
          f"{[d.name for d in seed_dirs]}")

    # Read ablation_summary.json from each seed dir
    all_summaries = []
    for sd in seed_dirs:
        summary_file = sd / "ablation_summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                all_summaries.append(json.load(f))
        else:
            print(f"  Warning: no ablation_summary.json in {sd}")

    if not all_summaries:
        print("No summaries found!")
        return

    # Get condition names from the first summary
    conditions = list(all_summaries[0].keys())
    print(f"Conditions: {conditions}")

    aggregated = {"num_seeds": len(all_summaries)}

    for cond in conditions:
        cond_results = []
        for summary in all_summaries:
            if cond in summary:
                cond_results.append(summary[cond]["metrics"])

        if not cond_results:
            continue

        # Identify numeric keys
        numeric_keys = [k for k in cond_results[0]
                        if isinstance(cond_results[0][k], (int, float))]

        cond_agg = {}
        for k in numeric_keys:
            vals = [r[k] for r in cond_results if k in r]
            n = len(vals)
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals))
            ci95 = 1.96 * std_v / np.sqrt(n) if n > 1 else 0.0
            cond_agg[k] = {
                "mean": mean_v,
                "std": std_v,
                "ci95": ci95,
                "values": vals,
            }

        aggregated[cond] = cond_agg

    # ── Pretty print ─────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    print(f"AGGREGATED ABLATION ({len(all_summaries)} seeds)")
    print(f"{'=' * 90}")
    print(f"  {'Condition':<20} {'Success':>16} {'Efficiency':>16} {'Coord':>16} {'Turns':>14}")
    print(f"  {'─'*20} {'─'*16} {'─'*16} {'─'*16} {'─'*14}")

    for cond in conditions:
        if cond not in aggregated:
            continue
        ca = aggregated[cond]
        sr = ca.get("task_success_rate", {})
        ae = ca.get("action_efficiency", {})
        cs = ca.get("coordination_score", {})
        tc = ca.get("avg_turn_count", {})

        s_str = f"{sr.get('mean',0):.1%}±{sr.get('std',0):.1%}"
        e_str = f"{ae.get('mean',0):.3f}±{ae.get('std',0):.3f}"
        c_str = f"{cs.get('mean',0):.3f}±{cs.get('std',0):.3f}"
        t_str = f"{tc.get('mean',0):.1f}±{tc.get('std',0):.1f}"
        print(f"  {cond:<20} {s_str:>16} {e_str:>16} {c_str:>16} {t_str:>14}")

    # Delta from full
    if "full" in aggregated:
        print(f"\n  Δ from full:")
        full_sr = aggregated["full"]["task_success_rate"]["mean"]
        for cond in conditions:
            if cond == "full" or cond not in aggregated:
                continue
            sr = aggregated[cond].get("task_success_rate", {}).get("mean", 0)
            delta = sr - full_sr
            print(f"    {cond:<20} Δsuccess = {delta:+.1%}")

    save_json(aggregated, args.output)
    print(f"\n✓ Saved → {args.output}")


if __name__ == "__main__":
    main()
