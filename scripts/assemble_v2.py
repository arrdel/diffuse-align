#!/usr/bin/env python
"""
Post-training: Average Stage 2 v2 checkpoints and reassemble final model.

Usage:
    python scripts/assemble_v2.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import OrderedDict

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def average_checkpoints(paths: list[str]) -> OrderedDict:
    """Average model state dicts from multiple checkpoints."""
    states = []
    for p in paths:
        ckpt = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            states.append(ckpt["model_state"])
        elif isinstance(ckpt, OrderedDict):
            states.append(ckpt)
        else:
            states.append(ckpt)
        print(f"  Loaded: {p} ({len(states[-1])} keys)")

    avg_state = OrderedDict()
    for key in states[0]:
        tensors = [s[key].float() for s in states]
        avg_state[key] = sum(tensors) / len(tensors)

    return avg_state


def main():
    ckpt_dir = Path("experiments/checkpoints")

    # Find Stage 2 v2 final checkpoints
    gpu4_dir = ckpt_dir / "stage2_gpu4_v2"
    gpu5_dir = ckpt_dir / "stage2_gpu5_v2"

    # Look for final checkpoint or latest step checkpoint
    def find_best(d: Path) -> str | None:
        final = d / "stage2_final.pt"
        if final.exists():
            return str(final)
        # Find latest step checkpoint
        ckpts = sorted(d.glob("stage2_step*.pt"))
        if ckpts:
            return str(ckpts[-1])
        return None

    gpu4_ckpt = find_best(gpu4_dir)
    gpu5_ckpt = find_best(gpu5_dir)

    if not gpu4_ckpt:
        print(f"ERROR: No checkpoint found in {gpu4_dir}")
        sys.exit(1)
    if not gpu5_ckpt:
        print(f"ERROR: No checkpoint found in {gpu5_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Stage 2 v2: Averaging checkpoints")
    print("=" * 60)

    avg_state = average_checkpoints([gpu4_ckpt, gpu5_ckpt])

    # Save averaged Stage 2
    avg_path = ckpt_dir / "stage2_v2_averaged.pt"
    torch.save(avg_state, avg_path)
    print(f"\nSaved averaged Stage 2 v2: {avg_path}")
    print(f"  Size: {avg_path.stat().st_size / 1e9:.2f} GB")

    # Now reassemble the full model
    print("\n" + "=" * 60)
    print("Reassembling final model with NL decoder")
    print("=" * 60)

    # Load existing final model
    old_final = ckpt_dir / "diffusealign_final.pt"
    if old_final.exists():
        print(f"Loading existing model: {old_final}")
        final_state = torch.load(old_final, map_location="cpu", weights_only=False)
    else:
        # Build from individual stages
        print("Building from individual stages...")
        final_state = OrderedDict()

        # Stage 1
        s1 = torch.load(ckpt_dir / "stage1_averaged.pt", map_location="cpu", weights_only=False)
        if isinstance(s1, dict) and "model_state" in s1:
            s1 = s1["model_state"]
        final_state.update(s1)

        # Stage 3
        s3 = torch.load(ckpt_dir / "stage3_averaged.pt", map_location="cpu", weights_only=False)
        if isinstance(s3, dict) and "model_state" in s3:
            s3 = s3["model_state"]
        final_state.update(s3)

    # Replace decoder weights with new v2 weights
    decoder_keys = [k for k in avg_state if "plan_decoder" in k]
    non_decoder_keys = [k for k in avg_state if "plan_decoder" not in k]

    print(f"  Total keys in averaged: {len(avg_state)}")
    print(f"  Decoder keys: {len(decoder_keys)}")
    print(f"  Non-decoder keys: {len(non_decoder_keys)}")

    # Remove old decoder keys from final
    old_decoder_keys = [k for k in final_state if "plan_decoder" in k]
    for k in old_decoder_keys:
        del final_state[k]
    print(f"  Removed {len(old_decoder_keys)} old decoder keys")

    # Add new decoder keys
    for k in decoder_keys:
        final_state[k] = avg_state[k]
    print(f"  Added {len(decoder_keys)} new decoder keys")

    # Save new final model
    new_final = ckpt_dir / "diffusealign_final_v2.pt"
    torch.save(final_state, new_final)
    print(f"\n✅ New final model: {new_final}")
    print(f"   Size: {new_final.stat().st_size / 1e9:.2f} GB")
    print(f"   Total keys: {len(final_state)}")

    # Also update the original path for evaluate.py compatibility
    backup_path = ckpt_dir / "diffusealign_final_v1_backup.pt"
    if old_final.exists() and not backup_path.exists():
        old_final.rename(backup_path)
        print(f"   Backed up old model to: {backup_path}")

    import shutil
    shutil.copy2(new_final, old_final)
    print(f"   Copied v2 to: {old_final}")


if __name__ == "__main__":
    main()
