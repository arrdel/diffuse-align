#!/usr/bin/env python
"""
Post-training: Average Stage 2 v3 checkpoints and reassemble final model.

v3 uses decoder trained on actual diffusion plan representations,
fixing the representation mismatch from v1/v2.

Usage:
    python scripts/assemble_v3.py
"""

from __future__ import annotations

import sys
import shutil
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

    # Stage 2 v3 directories
    gpu4_dir = ckpt_dir / "stage2v3_gpu4"
    gpu5_dir = ckpt_dir / "stage2v3_gpu5"

    # Use the best checkpoint from each GPU
    gpu4_best = gpu4_dir / "stage2v3_best.pt"
    gpu5_best = gpu5_dir / "stage2v3_best.pt"

    if not gpu4_best.exists():
        print(f"ERROR: No best checkpoint in {gpu4_dir}")
        sys.exit(1)
    if not gpu5_best.exists():
        print(f"ERROR: No best checkpoint in {gpu5_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Stage 2 v3: Averaging best checkpoints")
    print("=" * 60)

    avg_state = average_checkpoints([str(gpu4_best), str(gpu5_best)])

    # Save averaged Stage 2 v3
    avg_path = ckpt_dir / "stage2_v3_averaged.pt"
    torch.save(avg_state, avg_path)
    print(f"\nSaved averaged Stage 2 v3: {avg_path}")
    print(f"  Size: {avg_path.stat().st_size / 1e9:.2f} GB")

    # Reassemble the full model
    print("\n" + "=" * 60)
    print("Reassembling final model with v3 decoder")
    print("=" * 60)

    # Load existing final model as base (has stage1 + stage3)
    old_final = ckpt_dir / "diffusealign_final.pt"
    if old_final.exists():
        print(f"Loading base model: {old_final}")
        final_state = torch.load(old_final, map_location="cpu", weights_only=False)
    else:
        # Build from individual stages
        print("Building from individual stages...")
        final_state = OrderedDict()

        s1 = torch.load(ckpt_dir / "stage1_averaged.pt", map_location="cpu", weights_only=False)
        if isinstance(s1, dict) and "model_state" in s1:
            s1 = s1["model_state"]
        final_state.update(s1)

        s3 = torch.load(ckpt_dir / "stage3_averaged.pt", map_location="cpu", weights_only=False)
        if isinstance(s3, dict) and "model_state" in s3:
            s3 = s3["model_state"]
        final_state.update(s3)

    # Replace decoder weights with v3 averaged weights
    decoder_keys = [k for k in avg_state if "plan_decoder" in k]
    non_decoder_keys = [k for k in avg_state if "plan_decoder" not in k]

    print(f"  Averaged state: {len(avg_state)} total keys")
    print(f"    Decoder keys: {len(decoder_keys)}")
    print(f"    Non-decoder keys: {len(non_decoder_keys)}")

    # Remove old decoder keys
    old_decoder_keys = [k for k in final_state if "plan_decoder" in k]
    for k in old_decoder_keys:
        del final_state[k]
    print(f"  Removed {len(old_decoder_keys)} old decoder keys")

    # Add new v3 decoder keys
    for k in decoder_keys:
        final_state[k] = avg_state[k]
    print(f"  Added {len(decoder_keys)} new v3 decoder keys")

    # Also update any non-decoder keys (VQ codebook etc)
    for k in non_decoder_keys:
        final_state[k] = avg_state[k]
    if non_decoder_keys:
        print(f"  Updated {len(non_decoder_keys)} non-decoder keys")

    # Save
    new_final = ckpt_dir / "diffusealign_final_v3.pt"
    torch.save(final_state, new_final)
    print(f"\n✅ New final model (v3): {new_final}")
    print(f"   Size: {new_final.stat().st_size / 1e9:.2f} GB")
    print(f"   Total keys: {len(final_state)}")

    # Backup current and replace
    backup_path = ckpt_dir / "diffusealign_final_v2_backup.pt"
    if old_final.exists() and not backup_path.exists():
        shutil.copy2(old_final, backup_path)
        print(f"   Backed up v2 model to: {backup_path}")

    shutil.copy2(new_final, old_final)
    print(f"   Copied v3 to: {old_final}")

    print("\n🎉 Model assembly complete! Ready for evaluation.")


if __name__ == "__main__":
    main()
