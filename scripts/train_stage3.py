"""
Stage 3: Train Guidance Classifiers (diffusion + encoder + decoder frozen).

Loads the Stage 2 model (or Stage 1 + fresh decoder), freezes everything
except the guidance classifiers, and trains on trajectory labels:
    - task_completion classifier: P(success | plan, condition)
    - coordination classifier: coordination quality score

Usage:
    python scripts/train_stage3.py --config configs/default.yaml \
        --stage2_ckpt experiments/checkpoints/stage2_gpu4/stage2_final.pt \
        --gpu 4

    # Or if only stage 1 is available:
    python scripts/train_stage3.py --config configs/default.yaml \
        --stage1_ckpt experiments/checkpoints/stage1_averaged.pt \
        --gpu 4
"""

from __future__ import annotations

import argparse
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffuse_align import DiffuseAlign
from src.dataset import MultiAgentTrajectoryDataset, collate_trajectories
from src.agents import VOCAB_SIZE
from src.utils import (
    set_seed,
    load_config,
    count_parameters,
    format_params,
    AverageMeter,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DiffuseAlign Stage 3 (Guidance)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--stage1_ckpt", type=str, default="experiments/checkpoints/stage1_averaged.pt")
    parser.add_argument("--stage2_ckpt", type=str, default=None,
                        help="Stage 2 checkpoint (if available). Overrides stage1_ckpt.")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    return parser.parse_args()


def build_model(cfg) -> DiffuseAlign:
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


class GuidanceTrainingDataset(Dataset):
    """
    Dataset for training guidance classifiers.

    Each sample provides:
        - plan: The plan tensor
        - condition features (from task description)
        - task_success: Binary label (was the trajectory successful?)
        - coordination_quality: Heuristic score based on redundancy/conflicts
    """

    def __init__(self, base_dataset: MultiAgentTrajectoryDataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def _compute_coordination_quality(self, traj) -> float:
        """
        Heuristic coordination quality score.

        Penalizes:
          - Multiple agents doing the same action at the same step (redundancy)
          - Agents doing 'nop' or 'wait' when others are active (idle waste)
        """
        if not traj.steps:
            return 0.5

        # Group actions by step_idx
        step_actions: dict[int, list[tuple[int, str]]] = {}
        for s in traj.steps:
            if s.step_idx not in step_actions:
                step_actions[s.step_idx] = []
            step_actions[s.step_idx].append((s.agent_id, s.action))

        total_steps = len(step_actions)
        if total_steps == 0:
            return 0.5

        redundancy_count = 0
        idle_count = 0
        total_agent_actions = 0

        for step_idx, actions in step_actions.items():
            action_set = set()
            active_count = 0
            for aid, action in actions:
                total_agent_actions += 1
                if action in ("nop", "wait"):
                    idle_count += 1
                else:
                    active_count += 1
                    if action in action_set:
                        redundancy_count += 1
                    action_set.add(action)

        if total_agent_actions == 0:
            return 0.5

        # Score: 1.0 = perfect, 0.0 = terrible
        redundancy_penalty = redundancy_count / max(total_agent_actions, 1)
        idle_penalty = idle_count / max(total_agent_actions, 1)
        quality = max(0.0, 1.0 - redundancy_penalty - 0.3 * idle_penalty)

        return quality

    def __getitem__(self, idx):
        base_item = self.base_dataset[idx]
        traj = self.base_dataset.trajectories[idx]

        coord_quality = self._compute_coordination_quality(traj)

        return {
            **base_item,
            "task_success": base_item["success"],
            "coordination_quality": torch.tensor(coord_quality, dtype=torch.float32),
        }


def collate_guidance(batch):
    """Collate function for guidance training."""
    return {
        "plan": torch.stack([b["plan"] for b in batch]),
        "plan_actions": torch.stack([b["plan_actions"] for b in batch]),
        "validity_mask": torch.stack([b["validity_mask"] for b in batch]),
        "task_descriptions": [b["task_description"] for b in batch],
        "num_agents": torch.stack([b["num_agents"] for b in batch]),
        "plan_lengths": torch.stack([b["plan_length"] for b in batch]),
        "task_success": torch.stack([b["task_success"] for b in batch]),
        "coordination_quality": torch.stack([b["coordination_quality"] for b in batch]),
    }


def find_latest_checkpoint(output_dir: Path, prefix: str = "stage3") -> str | None:
    ckpts = sorted(output_dir.glob(f"{prefix}_step*.pt"))
    if not ckpts:
        return None
    def step_num(p):
        try:
            return int(p.stem.split("step")[1])
        except (IndexError, ValueError):
            return -1
    ckpts.sort(key=step_num)
    return str(ckpts[-1])


def cleanup_old_checkpoints(output_dir: Path, keep_last: int = 3, prefix: str = "stage3"):
    ckpts = sorted(output_dir.glob(f"{prefix}_step*.pt"))
    def step_num(p):
        try:
            return int(p.stem.split("step")[1])
        except (IndexError, ValueError):
            return -1
    ckpts.sort(key=step_num)
    if len(ckpts) > keep_last:
        for old in ckpts[:-keep_last]:
            old.unlink()
            print(f"  🗑️  Removed old checkpoint: {old.name}")


def main():
    args = parse_args()
    set_seed(args.seed + args.gpu)

    cfg = load_config(args.config)
    guidance_cfg = cfg.training.guidance_training

    gpu_id = args.gpu
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    # Hyperparams
    lr = args.lr or guidance_cfg.lr
    batch_size = args.batch_size or guidance_cfg.batch_size
    grad_accum = args.grad_accum
    num_epochs = args.num_epochs or guidance_cfg.num_epochs
    if args.smoke_test:
        num_epochs = 1
        batch_size = 8
        grad_accum = 1

    print("=" * 60)
    mode = "🔥 SMOKE TEST" if args.smoke_test else "🚀 FULL TRAINING"
    print(f"Stage 3: Guidance Classifiers — {mode}")
    print(f"  GPU: {gpu_id}, LR: {lr}, Batch: {batch_size}, Accum: {grad_accum}")
    print(f"  Epochs: {num_epochs}")
    print("=" * 60)

    # Output directory
    run_name = args.run_name or f"stage3_gpu{gpu_id}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    print("\nBuilding model...")
    model = build_model(cfg)

    # Load pretrained weights (prefer stage2, fallback stage1)
    load_path = args.stage2_ckpt or args.stage1_ckpt
    if load_path and os.path.exists(load_path):
        print(f"Loading checkpoint: {load_path}")
        state = torch.load(load_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print(f"⚠️  No pretrained checkpoint found. Training guidance from scratch.")

    model = model.to(device)

    # Freeze everything except guidance classifiers
    for name, param in model.named_parameters():
        if "guidance" not in name:
            param.requires_grad = False

    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    print(f"Total params: {format_params(total)}, Trainable (guidance): {format_params(trainable)}")

    # Dataset: load all 3 datasets
    print("\nLoading datasets...")
    all_trajectories = []
    for ds_cfg in cfg.data.train_datasets:
        ds = MultiAgentTrajectoryDataset(
            data_dir=ds_cfg.path,
            max_agents=cfg.model.diffusion.max_agents,
            max_steps=cfg.model.diffusion.max_plan_steps,
            plan_dim=cfg.model.diffusion.plan_dim,
        )
        all_trajectories.extend(ds.trajectories)
        print(f"  {ds_cfg.name}: {len(ds.trajectories)} trajectories")

    # Combined dataset
    combined_ds = MultiAgentTrajectoryDataset.__new__(MultiAgentTrajectoryDataset)
    combined_ds.max_agents = cfg.model.diffusion.max_agents
    combined_ds.max_steps = cfg.model.diffusion.max_plan_steps
    combined_ds.plan_dim = cfg.model.diffusion.plan_dim
    combined_ds.action_embed_dim = 256
    combined_ds.action_embedding = torch.nn.Embedding(VOCAB_SIZE, 256)
    combined_ds.trajectories = all_trajectories
    combined_ds.data_dir = Path("data/combined")

    guidance_dataset = GuidanceTrainingDataset(combined_ds)
    print(f"Total guidance samples: {len(guidance_dataset)}")

    # Log class balance
    success_count = sum(1 for t in all_trajectories if t.success)
    print(f"  Success rate: {success_count}/{len(all_trajectories)} "
          f"({100 * success_count / max(len(all_trajectories), 1):.1f}%)")

    train_loader = DataLoader(
        guidance_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if not args.smoke_test else 0,
        collate_fn=collate_guidance,
        pin_memory=True,
    )

    # Optimizer & scheduler
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    total_steps = num_epochs * len(train_loader) // grad_accum
    warmup_steps = min(200, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(1e-7 / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    max_agents_cfg = cfg.model.diffusion.max_agents

    # Resume logic
    start_step = 0
    start_epoch = 0
    resume_path = args.resume
    if resume_path is None and not args.smoke_test:
        resume_path = find_latest_checkpoint(output_dir)
        if resume_path:
            print(f"Auto-resuming from: {resume_path}")

    if resume_path and os.path.exists(resume_path):
        print(f"Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            start_step = ckpt.get("step", 0)
            start_epoch = ckpt.get("epoch", 0)
        print(f"✅ Resumed from step {start_step}, epoch {start_epoch}")

    # Checkpoint helper
    save_every = 500
    keep_last = 3

    def save_checkpoint(step, epoch, loss_avg, reason="periodic"):
        ckpt_path = output_dir / f"stage3_step{step}.pt"
        torch.save({
            "step": step,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "loss": loss_avg,
        }, ckpt_path)
        print(f"\n💾 Checkpoint ({reason}): {ckpt_path}  [step {step}, loss {loss_avg:.4f}]")
        cleanup_old_checkpoints(output_dir, keep_last=keep_last)

    # SIGTERM handler
    _interrupted = False
    def _signal_handler(signum, frame):
        nonlocal _interrupted
        sig_name = signal.Signals(signum).name
        print(f"\n⚠️  Received {sig_name} — saving checkpoint...")
        _interrupted = True
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # Positive class weight for imbalanced task_success labels
    n_total = len(all_trajectories)
    n_pos = max(success_count, 1)
    pos_weight = torch.tensor([(n_total - n_pos) / n_pos], device=device).clamp(max=20.0)
    print(f"  BCE pos_weight for task_success: {pos_weight.item():.1f}")

    # Training loop
    model.train()
    # Keep frozen parts in eval
    model.plan_diffusion.eval()
    model.plan_encoder.eval()
    model.role_masker.eval()
    model.plan_decoder.eval()

    loss_meter = AverageMeter("loss")
    task_loss_meter = AverageMeter("task")
    coord_loss_meter = AverageMeter("coord")
    global_step = start_step
    accum_count = 0
    start_time = time.time()
    _done = False

    print(f"\n🚀 Training guidance: {num_epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"   Total steps: ~{total_steps}, Warmup: {warmup_steps}")
    print(f"   Starting from step {start_step}, epoch {start_epoch}\n")

    for epoch in range(start_epoch, num_epochs):
        loss_meter.reset()
        task_loss_meter.reset()
        coord_loss_meter.reset()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in pbar:
            plans = batch["plan"].to(device)
            task_texts = batch["task_descriptions"]
            task_success = batch["task_success"].to(device)
            coord_quality = batch["coordination_quality"].to(device)

            # Get condition from plan encoder (frozen, no grad)
            num_agents_batch = batch["num_agents"].to(device)
            agent_states = torch.randn(
                plans.shape[0], max_agents_cfg, 128, device=device
            )
            with torch.no_grad():
                condition = model.plan_encoder(task_texts, agent_states)

            # Compute guidance losses
            labels = {
                "task_success": task_success,
                "coordination_quality": coord_quality,
            }
            losses = model.guidance.training_losses(plans, condition, labels)

            # Combined loss
            loss = torch.tensor(0.0, device=device)
            if "task_completion" in losses:
                task_loss = losses["task_completion"]
                loss = loss + task_loss
                task_loss_meter.update(task_loss.item())
            if "coordination" in losses:
                coord_loss = losses["coordination"]
                loss = loss + coord_loss
                coord_loss_meter.update(coord_loss.item())

            if loss.item() == 0.0:
                continue

            # Gradient accumulation
            (loss / grad_accum).backward()
            loss_meter.update(loss.item())

            accum_count += 1
            if accum_count >= grad_accum:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1

                pbar.set_postfix(
                    loss=f"{loss_meter.avg:.4f}",
                    task=f"{task_loss_meter.avg:.4f}",
                    coord=f"{coord_loss_meter.avg:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    step=global_step,
                )

                if not args.smoke_test and global_step % save_every == 0:
                    save_checkpoint(global_step, epoch, loss_meter.avg)

                if _interrupted:
                    if not args.smoke_test:
                        save_checkpoint(global_step, epoch, loss_meter.avg, reason="interrupted")
                    _done = True
                    break

        if _done:
            break

        # End-of-epoch checkpoint
        if not args.smoke_test:
            save_checkpoint(global_step, epoch + 1, loss_meter.avg, reason=f"epoch_{epoch+1}")

        print(f"  Epoch {epoch + 1}: loss={loss_meter.avg:.4f}, task={task_loss_meter.avg:.4f}, "
              f"coord={coord_loss_meter.avg:.4f}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Stage 3 {'interrupted' if _interrupted else 'complete'}!")
    print(f"  Steps: {global_step}, Loss: {loss_meter.avg:.4f}")
    print(f"  Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 60)

    if not args.smoke_test and not _interrupted:
        final_path = output_dir / "stage3_final.pt"
        torch.save(model.state_dict(), final_path)
        print(f"Final model: {final_path}")


if __name__ == "__main__":
    main()
