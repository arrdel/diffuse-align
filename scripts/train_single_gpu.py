"""
Single-GPU training script for DiffuseAlign (Stage 1: Diffusion Model).

This script trains on a single GPU with gradient accumulation.
For multi-GPU training, launch 8 instances with --gpu 0..7 and use
scripts/train_parallel.sh to manage them + periodic weight averaging.

Usage:
    # Smoke test (single GPU):
    python scripts/train_single_gpu.py --config configs/default.yaml --smoke_test --gpu 0

    # Full training (single GPU, grad accum):
    python scripts/train_single_gpu.py --config configs/default.yaml \
        --output_dir experiments/checkpoints --gpu 0

    # Multi-GPU via wrapper:
    bash scripts/train_parallel.sh
"""

from __future__ import annotations

import argparse
import glob
import os
import signal
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffuse_align import DiffuseAlign
from src.dataset import get_dataloader
from src.agents import VOCAB_SIZE
from src.utils import (
    set_seed,
    load_config,
    save_json,
    count_parameters,
    format_params,
    AverageMeter,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DiffuseAlign (single GPU)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--run_name", type=str, default=None, help="Name for output subdir (default: gpu{id})")
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None, help="Micro-batch per step")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_steps", type=int, default=None)
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


def find_latest_checkpoint(output_dir: Path) -> str | None:
    """Find the most recent checkpoint in output_dir by step number."""
    ckpts = sorted(output_dir.glob("stage1_step*.pt"))
    if not ckpts:
        return None
    # Sort by step number extracted from filename
    def step_num(p):
        try:
            return int(p.stem.split("step")[1])
        except (IndexError, ValueError):
            return -1
    ckpts.sort(key=step_num)
    return str(ckpts[-1])


def cleanup_old_checkpoints(output_dir: Path, keep_last: int = 3):
    """Remove old checkpoints, keeping only the most recent `keep_last`."""
    ckpts = sorted(output_dir.glob("stage1_step*.pt"))
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
    set_seed(args.seed + args.gpu)  # different seed per GPU for data diversity
    cfg = load_config(args.config)
    train_cfg = cfg.training

    gpu_id = args.gpu
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    print("=" * 60)
    mode = "🔥 SMOKE TEST" if args.smoke_test else "🚀 FULL TRAINING"
    print(f"{mode} — GPU {gpu_id}")

    if args.smoke_test:
        micro_batch = args.batch_size or 8
        grad_accum = args.grad_accum or 1
        max_steps = args.max_steps or 5
    else:
        # Default: effective batch 64 = micro 8 × accum 8
        micro_batch = args.batch_size or 8
        grad_accum = args.grad_accum or 8
        max_steps = args.max_steps or train_cfg.scheduler.num_training_steps

    effective_batch = micro_batch * grad_accum
    print(f"   Micro-batch: {micro_batch}, Grad accum: {grad_accum}")
    print(f"   Effective batch: {effective_batch}")
    print(f"   Max steps: {max_steps}")
    print("=" * 60)

    # Output directory
    run_name = args.run_name or f"gpu{gpu_id}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    keep_last = getattr(train_cfg.checkpoint, "keep_last", 3)

    # Build model
    print("\nBuilding model...")
    model = build_model(cfg)
    total_p = count_parameters(model, trainable_only=False)
    train_p = count_parameters(model, trainable_only=True)
    print(f"Total params: {format_params(total_p)}, Trainable: {format_params(train_p)}")
    model = model.to(device)
    model.train()

    # Data
    train_loader = get_dataloader(
        data_dir=cfg.data.train_datasets[0].path,
        batch_size=micro_batch,
        max_agents=cfg.model.diffusion.max_agents,
        max_steps=cfg.model.diffusion.max_plan_steps,
        plan_dim=cfg.model.diffusion.plan_dim,
        num_workers=2 if not args.smoke_test else 0,
    )
    print(f"Dataset: {len(train_loader.dataset)} trajectories, {len(train_loader)} batches/epoch")

    # Optimizer & scheduler (linear warmup + cosine decay)
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.optimizer.lr,
        weight_decay=train_cfg.optimizer.weight_decay,
        betas=tuple(train_cfg.optimizer.betas),
    )
    warmup_steps = getattr(train_cfg.scheduler, "warmup_steps", 1000)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        import math
        return max(1e-6 / train_cfg.optimizer.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    scheduler = LambdaLR(optimizer, lr_lambda)
    max_agents_cfg = cfg.model.diffusion.max_agents

    # ── Resume logic ──────────────────────────────────────────────
    # Priority: explicit --resume flag > auto-detect latest checkpoint
    start_step = 0
    resume_path = args.resume
    if resume_path is None and not args.smoke_test:
        resume_path = find_latest_checkpoint(output_dir)
        if resume_path:
            print(f"Auto-resuming from: {resume_path}")

    if resume_path:
        print(f"Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            start_step = ckpt.get("step", 0)
        else:
            model.load_state_dict(ckpt)
        print(f"✅ Resumed from step {start_step} (lr={scheduler.get_last_lr()[0]:.2e})")

    # ── Checkpoint save helper ────────────────────────────────────
    def save_checkpoint(step: int, loss_avg: float, reason: str = "periodic"):
        ckpt_path = output_dir / f"stage1_step{step}.pt"
        torch.save(
            {
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "loss": loss_avg,
                "config": str(args.config),
                "run_name": run_name,
                "seed": args.seed,
            },
            ckpt_path,
        )
        print(f"\n💾 Checkpoint ({reason}): {ckpt_path}  [step {step}, loss {loss_avg:.4f}]")
        cleanup_old_checkpoints(output_dir, keep_last=keep_last)

    # ── SIGTERM / SIGINT handler for graceful shutdown ────────────
    _interrupted = False
    def _signal_handler(signum, frame):
        nonlocal _interrupted
        sig_name = signal.Signals(signum).name
        print(f"\n⚠️  Received {sig_name} — will save checkpoint and exit after current step...")
        _interrupted = True

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    # W&B (only for GPU 0 in multi-GPU, or always for single)
    use_wandb = not args.smoke_test and gpu_id == 0
    if use_wandb:
        try:
            import wandb
            wandb.init(
                project="diffuse-align",
                name=f"stage1_gpu{gpu_id}",
                config={
                    "micro_batch": micro_batch,
                    "grad_accum": grad_accum,
                    "effective_batch": effective_batch,
                    "max_steps": max_steps,
                    "lr": train_cfg.optimizer.lr,
                    "gpu": gpu_id,
                },
            )
        except Exception as e:
            print(f"W&B init failed: {e}")
            use_wandb = False

    # Training loop
    loss_meter = AverageMeter("loss")
    global_step = start_step
    start_time = time.time()
    accum_count = 0
    _training_done = False

    print(f"\n🚀 Training from step {start_step} → {max_steps}...\n")

    num_epochs = max(1, (max_steps - start_step) * grad_accum // max(len(train_loader), 1) + 1)

    for epoch in range(num_epochs):
        loss_meter.reset()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in pbar:
            plans = batch["plan"].to(device)
            task_texts = batch["task_descriptions"]
            num_agents_batch = batch["num_agents"].to(device)
            plan_lengths = batch["plan_lengths"].to(device)
            agent_states = torch.randn(
                plans.shape[0], max_agents_cfg, 128, device=device
            )
            capabilities = torch.ones(
                plans.shape[0], max_agents_cfg, VOCAB_SIZE, device=device
            )

            # Encode condition (frozen encoder, no grad)
            with torch.no_grad():
                condition = model.plan_encoder(task_texts, agent_states)

            # Forward
            mask_result = model.role_masker(
                capabilities, plans, num_agents_batch, plan_lengths
            )
            combined_mask = mask_result["combined_mask"]
            validity_mask = mask_result["validity_mask"]
            loss = model.plan_diffusion.training_loss(
                plans, condition, role_mask=combined_mask, attn_mask=validity_mask
            )

            # Scale for gradient accumulation
            (loss / grad_accum).backward()
            loss_meter.update(loss.item())

            accum_count += 1
            if accum_count >= grad_accum:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), train_cfg.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accum_count = 0
                global_step += 1

                pbar.set_postfix(
                    loss=f"{loss_meter.avg:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    step=global_step,
                )

                if use_wandb and global_step % 100 == 0:
                    try:
                        import wandb

                        wandb.log(
                            {
                                "train/loss": loss_meter.avg,
                                "train/lr": scheduler.get_last_lr()[0],
                            },
                            step=global_step,
                        )
                    except Exception:
                        pass

                # Periodic checkpoint
                if (
                    not args.smoke_test
                    and global_step % train_cfg.checkpoint.save_every == 0
                ):
                    save_checkpoint(global_step, loss_meter.avg, reason="periodic")

                # Graceful shutdown on SIGTERM/SIGINT
                if _interrupted:
                    if not args.smoke_test:
                        save_checkpoint(global_step, loss_meter.avg, reason="interrupted")
                    print(f"🛑 Stopped at step {global_step}. Resume will be automatic next launch.")
                    _training_done = True
                    break

                if global_step >= max_steps:
                    _training_done = True
                    break

        if _training_done:
            break

    elapsed = time.time() - start_time
    used = torch.cuda.max_memory_allocated(gpu_id) / 1e9
    total_mem = torch.cuda.get_device_properties(gpu_id).total_memory / 1e9

    print(f"\n{'=' * 60}")
    print(f"Training {'interrupted' if _interrupted else 'complete'}! (GPU {gpu_id})")
    print(f"  Steps: {global_step}, Loss: {loss_meter.avg:.4f}")
    print(f"  Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"  Throughput: {global_step / max(elapsed, 1):.2f} steps/s")
    print(f"  GPU mem: {used:.1f}/{total_mem:.1f} GB")
    print("=" * 60)

    if args.smoke_test:
        print("\n✅ SMOKE TEST PASSED!")
    elif not _interrupted:
        final_path = output_dir / "stage1_final.pt"
        torch.save(model.state_dict(), final_path)
        print(f"Final model: {final_path}")

    if use_wandb:
        try:
            import wandb

            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
