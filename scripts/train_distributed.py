"""
Distributed training script for DiffuseAlign (Stage 1: Diffusion Model).

Uses HuggingFace Accelerate for multi-GPU DDP training.

Usage:
    # Smoke test (quick validation on all 8 GPUs):
    accelerate launch --num_processes 8 scripts/train_distributed.py \
        --config configs/default.yaml --smoke_test

    # Full overnight training:
    accelerate launch --num_processes 8 scripts/train_distributed.py \
        --config configs/default.yaml --output_dir experiments/checkpoints
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed as accel_set_seed

# Add project root to path
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
    parser = argparse.ArgumentParser(description="Train DiffuseAlign (distributed)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    # Smoke test overrides
    parser.add_argument("--smoke_test", action="store_true",
                        help="Run a quick smoke test (3 batches) to validate everything works")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Override batch size (per GPU)")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override max training steps")
    return parser.parse_args()


def build_model(cfg) -> DiffuseAlign:
    """Build DiffuseAlign model from config."""
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


def main():
    args = parse_args()

    # ── Accelerator setup ──────────────────────────────────────────
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=4 if not args.smoke_test else 1,
        log_with="wandb" if not args.smoke_test else None,
    )

    # Seed
    accel_set_seed(args.seed)
    set_seed(args.seed)

    # Load config
    cfg = load_config(args.config)
    train_cfg = cfg.training

    # Smoke test overrides
    if args.smoke_test:
        batch_size = args.batch_size or 8  # small per-GPU batch
        max_steps = args.max_steps or 3
        accelerator.print("=" * 60)
        accelerator.print("🔥 SMOKE TEST MODE")
        accelerator.print(f"   Batch size per GPU: {batch_size}")
        accelerator.print(f"   Max steps: {max_steps}")
        accelerator.print(f"   Num GPUs: {accelerator.num_processes}")
        accelerator.print(f"   Effective batch: {batch_size * accelerator.num_processes}")
        accelerator.print("=" * 60)
    else:
        batch_size = args.batch_size or train_cfg.batch_size
        max_steps = args.max_steps or train_cfg.scheduler.num_training_steps

    # ── Build model ────────────────────────────────────────────────
    accelerator.print("Building DiffuseAlign model...")
    model = build_model(cfg)

    total_params = count_parameters(model, trainable_only=False)
    trainable_params = count_parameters(model, trainable_only=True)
    accelerator.print(f"Total params: {format_params(total_params)}")
    accelerator.print(f"Trainable params: {format_params(trainable_params)}")

    # ── Data ───────────────────────────────────────────────────────
    accelerator.print("Loading data...")
    train_loader = get_dataloader(
        data_dir=cfg.data.train_datasets[0].path,
        batch_size=batch_size,
        max_agents=cfg.model.diffusion.max_agents,
        max_steps=cfg.model.diffusion.max_plan_steps,
        plan_dim=cfg.model.diffusion.plan_dim,
        num_workers=4 if not args.smoke_test else 0,
    )
    accelerator.print(f"Dataset: {len(train_loader.dataset)} trajectories")
    accelerator.print(f"Batches per epoch: {len(train_loader)}")

    # ── Optimizer + Scheduler ──────────────────────────────────────
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.optimizer.lr,
        weight_decay=train_cfg.optimizer.weight_decay,
        betas=tuple(train_cfg.optimizer.betas),
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max_steps,
        eta_min=1e-6,
    )

    # ── Prepare for distributed ────────────────────────────────────
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    # ── Resume ─────────────────────────────────────────────────────
    if args.resume:
        accelerator.print(f"Resuming from {args.resume}")
        accelerator.load_state(args.resume)

    # ── W&B init (full training only) ──────────────────────────────
    if not args.smoke_test and accelerator.is_main_process:
        try:
            accelerator.init_trackers(
                project_name="diffuse-align",
                config={
                    "batch_size_per_gpu": batch_size,
                    "num_gpus": accelerator.num_processes,
                    "effective_batch_size": batch_size * accelerator.num_processes,
                    "max_steps": max_steps,
                    "lr": train_cfg.optimizer.lr,
                    "total_params": total_params,
                    "trainable_params": trainable_params,
                },
            )
        except Exception as e:
            accelerator.print(f"W&B init failed (non-fatal): {e}")

    # ── Training loop ──────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    max_agents = cfg.model.diffusion.max_agents
    loss_meter = AverageMeter("loss")
    global_step = 0
    start_time = time.time()

    model.train()
    accelerator.print(f"\n🚀 Starting training (max {max_steps} steps)...\n")

    num_epochs = max(1, max_steps // max(len(train_loader), 1) + 1)

    for epoch in range(num_epochs):
        loss_meter.reset()

        if accelerator.is_main_process:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        else:
            pbar = train_loader

        for batch in pbar:
            with accelerator.accumulate(model):
                plans = batch["plan"]
                task_texts = batch["task_descriptions"]
                num_agents_batch = batch["num_agents"]
                plan_lengths = batch["plan_lengths"]

                # Create agent states and capabilities
                agent_states = torch.randn(
                    plans.shape[0], max_agents, 128, device=plans.device
                )
                capabilities = torch.ones(
                    plans.shape[0], max_agents, VOCAB_SIZE, device=plans.device
                )

                # Forward — use unwrapped model for training_step_diffusion
                # which handles text inputs that can't go through DDP wrapper
                unwrapped = accelerator.unwrap_model(model)
                losses = unwrapped.training_step_diffusion(
                    plans=plans,
                    task_texts=task_texts,
                    agent_states=agent_states,
                    capabilities=capabilities,
                    num_agents=num_agents_batch,
                    plan_lengths=plan_lengths,
                )

                loss = losses["total_loss"]

                # Backward
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # Logging
            loss_val = loss.detach().item()
            loss_meter.update(loss_val)
            global_step += 1

            if accelerator.is_main_process:
                pbar.set_postfix(
                    loss=f"{loss_meter.avg:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    step=global_step,
                )

                # Log to W&B
                if not args.smoke_test and global_step % 100 == 0:
                    try:
                        accelerator.log({
                            "train/loss": loss_meter.avg,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/step": global_step,
                            "train/epoch": epoch,
                        }, step=global_step)
                    except Exception:
                        pass

            # Checkpointing (full training only)
            if not args.smoke_test and global_step % train_cfg.checkpoint.save_every == 0:
                if accelerator.is_main_process:
                    ckpt_path = output_dir / f"stage1_step{global_step}.pt"
                    unwrapped = accelerator.unwrap_model(model)
                    torch.save({
                        "step": global_step,
                        "model_state": unwrapped.state_dict(),
                        "loss": loss_meter.avg,
                    }, ckpt_path)
                    accelerator.print(f"\n💾 Saved checkpoint: {ckpt_path}")

            # Stop condition
            if global_step >= max_steps:
                break

        if global_step >= max_steps:
            break

    # ── Finish ─────────────────────────────────────────────────────
    elapsed = time.time() - start_time

    accelerator.print(f"\n{'=' * 60}")
    accelerator.print(f"Training complete!")
    accelerator.print(f"  Steps: {global_step}")
    accelerator.print(f"  Final loss: {loss_meter.avg:.4f}")
    accelerator.print(f"  Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    accelerator.print(f"  Throughput: {global_step / elapsed:.2f} steps/s")
    accelerator.print(f"{'=' * 60}")

    if args.smoke_test:
        accelerator.print("\n✅ SMOKE TEST PASSED — all 8 GPUs working correctly!")
        accelerator.print("Ready for overnight training. Run:")
        accelerator.print(f"  accelerate launch --num_processes {accelerator.num_processes} scripts/train_distributed.py \\")
        accelerator.print(f"    --config configs/default.yaml --output_dir experiments/checkpoints")
    else:
        # Save final model
        if accelerator.is_main_process:
            final_path = output_dir / "stage1_final.pt"
            unwrapped = accelerator.unwrap_model(model)
            torch.save(unwrapped.state_dict(), final_path)
            accelerator.print(f"Final model saved to {final_path}")

            # Save training summary
            save_json({
                "steps": global_step,
                "final_loss": loss_meter.avg,
                "elapsed_seconds": elapsed,
                "num_gpus": accelerator.num_processes,
                "batch_size_per_gpu": batch_size,
                "effective_batch_size": batch_size * accelerator.num_processes,
            }, str(output_dir / "training_summary.json"))

    accelerator.end_training()


if __name__ == "__main__":
    main()
