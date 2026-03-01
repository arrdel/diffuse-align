"""
Training script for DiffuseAlign.

3-stage training pipeline:
    Stage 1: Train PlanDiffusion (main denoiser) — most compute-heavy
    Stage 2: Train PlanDecoder (plan → dialogue) — freeze diffusion
    Stage 3: Train guidance classifiers — freeze diffusion + decoder

Usage:
    python scripts/train.py --config configs/default.yaml --stage 1
    python scripts/train.py --config configs/default.yaml --stage 2
    python scripts/train.py --config configs/default.yaml --stage 3
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffuse_align import DiffuseAlign
from src.dataset import get_dataloader, collate_trajectories
from src.agents import AgentTeam, VOCAB_SIZE
from src.utils import (
    set_seed,
    load_config,
    save_json,
    count_parameters,
    format_params,
    get_device,
    AverageMeter,
    EarlyStopping,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DiffuseAlign")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from")
    parser.add_argument("--output_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
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


def train_stage1(model, cfg, device, output_dir):
    """Stage 1: Train diffusion model."""
    print("=" * 60)
    print("Stage 1: Training Plan Diffusion Model")
    print(f"Trainable params: {format_params(count_parameters(model))}")
    print("=" * 60)

    train_cfg = cfg.training
    data_cfg = cfg.data

    # Data loader
    train_loader = get_dataloader(
        data_dir=data_cfg.train_datasets[0].path,
        batch_size=train_cfg.batch_size,
        max_agents=cfg.model.diffusion.max_agents,
        max_steps=cfg.model.diffusion.max_plan_steps,
        plan_dim=cfg.model.diffusion.plan_dim,
    )

    # Optimizer
    optimizer = AdamW(
        model.plan_diffusion.parameters(),
        lr=train_cfg.optimizer.lr,
        weight_decay=train_cfg.optimizer.weight_decay,
        betas=tuple(train_cfg.optimizer.betas),
    )

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_cfg.scheduler.num_training_steps,
        eta_min=1e-6,
    )

    # Training loop
    loss_meter = AverageMeter("loss")
    global_step = 0
    num_epochs = train_cfg.scheduler.num_training_steps // max(len(train_loader), 1) + 1

    model.train()
    for epoch in range(num_epochs):
        loss_meter.reset()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in pbar:
            plans = batch["plan"].to(device)
            task_texts = batch["task_descriptions"]
            num_agents = batch["num_agents"].to(device)
            plan_lengths = batch["plan_lengths"].to(device)

            # Create dummy agent states and capabilities for now
            agent_states = torch.randn(
                plans.shape[0], cfg.model.diffusion.max_agents, 128, device=device
            )
            capabilities = torch.ones(
                plans.shape[0], cfg.model.diffusion.max_agents, VOCAB_SIZE, device=device
            )

            # Forward
            losses = model.training_step_diffusion(
                plans=plans,
                task_texts=task_texts,
                agent_states=agent_states,
                capabilities=capabilities,
                num_agents=num_agents,
                plan_lengths=plan_lengths,
            )

            loss = losses["total_loss"]

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            # Logging
            loss_meter.update(loss.item())
            global_step += 1

            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

            # Checkpointing
            if global_step % train_cfg.checkpoint.save_every == 0:
                ckpt_path = Path(output_dir) / f"stage1_step{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": loss_meter.avg,
                }, ckpt_path)
                print(f"\nSaved checkpoint: {ckpt_path}")

            if global_step >= train_cfg.scheduler.num_training_steps:
                break

        if global_step >= train_cfg.scheduler.num_training_steps:
            break

    # Final save
    final_path = Path(output_dir) / "stage1_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Stage 1 complete. Final model saved to {final_path}")
    print(f"Final loss: {loss_meter.avg:.4f}")


def train_stage2(model, cfg, device, output_dir):
    """Stage 2: Train plan-to-dialogue decoder (diffusion frozen)."""
    print("=" * 60)
    print("Stage 2: Training Plan-to-Dialogue Decoder")
    print("=" * 60)

    # Freeze diffusion and encoder
    for param in model.plan_diffusion.parameters():
        param.requires_grad = False
    for param in model.plan_encoder.parameters():
        param.requires_grad = False

    decoder_params = count_parameters(model.plan_decoder)
    print(f"Trainable decoder params: {format_params(decoder_params)}")

    dec_cfg = cfg.training.decoder_training

    optimizer = AdamW(
        model.plan_decoder.parameters(),
        lr=dec_cfg.lr,
    )

    # Training loop (simplified — in practice, need plan-utterance pairs)
    print("Stage 2 training requires plan-utterance paired data.")
    print("Skipping for now — will be implemented with data collection.")

    final_path = Path(output_dir) / "stage2_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Stage 2 placeholder saved to {final_path}")


def train_stage3(model, cfg, device, output_dir):
    """Stage 3: Train guidance classifiers (diffusion + decoder frozen)."""
    print("=" * 60)
    print("Stage 3: Training Guidance Classifiers")
    print("=" * 60)

    # Freeze everything except guidance
    for name, param in model.named_parameters():
        if "guidance" not in name:
            param.requires_grad = False

    guidance_params = count_parameters(model.guidance)
    print(f"Trainable guidance params: {format_params(guidance_params)}")

    print("Stage 3 training requires labeled trajectory data.")
    print("Skipping for now — will be implemented with data collection.")

    final_path = Path(output_dir) / "stage3_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Stage 3 placeholder saved to {final_path}")


def main():
    args = parse_args()
    set_seed(args.seed)

    # Load config
    cfg = load_config(args.config)

    # Device
    device = torch.device(args.device) if args.device else get_device()
    print(f"Using device: {device}")

    # Build model
    model = build_model(cfg)
    model = model.to(device)

    total_params = count_parameters(model, trainable_only=False)
    print(f"Total model parameters: {format_params(total_params)}")

    # Resume if needed
    if args.resume:
        print(f"Resuming from {args.resume}")
        state = torch.load(args.resume, map_location=device)
        if isinstance(state, dict) and "model_state" in state:
            model.load_state_dict(state["model_state"])
        else:
            model.load_state_dict(state)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    if args.stage == 1:
        train_stage1(model, cfg, device, str(output_dir))
    elif args.stage == 2:
        train_stage2(model, cfg, device, str(output_dir))
    elif args.stage == 3:
        train_stage3(model, cfg, device, str(output_dir))


if __name__ == "__main__":
    main()
