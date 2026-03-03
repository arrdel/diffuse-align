"""
Multi-GPU training script for DiffuseAlign (Stage 1: Diffusion Model).

Uses manual multi-GPU parallelism: the batch is chunked, each GPU runs
forward+backward on its chunk, and gradients are averaged on the primary GPU.

This avoids nn.DataParallel and NCCL DDP, which SIGSEGV on kernel 4.18.

Usage:
    # Smoke test:
    python scripts/train_multigpu.py --config configs/default.yaml --smoke_test

    # Full overnight training:
    python scripts/train_multigpu.py --config configs/default.yaml \
        --output_dir experiments/checkpoints
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
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
    parser = argparse.ArgumentParser(description="Train DiffuseAlign (multi-GPU)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--batch_size", type=int, default=None)
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


class MultiGPUTrainer:
    """
    Manual multi-GPU training via model replication + gradient averaging.

    - plan_encoder runs only on GPU 0 (text → condition).
    - plan_diffusion + role_masker are replicated to all GPUs.
    - Batch is chunked, each GPU does forward+backward.
    - Gradients are averaged back on GPU 0.
    """

    def __init__(self, model: DiffuseAlign, gpu_ids: list):
        self.model = model
        self.gpu_ids = gpu_ids
        self.primary = torch.device(f"cuda:{gpu_ids[0]}")
        self.num_gpus = len(gpu_ids)

        # Replicate trainable submodules to non-primary GPUs
        self.diffusion_replicas = {gpu_ids[0]: model.plan_diffusion}
        self.masker_replicas = {gpu_ids[0]: model.role_masker}

        for gid in gpu_ids[1:]:
            dev = torch.device(f"cuda:{gid}")
            self.diffusion_replicas[gid] = copy.deepcopy(model.plan_diffusion).to(dev)
            self.masker_replicas[gid] = copy.deepcopy(model.role_masker).to(dev)

    @torch.no_grad()
    def _sync_replicas(self):
        """Copy master params → all replicas (no NCCL needed)."""
        for gid in self.gpu_ids[1:]:
            dev = torch.device(f"cuda:{gid}")
            # Sync plan_diffusion parameters
            for p_master, p_replica in zip(
                self.model.plan_diffusion.parameters(),
                self.diffusion_replicas[gid].parameters(),
            ):
                p_replica.data.copy_(p_master.data.to(dev))
            # Sync plan_diffusion buffers
            for b_master, b_replica in zip(
                self.model.plan_diffusion.buffers(),
                self.diffusion_replicas[gid].buffers(),
            ):
                b_replica.data.copy_(b_master.data.to(dev))
            # Sync role_masker parameters
            for p_master, p_replica in zip(
                self.model.role_masker.parameters(),
                self.masker_replicas[gid].parameters(),
            ):
                p_replica.data.copy_(p_master.data.to(dev))
            # Sync role_masker buffers
            for b_master, b_replica in zip(
                self.model.role_masker.buffers(),
                self.masker_replicas[gid].buffers(),
            ):
                b_replica.data.copy_(b_master.data.to(dev))

    def _collect_grads(self):
        """Sum replica grads back to primary model (no NCCL needed)."""
        for gid in self.gpu_ids[1:]:
            for (name, p_master), (_, p_replica) in zip(
                self.model.plan_diffusion.named_parameters(),
                self.diffusion_replicas[gid].named_parameters(),
            ):
                if p_replica.grad is not None:
                    if p_master.grad is None:
                        p_master.grad = p_replica.grad.to(self.primary)
                    else:
                        p_master.grad.add_(p_replica.grad.to(self.primary))

            for (name, p_master), (_, p_replica) in zip(
                self.model.role_masker.named_parameters(),
                self.masker_replicas[gid].named_parameters(),
            ):
                if p_replica.grad is not None:
                    if p_master.grad is None:
                        p_master.grad = p_replica.grad.to(self.primary)
                    else:
                        p_master.grad.add_(p_replica.grad.to(self.primary))

    def forward_backward(self, plans, condition, capabilities, num_agents, plan_lengths):
        """Chunk batch across GPUs, forward+backward, return avg loss."""
        B = plans.shape[0]

        # Sync parameters to replicas
        if self.num_gpus > 1:
            self._sync_replicas()

        # Zero replica grads
        for gid in self.gpu_ids[1:]:
            self.diffusion_replicas[gid].zero_grad()
            self.masker_replicas[gid].zero_grad()

        # Compute chunk sizes
        base_chunk = B // self.num_gpus
        remainder = B % self.num_gpus
        chunks = []
        start = 0
        for i in range(self.num_gpus):
            size = base_chunk + (1 if i < remainder else 0)
            if size > 0:
                chunks.append((start, start + size, self.gpu_ids[i]))
            start += size

        gpu_losses = []
        all_losses = []  # Store loss tensors to keep references alive
        for idx, (s, e, gid) in enumerate(chunks):
            dev = torch.device(f"cuda:{gid}")
            # .clone() ensures no shared storage / view issues between chunks
            c_plans = plans[s:e].clone().to(dev)
            c_cond = condition[s:e].clone().to(dev)
            c_caps = capabilities[s:e].clone().to(dev)
            c_na = num_agents[s:e].clone().to(dev)
            c_pl = plan_lengths[s:e].clone().to(dev)

            diffusion = self.diffusion_replicas[gid]
            masker = self.masker_replicas[gid]

            mask_result = masker(c_caps, c_plans, c_na, c_pl)
            combined_mask = mask_result["combined_mask"]
            loss = diffusion.training_loss(c_plans, c_cond, combined_mask)

            scaled = loss * (e - s) / B
            all_losses.append((loss, scaled))
            print(f"  [DBG] Chunk {idx} GPU {gid}: loss id={id(loss)} grad_fn id={id(loss.grad_fn)} "
                  f"scaled grad_fn id={id(scaled.grad_fn)}")

        # Now backward all at once
        for idx, (loss, scaled) in enumerate(all_losses):
            gid = chunks[idx][2]
            is_last = (idx == len(all_losses) - 1)
            scaled.backward(retain_graph=not is_last)
            print(f"  [DBG] Backward chunk {idx} GPU {gid}: DONE")
            gpu_losses.append(loss.detach().item())

        # Collect gradients from replicas → primary
        if self.num_gpus > 1:
            self._collect_grads()

        return sum(gpu_losses) / len(gpu_losses)


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = load_config(args.config)
    train_cfg = cfg.training

    # GPU setup
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
    else:
        gpu_ids = list(range(torch.cuda.device_count()))
    num_gpus = len(gpu_ids)
    primary_device = torch.device(f"cuda:{gpu_ids[0]}")

    print("=" * 60)
    mode = "🔥 SMOKE TEST" if args.smoke_test else "🚀 FULL TRAINING"
    print(f"{mode}")
    print(f"   GPUs: {gpu_ids} ({num_gpus} total)")

    if args.smoke_test:
        total_batch_size = args.batch_size or (8 * num_gpus)
        max_steps = args.max_steps or 5
    else:
        total_batch_size = args.batch_size or train_cfg.batch_size
        max_steps = args.max_steps or train_cfg.scheduler.num_training_steps

    print(f"   Batch size: {total_batch_size} (≈{total_batch_size // num_gpus}/GPU)")
    print(f"   Max steps: {max_steps}")
    print("=" * 60)

    # Build model
    print("\nBuilding model...")
    model = build_model(cfg)
    total_p = count_parameters(model, trainable_only=False)
    train_p = count_parameters(model, trainable_only=True)
    print(f"Total params: {format_params(total_p)}, Trainable: {format_params(train_p)}")
    model = model.to(primary_device)

    # Multi-GPU trainer
    trainer = MultiGPUTrainer(model, gpu_ids)
    print(f"Manual multi-GPU across {num_gpus} GPUs\n")

    # Data
    train_loader = get_dataloader(
        data_dir=cfg.data.train_datasets[0].path,
        batch_size=total_batch_size,
        max_agents=cfg.model.diffusion.max_agents,
        max_steps=cfg.model.diffusion.max_plan_steps,
        plan_dim=cfg.model.diffusion.plan_dim,
        num_workers=4 if not args.smoke_test else 0,
    )
    print(f"Dataset: {len(train_loader.dataset)} trajectories, {len(train_loader)} batches/epoch")

    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=train_cfg.optimizer.lr,
        weight_decay=train_cfg.optimizer.weight_decay,
        betas=tuple(train_cfg.optimizer.betas),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=1e-6)
    max_agents_cfg = cfg.model.diffusion.max_agents

    # Resume
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=primary_device)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            start_step = ckpt.get("step", 0)
        else:
            model.load_state_dict(ckpt)
        print(f"Resumed from step {start_step}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # W&B
    use_wandb = not args.smoke_test
    if use_wandb:
        try:
            import wandb
            wandb.init(project="diffuse-align", name=f"stage1_{num_gpus}gpu",
                       config={"batch_size": total_batch_size, "num_gpus": num_gpus,
                               "max_steps": max_steps, "lr": train_cfg.optimizer.lr})
        except Exception as e:
            print(f"W&B init failed: {e}")
            use_wandb = False

    # Training loop
    loss_meter = AverageMeter("loss")
    global_step = start_step
    start_time = time.time()
    model.train()
    print(f"\n🚀 Training from step {start_step} → {max_steps}...\n")

    num_epochs = max(1, (max_steps - start_step) // max(len(train_loader), 1) + 1)

    for epoch in range(num_epochs):
        loss_meter.reset()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")

        for batch in pbar:
            plans = batch["plan"].to(primary_device)
            task_texts = batch["task_descriptions"]
            num_agents_batch = batch["num_agents"].to(primary_device)
            plan_lengths = batch["plan_lengths"].to(primary_device)
            agent_states = torch.randn(plans.shape[0], max_agents_cfg, 128, device=primary_device)
            capabilities = torch.ones(plans.shape[0], max_agents_cfg, VOCAB_SIZE, device=primary_device)

            # Condition (text encoder on primary GPU only)
            with torch.no_grad():
                condition = model.plan_encoder(task_texts, agent_states)

            # Forward + backward across all GPUs
            optimizer.zero_grad()
            loss_val = trainer.forward_backward(
                plans, condition, capabilities, num_agents_batch, plan_lengths
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()

            loss_meter.update(loss_val)
            global_step += 1
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}", step=global_step)

            if use_wandb and global_step % 100 == 0:
                try:
                    import wandb
                    wandb.log({"train/loss": loss_meter.avg, "train/lr": scheduler.get_last_lr()[0]}, step=global_step)
                except Exception:
                    pass

            if not args.smoke_test and global_step % train_cfg.checkpoint.save_every == 0:
                ckpt_path = output_dir / f"stage1_step{global_step}.pt"
                torch.save({"step": global_step, "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(), "loss": loss_meter.avg}, ckpt_path)
                print(f"\n💾 Checkpoint: {ckpt_path}")

            if global_step >= max_steps:
                break
        if global_step >= max_steps:
            break

    elapsed = time.time() - start_time
    gpu_mem = []
    for gid in gpu_ids:
        used = torch.cuda.max_memory_allocated(gid) / 1e9
        total = torch.cuda.get_device_properties(gid).total_mem / 1e9
        gpu_mem.append(f"GPU {gid}: {used:.1f}/{total:.1f} GB")

    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"  Steps: {global_step}, Loss: {loss_meter.avg:.4f}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Throughput: {global_step/max(elapsed,1):.2f} steps/s")
    for line in gpu_mem:
        print(f"  {line}")
    print("=" * 60)

    if args.smoke_test:
        print("\n✅ SMOKE TEST PASSED — all GPUs utilized!")
        print(f"For overnight training:\n  python scripts/train_multigpu.py --config configs/default.yaml --output_dir experiments/checkpoints")
    else:
        final_path = output_dir / "stage1_final.pt"
        torch.save(model.state_dict(), final_path)
        print(f"Final model: {final_path}")
        save_json({"steps": global_step, "loss": loss_meter.avg, "elapsed": elapsed,
                    "num_gpus": num_gpus, "gpu_memory": gpu_mem}, str(output_dir / "training_summary.json"))

    if use_wandb:
        try:
            import wandb; wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
