"""
Stage 2: Train Plan-to-Dialogue Decoder (diffusion + encoder frozen).

Loads the averaged Stage 1 model, freezes the diffusion model and encoder,
then trains the PlanToDialogueDecoder on (plan → utterance) pairs.

Usage:
    python scripts/train_stage2.py --config configs/default.yaml \
        --stage1_ckpt experiments/checkpoints/stage1_averaged.pt \
        --gpu 4

    # Multi-GPU (4 independent processes):
    bash scripts/train_stage2_parallel.sh
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
    parser = argparse.ArgumentParser(description="Train DiffuseAlign Stage 2 (Decoder)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--stage1_ckpt", type=str, default="experiments/checkpoints/stage1_averaged.pt")
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


class DecoderTrainingDataset(Dataset):
    """
    Wraps MultiAgentTrajectoryDataset and adds per-agent utterance targets.

    For each trajectory, creates plan → utterance pairs for each agent,
    flattening so each sample is (plan_for_one_agent, target_utterance).
    """

    def __init__(self, base_dataset: MultiAgentTrajectoryDataset):
        self.base_dataset = base_dataset
        # Pre-compute (traj_idx, agent_idx) pairs for all agents with valid steps
        self.samples = []
        for traj_idx, traj in enumerate(base_dataset.trajectories):
            # Collect per-agent utterances
            agent_utterances: dict[int, list[str]] = {}
            for step in traj.steps:
                aid = step.agent_id
                if aid >= base_dataset.max_agents:
                    continue
                if aid not in agent_utterances:
                    agent_utterances[aid] = []
                if step.utterance:
                    agent_utterances[aid].append(step.utterance)

            for aid, utts in agent_utterances.items():
                if utts:
                    # Join all utterances for this agent into one target string
                    target = " | ".join(utts)
                    self.samples.append((traj_idx, aid, target))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        traj_idx, agent_id, target_utterance = self.samples[idx]
        base_item = self.base_dataset[traj_idx]
        return {
            **base_item,
            "agent_id": torch.tensor(agent_id, dtype=torch.long),
            "target_utterance": target_utterance,
        }


def collate_decoder(batch):
    """Collate function for decoder training — includes utterances."""
    return {
        "plan": torch.stack([b["plan"] for b in batch]),
        "plan_actions": torch.stack([b["plan_actions"] for b in batch]),
        "validity_mask": torch.stack([b["validity_mask"] for b in batch]),
        "task_descriptions": [b["task_description"] for b in batch],
        "num_agents": torch.stack([b["num_agents"] for b in batch]),
        "plan_lengths": torch.stack([b["plan_length"] for b in batch]),
        "success": torch.stack([b["success"] for b in batch]),
        "agent_ids": torch.stack([b["agent_id"] for b in batch]),
        "target_utterances": [b["target_utterance"] for b in batch],
    }


def find_latest_checkpoint(output_dir: Path, prefix: str = "stage2") -> str | None:
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


def cleanup_old_checkpoints(output_dir: Path, keep_last: int = 3, prefix: str = "stage2"):
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
    dec_cfg = cfg.training.decoder_training

    gpu_id = args.gpu
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    # Hyperparams
    lr = args.lr or dec_cfg.lr
    batch_size = args.batch_size or dec_cfg.batch_size
    grad_accum = args.grad_accum
    num_epochs = args.num_epochs or dec_cfg.num_epochs
    if args.smoke_test:
        num_epochs = 1
        batch_size = 4
        grad_accum = 1

    print("=" * 60)
    mode = "🔥 SMOKE TEST" if args.smoke_test else "🚀 FULL TRAINING"
    print(f"Stage 2: Plan-to-Dialogue Decoder — {mode}")
    print(f"  GPU: {gpu_id}, LR: {lr}, Batch: {batch_size}, Accum: {grad_accum}")
    print(f"  Epochs: {num_epochs}")
    print("=" * 60)

    # Output directory
    run_name = args.run_name or f"stage2_gpu{gpu_id}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    print("\nBuilding model...")
    model = build_model(cfg)

    # Load Stage 1 weights
    stage1_path = args.stage1_ckpt
    if os.path.exists(stage1_path):
        print(f"Loading Stage 1 checkpoint: {stage1_path}")
        state = torch.load(stage1_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        # Load with strict=False since stage1 may not have decoder weights
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Loaded. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}")
        if missing:
            print(f"  Missing (expected — decoder/guidance): {missing[:5]}...")
    else:
        print(f"⚠️  Stage 1 checkpoint not found: {stage1_path}")
        print("   Training decoder from scratch (no pretrained diffusion).")

    model = model.to(device)

    # Freeze everything except decoder
    for name, param in model.named_parameters():
        if "plan_decoder" not in name:
            param.requires_grad = False

    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    print(f"Total params: {format_params(total)}, Trainable (decoder): {format_params(trainable)}")

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

    # Create a combined dataset
    combined_ds = MultiAgentTrajectoryDataset.__new__(MultiAgentTrajectoryDataset)
    combined_ds.max_agents = cfg.model.diffusion.max_agents
    combined_ds.max_steps = cfg.model.diffusion.max_plan_steps
    combined_ds.plan_dim = cfg.model.diffusion.plan_dim
    combined_ds.action_embed_dim = 256
    combined_ds.action_embedding = torch.nn.Embedding(VOCAB_SIZE, 256)
    combined_ds.trajectories = all_trajectories
    combined_ds.data_dir = Path("data/combined")

    decoder_dataset = DecoderTrainingDataset(combined_ds)
    print(f"Total decoder samples (agent-level): {len(decoder_dataset)}")

    train_loader = DataLoader(
        decoder_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if not args.smoke_test else 0,
        collate_fn=collate_decoder,
        pin_memory=True,
    )

    # Optimizer & scheduler
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    total_steps = num_epochs * len(train_loader) // grad_accum
    warmup_steps = min(500, total_steps // 10)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(1e-7 / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

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

    # Checkpoint helper — save every 250 steps (was 1000, too infrequent)
    save_every = 250
    keep_last = 3

    def save_checkpoint(step, epoch, loss_avg, reason="periodic"):
        ckpt_path = output_dir / f"stage2_step{step}.pt"
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

    # Training loop
    model.train()
    # But keep frozen parts in eval mode for efficiency
    model.plan_diffusion.eval()
    model.plan_encoder.eval()
    model.role_masker.eval()
    model.guidance.eval()

    loss_meter = AverageMeter("loss")
    nll_meter = AverageMeter("nll")
    commit_meter = AverageMeter("commit")
    global_step = start_step
    accum_count = 0
    start_time = time.time()
    _done = False

    max_agents_cfg = cfg.model.diffusion.max_agents

    print(f"\n🚀 Training decoder: {num_epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"   Total steps: ~{total_steps}, Warmup: {warmup_steps}")
    print(f"   Starting from step {start_step}, epoch {start_epoch}\n")

    for epoch in range(start_epoch, num_epochs):
        loss_meter.reset()
        nll_meter.reset()
        commit_meter.reset()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in pbar:
            plans = batch["plan"].to(device)
            target_utterances = batch["target_utterances"]
            agent_ids = batch["agent_ids"].to(device)

            # Create agent_ids as (batch, agents) — expand single agent_id
            batch_size_cur = plans.shape[0]
            # We need (batch, agents) for the decoder
            agent_ids_expanded = torch.zeros(
                batch_size_cur, max_agents_cfg, dtype=torch.long, device=device
            )
            for i in range(max_agents_cfg):
                agent_ids_expanded[:, i] = i

            # Forward through decoder
            decoder_out = model.plan_decoder(
                plan=plans,
                target_utterances=target_utterances,
                agent_ids=agent_ids_expanded,
            )

            loss = decoder_out["total_loss"]
            nll = decoder_out["nll_loss"]
            commit = decoder_out["commit_loss"]

            # Gradient accumulation
            (loss / grad_accum).backward()
            loss_meter.update(loss.item())
            nll_meter.update(nll.item())
            commit_meter.update(commit.item())

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
                    nll=f"{nll_meter.avg:.4f}",
                    commit=f"{commit_meter.avg:.4f}",
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

        print(f"  Epoch {epoch + 1}: loss={loss_meter.avg:.4f}, nll={nll_meter.avg:.4f}, "
              f"commit={commit_meter.avg:.4f}")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"Stage 2 {'interrupted' if _interrupted else 'complete'}!")
    print(f"  Steps: {global_step}, NLL: {nll_meter.avg:.4f}")
    print(f"  Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 60)

    if not args.smoke_test and not _interrupted:
        final_path = output_dir / "stage2_final.pt"
        torch.save(model.state_dict(), final_path)
        print(f"Final model: {final_path}")


if __name__ == "__main__":
    main()
