"""
Stage 2 v3: Train Plan-to-Dialogue Decoder on DIFFUSION-GENERATED plans.

Critical fix: Previous Stage 2 training used dataset action embeddings as plans,
but inference uses diffusion-generated plans — completely different distributions.
This script generates plans using the frozen diffusion model and trains the decoder
on those, ensuring train/inference distribution match.

Pipeline:
  1. Load frozen Stage 1 model (diffusion + encoder)
  2. For each training trajectory:
     a. Encode task + agent states → condition
     b. Run diffusion sampling → plan (same as inference)
     c. Pair plan with ground-truth utterances
  3. Train decoder on (diffusion_plan, utterance) pairs

Usage:
    CUDA_VISIBLE_DEVICES=4 python scripts/train_stage2_v3.py --gpu 0
    CUDA_VISIBLE_DEVICES=5 python scripts/train_stage2_v3.py --gpu 0 --seed 43
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffuse_align import DiffuseAlign
from src.dataset import MultiAgentTrajectoryDataset, Trajectory, TrajectoryStep
from src.agents import AgentTeam, AGENT_ARCHETYPES, VOCAB_SIZE, ACTION_VOCAB
from src.utils import (
    set_seed,
    load_config,
    count_parameters,
    format_params,
    AverageMeter,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train DiffuseAlign Stage 2 v3 (Decoder on diffusion plans)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--stage1_ckpt", type=str, default="experiments/checkpoints/stage1_averaged.pt")
    parser.add_argument("--output_dir", type=str, default="experiments/checkpoints")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--plan_batch_size", type=int, default=16, help="Batch size for plan generation")
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--warmup_frac", type=float, default=0.05)
    parser.add_argument("--skip_plan_cache", action="store_true", help="Regenerate plans even if cache exists")
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


# ─── Plan Cache: Generate plans from diffusion model ─────────────────────────

def generate_plan_cache(
    model: DiffuseAlign,
    trajectories: List[Trajectory],
    device: torch.device,
    max_agents: int,
    plan_batch_size: int = 16,
    cache_path: Optional[str] = None,
) -> List[torch.Tensor]:
    """
    Generate diffusion plans for all training trajectories.
    
    For each trajectory, runs the frozen diffusion model with the trajectory's
    task description and dummy agent states to produce a plan tensor.
    Returns a list of plan tensors on CPU.
    """
    if cache_path and os.path.exists(cache_path):
        print(f"  Loading plan cache from {cache_path}")
        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if len(cache) == len(trajectories):
            print(f"  ✅ Loaded {len(cache)} cached plans")
            return cache
        else:
            print(f"  ⚠️  Cache size mismatch ({len(cache)} vs {len(trajectories)}), regenerating")

    model.eval()
    plans = []
    
    # Build a default agent team for generating plans
    # Use first 2 archetypes as default since most trajectories have 2 agents
    archetype_names = list(AGENT_ARCHETYPES.keys())
    
    print(f"  Generating plans for {len(trajectories)} trajectories (batch_size={plan_batch_size})...")
    
    for start_idx in tqdm(range(0, len(trajectories), plan_batch_size), desc="  Gen plans"):
        end_idx = min(start_idx + plan_batch_size, len(trajectories))
        batch_trajs = trajectories[start_idx:end_idx]
        batch_size = len(batch_trajs)
        
        task_texts = [t.task_description for t in batch_trajs]
        
        # Build agent states and capabilities for each trajectory
        agent_states_list = []
        capabilities_list = []
        for traj in batch_trajs:
            n_agents = min(traj.num_agents, max_agents)
            # Use the actual roles from the trajectory
            roles = traj.agent_roles[:n_agents]
            # Pad roles if fewer than max_agents
            while len(roles) < max_agents:
                roles.append(roles[0] if roles else archetype_names[0])
            
            team = AgentTeam.from_archetypes(roles[:max_agents])
            agent_states_list.append(team.states_tensor())
            capabilities_list.append(team.capabilities_tensor())
        
        agent_states = torch.stack(agent_states_list).to(device)  # (B, A, state_dim)
        capabilities = torch.stack(capabilities_list).to(device)  # (B, A, num_actions)
        
        with torch.no_grad():
            plan = model.generate_plan(
                task_texts=task_texts,
                agent_states=agent_states,
                capabilities=capabilities,
                use_guidance=True,
            )  # (B, A, S, plan_dim)
        
        # Store each plan individually on CPU
        for i in range(batch_size):
            plans.append(plan[i].cpu())
    
    # Save cache
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(plans, cache_path)
        size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        print(f"  💾 Saved plan cache: {cache_path} ({size_mb:.0f} MB)")
    
    return plans


# ─── Dataset: pairs diffusion plans with utterances ──────────────────────────

class DiffusionPlanDecoderDataset(Dataset):
    """
    Dataset that pairs diffusion-generated plans with ground-truth utterances.
    
    Each sample: (plan_for_one_agent, target_utterance_string)
    where plan comes from the diffusion model, not from action embeddings.
    """
    
    def __init__(
        self,
        trajectories: List[Trajectory],
        plans: List[torch.Tensor],
        max_agents: int = 4,
    ):
        assert len(trajectories) == len(plans)
        self.trajectories = trajectories
        self.plans = plans  # List of (agents, steps, plan_dim) tensors
        self.max_agents = max_agents
        
        # Pre-compute (traj_idx, agent_idx, target_utterance) samples
        self.samples = []
        for traj_idx, traj in enumerate(trajectories):
            # Collect per-agent utterances
            agent_utterances: Dict[int, List[str]] = {}
            for step in traj.steps:
                aid = step.agent_id
                if aid >= max_agents:
                    continue
                if aid not in agent_utterances:
                    agent_utterances[aid] = []
                if step.utterance and step.utterance.strip():
                    agent_utterances[aid].append(step.utterance.strip())
            
            for aid, utts in agent_utterances.items():
                if utts:
                    target = " | ".join(utts)
                    self.samples.append((traj_idx, aid, target))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        traj_idx, agent_id, target_utterance = self.samples[idx]
        plan = self.plans[traj_idx]  # (agents, steps, plan_dim)
        
        return {
            "plan": plan,  # (agents, steps, plan_dim) — from diffusion model
            "agent_id": torch.tensor(agent_id, dtype=torch.long),
            "target_utterance": target_utterance,
            "traj_idx": traj_idx,
        }


def collate_diffusion_plans(batch):
    """Collate for diffusion plan decoder training."""
    return {
        "plan": torch.stack([b["plan"] for b in batch]),  # (B, agents, steps, plan_dim)
        "agent_ids": torch.stack([b["agent_id"] for b in batch]),  # (B,)
        "target_utterances": [b["target_utterance"] for b in batch],
    }


# ─── Training ────────────────────────────────────────────────────────────────

def cleanup_old_checkpoints(output_dir: Path, keep_last: int = 3, prefix: str = "stage2v3"):
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


def find_latest_checkpoint(output_dir: Path, prefix: str = "stage2v3") -> Optional[str]:
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


def main():
    args = parse_args()
    set_seed(args.seed)
    
    cfg = load_config(args.config)
    mc = cfg.model
    
    gpu_id = args.gpu
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    
    # Hyperparams
    lr = args.lr
    batch_size = args.batch_size
    grad_accum = args.grad_accum
    num_epochs = args.num_epochs
    max_agents = mc.diffusion.max_agents
    
    if args.smoke_test:
        num_epochs = 1
        batch_size = 4
        grad_accum = 1
    
    print("=" * 60)
    mode = "🔥 SMOKE TEST" if args.smoke_test else "🚀 FULL TRAINING"
    print(f"Stage 2 v3: Decoder on Diffusion Plans — {mode}")
    print(f"  GPU: {gpu_id}, LR: {lr}, Batch: {batch_size}, Accum: {grad_accum}")
    print(f"  Epochs: {num_epochs}, Seed: {args.seed}")
    print("=" * 60)
    
    # Output directory
    run_name = args.run_name or f"stage2v3_gpu{gpu_id}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.output_dir) / f"{run_name}.log"
    
    # Build model
    print("\nBuilding model...")
    model = build_model(cfg)
    
    # Load Stage 1 weights (diffusion + encoder)
    stage1_path = args.stage1_ckpt
    if os.path.exists(stage1_path):
        print(f"Loading Stage 1 checkpoint: {stage1_path}")
        state = torch.load(stage1_path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "model_state" in state:
            state = state["model_state"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"  Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print(f"⚠️  Stage 1 not found: {stage1_path}. Diffusion will be random!")
    
    model = model.to(device)
    
    # Load trajectories
    print("\nLoading trajectories...")
    all_trajectories = []
    for ds_cfg in cfg.data.train_datasets:
        ds = MultiAgentTrajectoryDataset(
            data_dir=ds_cfg.path,
            max_agents=max_agents,
            max_steps=mc.diffusion.max_plan_steps,
            plan_dim=mc.diffusion.plan_dim,
        )
        all_trajectories.extend(ds.trajectories)
        print(f"  {ds_cfg.name}: {len(ds.trajectories)} trajectories")
    
    if args.smoke_test:
        all_trajectories = all_trajectories[:50]
        print(f"  Smoke test: using {len(all_trajectories)} trajectories")
    
    print(f"Total trajectories: {len(all_trajectories)}")
    
    # Generate plan cache using diffusion model
    print("\nGenerating diffusion plans for training data...")
    cache_path = str(Path(args.output_dir) / "plan_cache_v3.pt")
    if args.smoke_test:
        cache_path = None  # Don't cache smoke test
    
    plans = generate_plan_cache(
        model=model,
        trajectories=all_trajectories,
        device=device,
        max_agents=max_agents,
        plan_batch_size=args.plan_batch_size,
        cache_path=cache_path if not args.skip_plan_cache else None,
    )
    
    # Verify distribution
    sample_plan = plans[0]
    print(f"\n  Plan shape: {sample_plan.shape}")
    print(f"  Plan stats: mean={sample_plan.mean():.4f}, std={sample_plan.std():.4f}")
    
    # Build decoder dataset
    print("\nBuilding decoder dataset...")
    decoder_dataset = DiffusionPlanDecoderDataset(
        trajectories=all_trajectories,
        plans=plans,
        max_agents=max_agents,
    )
    print(f"  Total decoder samples (agent-level): {len(decoder_dataset)}")
    
    # Show sample
    sample = decoder_dataset[0]
    print(f"  Sample plan shape: {sample['plan'].shape}")
    print(f"  Sample utterance: {sample['target_utterance'][:100]}...")
    
    train_loader = DataLoader(
        decoder_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2 if not args.smoke_test else 0,
        collate_fn=collate_diffusion_plans,
        pin_memory=True,
    )
    
    # Freeze everything except decoder
    for name, param in model.named_parameters():
        if "plan_decoder" not in name:
            param.requires_grad = False
    
    trainable = count_parameters(model, trainable_only=True)
    total = count_parameters(model, trainable_only=False)
    print(f"\nTotal params: {format_params(total)}, Trainable (decoder): {format_params(trainable)}")
    
    # Optimizer & scheduler
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )
    total_steps = num_epochs * len(train_loader) // grad_accum
    warmup_steps = max(50, int(total_steps * args.warmup_frac))
    
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(1e-7 / lr, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Resume logic
    start_step = 0
    start_epoch = 0
    resume_path = find_latest_checkpoint(output_dir)
    if resume_path:
        print(f"\nAuto-resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        if isinstance(ckpt, dict) and "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
            if "optimizer_state" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            if "scheduler_state" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            start_step = ckpt.get("step", 0)
            start_epoch = ckpt.get("epoch", 0)
        print(f"  ✅ Resumed from step {start_step}, epoch {start_epoch}")
    
    # Checkpoint helper
    save_every = 500
    keep_last = 3
    
    def save_checkpoint(step, epoch, loss_avg, reason="periodic"):
        ckpt_path = output_dir / f"stage2v3_step{step}.pt"
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
        # Also log
        with open(log_path, "a") as f:
            f.write(f"Checkpoint: step={step}, epoch={epoch}, loss={loss_avg:.4f}, reason={reason}\n")
    
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
    best_nll = float("inf")
    
    print(f"\n🚀 Training decoder: {num_epochs} epochs, {len(train_loader)} batches/epoch")
    print(f"   Total steps: ~{total_steps}, Warmup: {warmup_steps}")
    print(f"   Starting from step {start_step}, epoch {start_epoch}\n")
    
    with open(log_path, "a") as f:
        f.write(f"\n{'='*60}\nStarting Stage 2 v3 training at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"  LR: {lr}, Batch: {batch_size}, Accum: {grad_accum}, Epochs: {num_epochs}\n")
        f.write(f"  Total steps: ~{total_steps}, Warmup: {warmup_steps}\n")
        f.write(f"  Decoder samples: {len(decoder_dataset)}\n{'='*60}\n")
    
    for epoch in range(start_epoch, num_epochs):
        loss_meter.reset()
        nll_meter.reset()
        commit_meter.reset()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for batch in pbar:
            plans_batch = batch["plan"].to(device)  # (B, agents, steps, plan_dim)
            target_utterances = batch["target_utterances"]
            agent_ids = batch["agent_ids"].to(device)
            
            batch_size_cur = plans_batch.shape[0]
            
            # Create agent_ids as (batch, agents)
            agent_ids_expanded = torch.zeros(
                batch_size_cur, max_agents, dtype=torch.long, device=device
            )
            for i in range(max_agents):
                agent_ids_expanded[:, i] = i
            
            # Forward through decoder
            decoder_out = model.plan_decoder(
                plan=plans_batch,
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
        
        epoch_nll = nll_meter.avg
        epoch_msg = (f"  Epoch {epoch + 1}: loss={loss_meter.avg:.4f}, nll={epoch_nll:.4f}, "
                     f"commit={commit_meter.avg:.4f}")
        print(epoch_msg)
        
        with open(log_path, "a") as f:
            f.write(epoch_msg + "\n")
        
        # Save best model
        if epoch_nll < best_nll:
            best_nll = epoch_nll
            best_path = output_dir / "stage2v3_best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  🏆 New best NLL: {best_nll:.4f} → {best_path}")
        
        # End-of-epoch checkpoint
        if not args.smoke_test:
            save_checkpoint(global_step, epoch + 1, loss_meter.avg, reason=f"epoch_{epoch+1}")
    
    elapsed = time.time() - start_time
    final_msg = (f"\n{'=' * 60}\n"
                 f"Stage 2 v3 {'interrupted' if _interrupted else 'complete'}!\n"
                 f"  Steps: {global_step}, Best NLL: {best_nll:.4f}\n"
                 f"  Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)\n"
                 f"{'=' * 60}")
    print(final_msg)
    
    with open(log_path, "a") as f:
        f.write(final_msg + "\n")
    
    if not args.smoke_test and not _interrupted:
        final_path = output_dir / "stage2v3_final.pt"
        torch.save(model.state_dict(), final_path)
        print(f"Final model: {final_path}")


if __name__ == "__main__":
    main()
