"""Quick diagnostic to find the NaN source in DiffuseAlign training."""
import sys, torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diffuse_align import DiffuseAlign
from src.dataset import get_dataloader
from src.agents import VOCAB_SIZE
from src.utils import set_seed, load_config

torch.autograd.set_detect_anomaly(True)

set_seed(46)  # seed 42 + gpu_id 4, matching overnight run
cfg = load_config("configs/default.yaml")
train_cfg = cfg.training

device = torch.device("cuda:4")
torch.cuda.set_device(device)

model_cfg = cfg.model
model = DiffuseAlign(
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
).to(device)
model.train()

loader = get_dataloader(
    data_dir=cfg.data.train_datasets[0].path,
    batch_size=8,
    max_agents=cfg.model.diffusion.max_agents,
    max_steps=cfg.model.diffusion.max_plan_steps,
    plan_dim=cfg.model.diffusion.plan_dim,
    num_workers=0,
)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999),
)

max_agents_cfg = cfg.model.diffusion.max_agents
grad_accum = 8
accum_count = 0
global_step = 0

print(f"Running diagnostic: {len(loader)} micro-batches, grad_accum={grad_accum}")
print(f"Expected NaN around step 64 (micro-batch ~512)")

for epoch in range(2):
    for batch_idx, batch in enumerate(loader):
        plans = batch["plan"].to(device)
        task_texts = batch["task_descriptions"]
        num_agents_batch = batch["num_agents"].to(device)
        plan_lengths = batch["plan_lengths"].to(device)
        agent_states = torch.randn(plans.shape[0], max_agents_cfg, 128, device=device)
        capabilities = torch.ones(plans.shape[0], max_agents_cfg, VOCAB_SIZE, device=device)

        with torch.no_grad():
            condition = model.plan_encoder(task_texts, agent_states)

        mask_result = model.role_masker(capabilities, plans, num_agents_batch, plan_lengths)
        combined_mask = mask_result["combined_mask"]
        validity_mask = mask_result["validity_mask"]

        # Check for NaN in inputs
        if torch.isnan(plans).any():
            print(f"  ⚠️ NaN in plans! batch {batch_idx}")
        if torch.isnan(condition).any():
            print(f"  ⚠️ NaN in condition! batch {batch_idx}")
        if torch.isnan(combined_mask).any():
            print(f"  ⚠️ NaN in mask! batch {batch_idx}")

        loss = model.plan_diffusion.training_loss(
            plans, condition, role_mask=combined_mask, attn_mask=validity_mask
        )

        if torch.isnan(loss):
            print(f"\n🔴 NaN loss at epoch {epoch}, micro-batch {batch_idx}, step {global_step}")
            print(f"   plans: min={plans.min():.4f} max={plans.max():.4f} has_nan={torch.isnan(plans).any()}")
            print(f"   condition: min={condition.min():.4f} max={condition.max():.4f}")
            print(f"   mask sum={combined_mask.sum():.0f}")

            # Check model params for NaN
            nan_params = []
            for n, p in model.named_parameters():
                if torch.isnan(p).any():
                    nan_params.append(n)
            if nan_params:
                print(f"   NaN params: {nan_params[:10]}")
            else:
                print(f"   No NaN in model parameters!")

            # Check model params for Inf
            inf_params = []
            for n, p in model.named_parameters():
                if torch.isinf(p).any():
                    inf_params.append(n)
            if inf_params:
                print(f"   Inf params: {inf_params[:10]}")

            # Check param magnitudes
            for n, p in model.named_parameters():
                if p.abs().max() > 1e6:
                    print(f"   HUGE: {n} max={p.abs().max():.1f}")

            break

        (loss / grad_accum).backward()

        accum_count += 1
        if accum_count >= grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            accum_count = 0
            global_step += 1
            if global_step % 10 == 0:
                print(f"  step {global_step}: loss={loss.item():.4f}")
    else:
        print(f"Epoch {epoch+1} done, step={global_step}")
        continue
    break

if not torch.isnan(loss):
    print(f"\n✅ No NaN detected in {global_step} steps")
