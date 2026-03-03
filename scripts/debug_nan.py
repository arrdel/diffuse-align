"""Quick NaN diagnosis: run 100 steps with grad_accum=8."""
import torch
import sys

sys.path.insert(0, ".")
from src.diffuse_align import DiffuseAlign
from src.dataset import get_dataloader
from src.agents import VOCAB_SIZE
from src.utils import load_config, set_seed
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

set_seed(46)
cfg = load_config("configs/default.yaml")
mc = cfg.model

model = DiffuseAlign(
    plan_dim=mc.diffusion.plan_dim,
    hidden_dim=mc.diffusion.denoiser.hidden_dim,
    num_heads=mc.diffusion.denoiser.num_heads,
    num_layers=mc.diffusion.denoiser.num_layers,
    dropout=mc.diffusion.denoiser.dropout,
    condition_dim=mc.diffusion.denoiser.condition_dim,
    max_agents=mc.diffusion.max_agents,
    max_steps=mc.diffusion.max_plan_steps,
    num_train_timesteps=mc.diffusion.num_train_timesteps,
    num_inference_timesteps=mc.diffusion.num_inference_timesteps,
    unconditional_prob=mc.guidance.unconditional_prob,
    guidance_scale=mc.guidance.guidance_scale,
    task_guidance_weight=mc.guidance.task_completion.weight,
    safety_guidance_weight=mc.guidance.safety.weight,
    efficiency_guidance_weight=mc.guidance.efficiency.weight,
    coordination_guidance_weight=mc.guidance.coordination.weight,
    mask_type=mc.role_masking.mask_type,
).cuda()
model.train()

loader = get_dataloader(
    "data/alfworld_multi", batch_size=8, max_agents=4, max_steps=32, plan_dim=512, num_workers=0
)
optimizer = AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999),
)
scheduler = CosineAnnealingLR(optimizer, T_max=100000, eta_min=1e-6)

grad_accum = 8
accum = 0
global_step = 0
running_loss = 0.0

for epoch in range(3):
    for batch_idx, batch in enumerate(loader):
        plans = batch["plan"].cuda()
        na = batch["num_agents"].cuda()
        pl = batch["plan_lengths"].cuda()
        agent_states = torch.randn(plans.shape[0], 4, 128, device="cuda")
        caps = torch.ones(plans.shape[0], 4, VOCAB_SIZE, device="cuda")
        with torch.no_grad():
            cond = model.plan_encoder(batch["task_descriptions"], agent_states)
        mr = model.role_masker(caps, plans, na, pl)
        loss = model.plan_diffusion.training_loss(plans, cond, mr["combined_mask"])

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN at batch_idx={batch_idx} global_step={global_step}", flush=True)
            for n, p in model.plan_diffusion.named_parameters():
                if p.isnan().any() or p.isinf().any():
                    print(f"  BAD PARAM: {n}", flush=True)
            sys.exit(1)

        (loss / grad_accum).backward()
        running_loss += loss.item()
        accum += 1

        if accum >= grad_accum:
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            accum = 0
            global_step += 1
            avg = running_loss / grad_accum
            running_loss = 0.0
            if global_step % 20 == 0 or global_step <= 3:
                print(f"step {global_step}: loss={avg:.4f} grad_norm={gn:.2f}", flush=True)
            if global_step >= 100:
                break
    if global_step >= 100:
        break
    print(f"Epoch {epoch+1} done, step={global_step}", flush=True)

print(f"Done! {global_step} steps, no NaN", flush=True)
