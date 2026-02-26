"""
Plan Diffusion Model — the core of DiffuseAlign.

Implements a conditional diffusion model that denoises joint multi-agent action
plans. Instead of generating dialogue turn-by-turn, the model generates the
*entire* coordinated plan in a single denoising pass.

The plan is represented as a tensor of shape:
    (batch, max_agents, max_steps, action_dim)

where each entry is a latent action embedding. The diffusion process adds noise
to ground-truth plans during training, and the denoiser learns to recover the
clean plan conditioned on (task_spec, agent_states).

Architecture choice: We use a Transformer-based denoiser that attends across
both the agent and step dimensions, enabling cross-agent coordination.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


# ─── Sinusoidal Timestep Embedding ────────────────────────────────────────────

class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) integer timesteps in [0, T).
        Returns:
            (batch, dim) timestep embeddings.
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


# ─── Cross-Agent Transformer Block ────────────────────────────────────────────

class CrossAgentTransformerBlock(nn.Module):
    """
    A transformer block that attends across both agents and plan steps.

    The plan tensor is flattened from (agents, steps) into a single sequence,
    allowing each action to attend to all other agents' actions at all steps.
    This is what enables *joint* coordination — agent A's step 3 can directly
    attend to agent B's step 1.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        condition_dim: int = 512,
    ):
        super().__init__()

        # Self-attention over flattened (agents × steps) sequence
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention to conditioning (task spec + agent states)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cond_proj = nn.Linear(condition_dim, hidden_dim)

        # Feedforward
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

        # Timestep modulation (AdaLN-style)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # scale and shift
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        time_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, hidden_dim) — flattened plan.
            condition: (batch, cond_len, condition_dim) — conditioning vectors.
            time_emb: (batch, hidden_dim) — timestep embedding.
            mask: (batch, seq_len) — role-based attention mask.
        Returns:
            (batch, seq_len, hidden_dim)
        """
        # AdaLN: modulate with timestep
        time_params = self.time_mlp(time_emb)
        scale, shift = time_params.chunk(2, dim=-1)
        scale = scale.unsqueeze(1)  # (batch, 1, hidden_dim)
        shift = shift.unsqueeze(1)

        # Self-attention (joint across all agents and steps)
        h = self.norm1(x)
        h = h * (1 + scale) + shift
        key_padding_mask = None
        if mask is not None:
            # mask is (batch, seq_len) with 1=valid, 0=pad.
            # nn.MultiheadAttention key_padding_mask expects True=ignore.
            key_padding_mask = (mask < 0.5)
        h_attn, _ = self.self_attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + h_attn

        # Cross-attention to condition
        h = self.norm2(x)
        cond = self.cond_proj(condition)
        h_cross, _ = self.cross_attn(h, cond, cond)
        x = x + h_cross

        # Feedforward
        h = self.norm3(x)
        x = x + self.ffn(h)

        return x


# ─── Plan Denoiser (Transformer) ──────────────────────────────────────────────

class PlanDenoiserTransformer(nn.Module):
    """
    Transformer-based denoiser for joint multi-agent plans.

    Takes a noisy plan (flattened across agents and steps) and predicts the
    noise (epsilon) conditioned on timestep, task specification, and agent states.
    """

    def __init__(
        self,
        plan_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        time_embed_dim: int = 128,
        condition_dim: int = 512,
        max_agents: int = 4,
        max_steps: int = 32,
    ):
        super().__init__()

        self.max_agents = max_agents
        self.max_steps = max_steps
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(plan_dim, hidden_dim)

        # Positional embeddings
        self.agent_embed = nn.Embedding(max_agents, hidden_dim)
        self.step_embed = nn.Embedding(max_steps, hidden_dim)

        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            CrossAgentTransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                condition_dim=condition_dim,
            )
            for _ in range(num_layers)
        ])

        # Output projection
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, plan_dim)

    def forward(
        self,
        noisy_plan: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
        role_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            noisy_plan: (batch, max_agents, max_steps, plan_dim) — noisy joint plan.
            timestep: (batch,) — diffusion timestep.
            condition: (batch, cond_len, condition_dim) — task + agent state condition.
            role_mask: (batch, max_agents, max_steps) — binary mask for valid actions.
        Returns:
            predicted_noise: (batch, max_agents, max_steps, plan_dim)
        """
        batch_size = noisy_plan.shape[0]

        # Flatten agents × steps into a single sequence
        # (batch, agents, steps, dim) → (batch, agents*steps, dim)
        x = rearrange(noisy_plan, "b a s d -> b (a s) d")
        x = self.input_proj(x)

        # Add agent and step positional embeddings
        agent_ids = torch.arange(self.max_agents, device=x.device)
        step_ids = torch.arange(self.max_steps, device=x.device)

        # Create (agents*steps,) position indices
        agent_pos = repeat(agent_ids, "a -> (a s)", s=self.max_steps)
        step_pos = repeat(step_ids, "s -> (a s)", a=self.max_agents)

        x = x + self.agent_embed(agent_pos).unsqueeze(0) + self.step_embed(step_pos).unsqueeze(0)

        # Timestep embedding
        t_emb = self.time_embed(timestep)  # (batch, hidden_dim)

        # Flatten role mask for attention
        attn_mask = None
        if role_mask is not None:
            attn_mask = rearrange(role_mask, "b a s -> b (a s)")

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, condition, t_emb, mask=attn_mask)

        # Output projection
        x = self.final_norm(x)
        x = self.output_proj(x)

        # Reshape back to (batch, agents, steps, plan_dim)
        x = rearrange(x, "b (a s) d -> b a s d", a=self.max_agents, s=self.max_steps)

        return x


# ─── Plan Diffusion Model ─────────────────────────────────────────────────────

class PlanDiffusionModel(nn.Module):
    """
    Full diffusion model for joint multi-agent plan generation.

    Wraps the denoiser with DDPM/DDIM forward-backward processes.
    Supports classifier-free guidance via condition dropout.
    """

    def __init__(
        self,
        plan_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        time_embed_dim: int = 128,
        condition_dim: int = 512,
        max_agents: int = 4,
        max_steps: int = 32,
        num_train_timesteps: int = 1000,
        num_inference_timesteps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        prediction_type: str = "epsilon",
        unconditional_prob: float = 0.1,
    ):
        super().__init__()

        self.plan_dim = plan_dim
        self.max_agents = max_agents
        self.max_steps = max_steps
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps
        self.prediction_type = prediction_type
        self.unconditional_prob = unconditional_prob

        # Denoiser network
        self.denoiser = PlanDenoiserTransformer(
            plan_dim=plan_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            time_embed_dim=time_embed_dim,
            condition_dim=condition_dim,
            max_agents=max_agents,
            max_steps=max_steps,
        )

        # Noise schedule (cosine)
        self._build_noise_schedule(num_train_timesteps, beta_start, beta_end)

        # Null condition for classifier-free guidance
        self.null_condition = nn.Parameter(torch.randn(1, 1, condition_dim) * 0.01)

    def _build_noise_schedule(
        self, num_timesteps: int, beta_start: float, beta_end: float
    ):
        """Build cosine beta schedule (Improved DDPM)."""
        steps = torch.linspace(0, num_timesteps, num_timesteps + 1)
        alpha_bar = torch.cos(((steps / num_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        betas = torch.clamp(betas, min=1e-5, max=0.999)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: add noise to clean plan at timestep t.

        Args:
            x_start: (batch, agents, steps, plan_dim) — clean plan.
            t: (batch,) — timesteps.
            noise: Optional pre-sampled noise.
        Returns:
            (noisy_plan, noise)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        while sqrt_alpha.dim() < x_start.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)

        noisy = sqrt_alpha * x_start + sqrt_one_minus_alpha * noise
        return noisy, noise

    def training_loss(
        self,
        x_start: torch.Tensor,
        condition: torch.Tensor,
        role_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute training loss (simplified diffusion loss).

        Args:
            x_start: (batch, agents, steps, plan_dim) — ground truth plans.
            condition: (batch, cond_len, cond_dim) — conditioning vectors.
            role_mask: (batch, agents, steps) — soft/hard mask for loss weighting.
            attn_mask: (batch, agents, steps) — binary mask for attention padding.
                       If None, falls back to binarized role_mask.
        Returns:
            Scalar loss.
        """
        batch_size = x_start.shape[0]
        device = x_start.device

        # Sample random timesteps
        t = torch.randint(0, self.num_train_timesteps, (batch_size,), device=device)

        # Forward diffusion
        noisy_plan, noise = self.q_sample(x_start, t)

        # Classifier-free guidance: randomly drop condition
        if self.training and self.unconditional_prob > 0:
            mask = torch.rand(batch_size, device=device) < self.unconditional_prob
            null_cond = self.null_condition.expand(batch_size, -1, -1)
            condition = torch.where(
                mask[:, None, None].expand_as(condition),
                null_cond.expand_as(condition),
                condition,
            )

        # Use binary attention mask for denoiser (prevents NaN in softmax)
        # Fall back to binarized role_mask if no explicit attn_mask given
        denoiser_mask = attn_mask if attn_mask is not None else role_mask
        predicted = self.denoiser(noisy_plan, t, condition, denoiser_mask)

        # Target
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = x_start
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # MSE loss, optionally masked to valid actions only
        loss = F.mse_loss(predicted, target, reduction="none")

        if role_mask is not None:
            # Mask out invalid (padding) positions
            mask_expanded = role_mask.unsqueeze(-1).expand_as(loss)
            loss = (loss * mask_expanded).sum() / mask_expanded.sum().clamp(min=1)
        else:
            loss = loss.mean()

        return loss

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        role_mask: Optional[torch.Tensor] = None,
        guidance_fn: Optional[callable] = None,
        guidance_scale: float = 3.0,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a joint plan via DDIM sampling with optional guidance.

        Args:
            condition: (batch, cond_len, cond_dim).
            role_mask: (batch, agents, steps).
            guidance_fn: Optional callable for compositional guidance.
            guidance_scale: Classifier-free guidance scale.
            num_steps: Override inference steps.
        Returns:
            (batch, agents, steps, plan_dim) — denoised plan.
        """
        batch_size = condition.shape[0]
        device = condition.device
        num_steps = num_steps or self.num_inference_timesteps

        # Start from pure noise
        x = torch.randn(
            batch_size, self.max_agents, self.max_steps, self.plan_dim,
            device=device,
        )

        # DDIM timestep schedule
        step_ratio = self.num_train_timesteps // num_steps
        timesteps = torch.arange(0, self.num_train_timesteps, step_ratio, device=device)
        timesteps = timesteps.flip(0)  # Reverse: T → 0

        for i, t in enumerate(timesteps):
            t_batch = t.expand(batch_size)

            # Conditional prediction
            cond_pred = self.denoiser(x, t_batch, condition, role_mask)

            # Unconditional prediction (for CFG)
            if guidance_scale != 1.0:
                null_cond = self.null_condition.expand(batch_size, -1, -1)
                if null_cond.shape[1] < condition.shape[1]:
                    null_cond = null_cond.expand(-1, condition.shape[1], -1)
                uncond_pred = self.denoiser(x, t_batch, null_cond, role_mask)
                # Classifier-free guidance
                pred = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            else:
                pred = cond_pred

            # Additional compositional guidance
            if guidance_fn is not None:
                with torch.enable_grad():
                    x_temp = x.detach().requires_grad_(True)
                    guidance_grad = guidance_fn(x_temp, t)
                pred = pred - guidance_grad

            # DDIM update step
            if self.prediction_type == "epsilon":
                alpha_t = self.alphas_cumprod[t]
                alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0, device=device)

                x0_pred = (x - torch.sqrt(1 - alpha_t) * pred) / torch.sqrt(alpha_t)
                x0_pred = torch.clamp(x0_pred, -5.0, 5.0)  # Clip for stability

                # DDIM deterministic step (eta=0)
                x = torch.sqrt(alpha_prev) * x0_pred + torch.sqrt(1 - alpha_prev) * pred

            elif self.prediction_type == "sample":
                x = pred  # Direct sample prediction

        return x

    def forward(
        self,
        x_start: torch.Tensor,
        condition: torch.Tensor,
        role_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Training forward: returns loss."""
        return self.training_loss(x_start, condition, role_mask)
