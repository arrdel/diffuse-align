"""
DiffuseAlign — the main module that ties all components together.

DiffuseAlign: Diffusion-Based Joint Plan Generation for Multi-Agent Dialogue Coordination.

Architecture:

    Input: (task_spec, agent_team)
    ┌─────────────────────────────────────────────────────────┐
    │  1. PlanEncoder: task_text + agent_states → condition   │
    │  2. RoleMasker: agent capabilities → role masks         │
    │  3. PlanDiffusion: denoise joint plan with condition    │
    │     + compositional guidance (task, safety, efficiency) │
    │  4. PlanDecoder: plan → natural language dialogue       │
    └─────────────────────────────────────────────────────────┘
    Output: (joint_plan, dialogue_utterances, metrics)

Training is 3-stage:
    Stage 1: Train PlanDiffusion with condition dropout (main model)
    Stage 2: Train PlanDecoder on (plan, utterance) pairs (freeze diffusion)
    Stage 3: Train guidance classifiers on trajectory labels (freeze diffusion)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .plan_diffusion import PlanDiffusionModel
from .plan_encoder import PlanEncoder
from .role_masking import RoleMasker
from .guidance import CompositionalGuidance
from .plan_decoder import PlanToDialogueDecoder
from .agents import AgentTeam, VOCAB_SIZE


class DiffuseAlign(nn.Module):
    """
    DiffuseAlign: Full multi-agent dialogue coordination system.

    Combines plan encoder, diffusion model, role masking, compositional
    guidance, and plan-to-dialogue decoder into a unified system.
    """

    def __init__(
        self,
        # Diffusion model
        plan_dim: int = 512,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
        # Conditioning
        condition_dim: int = 512,
        agent_state_dim: int = 128,
        task_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        # Agents
        max_agents: int = 4,
        max_steps: int = 32,
        num_actions: int = VOCAB_SIZE,
        # Diffusion process
        num_train_timesteps: int = 1000,
        num_inference_timesteps: int = 50,
        unconditional_prob: float = 0.1,
        # Role masking
        mask_type: str = "soft",
        capability_dim: int = 64,
        # Guidance
        guidance_scale: float = 3.0,
        task_guidance_weight: float = 1.0,
        safety_guidance_weight: float = 0.5,
        efficiency_guidance_weight: float = 0.3,
        coordination_guidance_weight: float = 0.5,
        # Decoder
        decoder_backbone: str = "google/flan-t5-base",
    ):
        super().__init__()

        self.plan_dim = plan_dim
        self.max_agents = max_agents
        self.max_steps = max_steps
        self.guidance_scale = guidance_scale

        # ─── Plan Encoder ──────────────────────────────────────────
        self.plan_encoder = PlanEncoder(
            task_encoder_model=task_encoder_model,
            task_encoder_dim=384,
            agent_state_dim=agent_state_dim,
            condition_dim=condition_dim,
            max_agents=max_agents,
            num_fusion_heads=num_heads,
            num_fusion_layers=2,
            freeze_task_encoder=True,
        )

        # ─── Plan Diffusion Model ─────────────────────────────────
        self.plan_diffusion = PlanDiffusionModel(
            plan_dim=plan_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            condition_dim=condition_dim,
            max_agents=max_agents,
            max_steps=max_steps,
            num_train_timesteps=num_train_timesteps,
            num_inference_timesteps=num_inference_timesteps,
            unconditional_prob=unconditional_prob,
        )

        # ─── Role Masking ─────────────────────────────────────────
        self.role_masker = RoleMasker(
            num_actions=num_actions,
            capability_dim=capability_dim,
            plan_dim=plan_dim,
            max_agents=max_agents,
            max_steps=max_steps,
            mask_type=mask_type,
        )

        # ─── Compositional Guidance ───────────────────────────────
        self.guidance = CompositionalGuidance(
            plan_dim=plan_dim,
            condition_dim=condition_dim,
            max_agents=max_agents,
            max_steps=max_steps,
            task_weight=task_guidance_weight,
            safety_weight=safety_guidance_weight,
            efficiency_weight=efficiency_guidance_weight,
            coordination_weight=coordination_guidance_weight,
        )

        # ─── Plan-to-Dialogue Decoder ─────────────────────────────
        self.plan_decoder = PlanToDialogueDecoder(
            plan_dim=plan_dim,
            backbone=decoder_backbone,
        )

    # ──────────────────────────────────────────────────────────────────
    # Training (Stage 1: Diffusion)
    # ──────────────────────────────────────────────────────────────────

    def training_step_diffusion(
        self,
        plans: torch.Tensor,
        task_texts: List[str],
        agent_states: torch.Tensor,
        capabilities: torch.Tensor,
        num_agents: torch.Tensor,
        plan_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 1 training step: train the diffusion model.

        Args:
            plans: (batch, agents, steps, plan_dim) ground truth plans.
            task_texts: List of B task descriptions.
            agent_states: (batch, agents, state_dim).
            capabilities: (batch, agents, num_actions) binary.
            num_agents: (batch,) actual agent counts.
            plan_lengths: (batch,) actual plan lengths.
        Returns:
            Dict with "diffusion_loss" and auxiliary losses.
        """
        # Encode condition
        condition = self.plan_encoder(task_texts, agent_states)

        # Compute role masks
        mask_result = self.role_masker(
            capabilities, plans, num_agents, plan_lengths
        )
        combined_mask = mask_result["combined_mask"]

        # Diffusion loss
        diffusion_loss = self.plan_diffusion.training_loss(
            plans, condition, combined_mask
        )

        return {
            "diffusion_loss": diffusion_loss,
            "total_loss": diffusion_loss,
        }

    # ──────────────────────────────────────────────────────────────────
    # Training (Stage 2: Decoder)
    # ──────────────────────────────────────────────────────────────────

    def training_step_decoder(
        self,
        plans: torch.Tensor,
        target_utterances: List[str],
        agent_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 2 training step: train the plan-to-dialogue decoder.
        Diffusion model should be frozen.

        Args:
            plans: (batch, agents, steps, plan_dim).
            target_utterances: List of B*A target utterance strings.
            agent_ids: (batch, agents) role indices.
        Returns:
            Dict with "nll_loss", "commit_loss", "total_loss".
        """
        return self.plan_decoder(plans, target_utterances, agent_ids)

    # ──────────────────────────────────────────────────────────────────
    # Training (Stage 3: Guidance classifiers)
    # ──────────────────────────────────────────────────────────────────

    def training_step_guidance(
        self,
        plans: torch.Tensor,
        conditions: torch.Tensor,
        labels: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Stage 3 training step: train guidance classifiers.
        Both diffusion and decoder should be frozen.

        Args:
            plans: (batch, agents, steps, plan_dim).
            conditions: (batch, cond_len, cond_dim).
            labels: Dict with "task_success", "coordination_quality", etc.
        Returns:
            Dict of per-classifier losses.
        """
        return self.guidance.training_losses(plans, conditions, labels)

    # ──────────────────────────────────────────────────────────────────
    # Inference
    # ──────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_plan(
        self,
        task_texts: List[str],
        agent_states: torch.Tensor,
        capabilities: torch.Tensor,
        num_agents: Optional[torch.Tensor] = None,
        use_guidance: bool = True,
        guidance_scale: Optional[float] = None,
        active_guidance_signals: Optional[List[str]] = None,
        num_inference_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate a joint multi-agent plan via diffusion sampling.

        Args:
            task_texts: List of B task descriptions.
            agent_states: (batch, agents, state_dim).
            capabilities: (batch, agents, num_actions).
            num_agents: (batch,) actual agent counts.
            use_guidance: Whether to apply compositional guidance.
            guidance_scale: Override CFG scale.
            active_guidance_signals: Which guidance signals to apply.
            num_inference_steps: Override number of DDIM steps.
        Returns:
            (batch, agents, steps, plan_dim) — generated plan.
        """
        # Encode condition
        condition = self.plan_encoder(task_texts, agent_states)

        # Compute role mask (without plan, since plan doesn't exist yet)
        mask_result = self.role_masker(capabilities)
        role_mask = mask_result.get("combined_mask")

        # Guidance function
        guidance_fn = None
        if use_guidance:
            guidance_fn = self.guidance.get_guidance_fn(
                condition, active_guidance_signals
            )

        # Sample plan
        plan = self.plan_diffusion.sample(
            condition=condition,
            role_mask=role_mask,
            guidance_fn=guidance_fn,
            guidance_scale=guidance_scale or self.guidance_scale,
            num_steps=num_inference_steps,
        )

        return plan

    @torch.no_grad()
    def generate_dialogue(
        self,
        task_texts: List[str],
        agent_states: torch.Tensor,
        capabilities: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None,
        **plan_kwargs,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Full pipeline: generate plan then decode to dialogue.

        Args:
            task_texts, agent_states, capabilities: as in generate_plan.
            agent_ids: (batch, agents) role indices for the decoder.
            **plan_kwargs: Additional arguments for generate_plan.
        Returns:
            (plan, utterances) — the generated plan and NL dialogue.
        """
        # Generate plan
        plan = self.generate_plan(
            task_texts, agent_states, capabilities, **plan_kwargs
        )

        # Decode to dialogue
        utterances = self.plan_decoder.generate_utterances(
            plan, agent_ids=agent_ids
        )

        return plan, utterances

    # ──────────────────────────────────────────────────────────────────
    # Forward (training mode dispatch)
    # ──────────────────────────────────────────────────────────────────

    def forward(
        self,
        plans: torch.Tensor,
        task_texts: List[str],
        agent_states: torch.Tensor,
        capabilities: torch.Tensor,
        num_agents: torch.Tensor,
        plan_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Default forward: Stage 1 diffusion training."""
        return self.training_step_diffusion(
            plans, task_texts, agent_states, capabilities,
            num_agents, plan_lengths,
        )
