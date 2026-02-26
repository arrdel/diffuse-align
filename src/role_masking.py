"""
Role-Conditioned Masking — ensures the diffusion model respects agent capabilities.

Key idea: During denoising, each agent should only be assigned actions within
its capability set. Role masking provides this inductive bias by:

1. Hard masking: Zero out plan entries for actions an agent cannot perform.
2. Soft masking: Differentiable Gumbel-softmax attention reweighting that
   down-weights impossible actions but allows gradient flow.

This is a *structural* constraint — unlike guidance (which is a *soft* optimization
signal), role masking is a hard architectural boundary.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CapabilityEncoder(nn.Module):
    """
    Encodes an agent's capability set into a dense capability vector.

    Each agent archetype has a binary capability vector (which actions it can do).
    This module projects that into a dense embedding used for masking.
    """

    def __init__(
        self,
        num_actions: int = 128,
        capability_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_actions, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, capability_dim),
        )

    def forward(self, capability_vector: torch.Tensor) -> torch.Tensor:
        """
        Args:
            capability_vector: (batch, num_agents, num_actions) binary capabilities.
        Returns:
            (batch, num_agents, capability_dim) dense capability embeddings.
        """
        return self.encoder(capability_vector.float())


class ActionCapabilityScorer(nn.Module):
    """
    Scores whether each action in the plan is compatible with the assigned agent.

    Given an agent's capability embedding and a plan action embedding, outputs
    a compatibility score in [0, 1].
    """

    def __init__(
        self,
        capability_dim: int = 64,
        plan_dim: int = 512,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.scorer = nn.Sequential(
            nn.Linear(capability_dim + plan_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        capability_emb: torch.Tensor,
        action_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            capability_emb: (batch, num_agents, capability_dim)
            action_emb: (batch, num_agents, max_steps, plan_dim)
        Returns:
            scores: (batch, num_agents, max_steps) compatibility scores.
        """
        batch, agents, steps, plan_dim = action_emb.shape
        cap_dim = capability_emb.shape[-1]

        # Expand capability to match steps
        cap_expanded = capability_emb.unsqueeze(2).expand(-1, -1, steps, -1)

        # Concatenate and score
        combined = torch.cat([cap_expanded, action_emb], dim=-1)
        scores = self.scorer(combined).squeeze(-1)  # (batch, agents, steps)
        return torch.sigmoid(scores)


class RoleMasker(nn.Module):
    """
    Role-conditioned masking for the joint plan diffusion model.

    Provides two modes:
    - "hard": Binary mask — impossible actions are zeroed out.
    - "soft": Differentiable mask — uses Gumbel-sigmoid with temperature
              annealing so gradients still flow during training.

    Also generates a validity mask for the loss function (so we don't penalize
    the model for noise in padded/invalid positions).
    """

    def __init__(
        self,
        num_actions: int = 128,
        capability_dim: int = 64,
        plan_dim: int = 512,
        max_agents: int = 4,
        max_steps: int = 32,
        mask_type: str = "soft",
        temperature: float = 0.1,
    ):
        super().__init__()

        self.mask_type = mask_type
        self.temperature = temperature
        self.max_agents = max_agents
        self.max_steps = max_steps

        # Capability encoding
        self.capability_encoder = CapabilityEncoder(
            num_actions=num_actions,
            capability_dim=capability_dim,
        )

        # Compatibility scoring (for soft masking)
        if mask_type == "soft":
            self.scorer = ActionCapabilityScorer(
                capability_dim=capability_dim,
                plan_dim=plan_dim,
            )

        # Validity mask for padding
        # (agents or steps beyond the actual team size / plan length)
        self.register_buffer(
            "_default_validity",
            torch.ones(1, max_agents, max_steps),
        )

    def compute_validity_mask(
        self,
        num_agents: torch.Tensor,
        plan_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Create a binary mask for valid (non-padding) positions.

        Args:
            num_agents: (batch,) — actual number of agents per example.
            plan_lengths: (batch,) — actual plan length per example.
        Returns:
            (batch, max_agents, max_steps) binary mask.
        """
        batch_size = num_agents.shape[0]
        device = num_agents.device

        agent_mask = torch.arange(self.max_agents, device=device).unsqueeze(0) < num_agents.unsqueeze(1)
        step_mask = torch.arange(self.max_steps, device=device).unsqueeze(0) < plan_lengths.unsqueeze(1)

        # Outer product: valid only if both agent and step are valid
        validity = agent_mask.unsqueeze(2) & step_mask.unsqueeze(1)
        return validity.float()

    def forward(
        self,
        capabilities: torch.Tensor,
        plan: Optional[torch.Tensor] = None,
        num_agents: Optional[torch.Tensor] = None,
        plan_lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute role-conditioned masks.

        Args:
            capabilities: (batch, num_agents, num_actions) binary capability vectors.
            plan: (batch, agents, steps, plan_dim) — current plan (needed for soft masking).
            num_agents: (batch,) — actual agent count (for validity masking).
            plan_lengths: (batch,) — actual plan lengths.
        Returns:
            Dict with:
                "role_mask": (batch, agents, steps) — role-conditioned mask.
                "validity_mask": (batch, agents, steps) — padding validity mask.
                "combined_mask": (batch, agents, steps) — role * validity.
                "capability_emb": (batch, agents, cap_dim) — for downstream use.
        """
        batch_size = capabilities.shape[0]
        device = capabilities.device

        # Encode capabilities
        cap_emb = self.capability_encoder(capabilities)

        # Validity mask
        if num_agents is not None and plan_lengths is not None:
            validity_mask = self.compute_validity_mask(num_agents, plan_lengths)
        else:
            validity_mask = self._default_validity.expand(batch_size, -1, -1)

        # Role mask
        if self.mask_type == "hard":
            # Hard mask: each agent can only act if it has *any* capability
            # (fine-grained action-level masking happens in the action decoder)
            has_capability = capabilities.any(dim=-1)  # (batch, agents)
            role_mask = has_capability.unsqueeze(2).expand(-1, -1, self.max_steps).float()

        elif self.mask_type == "soft":
            if plan is not None:
                # Score each (agent, action) pair for compatibility
                raw_scores = self.scorer(cap_emb, plan)
                # Gumbel-sigmoid for differentiable masking
                if self.training:
                    gumbel_noise = -torch.log(-torch.log(
                        torch.rand_like(raw_scores).clamp(min=1e-8)
                    ))
                    role_mask = torch.sigmoid(
                        (raw_scores + gumbel_noise) / self.temperature
                    )
                else:
                    role_mask = (raw_scores > 0.5).float()
            else:
                # No plan yet (initial noise) — use uniform mask
                role_mask = torch.ones(
                    batch_size, self.max_agents, self.max_steps, device=device
                )
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")

        combined_mask = role_mask * validity_mask

        return {
            "role_mask": role_mask,
            "validity_mask": validity_mask,
            "combined_mask": combined_mask,
            "capability_emb": cap_emb,
        }
