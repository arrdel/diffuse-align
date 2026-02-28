"""
Compositional Guidance — training-free steering of diffusion sampling.

The key insight: rather than training separate models for each constraint,
we compose multiple gradient-based guidance signals at inference time:

    ε_guided = ε_uncond + s * (ε_cond - ε_uncond) + Σ_i w_i * ∇_x g_i(x, t)

where g_i are guidance functions for:
    1. Task completion — learned classifier predicting P(task_done | plan)
    2. Safety — rule-based detector blocking dangerous action sequences
    3. Efficiency — length penalty encouraging minimal plans
    4. Coordination — learned detector penalizing redundant/conflicting actions

This is the "compositional" part: different guidance functions can be mixed
and matched at inference time without retraining the base diffusion model.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Task Completion Guidance ──────────────────────────────────────────────────

class TaskCompletionClassifier(nn.Module):
    """
    Learns to predict whether a plan (possibly noisy) will lead to task completion.

    Trained on (plan, task_spec) → {0, 1} using completed/failed trajectories.
    At inference time, its gradient ∇_x P(success | x) steers the diffusion
    toward plans that are more likely to succeed.
    """

    def __init__(
        self,
        plan_dim: int = 512,
        condition_dim: int = 512,
        hidden_dim: int = 256,
        max_agents: int = 4,
        max_steps: int = 32,
    ):
        super().__init__()

        input_dim = plan_dim  # After aggregation

        self.plan_aggregator = nn.Sequential(
            nn.Linear(plan_dim, hidden_dim),
            nn.ReLU(),
        )

        self.condition_proj = nn.Linear(condition_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        plan: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            plan: (batch, agents, steps, plan_dim)
            condition: (batch, cond_len, cond_dim)
        Returns:
            (batch,) log-probability of task completion.
        """
        # Aggregate plan: mean over agents and steps
        plan_agg = self.plan_aggregator(plan.mean(dim=[1, 2]))  # (batch, hidden)

        # Aggregate condition: mean over tokens
        cond_agg = self.condition_proj(condition.mean(dim=1))  # (batch, hidden)

        # Classify
        combined = torch.cat([plan_agg, cond_agg], dim=-1)
        logit = self.classifier(combined).squeeze(-1)
        return logit  # Raw logit; use sigmoid for probability


# ─── Coordination Guidance ─────────────────────────────────────────────────────

class CoordinationClassifier(nn.Module):
    """
    Learns to detect redundant or conflicting actions across agents.

    Two agents doing the same thing = redundancy (waste).
    Two agents doing contradictory things = conflict (failure).

    The classifier outputs a "coordination quality" score that the guidance
    function maximizes.
    """

    def __init__(
        self,
        plan_dim: int = 512,
        hidden_dim: int = 256,
        max_agents: int = 4,
        max_steps: int = 32,
    ):
        super().__init__()

        # Pairwise agent interaction modeling
        self.agent_pair_encoder = nn.Sequential(
            nn.Linear(plan_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Per-step conflict/redundancy detection
        self.step_scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # [redundancy_score, conflict_score]
        )

        # Global coordination quality
        self.global_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, plan: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            plan: (batch, agents, steps, plan_dim)
        Returns:
            Dict with "coordination_score", "redundancy_map", "conflict_map".
        """
        batch, agents, steps, dim = plan.shape

        # Compute all pairwise agent interactions at each step
        redundancy_scores = []
        conflict_scores = []
        pair_features = []

        for i in range(agents):
            for j in range(i + 1, agents):
                pair_input = torch.cat([plan[:, i], plan[:, j]], dim=-1)  # (batch, steps, 2*dim)
                pair_enc = self.agent_pair_encoder(pair_input)  # (batch, steps, hidden)
                scores = self.step_scorer(pair_enc)  # (batch, steps, 2)
                redundancy_scores.append(scores[:, :, 0])
                conflict_scores.append(scores[:, :, 1])
                pair_features.append(pair_enc.mean(dim=1))  # (batch, hidden)

        if len(pair_features) > 0:
            # Aggregate pair features for global score
            pair_stack = torch.stack(pair_features, dim=1)  # (batch, num_pairs, hidden)
            global_feat = pair_stack.mean(dim=1)  # (batch, hidden)
            coordination_score = self.global_scorer(global_feat).squeeze(-1)

            redundancy_map = torch.stack(redundancy_scores, dim=1)  # (batch, pairs, steps)
            conflict_map = torch.stack(conflict_scores, dim=1)
        else:
            # Single agent — perfect coordination by definition
            coordination_score = torch.ones(batch, device=plan.device)
            redundancy_map = torch.zeros(batch, 1, steps, device=plan.device)
            conflict_map = torch.zeros(batch, 1, steps, device=plan.device)

        return {
            "coordination_score": coordination_score,
            "redundancy_map": redundancy_map,
            "conflict_map": conflict_map,
        }


# ─── Safety Guidance ───────────────────────────────────────────────────────────

class SafetyChecker(nn.Module):
    """
    Rule-based + learned safety checker.

    Detects dangerous action sequences in the plan (e.g., deleting files,
    bypassing access controls). Returns a safety score that guidance maximizes.
    """

    def __init__(
        self,
        plan_dim: int = 512,
        hidden_dim: int = 128,
        num_safety_rules: int = 16,
    ):
        super().__init__()

        # Learned safety pattern detectors
        self.safety_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(plan_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(num_safety_rules)
        ])

    def forward(self, plan: torch.Tensor) -> torch.Tensor:
        """
        Args:
            plan: (batch, agents, steps, plan_dim)
        Returns:
            (batch,) safety score in [0, 1] (1 = fully safe).
        """
        # Flatten across agents and steps
        flat = plan.reshape(plan.shape[0], -1, plan.shape[-1])  # (batch, A*S, dim)

        # Each safety head detects a different violation pattern
        violations = []
        for head in self.safety_heads:
            # Max-pool over positions: worst violation per safety rule
            scores = head(flat).squeeze(-1)  # (batch, A*S)
            violation = scores.max(dim=-1).values  # (batch,)
            violations.append(violation)

        # Stack and aggregate: safety = 1 - max_violation
        violations = torch.stack(violations, dim=-1)  # (batch, num_rules)
        max_violation = torch.sigmoid(violations.max(dim=-1).values)

        return 1.0 - max_violation


# ─── Efficiency Guidance ───────────────────────────────────────────────────────

class EfficiencyScorer(nn.Module):
    """
    Scores plan efficiency: shorter plans (fewer non-NOP actions) are better.

    Uses a learned "action significance" predictor to distinguish meaningful
    actions from padding/no-ops, then penalizes plans that are longer than
    a target ratio of the estimated minimum.
    """

    def __init__(
        self,
        plan_dim: int = 512,
        hidden_dim: int = 128,
        target_ratio: float = 1.2,
    ):
        super().__init__()
        self.target_ratio = target_ratio

        # Predicts whether each action slot is a meaningful action or NOP
        self.significance_predictor = nn.Sequential(
            nn.Linear(plan_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, plan: torch.Tensor) -> torch.Tensor:
        """
        Args:
            plan: (batch, agents, steps, plan_dim)
        Returns:
            (batch,) efficiency score — higher is better (closer to optimal length).
        """
        # Predict significance of each action slot
        significance = self.significance_predictor(plan).squeeze(-1)  # (batch, A, S)

        # Effective plan length = sum of significance scores
        effective_length = significance.sum(dim=[1, 2])  # (batch,)

        # We want effective_length ≈ target_ratio * min_length
        # Since we don't know min_length, use a soft penalty:
        # Score decreases quadratically as length increases beyond a threshold
        # Normalize by max possible length
        max_length = plan.shape[1] * plan.shape[2]
        normalized_length = effective_length / max_length

        # Efficiency: 1 when short, 0 when using all slots
        efficiency = 1.0 - normalized_length

        return efficiency


# ─── Compositional Guidance (Main Module) ──────────────────────────────────────

class CompositionalGuidance(nn.Module):
    """
    Composes multiple guidance signals for diffusion sampling.

    At inference time, combines:
        - Classifier-free guidance (task conditioning)
        - Task completion gradient
        - Safety gradient
        - Efficiency gradient
        - Coordination gradient

    Each can be weighted independently, enabling plug-and-play constraint composition.
    """

    def __init__(
        self,
        plan_dim: int = 512,
        condition_dim: int = 512,
        max_agents: int = 4,
        max_steps: int = 32,
        task_weight: float = 1.0,
        safety_weight: float = 0.5,
        efficiency_weight: float = 0.3,
        coordination_weight: float = 0.5,
    ):
        super().__init__()

        self.weights = {
            "task": task_weight,
            "safety": safety_weight,
            "efficiency": efficiency_weight,
            "coordination": coordination_weight,
        }

        # Guidance modules
        self.task_classifier = TaskCompletionClassifier(
            plan_dim=plan_dim,
            condition_dim=condition_dim,
            max_agents=max_agents,
            max_steps=max_steps,
        )

        self.coordination_classifier = CoordinationClassifier(
            plan_dim=plan_dim,
            max_agents=max_agents,
            max_steps=max_steps,
        )

        self.safety_checker = SafetyChecker(plan_dim=plan_dim)

        self.efficiency_scorer = EfficiencyScorer(plan_dim=plan_dim)

    def compute_guidance_gradient(
        self,
        plan: torch.Tensor,
        condition: torch.Tensor,
        timestep: torch.Tensor,
        active_signals: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Compute the composed guidance gradient for diffusion sampling.

        This function is called during the DDIM sampling loop. It computes
        gradients of each guidance signal w.r.t. the noisy plan, scales them
        by their weights, and returns the combined gradient.

        Args:
            plan: (batch, agents, steps, plan_dim) — current noisy plan (requires_grad).
            condition: (batch, cond_len, cond_dim).
            timestep: (batch,) — current diffusion timestep.
            active_signals: Which guidance signals to use. Default: all.
        Returns:
            (batch, agents, steps, plan_dim) — gradient to subtract from noise prediction.
        """
        if active_signals is None:
            active_signals = list(self.weights.keys())

        total_gradient = torch.zeros_like(plan)

        # Task completion guidance
        if "task" in active_signals and self.weights["task"] > 0:
            task_logit = self.task_classifier(plan, condition)
            task_score = torch.sigmoid(task_logit).sum()
            grad = torch.autograd.grad(task_score, plan, retain_graph=True)[0]
            total_gradient = total_gradient + self.weights["task"] * grad

        # Coordination guidance
        if "coordination" in active_signals and self.weights["coordination"] > 0:
            coord_result = self.coordination_classifier(plan)
            coord_score = coord_result["coordination_score"].sum()
            grad = torch.autograd.grad(coord_score, plan, retain_graph=True)[0]
            total_gradient = total_gradient + self.weights["coordination"] * grad

        # Safety guidance
        if "safety" in active_signals and self.weights["safety"] > 0:
            safety_score = self.safety_checker(plan).sum()
            grad = torch.autograd.grad(safety_score, plan, retain_graph=True)[0]
            total_gradient = total_gradient + self.weights["safety"] * grad

        # Efficiency guidance
        if "efficiency" in active_signals and self.weights["efficiency"] > 0:
            eff_score = self.efficiency_scorer(plan).sum()
            grad = torch.autograd.grad(eff_score, plan, retain_graph=True)[0]
            total_gradient = total_gradient + self.weights["efficiency"] * grad

        return total_gradient

    def get_guidance_fn(
        self,
        condition: torch.Tensor,
        active_signals: Optional[List[str]] = None,
    ) -> Callable:
        """
        Returns a guidance function compatible with PlanDiffusionModel.sample().

        The returned callable has signature: (plan, timestep) → gradient.
        """
        def guidance_fn(plan: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            return self.compute_guidance_gradient(
                plan, condition, timestep, active_signals
            )
        return guidance_fn

    def training_losses(
        self,
        plans: torch.Tensor,
        conditions: torch.Tensor,
        labels: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute training losses for the guidance classifiers.

        Args:
            plans: (batch, agents, steps, plan_dim) — ground truth plans.
            conditions: (batch, cond_len, cond_dim).
            labels: Dict with "task_success", "coordination_quality", etc.
        Returns:
            Dict of per-module losses.
        """
        losses = {}

        if "task_success" in labels:
            task_logit = self.task_classifier(plans, conditions)
            losses["task_completion"] = F.binary_cross_entropy_with_logits(
                task_logit, labels["task_success"].float()
            )

        if "coordination_quality" in labels:
            coord_result = self.coordination_classifier(plans)
            losses["coordination"] = F.mse_loss(
                torch.sigmoid(coord_result["coordination_score"]),
                labels["coordination_quality"].float(),
            )

        return losses
