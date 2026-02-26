"""
Plan Encoder — encodes task specifications and agent states into conditioning
vectors for the diffusion denoiser.

Architecture:
    - Task text → frozen sentence encoder → task embedding
    - Each agent's state (capabilities, inventory, position) → MLP → agent embedding
    - Cross-attention fusion: task attends to agent embeddings
    - Output: (batch, cond_len, cond_dim) conditioning tensor
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TaskEncoder(nn.Module):
    """
    Encodes natural-language task specifications into dense vectors.

    Uses a frozen sentence transformer for efficiency. The task description
    is tokenized and encoded, then projected to the conditioning dimension.
    """

    def __init__(
        self,
        pretrained_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        output_dim: int = 384,
        condition_dim: int = 512,
        freeze: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.condition_dim = condition_dim

        # Lazy import to avoid heavy dependency at module level
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
        self.encoder = AutoModel.from_pretrained(pretrained_model)

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.proj = nn.Linear(output_dim, condition_dim)

    def forward(self, task_texts: List[str]) -> torch.Tensor:
        """
        Args:
            task_texts: List of B task description strings.
        Returns:
            (batch, seq_len, condition_dim)
        """
        device = self.proj.weight.device
        encoded = self.tokenizer(
            task_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad() if not any(p.requires_grad for p in self.encoder.parameters()) else torch.enable_grad():
            outputs = self.encoder(**encoded)

        # Use all token embeddings (not just CLS) for richer conditioning
        token_embeddings = outputs.last_hidden_state  # (batch, seq_len, output_dim)
        return self.proj(token_embeddings)


class AgentStateEncoder(nn.Module):
    """
    Encodes each agent's structured state into a dense vector.

    Agent state includes:
        - capability_vector: binary vector of available actions
        - inventory: what the agent currently holds
        - position: where the agent is (for embodied tasks)
        - role_id: which archetype this agent is
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: List[int] = [256, 512],
        condition_dim: int = 512,
        max_agents: int = 4,
    ):
        super().__init__()
        self.max_agents = max_agents

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, condition_dim))

        self.mlp = nn.Sequential(*layers)

        # Learnable agent-order embedding
        self.agent_position_embed = nn.Embedding(max_agents, condition_dim)

    def forward(self, agent_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_states: (batch, num_agents, input_dim) — structured state per agent.
        Returns:
            (batch, num_agents, condition_dim)
        """
        batch_size, num_agents, _ = agent_states.shape

        encoded = self.mlp(agent_states)  # (batch, agents, cond_dim)

        # Add agent position embedding
        agent_ids = torch.arange(num_agents, device=agent_states.device)
        pos_embed = self.agent_position_embed(agent_ids)  # (agents, cond_dim)
        encoded = encoded + pos_embed.unsqueeze(0)

        return encoded


class TaskAgentFusion(nn.Module):
    """
    Fuses task encoding with agent state encodings via cross-attention.

    The task tokens attend to agent state tokens, allowing the task
    representation to be modulated by which agents are available and
    what their current states are.
    """

    def __init__(
        self,
        condition_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "cross_attn": nn.MultiheadAttention(
                    embed_dim=condition_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                ),
                "norm1": nn.LayerNorm(condition_dim),
                "norm2": nn.LayerNorm(condition_dim),
                "ffn": nn.Sequential(
                    nn.Linear(condition_dim, condition_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(condition_dim * 4, condition_dim),
                    nn.Dropout(dropout),
                ),
            })
            for _ in range(num_layers)
        ])

    def forward(
        self,
        task_tokens: torch.Tensor,
        agent_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            task_tokens: (batch, task_len, cond_dim) — from TaskEncoder.
            agent_tokens: (batch, num_agents, cond_dim) — from AgentStateEncoder.
        Returns:
            (batch, task_len + num_agents, cond_dim) — fused conditioning.
        """
        x = task_tokens

        for layer in self.layers:
            # Cross-attention: task attends to agents
            h = layer["norm1"](x)
            h, _ = layer["cross_attn"](h, agent_tokens, agent_tokens)
            x = x + h

            # FFN
            h = layer["norm2"](x)
            x = x + layer["ffn"](h)

        # Concatenate task and agent tokens for full conditioning
        return torch.cat([x, agent_tokens], dim=1)


class PlanEncoder(nn.Module):
    """
    Full plan encoder: Task text + Agent states → Conditioning tensor.

    This is the condition generator for the diffusion model.
    """

    def __init__(
        self,
        task_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        task_encoder_dim: int = 384,
        agent_state_dim: int = 128,
        condition_dim: int = 512,
        max_agents: int = 4,
        num_fusion_heads: int = 8,
        num_fusion_layers: int = 2,
        freeze_task_encoder: bool = True,
    ):
        super().__init__()

        self.task_encoder = TaskEncoder(
            pretrained_model=task_encoder_model,
            output_dim=task_encoder_dim,
            condition_dim=condition_dim,
            freeze=freeze_task_encoder,
        )

        self.agent_encoder = AgentStateEncoder(
            input_dim=agent_state_dim,
            hidden_dims=[256, condition_dim],
            condition_dim=condition_dim,
            max_agents=max_agents,
        )

        self.fusion = TaskAgentFusion(
            condition_dim=condition_dim,
            num_heads=num_fusion_heads,
            num_layers=num_fusion_layers,
        )

    def forward(
        self,
        task_texts: List[str],
        agent_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            task_texts: List of B task description strings.
            agent_states: (batch, num_agents, agent_state_dim).
        Returns:
            (batch, cond_len, condition_dim) — conditioning for diffusion model.
        """
        task_tokens = self.task_encoder(task_texts)
        agent_tokens = self.agent_encoder(agent_states)
        condition = self.fusion(task_tokens, agent_tokens)
        return condition
