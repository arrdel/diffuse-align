"""
Plan-to-Dialogue Decoder — converts abstract action plans into natural language.

The diffusion model generates plans as sequences of latent action embeddings.
This decoder translates them into actual dialogue utterances that agents produce
during execution.

Architecture: Flan-T5 backbone fine-tuned on (action_sequence → utterance) pairs.
Each plan step is decoded into a natural language turn, e.g.:

    Plan:  Agent[navigator] → NAVIGATE(kitchen) → LOOK(counter) → REPORT(found_mug)
    NL:    "I'll head to the kitchen to look for the mug. ... Found it on the counter!"
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PlanTokenizer(nn.Module):
    """
    Converts continuous plan embeddings into discrete action tokens
    that can be fed to the seq2seq decoder.

    Uses a learned codebook (VQ-style) to discretize latent plans.
    """

    def __init__(
        self,
        plan_dim: int = 512,
        codebook_size: int = 128,
        code_dim: int = 256,
    ):
        super().__init__()
        self.codebook_size = codebook_size

        # Project plan embeddings to code dimension
        self.pre_proj = nn.Linear(plan_dim, code_dim)

        # Codebook
        self.codebook = nn.Embedding(codebook_size, code_dim)
        nn.init.uniform_(self.codebook.weight, -1.0 / codebook_size, 1.0 / codebook_size)

    def quantize(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous plan embeddings to nearest codebook entries.

        Args:
            z: (..., code_dim) continuous embeddings.
        Returns:
            z_q: (..., code_dim) quantized embeddings.
            indices: (...,) codebook indices.
            commit_loss: scalar commitment loss.
        """
        z_flat = z.reshape(-1, z.shape[-1])

        # Distances to codebook entries
        dists = (
            z_flat.pow(2).sum(dim=-1, keepdim=True)
            + self.codebook.weight.pow(2).sum(dim=-1)
            - 2 * z_flat @ self.codebook.weight.t()
        )

        indices = dists.argmin(dim=-1)
        z_q = self.codebook(indices)

        # Reshape
        z_q = z_q.reshape(z.shape)
        indices = indices.reshape(z.shape[:-1])

        # Commitment loss
        commit_loss = F.mse_loss(z_q.detach(), z) + 0.25 * F.mse_loss(z_q, z.detach())

        # Straight-through estimator
        z_q = z + (z_q - z).detach()

        return z_q, indices, commit_loss

    def forward(self, plan: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            plan: (..., plan_dim) continuous plan embeddings.
        Returns:
            z_q, indices, commit_loss
        """
        z = self.pre_proj(plan)
        return self.quantize(z)


class PlanToDialogueDecoder(nn.Module):
    """
    Decodes quantized plan steps into natural language dialogue utterances.

    For each agent at each step, generates a natural language realization of
    the planned action. Uses a pre-trained seq2seq model (Flan-T5) fine-tuned
    on plan-to-utterance pairs.
    """

    def __init__(
        self,
        plan_dim: int = 512,
        codebook_size: int = 128,
        code_dim: int = 256,
        backbone: str = "google/flan-t5-base",
        max_utterance_length: int = 128,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        self.max_utterance_length = max_utterance_length

        # Plan tokenizer
        self.plan_tokenizer = PlanTokenizer(
            plan_dim=plan_dim,
            codebook_size=codebook_size,
            code_dim=code_dim,
        )

        # Seq2seq backbone
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        self.text_tokenizer = T5Tokenizer.from_pretrained(backbone)
        self.seq2seq = T5ForConditionalGeneration.from_pretrained(backbone)

        if freeze_backbone:
            for param in self.seq2seq.parameters():
                param.requires_grad = False
            # Unfreeze only the cross-attention layers for adapter-style training
            for name, param in self.seq2seq.named_parameters():
                if "EncDecAttention" in name:
                    param.requires_grad = True

        # Project quantized plan codes into T5 encoder space
        t5_dim = self.seq2seq.config.d_model
        self.code_to_t5 = nn.Sequential(
            nn.Linear(code_dim, t5_dim),
            nn.LayerNorm(t5_dim),
        )

        # Agent role prefix embeddings
        self.role_prefix = nn.Embedding(8, t5_dim)  # Up to 8 agent roles

    def encode_plan_for_decoder(
        self,
        plan: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert a plan into encoder hidden states for the T5 decoder.

        Args:
            plan: (batch, agents, steps, plan_dim).
            agent_ids: (batch, agents) — agent role indices.
        Returns:
            encoder_hidden: (batch * agents, steps + 1, t5_dim)
            commit_loss: scalar.
        """
        batch, agents, steps, dim = plan.shape

        # Flatten to per-agent plans
        flat_plan = rearrange(plan, "b a s d -> (b a) s d")

        # Quantize
        z_q, indices, commit_loss = self.plan_tokenizer(flat_plan)

        # Project to T5 space
        encoded = self.code_to_t5(z_q)  # (B*A, steps, t5_dim)

        # Prepend role prefix
        if agent_ids is not None:
            role_ids = rearrange(agent_ids, "b a -> (b a)")
            role_emb = self.role_prefix(role_ids).unsqueeze(1)  # (B*A, 1, t5_dim)
        else:
            role_emb = self.role_prefix(
                torch.zeros(batch * agents, dtype=torch.long, device=plan.device)
            ).unsqueeze(1)

        encoded = torch.cat([role_emb, encoded], dim=1)  # (B*A, steps+1, t5_dim)

        return encoded, commit_loss

    def forward(
        self,
        plan: torch.Tensor,
        target_utterances: Optional[List[str]] = None,
        agent_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Training forward: compute NLL loss for plan → utterance generation.

        Args:
            plan: (batch, agents, steps, plan_dim).
            target_utterances: List of B*A target utterance strings.
            agent_ids: (batch, agents) agent role indices.
        Returns:
            Dict with "nll_loss", "commit_loss".
        """
        batch, agents, steps, dim = plan.shape
        device = plan.device

        # Encode plan
        encoder_hidden, commit_loss = self.encode_plan_for_decoder(plan, agent_ids)

        if target_utterances is not None:
            # Tokenize targets
            targets = self.text_tokenizer(
                target_utterances,
                padding=True,
                truncation=True,
                max_length=self.max_utterance_length,
                return_tensors="pt",
            ).to(device)

            # Create attention mask for encoder hidden states
            attn_mask = torch.ones(
                encoder_hidden.shape[:2], device=device, dtype=torch.long
            )

            # Forward through T5 decoder
            outputs = self.seq2seq(
                encoder_outputs=(encoder_hidden,),
                attention_mask=attn_mask,
                labels=targets.input_ids,
            )

            return {
                "nll_loss": outputs.loss,
                "commit_loss": commit_loss,
                "total_loss": outputs.loss + 0.1 * commit_loss,
            }
        else:
            return {"commit_loss": commit_loss}

    @torch.no_grad()
    def generate_utterances(
        self,
        plan: torch.Tensor,
        agent_ids: Optional[torch.Tensor] = None,
        beam_size: int = 4,
        repetition_penalty: float = 2.5,
        length_penalty: float = 1.0,
        no_repeat_ngram_size: int = 3,
        do_sample: bool = False,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> List[str]:
        """
        Generate natural language utterances from a plan.

        Args:
            plan: (batch, agents, steps, plan_dim).
            agent_ids: (batch, agents) agent role indices.
            beam_size: Beam search width.
            repetition_penalty: Penalize repeated tokens (>1.0 = more penalty).
            length_penalty: >1.0 favors longer, <1.0 favors shorter.
            no_repeat_ngram_size: Prevent repeating n-grams of this size.
            do_sample: Use sampling instead of beam search.
            temperature: Sampling temperature (only if do_sample=True).
            top_p: Nucleus sampling threshold (only if do_sample=True).
        Returns:
            List of B*A generated utterance strings.
        """
        encoder_hidden, _ = self.encode_plan_for_decoder(plan, agent_ids)

        attn_mask = torch.ones(
            encoder_hidden.shape[:2], device=plan.device, dtype=torch.long
        )

        # Generate with beam search + anti-repetition controls
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)

        gen_kwargs = dict(
            encoder_outputs=encoder_outputs,
            attention_mask=attn_mask,
            max_new_tokens=self.max_utterance_length,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            early_stopping=True,
        )

        if do_sample:
            gen_kwargs.update(
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_beams=1,
            )
        else:
            gen_kwargs.update(
                num_beams=beam_size,
            )

        output_ids = self.seq2seq.generate(**gen_kwargs)

        utterances = self.text_tokenizer.batch_decode(
            output_ids, skip_special_tokens=True
        )

        return utterances
