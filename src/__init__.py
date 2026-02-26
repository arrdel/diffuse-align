"""
DiffuseAlign: Diffusion-Based Joint Plan Generation for Multi-Agent Dialogue Coordination.

A framework for coordinating multiple dialogue agents through joint trajectory
planning via conditional diffusion models, replacing greedy turn-by-turn generation
with holistic plan denoising.
"""

__version__ = "0.1.0"
__author__ = "Adele Chinda"

from .diffuse_align import DiffuseAlign
from .plan_diffusion import PlanDiffusionModel
from .plan_encoder import PlanEncoder
from .role_masking import RoleMasker
from .guidance import CompositionalGuidance
from .plan_decoder import PlanToDialogueDecoder
from .agents import AgentTeam, Agent

__all__ = [
    "DiffuseAlign",
    "PlanDiffusionModel",
    "PlanEncoder",
    "RoleMasker",
    "CompositionalGuidance",
    "PlanToDialogueDecoder",
    "AgentTeam",
    "Agent",
]
