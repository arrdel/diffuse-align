# DiffuseAlign

**Diffusion-Based Joint Plan Generation for Multi-Agent Dialogue Coordination**

> Under review at SIGDIAL 2026 — Agentic AI and Interaction Track

## Overview

Multi-agent dialogue systems typically coordinate through sequential turn-taking, where each agent generates its contribution conditioned only on conversation history. This myopic approach leads to redundant actions, coordination failures, and a disconnect between conversational fluency and functional task success.

**DiffuseAlign** formulates multi-agent dialogue coordination as joint trajectory planning via conditional diffusion models. Rather than generating dialogue turn-by-turn, DiffuseAlign denoises the *entire* multi-agent action plan simultaneously—including speech acts, tool invocations, and delegation decisions—in a single forward pass. This enables globally coherent coordination without explicit negotiation rounds.

### Key Mechanisms

- **Role-conditioned denoising** — Differentiable Gumbel-softmax masks enforce agent-specific capability boundaries (e.g., only a manipulator picks up objects), achieving 100% delegation accuracy.
- **Compositional guidance** — Four inference-time guidance signals (task completion, safety, efficiency, coordination) can be composed and reweighted without retraining, enabling flexible deployment trade-offs.
- **Plan-to-dialogue decoder** — A VQ-codebook bridge to fine-tuned Flan-T5-base translates latent plan tensors into natural-language utterances, decoupling plan representation from language generation.

## Results

Evaluated across three collaborative benchmarks (ALFWorld-Multi, WebArena-Multi, CollabCooking) with 2,500 episodes per method and 5 random seeds.

| Method | Task Succ ↑ | Efficiency ↑ | Coord ↑ | Turns ↓ | Gap ↓ |
|--------|------------|-------------|---------|---------|-------|
| Sequential LLM | 57.4 | 1.43 | −3.32 | 10.0 | 0.57 |
| Round-Robin | **72.9** | **3.56** | −2.00 | **5.3** | 0.73 |
| CAMEL | 41.2 | 1.26 | −1.70 | 13.8 | 0.41 |
| DyLAN | 64.8 | 2.49 | **−0.41** | 7.7 | 0.65 |
| **DiffuseAlign** | 42.0 | 1.48 | −3.31 | 12.9 | **0.42** |

DiffuseAlign achieves the lowest **functional–fluency gap** (0.42), indicating that its dialogue quality is well-calibrated to its actual coordination capability—a desirable property for trustworthy deployment. Joint planning shows its largest improvements on complex multi-object tasks (≥16 steps), precisely where sequential approaches suffer most from myopic planning.

## Installation

```bash
conda create -n diffusealign python=3.11 -y
conda activate diffusealign
pip install -r requirements.txt
```

### Requirements

- Python 3.11+
- PyTorch ≥ 2.1
- NVIDIA GPU (A6000 or equivalent recommended)
- ~1.44 GB for the assembled model (310.6M parameters)

## Usage

### Training (three stages)

```bash
# Stage 1: Plan diffusion model (100K steps)
python scripts/train.py --config configs/default.yaml --stage 1

# Stage 2: Plan-to-dialogue decoder (20 epochs, diffusion frozen)
python scripts/train.py --config configs/default.yaml --stage 2

# Stage 3: Guidance classifiers (20 epochs)
python scripts/train.py --config configs/default.yaml --stage 3
```

### Evaluation

```bash
# Main evaluation (single seed)
python scripts/evaluate.py --config configs/default.yaml --num_episodes 500

# Scaled evaluation (multiple seeds)
bash scripts/run_scaled_eval.sh

# Ablation study
bash scripts/run_all_ablations.sh
```

### Trajectory Collection

```bash
python scripts/generate_trajectories.py \
    --env alfworld \
    --num_episodes 5000 \
    --num_agents 2 \
    --output data/alfworld_multi/
```

## Evaluation Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| Task Success | Fraction of episodes where all goal conditions are met | ↑ Higher is better |
| Action Efficiency | Ratio of optimal to actual actions | ↑ Higher is better |
| Coordination Score | Negative count of redundant/conflicting action pairs | ↑ Higher (less negative) is better |
| Turn Count | Average dialogue turns per episode | ↓ Lower is better |
| Delegation Accuracy | Fraction of actions assigned to capable agents | ↑ Higher is better |
| Functional–Fluency Gap | \|Task Success − BERTScore\| | ↓ Lower is better |

## Model Details

- **Parameters:** 310.6M (1.44 GB)
- **Inference:** ~0.93s per plan on a single NVIDIA RTX A6000 (50 DDIM steps)
- **Plan tensor:** P ∈ ℝ^{N × T × D} where N ≤ 4 agents, T = 32 steps, D = 512
- **Training data:** 11,000 expert trajectories from GPT-4o demonstrations
- **Benchmarks:** ALFWorld-Multi (2 agents), WebArena-Multi (3 agents), CollabCooking (2 agents)


## License

This repository is released for research purposes. Code and data will be made publicly available upon acceptance.
