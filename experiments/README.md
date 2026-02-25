# Experiment outputs will be saved here.

## Directory Structure

After training and evaluation:

```
experiments/
├── checkpoints/
│   ├── stage1_final.pt      # Diffusion model
│   ├── stage2_final.pt      # + Plan decoder
│   └── stage3_final.pt      # + Guidance classifiers
├── eval_results.json         # Main evaluation report
├── ablation/                 # Ablation study results
│   ├── no_joint_planning.json
│   ├── no_role_masking.json
│   ├── no_guidance.json
│   └── ...
└── wandb/                    # W&B logs (auto-generated)
```
