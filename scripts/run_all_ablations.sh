#!/usr/bin/env bash
# Run ablation across all 5 seeds sequentially on one GPU
set -euo pipefail

cd /home/achinda1/projects/diffuse-align
CONDA="/home/achinda1/.conda/envs/diffusealign/bin/python"
CKPT="experiments/checkpoints/diffusealign_final.pt"
GPU="cuda:6"
EPISODES=200

for SEED in 42 123 456 789 2024; do
    OUT="experiments/scaled_ablation/seed_${SEED}"
    mkdir -p "$OUT"
    echo "=== Ablation seed=$SEED on $GPU ==="
    $CONDA scripts/ablation.py \
        --checkpoint "$CKPT" \
        --num_episodes $EPISODES \
        --device "$GPU" \
        --seed $SEED \
        --output_dir "$OUT" \
        --decode_utterances
    echo "=== Seed $SEED done ==="
done

echo "All ablation seeds complete. Aggregating..."
$CONDA scripts/aggregate_ablation.py \
    --input_dir experiments/scaled_ablation \
    --output experiments/scaled_ablation/aggregated.json
echo "Done."
