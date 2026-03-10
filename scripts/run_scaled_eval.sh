#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# run_scaled_eval.sh — Launch scaled evaluation across multiple GPUs
#
# Layout (8 GPUs):
#   GPU 0-4: Scaled eval seeds 42 123 456 789 2024  (500 ep each)
#   GPU 5:   Baselines (CPU-only, but use GPU 5 slot)
#   GPU 6:   Scaled ablation (sequential, single GPU)
#   GPU 7:   Reserved / spare
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

cd /home/achinda1/projects/diffuse-align

CONDA_ENV="/home/achinda1/.conda/envs/diffusealign/bin/python"
CHECKPOINT="experiments/checkpoints/diffusealign_final.pt"
NUM_EPISODES=500
ABLATION_EPISODES=200
OUTPUT_DIR="experiments/scaled"

mkdir -p "$OUTPUT_DIR"
mkdir -p experiments/scaled_ablation

SEEDS=(42 123 456 789 2024)

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  DiffuseAlign — Scaled Evaluation Launch                    ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  Episodes per seed:  $NUM_EPISODES                              ║"
echo "║  Seeds:              ${SEEDS[*]}               ║"
echo "║  Ablation episodes:  $ABLATION_EPISODES                              ║"
echo "║  Checkpoint:         diffusealign_final.pt                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Launch scaled eval (one seed per GPU) ─────────────────────────
echo "▶ Launching scaled evaluation (5 seeds × ${NUM_EPISODES} episodes)..."
for i in "${!SEEDS[@]}"; do
    SEED="${SEEDS[$i]}"
    GPU="$i"
    LOG="${OUTPUT_DIR}/seed_${SEED}.log"

    echo "  GPU $GPU → seed $SEED (log: $LOG)"
    nohup $CONDA_ENV scripts/scaled_eval.py \
        --checkpoint "$CHECKPOINT" \
        --num_episodes "$NUM_EPISODES" \
        --seed "$SEED" \
        --device "cuda:$GPU" \
        --decode_utterances \
        --output "${OUTPUT_DIR}/seed_${SEED}.json" \
        > "$LOG" 2>&1 &
done

# ── 2. Launch baselines (CPU-only, runs fast) ────────────────────────
echo ""
echo "▶ Launching baselines (${NUM_EPISODES} episodes × 5 seeds)..."
BASELINE_LOG="experiments/baseline_run.log"
nohup $CONDA_ENV scripts/run_baselines.py \
    --num_episodes "$NUM_EPISODES" \
    --seeds "${SEEDS[@]}" \
    --output experiments/baseline_results.json \
    > "$BASELINE_LOG" 2>&1 &
echo "  Baselines → $BASELINE_LOG"

# ── 3. Launch scaled ablation (sequential on GPU 6) ──────────────────
echo ""
echo "▶ Launching scaled ablation (${ABLATION_EPISODES} ep × 5 seeds) on GPU 6..."
for SEED in "${SEEDS[@]}"; do
    ABL_LOG="experiments/scaled_ablation/ablation_seed_${SEED}.log"
    ABL_OUT="experiments/scaled_ablation/seed_${SEED}"
    mkdir -p "$ABL_OUT"

    echo "  Ablation seed $SEED → $ABL_LOG"
    # These run sequentially per seed (8 conditions each)
    nohup $CONDA_ENV scripts/ablation.py \
        --checkpoint "$CHECKPOINT" \
        --num_episodes "$ABLATION_EPISODES" \
        --device "cuda:6" \
        --seed "$SEED" \
        --output_dir "$ABL_OUT" \
        --decode_utterances \
        > "$ABL_LOG" 2>&1 &

    # Wait for this seed's ablation to finish before starting next
    # (they share GPU 6)
    wait $!
done &  # The whole loop runs in background

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "All jobs launched.  Monitor with:"
echo "  tail -f ${OUTPUT_DIR}/seed_42.log"
echo "  tail -f experiments/baseline_run.log"
echo "  tail -f experiments/scaled_ablation/ablation_seed_42.log"
echo ""
echo "When all done, aggregate:"
echo "  python scripts/scaled_eval.py --aggregate --input_dir experiments/scaled --output experiments/scaled/aggregated.json"
echo "  python scripts/aggregate_ablation.py"
echo "═══════════════════════════════════════════════════════════════"
