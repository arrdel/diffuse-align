#!/usr/bin/env bash
# run_max_gpu.sh -- Maximum GPU utilization launcher
# Each process: ~2.5 GB VRAM, ~30% compute
# Target: 3 processes/GPU = ~7.5 GB, ~90% compute
# 8 GPUs x 3 slots = 24 concurrent slots
#
# Eval:     5 seeds on GPUs 0-1 (3+2)
# Ablation: 40 runs on GPUs 2-7 (round-robin, throttled)
set -euo pipefail

cd /home/achinda1/projects/diffuse-align

PY="/home/achinda1/.conda/envs/diffusealign/bin/python"
CKPT="experiments/checkpoints/diffusealign_final.pt"
EVAL_EPS=500
ABL_EPS=200
MAX_JOBS=24

mkdir -p experiments/scaled experiments/scaled_ablation

echo "=== MAX GPU UTILIZATION ==="
echo "  Eval: 5 seeds x $EVAL_EPS episodes (GPUs 0-1)"
echo "  Ablation: 40 runs x $ABL_EPS episodes (GPUs 2-7)"
echo ""

# --- Scaled eval: 5 seeds packed onto GPUs 0-1 ---
echo ">> Launching scaled evaluation..."
idx=0
for pair in "42:0" "123:0" "456:0" "789:1" "2024:1"; do
    SEED="${pair%%:*}"
    GPU="${pair##*:}"
    echo "  eval seed=$SEED -> cuda:$GPU"
    $PY scripts/scaled_eval.py \
        --checkpoint "$CKPT" \
        --num_episodes $EVAL_EPS \
        --seed "$SEED" \
        --device "cuda:$GPU" \
        --decode_utterances \
        --output "experiments/scaled/seed_${SEED}.json" \
        > "experiments/scaled/seed_${SEED}.log" 2>&1 &
done

# --- Ablation: 40 runs on GPUs 2-7 ---
echo ""
echo ">> Launching ablation (5 seeds x 8 conditions)..."
SLOT=0
for SEED in 42 123 456 789 2024; do
    OUT_DIR="experiments/scaled_ablation/seed_${SEED}"
    mkdir -p "$OUT_DIR"

    for ABL in full no_guidance no_coordination no_efficiency no_safety task_only low_cfg high_cfg; do
        # Round-robin GPUs 2-7
        GPU=$(( (SLOT % 6) + 2 ))
        SLOT=$((SLOT + 1))

        echo "  ablation $ABL seed=$SEED -> cuda:$GPU"
        $PY scripts/ablation.py \
            --checkpoint "$CKPT" \
            --num_episodes $ABL_EPS \
            --device "cuda:$GPU" \
            --seed "$SEED" \
            --output_dir "$OUT_DIR" \
            --ablations "$ABL" \
            --decode_utterances \
            > "${OUT_DIR}/${ABL}.log" 2>&1 &

        # Throttle: keep at most MAX_JOBS running
        while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
            sleep 2
        done
    done
done

echo ""
echo "=== All jobs launched. Waiting... ==="
wait
echo "=== All jobs done. Aggregating... ==="

# --- Aggregate eval ---
$PY scripts/scaled_eval.py \
    --aggregate \
    --input_dir experiments/scaled \
    --output experiments/scaled/aggregated.json

# --- Assemble per-seed ablation summaries ---
for SEED in 42 123 456 789 2024; do
    OUT_DIR="experiments/scaled_ablation/seed_${SEED}"
    $PY -c "
import json
from pathlib import Path
out_dir = Path('$OUT_DIR')
summary = {}
for f in sorted(out_dir.glob('*.json')):
    if f.name == 'ablation_summary.json':
        continue
    try:
        data = json.loads(f.read_text())
        summary[data.get('ablation', f.stem)] = data
    except Exception:
        pass
if summary:
    (out_dir / 'ablation_summary.json').write_text(json.dumps(summary, indent=2))
    print(f'  seed $SEED: {len(summary)} conditions assembled')
"
done

# --- Aggregate across ablation seeds ---
$PY scripts/aggregate_ablation.py \
    --input_dir experiments/scaled_ablation \
    --output experiments/scaled_ablation/aggregated.json

echo ""
echo "=== ALL DONE -- eval + ablation aggregated ==="
