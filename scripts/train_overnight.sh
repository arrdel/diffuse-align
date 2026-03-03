#!/bin/bash
# ============================================================
# DiffuseAlign — Overnight Training on GPUs 4,5,6,7
# 
# Launches 4 independent training processes (Local SGD),
# one per GPU. At the end, averages final checkpoints.
# ============================================================

set -euo pipefail

PROJECT_DIR="/home/achinda1/projects/diffuse-align"
cd "$PROJECT_DIR"

# Use the conda env Python directly (no need to activate)
PYTHON="/home/achinda1/.conda/envs/diffusealign/bin/python"

GPU_IDS=(4 5 6 7)
NUM_GPUS=${#GPU_IDS[@]}
CONFIG="configs/default.yaml"
OUTPUT_DIR="experiments/checkpoints"
BATCH=8
ACCUM=8
MAX_STEPS=100000

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

echo "============================================================"
echo "🚀 DiffuseAlign Stage 1 — Overnight Training"
echo "   Date: $(date)"
echo "   GPUs: ${GPU_IDS[*]} ($NUM_GPUS total)"
echo "   Micro-batch: $BATCH, Accum: $ACCUM → Effective: $((BATCH * ACCUM))"
echo "   Max steps: $MAX_STEPS"
echo "   Output: $OUTPUT_DIR"
echo "   Conda env: diffusealign"
echo "============================================================"
echo ""

mkdir -p "$OUTPUT_DIR"

# Clean up on exit — send SIGTERM so each process saves a checkpoint
PIDS=()
cleanup() {
    echo ""
    echo "Received signal — sending SIGTERM to training processes for graceful checkpoint..."
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    echo "Waiting for processes to save checkpoints and exit..."
    wait 2>/dev/null
    echo "All processes terminated. Checkpoints saved."
    echo "To resume tomorrow: bash scripts/train_overnight.sh"
}
trap cleanup SIGINT SIGTERM

# Launch one training process per GPU
for GPU_ID in "${GPU_IDS[@]}"; do
    LOG_FILE="$OUTPUT_DIR/gpu${GPU_ID}.log"
    echo "[$(date +%H:%M:%S)] Starting training on GPU $GPU_ID → $LOG_FILE"
    CUDA_VISIBLE_DEVICES=$GPU_ID \
        $PYTHON scripts/train_single_gpu.py \
            --config "$CONFIG" \
            --gpu 0 \
            --run_name "gpu${GPU_ID}" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size "$BATCH" \
            --grad_accum "$ACCUM" \
            --max_steps "$MAX_STEPS" \
            --seed $((42 + GPU_ID)) \
        > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All $NUM_GPUS processes launched. PIDs: ${PIDS[*]}"
echo "Monitor with:  tail -f $OUTPUT_DIR/gpu4.log"
echo ""

# Wait and report
FAILED=0
for i in "${!PIDS[@]}"; do
    GPU_ID="${GPU_IDS[$i]}"
    if ! wait "${PIDS[$i]}"; then
        echo "❌ [$(date +%H:%M:%S)] GPU $GPU_ID FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    else
        echo "✅ [$(date +%H:%M:%S)] GPU $GPU_ID completed"
    fi
done

echo ""
echo "============================================================"

if [ "$FAILED" -gt 0 ]; then
    echo "⚠️  $FAILED/$NUM_GPUS processes failed."
    echo "Check logs: $OUTPUT_DIR/gpu*.log"
else
    echo "All $NUM_GPUS processes completed successfully!"
    echo ""
    echo "Averaging final checkpoints..."
    $PYTHON -c "
import torch
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
gpu_ids = [4, 5, 6, 7]
finals = [output_dir / f'gpu{gid}' / 'stage1_final.pt' for gid in gpu_ids]
finals = [f for f in finals if f.exists()]

if not finals:
    print('No final checkpoints found.')
    exit(0)

print(f'Found {len(finals)} checkpoints: {[str(f) for f in finals]}')
avg_state = None
for f in finals:
    sd = torch.load(f, map_location='cpu', weights_only=False)
    if avg_state is None:
        avg_state = {k: v.float() for k, v in sd.items()}
    else:
        for k in avg_state:
            avg_state[k] += sd[k].float()

for k in avg_state:
    avg_state[k] /= len(finals)

out_path = output_dir / 'stage1_averaged.pt'
torch.save(avg_state, out_path)
print(f'✅ Averaged model saved to: {out_path}')
"
fi

echo ""
echo "Finished at: $(date)"
echo "============================================================"
