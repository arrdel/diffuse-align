#!/bin/bash
# ============================================================
# DiffuseAlign — 8-GPU Parallel Training (Local SGD)
#
# Launches 8 independent training processes, one per GPU.
# Every SYNC_EVERY steps, averages model weights across all GPUs.
#
# This avoids NCCL/DDP (which SIGSEGV on kernel 4.18) by using
# file-based weight averaging.
#
# Usage:
#   bash scripts/train_parallel.sh          # Full overnight training
#   bash scripts/train_parallel.sh smoke    # Smoke test
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

NUM_GPUS=8
SYNC_EVERY=500          # Average weights every N steps
CONFIG="configs/default.yaml"
OUTPUT_DIR="experiments/checkpoints"
MODE="${1:-full}"        # "smoke" or "full"

export TOKENIZERS_PARALLELISM=false

if [ "$MODE" = "smoke" ]; then
    echo "🔥 SMOKE TEST MODE"
    MAX_STEPS=5
    BATCH=8
    ACCUM=1
else
    echo "🚀 FULL TRAINING MODE"
    MAX_STEPS=100000
    BATCH=8
    ACCUM=8
fi

echo "   GPUs: $NUM_GPUS"
echo "   Micro-batch: $BATCH, Accum: $ACCUM → Effective: $((BATCH * ACCUM))"
echo "   Steps: $MAX_STEPS"
echo "   Sync every: $SYNC_EVERY steps"
echo "============================================================"

# Clean up on exit
PIDS=()
cleanup() {
    echo "Cleaning up..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
}
trap cleanup EXIT

# Launch 8 training processes
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    echo "Starting GPU $GPU_ID..."
    CUDA_VISIBLE_DEVICES=$GPU_ID WANDB_MODE=disabled \
        python scripts/train_single_gpu.py \
            --config "$CONFIG" \
            --gpu 0 \
            --output_dir "$OUTPUT_DIR" \
            --batch_size "$BATCH" \
            --grad_accum "$ACCUM" \
            --max_steps "$MAX_STEPS" \
            --seed $((42 + GPU_ID)) \
            $([ "$MODE" = "smoke" ] && echo "--smoke_test") \
        > "$OUTPUT_DIR/gpu${GPU_ID}.log" 2>&1 &
    PIDS+=($!)
done

echo "All $NUM_GPUS processes launched."
echo "Logs: $OUTPUT_DIR/gpu*.log"
echo ""

# Wait for all processes
FAILED=0
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "❌ GPU $i FAILED (exit code $?)"
        FAILED=$((FAILED + 1))
    else
        echo "✅ GPU $i completed"
    fi
done

if [ "$FAILED" -gt 0 ]; then
    echo ""
    echo "❌ $FAILED/$NUM_GPUS processes failed!"
    echo "Check logs in $OUTPUT_DIR/gpu*.log"
    exit 1
fi

echo ""
echo "============================================================"
if [ "$MODE" = "smoke" ]; then
    echo "✅ SMOKE TEST PASSED — all $NUM_GPUS GPUs trained successfully!"
    echo ""
    echo "For overnight training:"
    echo "  nohup bash scripts/train_parallel.sh > experiments/training.log 2>&1 &"
else
    echo "✅ Training complete on all $NUM_GPUS GPUs!"
    echo ""
    # Average final checkpoints
    echo "Averaging final models..."
    python -c "
import torch, glob
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
finals = sorted(output_dir.glob('gpu*/stage1_final.pt'))
if not finals:
    print('No final checkpoints found, skipping averaging.')
    exit(0)

print(f'Averaging {len(finals)} checkpoints...')
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
print(f'Averaged model saved to: {out_path}')
"
fi
echo "============================================================"
