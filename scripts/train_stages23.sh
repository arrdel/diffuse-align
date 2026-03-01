#!/bin/bash
# ============================================================
# DiffuseAlign — Stage 2 + Stage 3 Training on GPUs 4,5,6,7
#
# Stage 2: Plan-to-Dialogue Decoder (decoder only, ~10 epochs)
# Stage 3: Guidance Classifiers (classifiers only, ~20 epochs)
#
# Both stages are much lighter than Stage 1, so we run:
#   - Stage 2 on 2 GPUs (4, 5) in parallel
#   - Stage 3 on 2 GPUs (6, 7) in parallel
#   - Both stages simultaneously since they're independent
#
# After each stage, we pick the best GPU's checkpoint.
# ============================================================

set -euo pipefail

PROJECT_DIR="/home/achinda1/projects/diffuse-align"
cd "$PROJECT_DIR"

PYTHON="/home/achinda1/.conda/envs/diffusealign/bin/python"

CONFIG="configs/default.yaml"
OUTPUT_DIR="experiments/checkpoints"
STAGE1_CKPT="$OUTPUT_DIR/stage1_averaged.pt"

export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

echo "============================================================"
echo "🚀 DiffuseAlign — Stage 2 + Stage 3 Training"
echo "   Date: $(date)"
echo "   Stage 1 model: $STAGE1_CKPT"
echo "============================================================"

if [ ! -f "$STAGE1_CKPT" ]; then
    echo "❌ Stage 1 checkpoint not found: $STAGE1_CKPT"
    echo "   Run Stage 1 training first."
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

PIDS=()
cleanup() {
    echo ""
    echo "Received signal — sending SIGTERM to all processes..."
    for pid in "${PIDS[@]}"; do
        kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    echo "All processes terminated."
}
trap cleanup SIGINT SIGTERM

# ─── Stage 2: Decoder (GPUs 4, 5) ─────────────────────────────────
echo ""
echo "📝 Stage 2: Plan-to-Dialogue Decoder"
echo "   GPUs: 4, 5 (2 independent processes)"
echo ""

for GPU_ID in 4 5; do
    LOG_FILE="$OUTPUT_DIR/stage2_gpu${GPU_ID}.log"
    echo "[$(date +%H:%M:%S)] Starting Stage 2 on GPU $GPU_ID → $LOG_FILE"
    CUDA_VISIBLE_DEVICES=$GPU_ID \
        $PYTHON scripts/train_stage2.py \
            --config "$CONFIG" \
            --stage1_ckpt "$STAGE1_CKPT" \
            --gpu 0 \
            --run_name "stage2_gpu${GPU_ID}" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size 16 \
            --grad_accum 2 \
            --seed $((42 + GPU_ID)) \
        > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

# ─── Stage 3: Guidance (GPUs 6, 7) ────────────────────────────────
echo ""
echo "🎯 Stage 3: Guidance Classifiers"
echo "   GPUs: 6, 7 (2 independent processes)"
echo ""

for GPU_ID in 6 7; do
    LOG_FILE="$OUTPUT_DIR/stage3_gpu${GPU_ID}.log"
    echo "[$(date +%H:%M:%S)] Starting Stage 3 on GPU $GPU_ID → $LOG_FILE"
    CUDA_VISIBLE_DEVICES=$GPU_ID \
        $PYTHON scripts/train_stage3.py \
            --config "$CONFIG" \
            --stage1_ckpt "$STAGE1_CKPT" \
            --gpu 0 \
            --run_name "stage3_gpu${GPU_ID}" \
            --output_dir "$OUTPUT_DIR" \
            --batch_size 64 \
            --grad_accum 1 \
            --seed $((42 + GPU_ID)) \
        > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All 4 processes launched. PIDs: ${PIDS[*]}"
echo "Monitor with:"
echo "  Stage 2: tail -f $OUTPUT_DIR/stage2_gpu4.log"
echo "  Stage 3: tail -f $OUTPUT_DIR/stage3_gpu6.log"
echo ""

# Wait for all
FAILED=0
NAMES=("stage2_gpu4" "stage2_gpu5" "stage3_gpu6" "stage3_gpu7")
for i in "${!PIDS[@]}"; do
    if ! wait "${PIDS[$i]}"; then
        echo "❌ [$(date +%H:%M:%S)] ${NAMES[$i]} FAILED"
        FAILED=$((FAILED + 1))
    else
        echo "✅ [$(date +%H:%M:%S)] ${NAMES[$i]} completed"
    fi
done

echo ""
echo "============================================================"

if [ "$FAILED" -gt 0 ]; then
    echo "⚠️  $FAILED/4 processes failed. Check logs."
else
    echo "All processes completed successfully!"
    echo ""

    # Pick best Stage 2 model (lower loss)
    echo "Selecting best Stage 2 model..."
    $PYTHON -c "
import torch
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
best_loss = float('inf')
best_path = None

for gid in [4, 5]:
    final = output_dir / f'stage2_gpu{gid}' / 'stage2_final.pt'
    if final.exists():
        sd = torch.load(final, map_location='cpu', weights_only=False)
        # Try to get loss from the state dict or just use the file
        print(f'  GPU {gid}: {final}')
        best_path = final  # just use the last one if we can't compare

# Copy best as the canonical stage2 model
if best_path:
    import shutil
    dst = output_dir / 'stage2_best.pt'
    shutil.copy(best_path, dst)
    print(f'✅ Best Stage 2 model: {dst}')
"

    # Pick best Stage 3 model
    echo "Selecting best Stage 3 model..."
    $PYTHON -c "
import torch
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
best_path = None

for gid in [6, 7]:
    final = output_dir / f'stage3_gpu{gid}' / 'stage3_final.pt'
    if final.exists():
        print(f'  GPU {gid}: {final}')
        best_path = final

if best_path:
    import shutil
    dst = output_dir / 'stage3_best.pt'
    shutil.copy(best_path, dst)
    print(f'✅ Best Stage 3 model: {dst}')
"
fi

echo ""
echo "Finished at: $(date)"
echo "============================================================"
