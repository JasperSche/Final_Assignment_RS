#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs

# Format:
# run_id dim layers epochs lr reg train_batch pred_batch
RUNS=(
  "final_768d4l_e300_hl120_a0p5_3seed 768 4 300 0.0008 0.00025 4096 32"
  "final_768d5l_e300_hl120_a0p5_3seed 768 5 300 0.0007 0.0003 4096 32"
  "final_1024d4l_e300_hl120_a0p5_3seed 1024 4 300 0.0006 0.0003 4096 24"
  "final_1024d5l_e300_hl120_a0p5_3seed 1024 5 300 0.0006 0.0003 4096 24"
)

for RUN in "${RUNS[@]}"; do
  read -r RUN_ID DIM LAYERS EPOCHS LR REG TRAIN_BATCH PRED_BATCH <<< "$RUN"

  echo "============================================================"
  echo "Starting $RUN_ID"
  echo "dim=$DIM layers=$LAYERS epochs=$EPOCHS lr=$LR reg=$REG"
  echo "============================================================"

  python train_submission.py \
    --data_dir data \
    --output_root outputs \
    --run_id "$RUN_ID" \
    --dim "$DIM" \
    --layers "$LAYERS" \
    --epochs "$EPOCHS" \
    --batch_size "$TRAIN_BATCH" \
    --lr "$LR" \
    --reg "$REG" \
    --seeds 42,43,44 \
    --recent_half_lives 120 \
    --alphas 0.5 \
    --blend_norm rrf \
    --prediction_batch_size "$PRED_BATCH"

  echo "Finished $RUN_ID"
done

echo "All bigger-model runs completed."
