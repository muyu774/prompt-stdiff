#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs/revision_8gpu_logs/hybrid_mean
LOG="outputs/revision_8gpu_logs/hybrid_mean/gpu2_p04_mean_g03.log"
echo "[start] $(date '+%F %T') gpu2_p04_mean_g03" | tee -a "$LOG"
bash scripts/run_pems04.sh \
  --mode train \
  --config_file configs/pems04_hybrid_mean.yaml \
  --horizon_steps 12 \
  --history_steps 24 \
  --gpu_id 2 \
  --lr 2e-4 \
  --gamma 0.3 \
  --eval_interval 5 \
  --train_num_eval_samples 8 \
  --num_eval_samples 20 \
  --max_eval_batches 20 \
  --save_tag hybrid_mean_g03 | tee -a "$LOG"
echo "[done] $(date '+%F %T') gpu2_p04_mean_g03" | tee -a "$LOG"
