#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs/revision_8gpu_logs/hybrid_mean
LOG="outputs/revision_8gpu_logs/hybrid_mean/gpu7_p08_mean_nosem.log"
echo "[start] $(date '+%F %T') gpu7_p08_mean_nosem" | tee -a "$LOG"
bash scripts/run_pems08.sh \
  --mode train \
  --config_file configs/pems08_hybrid_mean.yaml \
  --horizon_steps 12 \
  --history_steps 24 \
  --gpu_id 7 \
  --lr 2e-4 \
  --gamma 0 \
  --eval_interval 5 \
  --train_num_eval_samples 8 \
  --num_eval_samples 20 \
  --max_eval_batches 20 \
  --disable_dynamic_semantic \
  --save_tag hybrid_mean_nosem | tee -a "$LOG"
echo "[done] $(date '+%F %T') gpu7_p08_mean_nosem" | tee -a "$LOG"
