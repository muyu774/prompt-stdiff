#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs/revision_8gpu_logs/hybrid_followup
LOG="outputs/revision_8gpu_logs/hybrid_followup/gpu1_p04_lr5e4_nosem.log"
echo "[start] $(date '+%F %T') gpu1_p04_lr5e4_nosem" | tee -a "$LOG"
bash scripts/run_pems04.sh \
  --mode train \
  --config_file configs/pems04_hybrid_mean.yaml \
  --horizon_steps 12 \
  --history_steps 24 \
  --gpu_id 1 \
  --lr 5e-4 \
  --gamma 0 \
  --eval_interval 5 \
  --train_num_eval_samples 8 \
  --num_eval_samples 20 \
  --max_eval_batches 20 \
  --disable_dynamic_semantic \
  --save_tag hybrid_follow_lr5e4_nosem | tee -a "$LOG"
echo "[done] $(date '+%F %T') gpu1_p04_lr5e4_nosem" | tee -a "$LOG"
