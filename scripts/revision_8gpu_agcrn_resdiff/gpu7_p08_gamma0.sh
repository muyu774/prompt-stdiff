#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
LOG_DIR="outputs/revision_8gpu_logs/agcrn_resdiff"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/gpu7_p08_gamma0.log"
echo "[start] $(date '+%F %T') gpu7_p08_gamma0" | tee -a "${LOG}"
bash scripts/run_pems08.sh \
  --mode train \
  --config_file configs/pems08_agcrn_resdiff.yaml \
  --horizon_steps 12 \
  --history_steps 24 \
  --gpu_id 7 \
  --lr 1e-3 \
  --gamma 0 \
  --eval_interval 5 \
  --train_num_eval_samples 8 \
  --num_eval_samples 20 \
  --max_eval_batches 20 \
  --save_tag gamma0 2>&1 | tee -a "${LOG}"
echo "[done] $(date '+%F %T') gpu7_p08_gamma0" | tee -a "${LOG}"
