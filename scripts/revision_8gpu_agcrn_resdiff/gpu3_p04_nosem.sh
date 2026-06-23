#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
LOG_DIR="outputs/revision_8gpu_logs/agcrn_resdiff"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/gpu3_p04_nosem.log"
echo "[start] $(date '+%F %T') gpu3_p04_nosem" | tee -a "${LOG}"
bash scripts/run_pems04.sh \
  --mode train \
  --config_file configs/pems04_agcrn_resdiff_nosem.yaml \
  --horizon_steps 12 \
  --history_steps 24 \
  --gpu_id 3 \
  --lr 1e-3 \
  --eval_interval 5 \
  --train_num_eval_samples 8 \
  --num_eval_samples 20 \
  --max_eval_batches 20 \
  --save_tag nosem 2>&1 | tee -a "${LOG}"
echo "[done] $(date '+%F %T') gpu3_p04_nosem" | tee -a "${LOG}"
