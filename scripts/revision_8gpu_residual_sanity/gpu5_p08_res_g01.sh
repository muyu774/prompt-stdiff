#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
LOG_DIR="${LOG_DIR:-outputs/revision_8gpu_logs/residual_sanity}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/gpu5_p08_res_g01.log") 2>&1

echo "[start] $(date '+%F %T') gpu5_p08_res_g01"
bash scripts/run_pems08.sh \
  --mode train \
  --config_file configs/pems08_residual_g01.yaml \
  --horizon_steps 12 \
  --history_steps 24 \
  --gpu_id 5 \
  --eval_interval 5 \
  --train_num_eval_samples 8 \
  --num_eval_samples 20 \
  --max_eval_batches 20 \
  --save_tag p08_res_g01
echo "[done] $(date '+%F %T') gpu5_p08_res_g01"
