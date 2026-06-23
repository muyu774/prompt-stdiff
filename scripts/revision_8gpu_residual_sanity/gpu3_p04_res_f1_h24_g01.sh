#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
LOG_DIR="${LOG_DIR:-outputs/revision_8gpu_logs/residual_sanity}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/gpu3_p04_res_f1_h24_g01.log") 2>&1

echo "[start] $(date '+%F %T') gpu3_p04_res_f1_h24_g01"
bash scripts/run_pems04.sh \
  --mode train \
  --config_file configs/pems04_residual_f1_h24_g01.yaml \
  --horizon_steps 12 \
  --history_steps 24 \
  --gpu_id 3 \
  --eval_interval 5 \
  --train_num_eval_samples 8 \
  --num_eval_samples 20 \
  --max_eval_batches 20 \
  --save_tag p04_res_f1_h24_g01
echo "[done] $(date '+%F %T') gpu3_p04_res_f1_h24_g01"
