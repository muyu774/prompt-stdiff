#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
LOG_DIR="${LOG_DIR:-outputs/revision_8gpu_logs/manual}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/gpu4_prompt_pems04_h12.log") 2>&1

echo "[start] $(date '+%F %T') gpu4_prompt_pems04_h12"
bash scripts/run_pems04.sh \
  --mode train \
  --horizon_steps 12 \
  --history_steps 12 \
  --gpu_id 4 \
  --lr 1e-3 \
  --gamma 0 \
  --eval_interval 5 \
  --train_num_eval_samples 8 \
  --num_eval_samples 20 \
  --max_eval_batches 20 \
  --save_tag "${PROMPT_EPOCH_TAG:-revision_h12}"
echo "[done] $(date '+%F %T') gpu4_prompt_pems04_h12"
