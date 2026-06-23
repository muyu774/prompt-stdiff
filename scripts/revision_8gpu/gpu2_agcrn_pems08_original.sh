#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
LOG_DIR="${LOG_DIR:-outputs/revision_8gpu_logs/manual}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/gpu2_agcrn_pems08_original.log") 2>&1

echo "[start] $(date '+%F %T') gpu2_agcrn_pems08_original"
python -m baselines.runners.run_agcrn \
  --config configs/pems08.yaml \
  --device cuda:2 \
  --agcrn_repo "${AGCRN_REPO:-baselines/external_repos/AGCRN}" \
  --input_feature_index 0 \
  --epochs "${AGCRN_EPOCHS:-50}" \
  --eval_interval 5 \
  --lr 2e-4 \
  --save_tag agcrn_original
echo "[done] $(date '+%F %T') gpu2_agcrn_pems08_original"
