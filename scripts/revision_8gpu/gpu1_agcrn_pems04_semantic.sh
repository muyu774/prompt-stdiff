#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
LOG_DIR="${LOG_DIR:-outputs/revision_8gpu_logs/manual}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/gpu1_agcrn_pems04_semantic.log") 2>&1

echo "[start] $(date '+%F %T') gpu1_agcrn_pems04_semantic"
python -m baselines.runners.run_agcrn \
  --config configs/pems04.yaml \
  --device cuda:1 \
  --agcrn_repo "${AGCRN_REPO:-baselines/external_repos/AGCRN}" \
  --input_feature_index 0 \
  --use_semantic \
  --semantic_proj_dim "${SEM_PROJ_DIM:-128}" \
  --epochs "${AGCRN_EPOCHS:-50}" \
  --eval_interval 5 \
  --lr 1e-3 \
  --save_tag agcrn_semantic
echo "[done] $(date '+%F %T') gpu1_agcrn_pems04_semantic"
