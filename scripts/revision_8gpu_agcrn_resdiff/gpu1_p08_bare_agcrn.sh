#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
LOG_DIR="outputs/revision_8gpu_logs/agcrn_resdiff"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/gpu1_p08_bare_agcrn.log"
echo "[start] $(date '+%F %T') gpu1_p08_bare_agcrn" | tee -a "${LOG}"
python scripts/eval_frozen_mean_predictor.py \
  --config configs/pems08_agcrn_resdiff.yaml \
  --gpu_id 1 \
  --output_csv outputs/agcrn_resdiff_results.csv \
  --results_md RESULTS.md 2>&1 | tee -a "${LOG}"
echo "[done] $(date '+%F %T') gpu1_p08_bare_agcrn" | tee -a "${LOG}"
