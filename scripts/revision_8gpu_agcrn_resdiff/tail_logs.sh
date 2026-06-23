#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
LOG_DIR="outputs/revision_8gpu_logs/agcrn_resdiff"
for f in "${LOG_DIR}"/gpu*.log; do
  [[ -f "${f}" ]] || continue
  echo "===== $(basename "${f}") ====="
  grep -E "Bare frozen AGCRN|Epoch [0-9]+ \\| val_mae=|horizon=12|Horizon 12|Traceback|ERROR|Non-finite|Killed|nan|\\[done\\]" "${f}" | tail -30 || true
done
