#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
LOG_DIR="${LOG_DIR:-outputs/revision_8gpu_logs/manual}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/gpu7_sensitivity_pems08_h12.log") 2>&1

echo "[start] $(date '+%F %T') gpu7_sensitivity_pems08_h12"
CKPT="${PEMS08_CKPT:-}"
if [[ -z "${CKPT}" ]]; then
  for candidate in     "outputs/checkpoints/pems08_h12_${PROMPT_EPOCH_TAG:-revision_h12}/best.pt"     "outputs/checkpoints/pems08/best.pt"     "outputs/checkpoints/pems08_full/best.pt"; do
    if [[ -f "${candidate}" ]]; then
      CKPT="${candidate}"
      break
    fi
  done
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  CKPT="$(find outputs/checkpoints -path '*pems08*' -type f -name 'best.pt' -print 2>/dev/null | sort | tail -n 1 || true)"
fi
if [[ -z "${CKPT}" || ! -f "${CKPT}" ]]; then
  echo "[skip] no PeMS08 checkpoint found. Set PEMS08_CKPT=/path/to/best.pt and rerun."
  echo "[debug] available checkpoints:"
  find outputs/checkpoints -type f -name '*.pt' -print 2>/dev/null | sort || true
  exit 0
fi
echo "[ckpt] ${CKPT}"
python scripts/sweep_hyperparam_sensitivity.py \
  --config configs/pems08.yaml \
  --ckpt "${CKPT}" \
  --out_dir outputs/sensitivity/pems08_h12_revision \
  --gpu_id 7 \
  --num_eval_samples 100 \
  --max_eval_batches 20
echo "[done] $(date '+%F %T') gpu7_sensitivity_pems08_h12"
