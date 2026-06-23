#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"
LOG_DIR="${LOG_DIR:-outputs/revision_8gpu_logs/manual}"
mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/gpu6_ddim_pems08_sweep.log") 2>&1

echo "[start] $(date '+%F %T') gpu6_ddim_pems08_sweep"
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
for S in 50 25 10 5; do
  python scripts/run_experiment_and_record.py \
    --config configs/pems08.yaml \
    --ckpt "${CKPT}" \
    --gpu_id 6 \
    --sampler ddim \
    --sampling_steps "${S}" \
    --num_eval_samples 100 \
    --method Prompt-STDiff \
    --setting "ddim_S${S}" \
    --title "DDIM Quality-Speed Sweep"
done
echo "[done] $(date '+%F %T') gpu6_ddim_pems08_sweep"
