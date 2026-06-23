#!/usr/bin/env bash
set -euo pipefail

# Sweep gamma at evaluation time for an existing checkpoint.
# This is useful because current training objective does not directly depend on gamma.
#
# Usage:
#   bash scripts/sweep_gamma_eval.sh --ckpt outputs/checkpoints/pems03_h12_lowmae_a/best.pt --gpu_id 0

CKPT=""
GPU_ID=0
HORIZON_STEPS=12
HISTORY_STEPS=24
GAMMAS="0.0 0.05 0.1 0.2"
SAVE_TAG_PREFIX="gamma_sweep"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
      CKPT="$2"
      shift 2
      ;;
    --gpu_id)
      GPU_ID="$2"
      shift 2
      ;;
    --horizon_steps)
      HORIZON_STEPS="$2"
      shift 2
      ;;
    --history_steps)
      HISTORY_STEPS="$2"
      shift 2
      ;;
    --gammas)
      GAMMAS="$2"
      shift 2
      ;;
    --save_tag_prefix)
      SAVE_TAG_PREFIX="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash scripts/sweep_gamma_eval.sh --ckpt PATH [--gpu_id 0] [--horizon_steps 12] [--history_steps 24]
                                  [--gammas "0.0 0.05 0.1 0.2"] [--save_tag_prefix STR]
EOF
      exit 0
      ;;
    *)
      echo "[error] Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ -z "${CKPT}" ]]; then
  echo "[error] --ckpt is required"
  exit 1
fi

for g in ${GAMMAS}; do
  tag="${SAVE_TAG_PREFIX}_g${g}"
  echo "========== gamma=${g} =========="
  bash scripts/run_pems03.sh \
    --mode eval \
    --horizon_steps "${HORIZON_STEPS}" \
    --history_steps "${HISTORY_STEPS}" \
    --gpu_id "${GPU_ID}" \
    --gamma "${g}" \
    --save_tag "${tag}" \
    --ckpt "${CKPT}" \
    --num_eval_samples 20
done

