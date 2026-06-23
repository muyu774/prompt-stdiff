#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/pems08.yaml"
CKPT=""
GPU_ID=0
NUM_EVAL_SAMPLES=20
MAX_EVAL_BATCHES=""
LATENCY_BATCH_SIZE=1
OUT_CSV="outputs/ddim_sweep/results.csv"
OUT_MD="outputs/ddim_sweep/RESULTS.md"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/sweep_ddim_sampling_steps.sh --config CONFIG --ckpt CKPT [--gpu_id N]
       [--num_eval_samples N] [--max_eval_batches N] [--latency_batch_size N]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --ckpt) CKPT="$2"; shift 2 ;;
    --gpu_id) GPU_ID="$2"; shift 2 ;;
    --num_eval_samples) NUM_EVAL_SAMPLES="$2"; shift 2 ;;
    --max_eval_batches) MAX_EVAL_BATCHES="$2"; shift 2 ;;
    --latency_batch_size) LATENCY_BATCH_SIZE="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] unknown arg: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "${CKPT}" ]]; then
  echo "[error] --ckpt is required"
  exit 1
fi

mkdir -p "$(dirname "${OUT_CSV}")"
rm -f "${OUT_CSV}" "${OUT_MD}"

for STEPS in 50 25 10 5; do
  echo "[sweep] DDIM sampling_steps=${STEPS}"
  CMD=(
    python scripts/run_experiment_and_record.py
    --config "${CONFIG}"
    --ckpt "${CKPT}"
    --gpu_id "${GPU_ID}"
    --sampler ddim
    --sampling_steps "${STEPS}"
    --num_eval_samples "${NUM_EVAL_SAMPLES}"
    --latency_batch_size "${LATENCY_BATCH_SIZE}"
    --method Prompt-STDiff
    --setting "ddim_s${STEPS}"
    --implementation ours
    --csv "${OUT_CSV}"
    --md "${OUT_MD}"
    --title "DDIM Quality-Speed Sweep"
  )
  if [[ -n "${MAX_EVAL_BATCHES}" ]]; then
    CMD+=(--max_eval_batches "${MAX_EVAL_BATCHES}")
  fi
  "${CMD[@]}"
done

python scripts/plot_ddim_quality_speed.py \
  --csv "${OUT_CSV}" \
  --out_png outputs/ddim_sweep/ddim_quality_speed.png \
  --out_pdf outputs/ddim_sweep/ddim_quality_speed.pdf
