#!/usr/bin/env bash
set -euo pipefail

# Train/evaluate DiffSTG on canonical PeMS04/PeMS08 splits and export NPZ samples.
#
# Smoke:
#   SMOKE=1 GPU_PEMS04=6 GPU_PEMS08=7 bash scripts/launch_diffstg_canonical.sh launch
#
# Full:
#   GPU_PEMS04=6 GPU_PEMS08=7 EPOCHS=50 NSAMPLE=20 bash scripts/launch_diffstg_canonical.sh launch

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

cmd="${1:-launch}"
LATEST_LINK="outputs/revision_8gpu_logs/diffstg_latest"
if [[ -z "${LOG_ROOT:-}" ]]; then
  if [[ "${cmd}" == "launch" ]]; then
    LOG_ROOT="outputs/revision_8gpu_logs/diffstg_$(date +%Y%m%d_%H%M%S)"
  elif [[ -L "${LATEST_LINK}" ]]; then
    LOG_ROOT="outputs/revision_8gpu_logs/$(readlink "${LATEST_LINK}")"
  else
    LOG_ROOT="outputs/revision_8gpu_logs/diffstg_latest_missing"
  fi
fi

DIFFSTG_REPO="${DIFFSTG_REPO:-baselines/external_repos/DiffSTG}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-1e-4}"
NSAMPLE="${NSAMPLE:-20}"
VALID_INTERVAL="${VALID_INTERVAL:-5}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-200}"
SAMPLE_STEPS="${SAMPLE_STEPS:-40}"
HIDDEN_SIZE="${HIDDEN_SIZE:-32}"
SEED="${SEED:-2022}"
TAG="${TAG:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  EPOCHS="${SMOKE_EPOCHS:-1}"
  NSAMPLE="${SMOKE_NSAMPLE:-2}"
  DIFFUSION_STEPS="${SMOKE_DIFFUSION_STEPS:-20}"
  SAMPLE_STEPS="${SMOKE_SAMPLE_STEPS:-5}"
  MAX_TRAIN_BATCHES="--max_train_batches ${SMOKE_TRAIN_BATCHES:-2}"
  MAX_EVAL_BATCHES="--max_eval_batches ${SMOKE_EVAL_BATCHES:-2}"
else
  MAX_TRAIN_BATCHES=""
  MAX_EVAL_BATCHES=""
fi

if [[ "${cmd}" == "launch" ]]; then
  mkdir -p "${LOG_ROOT}"
fi

launch_one() {
  local dataset="$1"
  local gpu="$2"
  local config="configs/${dataset}.yaml"
  local canonical="outputs/canonical_setup/${dataset}_h12_t12.npz"
  local suffix="${TAG:+_${TAG}}"
  local name="diffstg_${dataset}${suffix}"
  local log_file="${LOG_ROOT}/${name}.log"
  local save_dir="outputs/prob_baselines/DiffSTG/${dataset}${suffix}_run"
  local out_npz="outputs/prob_baselines/DiffSTG/${dataset}${suffix}_samples.npz"

  if [[ ! -f "${canonical}" ]]; then
    echo "[missing] ${canonical}. Run scripts/export_canonical_setup.py first." >&2
    return 1
  fi
  if [[ ! -d "${DIFFSTG_REPO}" ]]; then
    echo "[missing] DiffSTG repo: ${DIFFSTG_REPO}" >&2
    return 1
  fi

  echo "[launch] ${name} gpu=${gpu}"
  (
    echo "[start] $(date '+%F %T') ${name}"
    python scripts/run_diffstg_canonical.py \
      --config "${config}" \
      --canonical_npz "${canonical}" \
      --diffstg_repo "${DIFFSTG_REPO}" \
      --gpu_id "${gpu}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --valid_interval "${VALID_INTERVAL}" \
      --hidden_size "${HIDDEN_SIZE}" \
      --seed "${SEED}" \
      --diffusion_steps "${DIFFUSION_STEPS}" \
      --sample_steps "${SAMPLE_STEPS}" \
      --nsample "${NSAMPLE}" \
      --save_dir "${save_dir}" \
      --out_npz "${out_npz}" \
      ${MAX_TRAIN_BATCHES} \
      ${MAX_EVAL_BATCHES} \
      ${EXTRA_ARGS}
    echo "[eval] ${name}"
    python scripts/eval_probabilistic_npz.py \
      --pred_npz "${out_npz}" \
      --config "${config}" \
      --method DiffSTG \
      --setting official \
      --implementation official \
      --space normalized \
      --device "cuda:${gpu}" \
      --eval_batch_size 8 \
      --title "Probabilistic Baseline Results"
    echo "[done] $(date '+%F %T') ${name}"
  ) >"${log_file}" 2>&1 &
  echo $! >"${LOG_ROOT}/${name}.pid"
}

status() {
  echo "[logs] ${LOG_ROOT}"
  for pid_file in "${LOG_ROOT}"/*.pid; do
    [[ -e "${pid_file}" ]] || continue
    name="$(basename "${pid_file}" .pid)"
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "RUNNING ${name} pid=${pid}"
    else
      echo "DONE/EXITED ${name} pid=${pid}"
    fi
  done
}

tail_logs() {
  for log_file in "${LOG_ROOT}"/*.log; do
    [[ -e "${log_file}" ]] || continue
    echo "===== $(basename "${log_file}") ====="
    grep -E "\\[epoch|val_loss|\\[export|Test metrics|Horizon|Traceback|ERROR|RuntimeError|ValueError|ModuleNotFound|\\[done\\]" "${log_file}" | tail -80 || true
  done
}

case "${cmd}" in
  launch)
    launch_one pems04 "${GPU_PEMS04:-6}"
    launch_one pems08 "${GPU_PEMS08:-7}"
    ln -sfn "$(basename "${LOG_ROOT}")" "${LATEST_LINK}"
    echo "[logs] ${LOG_ROOT}"
    ;;
  status)
    status
    ;;
  tail)
    tail_logs
    ;;
  *)
    echo "Usage: $0 {launch|status|tail}" >&2
    exit 2
    ;;
esac
