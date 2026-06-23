#!/usr/bin/env bash
set -euo pipefail

# Evaluate exported probabilistic baseline NPZ files on 8 GPUs.
#
# Expected NPZ paths, checked in this order:
#   ${PRED_ROOT}/${method}/${dataset}_samples.npz
#   ${PRED_ROOT}/${method}_${dataset}_samples.npz
#   ${PRED_ROOT}/${method}/${dataset}.npz
#
# Each NPZ should contain one sample array key among:
#   samples, pred_samples, preds, forecasts, prediction
# with shape [S,B,H,N,F] or [B,S,H,N,F].
#
# Example:
#   METHODS="CSDI DiffSTG PriSTI SpecSTG" SPACE=normalized \
#     bash scripts/launch_prob_baseline_eval_8gpu.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

METHODS="${METHODS:-CSDI DiffSTG PriSTI SpecSTG}"
PRED_ROOT="${PRED_ROOT:-outputs/prob_baselines}"
LATEST_LINK="outputs/revision_8gpu_logs/prob_baselines_latest"
SPACE="${SPACE:-normalized}"
SETTING="${SETTING:-official}"
IMPLEMENTATION="${IMPLEMENTATION:-official}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
SAMPLE_AXIS="${SAMPLE_AXIS:-auto}"
SPLIT="${SPLIT:-test}"
CSV="${CSV:-outputs/results.csv}"
MD="${MD:-RESULTS.md}"
TITLE="${TITLE:-Probabilistic Baseline Results}"

cmd="${1:-launch}"
if [[ -z "${LOG_ROOT:-}" ]]; then
  if [[ "${cmd}" == "launch" ]]; then
    LOG_ROOT="outputs/revision_8gpu_logs/prob_baselines_$(date +%Y%m%d_%H%M%S)"
  elif [[ -L "${LATEST_LINK}" ]]; then
    LOG_ROOT="outputs/revision_8gpu_logs/$(readlink "${LATEST_LINK}")"
  else
    LOG_ROOT="outputs/revision_8gpu_logs/prob_baselines_latest_missing"
  fi
fi

if [[ "${cmd}" == "launch" ]]; then
  mkdir -p "${LOG_ROOT}"
fi

find_pred() {
  local method="$1"
  local dataset="$2"
  local candidates=(
    "${PRED_ROOT}/${method}/${dataset}_samples.npz"
    "${PRED_ROOT}/${method}_${dataset}_samples.npz"
    "${PRED_ROOT}/${method}/${dataset}.npz"
  )
  local p
  for p in "${candidates[@]}"; do
    if [[ -f "${p}" ]]; then
      echo "${p}"
      return 0
    fi
  done
  return 1
}

launch_eval() {
  local gpu="$1"
  local method="$2"
  local dataset="$3"
  local config="configs/${dataset}.yaml"
  local pred=""
  local name="${method}_${dataset}"
  local log_file="${LOG_ROOT}/${name}.log"

  if ! pred="$(find_pred "${method}" "${dataset}")"; then
    echo "[skip] ${name}: no NPZ under ${PRED_ROOT}"
    echo "[skip] ${name}: no NPZ under ${PRED_ROOT}" >"${log_file}"
    return 0
  fi

  echo "[launch] ${name} gpu=${gpu} pred=${pred}"
  (
    echo "[start] $(date '+%F %T') ${name}"
    echo "[pred] ${pred}"
    python scripts/eval_probabilistic_npz.py \
      --pred_npz "${pred}" \
      --config "${config}" \
      --method "${method}" \
      --setting "${SETTING}" \
      --implementation "${IMPLEMENTATION}" \
      --space "${SPACE}" \
      --sample_axis "${SAMPLE_AXIS}" \
      --split "${SPLIT}" \
      --eval_batch_size "${EVAL_BATCH_SIZE}" \
      --device "cuda:${gpu}" \
      --csv "${CSV}" \
      --md "${MD}" \
      --title "${TITLE}"
    echo "[done] $(date '+%F %T') ${name}"
  ) >"${log_file}" 2>&1 &
  echo $! >"${LOG_ROOT}/${name}.pid"
}

status() {
  echo "[logs] ${LOG_ROOT}"
  for pid_file in "${LOG_ROOT}"/*.pid; do
    [[ -e "${pid_file}" ]] || continue
    local name
    local pid
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
    grep -E "Test metrics|Horizon|Traceback|ERROR|ValueError|KeyError|\\[done\\]|\\[skip\\]" "${log_file}" | tail -40 || true
  done
}

case "${cmd}" in
  launch)
    gpu=0
    for method in ${METHODS}; do
      launch_eval "${gpu}" "${method}" pems04
      gpu=$((gpu + 1))
      launch_eval "${gpu}" "${method}" pems08
      gpu=$((gpu + 1))
      if [[ "${gpu}" -ge 8 ]]; then
        break
      fi
    done
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
