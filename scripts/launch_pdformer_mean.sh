#!/usr/bin/env bash
set -euo pipefail

# Canonical PDFormer deterministic mean candidates.
# Smoke:
#   SMOKE=1 GPU_PEMS04=0 GPU_PEMS08=1 bash scripts/launch_pdformer_mean.sh launch
# Full:
#   GPU_PEMS04=0 GPU_PEMS08=1 EPOCHS=80 BATCH_SIZE=16 bash scripts/launch_pdformer_mean.sh launch

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

cmd="${1:-launch}"
LATEST_LINK="outputs/revision_8gpu_logs/pdformer_mean_latest"
if [[ -z "${LOG_ROOT:-}" ]]; then
  if [[ "${cmd}" == "launch" ]]; then
    LOG_ROOT="outputs/revision_8gpu_logs/pdformer_mean_$(date +%Y%m%d_%H%M%S)"
  elif [[ -L "${LATEST_LINK}" ]]; then
    LOG_ROOT="outputs/revision_8gpu_logs/$(readlink "${LATEST_LINK}")"
  else
    LOG_ROOT="outputs/revision_8gpu_logs/pdformer_mean_latest_missing"
  fi
fi

PDFORMER_REPO="${PDFORMER_REPO:-baselines/external_repos/PDFormer}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PATIENCE="${PATIENCE:-20}"
EMBED_DIM="${EMBED_DIM:-64}"
SKIP_DIM="${SKIP_DIM:-256}"
ENC_DEPTH="${ENC_DEPTH:-4}"
DROP_PATH="${DROP_PATH:-0.1}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-5e-2}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
TAG_PREFIX="${TAG_PREFIX:-time_}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  EPOCHS="${SMOKE_EPOCHS:-1}"
  BATCH_SIZE="${SMOKE_BATCH_SIZE:-4}"
  PATIENCE="${SMOKE_PATIENCE:-1}"
  EXTRA_ARGS="${EXTRA_ARGS} --max_train_batches ${SMOKE_TRAIN_BATCHES:-2} --max_eval_batches ${SMOKE_EVAL_BATCHES:-2}"
fi

if [[ "${cmd}" == "launch" ]]; then
  mkdir -p "${LOG_ROOT}"
fi

launch_one() {
  local dataset="$1"
  local gpu="$2"
  local tag="$3"
  local seed="$4"
  local name="${dataset}_${tag}"
  local log_file="${LOG_ROOT}/${name}.log"

  echo "[launch] gpu=${gpu} ${name}"
  (
    echo "[start] $(date '+%F %T') ${name}"
    python scripts/run_pdformer_canonical.py \
      --config "configs/${dataset}.yaml" \
      --pdformer_repo "${PDFORMER_REPO}" \
      --gpu_id "${gpu}" \
      --tag "${tag}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --patience "${PATIENCE}" \
      --embed_dim "${EMBED_DIM}" \
      --skip_dim "${SKIP_DIM}" \
      --enc_depth "${ENC_DEPTH}" \
      --drop_path "${DROP_PATH}" \
      --lr "${LR}" \
      --weight_decay "${WEIGHT_DECAY}" \
      --seed "${seed}" \
      ${EXTRA_ARGS}
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
    grep -E "\[epoch|\[test|saved best|early_stop|Traceback|ERROR|RuntimeError|ModuleNotFound|ImportError|\[done\]" "${log_file}" | tail -80 || true
  done
}

case "${cmd}" in
  launch)
    launch_one pems04 "${GPU_PEMS04:-0}" "${TAG_PREFIX}pdformer_e${EMBED_DIM}_d${ENC_DEPTH}_s42" 42
    launch_one pems08 "${GPU_PEMS08:-1}" "${TAG_PREFIX}pdformer_e${EMBED_DIM}_d${ENC_DEPTH}_s42" 42
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
