#!/usr/bin/env bash
set -euo pipefail

# Launch seven STID-style deterministic frozen-mean candidate experiments.
# Recommended when GPUs 0,1,3,4,5,6,7 are free:
#   bash scripts/launch_7gpu_stid_mean.sh launch
#   bash scripts/launch_7gpu_stid_mean.sh status
#   bash scripts/launch_7gpu_stid_mean.sh tail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

cmd="${1:-launch}"
LATEST_LINK="outputs/revision_8gpu_logs/stid_mean_latest"
if [[ -z "${LOG_ROOT:-}" ]]; then
  if [[ "${cmd}" == "launch" ]]; then
    LOG_ROOT="outputs/revision_8gpu_logs/stid_mean_$(date +%Y%m%d_%H%M%S)"
  elif [[ -L "${LATEST_LINK}" ]]; then
    LOG_ROOT="outputs/revision_8gpu_logs/$(readlink "${LATEST_LINK}")"
  else
    LOG_ROOT="outputs/revision_8gpu_logs/stid_mean_latest_missing"
  fi
fi

EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PATIENCE="${PATIENCE:-25}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
RUN_IDS="${RUN_IDS:-1,2,3,4,5,6,7}"
TAG_PREFIX="${TAG_PREFIX:-time_}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  EPOCHS="${SMOKE_EPOCHS:-2}"
  BATCH_SIZE="${SMOKE_BATCH_SIZE:-16}"
  PATIENCE="${SMOKE_PATIENCE:-2}"
  EXTRA_ARGS="${EXTRA_ARGS} --max_train_batches ${SMOKE_TRAIN_BATCHES:-3} --max_eval_batches ${SMOKE_EVAL_BATCHES:-2}"
fi

if [[ "${cmd}" == "launch" ]]; then
  mkdir -p "${LOG_ROOT}"
fi

launch_one() {
  local run_id="$1"
  shift
  local gpu="$1"
  local dataset="$2"
  local tag="$3"
  local hidden="$4"
  local node_emb="$5"
  local horizon_emb="$6"
  local layers="$7"
  local dropout="$8"
  local lr="$9"
  local wd="${10}"
  local seed="${11}"
  local name="${dataset}_${tag}"
  local log_file="${LOG_ROOT}/${name}.log"

  if [[ ",${RUN_IDS}," != *",${run_id},"* ]]; then
    echo "[skip] id=${run_id} ${name}"
    return 0
  fi
  echo "[launch] id=${run_id} gpu=${gpu} ${name} hidden=${hidden} node=${node_emb} lr=${lr} seed=${seed}"
  (
    echo "[start] $(date '+%F %T') ${name}"
    python scripts/run_stid_mean.py \
      --config "configs/${dataset}.yaml" \
      --gpu_id "${gpu}" \
      --tag "${tag}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --patience "${PATIENCE}" \
      --hidden_dim "${hidden}" \
      --node_emb_dim "${node_emb}" \
      --horizon_emb_dim "${horizon_emb}" \
      --num_layers "${layers}" \
      --dropout "${dropout}" \
      --lr "${lr}" \
      --weight_decay "${wd}" \
      --input_feature_index 0 \
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
    local name pid
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
    grep -E "\[epoch|\[test|saved best|early_stop|Traceback|ERROR|RuntimeError|ModuleNotFound|\[done\]" "${log_file}" | tail -60 || true
  done
}

case "${cmd}" in
  launch)
    # 4x PeMS04, 3x PeMS08. These are frozen-mean candidates, not diffusion jobs.
    launch_one 1 0 pems04 "${TAG_PREFIX}stid_h128_n64_lr1e3_s42"   128  64 16 3 0.10 1e-3  1e-4 42
    launch_one 2 1 pems08 "${TAG_PREFIX}stid_h128_n64_lr1e3_s42"   128  64 16 3 0.10 1e-3  1e-4 42
    launch_one 3 3 pems04 "${TAG_PREFIX}stid_h256_n64_lr1e3_s3407" 256  64 16 3 0.10 1e-3  1e-4 3407
    launch_one 4 4 pems08 "${TAG_PREFIX}stid_h256_n64_lr1e3_s3407" 256  64 16 3 0.10 1e-3  1e-4 3407
    launch_one 5 5 pems04 "${TAG_PREFIX}stid_h256_n128_lr5e4_s2026" 256 128 32 4 0.20 5e-4  1e-4 2026
    launch_one 6 6 pems08 "${TAG_PREFIX}stid_h256_n128_lr5e4_s2026" 256 128 32 4 0.20 5e-4  1e-4 2026
    launch_one 7 7 pems04 "${TAG_PREFIX}stid_h384_n128_lr5e4_s777" 384 128 32 4 0.20 5e-4  5e-5 777
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
