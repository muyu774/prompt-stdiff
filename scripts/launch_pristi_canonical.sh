#!/usr/bin/env bash
set -euo pipefail

# Train/evaluate PriSTI on canonical PeMS04/PeMS08 splits and export NPZ samples.
#
# Smoke test:
#   SMOKE=1 GPU_PEMS04=4 GPU_PEMS08=5 bash scripts/launch_pristi_canonical.sh launch
#
# Full run:
#   GPU_PEMS04=4 GPU_PEMS08=5 EPOCHS=50 NSAMPLE=20 bash scripts/launch_pristi_canonical.sh launch

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

cmd="${1:-launch}"
LATEST_LINK="outputs/revision_8gpu_logs/pristi_latest"
if [[ -z "${LOG_ROOT:-}" ]]; then
  if [[ "${cmd}" == "launch" ]]; then
    LOG_ROOT="outputs/revision_8gpu_logs/pristi_$(date +%Y%m%d_%H%M%S)"
  elif [[ -L "${LATEST_LINK}" ]]; then
    LOG_ROOT="outputs/revision_8gpu_logs/$(readlink "${LATEST_LINK}")"
  else
    LOG_ROOT="outputs/revision_8gpu_logs/pristi_latest_missing"
  fi
fi

PRISTI_REPO="${PRISTI_REPO:-baselines/external_repos/PriSTI}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-1e-3}"
NSAMPLE="${NSAMPLE:-20}"
VALID_INTERVAL="${VALID_INTERVAL:-10}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-50}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  EPOCHS="${SMOKE_EPOCHS:-1}"
  NSAMPLE="${SMOKE_NSAMPLE:-2}"
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
  local name="pristi_${dataset}"
  local log_file="${LOG_ROOT}/${name}.log"

  if [[ ! -f "${canonical}" ]]; then
    echo "[missing] ${canonical}. Run scripts/export_canonical_setup.py first." >&2
    return 1
  fi
  if [[ ! -d "${PRISTI_REPO}" ]]; then
    echo "[missing] PriSTI repo: ${PRISTI_REPO}" >&2
    return 1
  fi

  echo "[launch] ${name} gpu=${gpu}"
  (
    echo "[start] $(date '+%F %T') ${name}"
    python scripts/run_pristi_canonical.py \
      --config "${config}" \
      --canonical_npz "${canonical}" \
      --pristi_repo "${PRISTI_REPO}" \
      --gpu_id "${gpu}" \
      --epochs "${EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --lr "${LR}" \
      --valid_interval "${VALID_INTERVAL}" \
      --diffusion_steps "${DIFFUSION_STEPS}" \
      --nsample "${NSAMPLE}" \
      --save_dir "outputs/prob_baselines/PriSTI/${dataset}_run" \
      --out_npz "outputs/prob_baselines/PriSTI/${dataset}_samples.npz" \
      ${MAX_TRAIN_BATCHES} \
      ${MAX_EVAL_BATCHES} \
      ${EXTRA_ARGS}
    echo "[eval] ${name}"
    python scripts/eval_probabilistic_npz.py \
      --pred_npz "outputs/prob_baselines/PriSTI/${dataset}_samples.npz" \
      --config "${config}" \
      --method PriSTI \
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
    launch_one pems04 "${GPU_PEMS04:-4}"
    launch_one pems08 "${GPU_PEMS08:-5}"
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
