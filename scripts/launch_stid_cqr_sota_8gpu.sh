#!/usr/bin/env bash
# launch_stid_cqr_sota_8gpu.sh  --  8-GPU reproducible entry point for the
# STID PeMS08 frozen-mean + CQR-load interval study.
#
# Usage (from repo root):
#   bash scripts/launch_stid_cqr_sota_8gpu.sh train    # launch 8-seed STID training
#   bash scripts/launch_stid_cqr_sota_8gpu.sh cqr      # run CQR evaluation on existing dumps
#   bash scripts/launch_stid_cqr_sota_8gpu.sh all      # train then cqr
#   bash scripts/launch_stid_cqr_sota_8gpu.sh status   # show running/finished jobs
#   bash scripts/launch_stid_cqr_sota_8gpu.sh tail     # grep key lines from logs
#
# Seeds and GPUs:
#   Seeds:  42  7  123  2024  2025  777  2026  3407  (8 seeds -> 8 GPUs 0-7)
#   GPUs:    0  1    2     3     4    5     6     7
#
# Environment overrides:
#   EPOCHS=120 PATIENCE=25 BATCH_SIZE=64  (training defaults)
#   DUMP_ROOT=outputs/frozen_dumps/stid_pems08  (where .npz dumps land)
#   CQR_ALPHA=0.10  TOD_BINS=6  LEVEL_BINS=5   (CQR evaluation defaults)
#   SKIP_SEEDS=""   comma-separated seeds to skip (e.g. "42,7")
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

# ------------------------------------------------------------------ config --
SEEDS=(42 7 123 2024 2025 777 2026 3407)
GPUS=(0 1 2 3 4 5 6 7)

EPOCHS="${EPOCHS:-120}"
BATCH_SIZE="${BATCH_SIZE:-64}"
PATIENCE="${PATIENCE:-25}"
HIDDEN_DIM="${HIDDEN_DIM:-256}"
NODE_EMB_DIM="${NODE_EMB_DIM:-64}"
HORIZON_EMB_DIM="${HORIZON_EMB_DIM:-16}"
NUM_LAYERS="${NUM_LAYERS:-3}"
DROPOUT="${DROPOUT:-0.10}"
LR="${LR:-1e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-4}"

DUMP_ROOT="${DUMP_ROOT:-outputs/frozen_dumps/stid_pems08}"
CQR_ALPHA="${CQR_ALPHA:-0.10}"
TOD_BINS="${TOD_BINS:-6}"
LEVEL_BINS="${LEVEL_BINS:-5}"
SKIP_SEEDS="${SKIP_SEEDS:-}"

LATEST_LINK="outputs/stid_cqr_sota_latest"
LOG_ROOT="${LOG_ROOT:-}"

cmd="${1:-help}"

# ---------------------------------------------------------------- helpers ---
_log_root() {
  if [[ -n "${LOG_ROOT}" ]]; then
    echo "${LOG_ROOT}"
    return
  fi
  case "${cmd}" in
    train|all)
      echo "outputs/stid_cqr_sota_$(date +%Y%m%d_%H%M%S)"
      ;;
    *)
      if [[ -L "${LATEST_LINK}" ]]; then
        echo "$(readlink -f "${LATEST_LINK}")"
      else
        echo "outputs/stid_cqr_sota_latest_missing"
      fi
      ;;
  esac
}

_skip_seed() {
  local s="$1"
  [[ -z "${SKIP_SEEDS}" ]] && return 1
  IFS=',' read -ra _sk <<< "${SKIP_SEEDS}"
  for _s in "${_sk[@]}"; do
    [[ "${_s}" == "${s}" ]] && return 0
  done
  return 1
}

# ----------------------------------------------------------------- train ----
do_train() {
  local log_root
  log_root="$(_log_root)"
  mkdir -p "${log_root}" "${DUMP_ROOT}"
  ln -sfn "$(realpath "${log_root}")" "${LATEST_LINK}" 2>/dev/null || \
    ln -sfn "${log_root}" "${LATEST_LINK}" || true

  for i in "${!SEEDS[@]}"; do
    local seed="${SEEDS[$i]}"
    local gpu="${GPUS[$i]}"
    if _skip_seed "${seed}"; then
      echo "[skip] seed=${seed}"
      continue
    fi
    local name="stid_pems08_s${seed}"
    local log_file="${log_root}/${name}.log"
    local dump_prefix="${DUMP_ROOT}/${name}"
    echo "[train] gpu=${gpu} seed=${seed} -> ${log_file}"
    (
      echo "[start] $(date '+%F %T') ${name}"
      python scripts/run_stid_mean.py \
        --config configs/pems08.yaml \
        --gpu_id "${gpu}" \
        --tag "stid_pems08_s${seed}" \
        --epochs "${EPOCHS}" \
        --batch_size "${BATCH_SIZE}" \
        --patience "${PATIENCE}" \
        --hidden_dim "${HIDDEN_DIM}" \
        --node_emb_dim "${NODE_EMB_DIM}" \
        --horizon_emb_dim "${HORIZON_EMB_DIM}" \
        --num_layers "${NUM_LAYERS}" \
        --dropout "${DROPOUT}" \
        --lr "${LR}" \
        --weight_decay "${WEIGHT_DECAY}" \
        --input_feature_index 0 \
        --seed "${seed}" \
        --dump_prefix "${dump_prefix}"
      echo "[done] $(date '+%F %T') ${name}"
    ) >"${log_file}" 2>&1 &
    echo $! >"${log_root}/${name}.pid"
  done
  echo "[train] launched ${#SEEDS[@]} jobs; logs in ${log_root}"
  echo "        monitor with:  bash $0 status"
  echo "        tail logs with: bash $0 tail"
}

# ------------------------------------------------------------------ cqr -----
do_cqr() {
  local log_root
  log_root="$(_log_root)"
  mkdir -p "${log_root}/cqr"

  for seed in "${SEEDS[@]}"; do
    if _skip_seed "${seed}"; then
      echo "[skip] seed=${seed}"
      continue
    fi
    local name="stid_pems08_s${seed}"
    local dump_prefix="${DUMP_ROOT}/${name}"
    local val_npz="${dump_prefix}_val.npz"
    local test_npz="${dump_prefix}_test.npz"
    local out_json="${log_root}/cqr/${name}_cqr.json"
    local log_file="${log_root}/cqr/${name}_cqr.log"

    if [[ ! -f "${val_npz}" || ! -f "${test_npz}" ]]; then
      echo "[warn] seed=${seed}: dumps not found at ${dump_prefix}_val/test.npz; skipping"
      continue
    fi

    echo "[cqr] seed=${seed} -> ${out_json}"
    (
      echo "[start] $(date '+%F %T') cqr seed=${seed}"
      python scripts/cqr_conditional_intervals.py \
        --val "${val_npz}" \
        --test "${test_npz}" \
        --alpha "${CQR_ALPHA}" \
        --level-bins "${LEVEL_BINS}" \
        --tod-bins "${TOD_BINS}" \
        --cqr-group nodeh \
        --label "stid_pems08_s${seed}" \
        --out-json "${out_json}"
      echo "[done] $(date '+%F %T') cqr seed=${seed}"
    ) >"${log_file}" 2>&1 &
    echo $! >"${log_root}/cqr/${name}_cqr.pid"
  done
  echo "[cqr] launched CQR jobs; results in ${log_root}/cqr/"
  echo "      wait for completion, then inspect JSON files for picp/mpiw/winkler."
}

# ---------------------------------------------------------------- status ----
do_status() {
  local log_root
  log_root="$(_log_root)"
  echo "[log_root] ${log_root}"
  local found=0
  local pid_file
  for pid_file in "${log_root}"/*.pid "${log_root}/cqr"/*.pid; do
    [[ -e "${pid_file}" ]] || continue
    found=1
    local name pid
    name="$(basename "${pid_file}" .pid)"
    pid="$(cat "${pid_file}")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "  RUNNING  ${name}  pid=${pid}"
    else
      echo "  DONE     ${name}  pid=${pid}"
    fi
  done
  [[ "${found}" -eq 0 ]] && echo "  (no .pid files found in ${log_root})"
}

# ------------------------------------------------------------------ tail ----
do_tail() {
  local log_root
  log_root="$(_log_root)"
  local log_file
  for log_file in "${log_root}"/*.log "${log_root}/cqr"/*.log; do
    [[ -e "${log_file}" ]] || continue
    echo "===== $(basename "${log_file}") ====="
    grep -E "\[epoch|\[test|saved best|early_stop|Traceback|ERROR|RuntimeError|ModuleNotFound|\[done\]|picp|winkler|mpiw|frozen_mu_test_mae" \
      "${log_file}" | tail -40 || true
  done
}

# ------------------------------------------------------------------- all ----
do_all() {
  do_train
  echo "[all] training launched; cqr will run after training finishes."
  echo "      to run cqr manually after training:  bash $0 cqr"
}

# ---------------------------------------------------------------- dispatch --
case "${cmd}" in
  train)   do_train   ;;
  cqr)     do_cqr     ;;
  all)     do_all     ;;
  status)  do_status  ;;
  tail)    do_tail    ;;
  help|--help|-h)
    echo "Usage: bash $0 {train|cqr|all|status|tail}"
    echo ""
    echo "  train   Launch 8-seed STID PeMS08 training on GPUs 0-7."
    echo "  cqr     Run CQR-load evaluation on existing frozen-mean dumps."
    echo "  all     train (launches training; cqr must be run separately after training)."
    echo "  status  Show running/finished jobs."
    echo "  tail    Grep key lines from training/cqr logs."
    echo ""
    echo "Seeds: ${SEEDS[*]}"
    echo "GPUs:  ${GPUS[*]}"
    echo ""
    echo "Key environment overrides:"
    echo "  EPOCHS=${EPOCHS}  PATIENCE=${PATIENCE}  BATCH_SIZE=${BATCH_SIZE}"
    echo "  DUMP_ROOT=${DUMP_ROOT}"
    echo "  CQR_ALPHA=${CQR_ALPHA}  TOD_BINS=${TOD_BINS}  LEVEL_BINS=${LEVEL_BINS}"
    echo "  SKIP_SEEDS=<comma-separated seeds to skip>"
    ;;
  *)
    echo "Unknown command: ${cmd}" >&2
    echo "Usage: bash $0 {train|cqr|all|status|tail|help}" >&2
    exit 2
    ;;
esac
