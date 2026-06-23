#!/usr/bin/env bash
set -euo pipefail

# Run pems08 training in parallel on GPU 0/1/2:
# - horizon 3  -> gpu 0
# - horizon 6  -> gpu 1
# - horizon 12 -> gpu 2
#
# Logs:
# - master log: outputs/exp_logs/<run_tag>/master.log
# - job logs:   outputs/exp_logs/<run_tag>/pems08_h{3,6,12}_g{0,1,2}.log
# - manifest:   outputs/exp_logs/<run_tag>/manifest.csv

CONFIG_FILE="configs/pems08.yaml"
LOG_ROOT="outputs/exp_logs"
SAVE_TAG=""
DRY_RUN=0

# Optional passthrough knobs for run_pems08.sh
HISTORY_STEPS=""
EVAL_INTERVAL=""
NUM_EVAL_SAMPLES=""
TRAIN_NUM_EVAL_SAMPLES=""
MAX_EVAL_BATCHES=""
LR=""
GAMMA=""
DISABLE_VAL=0
DISABLE_DYNAMIC_SEMANTIC=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_pems08_h3612_g012.sh [--config_file PATH] [--save_tag STR] [--log_root PATH] [--dry_run]
                                      [--history_steps N] [--eval_interval N]
                                      [--num_eval_samples N] [--train_num_eval_samples N]
                                      [--max_eval_batches N] [--lr FLOAT] [--gamma FLOAT]
                                      [--disable_val] [--disable_dynamic_semantic]

Examples:
  bash scripts/run_pems08_h3612_g012.sh
  bash scripts/run_pems08_h3612_g012.sh --config_file configs/pems08_full.yaml --save_tag full
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config_file)
      CONFIG_FILE="$2"
      shift 2
      ;;
    --save_tag)
      SAVE_TAG="$2"
      shift 2
      ;;
    --log_root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --dry_run)
      DRY_RUN=1
      shift
      ;;
    --history_steps)
      HISTORY_STEPS="$2"
      shift 2
      ;;
    --eval_interval)
      EVAL_INTERVAL="$2"
      shift 2
      ;;
    --num_eval_samples)
      NUM_EVAL_SAMPLES="$2"
      shift 2
      ;;
    --train_num_eval_samples)
      TRAIN_NUM_EVAL_SAMPLES="$2"
      shift 2
      ;;
    --max_eval_batches)
      MAX_EVAL_BATCHES="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --gamma)
      GAMMA="$2"
      shift 2
      ;;
    --disable_val)
      DISABLE_VAL=1
      shift
      ;;
    --disable_dynamic_semantic)
      DISABLE_DYNAMIC_SEMANTIC=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${LOG_ROOT}/${RUN_TAG}"
MASTER_LOG="${RUN_DIR}/master.log"
MANIFEST_CSV="${RUN_DIR}/manifest.csv"
mkdir -p "${RUN_DIR}"

log() {
  local ts
  ts="$(date '+%F %T')"
  echo "[${ts}] $*" | tee -a "${MASTER_LOG}"
}

horizons=(3 6 12)
gpus=(0 1 2)

echo "exp_id,horizon,gpu_id,pid,start_time,end_time,duration_sec,return_code,status,log_file" > "${MANIFEST_CSV}"

declare -a pids=()
declare -a exp_ids=()
declare -a hs=()
declare -a gs=()
declare -a starts=()
declare -a logs=()

for i in "${!horizons[@]}"; do
  h="${horizons[$i]}"
  g="${gpus[$i]}"
  exp_id="pems08_h${h}_g${g}"
  job_log="${RUN_DIR}/${exp_id}.log"
  start_iso="$(date '+%F %T')"
  start_epoch="$(date +%s)"

  cmd=(bash scripts/run_pems08.sh --mode train --config_file "${CONFIG_FILE}" --horizon_steps "${h}" --gpu_id "${g}")
  if [[ -n "${SAVE_TAG}" ]]; then
    cmd+=(--save_tag "${SAVE_TAG}")
  fi
  if [[ -n "${HISTORY_STEPS}" ]]; then
    cmd+=(--history_steps "${HISTORY_STEPS}")
  fi
  if [[ -n "${EVAL_INTERVAL}" ]]; then
    cmd+=(--eval_interval "${EVAL_INTERVAL}")
  fi
  if [[ -n "${NUM_EVAL_SAMPLES}" ]]; then
    cmd+=(--num_eval_samples "${NUM_EVAL_SAMPLES}")
  fi
  if [[ -n "${TRAIN_NUM_EVAL_SAMPLES}" ]]; then
    cmd+=(--train_num_eval_samples "${TRAIN_NUM_EVAL_SAMPLES}")
  fi
  if [[ -n "${MAX_EVAL_BATCHES}" ]]; then
    cmd+=(--max_eval_batches "${MAX_EVAL_BATCHES}")
  fi
  if [[ -n "${LR}" ]]; then
    cmd+=(--lr "${LR}")
  fi
  if [[ -n "${GAMMA}" ]]; then
    cmd+=(--gamma "${GAMMA}")
  fi
  if [[ "${DISABLE_VAL}" == "1" ]]; then
    cmd+=(--disable_val)
  fi
  if [[ "${DISABLE_DYNAMIC_SEMANTIC}" == "1" ]]; then
    cmd+=(--disable_dynamic_semantic)
  fi

  log "LAUNCH exp_id=${exp_id} cmd=${cmd[*]}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "${exp_id},${h},${g},-1,${start_iso},${start_iso},0,0,dry_run,${job_log}" >> "${MANIFEST_CSV}"
    continue
  fi

  (
    echo "[START] exp_id=${exp_id} start_time=${start_iso}"
    echo "[CMD] ${cmd[*]}"
    "${cmd[@]}"
  ) > "${job_log}" 2>&1 &

  pid=$!
  pids+=("${pid}")
  exp_ids+=("${exp_id}")
  hs+=("${h}")
  gs+=("${g}")
  starts+=("${start_epoch}")
  logs+=("${job_log}")
  log "STARTED exp_id=${exp_id} pid=${pid} log=${job_log}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
  log "DRY RUN done."
  exit 0
fi

fail_count=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  exp_id="${exp_ids[$i]}"
  h="${hs[$i]}"
  g="${gs[$i]}"
  job_log="${logs[$i]}"
  start_epoch="${starts[$i]}"

  rc=0
  if wait "${pid}"; then
    status="success"
    rc=0
  else
    status="failed"
    rc=$?
    fail_count=$((fail_count + 1))
  fi

  end_iso="$(date '+%F %T')"
  end_epoch="$(date +%s)"
  duration=$((end_epoch - start_epoch))
  echo "${exp_id},${h},${g},${pid},${start_epoch},${end_epoch},${duration},${rc},${status},${job_log}" >> "${MANIFEST_CSV}"
  log "FINISH exp_id=${exp_id} status=${status} rc=${rc} duration_sec=${duration} log=${job_log}"
  if [[ "${status}" == "failed" ]]; then
    log "---- failure tail: ${job_log} ----"
    tail -n 40 "${job_log}" | tee -a "${MASTER_LOG}" || true
    log "---- end failure tail: ${job_log} ----"
  fi
done

if [[ "${fail_count}" -gt 0 ]]; then
  log "Completed with failures: ${fail_count}/3 failed."
  exit 1
fi

log "All 3 pems08 runs finished successfully."
