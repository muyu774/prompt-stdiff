#!/usr/bin/env bash
set -euo pipefail

# Run 9 training experiments in parallel:
#   datasets: pems03, pems04, pems08
#   horizons: 3, 6, 12
#   GPUs:     0..8 (one job per GPU)
#
# Logs:
# - master log: outputs/exp_logs/<run_tag>/master.log
# - per-job log: outputs/exp_logs/<run_tag>/<exp_id>.log
# - manifest csv: outputs/exp_logs/<run_tag>/manifest.csv
#
# Usage:
#   bash scripts/run_9x_grid_train.sh
#   bash scripts/run_9x_grid_train.sh --dry_run
#   bash scripts/run_9x_grid_train.sh --log_root outputs/exp_logs_custom

DRY_RUN=0
SKIP_PREP=1
LOG_ROOT="outputs/exp_logs"
GRID_DISABLE_VAL="${GRID_DISABLE_VAL:-1}"          # default: disable val during 9x training
GRID_EVAL_INTERVAL="${GRID_EVAL_INTERVAL:-}"       # optional override when not disabled
GRID_NUM_EVAL_SAMPLES="${GRID_NUM_EVAL_SAMPLES:-}" # optional override when not disabled
GRID_TRAIN_NUM_EVAL_SAMPLES="${GRID_TRAIN_NUM_EVAL_SAMPLES:-}" # optional fast-train val samples
GRID_MAX_EVAL_BATCHES="${GRID_MAX_EVAL_BATCHES:-}"             # optional fast-train val subset
GRID_HISTORY_STEPS="${GRID_HISTORY_STEPS:-12}"                 # keep fixed history window by default
GRID_DISABLE_DYNAMIC_SEMANTIC="${GRID_DISABLE_DYNAMIC_SEMANTIC:-0}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry_run)
      DRY_RUN=1
      shift
      ;;
    --log_root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --skip_prep)
      SKIP_PREP=1
      shift
      ;;
    --with_prep)
      SKIP_PREP=0
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage:
  bash scripts/run_9x_grid_train.sh [--dry_run] [--log_root PATH] [--skip_prep|--with_prep]
Env knobs:
  GRID_DISABLE_VAL=1        # default, skip in-train validation
  GRID_EVAL_INTERVAL=5      # optional
  GRID_NUM_EVAL_SAMPLES=10  # optional
  GRID_TRAIN_NUM_EVAL_SAMPLES=8
  GRID_MAX_EVAL_BATCHES=20
  GRID_HISTORY_STEPS=12
  GRID_DISABLE_DYNAMIC_SEMANTIC=1
EOF
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown argument: $1"
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

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi -L | wc -l | tr -d ' ')"
  if [[ "${GPU_COUNT}" -lt 9 ]]; then
    log "ERROR: detected ${GPU_COUNT} GPU(s), but this script expects at least 9 (0-8)."
    exit 1
  fi
  log "Detected GPUs: ${GPU_COUNT}"
else
  log "WARNING: nvidia-smi not found. Skip GPU count check."
fi

datasets=(pems03 pems04 pems08)
horizons=(3 6 12)

declare -a pids=()
declare -a exp_ids=()
declare -a dss=()
declare -a hs=()
declare -a gpus=()
declare -a starts=()
declare -a logs=()

cleanup() {
  log "Received interrupt. Terminating running jobs..."
  for pid in "${pids[@]:-}"; do
    if kill -0 "${pid}" >/dev/null 2>&1; then
      kill "${pid}" >/dev/null 2>&1 || true
    fi
  done
}
trap cleanup INT TERM

echo "exp_id,dataset,horizon,gpu_id,pid,start_time,end_time,duration_sec,return_code,status,log_file" > "${MANIFEST_CSV}"

gpu_id=0
for ds in "${datasets[@]}"; do
  for h in "${horizons[@]}"; do
    exp_id="${ds}_h${h}_g${gpu_id}"
    run_script="scripts/run_${ds}.sh"
    if [[ ! -f "${run_script}" ]]; then
      log "ERROR: missing run script: ${run_script}"
      exit 1
    fi

    job_log="${RUN_DIR}/${exp_id}.log"
    start_iso="$(date '+%F %T')"
    start_epoch="$(date +%s)"
    cmd=(bash "${run_script}" --mode train --horizon_steps "${h}" --gpu_id "${gpu_id}")
    if [[ -n "${GRID_HISTORY_STEPS}" ]]; then
      cmd+=(--history_steps "${GRID_HISTORY_STEPS}")
    fi
    if [[ "${GRID_DISABLE_VAL}" == "1" ]]; then
      cmd+=(--disable_val)
    fi
    if [[ "${GRID_DISABLE_DYNAMIC_SEMANTIC}" == "1" ]]; then
      cmd+=(--disable_dynamic_semantic)
    fi
    if [[ -n "${GRID_EVAL_INTERVAL}" ]]; then
      cmd+=(--eval_interval "${GRID_EVAL_INTERVAL}")
    fi
    if [[ -n "${GRID_NUM_EVAL_SAMPLES}" ]]; then
      cmd+=(--num_eval_samples "${GRID_NUM_EVAL_SAMPLES}")
    fi
    if [[ -n "${GRID_TRAIN_NUM_EVAL_SAMPLES}" ]]; then
      cmd+=(--train_num_eval_samples "${GRID_TRAIN_NUM_EVAL_SAMPLES}")
    fi
    if [[ -n "${GRID_MAX_EVAL_BATCHES}" ]]; then
      cmd+=(--max_eval_batches "${GRID_MAX_EVAL_BATCHES}")
    fi

    log "LAUNCH exp_id=${exp_id} gpu=${gpu_id} cmd=${cmd[*]}"
    if [[ "${DRY_RUN}" -eq 1 ]]; then
      echo "${exp_id},${ds},${h},${gpu_id},-1,${start_iso},${start_iso},0,0,dry_run,${job_log}" >> "${MANIFEST_CSV}"
    else
      (
        echo "[START] exp_id=${exp_id} start_time=${start_iso}"
        echo "[CMD] ${cmd[*]}"
        SKIP_PREP="${SKIP_PREP}" "${cmd[@]}"
      ) > "${job_log}" 2>&1 &
      pid=$!
      pids+=("${pid}")
      exp_ids+=("${exp_id}")
      dss+=("${ds}")
      hs+=("${h}")
      gpus+=("${gpu_id}")
      starts+=("${start_epoch}")
      logs+=("${job_log}")
      log "STARTED exp_id=${exp_id} pid=${pid} log=${job_log}"
    fi

    gpu_id=$((gpu_id + 1))
  done
done

if [[ "${DRY_RUN}" -eq 1 ]]; then
  log "DRY RUN completed. Planned 9 experiments."
  exit 0
fi

fail_count=0
for i in "${!pids[@]}"; do
  pid="${pids[$i]}"
  exp_id="${exp_ids[$i]}"
  ds="${dss[$i]}"
  h="${hs[$i]}"
  gpu="${gpus[$i]}"
  start_epoch="${starts[$i]}"
  job_log="${logs[$i]}"

  rc=0
  if wait "${pid}"; then
    rc=0
    status="success"
  else
    rc=$?
    status="failed"
    fail_count=$((fail_count + 1))
  fi

  end_iso="$(date '+%F %T')"
  end_epoch="$(date +%s)"
  duration=$((end_epoch - start_epoch))

  echo "${exp_id},${ds},${h},${gpu},${pid},${start_epoch},${end_epoch},${duration},${rc},${status},${job_log}" >> "${MANIFEST_CSV}"
  log "FINISH exp_id=${exp_id} status=${status} rc=${rc} duration_sec=${duration} log=${job_log}"
  if [[ "${status}" == "failed" ]]; then
    log "---- failure tail: ${job_log} ----"
    tail -n 40 "${job_log}" | tee -a "${MASTER_LOG}" || true
    log "---- end failure tail: ${job_log} ----"
  fi
done

if [[ "${fail_count}" -gt 0 ]]; then
  log "Completed with failures: ${fail_count} / 9 experiments failed."
  exit 1
fi

log "All 9 experiments completed successfully."
