#!/usr/bin/env bash
set -euo pipefail

# Launch the highest-priority T-ITS revision experiments on 8 GPUs.
# Run from repo root:
#   bash scripts/launch_revision_8gpu.sh phase1
#   bash scripts/launch_revision_8gpu.sh status
#   bash scripts/launch_revision_8gpu.sh tail
#
# Phase 1 prioritizes the reviewer-critical fair semantic-access baselines,
# while also starting Prompt-STDiff main-model runs and one sensitivity queue.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

SUITE="${1:-phase1}"
DEFAULT_LOG_ROOT="outputs/revision_8gpu_logs/$(date +%Y%m%d_%H%M%S)"
if [[ -z "${LOG_ROOT:-}" && "${SUITE}" != "phase1" && -L outputs/revision_8gpu_logs/latest ]]; then
  LOG_ROOT="outputs/revision_8gpu_logs/$(readlink outputs/revision_8gpu_logs/latest)"
else
  LOG_ROOT="${LOG_ROOT:-${DEFAULT_LOG_ROOT}}"
fi
AGCRN_REPO="${AGCRN_REPO:-baselines/external_repos/AGCRN}"
SEM_PROJ_DIM="${SEM_PROJ_DIM:-128}"
AGCRN_EPOCHS="${AGCRN_EPOCHS:-50}"
PROMPT_EPOCH_TAG="${PROMPT_EPOCH_TAG:-revision_h12}"

mkdir -p "${LOG_ROOT}"

launch() {
  local name="$1"
  shift
  local log_file="${LOG_ROOT}/${name}.log"
  echo "[launch] ${name}"
  echo "[log] ${log_file}"
  (
    echo "[start] $(date '+%F %T') ${name}"
    echo "[cmd] $*"
    "$@"
    echo "[done] $(date '+%F %T') ${name}"
  ) >"${log_file}" 2>&1 &
  echo $! >"${LOG_ROOT}/${name}.pid"
}

launch_bash() {
  local name="$1"
  shift
  local log_file="${LOG_ROOT}/${name}.log"
  echo "[launch] ${name}"
  echo "[log] ${log_file}"
  (
    echo "[start] $(date '+%F %T') ${name}"
    echo "[cmd] $*"
    bash -lc "$*"
    echo "[done] $(date '+%F %T') ${name}"
  ) >"${log_file}" 2>&1 &
  echo $! >"${LOG_ROOT}/${name}.pid"
}

phase1() {
  echo "[suite] phase1 logs=${LOG_ROOT}"

  # GPU 0-3: highest-priority controlled semantic-access AGCRN baselines.
  launch agcrn_pems04_original \
    python -m baselines.runners.run_agcrn \
      --config configs/pems04.yaml \
      --device cuda:0 \
      --agcrn_repo "${AGCRN_REPO}" \
      --epochs "${AGCRN_EPOCHS}" \
      --eval_interval 5 \
      --input_feature_index 0 \
      --lr 1e-3 \
      --save_tag agcrn_original

  launch agcrn_pems04_semantic \
    python -m baselines.runners.run_agcrn \
      --config configs/pems04.yaml \
      --device cuda:1 \
      --agcrn_repo "${AGCRN_REPO}" \
      --use_semantic \
      --semantic_proj_dim "${SEM_PROJ_DIM}" \
      --epochs "${AGCRN_EPOCHS}" \
      --eval_interval 5 \
      --input_feature_index 0 \
      --lr 1e-3 \
      --save_tag agcrn_semantic

  launch agcrn_pems08_original \
    python -m baselines.runners.run_agcrn \
      --config configs/pems08.yaml \
      --device cuda:2 \
      --agcrn_repo "${AGCRN_REPO}" \
      --epochs "${AGCRN_EPOCHS}" \
      --eval_interval 5 \
      --input_feature_index 0 \
      --lr 2e-4 \
      --save_tag agcrn_original

  launch agcrn_pems08_semantic \
    python -m baselines.runners.run_agcrn \
      --config configs/pems08.yaml \
      --device cuda:3 \
      --agcrn_repo "${AGCRN_REPO}" \
      --use_semantic \
      --semantic_proj_dim "${SEM_PROJ_DIM}" \
      --epochs "${AGCRN_EPOCHS}" \
      --eval_interval 5 \
      --input_feature_index 0 \
      --lr 2e-4 \
      --save_tag agcrn_semantic

  # GPU 4-5: Prompt-STDiff main runs. Training-time validation is kept cheap;
  # run full-test evaluation after checkpoints are ready.
  launch prompt_pems04_h12 \
    bash scripts/run_pems04.sh \
      --mode train \
      --horizon_steps 12 \
      --history_steps 12 \
      --gpu_id 4 \
      --lr 1e-3 \
      --gamma 0 \
      --eval_interval 5 \
      --train_num_eval_samples 8 \
      --num_eval_samples 20 \
      --max_eval_batches 20 \
      --save_tag "${PROMPT_EPOCH_TAG}"

  launch prompt_pems08_h12 \
    bash scripts/run_pems08.sh \
      --mode train \
      --horizon_steps 12 \
      --history_steps 24 \
      --gpu_id 5 \
      --lr 2e-4 \
      --gamma 0 \
      --eval_interval 5 \
      --train_num_eval_samples 8 \
      --num_eval_samples 20 \
      --max_eval_batches 20 \
      --save_tag "${PROMPT_EPOCH_TAG}"

  # GPU 6: DDIM quality-speed eval queue if checkpoints already exist.
  launch_bash ddim_pems08_queue '
    set -euo pipefail
    CKPT="outputs/checkpoints/pems08_h12_'"${PROMPT_EPOCH_TAG}"'/best.pt"
    if [[ ! -f "${CKPT}" ]]; then
      CKPT="outputs/checkpoints/pems08/best.pt"
    fi
    if [[ ! -f "${CKPT}" ]]; then
      echo "[skip] no PeMS08 Prompt-STDiff checkpoint found for DDIM queue"
      exit 0
    fi
    for S in 50 25 10 5; do
      python scripts/run_experiment_and_record.py \
        --config configs/pems08.yaml \
        --ckpt "${CKPT}" \
        --gpu_id 6 \
        --sampler ddim \
        --sampling_steps "${S}" \
        --num_eval_samples 100 \
        --method Prompt-STDiff \
        --setting "ddim_S${S}" \
        --title "DDIM Quality-Speed Sweep"
    done
  '

  # GPU 7: sensitivity queue. This is intentionally sequential on one GPU.
  launch_bash sensitivity_pems08_h12 '
    set -euo pipefail
    CKPT="outputs/checkpoints/pems08_h12_'"${PROMPT_EPOCH_TAG}"'/best.pt"
    if [[ ! -f "${CKPT}" ]]; then
      CKPT="outputs/checkpoints/pems08/best.pt"
    fi
    if [[ ! -f "${CKPT}" ]]; then
      echo "[skip] no PeMS08 Prompt-STDiff checkpoint found for sensitivity queue"
      exit 0
    fi
    python scripts/sweep_hyperparam_sensitivity.py \
      --config configs/pems08.yaml \
      --ckpt "${CKPT}" \
      --out_dir outputs/sensitivity/pems08_h12_revision \
      --gpu_id 7 \
      --num_eval_samples 100 \
      --max_eval_batches 20
  '

  ln -sfn "$(basename "${LOG_ROOT}")" outputs/revision_8gpu_logs/latest
  echo "[suite] launched. Logs in ${LOG_ROOT}"
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
    echo "===== ${log_file} ====="
    tail -n 20 "${log_file}"
  done
}

case "${SUITE}" in
  phase1)
    phase1
    ;;
  status)
    status
    ;;
  tail)
    tail_logs
    ;;
  *)
    echo "Usage: bash scripts/launch_revision_8gpu.sh [phase1|status|tail]"
    exit 1
    ;;
esac
