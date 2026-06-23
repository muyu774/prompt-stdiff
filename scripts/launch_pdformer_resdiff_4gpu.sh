#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/wzy/prompt-stdiff
STDIF_PYTHON="${STDIF_PYTHON:-/mnt/data1/conda-ground/envs/stdiff/bin/python}"
if [[ -x "${STDIF_PYTHON}" ]]; then
  export PATH="$(dirname "${STDIF_PYTHON}"):${PATH}"
fi
LATEST_LINK="outputs/revision_8gpu_logs/pdformer_resdiff_latest"
if [[ "${1:-launch}" == "launch" ]]; then
  LOG_ROOT="outputs/revision_8gpu_logs/pdformer_resdiff_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$LOG_ROOT"
  ln -sfn "$(basename "$LOG_ROOT")" "$LATEST_LINK"
else
  LOG_ROOT="$LATEST_LINK"
fi

run_one() {
  local gpu="$1"
  local config="$2"
  local lr="$3"
  local tag="$4"
  local log="$LOG_ROOT/${tag}.log"
  echo "[launch] gpu=${gpu} tag=${tag} config=${config} lr=${lr} log=${log}"
  (
    echo "[start] $(date '+%F %T') ${tag}"
    CUDA_VISIBLE_DEVICES="${gpu}" bash scripts/run_pems08.sh \
      --mode train \
      --config_file "${config}" \
      --horizon_steps 12 \
      --history_steps 24 \
      --gpu_id 0 \
      --lr "${lr}" \
      --eval_interval 5 \
      --train_num_eval_samples 8 \
      --num_eval_samples 20 \
      --max_eval_batches 20 \
      --save_tag "${tag}"
    echo "[done] $(date '+%F %T') ${tag}"
  ) >"$log" 2>&1 &
  echo $! >"$LOG_ROOT/${tag}.pid"
}

case "${1:-launch}" in
  launch)
    run_one 0 configs/pems08_pdformer_resdiff.yaml 1e-3 full_lr1e3_s2
    run_one 1 configs/pems08_pdformer_resdiff_nosem.yaml 1e-3 nosem_lr1e3_s2
    run_one 2 configs/pems08_pdformer_resdiff.yaml 5e-4 full_lr5e4_s2
    run_one 3 configs/pems08_pdformer_resdiff_nosem.yaml 5e-4 nosem_lr5e4_s2
    echo "[logs] ${LOG_ROOT}"
    ;;
  status)
    echo "[logs] ${LOG_ROOT}"
    for p in "$LOG_ROOT"/*.pid; do
      [[ -e "$p" ]] || continue
      tag="$(basename "$p" .pid)"
      pid="$(cat "$p")"
      if kill -0 "$pid" 2>/dev/null; then
        echo "RUNNING ${tag} pid=${pid}"
      else
        echo "DONE/STOPPED ${tag} pid=${pid}"
      fi
    done
    ;;
  tail)
    for f in "$LOG_ROOT"/*.log; do
      echo "===== $(basename "$f") ====="
      grep -E "\[start\]|\[done\]|Epoch [0-9]+ \| val_mae=|horizon=12|Test metrics|Traceback|ERROR|Killed|Non-finite" "$f" | tail -30 || true
    done
    ;;
  *)
    echo "Usage: bash scripts/launch_pdformer_resdiff_4gpu.sh [launch|status|tail]" >&2
    exit 1
    ;;
esac
