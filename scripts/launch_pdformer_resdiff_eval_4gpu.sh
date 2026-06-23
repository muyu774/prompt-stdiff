#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/wzy/prompt-stdiff
STDIF_PYTHON="${STDIF_PYTHON:-/mnt/data1/conda-ground/envs/stdiff/bin/python}"
if [[ -x "${STDIF_PYTHON}" ]]; then
  export PATH="$(dirname "${STDIF_PYTHON}"):${PATH}"
fi
LATEST_LINK="outputs/revision_8gpu_logs/pdformer_resdiff_eval_latest"
if [[ "${1:-launch}" == "launch" ]]; then
  LOG_ROOT="outputs/revision_8gpu_logs/pdformer_resdiff_eval_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$LOG_ROOT"
  ln -sfn "$(basename "$LOG_ROOT")" "$LATEST_LINK"
else
  LOG_ROOT="$LATEST_LINK"
fi

run_eval() {
  local gpu="$1"
  local tag="$2"
  local config="$3"
  local ckpt="$4"
  local log="$LOG_ROOT/${tag}.log"
  echo "[launch] gpu=${gpu} tag=${tag} ckpt=${ckpt} log=${log}"
  (
    echo "[start] $(date +%F %T) ${tag}"
    CUDA_VISIBLE_DEVICES="${gpu}" python evaluate.py \
      --config "${config}" \
      --ckpt "${ckpt}" \
      --gpu_id 0
    echo "[done] $(date +%F %T) ${tag}"
  ) >"$log" 2>&1 &
  echo $! >"$LOG_ROOT/${tag}.pid"
}

case "${1:-launch}" in
  launch)
    run_eval 0 p08_pd_full_lr1e3_last configs/pems08_pdformer_resdiff.yaml outputs/checkpoints/pems08_pdformer_resdiff_full_lr1e3_s2/last.pt
    run_eval 1 p08_pd_full_lr5e4_last configs/pems08_pdformer_resdiff.yaml outputs/checkpoints/pems08_pdformer_resdiff_full_lr5e4_s2/last.pt
    run_eval 2 p08_pd_nosem_lr1e3_last configs/pems08_pdformer_resdiff_nosem.yaml outputs/checkpoints/pems08_pdformer_resdiff_nosem_nosem_lr1e3_s2/last.pt
    run_eval 4 p08_pd_nosem_lr5e4_last configs/pems08_pdformer_resdiff_nosem.yaml outputs/checkpoints/pems08_pdformer_resdiff_nosem_nosem_lr5e4_s2/last.pt
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
      grep -E "\[start\]|\[done\]|Loading residual stats|Loaded checkpoint|Test metrics|Horizon [0-9]+|Traceback|ERROR|Killed|Non-finite" "$f" | tail -60 || true
    done
    ;;
  *)
    echo "Usage: bash scripts/launch_pdformer_resdiff_eval_4gpu.sh [launch|status|tail]" >&2
    exit 1
    ;;
esac
