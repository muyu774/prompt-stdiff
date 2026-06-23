#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/wzy/prompt-stdiff
PY="/mnt/data1/conda-ground/envs/stdiff/bin/python"
cmd="${1:-launch}"
LATEST_LINK="outputs/revision_8gpu_logs/backbone_repair_pems08_latest"
if [[ -z "${LOG_ROOT:-}" ]]; then
  if [[ "$cmd" == "launch" ]]; then
    LOG_ROOT="outputs/revision_8gpu_logs/backbone_repair_pems08_$(date +%Y%m%d_%H%M%S)"
  elif [[ -L "$LATEST_LINK" ]]; then
    LOG_ROOT="outputs/revision_8gpu_logs/$(readlink "$LATEST_LINK")"
  else
    LOG_ROOT="outputs/revision_8gpu_logs/backbone_repair_pems08_latest_missing"
  fi
fi
CSV="outputs/p0_tables_20260623/backbone_repair_pems08.csv"
MD="outputs/p0_tables_20260623/backbone_repair_pems08.md"
mkdir -p "$LOG_ROOT" outputs/p0_tables_20260623
if [[ "$cmd" == "launch" ]]; then
  ln -sfn "$(basename "$LOG_ROOT")" "$LATEST_LINK"
fi

wait_stdiff_gpu() {
  local gpu="$1"
  while ps -u "$USER" -f | grep -E "evaluate.py|train.py|run_pdformer_canonical.py|run_agcrn.py" | grep -v grep | grep -q -- "--gpu_id ${gpu}\|cuda:${gpu}"; do
    echo "[wait] gpu=${gpu} still has stdiff job; sleep 180s" >&2
    sleep 180
  done
}

launch_agcrn() {
  local gpu="$1"
  local cfg="$2"
  local tag="$3"
  local lr="$4"
  local log_file="$LOG_ROOT/${tag}.log"
  echo "[queue] AGCRN ${tag} gpu=${gpu} cfg=${cfg} lr=${lr}"
  (
    wait_stdiff_gpu "$gpu"
    echo "[start] $(date +%F %T) ${tag} gpu=${gpu}"
    "$PY" -m baselines.runners.run_agcrn \
      --mode train \
      --config "$cfg" \
      --gpu_id "$gpu" \
      --agcrn_repo baselines/external_repos/AGCRN \
      --input_feature_index 0 \
      --epochs 120 \
      --eval_interval 5 \
      --patience 20 \
      --lr "$lr" \
      --weight_decay 0 \
      --batch_size 32 \
      --max_eval_batches 0 \
      --save_tag "$tag" \
      --output_csv "$CSV" \
      --results_md "$MD"
    echo "[done] $(date +%F %T) ${tag}"
  ) >"$log_file" 2>&1 &
  echo $! >"$LOG_ROOT/${tag}.pid"
}

launch_pdformer() {
  local gpu="$1"
  local tag="$2"
  local embed="$3"
  local depth="$4"
  local seed="$5"
  local log_file="$LOG_ROOT/${tag}.log"
  echo "[queue] PDFormer ${tag} gpu=${gpu} embed=${embed} depth=${depth} seed=${seed}"
  (
    wait_stdiff_gpu "$gpu"
    echo "[start] $(date +%F %T) ${tag} gpu=${gpu}"
    "$PY" scripts/run_pdformer_canonical.py \
      --config configs/pems08.yaml \
      --pdformer_repo baselines/external_repos/PDFormer \
      --gpu_id "$gpu" \
      --tag "$tag" \
      --epochs 120 \
      --batch_size 16 \
      --patience 25 \
      --embed_dim "$embed" \
      --skip_dim 256 \
      --enc_depth "$depth" \
      --drop_path 0.1 \
      --lr 1e-3 \
      --weight_decay 5e-2 \
      --seed "$seed" \
      --output_csv "$CSV" \
      --results_md "$MD"
    echo "[done] $(date +%F %T) ${tag}"
  ) >"$log_file" 2>&1 &
  echo $! >"$LOG_ROOT/${tag}.pid"
}

case "$cmd" in
  launch)
    launch_agcrn 0 configs/backbone_repair/pems08_agcrn_h12.yaml agcrn_pems08_h12_lr1e3_fullval 1e-3
    launch_agcrn 1 configs/backbone_repair/pems08_agcrn_h24.yaml agcrn_pems08_h24_lr1e3_fullval 1e-3
    launch_pdformer 2 time_pdformer_pems08_e128_d4_s44_fullval 128 4 44
    echo "[logs] $LOG_ROOT"
    ;;
  status)
    echo "[logs] $LOG_ROOT"
    for f in "$LOG_ROOT"/*.pid; do
      [ -e "$f" ] || continue
      name="$(basename "$f" .pid)"
      pid="$(cat "$f")"
      if kill -0 "$pid" 2>/dev/null; then echo "RUNNING $name pid=$pid"; else echo "DONE/EXITED $name pid=$pid"; fi
    done
    ;;
  tail)
    echo "[logs] $LOG_ROOT"
    for f in "$LOG_ROOT"/*.log; do
      [ -e "$f" ] || continue
      echo "===== $(basename "$f") ====="
      grep -E "\[wait\]|\[start\]|AGCRN Epoch|AGCRN test|\[epoch|\[test|saved best|early_stop|Traceback|ERROR|RuntimeError|ModuleNotFound|\[done\]" "$f" | tail -80 || true
    done
    ;;
  *)
    echo "Usage: $0 {launch|status|tail}" >&2
    exit 2
    ;;
esac
