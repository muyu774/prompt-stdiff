#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/wzy/prompt-stdiff

WAIT_DIR="outputs/revision_8gpu_logs/node_group_hscale_val_latest"
LOG_ROOT="outputs/revision_8gpu_logs/incident_tail_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
ln -sfn "$(basename "$LOG_ROOT")" outputs/revision_8gpu_logs/incident_tail_latest

echo "[wait] waiting for node-group validation pids from ${WAIT_DIR}" | tee -a "$LOG_ROOT/launcher.log"
if [[ -d "$WAIT_DIR" ]]; then
  while true; do
    running=0
    for p in "$WAIT_DIR"/*.pid; do
      [[ -f "$p" ]] || continue
      pid=$(cat "$p")
      if kill -0 "$pid" 2>/dev/null; then
        running=$((running + 1))
      fi
    done
    if [[ "$running" -eq 0 ]]; then
      break
    fi
    echo "[wait] still running=${running} $(date)" | tee -a "$LOG_ROOT/launcher.log"
    sleep 60
  done
fi

echo "[launch-start] $(date)" | tee -a "$LOG_ROOT/launcher.log"
run_one(){
  gpu="$1"
  cfg="$2"
  tag="$3"
  log="$LOG_ROOT/${tag}.log"
  echo "[launch] gpu=${gpu} tag=${tag} cfg=${cfg}" | tee -a "$LOG_ROOT/launcher.log"
  (
    echo "[start] $(date +%F %T) ${tag}"
    CUDA_VISIBLE_DEVICES="$gpu" bash scripts/run_pems08.sh \
      --mode train \
      --config_file "$cfg" \
      --horizon_steps 12 \
      --history_steps 24 \
      --gpu_id 0 \
      --lr 1e-3 \
      --eval_interval 5 \
      --train_num_eval_samples 8 \
      --num_eval_samples 20 \
      --max_eval_batches 20 \
      --save_tag "$tag"
    echo "[done] $(date +%F %T) ${tag}"
  ) >"$log" 2>&1 &
  echo $! >"$LOG_ROOT/${tag}.pid"
}

run_one 0 configs/pems08_pdformer_resdiff_incident_tail.yaml incident_tail_sem_student
run_one 1 configs/pems08_pdformer_resdiff_incident_tail_nosem.yaml incident_tail_nosem_student
run_one 3 configs/pems08_pdformer_resdiff_incident_tail_gaussian.yaml incident_tail_sem_gaussian
run_one 4 configs/pems08_pdformer_resdiff_incident_tail_df5.yaml incident_tail_sem_student_df5

echo "[logs] $LOG_ROOT" | tee -a "$LOG_ROOT/launcher.log"
