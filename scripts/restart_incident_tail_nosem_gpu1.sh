#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/wzy/prompt-stdiff
export PATH="/mnt/data1/conda-ground/envs/stdiff/bin:$PATH"
LOG_ROOT="outputs/revision_8gpu_logs/incident_tail_nosem_restart_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
ln -sfn "$(basename "$LOG_ROOT")" outputs/revision_8gpu_logs/incident_tail_nosem_restart_latest
log="$LOG_ROOT/incident_tail_nosem_student.log"
(
  echo "[start] $(date) incident_tail_nosem_student"
  CUDA_VISIBLE_DEVICES=1 bash scripts/run_pems08.sh \
    --mode train \
    --config_file configs/pems08_pdformer_resdiff_incident_tail_nosem.yaml \
    --horizon_steps 12 \
    --history_steps 24 \
    --gpu_id 0 \
    --lr 1e-3 \
    --eval_interval 5 \
    --train_num_eval_samples 8 \
    --num_eval_samples 20 \
    --max_eval_batches 20 \
    --save_tag incident_tail_nosem_student
  echo "[done] $(date) incident_tail_nosem_student"
) >"$log" 2>&1 &
echo $! >"$LOG_ROOT/incident_tail_nosem_student.pid"
echo "[logs] $LOG_ROOT"
sleep 6
tail -n 30 "$log"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
