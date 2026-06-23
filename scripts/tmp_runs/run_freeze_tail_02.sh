#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/wzy/prompt-stdiff
export PATH="/mnt/data1/conda-ground/envs/stdiff/bin:$PATH"
LOG_ROOT="outputs/revision_8gpu_logs/freeze_tail_02_20260623"
mkdir -p "$LOG_ROOT"

run_one(){
  local gpu="$1"
  local cfg="$2"
  local tag="$3"
  echo "[start] $(date +%F_%T) $tag gpu=$gpu"
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
  echo "[done] $(date +%F_%T) $tag"
}

run_one 0 configs/freeze_tail_20260623/pems08_tail_only_sem_full.yaml tail_only_sem > "$LOG_ROOT/gpu0_tail_only_sem.log" 2>&1 &
PID0=$!
run_one 2 configs/freeze_tail_20260623/pems08_tail_only_nosem_full.yaml tail_only_nosem > "$LOG_ROOT/gpu2_tail_only_nosem.log" 2>&1 &
PID2=$!

echo "[launch] gpu0_tail_only_sem pid=$PID0"
echo "[launch] gpu2_tail_only_nosem pid=$PID2"
wait $PID0 $PID2
echo "[all done] $(date +%F_%T) freeze tail 02"
