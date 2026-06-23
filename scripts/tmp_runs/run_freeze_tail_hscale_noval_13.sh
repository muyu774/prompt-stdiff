#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/wzy/prompt-stdiff
export PATH="/mnt/data1/conda-ground/envs/stdiff/bin:$PATH"
LOG_ROOT="outputs/revision_8gpu_logs/freeze_tail_hscale_noval_13_20260623"
mkdir -p "$LOG_ROOT"
run_one(){
  local gpu="$1"; local cfg="$2"; local tag="$3"
  echo "[start] $(date +%F_%T) $tag gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" bash scripts/run_pems08.sh \
    --mode train \
    --config_file "$cfg" \
    --horizon_steps 12 \
    --history_steps 24 \
    --gpu_id 0 \
    --lr 1e-3 \
    --disable_val \
    --save_tag "$tag"
  echo "[done] $(date +%F_%T) $tag"
}
run_one 1 configs/freeze_tail_hscale_20260623/pems08_tail_only_hscale_sem.yaml tail_only_hscale_sem > "$LOG_ROOT/gpu1_tail_only_hscale_sem.log" 2>&1 &
PID1=$!
run_one 3 configs/freeze_tail_hscale_20260623/pems08_tail_only_hscale_nosem.yaml tail_only_hscale_nosem > "$LOG_ROOT/gpu3_tail_only_hscale_nosem.log" 2>&1 &
PID3=$!
echo "[launch] gpu1_tail_only_hscale_sem pid=$PID1"
echo "[launch] gpu3_tail_only_hscale_nosem pid=$PID3"
wait $PID1 $PID3
echo "[all done] $(date +%F_%T) freeze tail hscale noval"
