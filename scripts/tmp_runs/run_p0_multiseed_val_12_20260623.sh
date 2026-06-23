#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/wzy/prompt-stdiff
PYBIN=/mnt/data1/conda-ground/envs/stdiff/bin/python
LOGDIR=outputs/revision_8gpu_logs/p0_multiseed_val_20260623
mkdir -p "$LOGDIR"
run_eval () {
  local SEED=$1
  local GPU=$2
  local CFG="configs/multiseed_pems08_hetero/pems08_pdformer_hetero_nosem_hscale_linear106_112_seed${SEED}.yaml"
  local CKPT="outputs/checkpoints/pems08_pdformer_resdiff_hetero_nosem_hscale_linear106_112_seed${SEED}/last.pt"
  local NAME="pems08_hscale_seed${SEED}_val"
  echo "[start] $(date +%F\ %T) ${NAME} gpu=${GPU}" | tee -a "$LOGDIR/master.log"
  "$PYBIN" evaluate.py \
    --config "$CFG" \
    --ckpt "$CKPT" \
    --split val \
    --gpu_id "$GPU" \
    > "$LOGDIR/${NAME}.log" 2>&1
  echo "[done] $(date +%F\ %T) ${NAME}" | tee -a "$LOGDIR/master.log"
}
run_eval 7 1 &
run_eval 123 2 &
wait
run_eval 2026 1 &
run_eval 3407 2 &
wait
printf "[all done] %s\n" "$(date +%F\ %T)" | tee -a "$LOGDIR/master.log"
