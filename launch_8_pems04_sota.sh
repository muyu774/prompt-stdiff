#!/usr/bin/env bash
set -e

mkdir -p logs

run_one () {
  GPU=$1
  LR=$2
  GAMMA=$3
  TAG=$4

  NAME="pems04_hybrid_mean_${TAG}"

  echo "[launch] gpu=${GPU} name=${NAME}"

  CUDA_VISIBLE_DEVICES=${GPU} nohup bash scripts/run_pems04.sh \
    --mode train \
    --config_file configs/pems04_hybrid_mean.yaml \
    --horizon_steps 12 \
    --history_steps 24 \
    --gpu_id 0 \
    --lr "${LR}" \
    --gamma "${GAMMA}" \
    --save_tag "${TAG}" \
    --eval_interval 1 \
    --train_num_eval_samples 20 \
    > "logs/${NAME}.log" 2>&1 &
}

run_one 0 3e-4 0.5 "lr3e4_g05"
run_one 1 5e-4 0.5 "lr5e4_g05"
run_one 2 7e-4 0.5 "lr7e4_g05"
run_one 3 3e-4 1.0 "lr3e4_g10"
run_one 4 5e-4 1.0 "lr5e4_g10"
run_one 5 7e-4 1.0 "lr7e4_g10"
run_one 6 3e-4 2.0 "lr3e4_g20"
run_one 7 5e-4 2.0 "lr5e4_g20"

wait
