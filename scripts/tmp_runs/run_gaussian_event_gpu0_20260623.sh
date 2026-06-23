#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/wzy/prompt-stdiff
PYBIN=/mnt/data1/conda-ground/envs/stdiff/bin/python
LOGDIR=outputs/revision_8gpu_logs/gaussian_event_gpu0_20260623
mkdir -p "$LOGDIR" outputs/event_subset
CONFIG=configs/hetero_horizon_fine_sweep/pems08_pdformer_hetero_nosem_hscale_linear106_112.yaml
CKPT=outputs/checkpoints/pems08_pdformer_resdiff_hetero_nosem_nosem_hetero_lr1e3/last.pt
run_one () {
  NAME=$1
  CSV=$2
  KIND=$3
  echo "[start] $(date +%F\ %T) $NAME" | tee -a "$LOGDIR/master.log"
  CUDA_VISIBLE_DEVICES=0 "$PYBIN" scripts/eval_gaussian_residual_event_subset.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --events_csv "$CSV" \
    --kind "$KIND" \
    --method GaussianResidualHetero \
    --setting event_subset \
    --gpu_id 0 \
    --num_eval_samples 20 \
    --seed 42 \
    --out_json "outputs/event_subset/GaussianResidualHetero_pems08_${KIND}_seed42.json" \
    > "$LOGDIR/${NAME}.log" 2>&1
  echo "[done] $(date +%F\ %T) $NAME" | tee -a "$LOGDIR/master.log"
}
run_one gaussian_drop outputs/pems08_extreme_drop_events.csv drop
run_one gaussian_spike outputs/pems08_extreme_events.csv spike
