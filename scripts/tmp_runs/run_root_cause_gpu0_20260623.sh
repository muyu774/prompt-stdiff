#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/wzy/prompt-stdiff
PYBIN=/mnt/data1/conda-ground/envs/stdiff/bin/python
LOGDIR=outputs/revision_8gpu_logs/root_cause_gpu0_20260623
mkdir -p "$LOGDIR" outputs/event_root_cause
CONFIG=configs/hetero_horizon_fine_sweep/pems08_pdformer_hetero_nosem_hscale_linear106_112.yaml
CKPT=outputs/checkpoints/pems08_pdformer_resdiff_hetero_nosem_nosem_hetero_lr1e3/last.pt
run_model () {
  NAME=$1
  CSV=$2
  KIND=$3
  echo "[start] $(date +%F\ %T) $NAME" | tee -a "$LOGDIR/master.log"
  CUDA_VISIBLE_DEVICES=0 "$PYBIN" scripts/eval_event_root_cause.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --events_csv "$CSV" \
    --kind "$KIND" \
    --method OursHeteroHScale \
    --setting event_root_cause \
    --gpu_id 0 \
    --num_eval_samples 20 \
    --out_json "outputs/event_root_cause/OursHeteroHScale_pems08_${KIND}_gpu0_20260623.json" \
    > "$LOGDIR/${NAME}.log" 2>&1
  echo "[done] $(date +%F\ %T) $NAME" | tee -a "$LOGDIR/master.log"
}
run_model ours_hscale_drop outputs/pems08_extreme_drop_events.csv drop
run_model ours_hscale_spike outputs/pems08_extreme_events.csv spike
