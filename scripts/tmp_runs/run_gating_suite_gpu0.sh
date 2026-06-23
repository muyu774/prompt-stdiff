#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate stdiff
cd /mnt/data/wzy/prompt-stdiff

CFG="configs/hetero_horizon_fine_sweep/pems08_pdformer_hetero_nosem_hscale_linear106_112.yaml"
CKPT="outputs/checkpoints/pems08_pdformer_resdiff_hetero_nosem_nosem_hetero_lr1e3/last.pt"
OUT="outputs/revision_8gpu_logs/gating_suite_20260623"
EVENTS="outputs/pems08_extreme_events.csv"
GPU=0

echo "[start] $(date +%F_%T) diffusion full-val"
python evaluate.py --config "$CFG" --ckpt "$CKPT" --split val --gpu_id "$GPU" | tee "$OUT/diffusion_full_val.log"

echo "[start] $(date +%F_%T) diffusion full-test"
python evaluate.py --config "$CFG" --ckpt "$CKPT" --split test --gpu_id "$GPU" | tee "$OUT/diffusion_full_test.log"

echo "[start] $(date +%F_%T) gaussian full-test"
python scripts/eval_gaussian_residual_baseline.py \
  --config "$CFG" \
  --ckpt "$CKPT" \
  --split test \
  --gpu_id "$GPU" \
  --num_eval_samples 20 \
  --seed 42 \
  --out_json "$OUT/gaussian_full_test_seed42.json" | tee "$OUT/gaussian_full_test_seed42.log"

echo "[start] $(date +%F_%T) gaussian drop event"
python scripts/eval_gaussian_residual_event_subset.py \
  --config "$CFG" \
  --ckpt "$CKPT" \
  --events_csv "$EVENTS" \
  --kind drop \
  --split test \
  --gpu_id "$GPU" \
  --num_eval_samples 20 \
  --seed 42 \
  --method GaussianResidualHetero \
  --setting pems08_drop \
  --out_json outputs/event_subset/GaussianResidualHetero_pems08_drop.json | tee "$OUT/gaussian_event_drop_seed42.log"

echo "[start] $(date +%F_%T) gaussian spike event"
python scripts/eval_gaussian_residual_event_subset.py \
  --config "$CFG" \
  --ckpt "$CKPT" \
  --events_csv "$EVENTS" \
  --kind spike \
  --split test \
  --gpu_id "$GPU" \
  --num_eval_samples 20 \
  --seed 42 \
  --method GaussianResidualHetero \
  --setting pems08_spike \
  --out_json outputs/event_subset/GaussianResidualHetero_pems08_spike.json | tee "$OUT/gaussian_event_spike_seed42.log"

echo "[done] $(date +%F_%T) gating suite"
