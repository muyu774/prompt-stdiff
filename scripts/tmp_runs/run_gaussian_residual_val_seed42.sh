#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate stdiff
cd /mnt/data/wzy/prompt-stdiff
echo "[start] gaussian_residual_val gpu=3 $(date +%F_%T)"
python scripts/eval_gaussian_residual_baseline.py \
  --config configs/hetero_horizon_fine_sweep/pems08_pdformer_hetero_nosem_hscale_linear106_112.yaml \
  --ckpt outputs/checkpoints/pems08_pdformer_resdiff_hetero_nosem_nosem_hetero_lr1e3/last.pt \
  --split val \
  --gpu_id 3 \
  --num_eval_samples 20 \
  --seed 42 \
  --out_json outputs/revision_8gpu_logs/gaussian_residual_gate/pems08_gaussian_residual_val_seed42.json
echo "[done] gaussian_residual_val $(date +%F_%T)"
