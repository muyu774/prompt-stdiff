#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate stdiff
cd /mnt/data/wzy/prompt-stdiff

while pgrep -f "pems08_pdformer_hetero_nosem_hscale_linear106_112_seed.*train.py" >/dev/null; do
  echo "[wait] multiseed training still running $(date +%F_%T)"
  sleep 300
done
while pgrep -f "scripts/tmp_runs/run_gating_suite_gpu0.sh|scripts/tmp_runs/run_uniform_wider_after_gating_gpu0.sh" >/dev/null; do
  echo "[wait] gpu0 gate/uniform queue still running $(date +%F_%T)"
  sleep 300
done

OUT="outputs/revision_8gpu_logs/multiseed_final_eval_20260623"
mkdir -p "$OUT"
GPU=0
for SEED in 7 123 2026 3407; do
  CFG="configs/multiseed_pems08_hetero/pems08_pdformer_hetero_nosem_hscale_linear106_112_seed${SEED}.yaml"
  CKPT="outputs/checkpoints/pems08_pdformer_resdiff_hetero_nosem_hscale_linear106_112_seed${SEED}/last.pt"
  echo "[start] $(date +%F_%T) seed=${SEED} full-test"
  python evaluate.py --config "$CFG" --ckpt "$CKPT" --split test --gpu_id "$GPU" | tee "$OUT/seed${SEED}_full_test.log"
done

echo "[done] multiseed final eval $(date +%F_%T)"
