#!/usr/bin/env bash
set -euo pipefail
source /root/miniconda3/etc/profile.d/conda.sh
conda activate stdiff
cd /mnt/data/wzy/prompt-stdiff

while pgrep -f "scripts/tmp_runs/run_gating_suite_gpu0.sh" >/dev/null; do
  echo "[wait] gating suite still running $(date +%F_%T)"
  sleep 300
done

BASE="configs/hetero_horizon_fine_sweep/pems08_pdformer_hetero_nosem_hscale_linear106_112.yaml"
CKPT="outputs/checkpoints/pems08_pdformer_resdiff_hetero_nosem_nosem_hetero_lr1e3/last.pt"
EVENTS="outputs/pems08_extreme_events.csv"
OUT="outputs/revision_8gpu_logs/uniform_wider_gate_20260623"
CFGDIR="configs/uniform_wider_gate_20260623"
mkdir -p "$OUT" "$CFGDIR" outputs/event_subset
GPU=0

for S in 1.25 1.50 2.00 3.00; do
  CFG="$CFGDIR/pems08_hscale_uniform_${S}.yaml"
  python - <<PY
from pathlib import Path
import yaml
from utils.config import load_config
cfg = load_config("$BASE")
cfg.setdefault("model", {})["residual_sample_scale"] = float("$S")
Path("$CFG").parent.mkdir(parents=True, exist_ok=True)
with open("$CFG", "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
  echo "[start] $(date +%F_%T) uniform scale=$S full-test"
  python evaluate.py --config "$CFG" --ckpt "$CKPT" --split test --gpu_id "$GPU" | tee "$OUT/diffusion_uniform_${S}_full_test.log"

  echo "[start] $(date +%F_%T) uniform scale=$S drop"
  python scripts/eval_event_subset.py \
    --config "$CFG" \
    --ckpt "$CKPT" \
    --events_csv "$EVENTS" \
    --kind drop \
    --split test \
    --gpu_id "$GPU" \
    --num_eval_samples 20 \
    --method "OursUniformScale${S}" \
    --setting pems08_drop \
    --out_json "outputs/event_subset/OursUniformScale${S}_pems08_drop.json" | tee "$OUT/diffusion_uniform_${S}_drop.log"

  echo "[start] $(date +%F_%T) uniform scale=$S spike"
  python scripts/eval_event_subset.py \
    --config "$CFG" \
    --ckpt "$CKPT" \
    --events_csv "$EVENTS" \
    --kind spike \
    --split test \
    --gpu_id "$GPU" \
    --num_eval_samples 20 \
    --method "OursUniformScale${S}" \
    --setting pems08_spike \
    --out_json "outputs/event_subset/OursUniformScale${S}_pems08_spike.json" | tee "$OUT/diffusion_uniform_${S}_spike.log"
done

echo "[done] uniform wider gate $(date +%F_%T)"
