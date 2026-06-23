#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/wzy/prompt-stdiff
mkdir -p outputs

for s in 1.0 2.0 3.0 4.0 5.0; do
  cfg="configs/.tmp_p08_agcrn_resdiff_scale_${s}.yaml"
  cp configs/pems08_agcrn_resdiff.yaml "$cfg"

  python - <<PY
import yaml
p = "$cfg"
with open(p) as f:
    cfg = yaml.safe_load(f)
cfg["model"]["residual_sample_scale"] = float("$s")
with open(p, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

  echo "===== scale=${s} ====="
  if [[ -f outputs/checkpoints/pems08_agcrn_resdiff_full/best.pt ]]; then
    ckpt="outputs/checkpoints/pems08_agcrn_resdiff_full/best.pt"
  else
    ckpt="outputs/checkpoints/pems08_agcrn_resdiff_full/last.pt"
  fi

  python evaluate.py \
    --config "$cfg" \
    --ckpt "$ckpt" \
    --gpu_id 1 | tee "outputs/scale_p08_${s}.log"
done
