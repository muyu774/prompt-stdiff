#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/wzy/prompt-stdiff

BASE_CFG=${BASE_CFG:-configs/hetero_horizon_fine_sweep/pems08_pdformer_hetero_nosem_hscale_linear106_112.yaml}
SEEDS_STR=${SEEDS:-7 2026}
GPUS_STR=${GPUS:-0 4}
EPOCHS=${EPOCHS:-50}
LR=${LR:-1e-3}
LOG_ROOT=${LOG_ROOT:-outputs/revision_8gpu_logs/multiseed_pems08_hetero_$(date +%Y%m%d_%H%M%S)}
CFG_DIR=${CFG_DIR:-configs/multiseed_pems08_hetero}

mkdir -p "$LOG_ROOT" "$CFG_DIR"
ln -sfn "$(basename "$LOG_ROOT")" outputs/revision_8gpu_logs/multiseed_pems08_hetero_latest

read -r -a SEEDS_ARR <<< "$SEEDS_STR"
read -r -a GPUS_ARR <<< "$GPUS_STR"

if [[ ${#SEEDS_ARR[@]} -gt ${#GPUS_ARR[@]} ]]; then
  echo "[error] need at least as many GPUS as SEEDS" >&2
  exit 1
fi

for i in "${!SEEDS_ARR[@]}"; do
  seed=${SEEDS_ARR[$i]}
  gpu=${GPUS_ARR[$i]}
  tag="seed${seed}"
  cfg="$CFG_DIR/pems08_pdformer_hetero_nosem_hscale_linear106_112_${tag}.yaml"
  log="$LOG_ROOT/${tag}_gpu${gpu}.log"

  python - <<PY
from pathlib import Path
import yaml
from utils.config import load_config
base = Path("$BASE_CFG")
out = Path("$cfg")
cfg = load_config(str(base))
cfg.setdefault("train", {})
cfg["train"]["seed"] = int("$seed")
cfg["train"]["epochs"] = int("$EPOCHS")
cfg["train"]["lr"] = float("$LR")
cfg["train"]["save_dir"] = "./outputs/checkpoints/pems08_pdformer_resdiff_hetero_nosem_hscale_linear106_112_$tag"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
print(f"[config] wrote {out}")
PY

  echo "[launch] seed=${seed} gpu=${gpu} cfg=${cfg} log=${log}"
  nohup bash -lc "source /root/miniconda3/etc/profile.d/conda.sh && conda activate stdiff && cd /mnt/data/wzy/prompt-stdiff && echo \"[start] seed=${seed} gpu=${gpu} cfg=${cfg}\" && python train.py --config \"${cfg}\" --gpu_id \"${gpu}\" && echo \"[done] seed=${seed} gpu=${gpu}\"" > "$log" 2>&1 &
done

echo "[logs] $LOG_ROOT"
