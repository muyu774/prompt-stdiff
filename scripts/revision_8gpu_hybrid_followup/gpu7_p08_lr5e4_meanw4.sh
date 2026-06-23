#!/usr/bin/env bash
set -euo pipefail
mkdir -p outputs/revision_8gpu_logs/hybrid_followup configs/.tmp_followup
LOG="outputs/revision_8gpu_logs/hybrid_followup/gpu7_p08_lr5e4_meanw4.log"
CFG="configs/.tmp_followup/pems08_hybrid_meanw4.yaml"
python - <<'PY'
from pathlib import Path
import yaml
from utils.config import load_config
cfg = load_config("configs/pems08_hybrid_mean.yaml")
cfg.setdefault("train", {})["loss_mean_weight"] = 4.0
cfg["train"]["save_dir"] = "./outputs/checkpoints/pems08_hybrid_meanw4"
Path("configs/.tmp_followup").mkdir(parents=True, exist_ok=True)
Path("configs/.tmp_followup/pems08_hybrid_meanw4.yaml").write_text(
    yaml.safe_dump(cfg, sort_keys=False, allow_unicode=False),
    encoding="utf-8",
)
PY
echo "[start] $(date '+%F %T') gpu7_p08_lr5e4_meanw4" | tee -a "$LOG"
bash scripts/run_pems08.sh \
  --mode train \
  --config_file "$CFG" \
  --horizon_steps 12 \
  --history_steps 24 \
  --gpu_id 7 \
  --lr 5e-4 \
  --gamma 0.3 \
  --eval_interval 5 \
  --train_num_eval_samples 8 \
  --num_eval_samples 20 \
  --max_eval_batches 20 \
  --save_tag hybrid_follow_meanw4 | tee -a "$LOG"
echo "[done] $(date '+%F %T') gpu7_p08_lr5e4_meanw4" | tee -a "$LOG"
