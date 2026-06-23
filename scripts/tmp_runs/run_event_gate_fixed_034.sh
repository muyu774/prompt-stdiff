#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/wzy/prompt-stdiff

PY=/mnt/data1/conda-ground/envs/stdiff/bin/python
BASE="configs/hetero_horizon_fine_sweep/pems08_pdformer_hetero_nosem_hscale_linear106_112.yaml"
CKPT="outputs/checkpoints/pems08_pdformer_resdiff_hetero_nosem_nosem_hetero_lr1e3/last.pt"
DROP_EVENTS="outputs/pems08_extreme_drop_events.csv"
SPIKE_EVENTS="outputs/pems08_extreme_events.csv"
OUT="outputs/revision_8gpu_logs/event_gate_fixed_20260623"
CFGDIR="configs/uniform_wider_gate_20260623"
mkdir -p "$OUT" "$CFGDIR" outputs/event_subset

run_gaussian () {
  local gpu=0
  echo "[start] $(date +%F_%T) gaussian drop gpu=$gpu"
  "$PY" scripts/eval_gaussian_residual_event_subset.py \
    --config "$BASE" --ckpt "$CKPT" --events_csv "$DROP_EVENTS" --kind drop --split test \
    --gpu_id "$gpu" --num_eval_samples 20 --seed 42 \
    --method GaussianResidualHetero --setting pems08_drop \
    --out_json outputs/event_subset/GaussianResidualHetero_pems08_drop.json \
    | tee "$OUT/gaussian_drop.log"

  echo "[start] $(date +%F_%T) gaussian spike gpu=$gpu"
  "$PY" scripts/eval_gaussian_residual_event_subset.py \
    --config "$BASE" --ckpt "$CKPT" --events_csv "$SPIKE_EVENTS" --kind spike --split test \
    --gpu_id "$gpu" --num_eval_samples 20 --seed 42 \
    --method GaussianResidualHetero --setting pems08_spike \
    --out_json outputs/event_subset/GaussianResidualHetero_pems08_spike.json \
    | tee "$OUT/gaussian_spike.log"
  echo "[done] $(date +%F_%T) gaussian events"
}

make_cfg () {
  local scale=$1
  local cfg="$CFGDIR/pems08_hscale_uniform_${scale}.yaml"
  "$PY" - <<PY
from pathlib import Path
import yaml
from utils.config import load_config
cfg = load_config("$BASE")
cfg.setdefault("model", {})["residual_sample_scale"] = float("$scale")
Path("$cfg").parent.mkdir(parents=True, exist_ok=True)
with open("$cfg", "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
  echo "$cfg"
}

run_uniform_scale () {
  local gpu=$1
  local scale=$2
  local cfg
  cfg=$(make_cfg "$scale")

  echo "[start] $(date +%F_%T) uniform scale=$scale full-test gpu=$gpu"
  "$PY" evaluate.py --config "$cfg" --ckpt "$CKPT" --split test --gpu_id "$gpu" \
    | tee "$OUT/uniform_${scale}_full_test.log"

  echo "[start] $(date +%F_%T) uniform scale=$scale drop gpu=$gpu"
  "$PY" scripts/eval_event_subset.py \
    --config "$cfg" --ckpt "$CKPT" --events_csv "$DROP_EVENTS" --kind drop --split test \
    --gpu_id "$gpu" --num_eval_samples 20 \
    --method "OursUniformScale${scale}" --setting pems08_drop \
    --out_json "outputs/event_subset/OursUniformScale${scale}_pems08_drop.json" \
    | tee "$OUT/uniform_${scale}_drop.log"

  echo "[start] $(date +%F_%T) uniform scale=$scale spike gpu=$gpu"
  "$PY" scripts/eval_event_subset.py \
    --config "$cfg" --ckpt "$CKPT" --events_csv "$SPIKE_EVENTS" --kind spike --split test \
    --gpu_id "$gpu" --num_eval_samples 20 \
    --method "OursUniformScale${scale}" --setting pems08_spike \
    --out_json "outputs/event_subset/OursUniformScale${scale}_pems08_spike.json" \
    | tee "$OUT/uniform_${scale}_spike.log"
  echo "[done] $(date +%F_%T) uniform scale=$scale"
}

run_gaussian > "$OUT/gpu0_gaussian.master.log" 2>&1 &
PID0=$!
(
  run_uniform_scale 3 1.25
  run_uniform_scale 3 1.50
) > "$OUT/gpu3_uniform_125_150.master.log" 2>&1 &
PID3=$!
(
  run_uniform_scale 4 2.00
  run_uniform_scale 4 3.00
) > "$OUT/gpu4_uniform_200_300.master.log" 2>&1 &
PID4=$!

echo "[launch] gpu0_gaussian pid=$PID0"
echo "[launch] gpu3_uniform pid=$PID3"
echo "[launch] gpu4_uniform pid=$PID4"
wait $PID0 $PID3 $PID4
echo "[all done] $(date +%F_%T) event gate fixed"
