#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/wzy/prompt-stdiff
PYBIN=/mnt/data1/conda-ground/envs/stdiff/bin/python
LOGDIR=outputs/revision_8gpu_logs/p0_prob_npz_eval_gpu3_20260623
OUTDIR=outputs/p0_tables_20260623
mkdir -p "$LOGDIR" "$OUTDIR"
CSV="$OUTDIR/prob_baselines_pems08.csv"
MD="$OUTDIR/prob_baselines_pems08.md"
run_one () {
  local NAME=$1
  local NPZ=$2
  local METHOD=$3
  local SETTING=$4
  echo "[start] $(date +%F\ %T) $NAME" | tee -a "$LOGDIR/master.log"
  set +e
  "$PYBIN" scripts/eval_probabilistic_npz.py \
    --pred_npz "$NPZ" \
    --config configs/pems08.yaml \
    --method "$METHOD" \
    --setting "$SETTING" \
    --implementation canonical_npz \
    --space normalized \
    --sample_axis 0 \
    --gpu_id 3 \
    --csv "$CSV" \
    --md "$MD" \
    --title "P0 PeMS08 probabilistic baselines" \
    --fail_on_sanity_warning \
    > "$LOGDIR/${NAME}.log" 2>&1
  local code=$?
  set -e
  echo "[done] $(date +%F\ %T) $NAME code=$code" | tee -a "$LOGDIR/master.log"
}
run_one csdi_pems08 outputs/prob_baselines/CSDI/pems08_samples.npz CSDI official
run_one pristi_pems08 outputs/prob_baselines/PriSTI/pems08_samples.npz PriSTI official
run_one diffstg_main_pems08 outputs/prob_baselines/DiffSTG/pems08_main_samples.npz DiffSTG canonical_main
run_one diffstg_h48_pems08 outputs/prob_baselines/DiffSTG/pems08_h48_seed3407_samples.npz DiffSTG canonical_h48_seed3407
printf "[all done] %s\n" "$(date +%F\ %T)" | tee -a "$LOGDIR/master.log"
