#!/usr/bin/env bash
set -uo pipefail
cd /mnt/data/wzy/prompt-stdiff
LOGDIR=outputs/revision_8gpu_logs/alignment_eval_gpu2_20260623
mkdir -p "$LOGDIR" outputs/event_subset outputs/event_root_cause
GPU=2
SAMPLES=20
PYBIN=/mnt/data1/conda-ground/envs/stdiff/bin/python

timestamp(){ date "+%Y-%m-%d %H:%M:%S"; }
run_cmd(){
  local name="$1"; shift
  echo "[start] $(timestamp) $name" | tee -a "$LOGDIR/launcher.log"
  "$@" > "$LOGDIR/${name}.log" 2>&1
  local code=$?
  echo "[done] $(timestamp) $name code=$code" | tee -a "$LOGDIR/launcher.log"
  return $code
}

MAIN_CFG=configs/hetero_horizon_fine_sweep/pems08_pdformer_hetero_nosem_hscale_linear106_112.yaml
MAIN_CKPT=outputs/checkpoints/pems08_pdformer_resdiff_hetero_nosem_nosem_hetero_lr1e3/last.pt
TAIL_SEM_CFG=configs/freeze_tail_hscale_20260623/pems08_tail_only_hscale_sem.yaml
TAIL_NOSEM_CFG=configs/freeze_tail_hscale_20260623/pems08_tail_only_hscale_nosem.yaml
TAIL_SEM_CKPT=outputs/checkpoints/pems08_tail_only_hscale_sem_tail_only_hscale_sem/last.pt
TAIL_NOSEM_CKPT=outputs/checkpoints/pems08_tail_only_hscale_nosem_tail_only_hscale_nosem/last.pt
DROP=outputs/pems08_extreme_drop_events.csv
SPIKE=outputs/pems08_extreme_events.csv

run_cmd tail_hscale_sem_full $PYBIN evaluate.py --config "$TAIL_SEM_CFG" --ckpt "$TAIL_SEM_CKPT" --gpu_id "$GPU" || true
run_cmd tail_hscale_nosem_full $PYBIN evaluate.py --config "$TAIL_NOSEM_CFG" --ckpt "$TAIL_NOSEM_CKPT" --gpu_id "$GPU" || true

for variant in sem nosem; do
  if [[ "$variant" == sem ]]; then
    CFG="$TAIL_SEM_CFG"; CKPT="$TAIL_SEM_CKPT"; METHOD=TailOnlyHScaleSem
  else
    CFG="$TAIL_NOSEM_CFG"; CKPT="$TAIL_NOSEM_CKPT"; METHOD=TailOnlyHScaleNoSem
  fi
  run_cmd ${variant}_drop_event $PYBIN scripts/eval_event_subset.py --config "$CFG" --ckpt "$CKPT" --events_csv "$DROP" --method "$METHOD" --kind drop --split test --num_eval_samples "$SAMPLES" --gpu_id "$GPU" --out_json outputs/event_subset/${METHOD}_pems08_drop.json || true
  run_cmd ${variant}_spike_event $PYBIN scripts/eval_event_subset.py --config "$CFG" --ckpt "$CKPT" --events_csv "$SPIKE" --method "$METHOD" --kind spike --split test --num_eval_samples "$SAMPLES" --gpu_id "$GPU" --out_json outputs/event_subset/${METHOD}_pems08_spike.json || true
  run_cmd ${variant}_drop_root $PYBIN scripts/eval_event_root_cause.py --config "$CFG" --ckpt "$CKPT" --events_csv "$DROP" --method "$METHOD" --kind drop --split test --num_eval_samples "$SAMPLES" --gpu_id "$GPU" --out_json outputs/event_root_cause/${METHOD}_pems08_drop.json || true
  run_cmd ${variant}_spike_root $PYBIN scripts/eval_event_root_cause.py --config "$CFG" --ckpt "$CKPT" --events_csv "$SPIKE" --method "$METHOD" --kind spike --split test --num_eval_samples "$SAMPLES" --gpu_id "$GPU" --out_json outputs/event_root_cause/${METHOD}_pems08_spike.json || true
done

run_cmd main_drop_root $PYBIN scripts/eval_event_root_cause.py --config "$MAIN_CFG" --ckpt "$MAIN_CKPT" --events_csv "$DROP" --method MainHScale --kind drop --split test --num_eval_samples "$SAMPLES" --gpu_id "$GPU" --out_json outputs/event_root_cause/MainHScale_pems08_drop.json || true
run_cmd main_spike_root $PYBIN scripts/eval_event_root_cause.py --config "$MAIN_CFG" --ckpt "$MAIN_CKPT" --events_csv "$SPIKE" --method MainHScale --kind spike --split test --num_eval_samples "$SAMPLES" --gpu_id "$GPU" --out_json outputs/event_root_cause/MainHScale_pems08_spike.json || true

if [[ -f outputs/prob_baselines/CSDI/pems08_samples.npz ]]; then
  run_cmd csdi_drop_root $PYBIN scripts/eval_event_root_cause.py --config configs/pems08.yaml --pred_npz outputs/prob_baselines/CSDI/pems08_samples.npz --events_csv "$DROP" --method CSDI --kind drop --split test --allow_truncate --gpu_id "$GPU" --out_json outputs/event_root_cause/CSDI_pems08_drop.json || true
  run_cmd csdi_spike_root $PYBIN scripts/eval_event_root_cause.py --config configs/pems08.yaml --pred_npz outputs/prob_baselines/CSDI/pems08_samples.npz --events_csv "$SPIKE" --method CSDI --kind spike --split test --allow_truncate --gpu_id "$GPU" --out_json outputs/event_root_cause/CSDI_pems08_spike.json || true
fi
if [[ -f outputs/prob_baselines/PriSTI/pems08_samples.npz ]]; then
  run_cmd pristi_drop_root $PYBIN scripts/eval_event_root_cause.py --config configs/pems08.yaml --pred_npz outputs/prob_baselines/PriSTI/pems08_samples.npz --events_csv "$DROP" --method PriSTI --kind drop --split test --allow_truncate --gpu_id "$GPU" --out_json outputs/event_root_cause/PriSTI_pems08_drop.json || true
  run_cmd pristi_spike_root $PYBIN scripts/eval_event_root_cause.py --config configs/pems08.yaml --pred_npz outputs/prob_baselines/PriSTI/pems08_samples.npz --events_csv "$SPIKE" --method PriSTI --kind spike --split test --allow_truncate --gpu_id "$GPU" --out_json outputs/event_root_cause/PriSTI_pems08_spike.json || true
fi

echo "[all done] $(timestamp) alignment eval gpu2" | tee -a "$LOGDIR/launcher.log"
