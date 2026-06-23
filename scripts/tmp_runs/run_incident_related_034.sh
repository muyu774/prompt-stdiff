#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/wzy/prompt-stdiff
export PATH="/mnt/data1/conda-ground/envs/stdiff/bin:$PATH"
LOG_ROOT="outputs/revision_8gpu_logs/incident_related_034_20260623"
mkdir -p "$LOG_ROOT" outputs/event_subset

run_eval_existing(){
  echo "[start] $(date +%F_%T) incident existing eval gpu=0"

  cfg="configs/pems08_pdformer_resdiff_incident_tail.yaml"
  ckpt="outputs/checkpoints/pems08_pdformer_resdiff_incident_tail_incident_tail_sem_student/last.pt"
  echo "[sem drop]"
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_event_subset.py --config "$cfg" --ckpt "$ckpt" --events_csv outputs/pems08_extreme_drop_events.csv --method Ours-IncidentTail-Sem --kind drop --split test --num_eval_samples 20 --gpu_id 0 --out_json outputs/event_subset/OursIncidentTailSem_pems08_drop_rerun.json
  echo "[sem spike]"
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_event_subset.py --config "$cfg" --ckpt "$ckpt" --events_csv outputs/pems08_extreme_events.csv --method Ours-IncidentTail-Sem --kind spike --split test --num_eval_samples 20 --gpu_id 0 --out_json outputs/event_subset/OursIncidentTailSem_pems08_spike_rerun.json
  echo "[sem full]"
  CUDA_VISIBLE_DEVICES=0 python evaluate.py --config "$cfg" --ckpt "$ckpt" --gpu_id 0 | tee "$LOG_ROOT/incident_tail_sem_full_eval.log"

  cfg="configs/pems08_pdformer_resdiff_incident_tail_nosem.yaml"
  ckpt="outputs/checkpoints/pems08_pdformer_resdiff_incident_tail_nosem_incident_tail_nosem_student/last.pt"
  echo "[nosem drop]"
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_event_subset.py --config "$cfg" --ckpt "$ckpt" --events_csv outputs/pems08_extreme_drop_events.csv --method Ours-IncidentTail-NoSem --kind drop --split test --num_eval_samples 20 --gpu_id 0 --out_json outputs/event_subset/OursIncidentTailNoSem_pems08_drop_rerun.json
  echo "[nosem spike]"
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_event_subset.py --config "$cfg" --ckpt "$ckpt" --events_csv outputs/pems08_extreme_events.csv --method Ours-IncidentTail-NoSem --kind spike --split test --num_eval_samples 20 --gpu_id 0 --out_json outputs/event_subset/OursIncidentTailNoSem_pems08_spike_rerun.json
  echo "[nosem full]"
  CUDA_VISIBLE_DEVICES=0 python evaluate.py --config "$cfg" --ckpt "$ckpt" --gpu_id 0 | tee "$LOG_ROOT/incident_tail_nosem_full_eval.log"

  echo "[done] $(date +%F_%T) incident existing eval"
}

run_train(){
  local gpu="$1"
  local cfg="$2"
  local tag="$3"
  echo "[start] $(date +%F_%T) train $tag gpu=$gpu"
  CUDA_VISIBLE_DEVICES="$gpu" bash scripts/run_pems08.sh \
    --mode train \
    --config_file "$cfg" \
    --horizon_steps 12 \
    --history_steps 24 \
    --gpu_id 0 \
    --lr 1e-3 \
    --eval_interval 5 \
    --train_num_eval_samples 8 \
    --num_eval_samples 20 \
    --max_eval_batches 20 \
    --save_tag "$tag"
  echo "[done] $(date +%F_%T) train $tag"
}

run_eval_existing > "$LOG_ROOT/gpu0_existing_eval.master.log" 2>&1 &
PID0=$!
run_train 3 configs/pems08_pdformer_resdiff_incident_tail_gaussian.yaml incident_tail_sem_gaussian > "$LOG_ROOT/gpu3_incident_tail_sem_gaussian_train.log" 2>&1 &
PID3=$!
run_train 4 configs/pems08_pdformer_resdiff_incident_tail_df5.yaml incident_tail_sem_student_df5 > "$LOG_ROOT/gpu4_incident_tail_sem_student_df5_train.log" 2>&1 &
PID4=$!

echo "[launch] gpu0_existing_eval pid=$PID0"
echo "[launch] gpu3_gaussian_train pid=$PID3"
echo "[launch] gpu4_df5_train pid=$PID4"
wait $PID0 $PID3 $PID4
echo "[all done] $(date +%F_%T) incident related 034"
