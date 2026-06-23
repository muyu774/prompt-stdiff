#!/usr/bin/env bash
set -euo pipefail
cd /mnt/data/wzy/prompt-stdiff
export PATH="/mnt/data1/conda-ground/envs/stdiff/bin:$PATH"
LOG_ROOT="outputs/revision_8gpu_logs/three_eval_023_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
ln -sfn "$(basename "$LOG_ROOT")" outputs/revision_8gpu_logs/three_eval_023_latest
mkdir -p outputs/event_subset

# GPU0: event subset for current best aggregate model.
(
  echo "[start] $(date) best_hetero_hscale_events"
  cfg="configs/hetero_horizon_fine_sweep/pems08_pdformer_hetero_nosem_hscale_linear106_112.yaml"
  ckpt="outputs/checkpoints/pems08_pdformer_resdiff_hetero_nosem_nosem_hetero_lr1e3/last.pt"
  echo "[drop event]"
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_event_subset.py --config "$cfg" --ckpt "$ckpt" --events_csv outputs/pems08_extreme_drop_events.csv --method Ours-Hetero-HScale --kind drop --split test --num_eval_samples 20 --gpu_id 0 --out_json outputs/event_subset/OursHeteroHScale_pems08_drop.json
  echo "[spike event]"
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_event_subset.py --config "$cfg" --ckpt "$ckpt" --events_csv outputs/pems08_extreme_events.csv --method Ours-Hetero-HScale --kind spike --split test --num_eval_samples 20 --gpu_id 0 --out_json outputs/event_subset/OursHeteroHScale_pems08_spike.json
  echo "[done] $(date) best_hetero_hscale_events"
) >"$LOG_ROOT/best_hetero_hscale_events.log" 2>&1 &
echo $! >"$LOG_ROOT/best_hetero_hscale_events.pid"

# GPU2: event + full-test for no-sem incident-tail ablation.
(
  echo "[start] $(date) incident_tail_nosem_eval"
  cfg="configs/pems08_pdformer_resdiff_incident_tail_nosem.yaml"
  ckpt="outputs/checkpoints/pems08_pdformer_resdiff_incident_tail_nosem_incident_tail_nosem_student/last.pt"
  echo "[drop event]"
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_event_subset.py --config "$cfg" --ckpt "$ckpt" --events_csv outputs/pems08_extreme_drop_events.csv --method Ours-IncidentTail-NoSem --kind drop --split test --num_eval_samples 20 --gpu_id 0 --out_json outputs/event_subset/OursIncidentTailNoSem_pems08_drop.json
  echo "[spike event]"
  CUDA_VISIBLE_DEVICES=2 python scripts/eval_event_subset.py --config "$cfg" --ckpt "$ckpt" --events_csv outputs/pems08_extreme_events.csv --method Ours-IncidentTail-NoSem --kind spike --split test --num_eval_samples 20 --gpu_id 0 --out_json outputs/event_subset/OursIncidentTailNoSem_pems08_spike.json
  echo "[full test]"
  CUDA_VISIBLE_DEVICES=2 python evaluate.py --config "$cfg" --ckpt "$ckpt" --gpu_id 0
  echo "[done] $(date) incident_tail_nosem_eval"
) >"$LOG_ROOT/incident_tail_nosem_eval.log" 2>&1 &
echo $! >"$LOG_ROOT/incident_tail_nosem_eval.pid"

# GPU3: validation-selected node-group calibration on test.
(
  echo "[start] $(date) node_group_wide_test"
  cfg="configs/node_group_hscale_pems08/p08_ng_wide.yaml"
  ckpt="outputs/checkpoints/pems08_pdformer_resdiff_hetero_nosem_nosem_hetero_lr1e3/last.pt"
  CUDA_VISIBLE_DEVICES=3 python evaluate.py --config "$cfg" --ckpt "$ckpt" --gpu_id 0
  echo "[done] $(date) node_group_wide_test"
) >"$LOG_ROOT/node_group_wide_test.log" 2>&1 &
echo $! >"$LOG_ROOT/node_group_wide_test.pid"

echo "[logs] $LOG_ROOT"
sleep 6
for f in "$LOG_ROOT"/*.log; do echo "===== $(basename "$f") ====="; tail -n 20 "$f"; done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
