#!/usr/bin/env bash
set -uo pipefail
cd /mnt/data/wzy/prompt-stdiff
PYBIN=/mnt/data1/conda-ground/envs/stdiff/bin/python
GPU=0
LOGDIR=outputs/revision_8gpu_logs/main_table_align_gpu0_20260623
mkdir -p "$LOGDIR" outputs/main_table_json
CFG=configs/hetero_horizon_fine_sweep/pems08_pdformer_hetero_nosem_hscale_linear106_112.yaml
CKPT=outputs/checkpoints/pems08_pdformer_resdiff_hetero_nosem_nosem_hetero_lr1e3/last.pt

ts(){ date "+%Y-%m-%d %H:%M:%S"; }
run(){ name="$1"; shift; echo "[start] $(ts) $name" | tee -a "$LOGDIR/launcher.log"; "$@" > "$LOGDIR/$name.log" 2>&1; code=$?; echo "[done] $(ts) $name code=$code" | tee -a "$LOGDIR/launcher.log"; return $code; }

run gaussian_val "$PYBIN" scripts/eval_gaussian_residual_baseline.py --config "$CFG" --ckpt "$CKPT" --split val --gpu_id "$GPU" --num_eval_samples 20 --seed 42 --out_json outputs/main_table_json/GaussianResidualHetero_pems08_val_seed42.json || true
run gaussian_test "$PYBIN" scripts/eval_gaussian_residual_baseline.py --config "$CFG" --ckpt "$CKPT" --split test --gpu_id "$GPU" --num_eval_samples 20 --seed 42 --out_json outputs/main_table_json/GaussianResidualHetero_pems08_test_seed42.json || true

echo "[all done] $(ts) main table align gpu0" | tee -a "$LOGDIR/launcher.log"
