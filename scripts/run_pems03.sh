#!/usr/bin/env bash
set -euo pipefail

DATASET="pems03"
MODE="train"          # train | eval | infer
HORIZON_STEPS=12
HISTORY_STEPS=""
GPU_ID=0
CKPT=""
OUT_PATH=""
EVAL_INTERVAL=""
NUM_EVAL_SAMPLES=""
TRAIN_NUM_EVAL_SAMPLES=""
MAX_EVAL_BATCHES=""
LR=""
GAMMA=""
SAVE_TAG=""
DISABLE_VAL=0
DISABLE_DYNAMIC_SEMANTIC=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_pems03.sh [--mode train|eval|infer] [--horizon_steps 3|6|12] [--history_steps N] [--gpu_id 0-9] [--ckpt PATH] [--out PATH]
                           [--eval_interval N] [--num_eval_samples N] [--train_num_eval_samples N]
                           [--max_eval_batches N] [--lr FLOAT] [--gamma FLOAT]
                           [--save_tag STR] [--disable_val] [--disable_dynamic_semantic]

Examples:
  bash scripts/run_pems03.sh --mode train --horizon_steps 12 --gpu_id 0
  bash scripts/run_pems03.sh --mode eval  --horizon_steps 6  --gpu_id 1 --ckpt outputs/checkpoints/pems03_h6/best.pt
  bash scripts/run_pems03.sh --mode infer --horizon_steps 3  --gpu_id 2 --ckpt outputs/checkpoints/pems03_h3/best.pt --out outputs/pems03_h3_preds.npy
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --horizon_steps)
      HORIZON_STEPS="$2"
      shift 2
      ;;
    --gpu_id)
      GPU_ID="$2"
      shift 2
      ;;
    --history_steps)
      HISTORY_STEPS="$2"
      shift 2
      ;;
    --ckpt)
      CKPT="$2"
      shift 2
      ;;
    --out)
      OUT_PATH="$2"
      shift 2
      ;;
    --eval_interval)
      EVAL_INTERVAL="$2"
      shift 2
      ;;
    --num_eval_samples)
      NUM_EVAL_SAMPLES="$2"
      shift 2
      ;;
    --train_num_eval_samples)
      TRAIN_NUM_EVAL_SAMPLES="$2"
      shift 2
      ;;
    --max_eval_batches)
      MAX_EVAL_BATCHES="$2"
      shift 2
      ;;
    --lr)
      LR="$2"
      shift 2
      ;;
    --gamma)
      GAMMA="$2"
      shift 2
      ;;
    --save_tag)
      SAVE_TAG="$2"
      shift 2
      ;;
    --disable_val)
      DISABLE_VAL=1
      shift
      ;;
    --disable_dynamic_semantic)
      DISABLE_DYNAMIC_SEMANTIC=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[error] Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

if ! [[ "${GPU_ID}" =~ ^[0-9]$ ]]; then
  echo "[error] --gpu_id must be an integer in [0,9], got: ${GPU_ID}"
  exit 1
fi
if ! [[ "${HORIZON_STEPS}" =~ ^[0-9]+$ ]]; then
  echo "[error] --horizon_steps must be a positive integer, got: ${HORIZON_STEPS}"
  exit 1
fi
if [[ "${MODE}" != "train" && "${MODE}" != "eval" && "${MODE}" != "infer" ]]; then
  echo "[error] --mode must be one of: train | eval | infer"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_CFG="${ROOT_DIR}/configs/${DATASET}.yaml"
TMP_CFG="${ROOT_DIR}/configs/.tmp_${DATASET}_h${HORIZON_STEPS}_g${GPU_ID}_$$.yaml"

python - <<PY
from pathlib import Path
import yaml
from utils.config import load_config

base_cfg = Path("${BASE_CFG}")
out_cfg = Path("${TMP_CFG}")
h = int("${HORIZON_STEPS}")
history_steps = "${HISTORY_STEPS}"

cfg = load_config(str(base_cfg))
cfg["dataset"]["horizon_steps"] = h
if history_steps != "":
    cfg["dataset"]["history_steps"] = int(history_steps)
save_tag = "${SAVE_TAG}"

ev = [x for x in [3, 6, 12] if x <= h]
if h not in ev:
    ev.append(h)
cfg.setdefault("train", {})
cfg["train"]["eval_horizons"] = ev
save_dir = f"./outputs/checkpoints/${DATASET}_h{h}"
if save_tag != "":
    save_dir = f"{save_dir}_{save_tag}"
cfg["train"]["save_dir"] = save_dir

disable_val = "${DISABLE_VAL}" == "1"
eval_interval = "${EVAL_INTERVAL}"
num_eval_samples = "${NUM_EVAL_SAMPLES}"
train_num_eval_samples = "${TRAIN_NUM_EVAL_SAMPLES}"
max_eval_batches = "${MAX_EVAL_BATCHES}"
lr = "${LR}"
gamma = "${GAMMA}"
if disable_val:
    cfg["train"]["eval_interval"] = 10**9
if eval_interval != "":
    cfg["train"]["eval_interval"] = int(eval_interval)
if num_eval_samples != "":
    cfg["train"]["num_eval_samples"] = int(num_eval_samples)
    cfg["train"]["train_num_eval_samples"] = int(num_eval_samples)
if train_num_eval_samples != "":
    cfg["train"]["train_num_eval_samples"] = int(train_num_eval_samples)
if max_eval_batches != "":
    cfg["train"]["max_eval_batches"] = int(max_eval_batches)
if lr != "":
    cfg["train"]["lr"] = float(lr)
if gamma != "":
    cfg["model"]["gamma"] = float(gamma)
if "${DISABLE_DYNAMIC_SEMANTIC}" == "1":
    cfg.setdefault("dataset", {}).setdefault("dynamic_semantic", {})
    cfg["dataset"]["dynamic_semantic"]["enabled"] = False

out_cfg.parent.mkdir(parents=True, exist_ok=True)
with out_cfg.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)
print(f"[config] wrote {out_cfg}")
PY

trap 'rm -f "${TMP_CFG}"' EXIT

if [[ "${MODE}" == "train" ]]; then
  python train.py --config "${TMP_CFG}" --gpu_id "${GPU_ID}"
elif [[ "${MODE}" == "eval" ]]; then
  if [[ -z "${CKPT}" ]]; then
    CKPT="outputs/checkpoints/${DATASET}_h${HORIZON_STEPS}${SAVE_TAG:+_${SAVE_TAG}}/best.pt"
  fi
  python evaluate.py --config "${TMP_CFG}" --ckpt "${CKPT}" --gpu_id "${GPU_ID}"
else
  if [[ -z "${CKPT}" ]]; then
    CKPT="outputs/checkpoints/${DATASET}_h${HORIZON_STEPS}${SAVE_TAG:+_${SAVE_TAG}}/best.pt"
  fi
  if [[ -z "${OUT_PATH}" ]]; then
    OUT_PATH="outputs/${DATASET}_h${HORIZON_STEPS}_preds.npy"
  fi
  python infer.py --config "${TMP_CFG}" --ckpt "${CKPT}" --split test --out "${OUT_PATH}" --gpu_id "${GPU_ID}"
fi
