#!/usr/bin/env bash
set -euo pipefail

cd /mnt/data/wzy/prompt-stdiff
mkdir -p outputs/resdiff_eval_logs

make_cfg () {
  base=$1
  out=$2
  scale=$3
  python - <<PY
import yaml
base = "$base"
out = "$out"
scale = float("$scale")
with open(base) as f:
    cfg = yaml.safe_load(f)
cfg["model"]["center_residual_samples"] = True
cfg["model"]["residual_sample_scale"] = scale
with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
}

run_eval () {
  gpu=$1
  name=$2
  cfg=$3
  ckpt=$4

  log="outputs/resdiff_eval_logs/${name}.log"
  echo "[start] $(date '+%F %T') ${name}" | tee "$log"
  CUDA_VISIBLE_DEVICES=$gpu python evaluate.py \
    --config "$cfg" \
    --ckpt "$ckpt" \
    --gpu_id 0 2>&1 | tee -a "$log"
  echo "[done] $(date '+%F %T') ${name}" | tee -a "$log"
}

# PeMS04 no-semantic: scale 3.0 / 4.0
make_cfg configs/pems04_agcrn_resdiff_nosem.yaml configs/.tmp_p04_nosem_s3.yaml 3.0
make_cfg configs/pems04_agcrn_resdiff_nosem.yaml configs/.tmp_p04_nosem_s4.yaml 4.0

# PeMS04 gamma0: scale 3.0 / 4.0
make_cfg configs/pems04_agcrn_resdiff.yaml configs/.tmp_p04_gamma0_s3.yaml 3.0
make_cfg configs/pems04_agcrn_resdiff.yaml configs/.tmp_p04_gamma0_s4.yaml 4.0
python - <<'PY'
import yaml
for p in ["configs/.tmp_p04_gamma0_s3.yaml", "configs/.tmp_p04_gamma0_s4.yaml"]:
    with open(p) as f:
        cfg = yaml.safe_load(f)
    cfg["model"]["gamma"] = 0.0
    with open(p, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
PY

# PeMS08 no-semantic/gamma0: use scale 2.0/3.0/4.0 candidates
make_cfg configs/pems08_agcrn_resdiff_nosem.yaml configs/.tmp_p08_nosem_s2.yaml 2.0
make_cfg configs/pems08_agcrn_resdiff_nosem.yaml configs/.tmp_p08_nosem_s3.yaml 3.0
make_cfg configs/pems08_agcrn_resdiff.yaml configs/.tmp_p08_gamma0_s3.yaml 3.0
python - <<'PY'
import yaml
p = "configs/.tmp_p08_gamma0_s3.yaml"
with open(p) as f:
    cfg = yaml.safe_load(f)
cfg["model"]["gamma"] = 0.0
with open(p, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY

run_eval 0 p04_nosem_s3 configs/.tmp_p04_nosem_s3.yaml outputs/checkpoints/pems04_agcrn_resdiff_nosem_nosem/last.pt &
run_eval 2 p04_nosem_s4 configs/.tmp_p04_nosem_s4.yaml outputs/checkpoints/pems04_agcrn_resdiff_nosem_nosem/last.pt &
run_eval 3 p04_gamma0_s3 configs/.tmp_p04_gamma0_s3.yaml outputs/checkpoints/pems04_agcrn_resdiff_gamma0/last.pt &
run_eval 4 p04_gamma0_s4 configs/.tmp_p04_gamma0_s4.yaml outputs/checkpoints/pems04_agcrn_resdiff_gamma0/last.pt &
run_eval 5 p08_nosem_s2 configs/.tmp_p08_nosem_s2.yaml outputs/checkpoints/pems08_agcrn_resdiff_nosem_nosem/best.pt &
run_eval 6 p08_nosem_s3 configs/.tmp_p08_nosem_s3.yaml outputs/checkpoints/pems08_agcrn_resdiff_nosem_nosem/best.pt &
run_eval 7 p08_gamma0_s3 configs/.tmp_p08_gamma0_s3.yaml outputs/checkpoints/pems08_agcrn_resdiff_gamma0/best.pt &

wait
echo "[all done] $(date '+%F %T')"
