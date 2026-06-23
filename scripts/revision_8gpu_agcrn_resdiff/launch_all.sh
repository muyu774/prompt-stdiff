#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
for s in scripts/revision_8gpu_agcrn_resdiff/gpu*.sh; do
  echo "[launch] ${s}"
  nohup bash "${s}" >/dev/null 2>&1 &
done
jobs -l
