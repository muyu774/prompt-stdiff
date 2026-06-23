#!/usr/bin/env bash
set -euo pipefail

# Fetch official / commonly used baseline repositories for the T-ITS revision.
# Run from repo root:
#   bash scripts/setup_external_baselines.sh
#
# The training runners in this repo load these repositories dynamically and do
# not modify their original code.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

BASE_DIR="${BASE_DIR:-baselines/external_repos}"
mkdir -p "${BASE_DIR}"

clone_or_skip() {
  local name="$1"
  local url="$2"
  local dst="${BASE_DIR}/${name}"
  if [[ -d "${dst}/.git" ]]; then
    echo "[skip] ${name} already exists at ${dst}"
    return 0
  fi
  if [[ -e "${dst}" ]]; then
    echo "[warn] ${dst} exists but is not a git repo; leaving it untouched"
    return 0
  fi
  echo "[clone] ${name} <- ${url}"
  git clone --depth 1 "${url}" "${dst}"
}

# Deterministic baselines for Prompt 1 fairness experiments.
clone_or_skip AGCRN https://github.com/LeiBAI/AGCRN.git
clone_or_skip Graph-WaveNet https://github.com/nnzhan/Graph-WaveNet.git
clone_or_skip PDFormer https://github.com/BUAABIGSCity/PDFormer.git

# Probabilistic baselines are listed here for reproducibility, but integration
# runners should be added one by one after deterministic baselines are stable.
# Uncomment after Prompt 1 deterministic baselines are finished.
# clone_or_skip CSDI https://github.com/ermongroup/CSDI.git
# clone_or_skip DiffSTG https://github.com/wenhaomin/DiffSTG.git
# clone_or_skip PriSTI https://github.com/LMZZML/PriSTI.git

echo "[check] discovered AGCRN files:"
find "${BASE_DIR}/AGCRN" -maxdepth 8 -type f \( -iname '*agcrn*.py' -o -name 'AGCRN.py' -o -name 'agcrn.py' \) -print 2>/dev/null || true

echo "[check] discovered Graph-WaveNet files:"
find "${BASE_DIR}/Graph-WaveNet" -maxdepth 8 -type f \( -iname '*gwnet*.py' -o -iname '*wavenet*.py' -o -name 'model.py' \) -print 2>/dev/null || true

echo "[check] discovered PDFormer files:"
find "${BASE_DIR}/PDFormer" -maxdepth 8 -type f \( -iname '*pdformer*.py' -o -name 'PDFormer.py' -o -name 'pdformer.py' \) -print 2>/dev/null || true

echo "[done] external baseline repos are under ${BASE_DIR}"
