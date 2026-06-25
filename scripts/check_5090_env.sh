#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# check_5090_env.sh  --  Decision (1) preflight for RTX 5090 (Blackwell sm_120)
#
# The `stdiff` env is built for A100 (sm_80). A 5090 needs CUDA 12.8+ /
# PyTorch 2.7+; on an older env you get `no kernel image is available` and the
# task either fails silently or falls back painfully slowly. Run this BEFORE
# putting any critical-path chain on a 5090.
#
# Usage:
#   bash scripts/check_5090_env.sh [GPU_INDEX]
#
#   GPU_INDEX  Physical GPU index to probe (default: 0, or $CUDA_VISIBLE_DEVICES
#              if it already pins a single device).
#
# Exit code:
#   0  -> 5090 is usable from this env (matmul is finite AND capability == (12, 0))
#   1  -> NOT usable (build a dedicated cu128 env for the 5090 first)
#
# A "pass" prints a finite matmul sum plus `(12, 0)`; anything else fails.
# ---------------------------------------------------------------------------
set -euo pipefail

GPU_INDEX="${1:-${CUDA_VISIBLE_DEVICES:-0}}"
# If CUDA_VISIBLE_DEVICES carried a list, keep only the first entry.
GPU_INDEX="${GPU_INDEX%%,*}"

if ! [[ "${GPU_INDEX}" =~ ^[0-9]+$ ]]; then
  echo "[check_5090_env] ERROR: GPU index must be a non-negative integer, got: ${GPU_INDEX}" >&2
  exit 1
fi

echo "[check_5090_env] probing physical GPU index ${GPU_INDEX} ..."

CUDA_VISIBLE_DEVICES="${GPU_INDEX}" python - <<'PY'
import sys

try:
    import torch
except Exception as exc:  # pragma: no cover - env issue
    print(f"[check_5090_env] FAIL: cannot import torch: {exc}")
    sys.exit(1)

print(f"[check_5090_env] torch={torch.__version__} cuda={torch.version.cuda}")

if not torch.cuda.is_available():
    print("[check_5090_env] FAIL: torch.cuda.is_available() is False")
    sys.exit(1)

try:
    x = torch.randn(4096, 4096, device="cuda")
    s = float((x @ x).sum())
    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
except Exception as exc:
    # The classic Blackwell-on-old-env signature: `no kernel image is available`.
    print(f"[check_5090_env] FAIL: CUDA kernel launch failed: {exc}")
    sys.exit(1)

print(f"[check_5090_env] matmul_sum={s}")
print(f"[check_5090_env] device={name} capability={cap}")

import math

if not math.isfinite(s):
    print("[check_5090_env] FAIL: matmul produced a non-finite value")
    sys.exit(1)

if tuple(cap) != (12, 0):
    print(
        f"[check_5090_env] WARN: capability {cap} != (12, 0); this probe targets "
        "the 5090 (sm_120). If this index is an A100 the env is fine for A100 "
        "but this is not a 5090."
    )
    sys.exit(1)

print("[check_5090_env] PASS: 5090 (sm_120) is usable from this env")
sys.exit(0)
PY
