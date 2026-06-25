#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# chain_dispatcher.sh -- pin one (seed, dataset) chain per GPU and run the full
# `backbone -> gate -> diffusion -> eval` pipeline, with per-card-type env
# gating (A100 vs 5090) and per-stage retries.
#
# Why "chains, not cards": every job here is a small independent chain
#   (seed, dataset) = backbone -> diffusion -> eval (graphs are 170-360 nodes,
#   nothing needs >32GB or multi-GPU). So we MAXIMISE the number of parallel
#   chains: 1 chain == 1 GPU, the whole chain (both the Gaussian-baseline and
#   diffusion arms) finishes on that GPU before the card is reused. A chain is
#   never split across GPUs, otherwise the in-chain GB-vs-diffusion comparison
#   gets polluted by hardware float differences and the cached `mu_s` can't be
#   read back.
#
# Decision (1) -- 5090 (Blackwell sm_120) needs CUDA 12.8+ / PyTorch 2.7+. Any
# chain assigned to a GPU typed `5090` in CARD_MAP is preflight-checked with
# scripts/check_5090_env.sh and is SKIPPED (not silently failed) if the env is
# too old. Keep the critical path on known-good A100s until that passes.
#
# ---------------------------------------------------------------------------
# QUICK START
#   1. Discover your real card<->index mapping (decision (2)):
#          bash scripts/chain_dispatcher.sh inspect
#      then set CARD_MAP to match, e.g. GPU4 is the 80GB A100 tenant:
#          export CARD_MAP="0:5090 1:5090 2:5090 3:5090 4:A100 5:A100 6:A100 7:A100"
#   2. (Optional) point CHAIN_MANIFEST at your own manifest file; otherwise the
#      built-in default manifest below is used.
#   3. Launch:
#          bash scripts/chain_dispatcher.sh run            # all chains
#          bash scripts/chain_dispatcher.sh run pems08_s42 # a subset by name
#          bash scripts/chain_dispatcher.sh status
#          bash scripts/chain_dispatcher.sh tail
#
# ---------------------------------------------------------------------------
# MANIFEST FORMAT (one chain per line, fields separated by ` @@ `):
#
#   NAME @@ GPU @@ BACKBONE_CMD @@ GATE_CMD @@ DIFFUSION_CMD @@ EVAL_CMD
#
#   NAME           unique chain id (also the log file stem)
#   GPU            physical GPU index this chain is PINNED to for ALL stages
#   *_CMD          shell command for that stage, or `-` to skip the stage
#
#   * Lines starting with `#` and blank lines are ignored.
#   * Every stage runs with CUDA_VISIBLE_DEVICES=<GPU> exported, so the inner
#     command should address the device as cuda:0 / --gpu_id 0.
#   * The GATE stage is the natural place for agcrn_backbone_gate.py: if it
#     exits non-zero the chain stops BEFORE wasting a diffusion run. If the gate
#     script does not exist yet the stage is treated as a soft skip (warn only).
# ---------------------------------------------------------------------------
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

FIELD_SEP=" @@ "
RETRIES="${RETRIES:-1}"                 # extra attempts per stage on failure
PREFLIGHT="${PREFLIGHT:-1}"             # 1 = run check_5090_env.sh before 5090 chains
GATE_SOFT="${GATE_SOFT:-1}"             # 1 = missing gate script is a soft skip
LOG_ROOT="${LOG_ROOT:-outputs/chain_dispatcher_logs/$(date +%Y%m%d_%H%M%S)}"
LATEST_LINK="outputs/chain_dispatcher_logs/latest"

# CARD_MAP: "INDEX:TYPE ..." where TYPE is A100 or 5090. EDIT to match your box
# (run `inspect` first). The default is a PLACEHOLDER and must be overridden.
CARD_MAP="${CARD_MAP:-0:A100 1:A100 2:A100 3:A100 4:A100 5:5090 6:5090 7:5090 8:5090 9:5090 10:5090 11:5090 12:5090}"

# ---------------------------------------------------------------------------
# Default manifest. Override by exporting CHAIN_MANIFEST=/path/to/manifest.
#
# Wave 1 (13 chains == 13 GPUs): 5 seed x 3 datasets = 15 > 13, so the third
# dataset starts with 3 seeds. PeMS04 keeps running where it is (do NOT move
# it), PeMS08 PDFormer goes on the A100s, the spare 5090s warm up the third
# dataset + Wave-2 work below.
#
# NOTE: the commands reference configs/scripts that already exist in this repo
# (run_pems0{3,4,8}.sh, baselines.runners.run_agcrn, the *_resdiff.yaml configs,
# and the Wave-2 eval scripts). Adjust seeds / tags / configs to taste.
# ---------------------------------------------------------------------------
read -r -d '' DEFAULT_MANIFEST <<'EOF' || true
# ===== Wave 1 / PeMS04 AGCRN -- DO NOT MOVE (already running on its 5 cards) =====
# Listed for completeness only; left disabled so the dispatcher won't relaunch
# them on top of the in-flight jobs. Uncomment only if you really mean to.
# pems04_s42   @@ 0 @@ python -m baselines.runners.run_agcrn --config configs/pems04.yaml --gpu_id 0 --epochs 50 --eval_interval 5 --lr 1e-3 --save_tag agcrn_s42 @@ python scripts/agcrn_backbone_gate.py --dataset pems04 --tag agcrn_s42 @@ bash scripts/run_pems04.sh --mode train --config_file configs/pems04_agcrn_resdiff.yaml --gpu_id 0 --save_tag s42 @@ bash scripts/run_pems04.sh --mode eval --config_file configs/pems04_agcrn_resdiff.yaml --gpu_id 0 --save_tag s42

# ===== Wave 1 / PeMS08 PDFormer -- A100 PRIORITY (heavier + env must be stable) =====
# same-backbone residual-generator verdict (GB vs diffusion) only needs a shared
# mu_s, so even the current PDFormer is enough for the C2 conclusion. Lowest-
# regret path is to fix ToD/DoW first (a couple of 5090s) then run all 5 seeds
# once; until then these reuse the existing PDFormer resdiff config.
pems08_pdformer_s42   @@ 4 @@ - @@ - @@ bash scripts/run_pems08.sh --mode train --config_file configs/pems08_pdformer_resdiff.yaml --gpu_id 0 --lr 1e-3 --save_tag s42 @@ bash scripts/run_pems08.sh --mode eval --config_file configs/pems08_pdformer_resdiff.yaml --gpu_id 0 --save_tag s42

# ===== Wave 1 / third dataset on the 5090s (env-gated) =====================
# METR-LA is the intended third dataset (independent network + rich incident
# regime, useful for C3). It is NOT yet wired into this repo (run_metrla.sh /
# configs/metrla*.yaml + data/metr_la/ must be added first), so it is left as a
# documented placeholder. Until then, PeMS03 is a real, supported third network
# that the 5090s can preheat right now:
pems03_agcrn_s42   @@ 5 @@ python -m baselines.runners.run_agcrn --config configs/pems03.yaml --gpu_id 0 --epochs 50 --eval_interval 5 --lr 1e-3 --save_tag agcrn_s42 @@ python scripts/agcrn_backbone_gate.py --dataset pems03 --tag agcrn_s42 @@ - @@ -
pems03_agcrn_s123  @@ 6 @@ python -m baselines.runners.run_agcrn --config configs/pems03.yaml --gpu_id 0 --epochs 50 --eval_interval 5 --lr 1e-3 --save_tag agcrn_s123 @@ python scripts/agcrn_backbone_gate.py --dataset pems03 --tag agcrn_s123 @@ - @@ -
pems03_agcrn_s2024 @@ 7 @@ python -m baselines.runners.run_agcrn --config configs/pems03.yaml --gpu_id 0 --epochs 50 --eval_interval 5 --lr 1e-3 --save_tag agcrn_s2024 @@ python scripts/agcrn_backbone_gate.py --dataset pems03 --tag agcrn_s2024 @@ - @@ -
# metrla_agcrn_s42 @@ 8 @@ python -m baselines.runners.run_agcrn --config configs/metrla.yaml --gpu_id 0 --epochs 50 --eval_interval 5 --lr 1e-3 --save_tag agcrn_s42 @@ python scripts/agcrn_backbone_gate.py --dataset metrla --tag agcrn_s42 @@ bash scripts/run_metrla.sh --mode train --config_file configs/metrla_agcrn_resdiff.yaml --gpu_id 0 --save_tag s42 @@ bash scripts/run_metrla.sh --mode eval --config_file configs/metrla_agcrn_resdiff.yaml --gpu_id 0 --save_tag s42

# ===== Wave 2 / 5090-runnable experiments (gate-green, highest leverage) =====
# External probabilistic baselines (P2-1): independent, embarrassingly parallel.
baseline_diffstg_pems08 @@ 9 @@ - @@ - @@ python scripts/run_diffstg_canonical.py --config configs/pems08.yaml --canonical_npz outputs/canonical/pems08_canonical.npz --gpu_id 0 --epochs 50 --nsample 20 --out_npz outputs/diffstg/pems08_diffstg.npz @@ python scripts/eval_probabilistic_npz.py --config configs/pems08.yaml --npz outputs/diffstg/pems08_diffstg.npz

# mean-failure warner bare row (C3, the highest-leverage thread): a small model,
# pure 5090 fodder.
warner_pems08 @@ 10 @@ - @@ - @@ - @@ python scripts/eval_frozen_mean_predictor.py --config configs/pems08_agcrn_resdiff.yaml --gpu_id 0

# conformal-on-drop (P1-2) is tiny -- co-locate / CPU is fine, parked on a 5090.
conformal_pems08 @@ 11 @@ - @@ - @@ - @@ python scripts/eval_conformal_residual_baseline.py --config configs/pems08_agcrn_resdiff.yaml --alpha 0.10 --gpu_id 0 --out_json outputs/conformal/pems08_conformal.json
EOF

log() { echo "[chain_dispatcher] $*"; }

card_type_of() {
  # $1 = gpu index ; echoes A100 / 5090 / UNKNOWN
  local idx="$1" pair
  for pair in ${CARD_MAP}; do
    if [[ "${pair%%:*}" == "${idx}" ]]; then
      echo "${pair##*:}"
      return 0
    fi
  done
  echo "UNKNOWN"
}

load_manifest() {
  if [[ -n "${CHAIN_MANIFEST:-}" ]]; then
    if [[ ! -f "${CHAIN_MANIFEST}" ]]; then
      log "ERROR: CHAIN_MANIFEST not found: ${CHAIN_MANIFEST}" >&2
      exit 1
    fi
    cat "${CHAIN_MANIFEST}"
  else
    printf '%s\n' "${DEFAULT_MANIFEST}"
  fi
}

# Parse a manifest line into the globals: C_NAME C_GPU C_BACKBONE C_GATE C_DIFF C_EVAL
parse_line() {
  local line="$1"
  IFS=$'\n' read -r -d '' -a _parts < <(printf '%s' "${line}" | sed "s/${FIELD_SEP}/\n/g" && printf '\0')
  C_NAME="$(echo "${_parts[0]:-}" | xargs)"
  C_GPU="$(echo "${_parts[1]:-}" | xargs)"
  C_BACKBONE="$(echo "${_parts[2]:--}" | sed 's/^ *//; s/ *$//')"
  C_GATE="$(echo "${_parts[3]:--}" | sed 's/^ *//; s/ *$//')"
  C_DIFF="$(echo "${_parts[4]:--}" | sed 's/^ *//; s/ *$//')"
  C_EVAL="$(echo "${_parts[5]:--}" | sed 's/^ *//; s/ *$//')"
}

# Run one stage command with retries. Returns 0 on success.
run_stage() {
  local name="$1" stage="$2" gpu="$3" cmd="$4" logf="$5"
  if [[ -z "${cmd}" || "${cmd}" == "-" ]]; then
    echo "[${name}] stage=${stage} SKIP (no command)" >>"${logf}"
    return 0
  fi

  # Soft-skip a missing gate script instead of hard-failing the chain.
  if [[ "${stage}" == "gate" && "${GATE_SOFT}" == "1" ]]; then
    local script_path
    script_path="$(awk '{print $1}' <<<"${cmd}")"
    if [[ "${script_path}" == python* ]]; then
      script_path="$(awk '{print $2}' <<<"${cmd}")"
    fi
    if [[ "${script_path}" == *.py && ! -f "${script_path}" ]]; then
      echo "[${name}] stage=gate SOFT-SKIP (script missing: ${script_path})" >>"${logf}"
      return 0
    fi
  fi

  local attempt=0 max=$((RETRIES + 1)) rc=0
  while (( attempt < max )); do
    attempt=$((attempt + 1))
    echo "[${name}] stage=${stage} attempt=${attempt}/${max} gpu=${gpu} $(date '+%F %T')" >>"${logf}"
    echo "[${name}]   cmd: ${cmd}" >>"${logf}"
    CUDA_VISIBLE_DEVICES="${gpu}" bash -lc "${cmd}" >>"${logf}" 2>&1
    rc=$?
    if (( rc == 0 )); then
      echo "[${name}] stage=${stage} OK $(date '+%F %T')" >>"${logf}"
      return 0
    fi
    echo "[${name}] stage=${stage} FAILED rc=${rc} $(date '+%F %T')" >>"${logf}"
  done
  return "${rc:-1}"
}

# Run a full chain (all stages) sequentially on its pinned GPU.
run_chain() {
  local name="$1" gpu="$2" backbone="$3" gate="$4" diff="$5" evalc="$6"
  local logf="${LOG_ROOT}/${name}.log"
  local card; card="$(card_type_of "${gpu}")"

  {
    echo "===== chain ${name} ====="
    echo "[start] $(date '+%F %T') gpu=${gpu} card=${card}"
  } >"${logf}"

  # Decision (1): env-gate any chain landing on a 5090.
  if [[ "${card}" == "5090" && "${PREFLIGHT}" == "1" ]]; then
    if ! bash scripts/check_5090_env.sh "${gpu}" >>"${logf}" 2>&1; then
      echo "[${name}] SKIP: 5090 env preflight FAILED on gpu=${gpu} -- build a cu128 env or move this chain to an A100" >>"${logf}"
      log "SKIP ${name}: 5090 env preflight failed on gpu=${gpu} (see ${logf})"
      return 0
    fi
    echo "[${name}] 5090 env preflight PASS on gpu=${gpu}" >>"${logf}"
  fi
  if [[ "${card}" == "UNKNOWN" ]]; then
    echo "[${name}] WARN: gpu=${gpu} not found in CARD_MAP; running without card-type gating" >>"${logf}"
    log "WARN ${name}: gpu=${gpu} not in CARD_MAP (set CARD_MAP for env gating)"
  fi

  local stage
  for stage in backbone gate diffusion eval; do
    local cmd
    case "${stage}" in
      backbone)  cmd="${backbone}" ;;
      gate)      cmd="${gate}" ;;
      diffusion) cmd="${diff}" ;;
      eval)      cmd="${evalc}" ;;
    esac
    if ! run_stage "${name}" "${stage}" "${gpu}" "${cmd}" "${logf}"; then
      echo "[${name}] CHAIN ABORTED at stage=${stage} $(date '+%F %T')" >>"${logf}"
      log "ABORT ${name}: stage ${stage} failed (see ${logf})"
      return 1
    fi
  done

  echo "[done] $(date '+%F %T') chain ${name}" >>"${logf}"
  log "DONE ${name} (gpu=${gpu}, card=${card})"
}

cmd_run() {
  mkdir -p "${LOG_ROOT}"
  ln -sfn "$(basename "${LOG_ROOT}")" "${LATEST_LINK}" 2>/dev/null || true
  log "logs -> ${LOG_ROOT}"
  log "CARD_MAP -> ${CARD_MAP}"

  local want=("$@")        # optional subset of chain names
  local launched=0
  local seen_gpus=""

  while IFS= read -r line; do
    line="${line%%$'\r'}"
    [[ -z "${line//[[:space:]]/}" ]] && continue
    [[ "${line#"${line%%[![:space:]]*}"}" == \#* ]] && continue

    parse_line "${line}"
    [[ -z "${C_NAME}" ]] && continue

    if (( ${#want[@]} > 0 )); then
      local match=0 w
      for w in "${want[@]}"; do [[ "${w}" == "${C_NAME}" ]] && match=1; done
      (( match == 0 )) && continue
    fi

    if ! [[ "${C_GPU}" =~ ^[0-9]+$ ]]; then
      log "ERROR ${C_NAME}: invalid GPU index '${C_GPU}'"; continue
    fi
    # Guard against double-pinning two chains to the same GPU in one wave.
    if [[ " ${seen_gpus} " == *" ${C_GPU} "* ]]; then
      log "WARN ${C_NAME}: gpu=${C_GPU} already used by an earlier chain in this run -- 1 chain == 1 GPU; launching anyway"
    fi
    seen_gpus="${seen_gpus} ${C_GPU}"

    log "launch ${C_NAME} on gpu=${C_GPU} (card=$(card_type_of "${C_GPU}"))"
    run_chain "${C_NAME}" "${C_GPU}" "${C_BACKBONE}" "${C_GATE}" "${C_DIFF}" "${C_EVAL}" &
    echo $! >"${LOG_ROOT}/${C_NAME}.pid"
    launched=$((launched + 1))
  done < <(load_manifest)

  log "launched ${launched} chain(s); waiting..."
  wait
  log "all chains finished -- see ${LOG_ROOT}"
}

cmd_status() {
  local root="${LOG_ROOT}"
  [[ -L "${LATEST_LINK}" && ! -d "${root}" ]] && root="outputs/chain_dispatcher_logs/$(readlink "${LATEST_LINK}")"
  log "logs -> ${root}"
  local p
  for p in "${root}"/*.pid; do
    [[ -e "${p}" ]] || continue
    local name pid
    name="$(basename "${p}" .pid)"; pid="$(cat "${p}")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "RUNNING ${name} pid=${pid}"
    else
      echo "DONE/STOPPED ${name} pid=${pid}"
    fi
  done
}

cmd_tail() {
  local root="${LOG_ROOT}"
  [[ -L "${LATEST_LINK}" && ! -d "${root}" ]] && root="outputs/chain_dispatcher_logs/$(readlink "${LATEST_LINK}")"
  local f
  for f in "${root}"/*.log; do
    [[ -e "${f}" ]] || continue
    echo "===== $(basename "${f}") ====="
    grep -E "\[start\]|\[done\]|stage=|SKIP|PASS|FAIL|ABORT|val_mae=|Test metrics|Traceback|ERROR|Killed|Non-finite" "${f}" | tail -25 || true
  done
}

cmd_list() {
  log "CARD_MAP -> ${CARD_MAP}"
  local line
  while IFS= read -r line; do
    line="${line%%$'\r'}"
    [[ -z "${line//[[:space:]]/}" ]] && continue
    [[ "${line#"${line%%[![:space:]]*}"}" == \#* ]] && continue
    parse_line "${line}"
    [[ -z "${C_NAME}" ]] && continue
    printf '%-24s gpu=%-3s card=%-7s backbone=%s gate=%s diffusion=%s eval=%s\n' \
      "${C_NAME}" "${C_GPU}" "$(card_type_of "${C_GPU}")" \
      "$([[ "${C_BACKBONE}" == "-" ]] && echo skip || echo yes)" \
      "$([[ "${C_GATE}" == "-" ]] && echo skip || echo yes)" \
      "$([[ "${C_DIFF}" == "-" ]] && echo skip || echo yes)" \
      "$([[ "${C_EVAL}" == "-" ]] && echo skip || echo yes)"
  done < <(load_manifest)
}

cmd_inspect() {
  log "decision (2): card <-> index mapping"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L
  else
    log "nvidia-smi not found in PATH"
  fi
}

usage() {
  cat <<EOF
Usage: bash scripts/chain_dispatcher.sh <command> [chain names...]

Commands:
  run [names...]   Launch all chains (or only the named ones), 1 chain per GPU,
                   running backbone -> gate -> diffusion -> eval with retries.
  status           Show RUNNING / DONE per chain for the latest run.
  tail             Tail the key lines of every chain log for the latest run.
  list             List chains in the manifest with their pinned GPU + card type.
  inspect          Run nvidia-smi -L to discover the card<->index mapping.

Key env vars:
  CARD_MAP         "IDX:TYPE ..." (TYPE = A100|5090). Default is a placeholder --
                   set it from \`inspect\` output.
  CHAIN_MANIFEST   Path to a custom manifest (default: built-in manifest).
  RETRIES          Extra attempts per stage on failure (default: ${RETRIES}).
  PREFLIGHT        1 = run check_5090_env.sh before any 5090 chain (default: 1).
  LOG_ROOT         Where logs go (default: timestamped dir under
                   outputs/chain_dispatcher_logs/).
EOF
}

case "${1:-}" in
  run)     shift; cmd_run "$@" ;;
  status)  cmd_status ;;
  tail)    cmd_tail ;;
  list)    cmd_list ;;
  inspect) cmd_inspect ;;
  ""|-h|--help|help) usage ;;
  *) echo "[chain_dispatcher] unknown command: $1" >&2; usage; exit 1 ;;
esac
