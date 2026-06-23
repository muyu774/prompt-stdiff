#!/usr/bin/env python3
"""Generate next-round sweep configs from leaderboard winners.

Overrides a small set of high-impact knobs and writes child configs to
configs/auto_sweep/ plus a new queue auto_research/queue_next.yaml.
Nothing is executed here.
"""
from __future__ import annotations
import argparse, itertools
from pathlib import Path
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SWEEP_DIR = ROOT / "configs" / "auto_sweep"
OUT_QUEUE = Path(__file__).parent / "queue_next.yaml"

GRID = {
    "train.lr": [5.0e-4, 1.0e-3, 2.0e-3],
    "model.residual_sample_scale": [1.0, 2.0, 3.0],
    "model.gamma": [0.0, 0.1],
}


def _set(d, dotted, v):
    ks = dotted.split(".")
    cur = d
    for k in ks[:-1]:
        cur = cur.setdefault(k, {})
    cur[ks[-1]] = v


def seeds_from_leaderboard():
    lb = ROOT / "outputs" / "auto_research" / "leaderboard.csv"
    if lb.exists():
        df = pd.read_csv(lb)
        if "config" in df.columns and not df.empty:
            return sorted(set(df["config"].dropna().tolist()))
    return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed-config", action="append", default=[])
    p.add_argument("--max-per-base", type=int, default=8)
    args = p.parse_args()
    bases = args.seed_config or seeds_from_leaderboard()
    if not bases:
        print("[WARN] No seed configs. Pass --seed-config configs/pems08_pdformer_resdiff.yaml")
        return
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    jobs = []
    for base in bases:
        bn = Path(base).stem
        cnt = 0
        for combo in combos:
            if cnt >= args.max_per_base:
                break
            child = {}
            # defaults path is relative to the child file's parent (configs/auto_sweep/)
            rel = Path("..") / Path(base).name if str(base).startswith("configs/") \
                else Path(base)
            child["defaults"] = [str(rel)]
            tag = []
            for k, v in zip(keys, combo):
                _set(child, k, v)
                tag.append(f"{k.split('.')[-1]}{v}")
            tagstr = "_".join(tag).replace(".", "p")
            _set(child, "train.save_dir", f"./outputs/checkpoints/auto_sweep/{bn}/{tagstr}")
            cp = SWEEP_DIR / f"{bn}__{tagstr}.yaml"
            with cp.open("w", encoding="utf-8") as f:
                yaml.safe_dump(child, f, sort_keys=False, allow_unicode=True)
            jobs.append({"name": f"{bn}__{tagstr}", "config": str(cp.relative_to(ROOT)),
                         "setting": "sweep", "tags": ["auto_sweep", bn]})
            cnt += 1
    spec = {"defaults": {"device": "auto", "num_eval_samples": 100, "sampler": "ddpm",
                         "method": "Prompt-STDiff", "implementation": "ours"},
            "gpus": [], "seeds": [2021], "jobs": jobs}
    with OUT_QUEUE.open("w", encoding="utf-8") as f:
        yaml.safe_dump(spec, f, sort_keys=False, allow_unicode=True)
    print(f"[OK] {len(jobs)} sweep configs in {SWEEP_DIR}")
    print(f"[OK] queue: {OUT_QUEUE}")
    print("Verify defaults parsing:")
    print("  python -c \"from utils.config import load_config; import glob; "
          "[load_config(p) for p in glob.glob('configs/auto_sweep/*.yaml')]; print('all OK')\"")


if __name__ == "__main__":
    main()
