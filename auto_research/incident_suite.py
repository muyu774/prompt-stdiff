#!/usr/bin/env python3
"""Incident/drop suite: event-subset metrics + mean-vs-dispersion root-cause.

Wraps existing repo scripts:
- scripts/eval_event_subset.py        (drop / spike subset prob metrics)
- scripts/eval_event_root_cause.py    (rho = |y-mu| / halfwidth)

Reads paths from auto_research/queue.yaml (events.drop_csv / mixed_csv /
rootcause_out_dir). Runs drop AND spike for a given checkpoint+config.
"""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _load_queue(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _run(cmd):
    print("[run]", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=str(ROOT)).returncode


def main():
    p = argparse.ArgumentParser(description="Incident/drop suite")
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--method", default="Prompt-STDiff")
    p.add_argument("--queue", default=str(ROOT / "auto_research" / "queue.yaml"))
    p.add_argument("--gpu_id", type=int, default=None)
    p.add_argument("--num_eval_samples", type=int, default=100)
    p.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddpm")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    spec = _load_queue(Path(args.queue))
    ev = spec.get("events", {})
    drop_csv = ROOT / ev.get("drop_csv", "outputs/pems08_extreme_drop_events.csv")
    mixed_csv = ROOT / ev.get("mixed_csv", "outputs/pems08_extreme_events.csv")
    out_dir = ROOT / ev.get("rootcause_out_dir", "outputs/revision_8gpu_logs/incident")
    out_dir.mkdir(parents=True, exist_ok=True)

    dev = ["--gpu_id", str(args.gpu_id)] if args.gpu_id is not None else ["--device", args.device]
    common = ["--config", args.config, "--ckpt", args.ckpt, "--method", args.method,
              "--num_eval_samples", str(args.num_eval_samples), "--sampler", args.sampler] + dev

    # ---- 1. DROP subset metrics (pure-drop CSV) ----
    _run([sys.executable, "scripts/eval_event_subset.py",
          "--events_csv", drop_csv, "--kind", "all", "--setting", "drop_subset",
          "--out_json", out_dir / "drop_subset_metrics.json", "--dedupe_positions"] + common)

    # ---- 2. SPIKE subset metrics (mixed CSV, filter kind=spike) ----
    _run([sys.executable, "scripts/eval_event_subset.py",
          "--events_csv", mixed_csv, "--kind", "spike", "--setting", "spike_subset",
          "--out_json", out_dir / "spike_subset_metrics.json", "--dedupe_positions"] + common)

    # ---- 3. DROP root-cause (rho) ----
    _run([sys.executable, "scripts/eval_event_root_cause.py",
          "--events_csv", drop_csv, "--kind", "all", "--setting", "drop_rootcause",
          "--out_json", out_dir / "drop_rootcause.json", "--dedupe_positions"] + common)

    # ---- 4. SPIKE root-cause (rho) ----
    _run([sys.executable, "scripts/eval_event_root_cause.py",
          "--events_csv", mixed_csv, "--kind", "spike", "--setting", "spike_rootcause",
          "--out_json", out_dir / "spike_rootcause.json", "--dedupe_positions"] + common)

    print(f"\n[OK] incident suite outputs in {out_dir}")
    for f in ["drop_subset_metrics.json", "spike_subset_metrics.json",
              "drop_rootcause.json", "spike_rootcause.json"]:
        print("  -", out_dir / f)


if __name__ == "__main__":
    main()
