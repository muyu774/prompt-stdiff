#!/usr/bin/env python3
"""One-command pipeline: run -> integrity -> analyze -> bootstrap -> assets."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def step(args_list, title):
    print(f"\n===== {title} =====", flush=True)
    rc = subprocess.run([sys.executable] + args_list, cwd=str(HERE.parent)).returncode
    print(f"[{title}] rc={rc}")
    return rc


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpus", default=None)
    p.add_argument("--queue", default=str(HERE / "queue.yaml"))
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--propose", action="store_true")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    if not args.skip_train:
        cmd = ["auto_research/orchestrator.py", "--queue", args.queue]
        if args.gpus:
            cmd += ["--gpus", args.gpus]
        step(cmd, "A. Run experiments")

    step(["auto_research/integrity_check.py"], "P0. Integrity check")
    step(["auto_research/analyze_results.py", "--use-clean"], "B. Analyze (validation-selected)")
    step(["auto_research/bootstrap_stats.py", "--metric", "crps"], "P0/P1. Paired bootstrap")

    if args.propose:
        step(["auto_research/propose_next.py",
              "--seed-config", "configs/pems08_pdformer_resdiff.yaml"], "C. Propose next round")

    assets = ["auto_research/make_paper_assets.py"]
    if args.no_plots:
        assets.append("--no-plots")
    step(assets, "D. Paper assets")

    print("\nAll done. See outputs/auto_research/ , outputs/paper_assets/ , "
          "outputs/revision_8gpu_logs/incident/")


if __name__ == "__main__":
    main()
