#!/usr/bin/env python3
"""Auto-research orchestrator: schedule training + recording across GPUs.

Reuses repo entrypoints:
- train.py
- scripts/run_experiment_and_record.py
"""
from __future__ import annotations
import argparse, json, os, queue, re, subprocess, sys, threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs" / "auto_research" / "logs"
LEDGER = ROOT / "outputs" / "auto_research" / "ledger.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def detect_gpus() -> List[int]:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                             capture_output=True, text=True, check=True)
        return [int(x) for x in out.stdout.split() if x.strip().isdigit()]
    except Exception:
        return []


@dataclass
class Job:
    name: str
    config: str
    setting: str = "eval"
    method: str = "Prompt-STDiff"
    implementation: str = "ours"
    device: Optional[str] = None
    num_eval_samples: int = 100
    sampler: str = "ddpm"
    seed: Optional[int] = None
    tags: List[str] = field(default_factory=list)


def _resolve_save_dir(config_path: str) -> Optional[Path]:
    try:
        sys.path.insert(0, str(ROOT))
        from utils.config import load_config
        cfg = load_config(config_path)
        sd = cfg.get("train", {}).get("save_dir")
        return Path(sd) if sd else None
    except Exception:
        return None


def _checkpoint_for(job: Job) -> Optional[Path]:
    sd = _resolve_save_dir(job.config)
    if sd is None:
        return None
    base = sd if sd.is_absolute() else (ROOT / sd)
    if job.seed is not None:
        return base / f"seed{job.seed}" / "best.pt"
    return base / "best.pt"


def _append_ledger(rec: Dict[str, Any]) -> None:
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def _run(cmd: List[str], log_path: Path, env: Dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"# CMD: {' '.join(cmd)}\n# START: {_now()}\n\n")
        log.flush()
        return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                env=env, cwd=str(ROOT)).wait()


def run_job(job: Job, gpu: Optional[int], force: bool, train_only: bool) -> Dict[str, Any]:
    env = dict(os.environ)
    device = job.device or "auto"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        device = "cuda:0"
    ckpt = _checkpoint_for(job)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"{job.name}" + (f"_seed{job.seed}" if job.seed is not None else "")
    base_log = LOG_DIR / f"{tag}_{ts}"
    res: Dict[str, Any] = {"name": job.name, "config": job.config, "setting": job.setting,
                           "seed": job.seed, "gpu": gpu, "device": device,
                           "start": _now(), "tags": job.tags}

    need_train = force or (ckpt is None) or (not ckpt.exists())
    if need_train:
        train_cmd = [sys.executable, "train.py", "--config", job.config, "--device", device]
        # NOTE: if train.py reads seed only from config, generate per-seed configs.
        rc = _run(train_cmd, base_log.with_suffix(".train.log"), env)
        res["train_rc"] = rc
        res["train_log"] = str(base_log.with_suffix(".train.log"))
        if rc != 0:
            res["status"] = "train_failed"; res["end"] = _now(); _append_ledger(res); return res
    else:
        res["train_rc"] = "skipped(existing_ckpt)"

    if train_only:
        res["status"] = "trained_only"; res["end"] = _now(); _append_ledger(res); return res

    if ckpt is None or not ckpt.exists():
        res["status"] = "no_checkpoint_for_eval"; res["end"] = _now(); _append_ledger(res); return res

    rec_cmd = [sys.executable, "scripts/run_experiment_and_record.py",
               "--config", job.config, "--ckpt", str(ckpt), "--device", device,
               "--num_eval_samples", str(job.num_eval_samples), "--sampler", job.sampler,
               "--method", job.method, "--setting", job.setting,
               "--implementation", job.implementation]
    rc = _run(rec_cmd, base_log.with_suffix(".eval.log"), env)
    res["eval_rc"] = rc
    res["eval_log"] = str(base_log.with_suffix(".eval.log"))
    res["checkpoint"] = str(ckpt)
    res["status"] = "ok" if rc == 0 else "eval_failed"
    res["end"] = _now(); _append_ledger(res); return res


def worker(gpu, jobq, args, results, lock):
    while True:
        try:
            job = jobq.get_nowait()
        except queue.Empty:
            return
        tagn = f"[gpu{gpu}]" if gpu is not None else "[cpu]"
        print(f"{tagn} START {job.name} seed={job.seed}", flush=True)
        r = run_job(job, gpu=gpu, force=args.force, train_only=args.train_only)
        with lock:
            results.append(r)
        print(f"{tagn} DONE  {job.name} seed={job.seed} -> {r.get('status')}", flush=True)
        jobq.task_done()


def build_jobs(spec, only_re):
    defaults = spec.get("defaults", {})
    seeds = spec.get("seeds", [None])
    jobs = []
    for j in spec.get("jobs", []):
        if only_re and not only_re.search(j["name"]):
            continue
        for seed in (seeds or [None]):
            jobs.append(Job(
                name=j["name"], config=j["config"],
                setting=j.get("setting", defaults.get("setting", "eval")),
                method=j.get("method", defaults.get("method", "Prompt-STDiff")),
                implementation=j.get("implementation", defaults.get("implementation", "ours")),
                device=j.get("device", defaults.get("device")),
                num_eval_samples=int(j.get("num_eval_samples", defaults.get("num_eval_samples", 100))),
                sampler=j.get("sampler", defaults.get("sampler", "ddpm")),
                seed=seed, tags=j.get("tags", []),
            ))
    return jobs


def parse_args():
    p = argparse.ArgumentParser(description="Auto-research orchestrator")
    p.add_argument("--queue", default=str(Path(__file__).parent / "queue.yaml"))
    p.add_argument("--gpus", default=None)
    p.add_argument("--only", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    spec = _load_yaml(Path(args.queue))
    only_re = re.compile(args.only) if args.only else None
    jobs = build_jobs(spec, only_re)

    if args.gpus is not None:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    elif spec.get("gpus"):
        gpus = [int(x) for x in spec["gpus"]]
    else:
        gpus = detect_gpus()

    print(f"Discovered {len(jobs)} job(s). GPUs={gpus or 'CPU-only'}")
    for j in jobs:
        ck = _checkpoint_for(j)
        st = "exists" if (ck and ck.exists()) else "missing"
        print(f"  - {j.name:38s} seed={str(j.seed):5s} ckpt={st}")
    if args.dry_run:
        return

    jobq = queue.Queue()
    for j in jobs:
        jobq.put(j)
    results, lock, threads = [], threading.Lock(), []
    pool = gpus if gpus else [None]
    for g in pool:
        t = threading.Thread(target=worker, args=(g, jobq, args, results, lock), daemon=True)
        t.start(); threads.append(t)
    for t in threads:
        t.join()
    ok = sum(1 for r in results if r.get("status") == "ok")
    print(f"\n=== Orchestrator finished: {ok}/{len(results)} ok ===\nLedger: {LEDGER}")


if __name__ == "__main__":
    main()
