"""Run the Perception Neuron -> Inspire Hand -> Isaac teleop pipeline.

Examples
--------
Offline dry-run on a recorded BVH using the dummy (no-GPU) backend::

    python -m teleop_inspire_isaac.scripts.run_teleop \
        --config teleop_inspire_isaac/config/default.yaml --max-frames 100

Live teleop from Axis Neuron into Isaac Gym::

    python -m teleop_inspire_isaac.scripts.run_teleop \
        --config my_isaac_config.yaml
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import yaml

from ..mocap.perception_neuron import AxisNeuronUDPSource, BVHFileSource
from ..pipeline import TeleopPipeline
from ..retarget.inspire_retargeter import InspireHandRetargeter, RetargetConfig
from ..sim.isaac_inspire_env import DummyInspireHand, IsaacInspireHand


def _resolve(base_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(base_dir, path)


def build_source(cfg: Dict[str, Any], base_dir: str):
    mocap = cfg.get("mocap", {})
    source = mocap.get("source", "bvh")
    if source == "bvh":
        b = mocap.get("bvh", {})
        return BVHFileSource(
            path=_resolve(base_dir, b["path"]),
            realtime=b.get("realtime", False),
            loop=b.get("loop", False),
        )
    if source == "udp":
        u = mocap.get("udp", {})
        return AxisNeuronUDPSource(
            ref_bvh=_resolve(base_dir, u["ref_bvh"]),
            host=u.get("host", "0.0.0.0"),
            port=u.get("port", 7002),
            timeout=u.get("timeout", 5.0),
            with_displacement=u.get("with_displacement", True),
        )
    raise ValueError(f"Unknown mocap source: {source}")


def build_retargeter(cfg: Dict[str, Any]) -> InspireHandRetargeter:
    r = cfg.get("retarget", {})
    rc = RetargetConfig(
        hand_prefix=r.get("hand_prefix", "RightHand"),
        flexion_axis=r.get("flexion_axis", 2),
        flexion_min_deg=r.get("flexion_min_deg", 0.0),
        flexion_max_deg=r.get("flexion_max_deg", 90.0),
        thumb_rot_axis=r.get("thumb_rot_axis", 1),
        thumb_rot_min_deg=r.get("thumb_rot_min_deg", -10.0),
        thumb_rot_max_deg=r.get("thumb_rot_max_deg", 60.0),
        output_min=r.get("output_min", 0),
        output_max=r.get("output_max", 1000),
        invert_flexion=r.get("invert_flexion", True),
        invert_thumb_rot=r.get("invert_thumb_rot", False),
    )
    return InspireHandRetargeter(rc)


def build_sim(cfg: Dict[str, Any], base_dir: str):
    s = cfg.get("sim", {})
    backend = s.get("backend", "dummy")
    if backend == "dummy":
        return DummyInspireHand(verbose=True)
    if backend == "isaac":
        i = s.get("isaac", {})
        return IsaacInspireHand(
            asset_root=_resolve(base_dir, i.get("asset_root", "assets")),
            asset_file=i.get("asset_file", "inspire_hand.urdf"),
            device=i.get("device", "cuda:0"),
            headless=i.get("headless", False),
            dt=i.get("dt", 1.0 / 60.0),
        )
    raise ValueError(f"Unknown sim backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after this many frames (default: run all).")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    base_dir = os.path.dirname(os.path.abspath(args.config))
    base_dir = os.path.dirname(base_dir)  # config/ -> package root

    source = build_source(cfg, base_dir)
    retargeter = build_retargeter(cfg)
    sim = build_sim(cfg, base_dir)
    smoothing = cfg.get("sim", {}).get("smoothing", 0.0)

    pipeline = TeleopPipeline(source, retargeter, sim, smoothing=smoothing)
    try:
        stats = pipeline.run(max_frames=args.max_frames)
    finally:
        source.close()
        sim.close()

    print(f"Done. Processed {stats.frames} frames.")
    if stats.last_command is not None:
        print(f"Last normalized command: {stats.last_command}")


if __name__ == "__main__":
    main()
