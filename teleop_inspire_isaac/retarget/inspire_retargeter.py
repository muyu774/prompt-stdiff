"""Retarget a human hand pose (mocap) onto the 6-DOF Inspire Hand.

The Inspire Hand (RH56 series) exposes **six** actuators, controlled in
this canonical order::

    0  little  (pinky)  flexion
    1  ring             flexion
    2  middle           flexion
    3  index            flexion
    4  thumb            flexion (bend)
    5  thumb            rotation (opposition)

Each actuator accepts an integer command in ``[output_min, output_max]``
(the Inspire SDK uses ``0..1000``). By the Inspire convention a *larger*
command means a *more open* finger, so by default human flexion is
inverted before scaling.

This module only depends on ``numpy`` and the parsed mocap frame, so it
runs and is unit-tested without any hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from ..mocap.bvh import BVHFrame

# Canonical Inspire actuator order.
INSPIRE_ACTUATORS = ("little", "ring", "middle", "index", "thumb_bend", "thumb_rot")


def _default_finger_joints(prefix: str) -> Dict[str, List[str]]:
    """Perception Neuron finger-joint names for a given hand prefix."""
    return {
        "thumb": [f"{prefix}Thumb1", f"{prefix}Thumb2", f"{prefix}Thumb3"],
        "index": [f"{prefix}Index1", f"{prefix}Index2", f"{prefix}Index3"],
        "middle": [f"{prefix}Middle1", f"{prefix}Middle2", f"{prefix}Middle3"],
        "ring": [f"{prefix}Ring1", f"{prefix}Ring2", f"{prefix}Ring3"],
        "little": [f"{prefix}Pinky1", f"{prefix}Pinky2", f"{prefix}Pinky3"],
    }


@dataclass
class RetargetConfig:
    """Configuration for :class:`InspireHandRetargeter`.

    Angle ranges are expressed in **degrees** for readability and converted
    internally. ``flexion_axis`` / ``thumb_rot_axis`` select the Euler
    component (0=X, 1=Y, 2=Z) that carries the relevant motion; this depends
    on the skeleton's joint coordinate frames and may need calibration.
    """

    # Joint name prefix, e.g. "RightHand" or "LeftHand".
    hand_prefix: str = "RightHand"
    # Per-finger joint name lists; defaults derived from ``hand_prefix``.
    finger_joints: Dict[str, List[str]] = field(default_factory=dict)

    # Which Euler axis encodes flexion, and its open/closed angle range.
    flexion_axis: int = 2  # Z by default
    flexion_min_deg: float = 0.0    # fully open  -> command output_max
    flexion_max_deg: float = 90.0   # fully closed -> command output_min

    # Thumb opposition / rotation (CMC joint, usually Thumb1).
    thumb_rot_axis: int = 1  # Y by default
    thumb_rot_min_deg: float = -10.0
    thumb_rot_max_deg: float = 60.0

    # Output command range (Inspire SDK uses 0..1000).
    output_min: int = 0
    output_max: int = 1000

    # Invert so that human flexion -> smaller (more closed) command,
    # matching the Inspire convention where larger == more open.
    invert_flexion: bool = True
    invert_thumb_rot: bool = False

    def resolved_finger_joints(self) -> Dict[str, List[str]]:
        if self.finger_joints:
            return self.finger_joints
        return _default_finger_joints(self.hand_prefix)


def _normalize(value: float, lo: float, hi: float) -> float:
    """Map ``value`` from ``[lo, hi]`` to ``[0, 1]``, clamped."""
    if hi == lo:
        return 0.0
    t = (value - lo) / (hi - lo)
    return float(min(1.0, max(0.0, t)))


class InspireHandRetargeter:
    """Convert mocap hand frames into Inspire Hand actuator commands."""

    def __init__(self, config: RetargetConfig | None = None):
        self.config = config or RetargetConfig()
        self._finger_joints = self.config.resolved_finger_joints()

    # -- per-finger primitives -------------------------------------------------

    def _finger_flexion_deg(self, frame: BVHFrame, finger: str) -> float:
        """Sum the flexion-axis rotation (deg) across a finger's joints."""
        axis = self.config.flexion_axis
        total = 0.0
        for joint in self._finger_joints.get(finger, []):
            euler_deg = np.degrees(frame.euler_rad(joint))
            total += float(euler_deg[axis])
        return total

    def _normalized_flexion(self, frame: BVHFrame, finger: str) -> float:
        """Return flexion in ``[0, 1]`` (0 open, 1 fully closed)."""
        cfg = self.config
        # Range is per-joint; scale by the number of joints summed.
        n = max(1, len(self._finger_joints.get(finger, [])))
        deg = self._finger_flexion_deg(frame, finger)
        return _normalize(deg, cfg.flexion_min_deg * n, cfg.flexion_max_deg * n)

    def _normalized_thumb_rot(self, frame: BVHFrame) -> float:
        cfg = self.config
        joints = self._finger_joints.get("thumb", [])
        if not joints:
            return 0.0
        euler_deg = np.degrees(frame.euler_rad(joints[0]))
        deg = float(euler_deg[cfg.thumb_rot_axis])
        return _normalize(deg, cfg.thumb_rot_min_deg, cfg.thumb_rot_max_deg)

    # -- public API ------------------------------------------------------------

    def normalized(self, frame: BVHFrame) -> np.ndarray:
        """Return the 6 actuator targets in ``[0, 1]`` (Inspire order).

        ``1`` always means *fully open*, ``0`` *fully closed*, after any
        configured inversion. This is the hardware-agnostic representation.
        """
        cfg = self.config
        out = np.zeros(len(INSPIRE_ACTUATORS), dtype=np.float64)
        for i, name in enumerate(INSPIRE_ACTUATORS):
            if name == "thumb_rot":
                t = self._normalized_thumb_rot(frame)
                if cfg.invert_thumb_rot:
                    t = 1.0 - t
            else:
                finger = "thumb" if name == "thumb_bend" else name
                t = self._normalized_flexion(frame, finger)
                if cfg.invert_flexion:
                    t = 1.0 - t
            out[i] = t
        return out

    def command(self, frame: BVHFrame) -> np.ndarray:
        """Return integer actuator commands in ``[output_min, output_max]``."""
        cfg = self.config
        norm = self.normalized(frame)
        scaled = cfg.output_min + norm * (cfg.output_max - cfg.output_min)
        return np.rint(scaled).astype(np.int64)

    def retarget_sequence(self, frames: Sequence[BVHFrame]) -> np.ndarray:
        """Vectorised retargeting of a sequence: returns ``[T, 6]`` commands."""
        return np.stack([self.command(f) for f in frames], axis=0)
