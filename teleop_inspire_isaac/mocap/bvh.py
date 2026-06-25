"""Minimal, dependency-light BVH parser.

BVH is the format exported by Noitom Axis Studio / Axis Neuron (and most
other motion-capture tools). This parser supports the standard
``HIERARCHY`` / ``MOTION`` layout and exposes per-joint channel values
for every frame.

Only the standard library and ``numpy`` are used so the parser runs
without any motion-capture hardware and is unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Rotation channels in the order they may appear in a BVH ``CHANNELS`` line.
ROTATION_CHANNELS = ("Xrotation", "Yrotation", "Zrotation")
POSITION_CHANNELS = ("Xposition", "Yposition", "Zposition")


@dataclass
class Joint:
    """A node in the BVH skeleton hierarchy."""

    name: str
    offset: np.ndarray
    channels: List[str] = field(default_factory=list)
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    is_end_site: bool = False


@dataclass
class BVHFrame:
    """A single frame of motion as ``joint -> {channel: value}``.

    Rotation values are stored in degrees, exactly as found in the BVH
    file (the on-disk convention). Use :meth:`euler_rad` for radians.
    """

    values: Dict[str, Dict[str, float]]

    def joint(self, name: str) -> Dict[str, float]:
        return self.values.get(name, {})

    def euler_rad(self, name: str) -> np.ndarray:
        """Return ``[Xrot, Yrot, Zrot]`` for a joint in radians.

        Missing channels default to ``0.0``.
        """
        ch = self.values.get(name, {})
        return np.radians(
            np.array([ch.get(c, 0.0) for c in ROTATION_CHANNELS], dtype=np.float64)
        )


@dataclass
class BVHData:
    """Parsed BVH document: skeleton plus a list of motion frames."""

    joints: Dict[str, Joint]
    root: str
    frame_time: float
    frames: List[BVHFrame]

    @property
    def num_frames(self) -> int:
        return len(self.frames)

    @property
    def fps(self) -> float:
        return 1.0 / self.frame_time if self.frame_time > 0 else 0.0

    def joint_names(self) -> List[str]:
        return [n for n, j in self.joints.items() if not j.is_end_site]


def _channels_per_joint(joints: Dict[str, Joint], order: List[str]) -> List[tuple]:
    """Flatten ``(joint_name, channel)`` pairs in motion-column order."""
    flat: List[tuple] = []
    for name in order:
        for ch in joints[name].channels:
            flat.append((name, ch))
    return flat


def parse_bvh(text: str) -> BVHData:
    """Parse a BVH document from a string."""
    tokens = text.split()
    idx = 0

    joints: Dict[str, Joint] = {}
    # Order in which joints (and their channels) are declared; this is the
    # column order used by the MOTION section.
    declared_order: List[str] = []
    root_name: Optional[str] = None
    stack: List[str] = []
    end_site_counter = 0

    def expect(tok: str) -> None:
        nonlocal idx
        if tokens[idx] != tok:
            raise ValueError(f"Expected '{tok}' but found '{tokens[idx]}'")
        idx += 1

    if tokens[idx] != "HIERARCHY":
        raise ValueError("BVH must start with HIERARCHY")
    idx += 1

    while tokens[idx] != "MOTION":
        tok = tokens[idx]
        if tok in ("ROOT", "JOINT"):
            idx += 1
            name = tokens[idx]
            idx += 1
            parent = stack[-1] if stack else None
            joint = Joint(name=name, offset=np.zeros(3), parent=parent)
            joints[name] = joint
            declared_order.append(name)
            if parent is not None:
                joints[parent].children.append(name)
            if tok == "ROOT":
                root_name = name
            stack.append(name)
            expect("{")
        elif tok == "End":
            idx += 1
            expect("Site")
            parent = stack[-1]
            name = f"{parent}_EndSite_{end_site_counter}"
            end_site_counter += 1
            joint = Joint(name=name, offset=np.zeros(3), parent=parent,
                          is_end_site=True)
            joints[name] = joint
            joints[parent].children.append(name)
            stack.append(name)
            expect("{")
        elif tok == "OFFSET":
            idx += 1
            off = np.array(tokens[idx:idx + 3], dtype=np.float64)
            idx += 3
            joints[stack[-1]].offset = off
        elif tok == "CHANNELS":
            idx += 1
            n = int(tokens[idx])
            idx += 1
            chans = tokens[idx:idx + n]
            idx += n
            joints[stack[-1]].channels = list(chans)
        elif tok == "}":
            idx += 1
            stack.pop()
        else:
            raise ValueError(f"Unexpected token in HIERARCHY: '{tok}'")

    if root_name is None:
        raise ValueError("No ROOT joint found")

    # MOTION
    expect("MOTION")
    expect("Frames:")
    num_frames = int(tokens[idx]); idx += 1
    expect("Frame")
    expect("Time:")
    frame_time = float(tokens[idx]); idx += 1

    columns = _channels_per_joint(joints, declared_order)
    n_cols = len(columns)

    rest = tokens[idx:]
    if len(rest) < num_frames * n_cols:
        raise ValueError(
            f"MOTION data too short: expected {num_frames * n_cols} values, "
            f"found {len(rest)}"
        )
    data = np.array(rest[: num_frames * n_cols], dtype=np.float64)
    data = data.reshape(num_frames, n_cols)

    frames: List[BVHFrame] = []
    for r in range(num_frames):
        values: Dict[str, Dict[str, float]] = {}
        for c, (jname, chan) in enumerate(columns):
            values.setdefault(jname, {})[chan] = float(data[r, c])
        frames.append(BVHFrame(values=values))

    return BVHData(joints=joints, root=root_name, frame_time=frame_time,
                   frames=frames)


def load_bvh(path: str) -> BVHData:
    """Load and parse a BVH file from disk."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return parse_bvh(f.read())
