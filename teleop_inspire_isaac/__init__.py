"""Teleoperation of an Inspire dexterous hand in Isaac simulation.

This package wires three stages together:

1. ``mocap``    – read hand motion from a Noitom Perception Neuron source
                  (BVH file playback or a live Axis Neuron UDP stream).
2. ``retarget`` – map the captured human-hand joint angles onto the
                  6-DOF Inspire Hand actuator command.
3. ``sim``      – send the actuator command to an Inspire Hand inside an
                  Isaac Gym / Isaac Lab simulation.

The heavy third-party runtimes (Noitom SDK, NVIDIA Isaac) are optional.
The mocap BVH parser and the retargeter depend only on ``numpy`` so they
run and are unit-tested without any hardware or GPU.
"""

from .retarget import InspireHandRetargeter, RetargetConfig
from .mocap import BVHData, BVHFrame, load_bvh

__all__ = [
    "InspireHandRetargeter",
    "RetargetConfig",
    "BVHData",
    "BVHFrame",
    "load_bvh",
]

__version__ = "0.1.0"
