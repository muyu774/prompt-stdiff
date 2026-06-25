"""End-to-end teleoperation pipeline.

Streams frames from a Perception Neuron mocap source, retargets each frame
onto Inspire Hand actuator commands, and applies them to an Isaac (or dummy)
simulation backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .mocap.perception_neuron import MocapSource
from .retarget.inspire_retargeter import InspireHandRetargeter
from .sim.isaac_inspire_env import InspireHandSim


@dataclass
class TeleopStats:
    frames: int = 0
    last_command: Optional[np.ndarray] = None


class TeleopPipeline:
    """Connect a :class:`MocapSource`, a retargeter and a sim backend."""

    def __init__(
        self,
        source: MocapSource,
        retargeter: InspireHandRetargeter,
        sim: InspireHandSim,
        smoothing: float = 0.0,
    ):
        """
        Parameters
        ----------
        smoothing:
            Exponential-moving-average factor in ``[0, 1)`` applied to the
            normalized targets to reduce jitter. ``0`` disables smoothing.
        """
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.source = source
        self.retargeter = retargeter
        self.sim = sim
        self.smoothing = smoothing
        self._ema: Optional[np.ndarray] = None

    def _smooth(self, target: np.ndarray) -> np.ndarray:
        if self.smoothing <= 0.0:
            return target
        if self._ema is None:
            self._ema = target.copy()
        else:
            a = self.smoothing
            self._ema = a * self._ema + (1.0 - a) * target
        return self._ema

    def run(self, max_frames: Optional[int] = None) -> TeleopStats:
        """Run the teleop loop until the source ends or ``max_frames``."""
        stats = TeleopStats()
        self.sim.reset()
        for frame in self.source.frames():
            target = self.retargeter.normalized(frame)
            target = self._smooth(target)
            self.sim.set_actuator_targets(target)
            self.sim.step()
            stats.frames += 1
            stats.last_command = target
            if max_frames is not None and stats.frames >= max_frames:
                break
        return stats
