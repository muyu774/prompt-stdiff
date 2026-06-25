"""Noitom Perception Neuron mocap sources.

Two concrete sources are provided behind a common :class:`MocapSource`
interface:

* :class:`BVHFileSource` – replays a recorded ``.bvh`` file (works
  offline, used by the tests).
* :class:`AxisNeuronUDPSource` – receives the live "BVH" data stream that
  Axis Studio / Axis Neuron broadcasts over UDP.

Both yield :class:`~teleop_inspire_isaac.mocap.bvh.BVHFrame` objects so the
downstream retargeter does not care where the motion came from.
"""

from __future__ import annotations

import socket
import time
from typing import Iterator, List, Optional

from .bvh import BVHData, BVHFrame, load_bvh


class MocapSource:
    """Abstract hand-motion source."""

    def joint_names(self) -> List[str]:
        raise NotImplementedError

    def frames(self) -> Iterator[BVHFrame]:
        raise NotImplementedError

    def close(self) -> None:
        pass

    def __enter__(self) -> "MocapSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class BVHFileSource(MocapSource):
    """Replay a recorded BVH file at (optionally) real-time speed."""

    def __init__(self, path: str, realtime: bool = False, loop: bool = False):
        self.data: BVHData = load_bvh(path)
        self.realtime = realtime
        self.loop = loop

    def joint_names(self) -> List[str]:
        return self.data.joint_names()

    def frames(self) -> Iterator[BVHFrame]:
        dt = self.data.frame_time
        while True:
            for frame in self.data.frames:
                if self.realtime and dt > 0:
                    time.sleep(dt)
                yield frame
            if not self.loop:
                break


class AxisNeuronUDPSource(MocapSource):
    """Receive the Axis Neuron / Axis Studio BVH data stream over UDP.

    Axis software can broadcast skeleton data as whitespace-separated
    ASCII ("BVH" output, *non* binary). Enable in Axis:
    ``Settings -> Output -> BVH -> UDP`` and pick the *string* format.

    The packet layout is one record per frame::

        <avatar_index> v1 v2 v3 ... vN

    where the values are the per-joint channels in the same order as the
    skeleton's reference BVH. Because the live stream omits the
    ``HIERARCHY`` block, a reference BVH file (``ref_bvh``) is required to
    recover joint/channel names.

    The binary BVH stream is intentionally not parsed here; configure Axis
    to emit the ASCII string format instead.
    """

    def __init__(
        self,
        ref_bvh: str,
        host: str = "0.0.0.0",
        port: int = 7002,
        timeout: Optional[float] = 5.0,
        with_displacement: bool = True,
    ):
        self.reference: BVHData = load_bvh(ref_bvh)
        self.host = host
        self.port = port
        self.timeout = timeout
        self.with_displacement = with_displacement
        # Column order recovered from the reference skeleton.
        self._columns = self._reference_columns()
        self._sock: Optional[socket.socket] = None

    def _reference_columns(self) -> List[tuple]:
        cols: List[tuple] = []
        for name in self.reference.joint_names():
            joint = self.reference.joints[name]
            channels = joint.channels
            if not self.with_displacement:
                channels = [c for c in channels if c.endswith("rotation")]
            for ch in channels:
                cols.append((name, ch))
        return cols

    def joint_names(self) -> List[str]:
        return self.reference.joint_names()

    def _ensure_socket(self) -> socket.socket:
        if self._sock is None:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port))
            if self.timeout is not None:
                sock.settimeout(self.timeout)
            self._sock = sock
        return self._sock

    def parse_packet(self, payload: str) -> Optional[BVHFrame]:
        """Turn one ASCII UDP payload into a :class:`BVHFrame`."""
        parts = payload.split()
        if not parts:
            return None
        # Drop a leading non-numeric avatar id / token if present.
        try:
            float(parts[0])
            numbers = parts
        except ValueError:
            numbers = parts[1:]
        if len(numbers) < len(self._columns):
            return None
        values: dict = {}
        for (jname, chan), raw in zip(self._columns, numbers):
            try:
                values.setdefault(jname, {})[chan] = float(raw)
            except ValueError:
                return None
        return BVHFrame(values=values)

    def frames(self) -> Iterator[BVHFrame]:
        sock = self._ensure_socket()
        while True:
            data, _addr = sock.recvfrom(65535)
            frame = self.parse_packet(data.decode("ascii", errors="ignore"))
            if frame is not None:
                yield frame

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
