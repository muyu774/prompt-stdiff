import os

import numpy as np

from teleop_inspire_isaac.mocap import BVHFileSource, load_bvh
from teleop_inspire_isaac.mocap.bvh import BVHFrame
from teleop_inspire_isaac.retarget import (
    INSPIRE_ACTUATORS,
    InspireHandRetargeter,
    RetargetConfig,
)
from teleop_inspire_isaac.sim import DummyInspireHand
from teleop_inspire_isaac.pipeline import TeleopPipeline

ASSET = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "assets", "sample_hand.bvh")
)


def _flat_frame(prefix="RightHand", flex_deg=0.0):
    values = {}
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        for i in (1, 2, 3):
            values[f"{prefix}{finger}{i}"] = {
                "Zrotation": flex_deg, "Xrotation": 0.0, "Yrotation": 0.0
            }
    return BVHFrame(values=values)


def test_actuator_count_and_order():
    assert len(INSPIRE_ACTUATORS) == 6
    r = InspireHandRetargeter()
    out = r.normalized(_flat_frame(flex_deg=0.0))
    assert out.shape == (6,)


def test_open_hand_is_fully_open():
    # flex 0 deg, invert_flexion -> normalized 1.0 (open) for all fingers.
    r = InspireHandRetargeter(RetargetConfig(invert_flexion=True))
    out = r.normalized(_flat_frame(flex_deg=0.0))
    # finger actuators (indices 0..3) should be ~1.0 (open)
    assert np.allclose(out[:4], 1.0)


def test_closed_hand_is_fully_closed():
    cfg = RetargetConfig(flexion_min_deg=0.0, flexion_max_deg=90.0,
                         invert_flexion=True)
    r = InspireHandRetargeter(cfg)
    # 90 deg per joint == max range per joint -> normalized flexion 1 -> open 0
    out = r.normalized(_flat_frame(flex_deg=90.0))
    assert np.allclose(out[:4], 0.0)


def test_command_range():
    cfg = RetargetConfig(output_min=0, output_max=1000)
    r = InspireHandRetargeter(cfg)
    cmd = r.command(_flat_frame(flex_deg=0.0))
    assert cmd.dtype == np.int64
    assert cmd.min() >= 0 and cmd.max() <= 1000


def test_monotonic_closing():
    r = InspireHandRetargeter()
    open_cmd = r.normalized(_flat_frame(flex_deg=0.0))
    half_cmd = r.normalized(_flat_frame(flex_deg=45.0))
    closed_cmd = r.normalized(_flat_frame(flex_deg=90.0))
    # As flexion increases, normalized "openness" must not increase.
    assert open_cmd[3] >= half_cmd[3] >= closed_cmd[3]


def test_pipeline_with_dummy_sim():
    source = BVHFileSource(ASSET)
    retargeter = InspireHandRetargeter()
    sim = DummyInspireHand()
    pipeline = TeleopPipeline(source, retargeter, sim, smoothing=0.5)
    stats = pipeline.run()
    assert stats.frames == 10
    assert len(sim.history) == 10
    # Every recorded target stays within [0, 1].
    for t in sim.history:
        assert t.min() >= 0.0 and t.max() <= 1.0


def test_retarget_sequence_shape():
    data = load_bvh(ASSET)
    r = InspireHandRetargeter()
    seq = r.retarget_sequence(data.frames)
    assert seq.shape == (10, 6)
