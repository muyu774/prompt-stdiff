import os

import numpy as np

from teleop_inspire_isaac.mocap import load_bvh, parse_bvh

ASSET = os.path.join(
    os.path.dirname(__file__), os.pardir, "assets", "sample_hand.bvh"
)

MINIMAL_BVH = """
HIERARCHY
ROOT RightHand
{
  OFFSET 0.0 0.0 0.0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT RightHandIndex1
  {
    OFFSET 0.5 0.0 0.0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0.3 0.0 0.0
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.0166667
0 0 0 0 0 0 10 0 0
0 0 0 0 0 0 20 0 0
"""


def test_parse_minimal():
    data = parse_bvh(MINIMAL_BVH)
    assert data.root == "RightHand"
    assert data.num_frames == 2
    assert "RightHandIndex1" in data.joint_names()
    # End sites are excluded from joint_names.
    assert all("EndSite" not in n for n in data.joint_names())
    assert abs(data.frame_time - 0.0166667) < 1e-9


def test_frame_euler_rad():
    data = parse_bvh(MINIMAL_BVH)
    euler = data.frames[1].euler_rad("RightHandIndex1")
    # Zrotation=20deg is the third (index 2) component.
    assert np.isclose(np.degrees(euler[2]), 20.0)
    assert np.isclose(euler[0], 0.0)


def test_load_sample_asset():
    data = load_bvh(os.path.abspath(ASSET))
    assert data.num_frames == 10
    names = data.joint_names()
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        assert f"RightHand{finger}1" in names


def test_too_short_motion_raises():
    bad = MINIMAL_BVH.replace("0 0 0 0 0 0 20 0 0", "0 0 0")
    try:
        parse_bvh(bad)
    except ValueError:
        return
    raise AssertionError("expected ValueError for truncated MOTION data")
