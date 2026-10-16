from __future__ import annotations

import sys
import unittest
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
if str(RELEASE / "src") not in sys.path:
    sys.path.insert(0, str(RELEASE / "src"))

from egoshift.taskc_occlusion_geometry import (  # noqa: E402
    OcclusionSearchConfig,
    propose_occluding_poses,
)


def _obj(object_id: str, x: float, z: float, width: float, height: float = 1.0):
    return {
        "objectId": object_id,
        "position": {"x": x, "y": 0.0, "z": z},
        "axisAlignedBoundingBox": {
            "center": {"x": x, "y": 0.0, "z": z},
            "size": {"x": width, "y": height, "z": width},
        },
    }


def _grid(x0=-6.0, x1=6.0, z0=-6.0, z1=6.0, step=0.25):
    positions = []
    x = x0
    while x <= x1:
        z = z0
        while z <= z1:
            positions.append({"x": round(x, 3), "y": 0.9, "z": round(z, 3)})
            z += step
        x += step
    return positions


class OcclusionProposalTests(unittest.TestCase):
    def test_proposes_a_pose_behind_a_larger_blocker(self) -> None:
        objects = [
            _obj("Mug|1", 0.0, 0.0, 0.2),          # small target
            _obj("Sofa|1", 0.0, 1.5, 1.6, 1.2),    # large blocker, 1.5 m away
        ]
        out = propose_occluding_poses(objects, _grid(), "Mug|1")
        self.assertTrue(out)
        best = out[0]
        self.assertEqual(best["occluder_id"], "Sofa|1")
        # Camera must sit beyond the blocker, i.e. further from the target.
        self.assertGreater(best["camera_to_target_meters"], 1.5)
        # And the blocker must be comfortably wider in angle than the target.
        self.assertGreaterEqual(best["angular_ratio"], 1.35)

    def test_a_blocker_too_small_to_cover_is_not_proposed(self) -> None:
        objects = [
            _obj("Sofa|1", 0.0, 0.0, 1.8, 1.2),    # large target
            _obj("Mug|1", 0.0, 1.5, 0.2, 0.15),    # tiny blocker cannot cover it
        ]
        self.assertEqual(propose_occluding_poses(objects, _grid(), "Sofa|1"), [])

    def test_structural_elements_are_never_blockers(self) -> None:
        objects = [
            _obj("Mug|1", 0.0, 0.0, 0.2),
            _obj("wall|4|2", 0.0, 1.5, 3.0, 2.5),
            _obj("door|1|1", 1.5, 0.0, 1.0, 2.0),
        ]
        # Hiding behind a wall is "in another room", not object occlusion.
        self.assertEqual(propose_occluding_poses(objects, _grid(), "Mug|1"), [])

    def test_same_type_blocker_is_rejected_as_a_referent_distractor(self) -> None:
        objects = [_obj("Chair|1", 0.0, 0.0, 0.5), _obj("Chair|2", 0.0, 1.5, 1.2)]
        self.assertEqual(propose_occluding_poses(objects, _grid(), "Chair|1"), [])

    def test_no_reachable_position_yields_no_proposal(self) -> None:
        objects = [_obj("Mug|1", 0.0, 0.0, 0.2), _obj("Sofa|1", 0.0, 1.5, 1.6, 1.2)]
        # Reachable positions exist but nowhere near the required standoff.
        far = [{"x": 50.0, "y": 0.9, "z": 50.0}]
        self.assertEqual(propose_occluding_poses(objects, far, "Mug|1"), [])

    def test_camera_yaw_points_back_at_the_target(self) -> None:
        import math

        objects = [_obj("Mug|1", 0.0, 0.0, 0.2), _obj("Sofa|1", 0.0, 2.0, 1.6, 1.2)]
        best = propose_occluding_poses(objects, _grid(), "Mug|1")[0]
        px, pz = best["position"]["x"], best["position"]["z"]
        expected = math.degrees(math.atan2(0.0 - px, 0.0 - pz)) % 360.0
        self.assertAlmostEqual(best["yaw"], expected, places=6)
        # Blocker sits on the +z side, so the camera must too.
        self.assertGreater(pz, 2.0)

    def test_one_proposal_per_blocker_ranked_toward_the_preferred_ratio(self) -> None:
        # Ranking prefers a comfortable margin, not a maximal one: the largest
        # ratios are tiny targets behind huge blockers, which barely clear the
        # source-pixel floor and produce repetitive items.
        objects = [
            _obj("Mug|1", 0.0, 0.0, 0.2),
            _obj("Sofa|1", 0.0, 1.5, 1.8, 1.2),
            _obj("Chair|1", 1.5, 0.0, 0.7, 1.0),
        ]
        config = OcclusionSearchConfig(preferred_angular_ratio=2.5)
        out = propose_occluding_poses(objects, _grid(), "Mug|1", config=config)
        ids = [p["occluder_id"] for p in out]
        self.assertEqual(len(ids), len(set(ids)))
        distances = [abs(p["angular_ratio"] - 2.5) for p in out]
        self.assertEqual(distances, sorted(distances))

    def test_a_blocker_in_another_room_is_rejected(self) -> None:
        # Across a wall the real occluder is the wall, whatever furniture
        # happens to fall in the line of sight. Without the room test this
        # produced "ball in the next room, behind a painting" items.
        rooms = [
            {"id": "room|1", "floorPolygon": [
                {"x": -4.0, "y": 0, "z": -4.0}, {"x": -4.0, "y": 0, "z": 0.5},
                {"x": 4.0, "y": 0, "z": 0.5}, {"x": 4.0, "y": 0, "z": -4.0}]},
            {"id": "room|2", "floorPolygon": [
                {"x": -4.0, "y": 0, "z": 0.5}, {"x": -4.0, "y": 0, "z": 6.0},
                {"x": 4.0, "y": 0, "z": 6.0}, {"x": 4.0, "y": 0, "z": 0.5}]},
        ]
        objects = [_obj("Mug|1", 0.0, 0.0, 0.2), _obj("Sofa|1", 0.0, 1.5, 1.6, 1.2)]
        # Target sits in room|1; every camera pose behind the blocker lands in
        # room|2, so nothing may be proposed.
        self.assertEqual(
            propose_occluding_poses(objects, _grid(), "Mug|1", rooms=rooms), []
        )
        # Same geometry, no room information supplied: the test cannot apply.
        self.assertTrue(propose_occluding_poses(objects, _grid(), "Mug|1"))

    def test_same_room_blocker_is_still_accepted(self) -> None:
        rooms = [{"id": "room|1", "floorPolygon": [
            {"x": -6.0, "y": 0, "z": -6.0}, {"x": -6.0, "y": 0, "z": 6.0},
            {"x": 6.0, "y": 0, "z": 6.0}, {"x": 6.0, "y": 0, "z": -6.0}]}]
        objects = [_obj("Mug|1", 0.0, 0.0, 0.2), _obj("Sofa|1", 0.0, 1.5, 1.6, 1.2)]
        out = propose_occluding_poses(objects, _grid(), "Mug|1", rooms=rooms)
        self.assertTrue(out)
        self.assertEqual(out[0]["room_id"], "room|1")

    def test_a_short_blocker_cannot_cover_a_tall_target(self) -> None:
        objects = [
            _obj("Painting|1", 0.0, 0.0, 0.6, 2.0),   # tall target
            _obj("Stool|1", 0.0, 1.5, 1.4, 0.4),      # wide but very short
        ]
        self.assertEqual(propose_occluding_poses(objects, _grid(), "Painting|1"), [])

    def test_config_ratio_controls_strictness_not_correctness(self) -> None:
        objects = [_obj("Mug|1", 0.0, 0.0, 0.35), _obj("Sofa|1", 0.0, 1.5, 1.0, 1.2)]
        lenient = propose_occluding_poses(
            objects, _grid(), "Mug|1", config=OcclusionSearchConfig(minimum_angular_ratio=1.0)
        )
        strict = propose_occluding_poses(
            objects, _grid(), "Mug|1", config=OcclusionSearchConfig(minimum_angular_ratio=6.0)
        )
        self.assertTrue(lenient)
        self.assertEqual(strict, [])


if __name__ == "__main__":
    unittest.main()
