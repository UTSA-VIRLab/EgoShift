from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

from egoshift.taskb_geometry import (
    box_center_bearing_degrees,
    canonical_zone,
    corner_post_diagnostic,
    horizontal_bearing_degrees,
    item_pose_diagnostic,
    post_action_bearing_degrees,
    source_envelope_diagnostic,
    wrap180,
)


ROOT = Path(__file__).resolve().parents[1]
INDEPENDENT_PATH = ROOT / "tools" / "independent_pose_geometry.py"


def load_independent():
    spec = importlib.util.spec_from_file_location("independent_pose_geometry", INDEPENDENT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load independent pose verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def box_at(x: float, z: float, radius: float = 0.1) -> list[list[float]]:
    return [
        [x + dx, y, z + dz]
        for dx in (-radius, radius)
        for y in (0.0, 1.0)
        for dz in (-radius, radius)
    ]


class PoseGeometryTests(unittest.TestCase):
    def test_cardinal_bearings_and_yaw(self) -> None:
        self.assertAlmostEqual(
            horizontal_bearing_degrees(
                camera_x=0, camera_z=0, camera_yaw_degrees=0, target_x=0, target_z=1
            ),
            0,
        )
        self.assertAlmostEqual(
            horizontal_bearing_degrees(
                camera_x=0, camera_z=0, camera_yaw_degrees=0, target_x=1, target_z=0
            ),
            90,
        )
        self.assertAlmostEqual(
            horizontal_bearing_degrees(
                camera_x=0, camera_z=0, camera_yaw_degrees=90, target_x=1, target_z=0
            ),
            0,
        )

    def test_box_center_and_action_sign(self) -> None:
        source = box_center_bearing_degrees(
            camera_position={"x": 0, "z": 0},
            camera_yaw_degrees=0,
            corners=box_at(1, 1),
        )
        self.assertAlmostEqual(source, 45.0)
        self.assertAlmostEqual(post_action_bearing_degrees(30, 30, "left"), 60)
        self.assertAlmostEqual(post_action_bearing_degrees(-30, 30, "right"), -60)

    def test_frozen_half_open_zone_boundaries(self) -> None:
        expected = {
            -180: "behind",
            -135: "to the left",
            -45: "in front",
            44.999: "in front",
            45: "to the right",
            134.999: "to the right",
            135: "behind",
            180: "behind",
        }
        for angle, label in expected.items():
            with self.subTest(angle=angle):
                self.assertEqual(canonical_zone(angle), label)

    def test_source_envelope_is_strict_and_not_visibility_inference(self) -> None:
        self.assertTrue(source_envelope_diagnostic(66.5)["strictly_inside"])
        self.assertFalse(source_envelope_diagnostic(66.58677555362947)["strictly_inside"])
        self.assertFalse(source_envelope_diagnostic(91.713)["strictly_inside"])

    def test_item_diagnostic_checks_gold_and_envelope(self) -> None:
        report = item_pose_diagnostic(
            camera_position={"x": 0, "z": 0},
            camera_yaw_degrees=0,
            corners=box_at(0, 2),
            rotation_degrees=90,
            turn_direction="left",
            expected_answer="to the right",
        )
        self.assertTrue(report["pass"])
        self.assertEqual(report["derived_answer"], "to the right")

    def test_corner_extent_must_stay_in_one_zone(self) -> None:
        clear = corner_post_diagnostic(
            camera_position={"x": 0, "z": 0},
            camera_yaw_degrees=0,
            corners=box_at(0, 2, radius=0.1),
            rotation_degrees=90,
            turn_direction="left",
            expected_answer="to the right",
        )
        self.assertTrue(clear["all_corners_match_expected"])
        self.assertGreater(clear["minimum_corner_boundary_margin_degrees"], 40)

    def test_independent_implementation_agrees_over_grid(self) -> None:
        independent = load_independent()
        for yaw in (0, 17, 90, 181, 359):
            for x, z in ((0, 2), (2, 0), (-2, 0.5), (1.2, -3.4)):
                corners = box_at(x, z)
                for rotation in (30, 60, 90, 120, 180):
                    for direction in ("left", "right"):
                        report = item_pose_diagnostic(
                            camera_position={"x": 0, "z": 0},
                            camera_yaw_degrees=yaw,
                            corners=corners,
                            rotation_degrees=rotation,
                            turn_direction=direction,
                        )
                        other = independent.verify(
                            camera_x=0,
                            camera_z=0,
                            yaw=yaw,
                            corners=corners,
                            rotation=rotation,
                            direction=direction,
                            half_width=66.58677555362947,
                        )
                        self.assertAlmostEqual(
                            report["source_center_bearing_degrees"],
                            other["source_bearing_degrees"],
                        )
                        self.assertAlmostEqual(
                            wrap180(report["post_center_bearing_degrees"]),
                            other["post_bearing_degrees"],
                        )
                        self.assertEqual(report["derived_answer"], other["answer"])
                        self.assertEqual(
                            report["source_envelope"]["strictly_inside"],
                            other["strictly_inside_source_envelope"],
                        )


if __name__ == "__main__":
    unittest.main()
