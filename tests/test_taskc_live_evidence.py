from __future__ import annotations

import unittest

from egoshift.taskc_live_evidence import (
    LiveEvidenceConfig,
    LiveEvidenceRejected,
    derive_live_full_occlusion,
    exact_mask_bbox,
    exact_mask_sha256,
    exact_object_mask,
)


class FakeEvent:
    def __init__(self, segmentation, objects, colors):
        import numpy as np

        self.instance_segmentation_frame = np.asarray(segmentation, dtype=np.uint8)
        self.object_id_to_color = dict(colors)
        self.color_to_object_id = {tuple(color): oid for oid, color in colors.items()}
        self.metadata = {
            "objects": objects,
            "cameraPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
            "agent": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "cameraHorizon": 0.0,
            },
        }


def _target_record():
    return {
        "objectId": "Mug|1",
        "position": {"x": 0.0, "y": 0.0, "z": 4.0},
        "axisAlignedBoundingBox": {
            "center": {"x": 0.0, "y": 0.0, "z": 4.0},
            "size": {"x": 2.0, "y": 2.0, "z": 1.0},
        },
    }


class LiveTaskCEvidenceTests(unittest.TestCase):
    def _events(self, occluder_id: str = "Chair|1"):
        import numpy as np

        source_segmentation = np.zeros((20, 20, 3), dtype=np.uint8)
        source_segmentation[6:14, 6:14] = (10, 20, 30)
        target_segmentation = np.zeros((20, 20, 3), dtype=np.uint8)
        target_segmentation[7:14, 7:14] = (40, 50, 60)
        objects = [
            _target_record(),
            {
                "objectId": occluder_id,
                "position": {"x": 0.0, "y": 0.0, "z": 2.5},
            },
        ]
        source = FakeEvent(
            source_segmentation,
            objects,
            {"Mug|1": (10, 20, 30)},
        )
        target = FakeEvent(
            target_segmentation,
            objects,
            {occluder_id: (40, 50, 60)},
        )
        return source, target

    def test_exact_mask_never_substitutes_same_type(self) -> None:
        source, target = self._events()
        source_mask = exact_object_mask(source, "Mug|1")
        target_mask = exact_object_mask(target, "Mug|1")
        self.assertEqual(int(source_mask.sum()), 64)
        self.assertEqual(int(target_mask.sum()), 0)
        self.assertEqual(exact_mask_bbox(source_mask), [6, 6, 14, 14])
        self.assertEqual(len(exact_mask_sha256(source_mask)), 64)

    def test_live_occlusion_uses_zero_target_mask_geometry_and_foreground_mask(self) -> None:
        source, target = self._events()
        evidence = derive_live_full_occlusion(
            source,
            target,
            "Mug|1",
            vertical_fov_degrees=90.0,
            config=LiveEvidenceConfig(
                minimum_source_pixels=30,
                minimum_occluder_overlap=0.05,
                minimum_one_edge_pixels=30,
                minimum_corner_pixels=30,
            ),
        )
        self.assertEqual(evidence["derived_class"], "occluded")
        self.assertEqual(evidence["source"]["bbox_xyxy"], [6, 6, 14, 14])
        self.assertEqual(evidence["target"]["pixel_count"], 0)
        self.assertEqual(evidence["occluder"]["object_id"], "Chair|1")
        self.assertGreater(evidence["occluder"]["overlap_ratio"], 0.05)
        self.assertEqual(
            evidence["projection"]["method"],
            "live_3d_aabb_convex_hull_x_exact_instance_masks",
        )

    def test_structural_foreground_is_not_accepted_as_occlusion_evidence(self) -> None:
        source, target = self._events("wall|1")
        with self.assertRaises(LiveEvidenceRejected) as raised:
            derive_live_full_occlusion(
                source,
                target,
                "Mug|1",
                vertical_fov_degrees=90.0,
                config=LiveEvidenceConfig(
                    minimum_source_pixels=30,
                    minimum_occluder_overlap=0.05,
                    minimum_one_edge_pixels=30,
                    minimum_corner_pixels=30,
                ),
            )
        self.assertEqual(raised.exception.reason, "no_nonstructural_foreground_overlap")


if __name__ == "__main__":
    unittest.main()

