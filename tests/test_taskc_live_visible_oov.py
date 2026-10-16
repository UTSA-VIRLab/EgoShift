from __future__ import annotations

import unittest

from egoshift.taskc_live_evidence import (
    LiveEvidenceConfig,
    LiveEvidenceRejected,
    derive_live_out_of_view,
    derive_live_visible,
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


def _config() -> LiveEvidenceConfig:
    return LiveEvidenceConfig(
        minimum_source_pixels=10,
        minimum_visible_target_pixels=5,
        minimum_one_edge_pixels=0,
        minimum_corner_pixels=0,
    )


def _record(object_id: str, x: float, z: float, size: float = 1.0):
    return {
        "objectId": object_id,
        "position": {"x": x, "y": 0.0, "z": z},
        "axisAlignedBoundingBox": {
            "center": {"x": x, "y": 0.0, "z": z},
            "size": {"x": size, "y": size, "z": size},
        },
    }


def _source_event(objects):
    import numpy as np

    segmentation = np.zeros((20, 20, 3), dtype=np.uint8)
    segmentation[6:14, 6:14] = (10, 20, 30)
    return FakeEvent(segmentation, objects, {"Mug|1": (10, 20, 30)})


class LiveVisibleTests(unittest.TestCase):
    def test_visible_pass_records_target_bbox(self) -> None:
        import numpy as np

        objects = [_record("Mug|1", 0.0, 4.0)]
        target_segmentation = np.zeros((20, 20, 3), dtype=np.uint8)
        target_segmentation[8:12, 8:12] = (10, 20, 30)
        target = FakeEvent(target_segmentation, objects, {"Mug|1": (10, 20, 30)})
        evidence = derive_live_visible(
            _source_event(objects), target, "Mug|1",
            vertical_fov_degrees=90.0, config=_config(),
        )
        self.assertEqual(evidence["derived_class"], "visible")
        self.assertEqual(evidence["target"]["pixel_count"], 16)
        self.assertEqual(evidence["target"]["bbox_xyxy"], [8, 8, 12, 12])

    def test_visible_rejects_below_pixel_floor(self) -> None:
        import numpy as np

        objects = [_record("Mug|1", 0.0, 4.0)]
        target_segmentation = np.zeros((20, 20, 3), dtype=np.uint8)
        target_segmentation[9:10, 9:12] = (10, 20, 30)
        target = FakeEvent(target_segmentation, objects, {"Mug|1": (10, 20, 30)})
        with self.assertRaises(LiveEvidenceRejected) as ctx:
            derive_live_visible(
                _source_event(objects), target, "Mug|1",
                vertical_fov_degrees=90.0, config=_config(),
            )
        self.assertEqual(ctx.exception.reason, "visible_target_pixels_below_floor")

    def test_visible_rejects_same_type_distractor(self) -> None:
        import numpy as np

        objects = [_record("Mug|1", 0.0, 4.0), _record("Mug|2", 1.0, 4.0)]
        target_segmentation = np.zeros((20, 20, 3), dtype=np.uint8)
        target_segmentation[8:12, 8:12] = (10, 20, 30)
        target_segmentation[2:5, 2:5] = (90, 90, 90)
        target = FakeEvent(
            target_segmentation, objects,
            {"Mug|1": (10, 20, 30), "Mug|2": (90, 90, 90)},
        )
        with self.assertRaises(LiveEvidenceRejected) as ctx:
            derive_live_visible(
                _source_event(objects), target, "Mug|1",
                vertical_fov_degrees=90.0, config=_config(),
            )
        self.assertEqual(ctx.exception.reason, "same_type_distractor_at_target")


class LiveOutOfViewTests(unittest.TestCase):
    def test_oov_pass_with_left_separating_plane(self) -> None:
        import numpy as np

        objects = [_record("Mug|1", -10.0, 1.0)]
        target = FakeEvent(
            np.zeros((20, 20, 3), dtype=np.uint8), objects, {"Mug|1": (10, 20, 30)}
        )
        evidence = derive_live_out_of_view(
            _source_event(objects), target, "Mug|1",
            vertical_fov_degrees=90.0, config=_config(),
        )
        self.assertEqual(evidence["derived_class"], "out_of_fov")
        self.assertTrue(evidence["frustum_separation"]["planes"]["left"])
        self.assertEqual(evidence["target"]["pixel_count"], 0)

    def test_oov_rejects_when_aabb_not_separable(self) -> None:
        import numpy as np

        objects = [_record("Mug|1", 0.0, 4.0)]
        target = FakeEvent(
            np.zeros((20, 20, 3), dtype=np.uint8), objects, {"Mug|1": (10, 20, 30)}
        )
        with self.assertRaises(LiveEvidenceRejected) as ctx:
            derive_live_out_of_view(
                _source_event(objects), target, "Mug|1",
                vertical_fov_degrees=90.0, config=_config(),
            )
        self.assertEqual(ctx.exception.reason, "no_separating_frustum_plane")

    def test_oov_rejects_nonzero_target_mask(self) -> None:
        import numpy as np

        objects = [_record("Mug|1", -10.0, 1.0)]
        target_segmentation = np.zeros((20, 20, 3), dtype=np.uint8)
        target_segmentation[0:3, 0:3] = (10, 20, 30)
        target = FakeEvent(target_segmentation, objects, {"Mug|1": (10, 20, 30)})
        with self.assertRaises(LiveEvidenceRejected) as ctx:
            derive_live_out_of_view(
                _source_event(objects), target, "Mug|1",
                vertical_fov_degrees=90.0, config=_config(),
            )
        self.assertEqual(ctx.exception.reason, "oov_target_mask_not_zero")


if __name__ == "__main__":
    unittest.main()
