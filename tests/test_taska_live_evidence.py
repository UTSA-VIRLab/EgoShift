from __future__ import annotations

import sys
import unittest
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
if str(RELEASE / "src") not in sys.path:
    sys.path.insert(0, str(RELEASE / "src"))

from egoshift.taska_live_evidence import (  # noqa: E402
    TaskAEvidenceConfig,
    derive_live_different_objects,
    derive_live_same_object,
)
from egoshift.taskc_live_evidence import LiveEvidenceRejected  # noqa: E402


class FakeEvent:
    def __init__(self, segmentation, objects, colors):
        import numpy as np

        self.instance_segmentation_frame = np.asarray(segmentation, dtype=np.uint8)
        self.object_id_to_color = dict(colors)
        self.color_to_object_id = {tuple(c): o for o, c in colors.items()}
        self.metadata = {
            "objects": objects,
            "cameraPosition": {"x": 0.0, "y": 0.0, "z": 0.0},
            "agent": {
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
                "cameraHorizon": 0.0,
            },
        }


def _record(object_id: str, x: float, z: float, size: float = 1.0):
    return {
        "objectId": object_id,
        "position": {"x": x, "y": 0.0, "z": z},
        "axisAlignedBoundingBox": {
            "center": {"x": x, "y": 0.0, "z": z},
            "size": {"x": size, "y": size, "z": size},
        },
    }


def _cfg(**kw):
    base = dict(minimum_visible_pixels=20, identifiable_minimum_pixels=5,
                minimum_coverage=0.0)
    base.update(kw)
    return TaskAEvidenceConfig(**base)


def _view(objects, colors, blocks):
    """blocks: list of (color, y0,y1, x0,x1)"""
    import numpy as np

    seg = np.zeros((40, 40, 3), dtype=np.uint8)
    for color, y0, y1, x0, x1 in blocks:
        seg[y0:y1, x0:x1] = color
    return FakeEvent(seg, objects, colors)


class TaskAEvidenceTests(unittest.TestCase):
    def test_same_object_across_two_views(self) -> None:
        objs = [_record("Painting|1", 0.0, 4.0)]
        colors = {"Painting|1": (10, 20, 30)}
        v1 = _view(objs, colors, [((10, 20, 30), 10, 20, 10, 20)])
        v2 = _view(objs, colors, [((10, 20, 30), 12, 24, 12, 24)])
        ev = derive_live_same_object(v1, v2, "Painting|1",
                                     vertical_fov_degrees=90.0, config=_cfg())
        self.assertEqual(ev["derived_class"], "same")
        self.assertEqual(ev["object_type"], "Painting")
        # The cue statistic must be present on both views and summarised.
        self.assertIn("view_coverage", ev["view_1"])
        self.assertIn("minimum_view_coverage", ev)
        self.assertGreater(ev["view_1"]["view_coverage"], 0.0)

    def test_different_objects_one_identifiable_per_view(self) -> None:
        objs = [_record("Painting|1", -1.0, 4.0), _record("Painting|2", 1.0, 4.0)]
        v1 = _view(objs, {"Painting|1": (10, 20, 30)},
                   [((10, 20, 30), 10, 20, 10, 20)])
        v2 = _view(objs, {"Painting|2": (40, 50, 60)},
                   [((40, 50, 60), 10, 20, 20, 30)])
        ev = derive_live_different_objects(v1, v2, "Painting|1", "Painting|2",
                                           vertical_fov_degrees=90.0, config=_cfg())
        self.assertEqual(ev["derived_class"], "different")
        self.assertEqual(ev["view_1"]["object_id"], "Painting|1")
        self.assertEqual(ev["view_2"]["object_id"], "Painting|2")

    def test_two_same_type_instances_in_one_view_is_ambiguous(self) -> None:
        # Both paintings identifiable in view 1: the box no longer picks out a
        # unique referent, so the item must be rejected.
        objs = [_record("Painting|1", -1.0, 4.0), _record("Painting|2", 1.0, 4.0)]
        v1 = _view(objs, {"Painting|1": (10, 20, 30), "Painting|2": (40, 50, 60)},
                   [((10, 20, 30), 10, 20, 10, 20), ((40, 50, 60), 10, 20, 25, 35)])
        v2 = _view(objs, {"Painting|1": (10, 20, 30)},
                   [((10, 20, 30), 12, 24, 12, 24)])
        with self.assertRaises(LiveEvidenceRejected) as ctx:
            derive_live_same_object(v1, v2, "Painting|1",
                                    vertical_fov_degrees=90.0, config=_cfg())
        self.assertIn("same_type_ambiguous", ctx.exception.reason)

    def test_pixel_floor_rejects_a_sliver(self) -> None:
        objs = [_record("Painting|1", 0.0, 4.0)]
        colors = {"Painting|1": (10, 20, 30)}
        v1 = _view(objs, colors, [((10, 20, 30), 10, 11, 10, 12)])   # 2 px
        v2 = _view(objs, colors, [((10, 20, 30), 12, 24, 12, 24)])
        with self.assertRaises(LiveEvidenceRejected) as ctx:
            derive_live_same_object(v1, v2, "Painting|1",
                                    vertical_fov_degrees=90.0, config=_cfg())
        self.assertIn("pixels_below_floor", ctx.exception.reason)

    def test_different_rejects_same_instance_and_type_mismatch(self) -> None:
        objs = [_record("Painting|1", 0.0, 4.0), _record("Sofa|1", 1.0, 4.0)]
        v1 = _view(objs, {"Painting|1": (10, 20, 30)}, [((10, 20, 30), 10, 20, 10, 20)])
        v2 = _view(objs, {"Sofa|1": (40, 50, 60)}, [((40, 50, 60), 10, 20, 10, 20)])
        with self.assertRaises(LiveEvidenceRejected):
            derive_live_different_objects(v1, v2, "Painting|1", "Painting|1",
                                          vertical_fov_degrees=90.0, config=_cfg())
        with self.assertRaises(LiveEvidenceRejected):
            derive_live_different_objects(v1, v2, "Painting|1", "Sofa|1",
                                          vertical_fov_degrees=90.0, config=_cfg())

    def test_coverage_is_unclipped_denominator(self) -> None:
        # A target whose projected hull runs off-frame must score coverage well
        # below 1.0 even though its visible mask is solid: the denominator is
        # deliberately not clipped, because clipping is what the cue is made of.
        objs = [_record("Painting|1", 0.0, 1.6, size=2.0)]
        colors = {"Painting|1": (10, 20, 30)}
        v = _view(objs, colors, [((10, 20, 30), 0, 40, 0, 40)])
        ev = derive_live_same_object(v, v, "Painting|1",
                                     vertical_fov_degrees=90.0, config=_cfg())
        self.assertLess(ev["view_1"]["view_coverage"], 1.0)
        self.assertTrue(ev["view_1"]["mask_touches_frame_edge"])


if __name__ == "__main__":
    unittest.main()
