from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
if str(RELEASE / "src") not in sys.path:
    sys.path.insert(0, str(RELEASE / "src"))

_spec = importlib.util.spec_from_file_location(
    "mine_taska_pairs", RELEASE / "tools" / "mine_taska_pairs.py"
)
miner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(miner)


class FakeEvent:
    def __init__(self, objects):
        self.metadata = {"objects": objects}


def _obj(object_id: str, object_type: str):
    return {
        "objectId": object_id,
        "objectType": object_type,
        "axisAlignedBoundingBox": {
            "center": {"x": 0.0, "y": 0.0, "z": 2.0},
            "size": {"x": 1.0, "y": 1.0, "z": 1.0},
        },
    }


class TaskATypeGroupingTests(unittest.TestCase):
    def test_grouping_key_is_the_object_id_prefix(self) -> None:
        # The evidence module decides "same type" from the objectId prefix, so
        # the miner must group on the same key or the two disagree.
        event = FakeEvent([_obj("Painting|2|6", "Painting"), _obj("Painting|3|1", "Painting")])
        grouped = miner._instances_by_type(event)
        self.assertEqual(sorted(grouped), ["Painting"])
        self.assertEqual(len(grouped["Painting"]), 2)

    def test_structural_elements_are_excluded_by_either_naming(self) -> None:
        # Regression: a door carries objectType "Doorway" but objectId prefix
        # "door". Excluding on objectType alone let it through and it then
        # grouped downstream as type "door".
        event = FakeEvent(
            [
                _obj("door|3|4", "Doorway"),
                _obj("window|1|2", "Window"),
                _obj("Bed|1", "Bed"),
            ]
        )
        grouped = miner._instances_by_type(event)
        self.assertEqual(sorted(grouped), ["Bed"])

    def test_objects_without_a_bounding_box_are_skipped(self) -> None:
        event = FakeEvent([{"objectId": "Mug|1", "objectType": "Mug"}])
        self.assertEqual(miner._instances_by_type(event), {})

    def test_preferred_distance_scales_with_object_size(self) -> None:
        near = miner._preferred_distance(0.10, 110.0)
        far = miner._preferred_distance(0.80, 110.0)
        self.assertLess(near, far)
        for radius in (0.01, 0.1, 1.0, 10.0):
            distance = miner._preferred_distance(radius, 110.0)
            self.assertGreaterEqual(distance, miner.DISTANCE_RANGE[0])
            self.assertLessEqual(distance, miner.DISTANCE_RANGE[1])


if __name__ == "__main__":
    unittest.main()
