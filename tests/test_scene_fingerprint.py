from __future__ import annotations

import unittest

from egoshift.scene_fingerprint import leakage_components, metadata_fingerprint


class SceneFingerprintTests(unittest.TestCase):
    def test_nonphysical_ids_do_not_change_fingerprint(self) -> None:
        first = {"scene_id": "scene_a", "source_house_index": 1, "objects": [{"x": 1}]}
        second = {"scene_id": "scene_b", "source_house_index": 99, "objects": [{"x": 1}]}
        self.assertEqual(metadata_fingerprint(first), metadata_fingerprint(second))

    def test_physical_content_changes_fingerprint(self) -> None:
        self.assertNotEqual(
            metadata_fingerprint({"objects": [{"x": 1}]}),
            metadata_fingerprint({"objects": [{"x": 2}]}),
        )

    def test_exact_and_conservative_aliases_join_components(self) -> None:
        mapping = {"a": "same", "b": "same", "c": "third", "d": "fourth"}
        components = leakage_components(mapping, conservative_pairs=[("c", "d")])
        self.assertEqual(components["a"], components["b"])
        self.assertEqual(components["c"], components["d"])
        self.assertNotEqual(components["a"], components["c"])


if __name__ == "__main__":
    unittest.main()
