from __future__ import annotations

import sys
import unittest
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
if str(RELEASE / "tools") not in sys.path:
    sys.path.insert(0, str(RELEASE / "tools"))

from run_local_models import ended_by_length, generation_limit, option_order_for  # noqa: E402


class LocalGenerationPolicyTests(unittest.TestCase):
    def test_panel_generation_allowance_overrides_short_default(self) -> None:
        self.assertEqual(generation_limit({"name": "ordinary"}), 48)
        self.assertEqual(
            generation_limit({"name": "QVQ", "max_new_tokens": 8192}), 8192)
        self.assertEqual(
            generation_limit({"name": "QVQ", "max_new_tokens": 8192}, 256), 256)

    def test_length_cap_is_not_confused_with_normal_eos(self) -> None:
        self.assertFalse(ended_by_length([1, 2], 3, 99))
        self.assertFalse(ended_by_length([1, 2, 99], 3, 99))
        self.assertTrue(ended_by_length([1, 2, 3], 3, 99))

    def test_generation_allowance_must_be_positive(self) -> None:
        with self.assertRaises(ValueError):
            generation_limit({"max_new_tokens": -1})

    def test_control_permutation_is_deterministic_and_non_identity(self) -> None:
        options = ["a", "b", "c", "d"]
        self.assertEqual(option_order_for(options, 0, "seed"), options)
        first = option_order_for(options, 1, "seed")
        self.assertNotEqual(first, options)
        self.assertEqual(first, option_order_for(options, 1, "seed"))


if __name__ == "__main__":
    unittest.main()
