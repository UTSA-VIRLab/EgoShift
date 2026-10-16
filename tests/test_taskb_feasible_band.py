from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
if str(RELEASE / "src") not in sys.path:
    sys.path.insert(0, str(RELEASE / "src"))

_spec = importlib.util.spec_from_file_location(
    "plan_taskb_targeted_mining", RELEASE / "tools" / "plan_taskb_targeted_mining.py"
)
planner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(planner)

CONE = 66.58677555362947
MARGIN = 15.0


class FeasibleBandTests(unittest.TestCase):
    def test_abundant_cell_has_a_wide_band_centred_on_zero(self) -> None:
        band = planner.feasible_centre_band(180, "left", "behind", CONE, MARGIN)
        self.assertIsNotNone(band)
        low, high = band
        self.assertAlmostEqual(low, -30.0, delta=0.2)
        self.assertAlmostEqual(high, 30.0, delta=0.2)

    def test_edge_cells_are_confined_to_the_cone_edge(self) -> None:
        for rotation, turn, answer, sign in (
            (180, "left", "to the left", 1.0),
            (180, "left", "to the right", -1.0),
            (90, "left", "in front", -1.0),
            (90, "right", "behind", -1.0),
        ):
            band = planner.feasible_centre_band(rotation, turn, answer, CONE, MARGIN)
            self.assertIsNotNone(band, f"{rotation}:{turn}:{answer}")
            low, high = band
            inner, outer = (low, high) if sign > 0 else (high, low)
            self.assertAlmostEqual(abs(inner), 60.0, delta=0.2)
            self.assertLessEqual(abs(outer), CONE)
            self.assertLess(high - low, 7.0)

    def test_the_historical_minus_55_aim_is_outside_the_feasible_band(self) -> None:
        # Regression guard for the r2c aiming defect: -55 deg satisfies the
        # cone and the answer zone but not the whole-box margin, so no jitter
        # around it could ever yield a row.
        band = planner.feasible_centre_band(90, "left", "in front", CONE, MARGIN)
        low, high = band
        for jitter in (-2.5, 0.0, 2.5):
            self.assertFalse(low <= -55.0 + jitter <= high)

    def test_an_impossible_margin_yields_no_band(self) -> None:
        self.assertIsNone(
            planner.feasible_centre_band(180, "left", "to the left", CONE, 60.0)
        )


if __name__ == "__main__":
    unittest.main()
