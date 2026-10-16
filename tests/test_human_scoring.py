from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "score_human_responses.py"
SPEC = importlib.util.spec_from_file_location("score_human_responses", SCRIPT)
assert SPEC and SPEC.loader
SCORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SCORER)


class HumanScoringTests(unittest.TestCase):
    def test_majority_tie_is_no_consensus_not_input_order(self) -> None:
        gold = {"i1": "left", "i2": "right"}
        answers = {
            "R01": {"i1": "left", "i2": "right"},
            "R02": {"i1": "right", "i2": "right"},
        }

        summary = SCORER.panel_summary(gold, answers, ["R01", "R02"])

        self.assertEqual(summary["majority_ties"], 1)
        self.assertIsNone(summary["per_item"]["i1"]["majority"])
        self.assertFalse(summary["per_item"]["i1"]["correct"])
        self.assertEqual(summary["majority_accuracy"], 0.5)
        self.assertEqual(summary["decided_majority_accuracy"], 1.0)

        reversed_summary = SCORER.panel_summary(gold, answers, ["R02", "R01"])
        self.assertEqual(summary["per_item"], reversed_summary["per_item"])

    def test_unanimous_panel_has_perfect_agreement(self) -> None:
        gold = {"i1": "same", "i2": "different"}
        answers = {
            "R01": dict(gold),
            "R02": dict(gold),
            "R03": dict(gold),
        }

        summary = SCORER.panel_summary(gold, answers, sorted(answers))

        self.assertEqual(summary["majority_accuracy"], 1.0)
        self.assertEqual(summary["majority_ties"], 0)
        self.assertEqual(summary["fleiss_kappa"], 1.0)
        self.assertTrue(all(row["agreement"] == 1.0 for row in summary["pairwise"]))

    def test_wilson_interval_contains_observed_proportion(self) -> None:
        low, high = SCORER.wilson_interval(51, 120)
        self.assertLessEqual(low, 51 / 120)
        self.assertGreaterEqual(high, 51 / 120)
        self.assertIsNone(SCORER.wilson_interval(0, 0))


if __name__ == "__main__":
    unittest.main()
