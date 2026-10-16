from __future__ import annotations

import importlib.util
import json
import unittest
from collections import Counter
from pathlib import Path

from egoshift.support import (
    GeometryConfig,
    analytic_support,
    cell_construction_diagnostics,
    finite_target_correct,
    population_target,
    support_table,
)
from egoshift.taskb_audit import audit_taskb_rows, parse_prompt_action


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "taskb_geometry_sensitivity.json"
ENUMERATOR_PATH = ROOT / "tools" / "independent_support_enumerator.py"


def load_independent_enumerator():
    spec = importlib.util.spec_from_file_location("independent_support_enumerator", ENUMERATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load independent enumerator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AnalyticSupportTests(unittest.TestCase):
    def test_release_supports(self) -> None:
        expected = {
            (30, "left"): ("in front", "to the right"),
            (30, "right"): ("in front", "to the left"),
            (60, "left"): ("in front", "to the right"),
            (60, "right"): ("in front", "to the left"),
            (90, "left"): ("in front", "to the right", "behind"),
            (90, "right"): ("in front", "behind", "to the left"),
            (120, "left"): ("to the right", "behind"),
            (120, "right"): ("behind", "to the left"),
            (180, "left"): ("to the right", "behind", "to the left"),
            (180, "right"): ("to the right", "behind", "to the left"),
        }
        for cell, support in expected.items():
            with self.subTest(cell=cell):
                self.assertEqual(analytic_support(*cell), support)

    def test_release_population_and_finite_targets(self) -> None:
        cells = {(float(rotation), turn): 114 for rotation in (30, 60, 90, 120, 180) for turn in ("left", "right")}
        supports = support_table((30, 60, 90, 120, 180))
        self.assertAlmostEqual(population_target(cells, supports), 13 / 30)
        self.assertEqual(finite_target_correct(cells, supports), 494)
        self.assertAlmostEqual(494 / 1140, 13 / 30)

    def test_finite_ceiling_and_excess(self) -> None:
        balanced = cell_construction_diagnostics(Counter({"a": 4, "b": 3, "c": 3}), ("a", "b", "c"))
        self.assertEqual(balanced["finite_target_correct"], 4)
        self.assertTrue(balanced["attains_target"])
        skewed = cell_construction_diagnostics(Counter({"a": 5, "b": 3, "c": 2}), ("a", "b", "c"))
        self.assertEqual(skewed["excess_correct"], 1)
        self.assertFalse(skewed["attains_target"])

    def test_nondivisible_small_and_empty_cells(self) -> None:
        small = cell_construction_diagnostics(Counter({"a": 1, "b": 1}), ("a", "b", "c"))
        self.assertEqual(small["finite_target_correct"], 1)
        self.assertTrue(small["attains_target"])
        empty = cell_construction_diagnostics(Counter(), ("a", "b", "c"))
        self.assertEqual(empty["finite_target_correct"], 0)
        self.assertTrue(empty["attains_target"])

    def test_boundary_touch_is_not_support(self) -> None:
        narrow = GeometryConfig(source_bearing_half_width_degrees=45.0)
        self.assertEqual(analytic_support(90, "left", narrow), ("to the right",))
        self.assertEqual(analytic_support(180, "left", narrow), ("behind",))

    def test_left_right_mirror_and_180_equivalence(self) -> None:
        mirror = {
            "in front": "in front",
            "to the right": "to the left",
            "behind": "behind",
            "to the left": "to the right",
        }
        for rotation in (30, 60, 90, 120, 180):
            left = set(analytic_support(rotation, "left"))
            right = set(analytic_support(rotation, "right"))
            self.assertEqual({mirror[label] for label in left}, right)
        self.assertEqual(analytic_support(180, "left"), analytic_support(180, "right"))

    def test_five_degree_action_calibration_preserves_support_sizes(self) -> None:
        baseline = [len(analytic_support(rotation, "left")) for rotation in (30, 60, 90, 120, 180)]
        for rotations in ((25, 55, 85, 115, 175), (35, 65, 95, 125, 185)):
            self.assertEqual([len(analytic_support(rotation, "left")) for rotation in rotations], baseline)

    def test_ten_degree_change_can_cross_support_transition(self) -> None:
        self.assertEqual(len(analytic_support(60, "left")), 2)
        self.assertEqual(len(analytic_support(70, "left")), 3)

    def test_seventy_five_is_maximal_for_the_proposed_support_pattern(self) -> None:
        at_limit = GeometryConfig(source_bearing_half_width_degrees=75.0)
        above_limit = GeometryConfig(source_bearing_half_width_degrees=75.0001)
        self.assertEqual(len(analytic_support(60, "left", at_limit)), 2)
        self.assertEqual(len(analytic_support(120, "left", at_limit)), 2)
        self.assertEqual(len(analytic_support(60, "left", above_limit)), 3)
        self.assertEqual(len(analytic_support(120, "left", above_limit)), 3)

    def test_bad_geometry_and_actions_fail(self) -> None:
        with self.assertRaises(ValueError):
            GeometryConfig(source_bearing_half_width_degrees=180)
        with self.assertRaises(ValueError):
            analytic_support(30, "clockwise")


class IndependentAgreementTests(unittest.TestCase):
    def test_dense_enumerator_agrees_for_all_sensitivity_configs(self) -> None:
        independent = load_independent_enumerator().run(CONFIG_PATH)
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        independent_by_id = {report["id"]: report for report in independent["configurations"]}
        for config in payload["configurations"]:
            geometry = GeometryConfig(
                source_bearing_half_width_degrees=float(
                    config["source_bearing_half_width_degrees"]
                ),
                zone_width_degrees=float(config["zone_width_degrees"]),
                zone_centers_degrees=config["zone_centers_degrees"],
            )
            observed = independent_by_id[config["id"]]["cells"]
            for rotation_value in config["rotations_degrees"]:
                rotation = float(rotation_value)
                display = int(rotation) if rotation.is_integer() else rotation
                for direction in config["turn_directions"]:
                    cell_id = f"{direction}_{display}"
                    with self.subTest(config=config["id"], cell=cell_id):
                        self.assertEqual(
                            observed[cell_id]["support"],
                            list(analytic_support(rotation, direction, geometry)),
                        )


class TaskBAuditTests(unittest.TestCase):
    def test_prompt_action_is_authoritative(self) -> None:
        self.assertEqual(
            parse_prompt_action("Staying in place, you turn left by 180 degrees. Where is it?"),
            (180.0, "left"),
        )

    def test_toy_balanced_pool_passes(self) -> None:
        rows = []
        for index, answer in enumerate(("in front", "to the right"), start=1):
            rows.append(
                {
                    "id": f"q{index}",
                    "task_family": "B",
                    "question": "You turn left by 30 degrees. Where is the target?",
                    "answer": answer,
                    "metadata": {"rotation_degrees": 30, "qb2_orig_turn": "left"},
                }
            )
        report = audit_taskb_rows(rows)
        self.assertTrue(report["pass"])
        self.assertEqual(report["finite_structural_balance_target_correct"], 1)

    def test_unreachable_label_fails(self) -> None:
        rows = [
            {
                "id": "q1",
                "task_family": "B",
                "question": "You turn left by 30 degrees. Where is the target?",
                "answer": "behind",
                "metadata": {"rotation_degrees": 30},
            }
        ]
        report = audit_taskb_rows(rows)
        self.assertFalse(report["pass"])
        self.assertEqual(report["invalid_support_items"], 1)

    def test_deterministic_geometry_and_positive_excess_fail(self) -> None:
        deterministic_rows = [
            {
                "id": "q1",
                "task_family": "B",
                "question": "You turn left by 90 degrees. Where is it?",
                "answer": "to the right",
                "metadata": {"rotation_degrees": 90},
            }
        ]
        deterministic = audit_taskb_rows(
            deterministic_rows, GeometryConfig(source_bearing_half_width_degrees=45.0)
        )
        self.assertFalse(deterministic["pass"])
        self.assertEqual(deterministic["deterministic_cells"], ["left_90"])

        skewed_rows = []
        for index, answer in enumerate(("in front", "in front", "to the right"), start=1):
            skewed_rows.append(
                {
                    "id": f"s{index}",
                    "task_family": "B",
                    "question": "You turn left by 30 degrees. Where is it?",
                    "answer": answer,
                    "metadata": {"rotation_degrees": 30},
                }
            )
        skewed = audit_taskb_rows(skewed_rows)
        self.assertTrue(skewed["pass"])
        self.assertEqual(skewed["total_excess_correct"], 0)
        # For n=3,k=2, the attainable maximum is already ceil(3/2)=2.
        # A stronger skew is required to create positive excess.
        skewed_rows.append(
            {
                "id": "s4",
                "task_family": "B",
                "question": "You turn left by 30 degrees. Where is it?",
                "answer": "in front",
                "metadata": {"rotation_degrees": 30},
            }
        )
        skewed = audit_taskb_rows(skewed_rows)
        self.assertFalse(skewed["pass"])
        self.assertEqual(skewed["total_excess_correct"], 1)

    def test_malformed_prompt_duplicate_and_rotation_mismatch_fail(self) -> None:
        rows = [
            {
                "id": "duplicate",
                "task_family": "B",
                "question": "You turn left by 30 degrees. Where is it?",
                "answer": "in front",
                "metadata": {"rotation_degrees": 60},
            },
            {
                "id": "duplicate",
                "task_family": "B",
                "question": "No action appears here.",
                "answer": "in front",
                "metadata": {"rotation_degrees": 30},
            },
        ]
        report = audit_taskb_rows(rows)
        self.assertFalse(report["pass"])
        self.assertEqual(report["duplicate_ids"], ["duplicate"])
        self.assertEqual(len(report["parsing_errors"]), 1)
        self.assertEqual(report["metadata_rotation_mismatch_ids"], ["duplicate"])


if __name__ == "__main__":
    unittest.main()
