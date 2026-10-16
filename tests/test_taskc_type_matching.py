from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
if str(RELEASE / "src") not in sys.path:
    sys.path.insert(0, str(RELEASE / "src"))

from egoshift.taskc_type_matching import (  # noqa: E402
    match_object_type_distribution,
    object_type_of,
    type_oracle_accuracy,
)

CLASSES = ("visible", "out_of_fov", "occluded")


def _row(object_type: str, answer: str, index: int) -> dict:
    return {"target_oid": f"{object_type}|{index}", "cls": answer, "key": f"{object_type}{index:04d}"}


def _pool(spec: dict[str, dict[str, int]]) -> list[dict]:
    rows = []
    index = 0
    for object_type, by_class in spec.items():
        for answer, count in by_class.items():
            for _ in range(count):
                rows.append(_row(object_type, answer, index))
                index += 1
    return rows


def _match(rows, cap=None):
    return match_object_type_distribution(
        rows,
        classes=CLASSES,
        class_of=lambda r: r["cls"],
        type_of=lambda r: object_type_of(r["target_oid"]),
        order_key=lambda r: r["key"],
        per_class_cap=cap,
    )


class TypeMatchingTests(unittest.TestCase):
    def test_the_measured_skew_is_removed(self) -> None:
        # Numbers mirror the real pool: pillows overwhelmingly occluded,
        # houseplants overwhelmingly visible.
        rows = _pool({
            "Pillow": {"occluded": 42, "out_of_fov": 3, "visible": 2},
            "HousePlant": {"occluded": 1, "out_of_fov": 2, "visible": 9},
            "Painting": {"occluded": 12, "out_of_fov": 14, "visible": 13},
        })
        before, _ = type_oracle_accuracy(
            [(object_type_of(r["target_oid"]), r["cls"]) for r in rows]
        )
        selected, report = _match(rows)
        after = report["object_type_oracle_after_matching"]
        self.assertGreater(before, 0.55)
        self.assertTrue(report["type_signal_removed"])
        self.assertAlmostEqual(after, report["majority_chance_after_matching"])
        self.assertLess(after, before)

    def test_every_type_contributes_equally_to_every_class(self) -> None:
        rows = _pool({
            "Pillow": {"occluded": 42, "out_of_fov": 3, "visible": 2},
            "Painting": {"occluded": 12, "out_of_fov": 14, "visible": 13},
        })
        selected, _ = _match(rows)
        grid: dict[tuple[str, str], int] = Counter()
        for row in selected:
            grid[(object_type_of(row["target_oid"]), row["cls"])] += 1
        for object_type in {object_type_of(r["target_oid"]) for r in selected}:
            counts = {grid[(object_type, c)] for c in CLASSES}
            self.assertEqual(len(counts), 1, f"{object_type} is not matched")

    def test_a_type_missing_from_one_class_is_dropped_entirely(self) -> None:
        # A type present in only one class is the purest form of the shortcut.
        rows = _pool({
            "BasketBall": {"occluded": 9, "out_of_fov": 0, "visible": 0},
            "Painting": {"occluded": 5, "out_of_fov": 5, "visible": 5},
        })
        selected, report = _match(rows)
        kept = {object_type_of(r["target_oid"]) for r in selected}
        self.assertNotIn("BasketBall", kept)
        self.assertIn("Painting", kept)
        self.assertEqual(report["types_dropped"], 1)

    def test_classes_come_out_equal_in_size(self) -> None:
        rows = _pool({
            "Pillow": {"occluded": 42, "out_of_fov": 3, "visible": 2},
            "Painting": {"occluded": 12, "out_of_fov": 14, "visible": 13},
            "Sofa": {"occluded": 2, "out_of_fov": 6, "visible": 11},
        })
        selected, report = _match(rows)
        counts = set(report["kept_by_class"].values())
        self.assertEqual(len(counts), 1)
        self.assertEqual(sum(report["kept_by_class"].values()), len(selected))

    def test_cap_preserves_the_match_while_shrinking(self) -> None:
        rows = _pool({
            "Pillow": {"occluded": 20, "out_of_fov": 20, "visible": 20},
            "Painting": {"occluded": 20, "out_of_fov": 20, "visible": 20},
            "Sofa": {"occluded": 20, "out_of_fov": 20, "visible": 20},
        })
        selected, report = _match(rows, cap=15)
        self.assertTrue(report["type_signal_removed"])
        for count in report["kept_by_class"].values():
            self.assertLessEqual(count, 15)
        grid: dict[tuple[str, str], int] = Counter()
        for row in selected:
            grid[(object_type_of(row["target_oid"]), row["cls"])] += 1
        for object_type in {object_type_of(r["target_oid"]) for r in selected}:
            counts = {grid[(object_type, c)] for c in CLASSES}
            self.assertEqual(len(counts), 1)

    def test_selection_is_deterministic(self) -> None:
        rows = _pool({
            "Pillow": {"occluded": 9, "out_of_fov": 4, "visible": 6},
            "Painting": {"occluded": 7, "out_of_fov": 7, "visible": 7},
        })
        first, _ = _match(rows)
        second, _ = _match(list(reversed(rows)))
        self.assertEqual([r["key"] for r in first], [r["key"] for r in second])

    def test_an_already_matched_pool_is_left_alone(self) -> None:
        rows = _pool({
            "Painting": {"occluded": 5, "out_of_fov": 5, "visible": 5},
            "Sofa": {"occluded": 3, "out_of_fov": 3, "visible": 3},
        })
        selected, report = _match(rows)
        self.assertEqual(len(selected), len(rows))
        self.assertEqual(report["items_discarded"], 0)


if __name__ == "__main__":
    unittest.main()
