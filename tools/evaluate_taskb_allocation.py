#!/usr/bin/env python3
"""Report the exact finite structural target for a Task-B cell allocation.

The equal-allocation design gives 13/30 = 43.333...%. If a cell cannot be
filled to its frozen size, the design must be re-frozen *before* selection and
the target recomputed exactly -- never quietly left at 43.33%. This tool makes
that recomputation auditable, and scans candidate re-freezes.
"""

from __future__ import annotations

import argparse
import json
import sys
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egoshift.support import analytic_support  # noqa: E402
from egoshift.taskb_candidate_r2 import action_supports  # noqa: E402


def evaluate(counts: dict[tuple[int, str], int], supports) -> dict:
    per_cell = []
    numerator = 0
    total = 0
    population = Fraction(0)
    for cell in sorted(counts):
        n = counts[cell]
        k = len(supports[cell])
        ceil_nk = -(-n // k)
        numerator += ceil_nk
        total += n
        per_cell.append(
            {"cell": f"{cell[0]}:{cell[1]}", "n": n, "k": k, "ceil_n_over_k": ceil_nk}
        )
    for cell in counts:
        if total:
            population += Fraction(counts[cell], total) * Fraction(
                1, len(supports[cell])
            )
    return {
        "per_cell": per_cell,
        "total_items": total,
        "finite_correct": numerator,
        "finite_target": float(numerator / total) if total else 0.0,
        "finite_target_exact": f"{numerator}/{total}",
        "population_target": float(population),
        "population_target_exact": str(population),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--role", default="hidden_test")
    parser.add_argument(
        "--cell", action="append", default=[],
        help="Override one cell, e.g. '180:left=30'. Repeatable.",
    )
    parser.add_argument(
        "--scan-rotation", type=int, default=None,
        help="Scan sizes for both presented turns of this rotation.",
    )
    parser.add_argument("--scan-values", default="78,60,52,45,39,30,26,20,15,12")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    half_width = float(config["geometry"]["source_bearing_half_width_degrees"])
    supports = action_supports(half_width)
    base_n = int(config["splits"][args.role]["items_per_action_cell"])
    counts = {cell: base_n for cell in supports}
    for override in args.cell:
        key, _, value = override.partition("=")
        rotation, _, turn = key.partition(":")
        cell = (int(rotation), turn)
        if cell not in counts:
            raise SystemExit(f"unknown cell {key}")
        counts[cell] = int(value)

    result = {
        "role": args.role,
        "source_bearing_half_width_degrees": half_width,
        "baseline_items_per_action_cell": base_n,
        "allocation": evaluate(counts, supports),
    }
    if args.scan_rotation is not None:
        scan = []
        for value in [int(v) for v in args.scan_values.split(",")]:
            trial = dict(counts)
            for cell in trial:
                if cell[0] == args.scan_rotation:
                    trial[cell] = value
            evaluated = evaluate(trial, supports)
            scan.append(
                {
                    "rotation": args.scan_rotation,
                    "items_per_presented_cell": value,
                    "total_items": evaluated["total_items"],
                    "finite_target_exact": evaluated["finite_target_exact"],
                    "finite_target": round(evaluated["finite_target"], 6),
                }
            )
        result["scan"] = scan
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
