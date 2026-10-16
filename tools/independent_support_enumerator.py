#!/usr/bin/env python3
"""Independent point-enumeration check of reachable answer supports.

This file intentionally does not import ``egoshift.support``.  It uses dense
interior samples from an explicitly declared source-bearing envelope and
assigns each transformed bearing directly to angular zones.  Agreement with
the interval-intersection implementation therefore checks both the derivation
and its sign convention; it does not verify item membership in the envelope.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def circular_distance(first: float, second: float) -> float:
    return abs((first - second + 180.0) % 360.0 - 180.0)


def enumerate_support(config: dict[str, Any], rotation: float, direction: str) -> list[str]:
    source_half_width = float(config["source_bearing_half_width_degrees"])
    half_width = float(config["zone_width_degrees"]) / 2.0
    centers = config["zone_centers_degrees"]
    # Midpoints exclude the source-envelope endpoints.  200,000 bins give a maximum
    # angular spacing below 0.001 degrees for all supplied configurations.
    bins = 200_000
    shift = rotation if direction == "left" else -rotation
    found: set[str] = set()
    for index in range(bins):
        source_bearing = -source_half_width + (index + 0.5) * (2.0 * source_half_width) / bins
        transformed = source_bearing + shift
        for label, center in centers.items():
            if circular_distance(transformed, float(center)) < half_width:
                found.add(label)
    return [label for label in centers if label in found]


def run(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    reports: list[dict[str, Any]] = []
    for config in payload["configurations"]:
        cells: dict[str, Any] = {}
        for rotation_value in config["rotations_degrees"]:
            rotation = float(rotation_value)
            display = int(rotation) if rotation.is_integer() else rotation
            for raw_direction in config["turn_directions"]:
                direction = str(raw_direction).lower()
                if direction not in {"left", "right"}:
                    raise ValueError(f"invalid turn direction: {raw_direction!r}")
                support = enumerate_support(config, rotation, direction)
                cells[f"{direction}_{display}"] = {
                    "rotation_degrees": display,
                    "turn_direction": direction,
                    "support": support,
                    "support_size": len(support),
                }
        reports.append({"id": config["id"], "cells": cells})
    return {
        "schema_version": "0.1.0",
        "method": "independent_dense_midpoint_enumeration",
        "imports_analytic_implementation": False,
        "source_bins": 200_000,
        "configurations": reports,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(json.dumps(run(args.config), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
