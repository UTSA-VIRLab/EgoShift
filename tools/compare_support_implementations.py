#!/usr/bin/env python3
"""Compare analytic and independently enumerated action supports."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

from egoshift.support import GeometryConfig, analytic_support


def _load_enumerator(path: Path):
    spec = importlib.util.spec_from_file_location("egoshift_independent_enumerator", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load independent verifier at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(config_path: Path, enumerator_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    independent = _load_enumerator(enumerator_path).run(config_path)
    independent_by_id = {row["id"]: row for row in independent["configurations"]}

    configurations: list[dict[str, Any]] = []
    total_cells = 0
    disagreements: list[str] = []
    for config in payload["configurations"]:
        geometry = GeometryConfig(
            source_bearing_half_width_degrees=float(
                config["source_bearing_half_width_degrees"]
            ),
            zone_width_degrees=float(config["zone_width_degrees"]),
            zone_centers_degrees=config["zone_centers_degrees"],
        )
        cells: dict[str, Any] = {}
        deterministic: list[str] = []
        inverse_support_sum = 0.0
        action_count = 0
        for rotation_value in config["rotations_degrees"]:
            rotation = float(rotation_value)
            display = int(rotation) if rotation.is_integer() else rotation
            for direction in config["turn_directions"]:
                action_count += 1
                total_cells += 1
                cell_id = f"{direction}_{display}"
                analytic = list(analytic_support(rotation, direction, geometry))
                enumerated = independent_by_id[config["id"]]["cells"][cell_id]["support"]
                agrees = analytic == enumerated
                if not agrees:
                    disagreements.append(f"{config['id']}:{cell_id}")
                if len(analytic) == 1:
                    deterministic.append(cell_id)
                if not analytic:
                    raise ValueError(f"empty support for {config['id']}:{cell_id}")
                inverse_support_sum += 1.0 / len(analytic)
                cells[cell_id] = {
                    "analytic_support": analytic,
                    "enumerated_support": enumerated,
                    "support_size": len(analytic),
                    "agrees": agrees,
                }
        configurations.append(
            {
                "id": config["id"],
                "cells": cells,
                "deterministic_cells": deterministic,
                "equal_action_mixture_structural_target": inverse_support_sum / action_count,
                "all_cells_agree": all(cell["agrees"] for cell in cells.values()),
            }
        )
    return {
        "schema_version": "0.1.0",
        "audit": "independent_support_implementation_agreement",
        "analytic_method": "periodic_positive_measure_interval_intersection",
        "independent_method": independent["method"],
        "independent_source_bins": independent["source_bins"],
        "tested_configurations": len(configurations),
        "tested_action_cells": total_cells,
        "disagreements": disagreements,
        "pass": not disagreements,
        "configurations": configurations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--enumerator", required=True, type=Path)
    args = parser.parse_args()
    report = compare(args.config, args.enumerator)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
