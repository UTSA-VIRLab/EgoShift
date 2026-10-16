"""Audit Task-B action cells against supports from a declared bearing envelope.

This module audits action parsing and realized counts.  It does not establish
that an item's source target actually lies inside the declared envelope; the
independent pose/AABB replay gate must run before this report is release-valid.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .records import read_jsonl
from .support import (
    GeometryConfig,
    analytic_support,
    cell_construction_diagnostics,
    finite_target_correct,
    population_target,
)


ACTION_PATTERN = re.compile(
    r"\bturn\s+(left|right)\s+by\s+([0-9]+(?:\.[0-9]+)?)\s+degrees?\b",
    flags=re.IGNORECASE,
)


def parse_prompt_action(question: str) -> tuple[float, str]:
    """Parse the declared turn from the user-visible Task-B question."""

    matches = ACTION_PATTERN.findall(question)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one turn action in question, found {len(matches)}")
    direction, rotation = matches[0]
    return float(rotation), direction.lower()


def _display_rotation(rotation: float) -> int | float:
    return int(rotation) if rotation.is_integer() else rotation


def _cell_id(rotation: float, direction: str) -> str:
    return f"{direction}_{_display_rotation(rotation)}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_taskb_rows(
    rows: list[dict[str, Any]],
    geometry: GeometryConfig | None = None,
    *,
    expected_actions: set[tuple[float, str]] | None = None,
    expected_items_per_cell: int | None = None,
) -> dict[str, Any]:
    """Audit action parsing, reachable labels, and finite construction balance."""

    geometry = geometry or GeometryConfig()
    cell_labels: dict[tuple[float, str], Counter[str]] = defaultdict(Counter)
    parsing_errors: list[dict[str, str]] = []
    metadata_rotation_mismatches: list[str] = []
    canonical_turn_missing_ids: list[str] = []
    canonical_turn_mismatch_ids: list[str] = []
    legacy_turn_values: Counter[str] = Counter()
    duplicate_ids: list[str] = []
    seen_ids: set[str] = set()

    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("id", f"row-{index}"))
        if row_id in seen_ids:
            duplicate_ids.append(row_id)
        seen_ids.add(row_id)
        if row.get("task_family") != "B":
            parsing_errors.append({"id": row_id, "error": "task_family is not B"})
            continue
        try:
            rotation, direction = parse_prompt_action(str(row.get("question", "")))
        except ValueError as exc:
            parsing_errors.append({"id": row_id, "error": str(exc)})
            continue

        metadata = row.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        declared_rotation = metadata.get("rotation_degrees")
        try:
            rotation_matches = float(declared_rotation) == rotation
        except (TypeError, ValueError):
            rotation_matches = False
        if not rotation_matches:
            metadata_rotation_mismatches.append(row_id)
        canonical_turn = metadata.get("turn_direction")
        if canonical_turn is None:
            canonical_turn_missing_ids.append(row_id)
        elif str(canonical_turn).strip().lower() != direction:
            canonical_turn_mismatch_ids.append(row_id)
        legacy_turn_values[str(metadata.get("qb2_orig_turn", "<missing>"))] += 1
        answer = str(row.get("answer", ""))
        cell_labels[(rotation, direction)][answer] += 1

    supports = {
        cell: analytic_support(cell[0], cell[1], geometry) for cell in sorted(cell_labels)
    }
    cell_counts = {cell: sum(counts.values()) for cell, counts in cell_labels.items()}
    cells: dict[str, Any] = {}
    invalid_support_items = 0
    empirical_oracle_correct = 0
    excess_correct = 0
    deterministic_cells: list[str] = []
    for cell in sorted(cell_labels):
        diagnostic = cell_construction_diagnostics(cell_labels[cell], supports[cell])
        cell_name = _cell_id(*cell)
        diagnostic["rotation_degrees"] = _display_rotation(cell[0])
        diagnostic["turn_direction"] = cell[1]
        cells[cell_name] = diagnostic
        invalid_support_items += sum(diagnostic["unexpected_label_counts"].values())
        empirical_oracle_correct += int(diagnostic["empirical_action_oracle_correct"])
        excess_correct += int(diagnostic["excess_correct"])
        if diagnostic["deterministic_support"]:
            deterministic_cells.append(cell_name)

    total = sum(cell_counts.values())
    finite_correct = finite_target_correct(cell_counts, supports) if total else 0
    population = population_target(cell_counts, supports) if total else None
    all_cells_attain = bool(cells) and all(cell["attains_target"] for cell in cells.values())

    merged_180_counts: Counter[str] = Counter()
    for cell, counts in cell_labels.items():
        if cell[0] == 180.0:
            merged_180_counts.update(counts)
    merged_180 = None
    if merged_180_counts:
        left_support = analytic_support(180.0, "left", geometry)
        right_support = analytic_support(180.0, "right", geometry)
        merged_180 = cell_construction_diagnostics(merged_180_counts, left_support)
        merged_180["left_right_supports_identical"] = left_support == right_support

    observed_actions = set(cell_counts)
    expected_actions = set(expected_actions) if expected_actions is not None else None
    missing_actions = sorted(expected_actions - observed_actions) if expected_actions is not None else []
    unexpected_actions = sorted(observed_actions - expected_actions) if expected_actions is not None else []
    count_mismatches: dict[str, dict[str, int]] = {}
    if expected_items_per_cell is not None:
        if expected_items_per_cell < 0:
            raise ValueError("expected_items_per_cell must be non-negative")
        cells_to_check = expected_actions if expected_actions is not None else observed_actions
        for cell in sorted(cells_to_check):
            observed_count = cell_counts.get(cell, 0)
            if observed_count != expected_items_per_cell:
                count_mismatches[_cell_id(*cell)] = {
                    "expected": expected_items_per_cell,
                    "observed": observed_count,
                }
    return {
        "schema_version": "0.1.0",
        "audit": "task_b_geometry_support_and_finite_target",
        "items_input": len(rows),
        "items_audited": total,
        "distinct_ids": len(seen_ids),
        "duplicate_ids": duplicate_ids,
        "action_source": "user_visible_question",
        "geometry": {
            "source_bearing_half_width_degrees": geometry.source_bearing_half_width_degrees,
            "source_envelope_semantics": "explicit_target_center_horizontal_bearing_gate_not_render_visibility",
            "zone_width_degrees": geometry.zone_width_degrees,
            "zone_centers_degrees": dict(geometry.zone_centers_degrees),
            "bearing_sign_convention": "positive_is_to_the_right",
            "left_turn_target_shift_sign": "positive",
            "boundary_policy": "positive_measure_overlap_only",
        },
        "parsing_errors": parsing_errors,
        "metadata_rotation_mismatch_ids": metadata_rotation_mismatches,
        "canonical_turn_direction": {
            "field": "metadata.turn_direction",
            "missing_count": len(canonical_turn_missing_ids),
            "missing_ids_preview": canonical_turn_missing_ids[:20],
            "mismatch_ids": canonical_turn_mismatch_ids,
            "release_requirement": "materialize and freeze this field during S05",
        },
        "legacy_qb2_orig_turn_values": dict(sorted(legacy_turn_values.items())),
        "action_contract": {
            "expected_cell_ids": (
                [_cell_id(*cell) for cell in sorted(expected_actions)]
                if expected_actions is not None
                else None
            ),
            "observed_cell_ids": [_cell_id(*cell) for cell in sorted(observed_actions)],
            "missing_cell_ids": [_cell_id(*cell) for cell in missing_actions],
            "unexpected_cell_ids": [_cell_id(*cell) for cell in unexpected_actions],
            "expected_items_per_cell": expected_items_per_cell,
            "item_count_mismatches": count_mismatches,
        },
        "cells": cells,
        "merged_180_degree_cell": merged_180,
        "structural_population_balance_target": population,
        "finite_structural_balance_target_correct": finite_correct,
        "finite_structural_balance_target_total": total,
        "finite_structural_balance_target_accuracy": finite_correct / total if total else None,
        "empirical_action_oracle_correct": empirical_oracle_correct,
        "empirical_action_oracle_accuracy": empirical_oracle_correct / total if total else None,
        "total_excess_correct": excess_correct,
        "invalid_support_items": invalid_support_items,
        "deterministic_cells": deterministic_cells,
        "all_cells_attain_finite_target": all_cells_attain,
        "pass": (
            not parsing_errors
            and not duplicate_ids
            and not metadata_rotation_mismatches
            and not canonical_turn_mismatch_ids
            and not missing_actions
            and not unexpected_actions
            and not count_mismatches
            and invalid_support_items == 0
            and not deterministic_cells
            and all_cells_attain
            and empirical_oracle_correct == finite_correct
        ),
    }


def audit_taskb_file(
    path: str | Path,
    geometry: GeometryConfig | None = None,
    *,
    expected_sha256: str | None = None,
    expected_actions: set[tuple[float, str]] | None = None,
    expected_items_per_cell: int | None = None,
) -> dict[str, Any]:
    """Read and audit a Task-B file, optionally enforcing its frozen hash."""

    input_path = Path(path)
    observed_sha256 = _sha256(input_path)
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise ValueError(
            f"Task-B SHA-256 mismatch: expected {expected_sha256}, observed {observed_sha256}"
        )
    report = audit_taskb_rows(
        read_jsonl(input_path),
        geometry,
        expected_actions=expected_actions,
        expected_items_per_cell=expected_items_per_cell,
    )
    report["input"] = {
        "path": str(input_path),
        "sha256": observed_sha256,
        "expected_sha256": expected_sha256,
        "hash_matches": expected_sha256 is None or observed_sha256 == expected_sha256,
    }
    return report
