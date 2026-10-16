"""Analytic reachable-answer support for action-conditioned bearing tasks.

The implementation deliberately operates only on a declared *source-bearing
envelope* and action geometry.  It does not infer that envelope from rendered
visibility, and it does not inspect question labels or benchmark answer counts
when deriving a support set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, isfinite
from typing import Iterable, Mapping


DEFAULT_ZONE_CENTERS: dict[str, float] = {
    "in front": 0.0,
    "to the right": 90.0,
    "behind": 180.0,
    "to the left": -90.0,
}


@dataclass(frozen=True)
class GeometryConfig:
    """Declared source-bearing/action/answer geometry.

    Bearings increase to the observer's right.  Consequently, a left camera
    rotation moves a stationary target toward positive relative bearings.
    ``source_bearing_half_width_degrees`` is a benchmark construction gate on
    the target center's horizontal-plane bearing in View 1.  It is not a claim
    about what a pitched camera can render and must be independently verified
    item by item.
    Zone boundaries are treated as measure-zero: a zone must overlap the
    reachable interval with positive angular width to enter the support.
    """

    source_bearing_half_width_degrees: float = 66.58677555362947
    zone_width_degrees: float = 90.0
    zone_centers_degrees: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_ZONE_CENTERS)
    )

    def __post_init__(self) -> None:
        if not isfinite(self.source_bearing_half_width_degrees) or not (
            0.0 < self.source_bearing_half_width_degrees < 180.0
        ):
            raise ValueError(
                "source_bearing_half_width_degrees must be finite and in (0, 180)"
            )
        if not isfinite(self.zone_width_degrees) or not (0.0 < self.zone_width_degrees <= 360.0):
            raise ValueError("zone_width_degrees must be finite and in (0, 360]")
        if not self.zone_centers_degrees:
            raise ValueError("at least one answer zone is required")
        if len(set(self.zone_centers_degrees)) != len(self.zone_centers_degrees):
            raise ValueError("answer-zone labels must be unique")
        for label, center in self.zone_centers_degrees.items():
            if not isinstance(label, str) or not label.strip():
                raise ValueError("answer-zone labels must be non-empty strings")
            if not isfinite(float(center)):
                raise ValueError(f"zone center for {label!r} must be finite")


def action_shift_degrees(rotation_degrees: float, turn_direction: str) -> float:
    """Return the change in target bearing induced by an in-place camera turn."""

    rotation = float(rotation_degrees)
    if not isfinite(rotation) or not (0.0 <= rotation <= 360.0):
        raise ValueError("rotation_degrees must be finite and in [0, 360]")
    direction = turn_direction.strip().lower()
    if direction == "left":
        return rotation
    if direction == "right":
        return -rotation
    raise ValueError("turn_direction must be 'left' or 'right'")


def _positive_interval_overlap(
    first_low: float,
    first_high: float,
    second_low: float,
    second_high: float,
    *,
    tolerance: float = 1e-12,
) -> bool:
    return min(first_high, second_high) - max(first_low, second_low) > tolerance


def analytic_support(
    rotation_degrees: float,
    turn_direction: str,
    geometry: GeometryConfig | None = None,
) -> tuple[str, ...]:
    """Derive all answer zones reachable from the declared source envelope.

    By construction, the initial target-center bearing lies in the explicit
    interval ``[-b, b]``.  The interval is shifted by the declared camera
    action and intersected with periodic copies of each answer zone.  Rendered
    visibility alone does not establish this precondition.  Touching at a
    single boundary point does not add a label to the population support.
    """

    geometry = geometry or GeometryConfig()
    half_width = geometry.source_bearing_half_width_degrees
    shift = action_shift_degrees(rotation_degrees, turn_direction)
    reachable_low = -half_width + shift
    reachable_high = half_width + shift
    half_zone = geometry.zone_width_degrees / 2.0

    support: list[str] = []
    for label, center_value in geometry.zone_centers_degrees.items():
        center = float(center_value)
        # Inputs are bounded to one revolution and the reachable interval is
        # shorter than one revolution.  Seven lifts are intentionally more
        # than needed and make non-normalized zone centers harmless.
        if any(
            _positive_interval_overlap(
                reachable_low,
                reachable_high,
                center + 360.0 * lift - half_zone,
                center + 360.0 * lift + half_zone,
            )
            for lift in range(-3, 4)
        ):
            support.append(label)
    return tuple(support)


def support_table(
    rotations_degrees: Iterable[float],
    turn_directions: Iterable[str] = ("left", "right"),
    geometry: GeometryConfig | None = None,
) -> dict[tuple[float, str], tuple[str, ...]]:
    """Return analytic support for every requested action cell."""

    geometry = geometry or GeometryConfig()
    table: dict[tuple[float, str], tuple[str, ...]] = {}
    for rotation in rotations_degrees:
        for direction in turn_directions:
            cell = (float(rotation), direction.strip().lower())
            table[cell] = analytic_support(cell[0], cell[1], geometry)
    return table


def population_target(
    cell_counts: Mapping[tuple[float, str], int],
    supports: Mapping[tuple[float, str], tuple[str, ...]],
) -> float:
    """Compute ``sum_z p(z)/|S(z)|`` for the empirical action mixture."""

    total = sum(cell_counts.values())
    if total <= 0:
        raise ValueError("cell_counts must contain at least one item")
    target = 0.0
    for cell, count in cell_counts.items():
        if count < 0:
            raise ValueError(f"negative item count for cell {cell!r}")
        if cell not in supports:
            raise ValueError(f"missing support for cell {cell!r}")
        support_size = len(supports[cell])
        if support_size == 0:
            raise ValueError(f"empty reachable support for cell {cell!r}")
        target += (count / total) / support_size
    return target


def finite_target_correct(
    cell_counts: Mapping[tuple[float, str], int],
    supports: Mapping[tuple[float, str], tuple[str, ...]],
) -> int:
    """Return the attainable finite construction target ``sum ceil(n_z/k_z)``."""

    correct = 0
    for cell, count in cell_counts.items():
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"cell count for {cell!r} must be a non-negative integer")
        if cell not in supports:
            raise ValueError(f"missing support for cell {cell!r}")
        support_size = len(supports[cell])
        if support_size == 0:
            raise ValueError(f"empty reachable support for cell {cell!r}")
        correct += ceil(count / support_size)
    return correct


def cell_construction_diagnostics(
    label_counts: Mapping[str, int], support: Iterable[str]
) -> dict[str, object]:
    """Compare one realized cell with its finite attainable target."""

    support_tuple = tuple(support)
    if not support_tuple:
        raise ValueError("support must not be empty")
    if len(set(support_tuple)) != len(support_tuple):
        raise ValueError("support labels must be unique")
    if any(not isinstance(value, int) or value < 0 for value in label_counts.values()):
        raise ValueError("label counts must be non-negative integers")

    total = sum(label_counts.values())
    counts_on_support = {label: int(label_counts.get(label, 0)) for label in support_tuple}
    unexpected = {
        label: count
        for label, count in label_counts.items()
        if label not in counts_on_support and count > 0
    }
    all_counts = list(label_counts.values()) or [0]
    empirical_oracle = max(all_counts)
    target = ceil(total / len(support_tuple))
    return {
        "items": total,
        "support": list(support_tuple),
        "support_size": len(support_tuple),
        "counts_on_support": counts_on_support,
        "unexpected_label_counts": unexpected,
        "finite_target_correct": target,
        "empirical_action_oracle_correct": empirical_oracle,
        "excess_correct": empirical_oracle - target,
        "attains_target": not unexpected and empirical_oracle == target,
        "deterministic_support": len(support_tuple) == 1,
    }
