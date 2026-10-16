"""Itemwise Task-B pose geometry independent of rendered visibility.

The formal support target is conditional on an explicit View-1 target-center
horizontal-bearing envelope.  This module computes the relevant bearing and
post-action gold from camera pose plus a 3D bounding box.  It deliberately does
not use masks, visible-pixel counts, stored answers, or camera field of view.
Those are separate evidence and quality gates.
"""

from __future__ import annotations

from math import atan2, cos, degrees, hypot, isfinite, radians, sin
from typing import Any, Iterable, Mapping, Sequence

from .support import action_shift_degrees


CANONICAL_LABELS = ("in front", "to the right", "behind", "to the left")


def wrap180(angle_degrees: float) -> float:
    """Wrap an angle to ``[-180, 180)`` with ``+180`` represented as ``-180``."""

    value = float(angle_degrees)
    if not isfinite(value):
        raise ValueError("angle must be finite")
    return (value + 180.0) % 360.0 - 180.0


def _coordinate(point: Mapping[str, Any] | Sequence[float], axis: str, index: int) -> float:
    try:
        value = point[axis] if isinstance(point, Mapping) else point[index]
        result = float(value)
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"point is missing finite {axis!r} coordinate") from exc
    if not isfinite(result):
        raise ValueError(f"point {axis!r} coordinate must be finite")
    return result


def box_center_xz(
    corners: Iterable[Mapping[str, Any] | Sequence[float]],
) -> tuple[float, float]:
    """Return the horizontal center of a 3D box from its corner points."""

    values = list(corners)
    if not values:
        raise ValueError("bounding box must contain at least one corner")
    xs = [_coordinate(point, "x", 0) for point in values]
    zs = [_coordinate(point, "z", 2) for point in values]
    return sum(xs) / len(xs), sum(zs) / len(zs)


def horizontal_bearing_degrees(
    *,
    camera_x: float,
    camera_z: float,
    camera_yaw_degrees: float,
    target_x: float,
    target_z: float,
) -> float:
    """Return target bearing in the camera yaw frame; positive means right."""

    values = (camera_x, camera_z, camera_yaw_degrees, target_x, target_z)
    if not all(isfinite(float(value)) for value in values):
        raise ValueError("camera and target coordinates/yaw must be finite")
    dx = float(target_x) - float(camera_x)
    dz = float(target_z) - float(camera_z)
    if hypot(dx, dz) <= 1e-12:
        raise ValueError("target center coincides with the camera in the horizontal plane")
    yaw = radians(float(camera_yaw_degrees))
    local_x = cos(yaw) * dx - sin(yaw) * dz
    local_z = sin(yaw) * dx + cos(yaw) * dz
    return wrap180(degrees(atan2(local_x, local_z)))


def box_center_bearing_degrees(
    *,
    camera_position: Mapping[str, Any],
    camera_yaw_degrees: float,
    corners: Iterable[Mapping[str, Any] | Sequence[float]],
) -> float:
    """Return horizontal bearing from camera pose to the 3D-box center."""

    target_x, target_z = box_center_xz(corners)
    return horizontal_bearing_degrees(
        camera_x=_coordinate(camera_position, "x", 0),
        camera_z=_coordinate(camera_position, "z", 2),
        camera_yaw_degrees=camera_yaw_degrees,
        target_x=target_x,
        target_z=target_z,
    )


def canonical_zone(bearing_degrees: float) -> str:
    """Assign the frozen half-open four-zone label.

    In ``[-180,180)`` coordinates the intervals are front ``[-45,45)``,
    right ``[45,135)``, behind ``[135,180) U [-180,-135)``, and left
    ``[-135,-45)``.
    """

    bearing = wrap180(bearing_degrees)
    if -45.0 <= bearing < 45.0:
        return "in front"
    if 45.0 <= bearing < 135.0:
        return "to the right"
    if -135.0 <= bearing < -45.0:
        return "to the left"
    return "behind"


def post_action_bearing_degrees(
    source_bearing_degrees: float,
    rotation_degrees: float,
    turn_direction: str,
) -> float:
    """Transform a stationary target bearing after an in-place camera turn."""

    return wrap180(
        float(source_bearing_degrees)
        + action_shift_degrees(rotation_degrees, turn_direction)
    )


def source_envelope_diagnostic(
    bearing_degrees: float,
    *,
    half_width_degrees: float = 66.58677555362947,
) -> dict[str, float | bool]:
    """Test strict membership in the declared symmetric source-bearing cone."""

    half_width = float(half_width_degrees)
    if not isfinite(half_width) or not (0.0 < half_width < 180.0):
        raise ValueError("half_width_degrees must be finite and in (0, 180)")
    bearing = wrap180(bearing_degrees)
    margin = half_width - abs(bearing)
    return {
        "bearing_degrees": bearing,
        "half_width_degrees": half_width,
        "margin_degrees": margin,
        "strictly_inside": margin > 0.0,
    }


def boundary_margin_degrees(bearing_degrees: float) -> float:
    """Return distance to the nearest frozen four-zone boundary."""

    bearing = wrap180(bearing_degrees)
    return min(abs(bearing - boundary) for boundary in (-135.0, -45.0, 45.0, 135.0))


def corner_post_diagnostic(
    *,
    camera_position: Mapping[str, Any],
    camera_yaw_degrees: float,
    corners: Iterable[Mapping[str, Any] | Sequence[float]],
    rotation_degrees: float,
    turn_direction: str,
    expected_answer: str,
) -> dict[str, Any]:
    """Verify that every 3D-box corner lands in one expected answer zone."""

    if expected_answer not in CANONICAL_LABELS:
        raise ValueError(f"unknown expected answer: {expected_answer!r}")
    corner_rows = list(corners)
    if not corner_rows:
        raise ValueError("bounding box must contain corners")
    post_bearings: list[float] = []
    labels: list[str] = []
    for corner in corner_rows:
        x = _coordinate(corner, "x", 0)
        z = _coordinate(corner, "z", 2)
        source = horizontal_bearing_degrees(
            camera_x=_coordinate(camera_position, "x", 0),
            camera_z=_coordinate(camera_position, "z", 2),
            camera_yaw_degrees=camera_yaw_degrees,
            target_x=x,
            target_z=z,
        )
        post = post_action_bearing_degrees(source, rotation_degrees, turn_direction)
        post_bearings.append(post)
        labels.append(canonical_zone(post))
    return {
        "post_corner_bearings_degrees": post_bearings,
        "corner_labels": labels,
        "all_corners_match_expected": all(label == expected_answer for label in labels),
        "minimum_corner_boundary_margin_degrees": min(
            boundary_margin_degrees(value) for value in post_bearings
        ),
    }


def item_pose_diagnostic(
    *,
    camera_position: Mapping[str, Any],
    camera_yaw_degrees: float,
    corners: Iterable[Mapping[str, Any] | Sequence[float]],
    rotation_degrees: float,
    turn_direction: str,
    expected_answer: str | None = None,
    source_half_width_degrees: float = 66.58677555362947,
) -> dict[str, Any]:
    """Compute source eligibility and post-action gold without stored-label input."""

    source = box_center_bearing_degrees(
        camera_position=camera_position,
        camera_yaw_degrees=camera_yaw_degrees,
        corners=corners,
    )
    post = post_action_bearing_degrees(source, rotation_degrees, turn_direction)
    derived_answer = canonical_zone(post)
    envelope = source_envelope_diagnostic(
        source, half_width_degrees=source_half_width_degrees
    )
    expected_valid = expected_answer is None or expected_answer in CANONICAL_LABELS
    if not expected_valid:
        raise ValueError(f"unknown expected answer: {expected_answer!r}")
    return {
        "source_center_bearing_degrees": source,
        "source_envelope": envelope,
        "rotation_degrees": float(rotation_degrees),
        "turn_direction": turn_direction.strip().lower(),
        "post_center_bearing_degrees": post,
        "derived_answer": derived_answer,
        "expected_answer": expected_answer,
        "answer_matches": expected_answer is None or derived_answer == expected_answer,
        "pass": bool(envelope["strictly_inside"])
        and (expected_answer is None or derived_answer == expected_answer),
    }
