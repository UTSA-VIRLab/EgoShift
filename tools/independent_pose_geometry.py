#!/usr/bin/env python3
"""Independent Task-B center-bearing and zone implementation.

This verifier intentionally imports no EgoShift geometry code.  It uses an
absolute world heading minus camera yaw, followed by a [0,360) zone partition,
which is algebraically equivalent but implementation-independent from the
camera-frame rotation used by ``egoshift.taskb_geometry``.
"""

from __future__ import annotations

from math import atan2, degrees
from typing import Iterable, Mapping, Sequence


def center_xz(corners: Iterable[Mapping[str, float] | Sequence[float]]) -> tuple[float, float]:
    rows = list(corners)
    if not rows:
        raise ValueError("no corners")
    xs = [float(row["x"] if isinstance(row, Mapping) else row[0]) for row in rows]
    zs = [float(row["z"] if isinstance(row, Mapping) else row[2]) for row in rows]
    return sum(xs) / len(xs), sum(zs) / len(zs)


def bearing(camera_x: float, camera_z: float, yaw: float, target_x: float, target_z: float) -> float:
    world_heading = degrees(atan2(target_x - camera_x, target_z - camera_z))
    return (world_heading - yaw + 180.0) % 360.0 - 180.0


def zone(angle: float) -> str:
    value = angle % 360.0
    if value >= 315.0 or value < 45.0:
        return "in front"
    if value < 135.0:
        return "to the right"
    if value < 225.0:
        return "behind"
    return "to the left"


def verify(
    *,
    camera_x: float,
    camera_z: float,
    yaw: float,
    corners: Iterable[Mapping[str, float] | Sequence[float]],
    rotation: float,
    direction: str,
    half_width: float,
) -> dict[str, object]:
    target_x, target_z = center_xz(corners)
    source = bearing(camera_x, camera_z, yaw, target_x, target_z)
    shift = rotation if direction.lower() == "left" else -rotation
    post = (source + shift + 180.0) % 360.0 - 180.0
    return {
        "source_bearing_degrees": source,
        "post_bearing_degrees": post,
        "answer": zone(post),
        "strictly_inside_source_envelope": abs(source) < half_width,
    }
