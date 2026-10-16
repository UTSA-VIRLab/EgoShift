"""Propose camera poses that should fully occlude a target, by construction.

The trajectory-template miner finds occlusions by accident: it plays fixed
ego-motion paths and checks whether a target happens to end up completely
behind furniture. In procedurally furnished rooms that happens for roughly six
targets in a hundred, which is far too rare to build a balanced class from.

Full occlusion is not accidental, though -- it is a solvable geometry problem.
For a target ``T`` and a blocker ``O`` that stands between the camera and the
target, the camera positions from which ``O`` completely covers ``T`` form a
region along the ray that runs from ``T`` through ``O`` and outward. This
module computes that region and proposes camera poses inside it.

Nothing here decides whether an item is occluded. Every proposal is still
rendered and put through the same live evidence gates as before: an exactly
empty target mask, a projected target region, a closer non-structural blocker
overlapping it, and no same-type distractor. This is search, not adjudication.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .taskc_source import STRUCTURAL_OCCLUDER_PREFIXES


@dataclass(frozen=True)
class OcclusionSearchConfig:
    """Search parameters. None of these is an acceptance gate."""

    # Blocker must be at least this much wider than the target, angularly, at
    # the proposed camera position. 1.0 means "just covers"; the margin buys
    # tolerance for a real mesh being smaller than its bounding box.
    minimum_angular_ratio: float = 1.35
    # How far behind the blocker to stand, as multiples of the blocker radius.
    standoff_multiples: tuple[float, ...] = (1.6, 2.4, 3.4)
    minimum_camera_distance: float = 0.8
    maximum_camera_distance: float = 6.0
    # Target and blocker must be meaningfully separated, or no camera position
    # sits cleanly "behind" the blocker.
    minimum_separation_meters: float = 0.35
    maximum_separation_meters: float = 6.0
    # Vertical: the blocker has to be tall enough to cover, not just wide.
    minimum_height_ratio: float = 0.75
    maximum_proposals_per_target: int = 12
    # An occlusion is only an *object* occlusion if the camera and the target
    # share a room. Across a wall the real blocker is the wall, whatever
    # furniture happens to sit in the line of sight.
    require_same_room: bool = True
    # Ranking preference, not a gate. Maximising the angular ratio selects
    # tiny targets behind huge blockers: trivial items that barely clear the
    # source-pixel floor and repeat the same few object types.
    preferred_angular_ratio: float = 2.5


def _xz(position: Mapping[str, Any]) -> tuple[float, float]:
    return float(position["x"]), float(position["z"])


def _box(record: Mapping[str, Any]) -> tuple[Mapping[str, float], float, float] | None:
    """Return (centre, horizontal radius, half height) for an object record."""

    box = record.get("axisAlignedBoundingBox") or {}
    centre = box.get("center") or record.get("position")
    if not isinstance(centre, Mapping):
        return None
    size = box.get("size")
    if isinstance(size, Mapping):
        radius = math.hypot(float(size["x"]), float(size["z"])) / 2.0
        half_height = float(size["y"]) / 2.0
        return centre, radius, half_height
    corners = box.get("cornerPoints")
    if isinstance(corners, list) and len(corners) >= 8:
        best = 0.0
        low = high = None
        for corner in corners[:8]:
            if isinstance(corner, Mapping):
                cx, cy, cz = float(corner["x"]), float(corner["y"]), float(corner["z"])
            else:
                cx, cy, cz = float(corner[0]), float(corner[1]), float(corner[2])
            best = max(best, math.hypot(cx - float(centre["x"]), cz - float(centre["z"])))
            low = cy if low is None else min(low, cy)
            high = cy if high is None else max(high, cy)
        if best <= 0.0 or low is None:
            return None
        return centre, best, (high - low) / 2.0
    return None


def _is_structural(object_id: str) -> bool:
    return str(object_id).casefold().startswith(STRUCTURAL_OCCLUDER_PREFIXES)


def _point_in_polygon(x: float, z: float, polygon: Sequence[Mapping[str, Any]]) -> bool:
    """Standard ray-crossing test in the xz plane."""

    inside = False
    count = len(polygon)
    for index in range(count):
        x0, z0 = float(polygon[index]["x"]), float(polygon[index]["z"])
        x1, z1 = float(polygon[(index + 1) % count]["x"]), float(
            polygon[(index + 1) % count]["z"]
        )
        if (z0 > z) != (z1 > z):
            crossing = x0 + (z - z0) * (x1 - x0) / (z1 - z0)
            if x < crossing:
                inside = not inside
    return inside


def room_of(
    x: float, z: float, rooms: Sequence[Mapping[str, Any]]
) -> str | None:
    """Which room contains this ground-plane point, if any."""

    for room in rooms:
        polygon = room.get("floorPolygon")
        if (
            isinstance(polygon, list)
            and len(polygon) >= 3
            and _point_in_polygon(x, z, polygon)
        ):
            return str(room.get("id") or "")
    return None


def _nearest_reachable(
    point: tuple[float, float],
    reachable: Sequence[Mapping[str, float]],
    limit: float,
) -> Mapping[str, float] | None:
    best = None
    best_distance = limit
    for position in reachable:
        px, pz = _xz(position)
        distance = math.hypot(px - point[0], pz - point[1])
        if distance < best_distance:
            best, best_distance = position, distance
    return best


def propose_occluding_poses(
    objects: Sequence[Mapping[str, Any]],
    reachable: Sequence[Mapping[str, float]],
    target_oid: str,
    *,
    config: OcclusionSearchConfig | None = None,
    snap_tolerance_meters: float = 0.6,
    rooms: Sequence[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    """Camera poses from which some object should completely hide the target.

    Returns proposals ordered by how comfortably the blocker covers the target,
    so the most promising renders are attempted first.
    """

    settings = config or OcclusionSearchConfig()
    records = {str(item.get("objectId")): item for item in objects if item.get("objectId")}
    target = records.get(target_oid)
    if target is None:
        return []
    target_box = _box(target)
    if target_box is None:
        return []
    target_centre, target_radius, target_half_height = target_box
    if target_radius <= 0.0:
        return []
    target_type = target_oid.split("|", 1)[0]
    tx, tz = _xz(target_centre)
    room_list = list(rooms or ())
    target_room = room_of(tx, tz, room_list) if room_list else None

    proposals: list[dict[str, Any]] = []
    for occluder_oid, record in records.items():
        if occluder_oid == target_oid or _is_structural(occluder_oid):
            continue
        # A same-type blocker would also be a referent distractor.
        if occluder_oid.split("|", 1)[0] == target_type:
            continue
        occluder_box = _box(record)
        if occluder_box is None:
            continue
        occluder_centre, occluder_radius, occluder_half_height = occluder_box
        if occluder_radius <= 0.0:
            continue
        if occluder_half_height < settings.minimum_height_ratio * target_half_height:
            continue
        ox, oz = _xz(occluder_centre)
        separation = math.hypot(ox - tx, oz - tz)
        if not (
            settings.minimum_separation_meters
            <= separation
            <= settings.maximum_separation_meters
        ):
            continue
        # Unit vector pointing from the target out through the blocker.
        ux, uz = (ox - tx) / separation, (oz - tz) / separation
        for multiple in settings.standoff_multiples:
            standoff = occluder_radius * multiple
            camera_x, camera_z = ox + ux * standoff, oz + uz * standoff
            to_target = separation + standoff
            if not (
                settings.minimum_camera_distance
                <= to_target
                <= settings.maximum_camera_distance
            ):
                continue
            # Angular half-widths from the proposed camera position.
            occluder_angle = math.atan2(occluder_radius, standoff)
            target_angle = math.atan2(target_radius, to_target)
            if target_angle <= 0.0:
                continue
            ratio = occluder_angle / target_angle
            if ratio < settings.minimum_angular_ratio:
                continue
            position = _nearest_reachable(
                (camera_x, camera_z), reachable, snap_tolerance_meters
            )
            if position is None:
                continue
            px, pz = _xz(position)
            if settings.require_same_room and room_list:
                if target_room is None or room_of(px, pz, room_list) != target_room:
                    continue
            yaw = math.degrees(math.atan2(tx - px, tz - pz)) % 360.0
            proposals.append(
                {
                    "occluder_id": occluder_oid,
                    "position": dict(position),
                    "yaw": yaw,
                    "angular_ratio": round(ratio, 4),
                    "standoff_meters": round(standoff, 4),
                    "camera_to_target_meters": round(
                        math.hypot(tx - px, tz - pz), 4
                    ),
                    "target_blocker_separation_meters": round(separation, 4),
                    "snap_offset_meters": round(
                        math.hypot(px - camera_x, pz - camera_z), 4
                    ),
                    "room_id": target_room,
                }
            )
    # Best coverage first, then the closest snap, then a stable tiebreak.
    proposals.sort(
        key=lambda item: (
            abs(item["angular_ratio"] - settings.preferred_angular_ratio),
            item["snap_offset_meters"],
            item["occluder_id"],
        )
    )
    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for proposal in proposals:
        if proposal["occluder_id"] in seen:
            continue
        seen.add(proposal["occluder_id"])
        deduped.append(proposal)
        if len(deduped) >= settings.maximum_proposals_per_target:
            break
    return deduped
