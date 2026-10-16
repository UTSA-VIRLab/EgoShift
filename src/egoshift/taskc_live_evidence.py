"""Live-render evidence primitives for targeted Task-C occlusion mining.

Nothing in this module consumes a stored visibility label.  A full-occlusion
transition is derived from an exact source instance mask, an exact zero target
mask, live camera/object geometry, and overlap between a projected target AABB
region and a closer non-structural object's exact target-frame mask.

NumPy and Pillow are development-only dependencies imported lazily by the
functions that need them; the benchmark scoring runtime remains dependency
free.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .hashing import canonical_json_bytes, sha256_bytes
from .taskc_source import (
    EDGE_TOLERANCE_PIXELS,
    MINIMUM_CORNER_PIXELS,
    MINIMUM_OCCLUDER_OVERLAP,
    MINIMUM_ONE_EDGE_PIXELS,
    MINIMUM_SOURCE_PIXELS,
    MINIMUM_VISIBLE_TARGET_PIXELS,
    STRUCTURAL_OCCLUDER_PREFIXES,
)


class LiveEvidenceRejected(ValueError):
    """Expected fail-closed rejection from live-render evidence."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class LiveEvidenceConfig:
    minimum_source_pixels: int = MINIMUM_SOURCE_PIXELS
    minimum_occluder_overlap: float = MINIMUM_OCCLUDER_OVERLAP
    edge_tolerance_pixels: int = EDGE_TOLERANCE_PIXELS
    minimum_one_edge_pixels: int = MINIMUM_ONE_EDGE_PIXELS
    minimum_corner_pixels: int = MINIMUM_CORNER_PIXELS
    foreground_depth_margin: float = 0.05
    minimum_visible_target_pixels: int = MINIMUM_VISIBLE_TARGET_PIXELS


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for live mask evidence") from exc
    return np


def _color_triplet(color: Any) -> tuple[int, int, int] | None:
    if isinstance(color, Mapping):
        try:
            return int(color["r"]), int(color["g"]), int(color["b"])
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(color, Sequence) and not isinstance(color, (str, bytes)) and len(color) >= 3:
        try:
            return int(color[0]), int(color[1]), int(color[2])
        except (TypeError, ValueError):
            return None
    return None


def _metadata_object_ids(event: Any) -> set[str]:
    metadata = getattr(event, "metadata", None) or {}
    return {
        str(item.get("objectId"))
        for item in metadata.get("objects", [])
        if isinstance(item, Mapping) and item.get("objectId")
    }


def exact_object_mask(event: Any, object_id: str):
    """Return the exact instance mask, never a same-type substitute.

    If the exact object exists in live metadata but has no segmentation color,
    the result is a proven all-zero mask.  If the object identity disappeared
    from live metadata, replay is rejected instead of guessing an alias.
    """

    np = _numpy()
    segmentation = getattr(event, "instance_segmentation_frame", None)
    if segmentation is None:
        raise LiveEvidenceRejected("missing_instance_segmentation")
    segmentation = np.asarray(segmentation)
    if segmentation.ndim != 3 or segmentation.shape[2] < 3:
        raise LiveEvidenceRejected("invalid_instance_segmentation_shape")
    if object_id not in _metadata_object_ids(event):
        raise LiveEvidenceRejected("exact_object_id_absent", object_id)

    color = None
    id_to_color = getattr(event, "object_id_to_color", None) or {}
    if object_id in id_to_color:
        color = _color_triplet(id_to_color[object_id])
    if color is None:
        color_to_id = getattr(event, "color_to_object_id", None) or {}
        for raw_color, mapped_id in color_to_id.items():
            if mapped_id == object_id:
                color = _color_triplet(raw_color)
                if color is not None:
                    break
    if color is None:
        return np.zeros(segmentation.shape[:2], dtype=bool)
    target = np.asarray(color, dtype=segmentation.dtype)
    return np.all(segmentation[..., :3] == target, axis=-1)


def exact_mask_bbox(mask: Any) -> list[int] | None:
    np = _numpy()
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise LiveEvidenceRejected("invalid_mask_shape")
    locations = np.argwhere(array)
    if locations.size == 0:
        return None
    y0, x0 = locations.min(axis=0)
    y1, x1 = locations.max(axis=0) + 1
    return [int(x0), int(y0), int(x1), int(y1)]


def exact_mask_sha256(mask: Any) -> str:
    np = _numpy()
    array = np.asarray(mask, dtype=bool)
    if array.ndim != 2:
        raise LiveEvidenceRejected("invalid_mask_shape")
    header = canonical_json_bytes(
        {
            "encoding": "numpy_packbits",
            "bitorder": "little",
            "height": int(array.shape[0]),
            "width": int(array.shape[1]),
        }
    )
    packed = np.packbits(array.reshape(-1), bitorder="little").tobytes()
    return hashlib.sha256(header + b"\n" + packed).hexdigest()


def _object_metadata(event: Any, object_id: str) -> Mapping[str, Any]:
    metadata = getattr(event, "metadata", None) or {}
    matches = [
        item
        for item in metadata.get("objects", [])
        if isinstance(item, Mapping) and item.get("objectId") == object_id
    ]
    if len(matches) != 1:
        raise LiveEvidenceRejected(
            "missing_or_ambiguous_live_object", f"{object_id}: matches={len(matches)}"
        )
    return matches[0]


def _camera(event: Any) -> tuple[Mapping[str, Any], dict[str, float]]:
    metadata = getattr(event, "metadata", None) or {}
    agent = metadata.get("agent") or {}
    position = metadata.get("cameraPosition") or agent.get("position")
    rotation = agent.get("rotation") or {}
    if not isinstance(position, Mapping):
        raise LiveEvidenceRejected("missing_live_camera_position")
    try:
        camera = {
            "x": float(position["x"]),
            "y": float(position["y"]),
            "z": float(position["z"]),
        }
        camera_rotation = {
            "x": float(metadata.get("cameraHorizon", agent.get("cameraHorizon", 0.0))),
            "y": float(rotation.get("y", 0.0)),
            "z": float(rotation.get("z", 0.0)),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise LiveEvidenceRejected("invalid_live_camera_pose", str(exc)) from exc
    return camera, camera_rotation


def _world_to_camera(
    point: Mapping[str, Any],
    camera: Mapping[str, float],
    rotation: Mapping[str, float],
) -> tuple[float, float, float]:
    try:
        dx = float(point["x"]) - camera["x"]
        dy = float(point["y"]) - camera["y"]
        dz = float(point["z"]) - camera["z"]
    except (KeyError, TypeError, ValueError) as exc:
        raise LiveEvidenceRejected("invalid_live_object_position", str(exc)) from exc
    yaw = math.radians(rotation["y"])
    pitch = math.radians(rotation["x"])
    cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
    x1 = cos_y * dx + sin_y * dz
    z1 = -sin_y * dx + cos_y * dz
    cos_p, sin_p = math.cos(-pitch), math.sin(-pitch)
    y2 = cos_p * dy - sin_p * z1
    z2 = sin_p * dy + cos_p * z1
    return x1, y2, z2


def _aabb_corners(record: Mapping[str, Any]) -> list[Mapping[str, float]]:
    aabb = record.get("axisAlignedBoundingBox")
    if not isinstance(aabb, Mapping):
        raise LiveEvidenceRejected("missing_live_target_aabb")
    corners = aabb.get("cornerPoints")
    if isinstance(corners, list) and len(corners) >= 8:
        normalized: list[Mapping[str, float]] = []
        for point in corners[:8]:
            if isinstance(point, Mapping):
                normalized.append(point)
            elif isinstance(point, Sequence) and len(point) >= 3:
                normalized.append({"x": point[0], "y": point[1], "z": point[2]})
        if len(normalized) == 8:
            return normalized
    center, size = aabb.get("center"), aabb.get("size")
    if not isinstance(center, Mapping) or not isinstance(size, Mapping):
        raise LiveEvidenceRejected("invalid_live_target_aabb")
    try:
        cx, cy, cz = (float(center[axis]) for axis in ("x", "y", "z"))
        hx, hy, hz = (float(size[axis]) / 2.0 for axis in ("x", "y", "z"))
    except (KeyError, TypeError, ValueError) as exc:
        raise LiveEvidenceRejected("invalid_live_target_aabb", str(exc)) from exc
    return [
        {"x": cx + sx * hx, "y": cy + sy * hy, "z": cz + sz * hz}
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    ]


def _convex_hull(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    values = sorted(set(points))
    if len(values) <= 2:
        return values

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in values:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(values):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def projected_aabb_region_mask(event: Any, object_id: str, vertical_fov_degrees: float):
    """Rasterize the live 3D-AABB convex hull in the current image frame."""

    np = _numpy()
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required for projected-region evidence") from exc
    segmentation = np.asarray(getattr(event, "instance_segmentation_frame", None))
    if segmentation.ndim != 3:
        raise LiveEvidenceRejected("missing_instance_segmentation")
    height, width = segmentation.shape[:2]
    camera, rotation = _camera(event)
    record = _object_metadata(event, object_id)
    corners = _aabb_corners(record)
    fov = float(vertical_fov_degrees)
    if not math.isfinite(fov) or not 0.0 < fov < 180.0:
        raise LiveEvidenceRejected("invalid_live_fov")
    half_vertical = math.radians(fov / 2.0)
    half_horizontal = math.atan(math.tan(half_vertical) * (width / height))
    projected: list[tuple[float, float]] = []
    depths: list[float] = []
    for corner in corners:
        x, y, z = _world_to_camera(corner, camera, rotation)
        if z <= 0.05:
            raise LiveEvidenceRejected("target_aabb_crosses_near_plane")
        projected.append(
            (
                x / (z * math.tan(half_horizontal)) * width / 2.0 + width / 2.0,
                -y / (z * math.tan(half_vertical)) * height / 2.0 + height / 2.0,
            )
        )
        depths.append(z)
    hull = _convex_hull(projected)
    if len(hull) < 3:
        raise LiveEvidenceRejected("degenerate_projected_target_region")
    canvas = Image.new("1", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(hull, fill=1)
    region = np.asarray(canvas, dtype=bool)
    if not bool(region.any()):
        raise LiveEvidenceRejected("projected_target_region_outside_frame")
    target_position = record.get("position")
    if not isinstance(target_position, Mapping):
        raise LiveEvidenceRejected("missing_live_object_position")
    target_local = _world_to_camera(target_position, camera, rotation)
    center_inside = (
        target_local[2] > 0.05
        and abs(math.atan2(target_local[0], target_local[2])) <= half_horizontal
        and abs(math.atan2(target_local[1], target_local[2])) <= half_vertical
    )
    if not center_inside:
        raise LiveEvidenceRejected("target_center_outside_live_frustum")
    return region, {
        "method": "live_3d_aabb_convex_hull_x_exact_instance_masks",
        "region_pixels": int(region.sum()),
        "region_sha256": exact_mask_sha256(region),
        "hull_xy": [[round(x, 6), round(y, 6)] for x, y in hull],
        "target_center_depth": round(float(target_local[2]), 8),
    }


def _edge_guard(
    bbox: Sequence[int], pixels: int, width: int, height: int, config: LiveEvidenceConfig
) -> dict[str, int]:
    x0, y0, x1, y1 = bbox
    edges = sum(
        (
            x0 <= config.edge_tolerance_pixels,
            y0 <= config.edge_tolerance_pixels,
            x1 >= width - config.edge_tolerance_pixels,
            y1 >= height - config.edge_tolerance_pixels,
        )
    )
    minimum = (
        config.minimum_corner_pixels
        if edges >= 2
        else config.minimum_one_edge_pixels
        if edges == 1
        else 0
    )
    if pixels < minimum:
        raise LiveEvidenceRejected(
            "source_clipped_view_guard",
            f"edges={edges}, pixels={pixels}, required={minimum}",
        )
    return {"edges_touched": edges, "minimum_pixels": minimum}


def derive_live_full_occlusion(
    source_event: Any,
    target_event: Any,
    target_oid: str,
    *,
    vertical_fov_degrees: float,
    config: LiveEvidenceConfig | None = None,
) -> dict[str, Any]:
    """Derive full occlusion from live pixels and geometry only."""

    np = _numpy()
    settings = config or LiveEvidenceConfig()
    source_mask = exact_object_mask(source_event, target_oid)
    target_mask = exact_object_mask(target_event, target_oid)
    source_pixels = int(source_mask.sum())
    target_pixels = int(target_mask.sum())
    if source_pixels < settings.minimum_source_pixels:
        raise LiveEvidenceRejected(
            "source_pixels_below_floor",
            f"{source_pixels}<{settings.minimum_source_pixels}",
        )
    if target_pixels != 0:
        raise LiveEvidenceRejected("target_mask_not_zero", str(target_pixels))
    bbox = exact_mask_bbox(source_mask)
    if bbox is None:
        raise LiveEvidenceRejected("missing_source_bbox")
    height, width = source_mask.shape
    edge = _edge_guard(bbox, source_pixels, width, height, settings)
    region, projection = projected_aabb_region_mask(
        target_event, target_oid, vertical_fov_degrees
    )

    target_type = target_oid.split("|", 1)[0]
    same_type_pixels = 0
    for other_oid in sorted(_metadata_object_ids(target_event)):
        if other_oid != target_oid and other_oid.split("|", 1)[0] == target_type:
            same_type_pixels += int(exact_object_mask(target_event, other_oid).sum())
    if same_type_pixels:
        raise LiveEvidenceRejected("same_type_distractor_at_target", str(same_type_pixels))

    camera, rotation = _camera(target_event)
    target_record = _object_metadata(target_event, target_oid)
    target_position = target_record.get("position")
    if not isinstance(target_position, Mapping):
        raise LiveEvidenceRejected("missing_live_object_position")
    target_depth = _world_to_camera(target_position, camera, rotation)[2]
    region_pixels = int(region.sum())
    overlaps: list[dict[str, Any]] = []
    for other_oid in sorted(_metadata_object_ids(target_event)):
        if other_oid == target_oid or other_oid.casefold().startswith(
            STRUCTURAL_OCCLUDER_PREFIXES
        ):
            continue
        other_record = _object_metadata(target_event, other_oid)
        other_position = other_record.get("position")
        if not isinstance(other_position, Mapping):
            continue
        other_depth = _world_to_camera(other_position, camera, rotation)[2]
        if other_depth <= 0.05 or other_depth + settings.foreground_depth_margin >= target_depth:
            continue
        other_mask = exact_object_mask(target_event, other_oid)
        intersection = int(np.logical_and(region, other_mask).sum())
        if intersection <= 0:
            continue
        overlaps.append(
            {
                "object_id": other_oid,
                "intersection_pixels": intersection,
                "overlap_ratio": round(intersection / region_pixels, 8),
                "center_depth": round(float(other_depth), 8),
                "mask_sha256": exact_mask_sha256(other_mask),
            }
        )
    overlaps.sort(key=lambda item: (-item["overlap_ratio"], item["object_id"]))
    if not overlaps or overlaps[0]["overlap_ratio"] < settings.minimum_occluder_overlap:
        maximum = overlaps[0]["overlap_ratio"] if overlaps else 0.0
        raise LiveEvidenceRejected(
            "no_nonstructural_foreground_overlap",
            f"maximum={maximum}, required={settings.minimum_occluder_overlap}",
        )
    occluder = overlaps[0]
    return {
        "derived_class": "occluded",
        "evidence_type": "live_exact_mask_and_geometry_replay",
        "target_oid": target_oid,
        "source": {
            "pixel_count": source_pixels,
            "bbox_xyxy": bbox,
            "mask_sha256": exact_mask_sha256(source_mask),
            **edge,
        },
        "target": {
            "pixel_count": target_pixels,
            "bbox_xyxy": None,
            "mask_sha256": exact_mask_sha256(target_mask),
        },
        "projection": projection,
        "occluder": occluder,
        "occluder_candidates": overlaps[:5],
        "same_type_distractor_pixels_target": same_type_pixels,
        "parameters": {
            "minimum_source_pixels": settings.minimum_source_pixels,
            "minimum_occluder_overlap": settings.minimum_occluder_overlap,
            "foreground_depth_margin": settings.foreground_depth_margin,
            "edge_tolerance_pixels": settings.edge_tolerance_pixels,
            "minimum_one_edge_pixels": settings.minimum_one_edge_pixels,
            "minimum_corner_pixels": settings.minimum_corner_pixels,
        },
        "evidence_sha256": sha256_bytes(
            canonical_json_bytes(
                (
                    target_oid,
                    exact_mask_sha256(source_mask),
                    exact_mask_sha256(target_mask),
                    projection,
                    occluder,
                )
            )
        ),
    }



def _source_view_gates(
    source_event: Any,
    target_oid: str,
    settings: LiveEvidenceConfig,
) -> tuple[Any, int, list[int], dict[str, int]]:
    """Shared strict source-view gates for every mined Task-C class."""

    source_mask = exact_object_mask(source_event, target_oid)
    source_pixels = int(source_mask.sum())
    if source_pixels < settings.minimum_source_pixels:
        raise LiveEvidenceRejected(
            "source_pixels_below_floor",
            f"{source_pixels}<{settings.minimum_source_pixels}",
        )
    bbox = exact_mask_bbox(source_mask)
    if bbox is None:
        raise LiveEvidenceRejected("missing_source_bbox")
    height, width = source_mask.shape
    edge = _edge_guard(bbox, source_pixels, width, height, settings)
    return source_mask, source_pixels, bbox, edge


def _reject_same_type_distractor(target_event: Any, target_oid: str) -> int:
    target_type = target_oid.split("|", 1)[0]
    same_type_pixels = 0
    for other_oid in sorted(_metadata_object_ids(target_event)):
        if other_oid != target_oid and other_oid.split("|", 1)[0] == target_type:
            same_type_pixels += int(exact_object_mask(target_event, other_oid).sum())
    if same_type_pixels:
        raise LiveEvidenceRejected(
            "same_type_distractor_at_target", str(same_type_pixels)
        )
    return 0


def derive_live_visible(
    source_event: Any,
    target_event: Any,
    target_oid: str,
    *,
    vertical_fov_degrees: float,
    config: LiveEvidenceConfig | None = None,
) -> dict[str, Any]:
    """Derive a strict VISIBLE transition from live pixels only.

    Gates, frozen before any mining result was inspected: the shared strict
    source-view gates; an exact live target-instance mask of at least
    ``minimum_visible_target_pixels`` in the target view; the same clipped-view
    edge guard applied to the target-view mask (stricter than the historical
    upstream, which patched clipping through sidecars); and zero same-type
    distractor pixels in the target view.
    """

    settings = config or LiveEvidenceConfig()
    source_mask, source_pixels, bbox, edge = _source_view_gates(
        source_event, target_oid, settings
    )
    target_mask = exact_object_mask(target_event, target_oid)
    target_pixels = int(target_mask.sum())
    if target_pixels < settings.minimum_visible_target_pixels:
        raise LiveEvidenceRejected(
            "visible_target_pixels_below_floor",
            f"{target_pixels}<{settings.minimum_visible_target_pixels}",
        )
    target_bbox = exact_mask_bbox(target_mask)
    if target_bbox is None:
        raise LiveEvidenceRejected("missing_visible_target_bbox")
    height, width = target_mask.shape
    target_edge = _edge_guard(target_bbox, target_pixels, width, height, settings)
    _reject_same_type_distractor(target_event, target_oid)
    return {
        "derived_class": "visible",
        "evidence_type": "live_exact_mask_and_geometry_replay",
        "target_oid": target_oid,
        "source": {
            "pixel_count": source_pixels,
            "bbox_xyxy": bbox,
            "mask_sha256": exact_mask_sha256(source_mask),
            **edge,
        },
        "target": {
            "pixel_count": target_pixels,
            "bbox_xyxy": target_bbox,
            "mask_sha256": exact_mask_sha256(target_mask),
            "edges_touched": target_edge["edges_touched"],
            "minimum_pixels": target_edge["minimum_pixels"],
        },
        "same_type_distractor_pixels_target": 0,
        "parameters": {
            "minimum_source_pixels": settings.minimum_source_pixels,
            "minimum_visible_target_pixels": settings.minimum_visible_target_pixels,
            "edge_tolerance_pixels": settings.edge_tolerance_pixels,
            "minimum_one_edge_pixels": settings.minimum_one_edge_pixels,
            "minimum_corner_pixels": settings.minimum_corner_pixels,
        },
        "evidence_sha256": sha256_bytes(
            canonical_json_bytes(
                (
                    "visible",
                    target_oid,
                    exact_mask_sha256(source_mask),
                    exact_mask_sha256(target_mask),
                )
            )
        ),
    }


def derive_live_out_of_view(
    source_event: Any,
    target_event: Any,
    target_oid: str,
    *,
    vertical_fov_degrees: float,
    config: LiveEvidenceConfig | None = None,
) -> dict[str, Any]:
    """Derive a strict OUT-OF-VIEW transition from live pixels and geometry.

    Gates, frozen before any mining result was inspected: the shared strict
    source-view gates; an exactly zero live target-instance mask in the target
    view; a separating frustum plane certifying the entire live target AABB
    lies outside the view frustum (an exact convex-separation test, stronger
    than a corner heuristic); and zero same-type distractor pixels in the
    target view.
    """

    settings = config or LiveEvidenceConfig()
    source_mask, source_pixels, bbox, edge = _source_view_gates(
        source_event, target_oid, settings
    )
    target_mask = exact_object_mask(target_event, target_oid)
    target_pixels = int(target_mask.sum())
    if target_pixels != 0:
        raise LiveEvidenceRejected("oov_target_mask_not_zero", str(target_pixels))

    np = _numpy()
    segmentation = np.asarray(
        getattr(target_event, "instance_segmentation_frame", None)
    )
    if segmentation.ndim != 3:
        raise LiveEvidenceRejected("missing_instance_segmentation")
    height, width = segmentation.shape[:2]
    fov = float(vertical_fov_degrees)
    if not math.isfinite(fov) or not 0.0 < fov < 180.0:
        raise LiveEvidenceRejected("invalid_live_fov")
    half_vertical = math.radians(fov / 2.0)
    half_horizontal = math.atan(math.tan(half_vertical) * (width / height))
    camera, rotation = _camera(target_event)
    record = _object_metadata(target_event, target_oid)
    corners = [
        _world_to_camera(corner, camera, rotation)
        for corner in _aabb_corners(record)
    ]
    tan_h = math.tan(half_horizontal)
    tan_v = math.tan(half_vertical)
    separating_planes = {
        "near": all(z <= 0.05 for _, _, z in corners),
        "left": all(x < -z * tan_h for x, _, z in corners),
        "right": all(x > z * tan_h for x, _, z in corners),
        "bottom": all(y < -z * tan_v for _, y, z in corners),
        "top": all(y > z * tan_v for _, y, z in corners),
    }
    if not any(separating_planes.values()):
        raise LiveEvidenceRejected(
            "no_separating_frustum_plane",
            "target AABB is not provably outside the live frustum",
        )
    _reject_same_type_distractor(target_event, target_oid)
    return {
        "derived_class": "out_of_fov",
        "evidence_type": "live_exact_mask_and_geometry_replay",
        "target_oid": target_oid,
        "source": {
            "pixel_count": source_pixels,
            "bbox_xyxy": bbox,
            "mask_sha256": exact_mask_sha256(source_mask),
            **edge,
        },
        "target": {
            "pixel_count": 0,
            "bbox_xyxy": None,
            "mask_sha256": exact_mask_sha256(target_mask),
        },
        "frustum_separation": {
            "method": "live_3d_aabb_convex_separating_plane",
            "planes": {k: bool(v) for k, v in separating_planes.items()},
            "half_horizontal_radians": round(half_horizontal, 8),
            "half_vertical_radians": round(half_vertical, 8),
        },
        "same_type_distractor_pixels_target": 0,
        "parameters": {
            "minimum_source_pixels": settings.minimum_source_pixels,
            "edge_tolerance_pixels": settings.edge_tolerance_pixels,
            "minimum_one_edge_pixels": settings.minimum_one_edge_pixels,
            "minimum_corner_pixels": settings.minimum_corner_pixels,
        },
        "evidence_sha256": sha256_bytes(
            canonical_json_bytes(
                (
                    "out_of_fov",
                    target_oid,
                    exact_mask_sha256(source_mask),
                    exact_mask_sha256(target_mask),
                    {k: bool(v) for k, v in sorted(separating_planes.items())},
                )
            )
        ),
    }
