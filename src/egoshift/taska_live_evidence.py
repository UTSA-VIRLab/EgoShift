"""Live-render evidence primitives for Task-A cross-view object identity.

Nothing here consumes a stored identity label. "Same" and "different" are
derived from exact instance masks and live object identity, and every item
carries the per-view coverage statistic that the known construction cue acts
on, so selection can match on it rather than discover it later.

The cue this module exists to expose: a *same* item re-observes one object and
re-observation tends to clip it, while a *different* item shows two cleanly
visible objects. Measured as
``visible_mask_pixels / unclipped_projected_silhouette_pixels`` -- the same
definition the historical pool used -- a coverage of 1.0 was 99.6% *different*.
Recording it per view is what makes cue-matched construction possible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .hashing import canonical_json_bytes, sha256_bytes
from .taskc_live_evidence import (
    LiveEvidenceRejected,
    _aabb_corners,
    _camera,
    _convex_hull,
    _metadata_object_ids,
    _numpy,
    _object_metadata,
    _world_to_camera,
    exact_mask_bbox,
    exact_mask_sha256,
    exact_object_mask,
)

COVERAGE_DEFINITION = "visible_mask_pixels / unclipped_projected_silhouette_pixels"
ANSWER_SAME = "same"
ANSWER_DIFFERENT = "different"


@dataclass(frozen=True)
class TaskAEvidenceConfig:
    """Gates frozen before any mined yield is inspected."""

    minimum_visible_pixels: int = 2000
    identifiable_minimum_pixels: int = 100
    minimum_coverage: float = 0.10
    edge_tolerance_pixels: int = 2
    # A referent is only unambiguous if exactly one instance of its type is
    # identifiable in the view that boxes it.
    required_same_type_visible: int = 1
    # Search aiming only, never a gate: the camera distance that renders a
    # target near this bbox width is computed per object from its AABB.
    target_bbox_width_pixels: float = 110.0


def _polygon_area(points: Sequence[tuple[float, float]]) -> float:
    """Shoelace area of a polygon, in pixel units, without clipping to frame."""

    if len(points) < 3:
        return 0.0
    total = 0.0
    for index in range(len(points)):
        x0, y0 = points[index]
        x1, y1 = points[(index + 1) % len(points)]
        total += x0 * y1 - x1 * y0
    return abs(total) / 2.0


def unclipped_projected_silhouette_pixels(
    event: Any, object_id: str, vertical_fov_degrees: float
) -> tuple[float, dict[str, Any]]:
    """Project the live 3D AABB and measure its hull area *before* frame clipping.

    The denominator of the coverage statistic must not be clipped: clipping it
    would hide exactly the effect the cue is made of.
    """

    np = _numpy()
    segmentation = np.asarray(getattr(event, "instance_segmentation_frame", None))
    if segmentation.ndim != 3:
        raise LiveEvidenceRejected("missing_instance_segmentation")
    height, width = segmentation.shape[:2]
    fov = float(vertical_fov_degrees)
    if not math.isfinite(fov) or not 0.0 < fov < 180.0:
        raise LiveEvidenceRejected("invalid_live_fov")
    half_vertical = math.radians(fov / 2.0)
    half_horizontal = math.atan(math.tan(half_vertical) * (width / height))
    camera, rotation = _camera(event)
    record = _object_metadata(event, object_id)
    projected: list[tuple[float, float]] = []
    for corner in _aabb_corners(record):
        x, y, z = _world_to_camera(corner, camera, rotation)
        if z <= 0.05:
            raise LiveEvidenceRejected("target_aabb_crosses_near_plane")
        projected.append(
            (
                x / (z * math.tan(half_horizontal)) * width / 2.0 + width / 2.0,
                -y / (z * math.tan(half_vertical)) * height / 2.0 + height / 2.0,
            )
        )
    hull = _convex_hull(projected)
    area = _polygon_area(hull)
    if area <= 0.0:
        raise LiveEvidenceRejected("degenerate_projected_silhouette")
    inside = sum(1 for x, y in hull if 0 <= x < width and 0 <= y < height)
    return area, {
        "method": "live_3d_aabb_convex_hull_shoelace_unclipped",
        "unclipped_silhouette_pixels": round(area, 4),
        "hull_vertices": len(hull),
        "hull_vertices_inside_frame": inside,
        "image_size": {"w": int(width), "h": int(height)},
    }


def _identifiable_same_type(
    event: Any, object_type: str, config: TaskAEvidenceConfig
) -> dict[str, int]:
    """Exact per-instance pixel counts for every identifiable same-type object."""

    counts: dict[str, int] = {}
    for other in sorted(_metadata_object_ids(event)):
        if other.split("|", 1)[0] != object_type:
            continue
        pixels = int(exact_object_mask(event, other).sum())
        if pixels >= config.identifiable_minimum_pixels:
            counts[other] = pixels
    return counts


def _view_evidence(
    event: Any,
    object_id: str,
    *,
    vertical_fov_degrees: float,
    config: TaskAEvidenceConfig,
    label: str,
) -> dict[str, Any]:
    """Gate one boxed view and measure its coverage statistic."""

    mask = exact_object_mask(event, object_id)
    pixels = int(mask.sum())
    if pixels < config.minimum_visible_pixels:
        raise LiveEvidenceRejected(
            f"{label}_pixels_below_floor", f"{pixels}<{config.minimum_visible_pixels}"
        )
    bbox = exact_mask_bbox(mask)
    if bbox is None:
        raise LiveEvidenceRejected(f"{label}_missing_bbox")
    silhouette, projection = unclipped_projected_silhouette_pixels(
        event, object_id, vertical_fov_degrees
    )
    coverage = pixels / silhouette
    if coverage < config.minimum_coverage:
        raise LiveEvidenceRejected(
            f"{label}_coverage_below_floor", f"{coverage:.4f}<{config.minimum_coverage}"
        )
    object_type = object_id.split("|", 1)[0]
    same_type = _identifiable_same_type(event, object_type, config)
    if len(same_type) != config.required_same_type_visible:
        raise LiveEvidenceRejected(
            f"{label}_same_type_ambiguous",
            f"{len(same_type)} identifiable {object_type} instances",
        )
    if object_id not in same_type:
        raise LiveEvidenceRejected(f"{label}_target_not_identifiable")
    height, width = mask.shape
    x0, y0, x1, y1 = bbox
    touches_edge = (
        x0 <= config.edge_tolerance_pixels
        or y0 <= config.edge_tolerance_pixels
        or x1 >= width - config.edge_tolerance_pixels
        or y1 >= height - config.edge_tolerance_pixels
    )
    return {
        "object_id": object_id,
        "object_type": object_type,
        "visible_pixels": pixels,
        "bbox_xyxy": bbox,
        "mask_sha256": exact_mask_sha256(mask),
        "view_coverage": round(coverage, 8),
        "coverage_definition": COVERAGE_DEFINITION,
        "mask_touches_frame_edge": bool(touches_edge),
        "identifiable_same_type_count": len(same_type),
        "projection": projection,
    }


def _assemble(
    answer_class: str,
    view1: Mapping[str, Any],
    view2: Mapping[str, Any],
    config: TaskAEvidenceConfig,
) -> dict[str, Any]:
    coverages = [float(view1["view_coverage"]), float(view2["view_coverage"])]
    evidence = {
        "derived_class": answer_class,
        "evidence_type": "live_exact_mask_identity",
        "object_type": view1["object_type"],
        "view_1": dict(view1),
        "view_2": dict(view2),
        # The cue statistic, recorded so selection can match on it.
        "minimum_view_coverage": round(min(coverages), 8),
        "maximum_view_coverage": round(max(coverages), 8),
        "any_view_touches_frame_edge": bool(
            view1["mask_touches_frame_edge"] or view2["mask_touches_frame_edge"]
        ),
        "coverage_definition": COVERAGE_DEFINITION,
        "parameters": {
            "minimum_visible_pixels": config.minimum_visible_pixels,
            "identifiable_minimum_pixels": config.identifiable_minimum_pixels,
            "minimum_coverage": config.minimum_coverage,
            "required_same_type_visible": config.required_same_type_visible,
            "edge_tolerance_pixels": config.edge_tolerance_pixels,
        },
    }
    evidence["evidence_sha256"] = sha256_bytes(
        canonical_json_bytes(
            (
                answer_class,
                view1["object_id"],
                view2["object_id"],
                view1["mask_sha256"],
                view2["mask_sha256"],
            )
        )
    )
    return evidence


def derive_live_same_object(
    view1_event: Any,
    view2_event: Any,
    object_id: str,
    *,
    vertical_fov_degrees: float,
    config: TaskAEvidenceConfig | None = None,
) -> dict[str, Any]:
    """One physical object, identifiable in both views, boxed in each."""

    settings = config or TaskAEvidenceConfig()
    view1 = _view_evidence(
        view1_event, object_id,
        vertical_fov_degrees=vertical_fov_degrees, config=settings, label="view1",
    )
    view2 = _view_evidence(
        view2_event, object_id,
        vertical_fov_degrees=vertical_fov_degrees, config=settings, label="view2",
    )
    return _assemble(ANSWER_SAME, view1, view2, settings)


def derive_live_different_objects(
    view1_event: Any,
    view2_event: Any,
    object_id_1: str,
    object_id_2: str,
    *,
    vertical_fov_degrees: float,
    config: TaskAEvidenceConfig | None = None,
) -> dict[str, Any]:
    """Two distinct instances of one type, one identifiable per view.

    Each view must contain exactly one identifiable instance of the type, and
    it must be the intended one -- otherwise "two different objects" is not a
    claim the viewer can check.
    """

    settings = config or TaskAEvidenceConfig()
    if object_id_1 == object_id_2:
        raise LiveEvidenceRejected("different_requires_distinct_instances")
    if object_id_1.split("|", 1)[0] != object_id_2.split("|", 1)[0]:
        raise LiveEvidenceRejected("different_requires_matching_type")
    view1 = _view_evidence(
        view1_event, object_id_1,
        vertical_fov_degrees=vertical_fov_degrees, config=settings, label="view1",
    )
    view2 = _view_evidence(
        view2_event, object_id_2,
        vertical_fov_degrees=vertical_fov_degrees, config=settings, label="view2",
    )
    return _assemble(ANSWER_DIFFERENT, view1, view2, settings)
