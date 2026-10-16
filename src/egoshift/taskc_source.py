"""Fail-closed preparation of the private Task-C candidate source.

The upstream QA file is a mixed-task construction artifact.  This module
streams it, replays simulator evidence from the saved trajectories, applies
the frozen physical-scene roles, and selects exact per-role/per-class quotas.
It deliberately does not infer a weaker label when evidence is absent.

The selected JSONL is private: it retains gold and audit provenance for the
later public/private materialization step.  Public release construction must
still remove every field other than the benchmark input surface.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import tempfile
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hashing import canonical_json_bytes, resolve_within, safe_relative_path, sha256_bytes, sha256_file
from .taskc_type_matching import match_object_type_distribution, object_type_of
from .split_roles import require_complete_split_roles


EXPECTED_UPSTREAM_SHA256 = "5caec6cd2343d3974cc91e0b799b3fac3851a0074e07deab94f6d808de70c8f3"

ROLE_ORDER = ("tuning_dev", "public_dev", "hidden_test")
CLASS_ORDER = ("visible", "out_of_fov", "occluded")

# How far above 1/3 a physical-target identity oracle may score before the
# selection is rejected. Small but non-zero: quota edges leave a handful of
# targets short of a full triple, which cannot be driven to exactly chance.
IDENTITY_ORACLE_TOLERANCE = 0.05
# Below this many selected rows the oracle is reported but not enforced --
# see the note at the check itself.
IDENTITY_ORACLE_MIN_ROWS = 100
DEFAULT_QUOTAS = {
    "tuning_dev": 44,
    "public_dev": 30,
    "hidden_test": 177,
}

CLASS_TO_ANSWER = {
    "visible": "Visible — It is visible in the current view, either fully or partially.",
    "out_of_fov": "Out of View — It is not in frame because the camera is not directed toward it.",
    "occluded": "Occluded — It is in frame but completely hidden by another object.",
}
CANONICAL_OPTIONS = tuple(CLASS_TO_ANSWER[name] for name in CLASS_ORDER)

# One question template for every row, whatever pipeline mined it. The
# upstream trajectory pool phrased the question one way with trajectory view
# numbers ("After the viewpoint changes from View 1 to View 4 ..."), the
# constructed-occlusion miner another ("You saw the X in View 1 ..."), and the
# two pipelines supply different classes -- so the phrasing itself was a
# perfect text-only class label. Every candidate's question is rebuilt from
# this template at normalization; only the referent noun varies.
QUESTION_TEMPLATE = (
    "You saw the {noun} in View 1. In View 2, what best describes its visibility?"
)


def display_noun(object_type: str) -> str:
    """Human noun for an objectId type prefix, e.g. GarbageCan -> garbage can."""

    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", object_type)
    return " ".join(spaced.split()).casefold() or "object"

LEGACY_CLASS_MAP = {
    "visible": "visible",
    "partially_occluded": "visible",
    "partial_occlusion": "visible",
    "out_of_fov": "out_of_fov",
    "fully_occluded": "occluded",
    "occluded": "occluded",
}

STRUCTURAL_OCCLUDER_PREFIXES = (
    "wall|",
    "door|",
    "window|",
    "room|",
    "floor|",
    "ceiling|",
)
# A structural element makes a poor *target* as well as a poor occluder.
# "Is the wall still visible?" is close to unanswerable and "is the room
# visible?" is meaningless; these also cannot be pointed at unambiguously by a
# box. Task B already excludes the same list from its targets, so applying it
# here keeps the two tasks consistent. Enforced at selection rather than in the
# miner so that already-mined candidates stay valid.
STRUCTURAL_TARGET_PREFIXES = (
    "wall|",
    "door|",
    "doorway|",
    "doorframe|",
    "window|",
    "windowframe|",
    "room|",
    "floor|",
    "ceiling|",
)


def is_structural_target(target_oid: str) -> bool:
    """True when the boxed referent is part of the building, not an object."""

    return str(target_oid).casefold().startswith(STRUCTURAL_TARGET_PREFIXES)
FRAME_PATTERN = re.compile(
    r"^(?P<scene>scene_[0-9]+)/trajectories/(?P<trajectory>[^/]+)/frames/"
    r"frame_(?P<step>[0-9]+)\.png$"
)
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")

MINIMUM_SOURCE_PIXELS = 500
# A "Visible" item must show enough of the object that a person can actually
# see it. At the old floor of 30 pixels -- 0.01% of a 640x480 frame -- the
# label was logically right and perceptually impossible: every one of a
# rater's visible-class errors was an item showing 62, 92 or 293 pixels, and
# she answered "Occluded" because she genuinely could not see the object.
# Gold a careful rater cannot verify is not gold. 300px costs ~7% of the
# visible pool and makes every released item human-checkable.
MINIMUM_VISIBLE_TARGET_PIXELS = 300
MINIMUM_OCCLUDER_OVERLAP = 0.05
MINIMUM_PARTIAL_OVERLAP = 0.10
EDGE_TOLERANCE_PIXELS = 2
MINIMUM_ONE_EDGE_PIXELS = 1500
MINIMUM_CORNER_PIXELS = 3000
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


class CandidateRejected(ValueError):
    """Expected item-level rejection with a stable audit reason."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class TaskCSourceConfig:
    """Inputs and frozen quota contract for :func:`prepare_taskc_source`."""

    source_path: Path
    split_roles_path: Path
    scenes_root: Path
    output_path: Path
    report_path: Path
    expected_source_sha256: str = EXPECTED_UPSTREAM_SHA256
    robust_sidecar_paths: tuple[Path, ...] = ()
    quotas: Mapping[str, int] | None = None
    minimum_source_pixels: int = MINIMUM_SOURCE_PIXELS
    minimum_visible_target_pixels: int = MINIMUM_VISIBLE_TARGET_PIXELS
    minimum_occluder_overlap: float = MINIMUM_OCCLUDER_OVERLAP
    minimum_partial_overlap: float = MINIMUM_PARTIAL_OVERLAP
    supplement_jsonl_paths: tuple[Path, ...] = ()
    supplements_only: bool = False


class _JsonLRU:
    """Small deterministic JSON cache; upstream rows are mostly scene-grouped."""

    def __init__(self, maximum: int = 192) -> None:
        self.maximum = maximum
        self._values: OrderedDict[str, Any] = OrderedDict()

    def load(self, path: Path) -> Any:
        key = str(path.resolve())
        if key in self._values:
            self._values.move_to_end(key)
            return self._values[key]
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self._values[key] = payload
        if len(self._values) > self.maximum:
            self._values.popitem(last=False)
        return payload


def _stable_digest(*values: object) -> str:
    return sha256_bytes(canonical_json_bytes(values))


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    payload = json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ).encode("utf-8") + b"\n"
    _atomic_write_bytes(path, payload)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    _atomic_write_bytes(path, payload)


def _validated_quotas(raw: Mapping[str, int] | None) -> dict[str, int]:
    quotas = dict(DEFAULT_QUOTAS if raw is None else raw)
    if set(quotas) != set(ROLE_ORDER):
        raise ValueError(f"quota roles must be exactly {ROLE_ORDER!r}")
    for role, value in quotas.items():
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(f"quota for {role!r} must be a non-negative integer")
        quotas[role] = int(value)
    return quotas


def _load_split_roles(path: Path) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    exposure_binding = require_complete_split_roles(payload)
    scenes = payload.get("scenes")
    if not isinstance(scenes, dict) or not scenes:
        raise ValueError("split-role audit lacks a non-empty scenes mapping")
    component_roles: dict[str, str] = {}
    for scene_id, record in scenes.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid split-role record for {scene_id!r}")
        role = str(record.get("role") or "")
        component = str(record.get("leakage_component_id") or "")
        if role not in ROLE_ORDER or not component:
            raise ValueError(f"invalid role/component for {scene_id!r}")
        old_role = component_roles.setdefault(component, role)
        if old_role != role:
            raise ValueError(
                f"physical component {component!r} crosses roles: {old_role!r}/{role!r}"
            )
    provenance = {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": payload.get("schema_version"),
        "split_contract_id": payload.get("split_contract_id"),
        "seed": payload.get("seed"),
        "scenes": len(scenes),
        "components": len(component_roles),
        "exposure_ledger": exposure_binding,
    }
    return scenes, provenance


def _load_robust_sidecars(
    paths: Sequence[Path],
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    merged: dict[str, Mapping[str, Any]] = {}
    provenance: list[dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"robust sidecar must be an object: {path}")
        provenance.append(
            {"path": str(path), "sha256": sha256_file(path), "items": len(payload)}
        )
        for item_id, entry in payload.items():
            if not isinstance(item_id, str) or not isinstance(entry, Mapping):
                raise ValueError(f"invalid robust sidecar entry in {path}")
            if item_id in merged and canonical_json_bytes(merged[item_id]) != canonical_json_bytes(entry):
                raise ValueError(f"conflicting robust sidecar entry for {item_id!r}")
            merged[item_id] = entry
    return merged, provenance


def _as_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise CandidateRejected("invalid_numeric_field", f"{name} is boolean")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise CandidateRejected("invalid_numeric_field", f"{name} is not an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise CandidateRejected("invalid_numeric_field", f"{name} is not integral")
    return result


def _as_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CandidateRejected("invalid_numeric_field", f"{name} is not numeric") from exc
    if not math.isfinite(result):
        raise CandidateRejected("invalid_numeric_field", f"{name} is not finite")
    return result


def _frame_parts(frame: Any) -> tuple[str, str, int, str]:
    try:
        normalized = safe_relative_path(frame)
    except (TypeError, ValueError) as exc:
        raise CandidateRejected("invalid_frame_path", str(exc)) from exc
    match = FRAME_PATTERN.fullmatch(normalized)
    if match is None:
        raise CandidateRejected("invalid_frame_path", normalized)
    return (
        match.group("scene"),
        match.group("trajectory"),
        int(match.group("step")),
        normalized,
    )


def _step_index(trajectory: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    steps = trajectory.get("steps")
    if not isinstance(steps, list):
        raise CandidateRejected("invalid_trajectory", "trajectory has no steps list")
    indexed: dict[int, Mapping[str, Any]] = {}
    for fallback, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise CandidateRejected("invalid_trajectory", "trajectory step is not an object")
        index = _as_int(step.get("step_index", fallback), "trajectory step_index")
        if index in indexed:
            raise CandidateRejected("invalid_trajectory", f"duplicate step_index={index}")
        indexed[index] = step
    return indexed


def _visibility_state(step: Mapping[str, Any], target_oid: str) -> str:
    states = step.get("visibility_states")
    record = states.get(target_oid) if isinstance(states, Mapping) else None
    if not isinstance(record, Mapping):
        raise CandidateRejected("missing_trajectory_visibility_state", target_oid)
    state = str(record.get("state") or "").upper()
    if state not in {"VISIBLE", "OUT_OF_FOV", "OCCLUDED"}:
        raise CandidateRejected("invalid_trajectory_visibility_state", state)
    return state


def _pixel_count(step: Mapping[str, Any], target_oid: str) -> int:
    counts = step.get("instance_pixel_counts")
    value = counts.get(target_oid, 0) if isinstance(counts, Mapping) else 0
    result = _as_int(value, "trajectory instance pixel count")
    if result < 0:
        raise CandidateRejected("invalid_trajectory_pixel_count", str(result))
    return result


def frustum_diagnostic(
    object_position: Mapping[str, Any],
    agent: Mapping[str, Any],
    vertical_fov_degrees: float,
    *,
    aspect: float = IMAGE_WIDTH / IMAGE_HEIGHT,
) -> dict[str, Any]:
    """Independently project an object center using saved pose and scene FOV.

    This mirrors the declared AI2-THOR coordinate convention but does not read
    the stored visibility label or the stored projection booleans.
    """

    try:
        camera_position = agent["position"]
        rotation = agent["rotation"]
        dx = float(object_position["x"]) - float(camera_position["x"])
        dy = float(object_position["y"]) - float(camera_position["y"])
        dz = float(object_position["z"]) - float(camera_position["z"])
        yaw = math.radians(float(rotation["y"]))
        pitch = math.radians(float(agent.get("cameraHorizon", 0.0)))
        fov = float(vertical_fov_degrees)
    except (KeyError, TypeError, ValueError) as exc:
        raise CandidateRejected("invalid_projection_inputs", str(exc)) from exc
    values = (dx, dy, dz, yaw, pitch, fov, aspect)
    if not all(math.isfinite(value) for value in values) or not (0.0 < fov < 180.0):
        raise CandidateRejected("invalid_projection_inputs", "non-finite pose or invalid FOV")

    cos_y, sin_y = math.cos(-yaw), math.sin(-yaw)
    local_x = cos_y * dx + sin_y * dz
    local_z_1 = -sin_y * dx + cos_y * dz
    cos_p, sin_p = math.cos(-pitch), math.sin(-pitch)
    local_y = cos_p * dy - sin_p * local_z_1
    local_z = sin_p * dy + cos_p * local_z_1

    half_vertical = math.radians(fov / 2.0)
    half_horizontal = math.atan(math.tan(half_vertical) * aspect)
    if local_z <= 0.05:
        inside = False
        horizontal_excess = math.pi
        vertical_excess = math.pi
    else:
        horizontal_angle = math.atan2(abs(local_x), local_z)
        vertical_angle = math.atan2(abs(local_y), local_z)
        horizontal_excess = horizontal_angle - half_horizontal
        vertical_excess = vertical_angle - half_vertical
        inside = horizontal_excess <= 0.0 and vertical_excess <= 0.0
    signed_margin = -max(horizontal_excess, vertical_excess)
    return {
        "inside": inside,
        "signed_margin_degrees": round(math.degrees(signed_margin), 8),
        "local_xyz": [round(local_x, 8), round(local_y, 8), round(local_z, 8)],
        "vertical_fov_degrees": fov,
        "aspect": aspect,
    }


def _object_position(scene_metadata: Mapping[str, Any], target_oid: str) -> Mapping[str, Any]:
    objects = scene_metadata.get("objects")
    if not isinstance(objects, list):
        raise CandidateRejected("invalid_scene_metadata", "objects is not a list")
    matches = [
        item.get("position")
        for item in objects
        if isinstance(item, Mapping) and item.get("objectId") == target_oid
    ]
    if len(matches) != 1 or not isinstance(matches[0], Mapping):
        raise CandidateRejected(
            "missing_or_ambiguous_object_position", f"{target_oid}: matches={len(matches)}"
        )
    return matches[0]


def _overlap_evidence(
    step: Mapping[str, Any], target_oid: str, occluder_id: str
) -> tuple[float, str]:
    direct = step.get("occlusion_evidence")
    target_direct = direct.get(target_oid) if isinstance(direct, Mapping) else None
    occluders = target_direct.get("occluders") if isinstance(target_direct, Mapping) else None
    if isinstance(occluders, Mapping) and occluder_id in occluders:
        return _as_float(occluders[occluder_id], "direct occluder overlap"), "occlusion_evidence"

    fallback = step.get("occluder_overlap")
    target_fallback = fallback.get(target_oid) if isinstance(fallback, Mapping) else None
    if isinstance(target_fallback, Mapping) and occluder_id in target_fallback:
        return _as_float(
            target_fallback[occluder_id], "trajectory occluder overlap"
        ), "occluder_overlap"
    raise CandidateRejected(
        "missing_explicit_occlusion_evidence", f"{target_oid} behind {occluder_id}"
    )


def _validate_stored_overlap(metadata: Mapping[str, Any], observed: float) -> None:
    stored = _as_float(metadata.get("occluder_overlap_ratio"), "stored occluder overlap")
    if abs(stored - observed) > 1e-5:
        raise CandidateRejected(
            "occluder_overlap_mismatch", f"stored={stored}, trajectory={observed}"
        )


def _edge_count(bbox: Sequence[int], width: int, height: int) -> int:
    x0, y0, x1, y1 = bbox
    return sum(
        (
            x0 <= EDGE_TOLERANCE_PIXELS,
            y0 <= EDGE_TOLERANCE_PIXELS,
            x1 >= width - EDGE_TOLERANCE_PIXELS,
            y1 >= height - EDGE_TOLERANCE_PIXELS,
        )
    )


def _validated_bbox(value: Any, width: int, height: int) -> list[int] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise CandidateRejected("invalid_robust_bbox", "bbox must be null or four integers")
    bbox = [_as_int(item, "robust bbox coordinate") for item in value]
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise CandidateRejected("invalid_robust_bbox", repr(bbox))
    return bbox


def _apply_robust_guard(
    row: Mapping[str, Any],
    entry: Mapping[str, Any],
    target_oid: str,
    canonical_class: str,
    source_step: int,
    target_step: int,
    source_pixels: int,
    target_pixels: int,
) -> dict[str, Any]:
    frames = row["frames"]
    sidecar_frames = entry.get("frames")
    if sidecar_frames is not None and sidecar_frames != frames:
        raise CandidateRejected("robust_sidecar_frame_mismatch", str(row.get("id")))
    size = entry.get("image_size")
    if not isinstance(size, Mapping):
        raise CandidateRejected("invalid_robust_sidecar", "missing image_size")
    width = _as_int(size.get("w"), "robust image width")
    height = _as_int(size.get("h"), "robust image height")
    if width <= 0 or height <= 0:
        raise CandidateRejected("invalid_robust_sidecar", "non-positive image size")
    raw_boxes = entry.get("boxes")
    if not isinstance(raw_boxes, list):
        raise CandidateRejected("invalid_robust_sidecar", "boxes is not a list")

    exact: dict[int, Mapping[str, Any]] = {}
    for box in raw_boxes:
        if not isinstance(box, Mapping) or box.get("object_id") != target_oid:
            continue
        index = _as_int(box.get("frame_index"), "robust frame_index")
        if index in {source_step, target_step}:
            if index in exact:
                raise CandidateRejected("ambiguous_robust_target_box", f"step={index}")
            exact[index] = box
    if set(exact) != {source_step, target_step}:
        raise CandidateRejected(
            "missing_exact_robust_target_box",
            f"needed={source_step,target_step}, observed={sorted(exact)}",
        )

    diagnostics: dict[str, Any] = {"image_size": {"w": width, "h": height}}
    for name, step, expected_pixels, frame_path in (
        ("source", source_step, source_pixels, frames[0]),
        ("target", target_step, target_pixels, frames[1]),
    ):
        box = exact[step]
        if box.get("frame_path") != frame_path:
            raise CandidateRejected("robust_sidecar_frame_mismatch", f"{name}: {frame_path}")
        pixels = _as_int(box.get("pixel_count"), "robust pixel_count")
        if pixels != expected_pixels:
            raise CandidateRejected(
                "robust_sidecar_pixel_mismatch",
                f"{name}: sidecar={pixels}, trajectory={expected_pixels}",
            )
        bbox = _validated_bbox(box.get("bbox_xyxy"), width, height)
        should_be_visible = name == "source" or canonical_class == "visible"
        if should_be_visible and bbox is None:
            raise CandidateRejected("missing_visible_robust_bbox", name)
        if not should_be_visible and bbox is not None:
            raise CandidateRejected("unexpected_invisible_robust_bbox", name)
        edges = _edge_count(bbox, width, height) if bbox is not None else 0
        minimum = MINIMUM_CORNER_PIXELS if edges >= 2 else MINIMUM_ONE_EDGE_PIXELS if edges == 1 else 0
        if pixels < minimum:
            raise CandidateRejected(
                "robust_clipped_view_guard",
                f"{name}: edges={edges}, pixels={pixels}, required={minimum}",
            )
        diagnostics[name] = {
            "bbox_xyxy": bbox,
            "pixel_count": pixels,
            "edges_touched": edges,
            "minimum_pixels": minimum,
        }
    answer_class = str(entry.get("answer_class") or "")
    if answer_class and LEGACY_CLASS_MAP.get(answer_class) != canonical_class:
        raise CandidateRejected("robust_sidecar_class_mismatch", answer_class)
    return diagnostics


def _candidate_sort_key(candidate: Mapping[str, Any]) -> tuple[object, ...]:
    quality = candidate["_quality"]
    return tuple(-float(value) for value in quality) + (candidate["_stable"],)


def _normalized_candidate(
    row: Mapping[str, Any],
    *,
    legacy_class: str,
    canonical_class: str,
    role: str,
    component: str,
    source_line: int,
    source_sha256: str,
    evidence: Mapping[str, Any],
    stable: str,
) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(row))
    removed_absolute_config_fields: dict[str, str] = {}
    for field in ("rules_config", "category_config"):
        value = normalized.pop(field, None)
        if isinstance(value, str):
            removed_absolute_config_fields[field] = Path(value).name
    metadata = dict(normalized.get("metadata") or {})
    metadata["legacy_answer_class"] = legacy_class
    metadata["answer_class"] = canonical_class
    normalized["metadata"] = metadata
    normalized["answer"] = CLASS_TO_ANSWER[canonical_class]
    normalized["options"] = list(CANONICAL_OPTIONS)
    normalized["question"] = QUESTION_TEMPLATE.format(
        noun=display_noun(str(normalized.get("target_oid") or "").split("|", 1)[0])
    )
    normalized["candidate_role"] = role
    normalized["_candidate_source"] = {
        "schema_version": "0.1.0",
        "upstream_sha256": source_sha256,
        "upstream_line": source_line,
        "upstream_id": row.get("id"),
        "physical_component_id": component,
        "role": role,
        "stable_identity_sha256": stable,
        "gold_evidence": dict(evidence),
        "stripped_config_basenames": removed_absolute_config_fields,
    }
    return normalized


def _evaluate_row(
    row: Mapping[str, Any],
    *,
    line_number: int,
    source_sha256: str,
    split_scenes: Mapping[str, Mapping[str, Any]],
    sidecars: Mapping[str, Mapping[str, Any]],
    scenes_root: Path,
    cache: _JsonLRU,
    config: TaskCSourceConfig,
) -> dict[str, Any]:
    if row.get("task_family") != "C":
        raise CandidateRejected("not_task_c")
    if is_structural_target(str(row.get("target_oid") or "")):
        raise CandidateRejected("structural_target", str(row.get("target_oid")))
    scene_id = str(row.get("scene_id") or "")
    split = split_scenes.get(scene_id)
    if split is None:
        raise CandidateRejected("scene_absent_from_split_roles", scene_id)
    if split.get("drop_exact_alias_nonrepresentative") is True:
        raise CandidateRejected("exact_alias_nonrepresentative", scene_id)
    role = str(split.get("role") or "")
    component = str(split.get("leakage_component_id") or "")
    if role not in ROLE_ORDER or not component:
        raise CandidateRejected("invalid_scene_role", scene_id)

    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        raise CandidateRejected("missing_metadata")
    legacy_class = str(metadata.get("answer_class") or "")
    canonical_class = LEGACY_CLASS_MAP.get(legacy_class)
    if canonical_class is None:
        raise CandidateRejected("unknown_answer_class", legacy_class)
    if str(row.get("candidate_answer_class") or legacy_class) != legacy_class:
        raise CandidateRejected("candidate_answer_class_mismatch")
    target_oid = str(row.get("target_oid") or "")
    if not target_oid:
        raise CandidateRejected("missing_target_oid")
    if metadata.get("object_ids") != [target_oid]:
        raise CandidateRejected("target_object_ids_mismatch", target_oid)

    frames = row.get("frames")
    if not isinstance(frames, list) or len(frames) != 2:
        raise CandidateRejected("invalid_frame_schema")
    source_parts = _frame_parts(frames[0])
    target_parts = _frame_parts(frames[1])
    source_step = _as_int(row.get("source_step"), "source_step")
    target_step = _as_int(row.get("target_step"), "target_step")
    if target_step <= source_step:
        raise CandidateRejected("non_forward_view_pair", f"{source_step}->{target_step}")
    if (
        source_parts[0] != scene_id
        or target_parts[0] != scene_id
        or source_parts[1] != target_parts[1]
        or source_parts[2] != source_step
        or target_parts[2] != target_step
    ):
        raise CandidateRejected("frame_trajectory_step_mismatch")
    trajectory_subdir = source_parts[1]
    if str(metadata.get("trajectory_subdir") or "") != trajectory_subdir:
        raise CandidateRejected("metadata_trajectory_mismatch")
    views_used = metadata.get("views_used")
    if views_used is not None and views_used != [source_step, target_step]:
        raise CandidateRejected("metadata_view_pair_mismatch")

    for frame in (source_parts[3], target_parts[3]):
        try:
            asset = resolve_within(scenes_root, frame)
        except ValueError as exc:
            raise CandidateRejected("invalid_frame_path", str(exc)) from exc
        if not asset.is_file():
            raise CandidateRejected("missing_frame_asset", frame)
    trajectory_relative = (
        f"{scene_id}/trajectories/{safe_relative_path(trajectory_subdir)}/trajectory.json"
    )
    trajectory_path = resolve_within(scenes_root, trajectory_relative)
    if not trajectory_path.is_file():
        raise CandidateRejected("missing_trajectory", trajectory_relative)
    scene_metadata_path = resolve_within(scenes_root, f"{scene_id}/metadata.json")
    if not scene_metadata_path.is_file():
        raise CandidateRejected("missing_scene_metadata", scene_id)
    trajectory = cache.load(trajectory_path)
    scene_metadata = cache.load(scene_metadata_path)
    if not isinstance(trajectory, Mapping) or not isinstance(scene_metadata, Mapping):
        raise CandidateRejected("invalid_evidence_json")
    steps = _step_index(trajectory)
    if source_step not in steps or target_step not in steps:
        raise CandidateRejected("missing_trajectory_step", f"{source_step},{target_step}")
    source_record, target_record = steps[source_step], steps[target_step]

    source_state = _visibility_state(source_record, target_oid)
    target_state = _visibility_state(target_record, target_oid)
    source_pixels = _pixel_count(source_record, target_oid)
    target_pixels = _pixel_count(target_record, target_oid)
    stored_source_pixels = _as_int(
        metadata.get("target_pixel_count_source"), "stored source pixels"
    )
    stored_target_pixels = _as_int(
        metadata.get("target_pixel_count_target"), "stored target pixels"
    )
    if source_pixels != stored_source_pixels or target_pixels != stored_target_pixels:
        raise CandidateRejected(
            "trajectory_pixel_mismatch",
            f"source={stored_source_pixels}/{source_pixels}, target={stored_target_pixels}/{target_pixels}",
        )
    if source_state != "VISIBLE":
        raise CandidateRejected("source_not_visible", source_state)
    if source_pixels < config.minimum_source_pixels:
        raise CandidateRejected(
            "source_pixels_below_floor", f"{source_pixels}<{config.minimum_source_pixels}"
        )
    distractor_pixels = _as_int(
        metadata.get("same_type_distractor_pixel_count_target", 0),
        "same-type distractor pixels",
    )
    if distractor_pixels != 0:
        raise CandidateRejected("same_type_distractor_at_target", str(distractor_pixels))

    object_position = _object_position(scene_metadata, target_oid)
    target_agent = target_record.get("agent")
    if not isinstance(target_agent, Mapping):
        raise CandidateRejected("missing_target_agent_pose")
    fov = _as_float(scene_metadata.get("camera_field_of_view"), "camera FOV")
    projection = frustum_diagnostic(object_position, target_agent, fov)
    evidence_kind = "visible_instance_mask"
    overlap = 0.0
    if canonical_class == "visible":
        if target_state != "VISIBLE" or target_pixels < config.minimum_visible_target_pixels:
            raise CandidateRejected(
                "visible_contract_failed", f"state={target_state}, pixels={target_pixels}"
            )
        if legacy_class in {"partially_occluded", "partial_occlusion"}:
            occluder_id = str(metadata.get("occluder_id") or "")
            if not occluder_id:
                raise CandidateRejected("partial_missing_occluder")
            overlap, evidence_kind = _overlap_evidence(target_record, target_oid, occluder_id)
            if overlap < config.minimum_partial_overlap:
                raise CandidateRejected(
                    "partial_overlap_below_floor",
                    f"{overlap}<{config.minimum_partial_overlap}",
                )
            _validate_stored_overlap(metadata, overlap)
            evidence_kind = f"partial_{evidence_kind}"
    elif canonical_class == "out_of_fov":
        if target_state != "OUT_OF_FOV" or target_pixels != 0:
            raise CandidateRejected(
                "out_of_fov_contract_failed", f"state={target_state}, pixels={target_pixels}"
            )
        if projection["inside"]:
            raise CandidateRejected("out_of_fov_geometry_disagrees")
        evidence_kind = "independent_out_of_fov_projection"
    else:
        if target_state != "OCCLUDED" or target_pixels != 0:
            raise CandidateRejected(
                "occluded_contract_failed", f"state={target_state}, pixels={target_pixels}"
            )
        if not projection["inside"]:
            raise CandidateRejected("occluded_geometry_disagrees")
        occluder_id = str(metadata.get("occluder_id") or "")
        if not occluder_id:
            raise CandidateRejected("occluded_missing_occluder")
        if occluder_id.casefold().startswith(STRUCTURAL_OCCLUDER_PREFIXES):
            raise CandidateRejected("structural_occluder", occluder_id)
        overlap, overlap_kind = _overlap_evidence(target_record, target_oid, occluder_id)
        if overlap < config.minimum_occluder_overlap:
            raise CandidateRejected(
                "occluder_overlap_below_floor",
                f"{overlap}<{config.minimum_occluder_overlap}",
            )
        _validate_stored_overlap(metadata, overlap)
        evidence_kind = overlap_kind

    robust = None
    item_id = str(row.get("id") or "")
    if item_id in sidecars:
        robust = _apply_robust_guard(
            row,
            sidecars[item_id],
            target_oid,
            canonical_class,
            source_step,
            target_step,
            source_pixels,
            target_pixels,
        )

    stable = _stable_digest(
        component,
        target_oid,
        canonical_class,
        scene_id,
        trajectory_subdir,
        source_step,
        target_step,
        item_id,
    )
    direct_rank = 2 if evidence_kind.endswith("occlusion_evidence") else 1
    primary = (
        target_pixels
        if canonical_class == "visible"
        else -float(projection["signed_margin_degrees"])
        if canonical_class == "out_of_fov"
        else overlap
    )
    evidence = {
        "canonical_class": canonical_class,
        "legacy_class": legacy_class,
        "source_state": source_state,
        "target_state": target_state,
        "source_pixels": source_pixels,
        "target_pixels": target_pixels,
        "projection": projection,
        "occluder_id": str(metadata.get("occluder_id") or ""),
        "occluder_overlap_ratio": overlap,
        "evidence_kind": evidence_kind,
        "robust_clipped_view": robust,
    }
    normalized = _normalized_candidate(
        row,
        legacy_class=legacy_class,
        canonical_class=canonical_class,
        role=role,
        component=component,
        source_line=line_number,
        source_sha256=source_sha256,
        evidence=evidence,
        stable=stable,
    )
    return {
        "_row": normalized,
        "_role": role,
        "_class": canonical_class,
        "_component": component,
        "_target": target_oid,
        "_scene": scene_id,
        "_stable": stable,
        "_quality": (1 if robust is not None else 0, direct_rank, primary, source_pixels),
        "_evidence_kind": evidence_kind,
        "_robust_applied": robust is not None,
    }


def _empty_report(config: TaskCSourceConfig, quotas: Mapping[str, int]) -> dict[str, Any]:
    return {
        "schema_version": "0.1.0",
        "audit": "taskc_hash_pinned_candidate_source",
        "pass": False,
        "source": {"path": str(config.source_path), "expected_sha256": config.expected_source_sha256},
        "split_roles": {"path": str(config.split_roles_path)},
        "robust_clipped_view_sidecars": [],
        "parameters": {
            "minimum_source_pixels": config.minimum_source_pixels,
            "minimum_visible_target_pixels": config.minimum_visible_target_pixels,
            "minimum_occluder_overlap": config.minimum_occluder_overlap,
            "minimum_partial_overlap": config.minimum_partial_overlap,
            "structural_occluder_prefixes": list(STRUCTURAL_OCCLUDER_PREFIXES),
            "edge_tolerance_pixels": EDGE_TOLERANCE_PIXELS,
            "minimum_one_edge_pixels": MINIMUM_ONE_EDGE_PIXELS,
            "minimum_corner_pixels": MINIMUM_CORNER_PIXELS,
        },
        "quota_contract": {
            role: {answer_class: quotas[role] for answer_class in CLASS_ORDER}
            for role in ROLE_ORDER
        },
        "counts": {},
        "rejections": {},
        "row_error_preview": [],
        "capacity": {},
        "pre_evidence_capacity": {},
        "selection": {},
        "fatal_errors": [],
        "output": {
            "path": str(config.output_path),
            "written": False,
            "stale_file_present_on_failure": False,
        },
    }


class StructuralTargetRejected(ValueError):
    """A supplement row whose boxed referent is part of the building.

    This is an expected quality filter, not pipeline corruption, so the
    selector drops the row and counts it rather than failing the build.
    """

    def __init__(self, target_oid: str) -> None:
        super().__init__(f"structural target: {target_oid}")
        self.target_oid = target_oid


class BelowVisibleFloorRejected(Exception):
    """A visible supplement whose View-2 sliver is too small for a person.

    Soft, counted rejection -- not pipeline corruption. The upstream path
    enforces the visible floor during replay; supplements used the pixel
    count only as a QUALITY rank, so with --supplements-only the floor was
    never applied at all and 62-pixel "Visible" gold reached a real rater.
    """


CROSS_CLASS_PRIORITY = {"occluded": 0, "out_of_fov": 1, "visible": 2}
SUPPLEMENT_EVIDENCE_TYPE = "live_exact_mask_and_geometry_replay"


def _supplement_candidate(
    row: Mapping[str, Any],
    *,
    split_scenes: Mapping[str, Mapping[str, Any]],
    path_label: str,
    line_number: int,
    minimum_visible_target_pixels: int = MINIMUM_VISIBLE_TARGET_PIXELS,
) -> dict[str, Any]:
    """Validate one live-mined supplement row into a selector candidate.

    Supplements are this project's own live-verified mined rows; any invalid
    row indicates pipeline corruption, so callers treat failures as fatal.
    """

    if row.get("task_family") != "C":
        raise ValueError(f"{path_label}:{line_number}: not a Task-C row")
    canonical_class = str(row.get("candidate_answer_class") or "")
    if canonical_class not in CLASS_ORDER:
        raise ValueError(f"{path_label}:{line_number}: invalid candidate class")
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("answer_class") != canonical_class:
        raise ValueError(f"{path_label}:{line_number}: metadata class mismatch")
    scene_id = str(row.get("scene_id") or "")
    split = split_scenes.get(scene_id)
    if split is None:
        raise ValueError(f"{path_label}:{line_number}: scene absent from split roles")
    if split.get("drop_exact_alias_nonrepresentative") is True:
        raise ValueError(f"{path_label}:{line_number}: exact-alias scene")
    role = str(split.get("role") or "")
    if row.get("candidate_role") != role:
        raise ValueError(f"{path_label}:{line_number}: stale supplement role")
    component = str(split.get("leakage_component_id") or "")
    source = row.get("_candidate_source")
    if not isinstance(source, Mapping):
        raise ValueError(f"{path_label}:{line_number}: missing candidate source")
    if source.get("physical_component_id") != component:
        raise ValueError(f"{path_label}:{line_number}: stale supplement component")
    evidence = source.get("gold_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("evidence_type") != SUPPLEMENT_EVIDENCE_TYPE:
        raise ValueError(f"{path_label}:{line_number}: missing live gold evidence")
    target_oid = str(row.get("target_oid") or "")
    if not target_oid:
        raise ValueError(f"{path_label}:{line_number}: missing target object")
    if is_structural_target(target_oid):
        raise StructuralTargetRejected(target_oid)
    stable = str(source.get("stable_identity_sha256") or "")
    if not stable:
        raise ValueError(f"{path_label}:{line_number}: missing stable identity")
    if canonical_class == "occluded":
        primary = float(metadata.get("occluder_overlap_ratio") or 0.0)
    elif canonical_class == "visible":
        primary = float(metadata.get("target_pixel_count_target") or 0)
        if primary < minimum_visible_target_pixels:
            raise BelowVisibleFloorRejected()
    else:
        primary = 0.0
    normalized_row = copy.deepcopy(dict(row))
    # Supplements bypass _normalized_candidate, so the canonical presentation
    # is applied here too: same question template, options, and answer string
    # as every upstream row, whatever the miner emitted.
    normalized_row["question"] = QUESTION_TEMPLATE.format(
        noun=display_noun(target_oid.split("|", 1)[0])
    )
    normalized_row["options"] = list(CANONICAL_OPTIONS)
    normalized_row["answer"] = CLASS_TO_ANSWER[canonical_class]
    return {
        "_row": normalized_row,
        "_role": role,
        "_class": canonical_class,
        "_component": component,
        "_target": target_oid,
        "_scene": scene_id,
        "_stable": stable,
        # Rank -1 sorts every supplement after every upstream candidate.
        "_quality": (
            -1,
            primary,
            float(metadata.get("target_pixel_count_source") or 0),
            0.0,
        ),
        "_evidence_kind": "live_mined_supplement",
        "_robust_applied": False,
    }


def prepare_taskc_source(config: TaskCSourceConfig) -> dict[str, Any]:
    """Prepare and audit the frozen private Task-C source.

    The report is always written when its parent directory is writable.  The
    selected JSONL is written only when every source/split invariant passes and
    every exact quota is feasible without replacement.
    """

    quotas = _validated_quotas(config.quotas)
    report = _empty_report(config, quotas)
    fatal_errors: list[str] = report["fatal_errors"]
    source_path = config.source_path.resolve()
    if not source_path.is_file():
        fatal_errors.append(f"source file does not exist: {source_path}")
        report["output"]["stale_file_present_on_failure"] = config.output_path.exists()
        _write_report(config.report_path, report)
        return report
    observed_sha256 = sha256_file(source_path)
    report["source"]["observed_sha256"] = observed_sha256
    report["source"]["bytes"] = source_path.stat().st_size
    if not HEX_SHA256.fullmatch(config.expected_source_sha256):
        fatal_errors.append("expected source SHA-256 is not a lowercase 64-hex digest")
    elif observed_sha256 != config.expected_source_sha256:
        fatal_errors.append(
            f"source hash mismatch: expected {config.expected_source_sha256}, observed {observed_sha256}"
        )
    if fatal_errors:
        report["output"]["stale_file_present_on_failure"] = config.output_path.exists()
        _write_report(config.report_path, report)
        return report

    try:
        split_scenes, split_provenance = _load_split_roles(config.split_roles_path.resolve())
        sidecars, sidecar_provenance = _load_robust_sidecars(
            tuple(path.resolve() for path in config.robust_sidecar_paths)
        )
    except Exception as exc:
        fatal_errors.append(f"{type(exc).__name__}: {exc}")
        report["output"]["stale_file_present_on_failure"] = config.output_path.exists()
        _write_report(config.report_path, report)
        return report
    report["split_roles"] = split_provenance
    report["robust_clipped_view_sidecars"] = sidecar_provenance
    report["counts"]["robust_sidecar_items_available"] = len(sidecars)

    counts: Counter[str] = Counter(report["counts"])
    rejections: Counter[str] = Counter()
    evidence_counts: Counter[str] = Counter()
    error_preview: list[dict[str, Any]] = []
    cache = _JsonLRU()
    candidates: list[dict[str, Any]] = []
    pre_evidence_keys: set[tuple[str, str, str]] = set()
    pre_evidence_roles: dict[tuple[str, str, str], str] = {}
    pre_evidence_rejections: Counter[str] = Counter()
    with source_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                counts["blank_lines"] += 1
                continue
            counts["source_rows"] += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                counts["malformed_json_rows"] += 1
                fatal_errors.append(f"malformed JSON at line {line_number}: {exc}")
                if len(error_preview) < 100:
                    error_preview.append({"line": line_number, "reason": "malformed_json"})
                continue
            if not isinstance(row, Mapping):
                counts["non_object_rows"] += 1
                fatal_errors.append(f"non-object JSON at line {line_number}")
                continue
            if row.get("task_family") != "C":
                counts["non_task_c_rows"] += 1
                continue
            counts["task_c_rows"] += 1
            if config.supplements_only:
                # Single-provenance mode. The legacy template pool renders at a
                # 90-degree vertical FOV along named orbit trajectories, while
                # the all-class miner renders at 120 degrees from jittered
                # poses. Mixing them puts a visible, class-correlated signature
                # in View 1 -- the same two-pipeline provenance that let a
                # pixel probe read the answer from View 1 alone. The upstream
                # file stays bound by SHA-256 for provenance and split roles;
                # only its rows are held out of the pool.
                counts["upstream_rows_excluded_single_provenance"] += 1
                continue
            # Capacity immediately before projection/occlusion-evidence replay.
            # This answers whether replay alone can fill a cell, without
            # allowing stored projection flags or stored labels to prove it.
            metadata = row.get("metadata")
            split = split_scenes.get(str(row.get("scene_id") or ""))
            legacy_class = (
                str(metadata.get("answer_class") or "")
                if isinstance(metadata, Mapping)
                else ""
            )
            canonical_class = LEGACY_CLASS_MAP.get(legacy_class)
            if split is None:
                pre_evidence_rejections["scene_absent_from_split_roles"] += 1
            elif split.get("drop_exact_alias_nonrepresentative") is True:
                pre_evidence_rejections["exact_alias_nonrepresentative"] += 1
            elif canonical_class is None or not isinstance(metadata, Mapping):
                pre_evidence_rejections["invalid_class_or_metadata"] += 1
            elif int(metadata.get("target_pixel_count_source") or 0) < config.minimum_source_pixels:
                pre_evidence_rejections["source_pixels_below_floor"] += 1
            elif canonical_class == "occluded" and str(
                metadata.get("occluder_id") or ""
            ).casefold().startswith(STRUCTURAL_OCCLUDER_PREFIXES):
                pre_evidence_rejections["structural_occluder"] += 1
            else:
                component = str(split.get("leakage_component_id") or "")
                role = str(split.get("role") or "")
                target_oid = str(row.get("target_oid") or "")
                if component and role in ROLE_ORDER and target_oid:
                    key = (component, target_oid, canonical_class)
                    old_role = pre_evidence_roles.setdefault(key, role)
                    if old_role != role:
                        fatal_errors.append(
                            "pre-evidence role crossover within uniqueness key " + repr(key)
                        )
                    pre_evidence_keys.add(key)
                else:
                    pre_evidence_rejections["invalid_role_component_or_target"] += 1
            try:
                candidate = _evaluate_row(
                    row,
                    line_number=line_number,
                    source_sha256=observed_sha256,
                    split_scenes=split_scenes,
                    sidecars=sidecars,
                    scenes_root=config.scenes_root.resolve(),
                    cache=cache,
                    config=config,
                )
            except CandidateRejected as exc:
                rejections[exc.reason] += 1
                if len(error_preview) < 100:
                    error_preview.append(
                        {
                            "line": line_number,
                            "id": row.get("id"),
                            "reason": exc.reason,
                            "detail": exc.detail,
                        }
                    )
                continue
            except Exception as exc:
                rejections["unexpected_row_error"] += 1
                fatal_errors.append(
                    f"unexpected row error at line {line_number}: {type(exc).__name__}: {exc}"
                )
                if len(error_preview) < 100:
                    error_preview.append(
                        {
                            "line": line_number,
                            "id": row.get("id"),
                            "reason": "unexpected_row_error",
                            "detail": f"{type(exc).__name__}: {exc}",
                        }
                    )
                continue
            candidates.append(candidate)
            counts["evidence_eligible_rows"] += 1
            if candidate["_robust_applied"]:
                counts["robust_sidecar_rows_applied"] += 1
            else:
                counts["rows_without_available_robust_sidecar"] += 1
            evidence_counts[candidate["_evidence_kind"]] += 1

    component_roles: dict[str, set[str]] = defaultdict(set)
    for candidate in candidates:
        component_roles[candidate["_component"]].add(candidate["_role"])
    role_crossovers = {
        component: sorted(roles)
        for component, roles in sorted(component_roles.items())
        if len(roles) > 1
    }
    if role_crossovers:
        fatal_errors.append(
            f"{len(role_crossovers)} physical components cross split roles"
        )

    by_unique_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for candidate in candidates:
        key = (candidate["_component"], candidate["_target"], candidate["_class"])
        old = by_unique_key.get(key)
        if old is None:
            by_unique_key[key] = candidate
            continue
        counts["duplicate_physical_target_class_rows"] += 1
        if old["_role"] != candidate["_role"]:
            fatal_errors.append(
                "role crossover within uniqueness key " + repr(key)
            )
            continue
        if _candidate_sort_key(candidate) < _candidate_sort_key(old):
            by_unique_key[key] = candidate
    unique_candidates = list(by_unique_key.values())
    counts["eligible_unique_physical_target_class"] = len(unique_candidates)

    supplement_provenance: list[dict[str, Any]] = []
    supplement_accepted: Counter[tuple[str, str]] = Counter()
    for supplement_path in tuple(config.supplement_jsonl_paths):
        resolved = supplement_path.resolve()
        entry: dict[str, Any] = {
            "path": str(supplement_path),
            "sha256": sha256_file(resolved),
            "rows": 0,
            "accepted_new_identities": 0,
            "duplicates_of_upstream_or_prior": 0,
            "structural_targets_dropped": 0,
            "below_visible_floor_dropped": 0,
        }
        try:
            with resolved.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    entry["rows"] += 1
                    try:
                        candidate = _supplement_candidate(
                            json.loads(line),
                            split_scenes=split_scenes,
                            path_label=str(supplement_path),
                            line_number=line_number,
                            minimum_visible_target_pixels=(
                                config.minimum_visible_target_pixels
                            ),
                        )
                    except BelowVisibleFloorRejected:
                        entry["below_visible_floor_dropped"] = (
                            entry.get("below_visible_floor_dropped", 0) + 1
                        )
                        continue
                    except StructuralTargetRejected:
                        entry["structural_targets_dropped"] = (
                            entry.get("structural_targets_dropped", 0) + 1
                        )
                        continue
                    key = (
                        candidate["_component"],
                        candidate["_target"],
                        candidate["_class"],
                    )
                    old_candidate = by_unique_key.get(key)
                    if old_candidate is None:
                        by_unique_key[key] = candidate
                        supplement_accepted[
                            (candidate["_role"], candidate["_class"])
                        ] += 1
                        entry["accepted_new_identities"] += 1
                    elif old_candidate["_quality"][0] == -1 and _candidate_sort_key(
                        candidate
                    ) < _candidate_sort_key(old_candidate):
                        by_unique_key[key] = candidate
                        entry["duplicates_of_upstream_or_prior"] += 1
                    else:
                        entry["duplicates_of_upstream_or_prior"] += 1
        except Exception as exc:
            fatal_errors.append(
                f"invalid supplement {supplement_path}: {type(exc).__name__}: {exc}"
            )
        supplement_provenance.append(entry)
    report["supplements"] = {
        "files": supplement_provenance,
        "accepted_new_identities_by_role_class": {
            f"{role}:{answer_class}": count
            for (role, answer_class), count in sorted(supplement_accepted.items())
        },
        "ordering_policy": (
            "upstream replayed candidates outrank live-mined supplements; "
            "supplements fill remaining quota deterministically by class "
            "quality then stable identity"
        ),
    }

    # DC-6: at most one selected row per physical component-target identity
    # PER CLASS, so one target may appear at most three times -- once with each
    # answer.
    #
    # The rule used to collapse each identity to a single row across all three
    # classes, which was right while the classes came from different targets.
    # Once one miner produces all three for the same target it backfires: it
    # pruned 25,624 rows and, for the 20,752 identities that had more than one
    # class, decided the class by a tie-break. The target's own properties then
    # predicted the answer, and a View-1 pixel probe read it at 43% against 33%
    # chance. Keying on the class instead bounds an object's contribution at
    # three (the original over-representation concern) while letting selection
    # draw target-matched triples, where the same object carries all three
    # answers and therefore leaks none of them.
    identity_map: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in by_unique_key.values():
        identity_map[
            (candidate["_component"], candidate["_target"], candidate["_class"])
        ].append(candidate)
    cross_class_pruned: Counter[str] = Counter()
    final_candidates: list[dict[str, Any]] = []
    for identity_candidates in identity_map.values():
        if len(identity_candidates) == 1:
            final_candidates.append(identity_candidates[0])
            continue
        keep = min(
            identity_candidates,
            key=lambda item: (
                CROSS_CLASS_PRIORITY[item["_class"]],
                _candidate_sort_key(item),
            ),
        )
        final_candidates.append(keep)
        for item in identity_candidates:
            if item is not keep:
                cross_class_pruned[item["_class"]] += 1
    report["cross_class_identity_cap"] = {
        "policy": (
            "one row per (component, target, class): an object appears at most "
            "once per answer, so it can contribute three rows and predicts none"
        ),
        "identities_with_multiple_classes": sum(
            1 for items in identity_map.values() if len(items) > 1
        ),
        "pruned_by_class": dict(sorted(cross_class_pruned.items())),
    }
    counts["cross_class_identity_pruned"] = sum(cross_class_pruned.values())

    # DC-6c: prefer TARGET-MATCHED TRIPLES. When one physical target yields all
    # three classes, taking all three makes the target itself carry no class
    # information: its object, its room, its clutter and its lighting are
    # identical across the three rows, so a View-1 probe has nothing to read.
    # Without this the class was effectively assigned by which rows a target
    # happened to produce -- objects visible from many standpoints became
    # "visible" items, cornered ones became "out_of_fov" -- and a DINOv2 probe
    # read View 1 alone at 43% against 33% chance, still 52% within a single
    # object type. Targets with fewer than three classes are kept only to top
    # up a class the triples cannot fill.
    triple_index: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for candidate in final_candidates:
        triple_index[(candidate["_component"], candidate["_target"])][
            candidate["_class"]
        ] = candidate
    # A triple must also share one View-1 image, byte for byte. The miner
    # renders a single source view and derives all three answers from it, so
    # this normally holds already -- but a target occasionally picks up its
    # third class from a second standpoint, and those rows do carry a faint
    # View-1 signal. Requiring identical source-frame bytes converts the
    # guarantee from statistical to structural: the three answers are attached
    # to the SAME pixels, so no View-1-only predictor can beat chance, and that
    # is checkable by hashing the release rather than by trusting a probe.
    def _view_one_sha(candidate: dict[str, Any]) -> str | None:
        return ((candidate["_row"].get("_candidate_source") or {})
                .get("source_frame_sha256"))

    complete = {}
    split_view_one = 0
    for key, classes in triple_index.items():
        if len(classes) != len(CLASS_ORDER):
            continue
        shas = {_view_one_sha(classes[name]) for name in CLASS_ORDER}
        if len(shas) != 1 or None in shas:
            split_view_one += 1
            continue
        complete[key] = classes
    for candidate in final_candidates:
        candidate["_triple_complete"] = (
            candidate["_component"], candidate["_target"]
        ) in complete
    report["target_matched_triples"] = {
        "rejected_for_split_view_one": split_view_one,
        "selection_policy": (
            "complete triples are taken first, so the same target supplies all "
            "three answers; cue-matched rows only top up a shortfall"
        ),
        "policy": (
            "one target contributing all three classes is preferred, so target "
            "identity and its scene carry no class information"
        ),
        "distinct_targets": len(triple_index),
        "targets_with_all_three_classes": len(complete),
        "rows_in_complete_triples": len(complete) * len(CLASS_ORDER),
        "targets_by_role": dict(
            Counter(
                classes[next(iter(classes))]["_role"] for classes in complete.values()
            )
        ),
    }
    counts["targets_with_all_three_classes"] = len(complete)

    # DC-6b: an exact rendered frame may back at most one selected row. The
    # miners legitimately reuse a view across targets -- one View-2 render
    # answers a different question for every object in it -- but releasing the
    # same PNG under several items makes the items non-independent and, worse,
    # ships one image with two different gold answers (73 such groups in the
    # 2026-08-23 pool). The candidate builder rejects duplicate released
    # content outright, so the cap belongs here, before quotas are measured.
    frame_owner: dict[str, dict[str, Any]] = {}
    frame_pruned: Counter[str] = Counter()

    def _frame_hashes(candidate: Mapping[str, Any]) -> list[str]:
        source = candidate["_row"].get("_candidate_source") or {}
        return [
            str(source[key])
            for key in ("source_frame_sha256", "target_frame_sha256")
            if source.get(key)
        ]

    content_ordered = sorted(
        final_candidates,
        key=lambda item: (CROSS_CLASS_PRIORITY[item["_class"]], _candidate_sort_key(item)),
    )
    content_kept: list[dict[str, Any]] = []
    for candidate in content_ordered:
        hashes = _frame_hashes(candidate)
        if any(digest in frame_owner for digest in hashes):
            frame_pruned[candidate["_class"]] += 1
            continue
        for digest in hashes:
            frame_owner[digest] = candidate
        content_kept.append(candidate)
    report["exact_frame_content_cap"] = {
        "policy": "one selected row per exact rendered frame, either view",
        "rows_before": len(final_candidates),
        "rows_after": len(content_kept),
        "pruned_by_class": dict(sorted(frame_pruned.items())),
        "rows_without_frame_hashes": sum(
            1 for candidate in final_candidates if not _frame_hashes(candidate)
        ),
    }
    counts["exact_frame_content_pruned"] = sum(frame_pruned.values())
    final_candidates = content_kept

    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in final_candidates:
        pools[(candidate["_role"], candidate["_class"])].append(candidate)
    for pool in pools.values():
        pool.sort(key=lambda item: (not item.get("_triple_complete", False),
                                    _candidate_sort_key(item)))

    # Construction-cue matching. The three classes come from two different
    # pipelines (trajectory replay for visible/out_of_fov, constructed
    # occlusion search for most occluded rows), and a balanced probe read the
    # pipeline -- hence the class -- from View-1 statistics alone: 79% from
    # the source pixel count, with the box centre similarly skewed before the
    # miner gained aim jitter. Selection therefore keeps an equal number of
    # every (object type x source-pixel bin x box-centre bin) cell in every
    # class, so the noun, the target's apparent size, and its frame position
    # carry zero class information by construction. Bin edges are pooled
    # quantiles of the final candidates and are recorded in the report.
    def _cue_features(candidate: Mapping[str, Any]) -> tuple[float, float] | None:
        evidence = candidate["_row"]["_candidate_source"]["gold_evidence"]
        source = evidence.get("source") or {}
        bbox = source.get("bbox_xyxy")
        pixels = source.get("pixel_count")
        if not (isinstance(bbox, Sequence) and len(bbox) == 4) or pixels is None:
            return None
        size = (
            evidence.get("image_size")
            or (evidence.get("projection") or {}).get("image_size")
            or {"w": 640, "h": 480}
        )
        centre_x = (float(bbox[0]) + float(bbox[2])) / 2.0 / float(size["w"])
        return math.log(max(1.0, float(pixels))), centre_x

    def _quantile_edges(values: list[float], bins: int) -> list[float]:
        ordered = sorted(values)
        return [ordered[int(len(ordered) * k / bins)] for k in range(1, bins)]

    def _bin_index(value: float, edges: list[float]) -> int:
        for index, edge in enumerate(edges):
            if value < edge:
                return index
        return len(edges)

    featured = [
        (candidate, features)
        for candidate in final_candidates
        if (features := _cue_features(candidate)) is not None
    ]
    cue_ineligible = Counter(
        (candidate["_role"], candidate["_class"])
        for candidate in final_candidates
        if _cue_features(candidate) is None
    )
    matching_engaged = bool(featured)
    pixel_edges = (
        _quantile_edges([f[0] for _, f in featured], 4) if matching_engaged else []
    )
    centre_edges = (
        _quantile_edges([f[1] for _, f in featured], 3) if matching_engaged else []
    )

    def _cue_cell(candidate: Mapping[str, Any]) -> str:
        pixels_log, centre_x = _cue_features(candidate)
        return (
            f"{object_type_of(candidate['_target'])}"
            f"|p{_bin_index(pixels_log, pixel_edges)}"
            f"|x{_bin_index(centre_x, centre_edges)}"
        )

    # Complete triples are already stronger than anything the cue matcher can
    # achieve: the same physical target carries all three answers, so object
    # type, room, clutter and lighting are identical across the classes rather
    # than merely equalised bin-by-bin. Selecting triples first is therefore
    # both the stricter control AND the larger pool -- matching cells cost
    # 88 items per hidden class where triples offer thousands. The cue matcher
    # is kept for the top-up, where rows are not triple-backed.
    triples_by_role: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key, classes in complete.items():
        triples_by_role[classes[CLASS_ORDER[0]]["_role"]].append(key)
    for role, keys in triples_by_role.items():
        keys.sort(key=lambda k: min(
            _candidate_sort_key(complete[k][name]) for name in CLASS_ORDER
        ))

    matched_by_role_class: dict[tuple[str, str], list[dict[str, Any]]] = {}
    matching_report: dict[str, Any] = {
        "engaged": matching_engaged,
        "policy": (
            "equal count per (object_type x source-pixel bin x box-centre "
            "bin) in every class; cells absent from any class are dropped"
        ),
        "pixel_log_bin_edges": [round(edge, 6) for edge in pixel_edges],
        "centre_x_bin_edges": [round(edge, 6) for edge in centre_edges],
        "ineligible_missing_source_stats": {
            f"{role}:{answer_class}": count
            for (role, answer_class), count in sorted(cue_ineligible.items())
        },
        "roles": {},
    }
    if matching_engaged:
        for role in ROLE_ORDER:
            eligible = [c for c, _features in featured if c["_role"] == role]
            matched, role_match_report = match_object_type_distribution(
                eligible,
                classes=CLASS_ORDER,
                class_of=lambda item: item["_class"],
                type_of=_cue_cell,
                order_key=_candidate_sort_key,
                per_class_cap=quotas[role],
            )
            for candidate in matched:
                matched_by_role_class.setdefault(
                    (role, candidate["_class"]), []
                ).append(candidate)
            matching_report["roles"][role] = role_match_report
            if not role_match_report["type_signal_removed"]:
                raise ValueError(
                    f"{role}: cue matching left a residual composite-cell signal"
                )
    else:
        # Only reachable when no candidate carries source statistics at all
        # (synthetic fixtures). A real build must engage matching; the audit
        # asserts on this flag.
        for candidate in final_candidates:
            matched_by_role_class.setdefault(
                (candidate["_role"], candidate["_class"]), []
            ).append(candidate)
    report["construction_cue_matching"] = matching_report

    capacity: dict[str, dict[str, Any]] = {}
    missing_cells: list[dict[str, Any]] = []
    selected: list[dict[str, Any]] = []
    selection_report: dict[str, dict[str, Any]] = {}
    for role in ROLE_ORDER:
        capacity[role] = {}
        selection_report[role] = {}
        for answer_class in CLASS_ORDER:
            triple_rows = [
                complete[key][answer_class]
                for key in triples_by_role.get(role, [])[: quotas[role]]
            ]
            used_ids = {row["_stable"] for row in triple_rows}
            top_up = [
                row for row in sorted(
                    matched_by_role_class.get((role, answer_class), []),
                    key=_candidate_sort_key,
                )
                if row["_stable"] not in used_ids
            ]
            pool = triple_rows + top_up
            required = quotas[role]
            available = len(pool)
            feasible = available >= required
            capacity[role][answer_class] = {
                "available_unique": available,
                "unmatched_pool_available": len(pools[(role, answer_class)]),
                "required": required,
                "surplus": available - required,
                "feasible": feasible,
            }
            if not feasible:
                missing_cells.append(
                    {
                        "role": role,
                        "answer_class": answer_class,
                        "available": available,
                        "required": required,
                        "shortfall": required - available,
                    }
                )
                selection_report[role][answer_class] = {"selected": 0}
                continue
            chosen = pool[:required]
            selected.extend(chosen)
            selection_report[role][answer_class] = {
                "selected": len(chosen),
                "scenes": len({item["_scene"] for item in chosen}),
                "physical_components": len({item["_component"] for item in chosen}),
                "targets": len({item["_target"] for item in chosen}),
            }

    report["counts"] = dict(sorted(counts.items()))
    report["rejections"] = dict(sorted(rejections.items()))
    report["evidence_kind_counts"] = dict(sorted(evidence_counts.items()))
    report["row_error_preview"] = error_preview
    report["role_crossovers"] = role_crossovers
    pre_evidence_counts: dict[str, dict[str, int]] = {
        role: {answer_class: 0 for answer_class in CLASS_ORDER} for role in ROLE_ORDER
    }
    for key in pre_evidence_keys:
        pre_evidence_counts[pre_evidence_roles[key]][key[2]] += 1
    report["pre_evidence_capacity"] = {
        "definition": (
            "unique (physical_component,target,class) after split-role, exact-alias, "
            "source-pixel, and known-structural-occluder gates; before independent "
            "projection and explicit occlusion-evidence replay"
        ),
        "counts": pre_evidence_counts,
        "rejections": dict(sorted(pre_evidence_rejections.items())),
    }
    report["capacity"] = capacity
    report["selection"] = selection_report
    report["missing_quota_cells"] = missing_cells

    pass_audit = not fatal_errors and not missing_cells
    if pass_audit:
        output_rows = [candidate["_row"] for candidate in selected]
        expected_rows = sum(quotas.values()) * len(CLASS_ORDER)
        if len(output_rows) != expected_rows:
            fatal_errors.append(
                f"internal selection count mismatch: {len(output_rows)}/{expected_rows}"
            )
            pass_audit = False
        unique_output = {
            (
                row["_candidate_source"]["physical_component_id"],
                row["target_oid"],
                row["metadata"]["answer_class"],
            )
            for row in output_rows
        }
        if len(unique_output) != len(output_rows):
            fatal_errors.append("selected output violates physical-component/target/class uniqueness")
            pass_audit = False
        # Target identity must carry no information about the answer. The old
        # contract enforced this by letting each physical object appear under a
        # single class -- which is exactly backwards: a target seen only as
        # "occluded" IS the leak, because memorising the object then predicts
        # the label. Target-matched triples invert it, putting the same object
        # under all three answers, so identity becomes uninformative by
        # construction rather than by exclusion. The check below therefore
        # measures the leak directly: an oracle that knows the physical target
        # and guesses its most frequent class must sit at chance.
        identity_classes: dict[tuple[str, str], Counter] = defaultdict(Counter)
        for row in output_rows:
            identity_classes[
                (
                    row["_candidate_source"]["physical_component_id"],
                    row["target_oid"],
                )
            ][row["metadata"]["answer_class"]] += 1
        identity_oracle = sum(
            counts.most_common(1)[0][1] for counts in identity_classes.values()
        ) / max(1, len(output_rows))
        complete = sum(1 for c in identity_classes.values() if len(c) == len(CLASS_ORDER))
        report["target_identity_oracle"] = {
            "accuracy": round(identity_oracle, 6),
            "chance": round(1.0 / len(CLASS_ORDER), 6),
            "tolerance": IDENTITY_ORACLE_TOLERANCE,
            "distinct_physical_targets": len(identity_classes),
            "targets_covering_all_classes": complete,
        }
        # The statistic is only interpretable once each class holds enough rows
        # for triples to form at all: in a handful of rows every target is
        # necessarily single-class and the oracle reads 1.0 with no leak to
        # speak of. It is always reported; it is enforced at benchmark scale,
        # which every released selection is far above.
        enforce_oracle = len(output_rows) >= IDENTITY_ORACLE_MIN_ROWS
        report["target_identity_oracle"]["enforced"] = enforce_oracle
        if enforce_oracle and identity_oracle > 1.0 / len(CLASS_ORDER) + IDENTITY_ORACLE_TOLERANCE:
            fatal_errors.append(
                "physical-target identity predicts the answer class above chance "
                f"({identity_oracle:.4f} > {1.0 / len(CLASS_ORDER):.4f} + "
                f"{IDENTITY_ORACLE_TOLERANCE})"
            )
            pass_audit = False
    if pass_audit:
        _write_jsonl(config.output_path, output_rows)
        report["output"] = {
            "path": str(config.output_path),
            "written": True,
            "rows": len(output_rows),
            "sha256": sha256_file(config.output_path),
            "stale_file_present_on_failure": False,
        }
        report["pass"] = True
    else:
        report["output"]["stale_file_present_on_failure"] = config.output_path.exists()
    _write_report(config.report_path, report)
    return report
