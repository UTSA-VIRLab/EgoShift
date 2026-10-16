"""Deterministic, fail-closed S05 construction for the Task-B candidate.

This builder consumes only hash-pinned outputs of the corrected static pose
geometry index.  It uses a binary mixed-integer program so that exact finite
support balance and all diversity caps are simultaneous hard constraints.  An
infeasible problem is reported as a capacity blocker; no gate is relaxed.
"""

from __future__ import annotations

import json
import math
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .hashing import safe_relative_path, sha256_file, sha256_json
from .split_roles import require_complete_split_roles
from .support import GeometryConfig, analytic_support, cell_construction_diagnostics
from .taskb_geometry import canonical_zone, post_action_bearing_degrees


SCHEMA_VERSION = "0.1.0"
ROLES = ("hidden_test", "public_dev", "tuning_dev")
ROTATIONS = (30, 60, 90, 120, 180)
TURNS = ("left", "right")
PUBLIC_FIELDS = ("id", "task_family", "format", "frames", "question", "options")


class CandidateCapacityError(RuntimeError):
    """Raised when the frozen hard constraints have no feasible candidate."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


@dataclass(frozen=True)
class SourceSpec:
    name: str
    kind: str
    index_path: Path
    geometry_report_path: Path
    asset_root: Path
    asset_trust_root: Path
    upstream_merge_report_path: Path | None = None


@dataclass(frozen=True)
class Assignment:
    candidate_index: int
    rotation_degrees: int
    turn_direction: str
    answer: str

    @property
    def stratum(self) -> tuple[int, str, str]:
        return (self.rotation_degrees, self.turn_direction, self.answer)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _canonical_category(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold()) or "object"


def _display_category(value: str) -> str:
    known = {
        "countertop": "countertop",
        "doorframe": "doorframe",
        "floorlamp": "floor lamp",
        "desklamp": "desk lamp",
        "tvstand": "TV stand",
        "houseplant": "house plant",
        "garbagecan": "garbage can",
        "coffeemachine": "coffee machine",
        "teddybear": "teddy bear",
        "sidetable": "side table",
        "diningtable": "dining table",
        "washingmachine": "washing machine",
        "clothesdryer": "clothes dryer",
        "laundryhamper": "laundry hamper",
        "baseballbat": "baseball bat",
        "alarmclock": "alarm clock",
        "dishsponge": "dish sponge",
        "papertowelroll": "paper towel roll",
        "toiletpaper": "toilet paper",
        "winebottle": "wine bottle",
        "soapbottle": "soap bottle",
        "spraybottle": "spray bottle",
        "peppershaker": "pepper shaker",
        "saltshaker": "salt shaker",
        "vacuumcleaner": "vacuum cleaner",
        "tissuebox": "tissue box",
        "coffeetable": "coffee table",
    }
    canonical = _canonical_category(value)
    if canonical in known:
        return known[canonical]
    return (
        re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
        .replace("_", " ")
        .strip()
        .casefold()
        or "object"
    )


def _resolve_source_asset(source: SourceSpec, relative: str) -> Path:
    """Resolve a source asset while permitting the declared in-workspace symlink forest.

    The reservoir intentionally mixes physical directories with scene-level
    symlinks into ``data/sim/taskb_stage``.  Lexical traversal is still
    forbidden, and every resolved target must remain below the explicit trust
    root supplied by the caller.
    """

    normalized = safe_relative_path(relative)
    candidate = (source.asset_root.absolute() / normalized).resolve()
    trust_root = source.asset_trust_root.resolve()
    try:
        candidate.relative_to(trust_root)
    except ValueError as exc:
        raise ValueError(
            f"source asset escapes declared trust root: {relative!r} -> {candidate}"
        ) from exc
    return candidate


def _require_close(label: str, observed: object, expected: object, tolerance: float = 1e-9) -> None:
    try:
        matches = math.isclose(
            float(observed), float(expected), rel_tol=0.0, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        matches = False
    if not matches:
        raise ValueError(f"{label} mismatch: observed={observed!r}, expected={expected!r}")


def validate_contract(config: Mapping[str, Any]) -> None:
    """Reject contract edits that would silently weaken the frozen S05 build."""

    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected Task-B candidate contract schema")
    geometry = config.get("geometry") or {}
    _require_close(
        "source-bearing half-width",
        geometry.get("source_bearing_half_width_degrees"),
        66.58677555362947,
        1e-12,
    )
    if geometry.get("source_bearing_boundary_policy") != "strict":
        raise ValueError("source-bearing boundary policy must remain strict")
    # Re-frozen 2026-08-22 to the MILP-CERTIFIED allocation (exact solver on
    # the mined reservoir+r2c+r2f pools under split_roles_v3, all caps and
    # content dedup active; audit: taskb_certified_allocation_v3.json).
    # Previously the r2e/r2f MINING DRAFT: per-rotation sizes are
    # the v3-component-scaled aspiration (x1.3 overshoot) that drives the
    # expansion mining sweep. Not a certified size; the post-cap MILP
    # certifies the real allocation and this table is re-frozen then.
    # Previously: The uniform 78/12/20 allocation is
    # not constructible: the 180-degree cells, and to a lesser degree the
    # 90-degree cells, need a source bearing in a narrow band at the edge of the
    # declared cone, and an exhaustive sweep of every available scene returned
    # 16 and 13 rows against the 52 required. Per-rotation sizes are pinned here
    # alongside the totals so neither can drift silently.
    expected_splits = {
        "hidden_test": (135, 1098, 2, 1, 1, 55, True),
        "public_dev": (54, 440, 2, 1, 1, 22, False),
        "tuning_dev": (78, 592, 2, 1, 1, 30, False),
    }
    expected_by_rotation = {
        "hidden_test": {30: 135, 60: 135, 90: 117, 120: 134, 180: 28},
        "public_dev": {30: 54, 60: 54, 90: 47, 120: 54, 180: 11},
        "tuning_dev": {30: 78, 60: 77, 90: 51, 120: 76, 180: 14},
    }
    splits = config.get("splits") or {}
    if set(splits) != set(expected_splits):
        raise ValueError("Task-B split set differs from the frozen contract")
    keys = (
        "items_per_action_cell",
        "total_items",
        "maximum_per_physical_component",
        "maximum_per_source_frame",
        "maximum_per_target_object",
        "maximum_single_target_category_items",
        "requires_never_outcome_exposed_component",
    )
    for role, values in expected_splits.items():
        observed = tuple(splits[role].get(key) for key in keys)
        if observed != values:
            raise ValueError(
                f"{role} contract differs from frozen quotas/caps: {observed!r}"
            )
        by_rotation = {
            int(k): int(v)
            for k, v in (splits[role].get("items_per_action_cell_by_rotation") or {}).items()
        }
        if by_rotation != expected_by_rotation[role]:
            raise ValueError(
                f"{role} per-rotation sizes differ from the frozen contract: {by_rotation!r}"
            )
        declared = sum(
            by_rotation.get(rotation, values[0])
            for rotation in ROTATIONS
            for _turn in TURNS
        )
        if declared != values[1]:
            raise ValueError(
                f"{role} per-rotation sizes sum to {declared}, not {values[1]}"
            )
    actions = config.get("actions") or {}
    if tuple(actions.get("rotations_degrees") or ()) != ROTATIONS:
        raise ValueError("rotation set differs from frozen contract")
    if tuple(actions.get("presented_turn_directions") or ()) != TURNS:
        raise ValueError("turn-direction set differs from frozen contract")
    if tuple((config.get("presentation") or {}).get("public_question_fields") or ()) != PUBLIC_FIELDS:
        raise ValueError("public question allowlist differs from frozen contract")


def action_supports(half_width: float) -> dict[tuple[int, str], tuple[str, ...]]:
    geometry = GeometryConfig(source_bearing_half_width_degrees=half_width)
    return {
        (rotation, turn): analytic_support(rotation, turn, geometry)
        for rotation in ROTATIONS
        for turn in TURNS
    }


def finite_label_quotas(
    *,
    role: str,
    items_per_action_cell: int,
    seed: str,
    half_width: float,
    items_by_rotation: Mapping[int, int] | None = None,
) -> dict[tuple[int, str, str], int]:
    """Freeze an attainable floor/ceil label allocation for every action cell.

    ``items_by_rotation`` optionally overrides the per-cell size for individual
    rotations. Cell sizes need not be equal: the finite target
    ``N^-1 sum_z ceil(n_z / |S(z)|)`` is defined for any allocation, and a
    rotation whose feasible support cannot be constructed at scale must be
    allowed a smaller size rather than be filled with an unbalanced label mix.
    Omitting the mapping reproduces the uniform allocation exactly.
    """

    overrides = {int(k): int(v) for k, v in (items_by_rotation or {}).items()}
    result: dict[tuple[int, str, str], int] = {}
    for (rotation, turn), support in action_supports(half_width).items():
        cell_items = overrides.get(int(rotation), items_per_action_cell)
        base, remainder = divmod(cell_items, len(support))
        ordered = sorted(
            support,
            key=lambda label: sha256_json(
                [seed, "finite-remainder", role, rotation, turn, label]
            ),
        )
        plus_one = set(ordered[:remainder])
        for label in support:
            result[(rotation, turn, label)] = base + int(label in plus_one)
    return result


def _validate_geometry_report(
    spec: SourceSpec, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    index_path = spec.index_path.resolve()
    report_path = spec.geometry_report_path.resolve()
    report = _load_json(report_path)
    if report.get("audit") != "taskb_static_pose_geometry_source_index" or report.get("pass") is not True:
        raise ValueError(f"{spec.name}: geometry-index audit is absent or failing")
    if report.get("source_kind") != spec.kind:
        raise ValueError(f"{spec.name}: source kind disagrees with geometry-index audit")
    observed_index_hash = sha256_file(index_path)
    if report.get("output", {}).get("sha256") != observed_index_hash:
        raise ValueError(f"{spec.name}: geometry-index output hash mismatch")
    parameters = report.get("parameters") or {}
    gates = config["geometry"]
    comparisons = {
        "source_bearing_half_width_degrees": gates["source_bearing_half_width_degrees"],
        "minimum_visible_pixels": gates["minimum_visible_pixels"],
        "minimum_whole_box_boundary_margin_degrees": gates[
            "minimum_whole_box_boundary_margin_degrees"
        ],
        "same_type_pixel_threshold": gates["same_type_visible_pixel_threshold"],
        "render_vertical_fov_degrees": gates["render_vertical_fov_degrees"],
    }
    if spec.kind == "reservoir":
        comparisons["minimum_coverage"] = gates[
            "minimum_reservoir_silhouette_coverage"
        ]
    else:
        comparisons["minimum_target_mask_edge_margin_pixels_for_targeted"] = gates[
            "minimum_targeted_mask_edge_margin_pixels"
        ]
    for key, expected in comparisons.items():
        _require_close(f"{spec.name} geometry report {key}", parameters.get(key), expected)
    upstream: dict[str, Any] = {}
    if spec.kind == "targeted":
        if spec.upstream_merge_report_path is None:
            raise ValueError("targeted source requires the fail-closed merge report")
        merge_path = spec.upstream_merge_report_path.resolve()
        upstream = _load_json(merge_path)
        if upstream.get("audit") != "taskb_targeted_r2_fail_closed_merge":
            raise ValueError("targeted upstream merge audit is absent or failing")
        # The builder requires the trust-boundary result (every merged row
        # independently re-verified), not contract fulfilment: a short mining
        # round is legitimate source material, and this builder's own capacity
        # solver is what fails closed when the pooled sources cannot fill the
        # finite quotas.
        verification = upstream.get("verification_pass")
        if verification is None:
            verification = upstream.get("pass")
        if verification is not True:
            raise ValueError("targeted upstream merge audit is absent or failing")
        if report.get("input", {}).get("sha256") != upstream.get("output", {}).get("sha256"):
            raise ValueError("targeted merge-to-geometry hash chain is broken")
        upstream = {
            "path": str(merge_path),
            "sha256": sha256_file(merge_path),
            "merged_source_sha256": upstream["output"]["sha256"],
            "verification_pass": True,
            "quota_pass": upstream.get("quota_pass", upstream.get("pass")),
            "unmet_global_quotas": upstream.get("unmet_global_quotas") or {},
        }
    return report, {
        "name": spec.name,
        "kind": spec.kind,
        "index": {"path": str(index_path), "sha256": observed_index_hash},
        "geometry_report": {"path": str(report_path), "sha256": sha256_file(report_path)},
        "asset_root": str(spec.asset_root.resolve()),
        "asset_trust_root": str(spec.asset_trust_root.resolve()),
        "upstream_merge_report": upstream or None,
    }


def _validate_source_row(
    row: dict[str, Any],
    *,
    source: SourceSpec,
    line_number: int,
    roles: Mapping[str, Any],
    config: Mapping[str, Any],
    supports: Mapping[tuple[int, str], tuple[str, ...]],
    expected_upstream_source_hash: str,
) -> dict[str, Any]:
    label = f"{source.name} line {line_number}"
    if row.get("source_kind") != source.kind:
        raise ValueError(f"{label}: source_kind mismatch")
    if row.get("gate1_pre") is not True:
        raise ValueError(f"{label}: upstream gate1_pre is not true")
    if row.get("source_file_sha256") != expected_upstream_source_hash:
        raise ValueError(f"{label}: upstream source hash disagrees with geometry audit")
    scene_id = str(row.get("scene_id") or "")
    role_record = roles.get(scene_id)
    if not isinstance(role_record, dict):
        raise ValueError(f"{label}: scene absent from frozen role ledger")
    if role_record.get("drop_exact_alias_nonrepresentative") is not False:
        raise ValueError(f"{label}: exact-alias nonrepresentative is forbidden")
    role = str(role_record.get("role"))
    if role not in ROLES:
        raise ValueError(f"{label}: invalid split role")
    if role == "hidden_test" and role_record.get("outcome_exposed_component") is not False:
        raise ValueError(f"{label}: hidden component has outcome exposure")
    component = str(role_record.get("leakage_component_id") or "")
    if row.get("physical_scene_component_id") not in (None, component):
        raise ValueError(f"{label}: physical component disagrees with ledger")

    rotation = int(row.get("rotation_degrees"))
    raw_turn = str(row.get("turn") or "").lower()
    if rotation not in ROTATIONS:
        raise ValueError(f"{label}: unsupported rotation")
    if rotation != 180 and raw_turn not in TURNS:
        raise ValueError(f"{label}: unsupported turn direction")
    if rotation == 180 and raw_turn not in {"left", "right", "any", "free180"}:
        raise ValueError(f"{label}: unsupported 180-degree source turn")
    answer = str(row.get("answer") or "")
    support_turn = "left" if rotation == 180 else raw_turn
    if answer not in supports[(rotation, support_turn)]:
        raise ValueError(f"{label}: answer is outside analytic support")

    evidence = row.get("geometry_evidence")
    if not isinstance(evidence, dict):
        raise ValueError(f"{label}: missing corrected geometry evidence")
    gates = config["geometry"]
    half_width = float(gates["source_bearing_half_width_degrees"])
    _require_close(
        f"{label} evidence half-width",
        evidence.get("source_envelope_half_width_degrees"),
        half_width,
        1e-12,
    )
    source_bearing = float(evidence.get("source_center_bearing_degrees"))
    if not abs(source_bearing) < half_width:
        raise ValueError(f"{label}: source center is outside the strict cone")
    _require_close(
        f"{label} source margin",
        evidence.get("source_envelope_margin_degrees"),
        half_width - abs(source_bearing),
        2e-5,
    )
    if evidence.get("derived_answer") != answer:
        raise ValueError(f"{label}: pose-derived answer disagrees with stored answer")
    recomputed_post = post_action_bearing_degrees(
        source_bearing, rotation, support_turn
    )
    _require_close(
        f"{label} post-action center bearing",
        evidence.get("post_center_bearing_degrees"),
        recomputed_post,
        2e-5,
    )
    if canonical_zone(recomputed_post) != answer:
        raise ValueError(f"{label}: center bearing/action does not derive stored answer")
    whole = evidence.get("whole_box")
    if not isinstance(whole, dict) or whole.get("all_corners_match_expected") is not True:
        raise ValueError(f"{label}: whole-box zone gate failed")
    margin = float(whole.get("minimum_corner_boundary_margin_degrees"))
    if margin < float(gates["minimum_whole_box_boundary_margin_degrees"]):
        raise ValueError(f"{label}: whole-box boundary margin gate failed")
    _require_close(
        f"{label} stored/replayed margin",
        row.get("extent_near_margin_deg"),
        margin,
        # Historical reservoir rows store this diagnostic to two decimals;
        # the corrected index carries the replayed full-precision value.
        0.011,
    )
    if int(evidence.get("same_type_visible_count", -1)) != int(
        gates["same_type_visible_count"]
    ):
        raise ValueError(f"{label}: same-type uniqueness gate failed")
    if int(row.get("visible_px") or 0) < int(gates["minimum_visible_pixels"]):
        raise ValueError(f"{label}: visible-pixel gate failed")
    if source.kind == "reservoir" and float(row.get("coverage") or 0.0) < float(
        gates["minimum_reservoir_silhouette_coverage"]
    ):
        raise ValueError(f"{label}: silhouette-coverage gate failed")
    if source.kind == "targeted":
        bbox = evidence.get("target_bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"{label}: targeted bbox evidence is missing")
        x0, y0, x1, y1 = [int(value) for value in bbox]
        if min(x0, y0, 640 - x1, 480 - y1) < int(
            gates["minimum_targeted_mask_edge_margin_pixels"]
        ):
            raise ValueError(f"{label}: targeted mask-edge gate failed")
    _require_close(
        f"{label} render vertical FOV",
        evidence.get("render_vertical_fov_degrees"),
        gates["render_vertical_fov_degrees"],
    )

    frames = row.get("frames")
    if not isinstance(frames, list) or len(frames) != 1:
        raise ValueError(f"{label}: exactly one source frame is required")
    source_frame = str(frames[0])
    asset_path = _resolve_source_asset(source, source_frame)
    if not asset_path.is_file() or asset_path.stat().st_size <= 8:
        raise ValueError(f"{label}: source frame is missing or empty")
    with asset_path.open("rb") as handle:
        if handle.read(8) != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"{label}: source frame is not a PNG")
    target_oid = str(row.get("target_oid") or "")
    target_type = str(row.get("target_type") or target_oid.split("|")[0])
    if not target_oid or not target_type:
        raise ValueError(f"{label}: target identity/category is missing")

    result = dict(row)
    result.update(
        {
            "_source_name": source.name,
            "_source_kind": source.kind,
            "_source_line": line_number,
            "_source_frame": source_frame,
            "_asset_root": str(source.asset_root.resolve()),
            "_asset_path": asset_path,
            "_role": role,
            "_component": component,
            "_outcome_exposed_component": bool(
                role_record.get("outcome_exposed_component")
            ),
            "_category": _canonical_category(target_type),
            "_display_category": _display_category(target_type),
            "_source_observation_key": sha256_json(
                [source.name, component, scene_id, target_oid, source_frame]
            ),
            "_source_action_key": sha256_json(
                [source.name, component, scene_id, target_oid, source_frame, rotation, raw_turn]
            ),
            "_frame_path_key": f"{source.asset_root.resolve()}::{source_frame}",
            "_object_key": f"{component}::{target_oid}",
            "_stable": sha256_json(
                [component, scene_id, target_oid, source_frame, rotation, raw_turn]
            ),
        }
    )
    return result


def load_candidates(
    *,
    source_specs: Sequence[SourceSpec],
    roles_payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require_complete_split_roles(roles_payload)
    roles = roles_payload.get("scenes")
    if not isinstance(roles, dict):
        raise ValueError("split-role ledger is absent or failing")
    half_width = float(config["geometry"]["source_bearing_half_width_degrees"])
    supports = action_supports(half_width)
    candidates_by_action: dict[str, dict[str, Any]] = {}
    conflicting_gold: list[dict[str, Any]] = []
    input_records: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for spec in source_specs:
        geometry_report, input_record = _validate_geometry_report(spec, config)
        input_records.append(input_record)
        with spec.index_path.resolve().open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                counts["input_rows"] += 1
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{spec.name} line {line_number}: row is not an object")
                role_record = roles.get(str(row.get("scene_id") or ""))
                if not isinstance(role_record, dict):
                    raise ValueError(
                        f"{spec.name} line {line_number}: scene absent from role ledger"
                    )
                if role_record.get("drop_exact_alias_nonrepresentative") is True:
                    # The physical-component ledger prospectively designates
                    # one representative.  Nonrepresentative aliases are
                    # expected upstream capacity, never candidate items.
                    counts["dropped_exact_alias_nonrepresentative_rows"] += 1
                    continue
                candidate = _validate_source_row(
                    row,
                    source=spec,
                    line_number=line_number,
                    roles=roles,
                    config=config,
                    supports=supports,
                    expected_upstream_source_hash=str(
                        geometry_report.get("input", {}).get("sha256") or ""
                    ),
                )
                key = candidate["_source_action_key"]
                prior = candidates_by_action.get(key)
                if prior is not None:
                    counts["duplicate_source_action_rows"] += 1
                    if prior["answer"] != candidate["answer"]:
                        conflicting_gold.append(
                            {
                                "source_action_key": key,
                                "first": [prior["_source_name"], prior["_source_line"], prior["answer"]],
                                "second": [candidate["_source_name"], candidate["_source_line"], candidate["answer"]],
                            }
                        )
                        continue
                    old_quality = (
                        float(prior["geometry_evidence"]["whole_box"]["minimum_corner_boundary_margin_degrees"]),
                        float(prior.get("coverage") or 0.0),
                        int(prior.get("visible_px") or 0),
                        prior["_stable"],
                    )
                    new_quality = (
                        float(candidate["geometry_evidence"]["whole_box"]["minimum_corner_boundary_margin_degrees"]),
                        float(candidate.get("coverage") or 0.0),
                        int(candidate.get("visible_px") or 0),
                        candidate["_stable"],
                    )
                    if new_quality > old_quality:
                        candidates_by_action[key] = candidate
                else:
                    candidates_by_action[key] = candidate
    if conflicting_gold:
        raise ValueError(f"source pool has {len(conflicting_gold)} conflicting-gold identities")
    candidates = sorted(candidates_by_action.values(), key=lambda row: row["_stable"])
    counts["eligible_unique_source_actions"] = len(candidates)
    counts.update(Counter(f"role_{row['_role']}" for row in candidates))
    return candidates, {
        "sources": input_records,
        "counts": dict(sorted(counts.items())),
        "conflicting_gold": conflicting_gold,
    }


def _assignments_for_role(
    candidates: Sequence[Mapping[str, Any]],
    role: str,
    quotas: Mapping[tuple[int, str, str], int],
) -> list[Assignment]:
    assignments: list[Assignment] = []
    for index, row in enumerate(candidates):
        if row["_role"] != role:
            continue
        rotation = int(row["rotation_degrees"])
        answer = str(row["answer"])
        possible_turns = TURNS if rotation == 180 else (str(row["turn"]).lower(),)
        for turn in possible_turns:
            if quotas.get((rotation, turn, answer), 0) > 0:
                assignments.append(Assignment(index, rotation, turn, answer))
    return assignments


def _quality_tuple(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        -float(row["geometry_evidence"]["whole_box"]["minimum_corner_boundary_margin_degrees"]),
        -float(row.get("coverage") or 0.0),
        -int(row.get("visible_px") or 0),
        str(row["_stable"]),
    )


def _capacity_diagnostics(
    assignments: Sequence[Assignment],
    candidates: Sequence[Mapping[str, Any]],
    quotas: Mapping[tuple[int, str, str], int],
    caps: Mapping[str, Any],
) -> dict[str, Any]:
    raw = Counter(assignment.stratum for assignment in assignments)
    by_category = Counter(str(candidates[a.candidate_index]["_category"]) for a in assignments)
    by_component = Counter(str(candidates[a.candidate_index]["_component"]) for a in assignments)
    by_object = Counter(str(candidates[a.candidate_index]["_object_key"]) for a in assignments)
    by_frame = Counter(str(candidates[a.candidate_index]["_frame_path_key"]) for a in assignments)
    return {
        "required_by_stratum": {str(key): value for key, value in sorted(quotas.items())},
        "raw_assignment_capacity_by_stratum": {
            str(key): raw[key] for key in sorted(quotas)
        },
        "raw_stratum_shortfalls": {
            str(key): {"required": value, "available": raw[key], "shortfall": value - raw[key]}
            for key, value in sorted(quotas.items())
            if raw[key] < value
        },
        "distinct_categories": len(by_category),
        "category_assignment_capacity": dict(by_category.most_common()),
        "distinct_physical_components": len(by_component),
        "distinct_target_objects": len(by_object),
        "distinct_source_frames": len(by_frame),
        "caps": dict(caps),
    }


def solve_role(
    *,
    candidates: Sequence[dict[str, Any]],
    role: str,
    quotas: Mapping[tuple[int, str, str], int],
    caps: Mapping[str, Any],
    extra_content_groups: Sequence[set[int]] = (),
    banned_candidate_indices: set[int] | None = None,
) -> tuple[list[Assignment], dict[str, Any]]:
    """Solve one role exactly with all quotas and caps as hard constraints."""

    assignments = _assignments_for_role(candidates, role, quotas)
    diagnostics = _capacity_diagnostics(assignments, candidates, quotas, caps)
    if diagnostics["raw_stratum_shortfalls"]:
        raise CandidateCapacityError(
            f"{role} has raw action/answer capacity shortfalls",
            {"role": role, "kind": "raw_stratum_shortfall", **diagnostics},
        )
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
        from scipy.sparse import coo_matrix
    except ImportError as exc:
        raise RuntimeError(
            "Task-B candidate selection requires scipy.optimize.milp (SciPy >=1.11)"
        ) from exc

    variable_count = len(assignments)
    costs = np.zeros(variable_count, dtype=float)
    by_stratum_variables: dict[tuple[int, str, str], list[int]] = defaultdict(list)
    by_candidate_variables: dict[int, list[int]] = defaultdict(list)
    for variable, assignment in enumerate(assignments):
        by_stratum_variables[assignment.stratum].append(variable)
        by_candidate_variables[assignment.candidate_index].append(variable)
    for stratum, variables in by_stratum_variables.items():
        ordered = sorted(
            variables,
            key=lambda variable: (
                _quality_tuple(candidates[assignments[variable].candidate_index]),
                sha256_json(
                    [
                        "milp-tie",
                        role,
                        stratum,
                        candidates[assignments[variable].candidate_index]["_stable"],
                    ]
                ),
            ),
        )
        denominator = max(1, len(ordered))
        for rank, variable in enumerate(ordered):
            costs[variable] = rank / denominator
            costs[variable] += int(
                sha256_json(["objective", role, variable])[:12], 16
            ) / (16**12) * 1e-7

    constraint_rows: list[list[int]] = []
    lowers: list[float] = []
    uppers: list[float] = []

    def add_constraint(variables: Iterable[int], low: float, high: float) -> None:
        values = sorted(set(variables))
        constraint_rows.append(values)
        lowers.append(low)
        uppers.append(high)

    for stratum, required in sorted(quotas.items()):
        add_constraint(by_stratum_variables[stratum], required, required)
    for variables in by_candidate_variables.values():
        if len(variables) > 1:
            add_constraint(variables, -np.inf, 1)

    group_specs = (
        ("_component", int(caps["maximum_per_physical_component"])),
        ("_frame_path_key", int(caps["maximum_per_source_frame"])),
        ("_object_key", int(caps["maximum_per_target_object"])),
        ("_category", int(caps["maximum_single_target_category_items"])),
    )
    for field, cap in group_specs:
        groups: dict[str, list[int]] = defaultdict(list)
        for variable, assignment in enumerate(assignments):
            groups[str(candidates[assignment.candidate_index][field])].append(variable)
        for variables in groups.values():
            add_constraint(variables, -np.inf, cap)
    for candidate_group in extra_content_groups:
        variables = [
            variable
            for candidate_index in candidate_group
            for variable in by_candidate_variables.get(candidate_index, [])
        ]
        if variables:
            add_constraint(variables, -np.inf, 1)

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    for constraint_index, variables in enumerate(constraint_rows):
        row_indices.extend([constraint_index] * len(variables))
        column_indices.extend(variables)
        values.extend([1.0] * len(variables))
    matrix = coo_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(constraint_rows), variable_count),
    ).tocsr()
    upper_bounds = np.ones(variable_count, dtype=float)
    for candidate_index in banned_candidate_indices or set():
        for variable in by_candidate_variables.get(candidate_index, []):
            upper_bounds[variable] = 0.0
    result = milp(
        c=costs,
        integrality=np.ones(variable_count, dtype=np.uint8),
        bounds=Bounds(np.zeros(variable_count), upper_bounds),
        constraints=LinearConstraint(matrix, np.asarray(lowers), np.asarray(uppers)),
        options={"presolve": True, "mip_rel_gap": 0.0, "time_limit": 600.0},
    )
    if not result.success or result.x is None:
        raise CandidateCapacityError(
            f"{role} joint hard-cap program is infeasible or unsolved: {result.message}",
            {
                "role": role,
                "kind": "joint_hard_cap_infeasibility",
                "solver_status": int(result.status),
                "solver_message": str(result.message),
                "variables": variable_count,
                "constraints": len(constraint_rows),
                **diagnostics,
            },
        )
    chosen = [
        assignments[index]
        for index, value in enumerate(result.x)
        if float(value) > 0.5
    ]
    chosen_counts = Counter(item.stratum for item in chosen)
    if chosen_counts != Counter(quotas):
        raise RuntimeError("MILP returned a selection that violates exact quotas")
    if len({item.candidate_index for item in chosen}) != len(chosen):
        raise RuntimeError("MILP returned a reused source observation")
    return chosen, {
        "solver": "scipy.optimize.milp/HiGHS",
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "objective": float(result.fun),
        "variables": variable_count,
        "constraints": len(constraint_rows),
        "selected": len(chosen),
        "capacity": diagnostics,
    }


def _asset_hash(row: dict[str, Any]) -> str:
    value = row.get("_rgb_sha256")
    if value is None:
        value = sha256_file(row["_asset_path"])
        row["_rgb_sha256"] = value
    return str(value)


def solve_with_content_deduplication(
    *,
    candidates: list[dict[str, Any]],
    role: str,
    quotas: Mapping[tuple[int, str, str], int],
    caps: Mapping[str, Any],
    globally_forbidden_hashes: set[str],
) -> tuple[list[Assignment], dict[str, Any]]:
    content_groups: list[set[int]] = []
    banned: set[int] = set()
    for iteration in range(1, 21):
        chosen, solver = solve_role(
            candidates=candidates,
            role=role,
            quotas=quotas,
            caps=caps,
            extra_content_groups=content_groups,
            banned_candidate_indices=banned,
        )
        hashes: dict[str, list[int]] = defaultdict(list)
        for assignment in chosen:
            hashes[_asset_hash(candidates[assignment.candidate_index])].append(
                assignment.candidate_index
            )
        new_information = False
        for digest, indices in hashes.items():
            unique_indices = set(indices)
            if digest in globally_forbidden_hashes:
                before = len(banned)
                banned.update(unique_indices)
                new_information |= len(banned) != before
            elif len(unique_indices) > 1:
                if unique_indices not in content_groups:
                    content_groups.append(unique_indices)
                    new_information = True
        if not new_information:
            solver.update(
                {
                    "content_deduplication_iterations": iteration,
                    "content_groups_added": len(content_groups),
                    "candidates_banned_for_cross_split_rgb_overlap": len(banned),
                }
            )
            return chosen, solver
    raise CandidateCapacityError(
        f"{role} exact-RGB deduplication did not converge",
        {
            "role": role,
            "kind": "content_deduplication_nonconvergence",
            "iterations": 20,
            "content_groups": len(content_groups),
            "banned_candidates": len(banned),
        },
    )


def _question(config: Mapping[str, Any], category: str, turn: str, rotation: int) -> str:
    return str(config["presentation"]["question_template"]).format(
        category=category, turn=turn, rotation=rotation
    )


def _assign_options(rows: list[dict[str, Any]], labels: Sequence[str], seed: str) -> None:
    groups: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["_split"],
                row["_rotation_degrees"],
                row["_turn_direction"],
                row["_answer"],
            )
        ].append(row)
    for key, group in sorted(groups.items(), key=lambda value: repr(value[0])):
        offset = int(sha256_json([seed, "answer-position", key]), 16) % len(labels)
        ordered = sorted(group, key=lambda row: sha256_json([seed, "option-row", row["id"]]))
        for index, row in enumerate(ordered):
            answer_position = (offset + index) % len(labels)
            distractors = [label for label in labels if label != row["_answer"]]
            distractors.sort(
                key=lambda label: sha256_json(
                    [seed, "distractor-order", row["id"], label]
                )
            )
            options = list(distractors)
            options.insert(answer_position, row["_answer"])
            row["options"] = options


def _jsonl_write(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    count = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            count += 1
    return count


def _artifact_record(root: Path, path: Path, *, rows: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if rows is not None:
        record["rows"] = rows
    return record


def _selection_audit(
    *,
    rows_by_role: Mapping[str, list[dict[str, Any]]],
    config: Mapping[str, Any],
    solvers: Mapping[str, Any],
) -> dict[str, Any]:
    split_reports: dict[str, Any] = {}
    component_sets: dict[str, set[str]] = {}
    all_ids: list[str] = []
    all_hashes: list[str] = []
    public_field_violations: list[dict[str, Any]] = []
    forbidden = set(config["separation"]["forbidden_public_fields"])
    half_width = float(config["geometry"]["source_bearing_half_width_degrees"])
    supports = action_supports(half_width)
    for role, rows in rows_by_role.items():
        component_counts = Counter(row["_source"]["_component"] for row in rows)
        frame_counts = Counter(row["_source"]["_rgb_sha256"] for row in rows)
        object_counts = Counter(row["_source"]["_object_key"] for row in rows)
        category_counts = Counter(row["_source"]["_category"] for row in rows)
        action_answer = Counter(
            (row["_rotation_degrees"], row["_turn_direction"], row["_answer"])
            for row in rows
        )
        action_diagnostics: dict[str, Any] = {}
        for rotation in ROTATIONS:
            for turn in TURNS:
                labels = Counter(
                    row["_answer"]
                    for row in rows
                    if row["_rotation_degrees"] == rotation
                    and row["_turn_direction"] == turn
                )
                action_diagnostics[f"{rotation}:{turn}"] = cell_construction_diagnostics(
                    labels, supports[(rotation, turn)]
                )
        option_groups: dict[str, Counter[int]] = defaultdict(Counter)
        for row in rows:
            position = row["options"].index(row["_answer"])
            key = (
                row["_rotation_degrees"],
                row["_turn_direction"],
                row["_answer"],
            )
            option_groups[str(key)][position] += 1
            public = {key: row[key] for key in PUBLIC_FIELDS}
            if set(public) != set(PUBLIC_FIELDS) or set(public) & forbidden:
                public_field_violations.append({"id": row["id"], "fields": sorted(public)})
        split_reports[role] = {
            "items": len(rows),
            "action_answer_counts": {
                str(key): value for key, value in sorted(action_answer.items())
            },
            "finite_action_cell_diagnostics": action_diagnostics,
            "category_counts": dict(category_counts.most_common()),
            "distinct_categories": len(category_counts),
            "maximum_category_count": max(category_counts.values(), default=0),
            "physical_components": len(component_counts),
            "maximum_per_physical_component": max(component_counts.values(), default=0),
            "source_frames": len(frame_counts),
            "maximum_per_exact_rgb": max(frame_counts.values(), default=0),
            "target_objects": len(object_counts),
            "maximum_per_target_object": max(object_counts.values(), default=0),
            "outcome_exposed_hidden_items": sum(
                row["_source"]["_outcome_exposed_component"]
                for row in rows
                if role == "hidden_test"
            ),
            "source_kind_counts": dict(
                Counter(row["_source"]["_source_kind"] for row in rows)
            ),
            "answer_position_counts_by_action_answer": {
                key: dict(sorted(value.items())) for key, value in sorted(option_groups.items())
            },
            "answer_position_max_min_gap": max(
                (
                    max(value.get(index, 0) for index in range(4))
                    - min(value.get(index, 0) for index in range(4))
                    for value in option_groups.values()
                ),
                default=0,
            ),
            "solver": solvers[role],
        }
        component_sets[role] = set(component_counts)
        all_ids.extend(row["id"] for row in rows)
        all_hashes.extend(row["_source"]["_rgb_sha256"] for row in rows)
    overlap = {}
    for first_index, first in enumerate(ROLES):
        for second in ROLES[first_index + 1 :]:
            overlap[f"{first}__{second}"] = len(component_sets[first] & component_sets[second])
    caps_pass = all(
        split_reports[role]["items"] == int(config["splits"][role]["total_items"])
        and split_reports[role]["maximum_per_physical_component"]
        <= int(config["splits"][role]["maximum_per_physical_component"])
        and split_reports[role]["maximum_per_exact_rgb"]
        <= int(config["splits"][role]["maximum_per_source_frame"])
        and split_reports[role]["maximum_per_target_object"]
        <= int(config["splits"][role]["maximum_per_target_object"])
        and split_reports[role]["maximum_category_count"]
        <= int(config["splits"][role]["maximum_single_target_category_items"])
        and split_reports[role]["outcome_exposed_hidden_items"] == 0
        and split_reports[role]["answer_position_max_min_gap"] <= 1
        and all(
            value["attains_target"]
            for value in split_reports[role]["finite_action_cell_diagnostics"].values()
        )
        for role in ROLES
    )
    return {
        "splits": split_reports,
        "cross_role_physical_component_overlap": overlap,
        "duplicate_public_ids": len(all_ids) - len(set(all_ids)),
        "duplicate_exact_rgb_content": len(all_hashes) - len(set(all_hashes)),
        "public_field_violations": public_field_violations,
        "hard_gates_pass": caps_pass
        and not any(overlap.values())
        and len(all_ids) == len(set(all_ids))
        and len(all_hashes) == len(set(all_hashes))
        and not public_field_violations,
    }


def build_candidate(
    *,
    config_path: Path,
    split_roles_path: Path,
    source_specs: Sequence[SourceSpec],
    output_root: Path,
    audit_report_path: Path,
) -> dict[str, Any]:
    """Build all three Task-B roles and materialize public/private artifacts."""

    config_path = config_path.resolve()
    split_roles_path = split_roles_path.resolve()
    output_root = output_root.resolve()
    audit_report_path = audit_report_path.resolve()
    if output_root.exists() or audit_report_path.exists():
        raise ValueError("refusing to overwrite candidate output or audit report")
    config = _load_json(config_path)
    validate_contract(config)
    roles_payload = _load_json(split_roles_path)
    candidates, source_audit = load_candidates(
        source_specs=source_specs, roles_payload=roles_payload, config=config
    )
    seed = str(config["seed"])
    half_width = float(config["geometry"]["source_bearing_half_width_degrees"])

    selected_by_role: dict[str, list[Assignment]] = {}
    solver_reports: dict[str, Any] = {}
    global_hashes: set[str] = set()
    # Hidden is frozen first; development roles can never displace a hidden
    # item and physical components are already role-exclusive.
    for role in ROLES:
        split = config["splits"][role]
        quotas = finite_label_quotas(
            role=role,
            items_per_action_cell=int(split["items_per_action_cell"]),
            items_by_rotation=split.get("items_per_action_cell_by_rotation"),
            seed=seed,
            half_width=half_width,
        )
        selected, solver = solve_with_content_deduplication(
            candidates=candidates,
            role=role,
            quotas=quotas,
            caps=split,
            globally_forbidden_hashes=global_hashes,
        )
        selected_by_role[role] = selected
        solver_reports[role] = solver
        global_hashes.update(
            _asset_hash(candidates[assignment.candidate_index]) for assignment in selected
        )

    labels = tuple(config["actions"]["answer_labels"])
    rows_by_role: dict[str, list[dict[str, Any]]] = {}
    all_ids: set[str] = set()
    for role in ROLES:
        rows: list[dict[str, Any]] = []
        for assignment in selected_by_role[role]:
            source = candidates[assignment.candidate_index]
            item_digest = sha256_json(
                [
                    config["candidate_id"],
                    config["construction_id"],
                    "Task-B",
                    role,
                    source["_component"],
                    source["scene_id"],
                    source["target_oid"],
                    source["_source_frame"],
                    assignment.rotation_degrees,
                    assignment.turn_direction,
                ]
            )
            item_id = str(config["presentation"]["opaque_id_prefix"]) + item_digest[:24]
            if item_id in all_ids:
                raise RuntimeError(f"opaque public ID collision: {item_id}")
            all_ids.add(item_id)
            public_asset = (
                f"{config['presentation']['public_asset_prefix']}/"
                f"{source['_rgb_sha256']}.png"
            )
            rows.append(
                {
                    "id": item_id,
                    "task_family": config["presentation"]["task_family"],
                    "format": config["presentation"]["format"],
                    "frames": [public_asset],
                    "question": _question(
                        config,
                        source["_display_category"],
                        assignment.turn_direction,
                        assignment.rotation_degrees,
                    ),
                    "options": list(labels),
                    "_answer": assignment.answer,
                    "_split": role,
                    "_rotation_degrees": assignment.rotation_degrees,
                    "_turn_direction": assignment.turn_direction,
                    "_source": source,
                }
            )
        _assign_options(rows, labels, seed)
        rows.sort(key=lambda row: row["id"])
        rows_by_role[role] = rows

    selection_audit = _selection_audit(
        rows_by_role=rows_by_role, config=config, solvers=solver_reports
    )
    if not selection_audit["hard_gates_pass"]:
        raise RuntimeError("post-selection Task-B hard-gate audit failed")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_root.parent, prefix=f".{output_root.name}.building."
    ) as temporary:
        staging = Path(temporary)
        artifact_records: list[dict[str, Any]] = []
        copied_hashes: set[str] = set()
        for role in ROLES:
            rows = rows_by_role[role]
            public_rows = [
                {field: row[field] for field in PUBLIC_FIELDS} for row in rows
            ]
            question_path = staging / "public" / "questions" / f"task_b_{role}.jsonl"
            row_count = _jsonl_write(question_path, public_rows)
            artifact_records.append(
                _artifact_record(staging, question_path, rows=row_count)
            )
            gold_rows = [
                {
                    "id": row["id"],
                    "answer": row["_answer"],
                    "answer_index": row["options"].index(row["_answer"]),
                }
                for row in rows
            ]
            if role == "hidden_test":
                gold_path = staging / "private" / "gold" / "task_b_hidden_test.jsonl"
            else:
                gold_path = staging / "development" / "answers" / f"task_b_{role}.jsonl"
            gold_count = _jsonl_write(gold_path, gold_rows)
            artifact_records.append(_artifact_record(staging, gold_path, rows=gold_count))

            provenance_rows = []
            for row in rows:
                source = row["_source"]
                provenance_rows.append(
                    {
                        "id": row["id"],
                        "split": role,
                        "answer": row["_answer"],
                        "answer_index": row["options"].index(row["_answer"]),
                        "rotation_degrees": row["_rotation_degrees"],
                        "turn_direction": row["_turn_direction"],
                        "source_name": source["_source_name"],
                        "source_kind": source["_source_kind"],
                        "source_line": source["_source_line"],
                        "source_frame": source["_source_frame"],
                        "source_rgb_sha256": source["_rgb_sha256"],
                        "scene_id": source["scene_id"],
                        "physical_scene_component_id": source["_component"],
                        "outcome_exposed_component": source[
                            "_outcome_exposed_component"
                        ],
                        "target_oid": source["target_oid"],
                        "target_type": source.get("target_type"),
                        "target_category": source["_category"],
                        "visible_pixels": source.get("visible_px"),
                        "silhouette_coverage": source.get("coverage"),
                        "geometry_evidence": source["geometry_evidence"],
                        "targeted": source["_source_kind"] == "targeted",
                    }
                )
            provenance_path = (
                staging / "private" / "provenance" / f"task_b_{role}.jsonl"
            )
            provenance_count = _jsonl_write(provenance_path, provenance_rows)
            artifact_records.append(
                _artifact_record(staging, provenance_path, rows=provenance_count)
            )

            for row in rows:
                source = row["_source"]
                digest = source["_rgb_sha256"]
                if digest in copied_hashes:
                    raise RuntimeError("exact RGB duplicate escaped the selection audit")
                copied_hashes.add(digest)
                destination = staging / row["frames"][0]
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source["_asset_path"], destination)
                if sha256_file(destination) != digest:
                    raise RuntimeError("materialized public asset hash mismatch")
                artifact_records.append(_artifact_record(staging, destination))

        artifact_records.sort(key=lambda record: record["path"])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "manifest": "egoshift_taskb_s05_static_candidate",
            "candidate_id": config["candidate_id"],
            "construction_id": config["construction_id"],
            "status": "static_candidate_only_not_release_ready",
            "task_family": "B",
            "config": {"sha256": sha256_file(config_path)},
            "split_roles": {"sha256": sha256_file(split_roles_path)},
            "counts": {
                "items": sum(len(rows) for rows in rows_by_role.values()),
                "items_by_split": {
                    role: len(rows_by_role[role]) for role in ROLES
                },
                "public_assets": len(copied_hashes),
            },
            "artifacts": artifact_records,
            "release_ready": False,
            "remaining_blockers": config["downstream_release_blockers"],
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifest_hash = sha256_file(manifest_path)
        staging.rename(output_root)

    audit = {
        "schema_version": SCHEMA_VERSION,
        "audit": "taskb_candidate_r2_static_construction",
        "candidate_id": config["candidate_id"],
        "construction_id": config["construction_id"],
        "status": "static_candidate_only_not_release_ready",
        "inputs": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "split_roles": {
                "path": str(split_roles_path),
                "sha256": sha256_file(split_roles_path),
            },
            **source_audit,
        },
        "contract": {
            "source_bearing_half_width_degrees": half_width,
            "source_boundary_policy": "strict",
            "items_per_action_cell_by_rotation": {
                role: config["splits"][role].get("items_per_action_cell_by_rotation")
                for role in ROLES
            },
            "items_per_action_cell": {
                role: int(config["splits"][role]["items_per_action_cell"])
                for role in ROLES
            },
            "hard_caps": config["splits"],
            "public_question_fields": list(PUBLIC_FIELDS),
        },
        "selection": selection_audit,
        "output": {
            "root": str(output_root),
            "manifest": {
                "path": str(output_root / "manifest.json"),
                "sha256": manifest_hash,
            },
        },
        "public_private_separation": {
            "questions": "public/questions/task_b_<split>.jsonl",
            "development_answers": "development/answers/task_b_<development-role>.jsonl",
            "hidden_gold": "private/gold/task_b_hidden_test.jsonl",
            "private_provenance": "private/provenance/task_b_<split>.jsonl",
            "public_allowlist_pass": not selection_audit["public_field_violations"],
        },
        "release_ready": False,
        "remaining_blockers": config["downstream_release_blockers"],
        "pass": True,
    }
    audit_report_path.parent.mkdir(parents=True, exist_ok=True)
    audit_report_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return audit
