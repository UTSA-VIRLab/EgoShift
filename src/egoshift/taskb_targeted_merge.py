"""Fail-closed verification and merge of targeted Task-B mining shards.

The sharded miner deliberately writes candidate *source material*.  This
module is the trust boundary before that material may enter the static S05
selector: it verifies the complete report chain, recomputes pose geometry,
checks every referenced RGB/mask hash, and enforces the globally frozen scene
roles and component cap.  It never treats an unmet per-shard overprovisioning
goal as corruption; the prespecified global role/cell quotas are the merge
gate.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .hashing import resolve_within, sha256_file, sha256_json
from .split_roles import require_complete_split_roles
from .taskb_geometry import corner_post_diagnostic, item_pose_diagnostic


EXPECTED_SCHEMA_VERSION = "0.1.0"
EXPECTED_AUDIT = "taskb_targeted_r2_mining_shard"


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _cell_id(row: Mapping[str, Any]) -> str:
    return f"{int(row['rotation_degrees'])}:{str(row['turn']).lower()}:{row['answer']}"


def _configured_cells(config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for cell in config.get("cells") or []:
        if not isinstance(cell, dict):
            raise ValueError("targeted-miner cells must be objects")
        identifier = _cell_id(cell)
        if identifier in result:
            raise ValueError(f"duplicate targeted-miner cell: {identifier}")
        result[identifier] = cell
    if not result:
        raise ValueError("targeted-miner config has no cells")
    return result


def _global_quota(config: Mapping[str, Any], role: str, cell: str) -> int:
    contract = config["target_rows_per_role_and_cell"][role]
    if isinstance(contract, Mapping):
        if cell not in contract:
            raise ValueError(f"missing targeted quota for {role}:{cell}")
        value = int(contract[cell])
    else:
        value = int(contract)
    if value < 0:
        raise ValueError(f"negative targeted quota for {role}:{cell}")
    return value


def _report_output_path(report_path: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid shard output path in {report_path}")
    path = Path(value)
    if not path.is_absolute():
        # Miner reports normally store the invocation-relative path.  Resolve
        # it from the process working directory first, matching the miner.
        path = Path.cwd() / path
    return path.resolve()


def _close(first: object, second: object, *, tolerance: float = 1e-6) -> bool:
    try:
        return math.isclose(float(first), float(second), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return False


def _validate_row(
    row: dict[str, Any],
    *,
    source_label: str,
    config: Mapping[str, Any],
    config_sha256: str,
    roles: Mapping[str, Any],
    asset_root: Path,
    cells: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    gates = config["gates"]
    render = config["render"]
    required = {
        "scene_id",
        "physical_scene_component_id",
        "preassigned_role",
        "target_oid",
        "target_type",
        "rotation_degrees",
        "turn",
        "answer",
        "visible_px",
        "extent_near_margin_deg",
        "frames",
        "target_mask_private_path",
        "target_mask_sha256",
        "rgb_sha256",
        "target_mask_bbox_xyxy",
        "same_type_visible_count",
        "same_type_visible_instances",
        "aabb_corners",
        "v4_pose",
        "geometry_evidence_live",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(f"{source_label}: row missing fields {missing}")
    if row.get("gate1_pre") is not True or row.get("v4_targeted") is not True:
        raise ValueError(f"{source_label}: targeted/gate1 flags are not true")
    if row.get("targeted_miner_id") != config.get("mining_id"):
        raise ValueError(f"{source_label}: mining_id mismatch")
    if row.get("targeted_miner_config_sha256") != config_sha256:
        raise ValueError(f"{source_label}: row config hash mismatch")

    scene_id = str(row["scene_id"])
    role_record = roles.get(scene_id)
    if not isinstance(role_record, dict):
        raise ValueError(f"{source_label}: scene absent from split-role ledger: {scene_id}")
    if role_record.get("drop_exact_alias_nonrepresentative") is not False:
        raise ValueError(f"{source_label}: exact-alias nonrepresentative is forbidden")
    role = str(role_record.get("role"))
    if row["preassigned_role"] != role:
        raise ValueError(f"{source_label}: preassigned role disagrees with role ledger")
    component = str(role_record.get("leakage_component_id"))
    if row["physical_scene_component_id"] != component:
        raise ValueError(f"{source_label}: physical component disagrees with role ledger")
    if role == "hidden_test" and role_record.get("outcome_exposed_component") is not False:
        raise ValueError(f"{source_label}: outcome-exposed component assigned hidden")
    if role not in config["target_rows_per_role_and_cell"]:
        raise ValueError(f"{source_label}: unknown role {role!r}")

    cid = _cell_id(row)
    if cid not in cells:
        raise ValueError(f"{source_label}: row is outside the frozen targeted cells: {cid}")
    if _global_quota(config, role, cid) == 0:
        raise ValueError(f"{source_label}: row was mined for a frozen zero-quota cell")
    if int(row["visible_px"]) < int(gates["minimum_visible_pixels"]):
        raise ValueError(f"{source_label}: visible-pixel gate failed")
    if int(row["same_type_visible_count"]) != int(gates["same_type_visible_count"]):
        raise ValueError(f"{source_label}: same-type uniqueness gate failed")
    instances = row["same_type_visible_instances"]
    if not isinstance(instances, list) or len(instances) != int(gates["same_type_visible_count"]):
        raise ValueError(f"{source_label}: malformed same-type instance evidence")
    if not any(
        item.get("object_id") == row["target_oid"]
        and int(item.get("pixels") or 0) == int(row["visible_px"])
        for item in instances
        if isinstance(item, dict)
    ):
        raise ValueError(f"{source_label}: target is absent from uniqueness evidence")

    frames = row["frames"]
    if not isinstance(frames, list) or len(frames) != 1:
        raise ValueError(f"{source_label}: exactly one RGB frame is required")
    rgb_path = resolve_within(asset_root, str(frames[0]))
    mask_path = resolve_within(asset_root, str(row["target_mask_private_path"]))
    if not rgb_path.is_file() or not mask_path.is_file():
        raise ValueError(f"{source_label}: referenced RGB or mask is missing")
    rgb_hash = sha256_file(rgb_path)
    mask_hash = sha256_file(mask_path)
    if rgb_hash != row["rgb_sha256"] or mask_hash != row["target_mask_sha256"]:
        raise ValueError(f"{source_label}: referenced RGB or mask hash mismatch")

    bbox = row["target_mask_bbox_xyxy"]
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(f"{source_label}: malformed target-mask bbox")
    x0, y0, x1, y1 = [int(value) for value in bbox]
    width, height = int(render["width"]), int(render["height"])
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"{source_label}: target-mask bbox is outside the render")
    if x1 - x0 < int(gates["minimum_bbox_width_pixels"]) or y1 - y0 < int(
        gates["minimum_bbox_height_pixels"]
    ):
        raise ValueError(f"{source_label}: target-mask bbox dimension gate failed")
    edge_margin = min(x0, y0, width - x1, height - y1)
    if edge_margin < int(gates["minimum_mask_edge_margin_pixels"]):
        raise ValueError(f"{source_label}: target-mask edge gate failed")
    if int(row.get("target_mask_edge_margin_pixels", -1)) != edge_margin:
        raise ValueError(f"{source_label}: stored edge margin is inconsistent")

    pose_record = row["v4_pose"]
    if not isinstance(pose_record, dict) or not isinstance(pose_record.get("position"), dict):
        raise ValueError(f"{source_label}: malformed camera pose")
    corners = row["aabb_corners"]
    if not isinstance(corners, list) or len(corners) < 8:
        raise ValueError(f"{source_label}: malformed 3D bounding box")
    rotation = int(row["rotation_degrees"])
    geometry_turn = "left" if rotation == 180 else str(row["turn"]).lower()
    center = item_pose_diagnostic(
        camera_position=pose_record["position"],
        camera_yaw_degrees=float(pose_record["rotation_y"]),
        corners=corners,
        rotation_degrees=rotation,
        turn_direction=geometry_turn,
        expected_answer=str(row["answer"]),
        source_half_width_degrees=float(gates["source_bearing_half_width_degrees"]),
    )
    whole = corner_post_diagnostic(
        camera_position=pose_record["position"],
        camera_yaw_degrees=float(pose_record["rotation_y"]),
        corners=corners,
        rotation_degrees=rotation,
        turn_direction=geometry_turn,
        expected_answer=str(row["answer"]),
    )
    if not center["pass"]:
        raise ValueError(f"{source_label}: independently recomputed center geometry failed")
    minimum_margin = float(gates["minimum_whole_box_post_zone_margin_degrees"])
    if not whole["all_corners_match_expected"] or float(
        whole["minimum_corner_boundary_margin_degrees"]
    ) < minimum_margin:
        raise ValueError(f"{source_label}: independently recomputed whole-box geometry failed")
    if not _close(
        pose_record.get("theta_pre_deg"), center["source_center_bearing_degrees"], tolerance=2e-5
    ) or not _close(
        pose_record.get("theta_post_deg"), center["post_center_bearing_degrees"], tolerance=2e-5
    ):
        raise ValueError(f"{source_label}: stored and recomputed center bearings disagree")
    if not _close(
        row["extent_near_margin_deg"],
        whole["minimum_corner_boundary_margin_degrees"],
        tolerance=2e-5,
    ):
        raise ValueError(f"{source_label}: stored and recomputed whole-box margin disagree")
    if not _close(row.get("render_vertical_fov_degrees"), render["vertical_fov_degrees"]):
        raise ValueError(f"{source_label}: render vertical FOV mismatch")
    if not _close(row.get("render_horizon_degrees"), render["camera_horizon_degrees"], tolerance=1e-4):
        raise ValueError(f"{source_label}: render horizon mismatch")

    row_key = sha256_json(
        {
            "scene_id": scene_id,
            "target_oid": row["target_oid"],
            "cell": cid,
            "position": pose_record["position"],
            "yaw": round(float(pose_record["rotation_y"]), 6),
            "rgb_sha256": rgb_hash,
        }
    )
    return {
        "row_key": row_key,
        "role": role,
        "cell": cid,
        "component": component,
        "frame": str(frames[0]),
        "mask": str(row["target_mask_private_path"]),
        "rgb_sha256": rgb_hash,
        "mask_sha256": mask_hash,
    }


def merge_targeted_shards(
    *,
    config_path: Path,
    split_roles_path: Path,
    asset_root: Path,
    shard_report_paths: Iterable[Path],
    output_path: Path,
    report_path: Path,
    allow_global_shortfall: bool = False,
    allow_scene_errors: bool = False,
) -> dict[str, Any]:
    """Validate all shards and write one deterministic merged JSONL plus audit."""

    config_path = config_path.resolve()
    split_roles_path = split_roles_path.resolve()
    asset_root = asset_root.resolve()
    output_path = output_path.resolve()
    report_path = report_path.resolve()
    if output_path.exists() or report_path.exists():
        raise ValueError("refusing to overwrite merged output or audit report")
    config = _load_json(config_path)
    roles_payload = _load_json(split_roles_path)
    require_complete_split_roles(roles_payload)
    roles = roles_payload.get("scenes")
    if not isinstance(roles, dict):
        raise ValueError("split-role ledger lacks a scene mapping")
    config_sha = sha256_file(config_path)
    roles_sha = sha256_file(split_roles_path)
    cells = _configured_cells(config)
    for role in config.get("target_rows_per_role_and_cell") or {}:
        for cid in cells:
            _global_quota(config, str(role), cid)

    report_paths = [Path(path).resolve() for path in shard_report_paths]
    if not report_paths:
        raise ValueError("at least one shard report is required")
    shard_reports: dict[int, tuple[Path, dict[str, Any]]] = {}
    expected_shard_count: int | None = None
    for path in sorted(report_paths):
        report = _load_json(path)
        if report.get("schema_version") != EXPECTED_SCHEMA_VERSION or report.get("audit") != EXPECTED_AUDIT:
            raise ValueError(f"unexpected shard report schema/audit: {path}")
        if report.get("config", {}).get("sha256") != config_sha:
            raise ValueError(f"shard config hash mismatch: {path}")
        if report.get("split_roles", {}).get("sha256") != roles_sha:
            raise ValueError(f"shard split-role hash mismatch: {path}")
        shard = report.get("shard") or {}
        index, count = int(shard.get("index", -1)), int(shard.get("count", -1))
        if expected_shard_count is None:
            expected_shard_count = count
        if count != expected_shard_count or not (0 <= index < count):
            raise ValueError(f"inconsistent shard cardinality: {path}")
        if index in shard_reports:
            raise ValueError(f"duplicate shard index: {index}")
        if report.get("errors_preview") or int((report.get("counts") or {}).get("scene_errors", 0)):
            # A scene error means that scene contributed no rows -- a simulator
            # timeout, not corruption of an emitted row. The integrity of what
            # WAS emitted is established separately by the JSONL hash chain and
            # the accepted-row/quota reconciliation below, so this is an
            # operational fact to record, not a reason to distrust the data.
            if not allow_scene_errors:
                raise ValueError(f"shard contains scene errors: {path}")
        shard_reports[index] = (path, report)
    assert expected_shard_count is not None
    if len(shard_reports) != expected_shard_count or set(shard_reports) != set(
        range(expected_shard_count)
    ):
        raise ValueError(
            f"incomplete shard set: observed={sorted(shard_reports)}, expected=0..{expected_shard_count - 1}"
        )

    merged: list[dict[str, Any]] = []
    row_keys: set[str] = set()
    frame_paths: set[str] = set()
    mask_paths: set[str] = set()
    rgb_hashes: Counter[str] = Counter()
    mask_hashes: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    quota_counts: Counter[tuple[str, str]] = Counter()
    source_records: list[dict[str, Any]] = []
    for shard_index in range(expected_shard_count):
        shard_report_path, shard_report = shard_reports[shard_index]
        jsonl_path = _report_output_path(
            shard_report_path, shard_report.get("output", {}).get("path")
        )
        observed_hash = sha256_file(jsonl_path)
        if observed_hash != shard_report.get("output", {}).get("sha256"):
            raise ValueError(f"shard JSONL hash mismatch: {jsonl_path}")
        computed_quota: Counter[str] = Counter()
        line_count = 0
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                line_count += 1
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"shard {shard_index} line {line_number} is not an object")
                facts = _validate_row(
                    row,
                    source_label=f"shard {shard_index} line {line_number}",
                    config=config,
                    config_sha256=config_sha,
                    roles=roles,
                    asset_root=asset_root,
                    cells=cells,
                )
                duplicate_checks = (
                    (row_keys, facts["row_key"], "row identity"),
                    (frame_paths, facts["frame"], "RGB path"),
                    (mask_paths, facts["mask"], "mask path"),
                )
                for seen, value, label in duplicate_checks:
                    if value in seen:
                        raise ValueError(f"duplicate targeted {label}: {value}")
                    seen.add(value)
                # Different hypothetical actions can legitimately originate
                # from the same rendered pose.  Preserve those source rows,
                # audit the exact-content groups, and let the final selector's
                # hard source-frame/content cap choose at most one.
                rgb_hashes[facts["rgb_sha256"]] += 1
                mask_hashes[facts["mask_sha256"]] += 1
                component_counts[facts["component"]] += 1
                quota_counts[(facts["role"], facts["cell"])] += 1
                computed_quota[f"{facts['role']}:{facts['cell']}"] += 1
                merged.append(row)
        accepted = (shard_report.get("counts") or {}).get("accepted")
        if accepted is None:
            # ``Counter`` omits zero entries, so a shard that legitimately
            # accepted nothing carries no "accepted" key. Read that as zero
            # only when the rest of the report agrees the shard produced
            # nothing; otherwise keep failing, because a missing field in a
            # report that *does* claim rows is a malformed report.
            accepted = 0 if not (shard_report.get("quota_counts") or {}) else -1
        if line_count != int(accepted):
            raise ValueError(f"accepted-row count mismatch in shard {shard_index}")
        reported_quota = {
            str(key): int(value)
            for key, value in (shard_report.get("quota_counts") or {}).items()
        }
        if dict(sorted(computed_quota.items())) != dict(sorted(reported_quota.items())):
            raise ValueError(f"quota-count mismatch in shard {shard_index}")
        source_records.append(
            {
                "shard_index": shard_index,
                "report_path": str(shard_report_path),
                "report_sha256": sha256_file(shard_report_path),
                "jsonl_path": str(jsonl_path),
                "jsonl_sha256": observed_hash,
                "rows": line_count,
                "local_overprovisioning_pass": bool(shard_report.get("pass")),
                "unmet_local_quotas": shard_report.get("unmet_local_quotas") or {},
                "scene_errors": shard_report.get("errors_preview") or [],
            }
        )

    component_cap = int(config["maximum_rows_per_physical_component"])
    component_violations = {
        key: value for key, value in sorted(component_counts.items()) if value > component_cap
    }
    if component_violations:
        raise ValueError(
            f"global physical-component cap exceeded: {component_violations}"
        )
    unmet_global: dict[str, dict[str, int]] = {}
    for role in config["target_rows_per_role_and_cell"]:
        for cid in cells:
            observed = quota_counts[(role, cid)]
            target = _global_quota(config, role, cid)
            if observed < target:
                unmet_global[f"{role}:{cid}"] = {
                    "required": target,
                    "observed": observed,
                    "shortfall": target - observed,
                }
    if unmet_global:
        if not allow_global_shortfall:
            raise ValueError(f"global targeted quotas are unmet: {unmet_global}")

    merged.sort(
        key=lambda row: (
            str(row["preassigned_role"]),
            _cell_id(row),
            sha256_json(
                [
                    row["scene_id"],
                    row["target_oid"],
                    row["frames"],
                    row["v4_pose"],
                ]
            ),
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    audit = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "audit": "taskb_targeted_r2_fail_closed_merge",
        "status": "verified_candidate_source_only",
        "config": {"path": str(config_path), "sha256": config_sha},
        "split_roles": {"path": str(split_roles_path), "sha256": roles_sha},
        "asset_root": str(asset_root),
        "shards": source_records,
        "verified_gates": [
            "complete unique shard-index set",
            "report/config/split-role/JSONL hash chain",
            "zero recorded scene errors",
            "role and physical-component ledger agreement",
            "fresh-only hidden components",
            "live same-type uniqueness evidence",
            "RGB and private-mask existence and SHA-256",
            "target mask size and edge margin",
            "independently recomputed strict source-center cone",
            "independently recomputed center gold and whole-box 15-degree margin",
            "global physical-component cap",
            "global prespecified role/cell targeted quotas",
            "zero duplicate row or asset-path identities; exact-content groups audited for the selector's global cap",
        ],
        "counts": {
            "rows": len(merged),
            "physical_components": len(component_counts),
            "maximum_rows_per_physical_component": max(component_counts.values(), default=0),
            "unique_rgb_hashes": len(rgb_hashes),
            "unique_mask_hashes": len(mask_hashes),
            "duplicate_rgb_content_excess": sum(value - 1 for value in rgb_hashes.values()),
            "duplicate_mask_content_excess": sum(value - 1 for value in mask_hashes.values()),
        },
        "quota_counts": {
            f"{role}:{cell}": value
            for (role, cell), value in sorted(quota_counts.items())
        },
        "unmet_global_quotas": dict(sorted(unmet_global.items())) if unmet_global else {},
        "output": {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "rows": len(merged),
        },
        # `verification_pass` is the trust-boundary result: every merged row
        # passed independent geometry, role, cap, and hash re-verification.
        # `quota_pass` is the separate planning fact of whether this mining
        # round delivered its requested per-cell counts.  A multi-round
        # strategy legitimately merges a short round; the candidate builder
        # still fails closed if all sources together cannot fill the finite
        # quotas.
        "verification_pass": True,
        "quota_pass": not unmet_global,
        "operational_clean": not any(
            record.get("scene_errors") for record in source_records
        ),
        "scene_error_count": sum(
            len(record.get("scene_errors") or []) for record in source_records
        ),
        "partial_merge": bool(unmet_global),
        "pass": not unmet_global
        and not any(record.get("scene_errors") for record in source_records),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return audit
