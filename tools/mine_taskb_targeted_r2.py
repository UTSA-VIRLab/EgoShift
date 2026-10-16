#!/usr/bin/env python3
"""Mine clean Task-B rare-cell poses under the candidate-r2 contract.

Unlike the historical targeted miner, this version matches the reservoir
render protocol (vertical FOV 120, horizon 30), enforces live same-type
uniqueness, rejects masks near an image edge, verifies the explicit source
bearing cone, and checks every 3D-box corner after the stated turn.

The tool is shard-safe: every shard writes a distinct JSONL and asset root.
Rows remain candidate source material and are not a release split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from egoshift.split_roles import require_complete_split_roles
from egoshift.taskb_geometry import corner_post_diagnostic, item_pose_diagnostic


def stable_digest(*values: object) -> str:
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aabb_corners(obj: dict[str, Any]) -> list[Any] | None:
    box = obj.get("axisAlignedBoundingBox") or {}
    corners = box.get("cornerPoints")
    if isinstance(corners, list) and len(corners) >= 8:
        return corners[:8]
    center, size = box.get("center"), box.get("size")
    if not isinstance(center, dict) or not isinstance(size, dict):
        return None
    return [
        {
            "x": center["x"] + sx * size["x"] / 2,
            "y": center["y"] + sy * size["y"] / 2,
            "z": center["z"] + sz * size["z"] / 2,
        }
        for sx in (-1, 1)
        for sy in (-1, 1)
        for sz in (-1, 1)
    ]


def mask_bbox(mask: Any) -> tuple[int, int, int, int] | None:
    import numpy as np

    if mask is None or not bool(mask.any()):
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def world_heading(camera: dict[str, float], target: dict[str, float]) -> float:
    return math.degrees(
        math.atan2(target["x"] - camera["x"], target["z"] - camera["z"])
    )


def horizontal_radius(obj: dict[str, Any]) -> float | None:
    """Largest horizontal offset from the target centre to an AABB corner."""

    corners = aabb_corners(obj)
    if not corners:
        return None
    centre = target_center(obj)
    best = 0.0
    for corner in corners:
        if isinstance(corner, dict):
            dx = float(corner["x"]) - float(centre["x"])
            dz = float(corner["z"]) - float(centre["z"])
        else:
            dx = float(corner[0]) - float(centre["x"])
            dz = float(corner[2]) - float(centre["z"])
        best = max(best, math.hypot(dx, dz))
    return best or None


def target_center(obj: dict[str, Any]) -> dict[str, float]:
    box = obj.get("axisAlignedBoundingBox") or {}
    if isinstance(box.get("center"), dict):
        return box["center"]
    return obj["position"]


def role_needs(
    role: str,
    cells: list[dict[str, Any]],
    counts: Counter[tuple[str, str]],
    per_role_target: dict[str, Any],
    shard_count: int,
) -> list[dict[str, Any]]:
    """Return cells whose frozen shard-local capacity target is not yet met.

    The preferred contract supplies an explicit global target for every role and
    rare cell.  Stable scene sharding converts it to ``ceil(global/N)`` without
    the old unconditional ``+2`` overshoot.  The merger still enforces global
    sufficiency and may request a narrowly scoped deficit rerun.  A scalar value
    remains accepted only for backwards-compatible smoke tests.
    """

    role_contract = per_role_target[role]
    needed: list[dict[str, Any]] = []
    for cell in cells:
        cid = cell_id(cell)
        if isinstance(role_contract, dict):
            if cid not in role_contract:
                raise ValueError(f"missing targeted quota for {role}:{cid}")
            global_goal = int(role_contract[cid])
        else:
            global_goal = int(role_contract)
        local_goal = math.ceil(global_goal / shard_count)
        if counts[(role, cid)] < local_goal:
            needed.append(cell)
    return needed


def cell_id(cell: dict[str, Any]) -> str:
    return f"{cell['rotation_degrees']}:{cell['turn']}:{cell['answer']}"


def propose(
    objects: list[dict[str, Any]],
    reachable: list[dict[str, float]],
    cells: list[dict[str, Any]],
    *,
    seed: str,
    scene_id: str,
    max_positions: int,
    distance_min: float,
    distance_max: float,
    degrees_per_pixel: float | None = None,
) -> list[tuple[dict[str, Any], dict[str, Any], dict[str, float], float]]:
    proposals: list[tuple[dict[str, Any], dict[str, Any], dict[str, float], float]] = []
    for obj in objects:
        center = target_center(obj)
        positions = []
        for position in reachable:
            distance = math.hypot(position["x"] - center["x"], position["z"] - center["z"])
            if distance_min <= distance <= distance_max:
                positions.append((distance, position))
        radius = horizontal_radius(obj)
        for cell in cells:
            desired = float(cell["desired_source_bearing_degrees"])
            # Search-aiming only; never a gate.  A cell whose feasible centre
            # band sits at the cone edge admits only targets whose horizontal
            # angular extent is small enough that the frozen whole-box margin
            # and the frame edge can both be satisfied.  Angular width scales
            # as 1/distance, so the distance that lands a given object in that
            # width window is computable rather than something to sample for.
            target_width_px = cell.get("target_bbox_width_pixels")
            preferred = float(cell.get("preferred_distance_meters", 2.5))
            if target_width_px and degrees_per_pixel and radius:
                half_angle = math.radians(
                    float(target_width_px) * float(degrees_per_pixel) / 2.0
                )
                if half_angle > 0:
                    preferred = radius / math.tan(half_angle)
            jitters = tuple(
                float(value)
                for value in cell.get("bearing_jitters_degrees", (0.0, -2.5, 2.5))
            ) or (0.0,)
            ordered = sorted(
                positions,
                key=lambda item: (
                    abs(item[0] - preferred),
                    stable_digest(seed, scene_id, obj["objectId"], item[1]),
                ),
            )
            for position_index, (_distance, position) in enumerate(ordered[:max_positions]):
                theta = desired + jitters[position_index % len(jitters)]
                yaw = (world_heading(position, center) - theta) % 360.0
                proposals.append((cell, obj, position, yaw))
    proposals.sort(
        key=lambda row: stable_digest(
            seed,
            scene_id,
            cell_id(row[0]),
            row[1]["objectId"],
            row[2],
            round(row[3], 6),
        )
    )
    return proposals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split-roles", required=True, type=Path)
    parser.add_argument("--scenes-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--max-scenes", type=int, default=0)
    args = parser.parse_args()
    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError("shard-index must be in [0,num-shards)")
    if args.output_jsonl.exists() or args.report.exists():
        raise ValueError("refusing to overwrite an existing shard output")

    config = json.loads(args.config.read_text(encoding="utf-8"))
    split_roles = json.loads(args.split_roles.read_text(encoding="utf-8"))
    require_complete_split_roles(split_roles)
    render = config["render"]
    horizontal_fov = 2.0 * math.degrees(
        math.atan(
            math.tan(math.radians(float(render["vertical_fov_degrees"]) / 2.0))
            * float(render["width"])
            / float(render["height"])
        )
    )
    degrees_per_pixel = horizontal_fov / float(render["width"])
    gates = config["gates"]
    cells = config["cells"]
    per_role_target = config["target_rows_per_role_and_cell"]
    seed = str(config["seed"])
    rng = random.Random(int(stable_digest(seed, args.shard_index), 16))

    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering
    import numpy as np
    from PIL import Image

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    # Stream accepted rows immediately.  A completed shard is certified only by
    # its report; an interrupted JSONL is a recoverable diagnostic, never an
    # implicitly passing input.
    args.output_jsonl.touch(exist_ok=False)
    scene_records = split_roles["scenes"]
    scene_ids = [
        scene_id
        for scene_id, record in sorted(scene_records.items())
        if not record["drop_exact_alias_nonrepresentative"]
        and (args.scenes_root / scene_id / "house.json").is_file()
        and int(stable_digest(seed, scene_id), 16) % args.num_shards == args.shard_index
    ]
    if args.max_scenes:
        scene_ids = scene_ids[: args.max_scenes]

    accepted: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    quota_counts: Counter[tuple[str, str]] = Counter()
    component_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    errors: list[dict[str, str]] = []
    for scene_index, scene_id in enumerate(scene_ids, start=1):
        role_record = scene_records[scene_id]
        role = role_record["role"]
        needed = role_needs(role, cells, quota_counts, per_role_target, args.num_shards)
        if not needed:
            if all(not role_needs(r, cells, quota_counts, per_role_target, args.num_shards) for r in per_role_target):
                break
            continue
        component_id = role_record["leakage_component_id"]
        if component_counts[component_id] >= int(config["maximum_rows_per_physical_component"]):
            continue
        controller = None
        try:
            house = json.loads(
                (args.scenes_root / scene_id / "house.json").read_text(encoding="utf-8")
            )
            controller = Controller(
                scene=house,
                width=int(render["width"]),
                height=int(render["height"]),
                fieldOfView=float(render["vertical_fov_degrees"]),
                renderDepthImage=False,
                renderInstanceSegmentation=True,
                platform=CloudRendering,
                gpu_device=args.gpu_device,
            )
            event = controller.step(action="GetReachablePositions")
            reachable = event.metadata.get("actionReturn") or []
            excluded_types = {value.casefold() for value in gates["excluded_target_types"]}
            objects = [
                obj
                for obj in event.metadata.get("objects") or []
                if obj.get("objectId")
                and aabb_corners(obj)
                and str(obj.get("objectType", "")).casefold() not in excluded_types
            ]
            objects.sort(
                key=lambda obj: (
                    type_counts[str(obj.get("objectType"))],
                    stable_digest(seed, scene_id, obj["objectId"]),
                )
            )
            proposals = propose(
                objects[: int(config.get("maximum_objects_per_scene", 40))],
                reachable,
                needed,
                seed=seed,
                scene_id=scene_id,
                max_positions=int(config["maximum_positions_per_object_cell"]),
                distance_min=float(config["distance_range_meters"][0]),
                distance_max=float(config["distance_range_meters"][1]),
                degrees_per_pixel=degrees_per_pixel,
            )
            rng.shuffle(proposals)
            proposals = proposals[: int(config["maximum_render_attempts_per_scene"])]
            for cell, obj, position, yaw in proposals:
                cid = cell_id(cell)
                if cell not in role_needs(role, cells, quota_counts, per_role_target, args.num_shards):
                    continue
                if component_counts[component_id] >= int(config["maximum_rows_per_physical_component"]):
                    break
                counts["render_attempts"] += 1
                event = controller.step(
                    action="TeleportFull",
                    position=position,
                    rotation={"x": 0.0, "y": yaw, "z": 0.0},
                    horizon=float(render["camera_horizon_degrees"]),
                    standing=True,
                    forceAction=True,
                )
                if not event.metadata.get("lastActionSuccess", False):
                    counts["reject_teleport"] += 1
                    continue
                oid = obj["objectId"]
                mask = (event.instance_masks or {}).get(oid)
                visible_pixels = int(np.count_nonzero(mask)) if mask is not None else 0
                if visible_pixels < int(gates["minimum_visible_pixels"]):
                    counts["reject_visible_pixels"] += 1
                    continue
                bbox = mask_bbox(mask)
                if bbox is None:
                    counts["reject_missing_bbox"] += 1
                    continue
                x0, y0, x1, y1 = bbox
                if x1 - x0 < int(gates["minimum_bbox_width_pixels"]) or y1 - y0 < int(
                    gates["minimum_bbox_height_pixels"]
                ):
                    counts["reject_bbox_dimensions"] += 1
                    continue
                edge_margin = min(
                    x0,
                    y0,
                    int(render["width"]) - x1,
                    int(render["height"]) - y1,
                )
                if edge_margin < int(gates["minimum_mask_edge_margin_pixels"]):
                    counts["reject_edge_margin"] += 1
                    continue
                visible_same_type: list[dict[str, Any]] = []
                target_type = str(obj["objectType"]).casefold()
                for other_id, other_mask in (event.instance_masks or {}).items():
                    if other_id.split("|")[0].casefold() != target_type:
                        continue
                    pixels = int(np.count_nonzero(other_mask))
                    if pixels >= int(gates["same_type_visible_pixel_threshold"]):
                        visible_same_type.append({"object_id": other_id, "pixels": pixels})
                if len(visible_same_type) != int(gates["same_type_visible_count"]):
                    counts["reject_same_type_uniqueness"] += 1
                    continue
                live_obj = next(
                    (item for item in event.metadata.get("objects") or [] if item.get("objectId") == oid),
                    None,
                )
                corners = aabb_corners(live_obj or obj)
                if not corners:
                    counts["reject_missing_live_box"] += 1
                    continue
                agent = event.metadata["agent"]
                geometry_turn = "left" if int(cell["rotation_degrees"]) == 180 else cell["turn"]
                pose = item_pose_diagnostic(
                    camera_position=agent["position"],
                    camera_yaw_degrees=agent["rotation"]["y"],
                    corners=corners,
                    rotation_degrees=cell["rotation_degrees"],
                    turn_direction=geometry_turn,
                    expected_answer=cell["answer"],
                    source_half_width_degrees=float(gates["source_bearing_half_width_degrees"]),
                )
                if not pose["pass"]:
                    counts["reject_center_geometry"] += 1
                    continue
                extent = corner_post_diagnostic(
                    camera_position=agent["position"],
                    camera_yaw_degrees=agent["rotation"]["y"],
                    corners=corners,
                    rotation_degrees=cell["rotation_degrees"],
                    turn_direction=geometry_turn,
                    expected_answer=cell["answer"],
                )
                if not extent["all_corners_match_expected"] or float(
                    extent["minimum_corner_boundary_margin_degrees"]
                ) < float(gates["minimum_whole_box_post_zone_margin_degrees"]):
                    counts["reject_whole_box_geometry"] += 1
                    continue

                pose_key = stable_digest(
                    seed,
                    scene_id,
                    oid,
                    cid,
                    {"x": position["x"], "z": position["z"]},
                    round(float(agent["rotation"]["y"]), 6),
                )
                frame_rel = f"{scene_id}/targeted_r2/{pose_key[:24]}.png"
                mask_rel = f"{scene_id}/targeted_r2_private_masks/{pose_key[:24]}.png"
                frame_path = args.output_root / frame_rel
                mask_path = args.output_root / mask_rel
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(event.frame).save(frame_path)
                Image.fromarray((mask.astype("uint8") * 255)).save(mask_path)
                row = {
                    "scene_id": scene_id,
                    "physical_scene_component_id": component_id,
                    "preassigned_role": role,
                    "target_oid": oid,
                    "target_type": obj["objectType"],
                    "rotation_degrees": int(cell["rotation_degrees"]),
                    "turn": cell["turn"],
                    "answer": cell["answer"],
                    "visible_px": visible_pixels,
                    "extent_near_margin_deg": round(
                        float(extent["minimum_corner_boundary_margin_degrees"]), 8
                    ),
                    "gate1_pre": True,
                    "frames": [frame_rel],
                    "target_mask_private_path": mask_rel,
                    "target_mask_sha256": sha256_file(mask_path),
                    "rgb_sha256": sha256_file(frame_path),
                    "target_mask_bbox_xyxy": [x0, y0, x1, y1],
                    "target_mask_edge_margin_pixels": edge_margin,
                    "same_type_visible_count": len(visible_same_type),
                    "same_type_visible_instances": visible_same_type,
                    "aabb_corners": corners,
                    "render_vertical_fov_degrees": float(render["vertical_fov_degrees"]),
                    "render_horizon_degrees": float(agent.get("cameraHorizon", render["camera_horizon_degrees"])),
                    "v4_pose": {
                        "position": agent["position"],
                        "rotation_y": float(agent["rotation"]["y"]),
                        "theta_pre_deg": round(float(pose["source_center_bearing_degrees"]), 8),
                        "theta_post_deg": round(float(pose["post_center_bearing_degrees"]), 8),
                    },
                    "geometry_evidence_live": {"center": pose, "whole_box": extent},
                    "v4_targeted": True,
                    "targeted_miner_id": config["mining_id"],
                    "targeted_miner_config_sha256": sha256_file(args.config),
                }
                accepted.append(row)
                with args.output_jsonl.open("a", encoding="utf-8") as checkpoint:
                    checkpoint.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    checkpoint.flush()
                quota_counts[(role, cid)] += 1
                component_counts[component_id] += 1
                type_counts[str(obj["objectType"])] += 1
                counts["accepted"] += 1
        except Exception as exc:
            counts["scene_errors"] += 1
            if len(errors) < 100:
                errors.append({"scene_id": scene_id, "error": f"{type(exc).__name__}: {exc}"})
        finally:
            if controller is not None:
                try:
                    controller.stop()
                except Exception:
                    pass
        if scene_index % 25 == 0:
            print(
                f"shard={args.shard_index} scenes={scene_index}/{len(scene_ids)} "
                f"accepted={len(accepted)} counts={dict(quota_counts)}",
                flush=True,
            )

    unmet: dict[str, dict[str, int]] = {}
    for role in per_role_target:
        for cell in cells:
            cid = cell_id(cell)
            role_contract = per_role_target[role]
            if isinstance(role_contract, dict):
                local_goal = math.ceil(int(role_contract[cid]) / args.num_shards)
            else:
                local_goal = math.ceil(int(role_contract) / args.num_shards)
            observed = quota_counts[(role, cid)]
            if observed < local_goal:
                unmet[f"{role}:{cid}"] = {"goal": local_goal, "observed": observed}
    report = {
        "schema_version": "0.1.0",
        "audit": "taskb_targeted_r2_mining_shard",
        "config": {"path": str(args.config), "sha256": sha256_file(args.config)},
        "split_roles": {"path": str(args.split_roles), "sha256": sha256_file(args.split_roles)},
        "shard": {"index": args.shard_index, "count": args.num_shards, "gpu": args.gpu_device},
        "scene_count": len(scene_ids),
        # Always explicit: Counter drops zero entries, and a zero-yield shard
        # must still state that it accepted nothing.
        "counts": {"accepted": int(counts.get("accepted", 0)), **dict(sorted(counts.items()))},
        "quota_counts": {f"{role}:{cid}": value for (role, cid), value in sorted(quota_counts.items())},
        "unmet_local_quotas": unmet,
        "target_type_counts": dict(type_counts.most_common()),
        "errors_preview": errors,
        "output": {"path": str(args.output_jsonl), "sha256": sha256_file(args.output_jsonl)},
        "pass": not unmet and not errors,
    }
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"pass": report["pass"], "counts": report["counts"], "unmet": unmet}, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
