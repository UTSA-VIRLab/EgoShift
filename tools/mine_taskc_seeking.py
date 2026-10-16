#!/usr/bin/env python3
"""Mine Task-C full occlusions by construction rather than by accident.

The template miner plays fixed ego-motion paths and keeps whatever occlusions
happen to occur, which yields roughly six per hundred rooms -- too rare to
build a balanced class from. This miner instead computes, for each target, the
camera positions from which a specific piece of furniture should completely
cover it, and renders those.

The acceptance test is unchanged: every candidate goes through
``derive_live_full_occlusion`` exactly as before -- exactly empty target mask,
projected target region, a closer non-structural blocker overlapping it, no
same-type distractor, and the clipped-view guard on the source frame. Only the
search changed, never the standard.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

RELEASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = RELEASE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from egoshift.exact_mask_bbox import (  # noqa: E402
    COORDINATE_CONVENTION,
    EXACT_BBOX_SOURCE,
    EXACT_EVIDENCE_TYPE,
)
from egoshift.hashing import canonical_json_bytes, sha256_bytes, sha256_file  # noqa: E402
from egoshift.split_roles import require_complete_split_roles  # noqa: E402
from egoshift.taskc_live_evidence import (  # noqa: E402
    LiveEvidenceConfig,
    LiveEvidenceRejected,
    derive_live_full_occlusion,
    derive_live_out_of_view,
    derive_live_visible,
    exact_mask_bbox,
    exact_mask_sha256,
    exact_object_mask,
)
from egoshift.taskc_occlusion_geometry import (  # noqa: E402
    OcclusionSearchConfig,
    propose_occluding_poses,
)
from egoshift.taskc_source import (  # noqa: E402
    CANONICAL_OPTIONS,
    CLASS_TO_ANSWER,
    is_structural_target,
)

MINER_SEED = "egoshift-taskc-seeking-occlusion-v1"
ROLE_ORDER = ("tuning_dev", "public_dev", "hidden_test")
RENDER = {
    "width": 640,
    "height": 480,
    "vertical_fov_degrees": 90.0,
    "horizon": 0.0,
    # Both views aim at a deterministically jittered offset from the target.
    # Exact aiming put the boxed referent at cx=0.500+/-0.011 in every View 1
    # (and the blocker dead-centre in View 2), while the trajectory-mined
    # visible/out-of-fov pool scatters targets across the frame -- a balanced
    # probe separated the pipelines at 79% from one statistic. Jitter restores
    # overlap so the selection-layer matcher can equalise the residual.
    "yaw_jitter_degrees": 14.0,
    "horizon_jitter_degrees": 8.0,
}
SOURCE_DISTANCE_RANGE = (1.0, 4.5)


def _stable(*values: object) -> str:
    return sha256_bytes(canonical_json_bytes([MINER_SEED, *values]))


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _save_png(path: Path, frame: Any) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        Image.fromarray(frame).convert("RGB").save(
            temporary, format="PNG", compress_level=9
        )
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _display(object_type: str) -> str:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", object_type)
    return " ".join(spaced.split()).casefold() or "object"


def _load_roles(path: Path, roles: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    binding = require_complete_split_roles(payload)
    scenes = []
    for scene_id, record in sorted(payload["scenes"].items()):
        if record.get("drop_exact_alias_nonrepresentative") is True:
            continue
        if record.get("role") not in roles:
            continue
        scenes.append(
            {
                "scene_id": scene_id,
                "component": record["leakage_component_id"],
                "role": record["role"],
            }
        )
    return scenes, {
        "path": str(path),
        "sha256": sha256_file(path),
        "exposure_manifest_sha256": binding["manifest_sha256"],
        "scenes_in_requested_roles": len(scenes),
    }


def _contract(
    evidence: LiveEvidenceConfig, search: OcclusionSearchConfig
) -> dict[str, Any]:
    payload = {
        "miner_seed": MINER_SEED,
        "render": dict(RENDER),
        "evidence": evidence.__dict__,
        "search": {
            **{k: v for k, v in search.__dict__.items()},
            "standoff_multiples": list(search.standoff_multiples),
        },
        "acceptance": "derive_live_full_occlusion / _visible / _out_of_view (unchanged gates)",
        "classes": ["occluded", "visible", "out_of_fov"],
        "search_only": True,
    }
    return {"sha256": sha256_bytes(canonical_json_bytes(payload)), **payload}


def _teleport(
    controller: Any,
    position: Mapping[str, float],
    yaw: float,
    horizon: float | None = None,
) -> Any:
    event = controller.step(
        action="TeleportFull",
        position=dict(position),
        rotation={"x": 0.0, "y": float(yaw), "z": 0.0},
        horizon=float(RENDER["horizon"] if horizon is None else horizon),
        standing=True,
        forceAction=True,
    )
    if not event.metadata.get("lastActionSuccess", False):
        raise LiveEvidenceRejected("teleport_failed")
    return event


def _jitter(spread: float, *key: object) -> float:
    """Deterministic uniform in [-spread, +spread], keyed by identity."""

    unit = int(_stable("aimjitter", *key)[:12], 16) / float(16**12)
    return (2.0 * unit - 1.0) * spread


def _source_views(
    controller: Any,
    reachable: Sequence[Mapping[str, float]],
    centre: Mapping[str, float],
    *,
    maximum: int,
    scene_id: str,
    target_oid: str,
) -> list[tuple[dict[str, Any], Any]]:
    """Poses that plausibly show the target cleanly, for the View-1 frame."""

    tx, tz = float(centre["x"]), float(centre["z"])
    scored = []
    for position in reachable:
        distance = math.hypot(float(position["x"]) - tx, float(position["z"]) - tz)
        if not SOURCE_DISTANCE_RANGE[0] <= distance <= SOURCE_DISTANCE_RANGE[1]:
            continue
        scored.append((abs(distance - 2.2), distance, position))
    scored.sort(key=lambda item: (item[0], item[2]["x"], item[2]["z"]))
    views = []
    for _, distance, position in scored[: maximum * 4]:
        aim = math.degrees(
            math.atan2(tx - float(position["x"]), tz - float(position["z"]))
        ) % 360.0
        key = (scene_id, target_oid, position["x"], position["z"])
        yaw = (aim + _jitter(float(RENDER["yaw_jitter_degrees"]), "src-yaw", *key)) % 360.0
        horizon = float(RENDER["horizon"]) + _jitter(
            float(RENDER["horizon_jitter_degrees"]), "src-horizon", *key
        )
        try:
            event = _teleport(controller, position, yaw, horizon)
        except LiveEvidenceRejected:
            continue
        views.append(
            (
                {
                    "position": dict(position),
                    "yaw": yaw,
                    "horizon": horizon,
                    "distance": distance,
                },
                event,
            )
        )
        if len(views) >= maximum:
            break
    return views


def _materialize(
    *,
    scene: Mapping[str, str],
    scene_id: str,
    target_oid: str,
    evidence: Mapping[str, Any],
    source_pose: Mapping[str, Any],
    target_pose: Mapping[str, Any],
    source_event: Any,
    target_event: Any,
    asset_root: Path,
    house_sha256: str,
    contract_sha256: str,
    stable: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    answer_class = str(evidence["derived_class"])
    prefix = {"occluded": "tcs_", "visible": "tcv_", "out_of_fov": "tcu_"}[answer_class]
    item_id = prefix + stable[:20]
    base = Path(scene_id) / "taskc_seeking" / item_id
    relatives = [str(base / "view_1.png"), str(base / "view_2.png")]
    hashes = []
    for relative, event in zip(relatives, (source_event, target_event)):
        path = asset_root / relative
        _save_png(path, event.frame)
        hashes.append(sha256_file(path))
    display = _display(target_oid.split("|", 1)[0])
    row = {
        "id": item_id,
        "scene_id": scene_id,
        "task_family": "C",
        "format": "mcq",
        "frames": relatives,
        "question": (
            f"You saw the {display} in View 1. "
            "In View 2, what best describes its visibility?"
        ),
        "options": list(CANONICAL_OPTIONS),
        "answer": CLASS_TO_ANSWER[answer_class],
        "metadata": {
            "answer_class": answer_class,
            "target_pixel_count_source": evidence["source"]["pixel_count"],
            "target_pixel_count_target": (evidence.get("target") or {}).get(
                "pixel_count", 0
            ),
            "same_type_distractor_pixel_count_target": evidence[
                "same_type_distractor_pixels_target"
            ],
            "target_visibility_state_target": {
                "occluded": "OCCLUDED",
                "visible": "VISIBLE",
                "out_of_fov": "OUT_OF_VIEW",
            }[answer_class],
            "occluder_id": (evidence.get("occluder") or {}).get("object_id"),
            "occluder_overlap_ratio": (evidence.get("occluder") or {}).get(
                "overlap_ratio", 0.0
            ),
            "object_ids": [target_oid],
            "trajectory_template": f"constructed_{answer_class}_pose",
            "ground_truth_derivation": "live_exact_mask_and_geometry_replay",
        },
        "target_oid": target_oid,
        "source_step": 0,
        "target_step": 1,
        "candidate_answer_class": answer_class,
        "candidate_role": scene["role"],
        "_candidate_source": {
            "schema_version": "0.1.0",
            "physical_component_id": scene["component"],
            "role": scene["role"],
            "stable_identity_sha256": stable,
            "house_sha256": house_sha256,
            "targeted_template_contract_sha256": contract_sha256,
            "template": f"constructed_{answer_class}_pose",
            "source_pose": dict(source_pose),
            "target_pose": dict(target_pose),
            "source_frame_sha256": hashes[0],
            "target_frame_sha256": hashes[1],
            "frame_asset_root": str(asset_root),
            "gold_evidence": evidence,
        },
    }
    sidecar = {
        "evidence_type": EXACT_EVIDENCE_TYPE,
        "bbox_source": EXACT_BBOX_SOURCE,
        "coordinate_convention": COORDINATE_CONVENTION,
        "object_id": target_oid,
        "frame_path": relatives[0],
        "frame_index": 0,
        "image_size": {
            "w": int(source_event.frame.shape[1]),
            "h": int(source_event.frame.shape[0]),
        },
        "bbox_xyxy": evidence["source"]["bbox_xyxy"],
        "pixel_count": evidence["source"]["pixel_count"],
        "frame_sha256": hashes[0],
        "mask_sha256": evidence["source"]["mask_sha256"],
    }
    return row, sidecar


def _mine_scene(
    scene: Mapping[str, str],
    *,
    scenes_root: Path,
    asset_root: Path,
    fragment_path: Path,
    contract: Mapping[str, Any],
    evidence_config: LiveEvidenceConfig,
    search_config: OcclusionSearchConfig,
    gpu_device: int | None,
    max_targets: int,
    max_source_views: int,
) -> dict[str, Any]:
    scene_id = scene["scene_id"]
    house_path = scenes_root / scene_id / "house.json"
    if not house_path.is_file():
        return {"scene": dict(scene), "pass": False, "error": "missing house.json",
                "rows": [], "items": {}, "rejections": {}}
    house_sha256 = sha256_file(house_path)
    if fragment_path.is_file():
        cached = json.loads(fragment_path.read_text(encoding="utf-8"))
        if (
            cached.get("house_sha256") == house_sha256
            and cached.get("contract_sha256") == contract["sha256"]
        ):
            cached["resumed"] = True
            return cached

    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    house = json.loads(house_path.read_text(encoding="utf-8"))
    kwargs: dict[str, Any] = {
        "scene": house,
        "width": int(RENDER["width"]),
        "height": int(RENDER["height"]),
        "fieldOfView": float(RENDER["vertical_fov_degrees"]),
        "renderDepthImage": False,
        "renderInstanceSegmentation": True,
        "platform": CloudRendering,
    }
    if gpu_device is not None:
        kwargs["gpu_device"] = gpu_device

    controller = None
    rows: list[dict[str, Any]] = []
    items: dict[str, Any] = {}
    rejections: Counter[str] = Counter()
    error: str | None = None
    try:
        controller = Controller(**kwargs)
        base = controller.step(action="GetReachablePositions")
        reachable = base.metadata.get("actionReturn") or []
        if not reachable:
            raise LiveEvidenceRejected("no_reachable_positions")
        objects = [
            item for item in (base.metadata.get("objects") or [])
            if item.get("objectId") and not is_structural_target(str(item["objectId"]))
        ]
        # Try targets that have the most proposals first.
        ranked = []
        for item in objects:
            oid = str(item["objectId"])
            proposals = propose_occluding_poses(
                base.metadata.get("objects") or [], reachable, oid,
                config=search_config,
                rooms=house.get("rooms") or [],
            )
            if proposals:
                ranked.append((len(proposals), oid, proposals))
        ranked.sort(key=lambda entry: (-entry[0], entry[1]))
        if not ranked:
            rejections["no_geometric_proposal"] += 1
        for _, target_oid, proposals in ranked[:max_targets]:
            centre = None
            for item in objects:
                if str(item["objectId"]) == target_oid:
                    box = item.get("axisAlignedBoundingBox") or {}
                    centre = box.get("center") or item.get("position")
                    break
            if not isinstance(centre, Mapping):
                continue
            sources = _source_views(
                controller,
                reachable,
                centre,
                maximum=max_source_views,
                scene_id=scene_id,
                target_oid=target_oid,
            )
            if not sources:
                rejections["no_source_view"] += 1
                continue
            accepted = False
            for proposal in proposals:
                if accepted:
                    break
                proposal_key = (
                    scene_id, target_oid, proposal["occluder_id"],
                    proposal["position"]["x"], proposal["position"]["z"],
                )
                proposal_yaw = (
                    float(proposal["yaw"])
                    + _jitter(
                        float(RENDER["yaw_jitter_degrees"]), "tgt-yaw", *proposal_key
                    )
                ) % 360.0
                proposal_horizon = float(RENDER["horizon"]) + _jitter(
                    float(RENDER["horizon_jitter_degrees"]), "tgt-horizon", *proposal_key
                )
                try:
                    target_event = _teleport(
                        controller,
                        proposal["position"],
                        proposal_yaw,
                        proposal_horizon,
                    )
                except LiveEvidenceRejected as exc:
                    rejections[exc.reason] += 1
                    continue
                for source_pose, source_event in sources:
                    try:
                        evidence = derive_live_full_occlusion(
                            source_event,
                            target_event,
                            target_oid,
                            vertical_fov_degrees=float(
                                RENDER["vertical_fov_degrees"]
                            ),
                            config=evidence_config,
                        )
                    except LiveEvidenceRejected as exc:
                        rejections[exc.reason] += 1
                        continue
                    stable = _stable(
                        "item", scene["component"], target_oid,
                        proposal["occluder_id"], evidence["evidence_sha256"],
                    )
                    row, sidecar = _materialize(
                        scene=scene,
                        scene_id=scene_id,
                        target_oid=target_oid,
                        evidence=evidence,
                        source_pose=source_pose,
                        target_pose={
                            "position": proposal["position"],
                            "yaw": proposal_yaw,
                            "horizon": proposal_horizon,
                            "proposal": proposal,
                        },
                        source_event=source_event,
                        target_event=target_event,
                        asset_root=asset_root,
                        house_sha256=house_sha256,
                        contract_sha256=contract["sha256"],
                        stable=stable,
                    )
                    rows.append(row)
                    items[row["id"]] = sidecar
                    accepted = True
                    break

            # Visible and out-of-view items for the SAME target, from the SAME
            # source-view distribution. Drawing the three classes from two
            # different miners made the pipeline itself the label: a DINOv2
            # probe read the producing miner from View 1 at 96.3% and therefore
            # the class at 66%, while within one pipeline View 1 carried
            # nothing (51.4% on a two-way split). Same poses, same jitter, same
            # scenes for every class removes that by construction.
            for other_class, derive in (
                ("visible", derive_live_visible),
                ("out_of_fov", derive_live_out_of_view),
            ):
                if len(sources) < 2 and other_class == "visible":
                    continue
                emitted = False
                for index, (source_pose, source_event) in enumerate(sources):
                    if emitted:
                        break
                    if other_class == "visible":
                        partners = [
                            (pose, event)
                            for j, (pose, event) in enumerate(sources)
                            if j != index
                        ]
                    else:
                        # Turn away from the target from the same standpoint, so
                        # the camera position distribution is identical and only
                        # the heading differs.
                        partners = []
                        for turn in (150.0, 180.0, 120.0, -150.0, -120.0):
                            key = (scene_id, target_oid, other_class, turn, index)
                            yaw = (
                                float(source_pose["yaw"]) + turn
                                + _jitter(float(RENDER["yaw_jitter_degrees"]), "away-yaw", *key)
                            ) % 360.0
                            horizon = float(RENDER["horizon"]) + _jitter(
                                float(RENDER["horizon_jitter_degrees"]), "away-horizon", *key
                            )
                            try:
                                partners.append((
                                    {"position": dict(source_pose["position"]),
                                     "yaw": yaw, "horizon": horizon},
                                    _teleport(controller, source_pose["position"], yaw, horizon),
                                ))
                            except LiveEvidenceRejected as exc:
                                rejections[exc.reason] += 1
                    for partner_pose, partner_event in partners:
                        try:
                            other_evidence = derive(
                                source_event,
                                partner_event,
                                target_oid,
                                vertical_fov_degrees=float(RENDER["vertical_fov_degrees"]),
                                config=evidence_config,
                            )
                        except LiveEvidenceRejected as exc:
                            rejections[f"{other_class}:{exc.reason}"] += 1
                            continue
                        other_stable = _stable(
                            "item", scene["component"], target_oid, other_class,
                            other_evidence["evidence_sha256"],
                        )
                        other_row, other_sidecar = _materialize(
                            scene=scene,
                            scene_id=scene_id,
                            target_oid=target_oid,
                            evidence=other_evidence,
                            source_pose=source_pose,
                            target_pose=dict(partner_pose),
                            source_event=source_event,
                            target_event=partner_event,
                            asset_root=asset_root,
                            house_sha256=house_sha256,
                            contract_sha256=contract["sha256"],
                            stable=other_stable,
                        )
                        rows.append(other_row)
                        items[other_row["id"]] = other_sidecar
                        emitted = True
                        break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if controller is not None:
            try:
                controller.stop()
            except Exception:
                pass

    fragment = {
        "schema_version": "0.1.0",
        "scene": dict(scene),
        "house_sha256": house_sha256,
        "contract_sha256": contract["sha256"],
        "pass": error is None,
        "error": error,
        "rows": rows,
        "items": items,
        "rejections": dict(sorted(rejections.items())),
        "resumed": False,
    }
    _atomic_json(fragment_path, fragment)
    return fragment


def main(argv: list[str] | None = None) -> int:
    root = RELEASE_ROOT.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-roles", required=True, type=Path)
    parser.add_argument("--scenes-root", type=Path, default=root / "data/sim/scenes")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--role", action="append", choices=ROLE_ORDER, default=None)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--gpu-device", type=int, default=None)
    parser.add_argument("--max-components", type=int, default=0)
    parser.add_argument("--max-targets-per-scene", type=int, default=6)
    parser.add_argument("--max-source-views", type=int, default=3)
    parser.add_argument("--minimum-source-pixels", type=int, default=500)
    parser.add_argument("--minimum-occluder-overlap", type=float, default=0.05)
    parser.add_argument("--minimum-angular-ratio", type=float, default=1.35)
    parser.add_argument(
        "--scene-allowlist",
        type=Path,
        default=None,
        help=(
            "Optional file of scene ids (one per line). Restricts the sweep; "
            "used to re-shoot with aim jitter only the scenes a previous full "
            "sweep proved can yield accepted occlusions, since jitter does not "
            "change which scenes those are."
        ),
    )
    args = parser.parse_args(argv)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard-index must be in [0, num-shards)")

    evidence_config = LiveEvidenceConfig(
        minimum_source_pixels=args.minimum_source_pixels,
        minimum_occluder_overlap=args.minimum_occluder_overlap,
    )
    search_config = OcclusionSearchConfig(
        minimum_angular_ratio=args.minimum_angular_ratio
    )
    contract = _contract(evidence_config, search_config)
    scenes, provenance = _load_roles(
        args.split_roles.resolve(), set(args.role or ROLE_ORDER)
    )
    scenes = [
        s for s in scenes
        if (args.scenes_root.resolve() / s["scene_id"] / "house.json").is_file()
    ]
    if args.scene_allowlist is not None:
        allowed = {
            line.strip()
            for line in args.scene_allowlist.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        scenes = [s for s in scenes if s["scene_id"] in allowed]
    scenes = [
        s for s in scenes
        if int(_stable("shard", s["component"])[:8], 16) % args.num_shards
        == args.shard_index
    ]
    if args.max_components > 0:
        seen: set[str] = set()
        kept = []
        for scene in scenes:
            if scene["component"] not in seen and len(seen) >= args.max_components:
                continue
            seen.add(scene["component"])
            kept.append(scene)
        scenes = kept

    tag = f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    shard_root = args.output_root / tag
    asset_root = shard_root / "assets"
    fragment_root = shard_root / "fragments"
    rows: list[dict[str, Any]] = []
    sidecar: dict[str, Any] = {}
    rejections: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    resumed = 0
    for index, scene in enumerate(scenes, start=1):
        name = _stable("fragment", scene["component"], scene["scene_id"])[:24] + ".json"
        print(
            f"[taskc-seek] shard {args.shard_index}/{args.num_shards} "
            f"scene {index}/{len(scenes)} {scene['scene_id']} ({scene['role']}) "
            f"rows={len(rows)}",
            flush=True,
        )
        fragment = _mine_scene(
            scene,
            scenes_root=args.scenes_root.resolve(),
            asset_root=asset_root.resolve(),
            fragment_path=fragment_root / name,
            contract=contract,
            evidence_config=evidence_config,
            search_config=search_config,
            gpu_device=args.gpu_device,
            max_targets=args.max_targets_per_scene,
            max_source_views=args.max_source_views,
        )
        resumed += 1 if fragment.get("resumed") else 0
        rejections.update(fragment.get("rejections") or {})
        if not fragment.get("pass"):
            errors.append({"scene": fragment.get("scene"), "error": fragment.get("error")})
        rows.extend(fragment.get("rows") or [])
        sidecar.update(fragment.get("items") or {})

    # One row per physical component, target AND CLASS. The key used to omit
    # the class, which was right when this miner only produced occlusions: now
    # that it emits all three classes for the same target, that key threw away
    # two of every three rows (21,763 of 46,612 on the 2026-08-24 sweep) and,
    # worse, decided each target's class by whichever row happened to win the
    # tie-break. That made the target's own properties predict the class, which
    # a View-1 pixel probe duly read at 43% against 33% chance. Keeping one row
    # per class per target lets selection draw matched triples instead.
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["_candidate_source"]["physical_component_id"], row["target_oid"],
               row["candidate_answer_class"])
        prior = unique.get(key)
        if prior is None or (
            float(row["metadata"]["occluder_overlap_ratio"])
            > float(prior["metadata"]["occluder_overlap_ratio"])
        ):
            unique[key] = row
    selected = sorted(
        unique.values(), key=lambda r: r["_candidate_source"]["stable_identity_sha256"]
    )

    jsonl = args.output_root / f"taskc_seeking_private.{tag}.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    _atomic_json(
        args.output_root / f"taskc_seeking_exact_bbox.{tag}.json",
        {
            "schema_version": "0.1.0",
            "evidence_type": EXACT_EVIDENCE_TYPE,
            "items": {row["id"]: sidecar[row["id"]] for row in selected if row["id"] in sidecar},
        },
    )
    report = {
        "schema_version": "0.1.0",
        "audit": "taskc_seeking_occlusion_shard",
        "contract": contract,
        "split_roles": provenance,
        "shard": {"index": args.shard_index, "count": args.num_shards, "gpu": args.gpu_device},
        "scenes_processed": len(scenes),
        "scenes_resumed": resumed,
        "rows_before_identity_cap": len(rows),
        "rows": len(selected),
        "rows_by_role": dict(Counter(r["candidate_role"] for r in selected)),
        "occlusions_per_component": (
            round(len(selected) / len({s["component"] for s in scenes}), 4)
            if scenes else 0.0
        ),
        "rejections": dict(sorted(rejections.items())),
        "scene_errors": errors,
        "output": {"path": str(jsonl), "sha256": sha256_file(jsonl), "rows": len(selected)},
        "pass": not errors,
    }
    _atomic_json(args.output_root / f"taskc_seeking_audit.{tag}.json", report)
    print(
        json.dumps(
            {k: report[k] for k in
             ("rows", "rows_by_role", "scenes_processed", "occlusions_per_component")},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
