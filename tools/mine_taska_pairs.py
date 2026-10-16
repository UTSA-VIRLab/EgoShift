#!/usr/bin/env python3
"""Mine fresh, live-verified Task-A cross-view identity pairs.

Task A is a positive control, not a difficulty claim: it shows that the same
models, on the same renderer, read these scenes well. That only works if the
pool is clean, so this miner derives identity from exact live instance masks
rather than from any stored label, and records the per-view coverage statistic
the known construction cue acts on so that selection can match on it.

Both classes are mined from the same camera poses wherever possible. A "same"
item boxes one instance in two views; a "different" item boxes two distinct
instances of one type, one per view. Every boxed view must contain exactly one
identifiable instance of that type, or the box does not pick out a referent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
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

from egoshift.hashing import canonical_json_bytes, sha256_bytes, sha256_file  # noqa: E402
from egoshift.split_roles import require_complete_split_roles  # noqa: E402
from egoshift.taska_live_evidence import (  # noqa: E402
    ANSWER_DIFFERENT,
    ANSWER_SAME,
    TaskAEvidenceConfig,
    derive_live_different_objects,
    derive_live_same_object,
)
from egoshift.taskc_live_evidence import LiveEvidenceRejected  # noqa: E402

MINER_SEED = "egoshift-taska-crossview-identity-v1"
ROLE_ORDER = ("tuning_dev", "public_dev", "hidden_test")
RENDER = {
    "width": 640,
    "height": 480,
    "vertical_fov_degrees": 90.0,
    "horizon": 0.0,
    # Every view aims at a jittered offset from the target, drawn from the
    # same distribution for both classes. Exact aiming placed the boxed target
    # at the same frame position in both views of a *same* pair and at
    # systematically different positions for a *different* pair; a grouped
    # gradient-boosting probe read the answer from box-centre distance alone
    # at 68% on the 2026-08-22 pool. Jitter removes the mechanism; the
    # cue-matched selector still matches the residual.
    "yaw_jitter_degrees": 12.0,
    "horizon_jitter_degrees": 8.0,
}
DISTANCE_RANGE = (1.0, 5.0)
# Types whose instances are not reliably distinguishable as objects.
EXCLUDED_TYPES = {
    "wall", "floor", "ceiling", "room", "door", "doorway", "window",
    "doorframe", "windowframe",
}

OPTION_SAME = "Same object seen from different viewpoints"


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


def _plural(word: str) -> str:
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


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


def _contract(config: TaskAEvidenceConfig) -> dict[str, Any]:
    payload = {
        "miner_seed": MINER_SEED,
        "render": dict(RENDER),
        "distance_range_meters": list(DISTANCE_RANGE),
        "excluded_types": sorted(EXCLUDED_TYPES),
        "evidence": config.__dict__,
        "classes": [ANSWER_SAME, ANSWER_DIFFERENT],
        "strict_exact_object_id": True,
        "referent_rule": "exactly one identifiable same-type instance per boxed view",
    }
    return {"sha256": sha256_bytes(canonical_json_bytes(payload)), **payload}


def _heading(camera: Mapping[str, float], target: Mapping[str, float]) -> float:
    return math.degrees(math.atan2(target["x"] - camera["x"], target["z"] - camera["z"]))


def _viewpoints(
    reachable: Sequence[Mapping[str, float]],
    centre: Mapping[str, float],
    *,
    rng: random.Random,
    maximum: int,
    preferred_distance: float = 2.5,
) -> list[dict[str, Any]]:
    """Camera poses aimed at a point, spread over distinct viewing directions."""

    candidates = []
    for position in reachable:
        distance = math.hypot(position["x"] - centre["x"], position["z"] - centre["z"])
        if not DISTANCE_RANGE[0] <= distance <= DISTANCE_RANGE[1]:
            continue
        candidates.append((_heading(position, centre) % 360.0, distance, position))
    if not candidates:
        return []
    # Spread over heading so the two views are genuinely different viewpoints.
    buckets: dict[int, tuple[float, float, Mapping[str, float]]] = {}
    for heading, distance, position in candidates:
        key = int(heading // 30.0)
        best = buckets.get(key)
        if best is None or abs(distance - preferred_distance) < abs(
            best[1] - preferred_distance
        ):
            buckets[key] = (heading, distance, position)
    ordered = [buckets[k] for k in sorted(buckets)]
    rng.shuffle(ordered)
    yaw_jitter = float(RENDER["yaw_jitter_degrees"])
    horizon_jitter = float(RENDER["horizon_jitter_degrees"])
    return [
        {
            "position": dict(position),
            "yaw": (heading + rng.uniform(-yaw_jitter, yaw_jitter)) % 360.0,
            "horizon": float(RENDER["horizon"])
            + rng.uniform(-horizon_jitter, horizon_jitter),
            "distance": distance,
        }
        for heading, distance, position in ordered[:maximum]
    ]


def _render(controller: Any, pose: Mapping[str, Any]) -> Any:
    event = controller.step(
        action="TeleportFull",
        position=pose["position"],
        rotation={"x": 0.0, "y": pose["yaw"], "z": 0.0},
        horizon=float(pose.get("horizon", RENDER["horizon"])),
        standing=True,
        forceAction=True,
    )
    if not event.metadata.get("lastActionSuccess", False):
        raise LiveEvidenceRejected("teleport_failed")
    return event


def _instances_by_type(event: Any) -> dict[str, list[str]]:
    """Group instances by the objectId prefix, which is the identity key.

    The evidence module decides "same type" from the objectId prefix, so the
    miner must group and exclude on the same key. Grouping on ``objectType``
    instead let structural elements through: a door carries objectType
    ``Doorway`` but objectId prefix ``door``, so it evaded the exclusion list
    and then grouped as type ``door`` downstream.
    """

    grouped: dict[str, list[str]] = {}
    for item in event.metadata.get("objects") or []:
        oid = item.get("objectId")
        if not oid:
            continue
        prefix = str(oid).split("|", 1)[0]
        otype = str(item.get("objectType") or "")
        if prefix.casefold() in EXCLUDED_TYPES or otype.casefold() in EXCLUDED_TYPES:
            continue
        box = item.get("axisAlignedBoundingBox") or {}
        if not (box.get("cornerPoints") or (box.get("center") and box.get("size"))):
            continue
        grouped.setdefault(prefix, []).append(str(oid))
    return {k: sorted(v) for k, v in grouped.items()}


def _radius(event: Any, object_id: str) -> float | None:
    """Largest horizontal offset from an object's centre to its AABB corner."""

    for item in event.metadata.get("objects") or []:
        if item.get("objectId") != object_id:
            continue
        box = item.get("axisAlignedBoundingBox") or {}
        corners = box.get("cornerPoints")
        centre = box.get("center") or item.get("position")
        if not isinstance(centre, Mapping):
            return None
        if isinstance(corners, list) and len(corners) >= 8:
            best = 0.0
            for corner in corners[:8]:
                if isinstance(corner, Mapping):
                    dx = float(corner["x"]) - float(centre["x"])
                    dz = float(corner["z"]) - float(centre["z"])
                else:
                    dx = float(corner[0]) - float(centre["x"])
                    dz = float(corner[2]) - float(centre["z"])
                best = max(best, math.hypot(dx, dz))
            return best or None
        size = box.get("size")
        if isinstance(size, Mapping):
            return math.hypot(float(size["x"]), float(size["z"])) / 2.0 or None
        return None
    return None


def _preferred_distance(radius: float | None, target_bbox_width_pixels: float) -> float:
    """Distance that renders an object of this size at a workable pixel width.

    The visible-pixel floor is a gate and stays fixed; what we control is where
    the camera stands. Angular width scales as 1/distance, so the distance that
    lands a given object near the target width is computable rather than
    something to sample for.
    """

    if not radius or radius <= 0.0:
        return 2.5
    half_vertical = math.radians(float(RENDER["vertical_fov_degrees"]) / 2.0)
    half_horizontal = math.atan(
        math.tan(half_vertical) * float(RENDER["width"]) / float(RENDER["height"])
    )
    degrees_per_pixel = math.degrees(2.0 * half_horizontal) / float(RENDER["width"])
    half_angle = math.radians(target_bbox_width_pixels * degrees_per_pixel / 2.0)
    if half_angle <= 0.0:
        return 2.5
    distance = radius / math.tan(half_angle)
    return min(max(distance, DISTANCE_RANGE[0]), DISTANCE_RANGE[1])


def _centre(event: Any, object_id: str) -> Mapping[str, float] | None:
    for item in event.metadata.get("objects") or []:
        if item.get("objectId") == object_id:
            box = item.get("axisAlignedBoundingBox") or {}
            if isinstance(box.get("center"), Mapping):
                return box["center"]
            return item.get("position")
    return None


def _materialize(
    candidate: Mapping[str, Any],
    *,
    asset_root: Path,
    scene_id: str,
    house_sha256: str,
    contract_sha256: str,
) -> dict[str, Any]:
    evidence = candidate["evidence"]
    answer_class = evidence["derived_class"]
    prefix = "taa_" if answer_class == ANSWER_SAME else "tad_"
    item_id = prefix + candidate["stable"][:20]
    base = Path(scene_id) / "taska_pairs" / item_id
    relatives = [str(base / "view_1.png"), str(base / "view_2.png")]
    hashes = []
    for relative, event in zip(relatives, (candidate["event_1"], candidate["event_2"])):
        path = asset_root / relative
        _save_png(path, event.frame)
        hashes.append(sha256_file(path))
    noun = _display(evidence["object_type"])
    question = (
        f"Looking at View 1 and View 2, do these {_plural(noun)}, each highlighted "
        "with a box, show the same object or two different objects?"
    )
    option_different = f"Two different {_plural(noun)}"
    return {
        "id": item_id,
        "scene_id": scene_id,
        "task_family": "A",
        "format": "mcq",
        "frames": relatives,
        "question": question,
        "options": [OPTION_SAME, option_different],
        "answer": OPTION_SAME if answer_class == ANSWER_SAME else option_different,
        "metadata": {
            "answer_class": answer_class,
            "object_type": evidence["object_type"],
            "object_ids": [
                evidence["view_1"]["object_id"],
                evidence["view_2"]["object_id"],
            ],
            "taska_view_coverage_source": evidence["view_1"]["view_coverage"],
            "taska_view_coverage_target": evidence["view_2"]["view_coverage"],
            "taska_min_view_coverage": evidence["minimum_view_coverage"],
            "taska_view_coverage_definition": evidence["coverage_definition"],
            "taska_any_view_touches_frame_edge": evidence["any_view_touches_frame_edge"],
            "taska_visible_pixels": [
                evidence["view_1"]["visible_pixels"],
                evidence["view_2"]["visible_pixels"],
            ],
            "taska_identifiable_same_type_count": [
                evidence["view_1"]["identifiable_same_type_count"],
                evidence["view_2"]["identifiable_same_type_count"],
            ],
            "ground_truth_derivation": "live_exact_mask_identity",
        },
        "candidate_answer_class": answer_class,
        "candidate_role": candidate["role"],
        "_candidate_source": {
            "schema_version": "0.1.0",
            "physical_component_id": candidate["component"],
            "role": candidate["role"],
            "stable_identity_sha256": candidate["stable"],
            "house_sha256": house_sha256,
            "contract_sha256": contract_sha256,
            "view_1_bbox_xyxy": evidence["view_1"]["bbox_xyxy"],
            "view_2_bbox_xyxy": evidence["view_2"]["bbox_xyxy"],
            "frame_sha256": hashes,
            "frame_asset_root": str(asset_root),
            "poses": [candidate["pose_1"], candidate["pose_2"]],
            "gold_evidence": evidence,
        },
    }


def _mine_scene(
    scene: Mapping[str, str],
    *,
    scenes_root: Path,
    asset_root: Path,
    fragment_path: Path,
    contract: Mapping[str, Any],
    config: TaskAEvidenceConfig,
    gpu_device: int | None,
    max_views: int,
    max_types: int,
) -> dict[str, Any]:
    scene_id = scene["scene_id"]
    house_path = scenes_root / scene_id / "house.json"
    if not house_path.is_file():
        return {
            "scene": dict(scene),
            "pass": False,
            "error": "missing house.json",
            "rows": [],
            "rejections": {},
        }
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
    rng = random.Random(int(_stable("scene", scene_id)[:16], 16))
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
    seen_keys: set[str] = set()
    rejections: Counter[str] = Counter()
    error: str | None = None
    try:
        controller = Controller(**kwargs)
        event = controller.step(action="GetReachablePositions")
        reachable = event.metadata.get("actionReturn") or []
        if not reachable:
            raise LiveEvidenceRejected("no_reachable_positions")
        grouped = _instances_by_type(event)
        fov = float(RENDER["vertical_fov_degrees"])
        target_width = float(config.target_bbox_width_pixels)

        def radius_of(oid: str) -> float:
            return _radius(event, oid) or 0.0

        # Prefer types that can supply both classes, then physically larger
        # types: a target must clear the visible-pixel gate, and small props
        # cannot at any reachable distance. This is candidate ordering, not a
        # gate -- the gate still decides.
        types = sorted(
            grouped,
            key=lambda name: (
                -min(len(grouped[name]), 2),
                -radius_of(grouped[name][0]),
                name,
            ),
        )[:max_types]

        def poses_for(oid: str) -> list[dict[str, Any]]:
            anchor = _centre(event, oid)
            if anchor is None:
                return []
            return _viewpoints(
                reachable,
                anchor,
                rng=rng,
                maximum=max_views,
                preferred_distance=_preferred_distance(radius_of(oid), target_width),
            )

        def render_all(poses: Sequence[Mapping[str, Any]]) -> list[tuple[dict, Any]]:
            rendered = []
            for pose in poses:
                try:
                    rendered.append((dict(pose), _render(controller, pose)))
                except LiveEvidenceRejected as exc:
                    rejections[exc.reason] += 1
            return rendered

        for object_type in types:
            instances = grouped[object_type]
            views_a = render_all(poses_for(instances[0]))
            if len(views_a) < 2:
                rejections["too_few_viewpoints"] += 1
            # "same": two views both aimed at the one instance.
            same_key = f"{object_type}:{ANSWER_SAME}"
            for i in range(len(views_a)):
                if same_key in seen_keys:
                    break
                for j in range(i + 1, len(views_a)):
                    (pose1, ev1), (pose2, ev2) = views_a[i], views_a[j]
                    try:
                        evidence = derive_live_same_object(
                            ev1, ev2, instances[0],
                            vertical_fov_degrees=fov, config=config,
                        )
                    except LiveEvidenceRejected as exc:
                        rejections[f"{ANSWER_SAME}:{exc.reason}"] += 1
                        continue
                    rows.append(
                        _materialize(
                            {
                                "component": scene["component"],
                                "role": scene["role"],
                                "pose_1": pose1, "pose_2": pose2,
                                "event_1": ev1, "event_2": ev2,
                                "evidence": evidence,
                                "stable": _stable(
                                    "item", scene["component"], object_type,
                                    ANSWER_SAME, evidence["evidence_sha256"],
                                ),
                            },
                            asset_root=asset_root, scene_id=scene_id,
                            house_sha256=house_sha256,
                            contract_sha256=contract["sha256"],
                        )
                    )
                    seen_keys.add(same_key)
                    break
            # "different": view 2 must be aimed at the *second* instance, or it
            # is simply not in frame. Reusing the first instance's poses here
            # was why this class produced nothing.
            if len(instances) < 2:
                continue
            diff_key = f"{object_type}:{ANSWER_DIFFERENT}"
            views_b = render_all(poses_for(instances[1]))
            for pose1, ev1 in views_a:
                if diff_key in seen_keys:
                    break
                for pose2, ev2 in views_b:
                    try:
                        evidence = derive_live_different_objects(
                            ev1, ev2, instances[0], instances[1],
                            vertical_fov_degrees=fov, config=config,
                        )
                    except LiveEvidenceRejected as exc:
                        rejections[f"{ANSWER_DIFFERENT}:{exc.reason}"] += 1
                        continue
                    rows.append(
                        _materialize(
                            {
                                "component": scene["component"],
                                "role": scene["role"],
                                "pose_1": pose1, "pose_2": pose2,
                                "event_1": ev1, "event_2": ev2,
                                "evidence": evidence,
                                "stable": _stable(
                                    "item", scene["component"], object_type,
                                    ANSWER_DIFFERENT, evidence["evidence_sha256"],
                                ),
                            },
                            asset_root=asset_root, scene_id=scene_id,
                            house_sha256=house_sha256,
                            contract_sha256=contract["sha256"],
                        )
                    )
                    seen_keys.add(diff_key)
                    break
    except Exception as exc:  # simulator failures are operational, not corruption
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
    parser.add_argument("--max-views-per-type", type=int, default=3)
    parser.add_argument("--max-types-per-scene", type=int, default=6)
    parser.add_argument("--minimum-visible-pixels", type=int, default=2000)
    parser.add_argument("--minimum-coverage", type=float, default=0.10)
    args = parser.parse_args(argv)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard-index must be in [0, num-shards)")

    config = TaskAEvidenceConfig(
        minimum_visible_pixels=args.minimum_visible_pixels,
        minimum_coverage=args.minimum_coverage,
    )
    contract = _contract(config)
    scenes, provenance = _load_roles(
        args.split_roles.resolve(), set(args.role or ROLE_ORDER)
    )
    scenes = [
        s
        for s in scenes
        if (args.scenes_root.resolve() / s["scene_id"] / "house.json").is_file()
    ]
    scenes = [
        s
        for s in scenes
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
    rejections: Counter[str] = Counter()
    errors: list[dict[str, Any]] = []
    resumed = 0
    for index, scene in enumerate(scenes, start=1):
        name = _stable("fragment", scene["component"], scene["scene_id"])[:24] + ".json"
        print(
            f"[taska] shard {args.shard_index}/{args.num_shards} "
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
            config=config,
            gpu_device=args.gpu_device,
            max_views=args.max_views_per_type,
            max_types=args.max_types_per_scene,
        )
        resumed += 1 if fragment.get("resumed") else 0
        rejections.update(fragment.get("rejections") or {})
        if not fragment.get("pass"):
            errors.append(
                {"scene": fragment.get("scene"), "error": fragment.get("error")}
            )
        rows.extend(fragment.get("rows") or [])

    # One row per physical component, type and class: a component that supplies
    # many near-identical pairs would otherwise dominate the pool.
    unique: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            row["_candidate_source"]["physical_component_id"],
            row["metadata"]["object_type"],
            row["candidate_answer_class"],
        )
        prior = unique.get(key)
        if (
            prior is None
            or row["_candidate_source"]["stable_identity_sha256"]
            < prior["_candidate_source"]["stable_identity_sha256"]
        ):
            unique[key] = row
    selected = sorted(
        unique.values(),
        key=lambda r: r["_candidate_source"]["stable_identity_sha256"],
    )

    jsonl = args.output_root / f"taska_pairs_private.{tag}.jsonl"
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("w", encoding="utf-8") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": "0.1.0",
        "audit": "taska_pair_mining_shard",
        "contract": contract,
        "split_roles": provenance,
        "shard": {
            "index": args.shard_index,
            "count": args.num_shards,
            "gpu": args.gpu_device,
        },
        "scenes_processed": len(scenes),
        "scenes_resumed": resumed,
        "rows_before_identity_cap": len(rows),
        "rows": len(selected),
        "rows_by_class": dict(Counter(r["candidate_answer_class"] for r in selected)),
        "rows_by_role": dict(Counter(r["candidate_role"] for r in selected)),
        "rejections": dict(sorted(rejections.items())),
        "scene_errors": errors,
        "output": {
            "path": str(jsonl),
            "sha256": sha256_file(jsonl),
            "rows": len(selected),
        },
        "pass": not errors,
    }
    _atomic_json(args.output_root / f"taska_pairs_audit.{tag}.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("rows", "rows_by_class", "rows_by_role", "scenes_processed")
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
