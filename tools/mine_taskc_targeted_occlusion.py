#!/usr/bin/env python3
"""Mine fresh, live-verified Task-C occlusions in frozen physical split roles.

The miner does not consume stored Task-C answers.  It rerenders hash-pinned
scene houses with deterministic ego-motion templates, derives occlusion from
exact live instance masks plus live 3D geometry, and emits resumable per-scene
fragments, a private candidate JSONL, and an exact-mask bbox sidecar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
)
from egoshift.taskc_source import (  # noqa: E402
    CANONICAL_OPTIONS,
    CLASS_TO_ANSWER,
    DEFAULT_QUOTAS,
    ROLE_ORDER,
)


MINER_SEED = "egoshift-taskc-targeted-occlusion-v1"
EGO_ACTIONS = {
    "Pass",
    "MoveAhead",
    "MoveBack",
    "MoveLeft",
    "MoveRight",
    "RotateLeft",
    "RotateRight",
    "LookUp",
    "LookDown",
    "Stand",
    "Crouch",
}

# Frozen actions copied from the repository's Task-C enrichment family.  The
# contract hash below makes any future change an explicit new mining version.
TARGETED_TEMPLATES: dict[str, tuple[Mapping[str, Any], ...]] = {
    "lateral_full_sweep_right": (
        {"action": "Pass"},
        {"action": "MoveRight", "moveMagnitude": 0.25},
        {"action": "MoveRight", "moveMagnitude": 0.25},
        {"action": "MoveAhead", "moveMagnitude": 0.25},
        {"action": "MoveRight", "moveMagnitude": 0.45},
        {"action": "RotateLeft", "degrees": 15},
        {"action": "MoveRight", "moveMagnitude": 0.45},
    ),
    "lateral_full_sweep_left": (
        {"action": "Pass"},
        {"action": "MoveLeft", "moveMagnitude": 0.25},
        {"action": "MoveLeft", "moveMagnitude": 0.25},
        {"action": "MoveAhead", "moveMagnitude": 0.25},
        {"action": "MoveLeft", "moveMagnitude": 0.45},
        {"action": "RotateRight", "degrees": 15},
        {"action": "MoveLeft", "moveMagnitude": 0.45},
    ),
    "approach_full_occlusion_zigzag": (
        {"action": "Pass"},
        {"action": "MoveAhead", "moveMagnitude": 0.30},
        {"action": "MoveRight", "moveMagnitude": 0.25},
        {"action": "MoveAhead", "moveMagnitude": 0.35},
        {"action": "MoveLeft", "moveMagnitude": 0.25},
        {"action": "MoveAhead", "moveMagnitude": 0.40},
        {"action": "MoveAhead", "moveMagnitude": 0.40},
    ),
    "slow_approach_partial_full": (
        {"action": "Pass"},
        {"action": "MoveAhead", "moveMagnitude": 0.20},
        {"action": "MoveAhead", "moveMagnitude": 0.20},
        {"action": "MoveAhead", "moveMagnitude": 0.20},
        {"action": "MoveAhead", "moveMagnitude": 0.20},
        {"action": "MoveAhead", "moveMagnitude": 0.20},
        {"action": "MoveAhead", "moveMagnitude": 0.20},
        {"action": "MoveAhead", "moveMagnitude": 0.20},
    ),
    "lateral_partial_to_full_right": (
        {"action": "Pass"},
        {"action": "MoveRight", "moveMagnitude": 0.20},
        {"action": "MoveRight", "moveMagnitude": 0.20},
        {"action": "MoveAhead", "moveMagnitude": 0.20},
        {"action": "MoveRight", "moveMagnitude": 0.20},
        {"action": "MoveRight", "moveMagnitude": 0.20},
        {"action": "MoveRight", "moveMagnitude": 0.45},
    ),
    "lateral_partial_to_full_left": (
        {"action": "Pass"},
        {"action": "MoveLeft", "moveMagnitude": 0.20},
        {"action": "MoveLeft", "moveMagnitude": 0.20},
        {"action": "MoveAhead", "moveMagnitude": 0.20},
        {"action": "MoveLeft", "moveMagnitude": 0.20},
        {"action": "MoveLeft", "moveMagnitude": 0.20},
        {"action": "MoveLeft", "moveMagnitude": 0.45},
    ),
}


def _atomic_bytes(path: Path, payload: bytes) -> None:
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


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_bytes(
        path,
        json.dumps(
            payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ).encode("utf-8")
        + b"\n",
    )


def _stable(*values: object) -> str:
    return sha256_bytes(canonical_json_bytes((MINER_SEED, *values)))


def _shard(component: str, total: int) -> int:
    return int(_stable("shard", component), 16) % total


def _suffix(path: Path, shard_index: int, num_shards: int) -> Path:
    return path.with_name(
        f"{path.stem}.shard_{shard_index:02d}_of_{num_shards:02d}{path.suffix}"
    )


def _load_roles(path: Path, requested_roles: set[str]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    exposure_binding = require_complete_split_roles(payload)
    scenes: list[dict[str, str]] = []
    component_roles: dict[str, str] = {}
    for scene_id, record in sorted((payload.get("scenes") or {}).items()):
        if record.get("drop_exact_alias_nonrepresentative") is True:
            continue
        role = str(record.get("role") or "")
        component = str(record.get("leakage_component_id") or "")
        if role not in ROLE_ORDER or not component:
            raise ValueError(f"invalid split-role record for {scene_id}")
        previous = component_roles.setdefault(component, role)
        if previous != role:
            raise ValueError(f"component {component} crosses roles")
        if role in requested_roles:
            scenes.append({"scene_id": scene_id, "component": component, "role": role})
    scenes.sort(key=lambda item: (_stable("scene-order", item["component"], item["scene_id"]), item["scene_id"]))
    return scenes, {
        "path": str(path),
        "sha256": sha256_file(path),
        "schema_version": payload.get("schema_version"),
        "split_contract_id": payload.get("split_contract_id"),
        "exposure_ledger": exposure_binding,
    }


def _template_contract(names: Sequence[str], evidence: LiveEvidenceConfig) -> dict[str, Any]:
    payload = {
        "miner_seed": MINER_SEED,
        "templates": {name: TARGETED_TEMPLATES[name] for name in names},
        "evidence": evidence.__dict__,
        "image_size": {"w": 640, "h": 480},
        "vertical_fov_degrees": 90.0,
        "strict_exact_object_id": True,
        "same_type_fallback": False,
        "mined_classes": ["occluded", "visible", "out_of_fov"],
        "class_acceptance": {
            "occluded": "best_by_overlap_then_source_pixels_across_all_pairs",
            "visible": "first_accepted_pair_per_template_then_best_across_templates",
            "out_of_fov": "first_accepted_pair_per_template_then_best_across_templates",
        },
    }
    return {"sha256": sha256_bytes(canonical_json_bytes(payload)), **payload}


def _event_object_ids(event: Any) -> list[str]:
    metadata = getattr(event, "metadata", None) or {}
    return sorted(
        str(record["objectId"])
        for record in metadata.get("objects", [])
        if isinstance(record, Mapping) and record.get("objectId")
    )


def _action_succeeded(event: Any) -> tuple[bool, str]:
    metadata = getattr(event, "metadata", None) or {}
    success = metadata.get("lastActionSuccess")
    if success is False:
        return False, str(metadata.get("errorMessage") or "action failed")
    return True, ""


def _render_template(
    controller: Any,
    house: Mapping[str, Any],
    actions: Sequence[Mapping[str, Any]],
) -> tuple[list[Any], str | None]:
    controller.reset(scene=house)
    events: list[Any] = []
    for index, raw_action in enumerate(actions):
        action = dict(raw_action)
        if action.get("action") not in EGO_ACTIONS:
            raise ValueError(f"non-ego action forbidden: {action!r}")
        event = controller.step(**action)
        success, detail = _action_succeeded(event)
        if not success:
            return events, f"step={index}: {detail}"
        events.append(event)
    return events, None


CLASS_ORDER = ("occluded", "visible", "out_of_fov")
CLASS_STATE_NAME = {
    "occluded": "OCCLUDED",
    "visible": "VISIBLE",
    "out_of_fov": "OUT_OF_FOV",
}
CLASS_ID_PREFIX = {"occluded": "tco_", "visible": "tcv_", "out_of_fov": "tcf_"}
CLASS_DERIVERS = (
    ("occluded", derive_live_full_occlusion),
    ("visible", derive_live_visible),
    ("out_of_fov", derive_live_out_of_view),
)


def _candidate_score(candidate: Mapping[str, Any]) -> tuple[object, ...]:
    evidence = candidate["evidence"]
    derived_class = evidence["derived_class"]
    if derived_class == "occluded":
        primary = (
            -float(evidence["occluder"]["overlap_ratio"]),
            -int(evidence["source"]["pixel_count"]),
        )
    elif derived_class == "visible":
        primary = (
            -int(evidence["target"]["pixel_count"]),
            -int(evidence["source"]["pixel_count"]),
        )
    else:
        primary = (0.0, -int(evidence["source"]["pixel_count"]))
    return (
        CLASS_ORDER.index(derived_class),
        *primary,
        int(candidate["target_step"]) - int(candidate["source_step"]),
        candidate["stable"],
    )


def _mine_events(
    events: Sequence[Any],
    *,
    scene_id: str,
    component: str,
    role: str,
    template: str,
    evidence_config: LiveEvidenceConfig,
) -> tuple[dict[str, dict[str, Any]], Counter[str]]:
    best: dict[str, dict[str, Any]] = {}
    rejections: Counter[str] = Counter()
    if len(events) < 2:
        return best, rejections
    common_objects = set(_event_object_ids(events[0]))
    for event in events[1:]:
        common_objects.intersection_update(_event_object_ids(event))
    for target_oid in sorted(common_objects):
        for source_step in range(len(events) - 1):
            for target_step in range(source_step + 1, len(events)):
                for derived_class, derive in CLASS_DERIVERS:
                    key = f"{target_oid}\u0000{derived_class}"
                    if derived_class != "occluded" and key in best:
                        continue  # frozen first-accept rule for abundant classes
                    try:
                        evidence = derive(
                            events[source_step],
                            events[target_step],
                            target_oid,
                            vertical_fov_degrees=90.0,
                            config=evidence_config,
                        )
                    except LiveEvidenceRejected as exc:
                        rejections[f"{derived_class}:{exc.reason}"] += 1
                        continue
                    stable = _stable(
                        "candidate",
                        component,
                        target_oid,
                        derived_class,
                        template,
                        source_step,
                        target_step,
                        evidence["evidence_sha256"],
                    )
                    candidate = {
                        "scene_id": scene_id,
                        "component": component,
                        "role": role,
                        "template": template,
                        "source_step": source_step,
                        "target_step": target_step,
                        "target_oid": target_oid,
                        "evidence": evidence,
                        "stable": stable,
                        "source_event": events[source_step],
                        "target_event": events[target_step],
                    }
                    old = best.get(key)
                    if old is None or _candidate_score(candidate) < _candidate_score(old):
                        best[key] = candidate
    return best, rejections


def _save_png(path: Path, frame: Any) -> None:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required for targeted replay frames") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        Image.fromarray(frame).convert("RGB").save(
            temporary, format="PNG", optimize=False, compress_level=9
        )
        temporary.replace(path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _materialize_candidate(
    candidate: Mapping[str, Any],
    *,
    asset_root: Path,
    house_sha256: str,
    contract_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    derived_class = candidate["evidence"]["derived_class"]
    item_id = CLASS_ID_PREFIX[derived_class] + candidate["stable"][:20]
    class_dir = {
        "occluded": "targeted_occlusion",
        "visible": "targeted_visible",
        "out_of_fov": "targeted_out_of_fov",
    }[derived_class]
    base = Path(candidate["scene_id"]) / class_dir / item_id
    source_relative = str(base / "view_1.png")
    target_relative = str(base / "view_2.png")
    source_path = asset_root / source_relative
    target_path = asset_root / target_relative
    _save_png(source_path, candidate["source_event"].frame)
    _save_png(target_path, candidate["target_event"].frame)
    source_sha256 = sha256_file(source_path)
    target_sha256 = sha256_file(target_path)
    target_oid = candidate["target_oid"]
    category = target_oid.split("|", 1)[0]
    display = " ".join(
        part for part in __import__("re").sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", category).split()
    ).casefold()
    evidence = candidate["evidence"]
    row = {
        "id": item_id,
        "scene_id": candidate["scene_id"],
        "task_family": "C",
        "format": "mcq",
        "frames": [source_relative, target_relative],
        "question": (
            f"You saw the {display or 'object'} in View 1. "
            "In View 2, what best describes its visibility?"
        ),
        "options": list(CANONICAL_OPTIONS),
        "answer": CLASS_TO_ANSWER[derived_class],
        "metadata": {
            "answer_class": derived_class,
            "target_pixel_count_source": evidence["source"]["pixel_count"],
            "target_pixel_count_target": evidence["target"]["pixel_count"],
            "same_type_distractor_pixel_count_target": evidence[
                "same_type_distractor_pixels_target"
            ],
            "target_visibility_state_target": CLASS_STATE_NAME[derived_class],
            **(
                {
                    "occluder_id": evidence["occluder"]["object_id"],
                    "occluder_overlap_ratio": evidence["occluder"]["overlap_ratio"],
                }
                if derived_class == "occluded"
                else {}
            ),
            **(
                {"target_bbox_xyxy_target": evidence["target"]["bbox_xyxy"]}
                if derived_class == "visible"
                else {}
            ),
            "object_ids": [target_oid],
            "trajectory_template": candidate["template"],
            "views_used": [candidate["source_step"], candidate["target_step"]],
            "ground_truth_derivation": "live_exact_mask_and_geometry_replay",
        },
        "target_oid": target_oid,
        "source_step": candidate["source_step"],
        "target_step": candidate["target_step"],
        "candidate_answer_class": derived_class,
        "candidate_role": candidate["role"],
        "_candidate_source": {
            "schema_version": "0.1.0",
            "physical_component_id": candidate["component"],
            "role": candidate["role"],
            "stable_identity_sha256": candidate["stable"],
            "house_sha256": house_sha256,
            "targeted_template_contract_sha256": contract_sha256,
            "template": candidate["template"],
            "actions": list(TARGETED_TEMPLATES[candidate["template"]]),
            "source_frame_sha256": source_sha256,
            "target_frame_sha256": target_sha256,
            "frame_asset_root": str(asset_root),
            "gold_evidence": evidence,
        },
    }
    sidecar_entry = {
        "evidence_type": EXACT_EVIDENCE_TYPE,
        "bbox_source": EXACT_BBOX_SOURCE,
        "coordinate_convention": COORDINATE_CONVENTION,
        "object_id": target_oid,
        "frame_path": source_relative,
        "frame_index": candidate["source_step"],
        "image_size": {
            "w": int(candidate["source_event"].frame.shape[1]),
            "h": int(candidate["source_event"].frame.shape[0]),
        },
        "bbox_xyxy": evidence["source"]["bbox_xyxy"],
        "pixel_count": evidence["source"]["pixel_count"],
        "frame_sha256": source_sha256,
        "mask_sha256": evidence["source"]["mask_sha256"],
    }
    return row, sidecar_entry


def _mine_scene(
    scene: Mapping[str, str],
    *,
    scenes_root: Path,
    asset_root: Path,
    fragment_path: Path,
    template_names: Sequence[str],
    contract: Mapping[str, Any],
    evidence_config: LiveEvidenceConfig,
    gpu_device: int | None,
) -> dict[str, Any]:
    scene_id = scene["scene_id"]
    house_path = scenes_root / scene_id / "house.json"
    if not house_path.is_file():
        return {"scene": dict(scene), "pass": False, "error": "missing house.json", "rows": [], "items": {}}
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
        "width": 640,
        "height": 480,
        "fieldOfView": 90.0,
        "renderDepthImage": False,
        "renderInstanceSegmentation": True,
        "platform": CloudRendering,
    }
    if gpu_device is not None:
        kwargs["gpu_device"] = gpu_device
    controller = None
    best: dict[str, dict[str, Any]] = {}
    rejection_counts: Counter[str] = Counter()
    template_errors: list[dict[str, str]] = []
    try:
        controller = Controller(**kwargs)
        for template in template_names:
            try:
                events, action_failure = _render_template(
                    controller, house, TARGETED_TEMPLATES[template]
                )
                if action_failure is not None:
                    rejection_counts["targeted_action_failed"] += 1
                mined, rejected = _mine_events(
                    events,
                    scene_id=scene_id,
                    component=scene["component"],
                    role=scene["role"],
                    template=template,
                    evidence_config=evidence_config,
                )
                rejection_counts.update(rejected)
                for key, candidate in mined.items():
                    old = best.get(key)
                    if old is None or _candidate_score(candidate) < _candidate_score(old):
                        best[key] = candidate
            except Exception as exc:
                template_errors.append(
                    {"template": template, "error": f"{type(exc).__name__}: {exc}"}
                )
    finally:
        if controller is not None:
            try:
                controller.stop()
            except Exception:
                pass

    rows: list[dict[str, Any]] = []
    items: dict[str, dict[str, Any]] = {}
    for candidate in sorted(best.values(), key=_candidate_score):
        row, entry = _materialize_candidate(
            candidate,
            asset_root=asset_root,
            house_sha256=house_sha256,
            contract_sha256=contract["sha256"],
        )
        rows.append(row)
        items[row["id"]] = entry
    fragment = {
        "schema_version": "0.1.0",
        "scene": dict(scene),
        "house_sha256": house_sha256,
        "contract_sha256": contract["sha256"],
        "pass": not template_errors,
        "rows": rows,
        "items": items,
        "rejections": dict(sorted(rejection_counts.items())),
        "template_errors": template_errors,
        "resumed": False,
    }
    _atomic_json(fragment_path, fragment)
    return fragment


def build_parser() -> argparse.ArgumentParser:
    root = RELEASE_ROOT.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-roles",
        type=Path,
        default=root / "work/candidate_r2/audits/split_roles.json",
    )
    parser.add_argument("--scenes-root", type=Path, default=root / "data/sim/scenes")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root / "work/candidate_r2/taskc_targeted_occlusion_v1",
    )
    parser.add_argument("--role", action="append", choices=ROLE_ORDER, default=None)
    parser.add_argument(
        "--template", action="append", choices=tuple(TARGETED_TEMPLATES), default=None
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--gpu-device", type=int, default=None)
    parser.add_argument(
        "--max-components",
        type=int,
        default=0,
        help="Debug-only cap after deterministic shard ordering; zero means all.",
    )
    parser.add_argument("--minimum-source-pixels", type=int, default=500)
    parser.add_argument("--minimum-occluder-overlap", type=float, default=0.05)
    parser.add_argument(
        "--scene-allowlist",
        type=Path,
        default=None,
        help=(
            "Optional file of scene ids (one per line). Restricts the sweep "
            "to exactly those scenes -- used to draw every Task-C class from "
            "the same scene population as the constructed-occlusion pool."
        ),
    )
    return parser


def _row_quality(row: Mapping[str, Any]) -> tuple[object, ...]:
    metadata = row["metadata"]
    derived_class = row["candidate_answer_class"]
    if derived_class == "occluded":
        primary = (
            -float(metadata["occluder_overlap_ratio"]),
            -int(metadata["target_pixel_count_source"]),
        )
    elif derived_class == "visible":
        primary = (
            -int(metadata["target_pixel_count_target"]),
            -int(metadata["target_pixel_count_source"]),
        )
    else:
        primary = (0.0, -int(metadata["target_pixel_count_source"]))
    return (*primary, row["_candidate_source"]["stable_identity_sha256"])


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("shard-index must be in [0, num-shards)")
    requested_roles = set(args.role or ROLE_ORDER)
    template_names = tuple(args.template or TARGETED_TEMPLATES)
    evidence_config = LiveEvidenceConfig(
        minimum_source_pixels=args.minimum_source_pixels,
        minimum_occluder_overlap=args.minimum_occluder_overlap,
    )
    contract = _template_contract(template_names, evidence_config)
    scenes, split_provenance = _load_roles(args.split_roles.resolve(), requested_roles)
    role_scene_count = len(scenes)
    scenes = [
        scene
        for scene in scenes
        if (args.scenes_root.resolve() / scene["scene_id"] / "house.json").is_file()
    ]
    locally_available_scene_count = len(scenes)
    if args.scene_allowlist is not None:
        allowed_scenes = {
            line.strip()
            for line in args.scene_allowlist.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        scenes = [scene for scene in scenes if scene["scene_id"] in allowed_scenes]
    scenes = [
        scene
        for scene in scenes
        if _shard(scene["component"], args.num_shards) == args.shard_index
    ]
    if args.max_components > 0:
        allowed = []
        components: set[str] = set()
        for scene in scenes:
            if scene["component"] not in components and len(components) >= args.max_components:
                continue
            components.add(scene["component"])
            allowed.append(scene)
        scenes = allowed

    shard_root = args.output_root / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}"
    asset_root = shard_root / "assets"
    fragment_root = shard_root / "fragments"
    fragments: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes, start=1):
        fragment_name = _stable("fragment", scene["component"], scene["scene_id"])[0:24] + ".json"
        print(
            f"[taskc-targeted] shard {args.shard_index}/{args.num_shards} "
            f"scene {index}/{len(scenes)} {scene['scene_id']} ({scene['role']})",
            flush=True,
        )
        # Controller creation can time out under fleet load; that is a
        # transient host condition, not scene evidence, so retry before
        # recording the scene as errored instead of letting one timeout kill
        # the whole shard.
        fragment = None
        for attempt in range(3):
            try:
                fragment = _mine_scene(
                    scene,
                    scenes_root=args.scenes_root.resolve(),
                    asset_root=asset_root.resolve(),
                    fragment_path=fragment_root / fragment_name,
                    template_names=template_names,
                    contract=contract,
                    evidence_config=evidence_config,
                    gpu_device=args.gpu_device,
                )
                break
            except Exception as exc:
                print(
                    f"[taskc-targeted] shard {args.shard_index} scene "
                    f"{scene['scene_id']} attempt {attempt + 1} failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
        if fragment is None:
            fragment = {
                "scene": dict(scene),
                "pass": False,
                "error": "scene mining failed after 3 attempts",
                "rows": [],
                "items": {},
            }
        fragments.append(fragment)

    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    sidecar_items: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    rejections: Counter[str] = Counter()
    resumed = 0
    for fragment in fragments:
        if fragment.get("resumed"):
            resumed += 1
        rejections.update(fragment.get("rejections") or {})
        if not fragment.get("pass"):
            errors.append(
                {
                    "scene": fragment.get("scene"),
                    "error": fragment.get("error"),
                    "template_errors": fragment.get("template_errors"),
                }
            )
        for row in fragment.get("rows") or []:
            key = (
                row["_candidate_source"]["physical_component_id"],
                row["target_oid"],
                row["candidate_answer_class"],
            )
            old = by_key.get(key)
            if old is None or _row_quality(row) < _row_quality(old):
                by_key[key] = row
    rows = sorted(
        by_key.values(),
        key=lambda row: (
            ROLE_ORDER.index(row["candidate_role"]),
            CLASS_ORDER.index(row["candidate_answer_class"]),
            *_row_quality(row),
        ),
    )
    for row in rows:
        for fragment in fragments:
            entry = (fragment.get("items") or {}).get(row["id"])
            if entry is not None:
                sidecar_items[row["id"]] = entry
                break
        if row["id"] not in sidecar_items:
            errors.append({"id": row["id"], "error": "missing exact bbox sidecar entry"})

    jsonl_path = _suffix(
        args.output_root / "targeted_occlusion_private.jsonl",
        args.shard_index,
        args.num_shards,
    )
    sidecar_path = _suffix(
        args.output_root / "targeted_occlusion_exact_bbox.json",
        args.shard_index,
        args.num_shards,
    )
    report_path = _suffix(
        args.output_root / "targeted_occlusion_audit.json",
        args.shard_index,
        args.num_shards,
    )
    _atomic_bytes(jsonl_path, b"".join(canonical_json_bytes(row) + b"\n" for row in rows))
    _atomic_json(
        sidecar_path,
        {
            "schema_version": "0.1.0",
            "evidence_type": EXACT_EVIDENCE_TYPE,
            "coordinate_convention": COORDINATE_CONVENTION,
            "items": sidecar_items,
        },
    )
    role_counts = Counter(row["candidate_role"] for row in rows)
    report = {
        "schema_version": "0.1.0",
        "audit": "taskc_targeted_live_occlusion_shard",
        "pass": not errors and len(rows) == len(sidecar_items),
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "split_roles": split_provenance,
        "roles": sorted(requested_roles),
        "contract": contract,
        "counts": {
            "role_scenes_in_split_contract": role_scene_count,
            "role_scenes_with_local_house": locally_available_scene_count,
            "scenes_assigned": len(scenes),
            "components_assigned": len({scene["component"] for scene in scenes}),
            "fragments_resumed": resumed,
            "verified_unique_candidates": len(rows),
            "verified_by_role": dict(sorted(role_counts.items())),
        },
        "final_quota_reference": dict(DEFAULT_QUOTAS),
        "rejections": dict(sorted(rejections.items())),
        "errors": errors,
        "outputs": {
            "private_jsonl": {"path": str(jsonl_path), "sha256": sha256_file(jsonl_path)},
            "exact_bbox_sidecar": {
                "path": str(sidecar_path),
                "sha256": sha256_file(sidecar_path),
            },
            "asset_root": str(asset_root),
        },
    }
    _atomic_json(report_path, report)
    print(json.dumps({"pass": report["pass"], "counts": report["counts"], "report": str(report_path)}, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
