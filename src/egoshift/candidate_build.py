"""Deterministic construction of an audited, non-release EgoShift candidate.

The builder separates public model inputs from private gold/provenance and
never copies simulator assets.  Asset redistribution is a later external gate.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .records import read_json, read_jsonl, write_jsonl_lines
from .support import GeometryConfig, analytic_support
from .taskb_audit import audit_taskb_rows


TASK_B_LABELS = ("in front", "to the right", "behind", "to the left")
TASK_B_PLAN: dict[tuple[int, str], tuple[str, ...]] = {
    (30, "left"): ("in front", "to the right"),
    (30, "right"): ("in front", "to the left"),
    (60, "left"): ("in front", "to the right"),
    (60, "right"): ("in front", "to the left"),
    (90, "left"): ("in front", "to the right", "behind"),
    (90, "right"): ("in front", "to the left", "behind"),
    (120, "left"): ("to the right", "behind"),
    (120, "right"): ("to the left", "behind"),
    (180, "any"): ("to the right", "behind", "to the left"),
}
TASK_C_CLASS_TO_ANSWER = {
    "visible": "Visible — It is visible in the current view, either fully or partially.",
    "out_of_fov": "Out of View — It is not in frame because the camera is not directed toward it.",
    "occluded": "Occluded — It is in frame but completely hidden by another object.",
}
TASK_C_LABELS = tuple(TASK_C_CLASS_TO_ANSWER.values())
PUBLIC_FIELDS = ("id", "task_family", "format", "frames", "question", "options")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_digest(*values: object) -> str:
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_category(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold()) or "object"


def _display_category(value: str) -> str:
    substitutions = {
        "countertop": "countertop",
        "doorframe": "doorframe",
        "doorway": "doorway",
        "floorlamp": "floor lamp",
        "desklamp": "desk lamp",
        "tvstand": "TV stand",
        "houseplant": "house plant",
        "garbagecan": "garbage can",
        "coffeemachine": "coffee machine",
        "armchair": "armchair",
        "teddybear": "teddy bear",
        "sidetable": "side table",
        "diningtable": "dining table",
        "washingmachine": "washing machine",
        "clothesdryer": "clothes dryer",
        "laundryhamper": "laundry hamper",
        "basketball": "basketball",
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
    category = _canonical_category(value)
    if category in substitutions:
        return substitutions[category]
    # Fallback CamelCase splitting preserves names absent from the small map.
    split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value).replace("_", " ").strip()
    return split.casefold() or "object"


def scene_role(scene_id: str, seed_string: str) -> str:
    value = int(stable_digest(seed_string, scene_id), 16)
    return "tuning_dev" if value % 2 == 0 else "public_dev"


def _task_b_quotas(items_per_rotation: int) -> dict[tuple[int, str, str], int]:
    quotas: dict[tuple[int, str, str], int] = {}
    for (rotation, turn), answers in TASK_B_PLAN.items():
        directions = 1 if rotation == 180 else 2
        denominator = directions * len(answers)
        if items_per_rotation % denominator:
            raise ValueError(
                f"{items_per_rotation} items at rotation {rotation} cannot be exactly "
                f"balanced across {denominator} action/answer cells"
            )
        per_answer = items_per_rotation // denominator
        for answer in answers:
            quotas[(rotation, turn, answer)] = per_answer
    return quotas


def _task_b_identity(row: Mapping[str, Any]) -> tuple[object, ...]:
    rotation = int(row["rotation_degrees"])
    turn = "any" if rotation == 180 else str(row["turn"]).lower()
    return (
        row["scene_id"],
        row["target_oid"],
        tuple(row["frames"]),
        rotation,
        turn,
    )


def load_task_b_candidates(
    repo_root: Path,
    sources: Iterable[tuple[str, Mapping[str, Any]]],
    *,
    minimum_pixels: int,
    minimum_coverage: float,
    minimum_margin: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load, gate, deduplicate, and geometry-check raw Task-B candidates."""

    geometry = GeometryConfig()
    by_identity: dict[tuple[object, ...], dict[str, Any]] = {}
    answer_conflicts: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    source_hashes: dict[str, str] = {}

    for source_name, source in sources:
        source_path = repo_root / str(source["path"])
        asset_root = repo_root / str(source["asset_root"])
        observed_hash = sha256_file(source_path)
        if observed_hash != source["sha256"]:
            raise ValueError(
                f"{source_name} hash mismatch: expected {source['sha256']}, observed {observed_hash}"
            )
        source_hashes[source_name] = observed_hash
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                counts["raw_rows"] += 1
                row = json.loads(line)
                if not row.get("gate1_pre"):
                    counts["reject_gate1"] += 1
                    continue
                if int(row.get("visible_px") or 0) < minimum_pixels:
                    counts["reject_pixels"] += 1
                    continue
                if float(row.get("coverage") or 0.0) < minimum_coverage:
                    counts["reject_coverage"] += 1
                    continue
                if float(row.get("extent_near_margin_deg") or 0.0) < minimum_margin:
                    counts["reject_margin"] += 1
                    continue
                rotation = int(row["rotation_degrees"])
                raw_turn = str(row.get("turn", "")).lower()
                turn = "any" if rotation == 180 else raw_turn
                cell = (rotation, turn)
                if cell not in TASK_B_PLAN or row.get("answer") not in TASK_B_PLAN[cell]:
                    counts["reject_plan"] += 1
                    continue
                support_turn = "left" if turn == "any" else turn
                if row["answer"] not in analytic_support(rotation, support_turn, geometry):
                    counts["reject_analytic_support"] += 1
                    continue
                frames = row.get("frames")
                if not isinstance(frames, list) or len(frames) != 1:
                    counts["reject_frame_schema"] += 1
                    continue
                asset_path = asset_root / str(frames[0])
                if not asset_path.is_file():
                    counts["reject_missing_asset"] += 1
                    continue

                identity = _task_b_identity(row)
                answer = str(row["answer"])
                existing = by_identity.get(identity)
                if existing is not None and existing["answer"] != answer:
                    answer_conflicts.append(
                        {
                            "identity_sha256": stable_digest(*identity),
                            "first_answer": existing["answer"],
                            "second_answer": answer,
                            "source": source_name,
                            "line": line_number,
                        }
                    )
                    continue

                target_type = str(row.get("target_type") or str(row["target_oid"]).split("|")[0])
                candidate = dict(row)
                candidate.update(
                    {
                        "_identity": identity,
                        "_source_name": source_name,
                        "_source_line": line_number,
                        "_asset_root": str(source["asset_root"]),
                        "_asset_path": asset_path,
                        "_source_frame": str(frames[0]),
                        "_category": _canonical_category(target_type),
                        "_display_category": _display_category(target_type),
                        "_stable": stable_digest(*identity),
                    }
                )
                if existing is None:
                    by_identity[identity] = candidate
                    counts["kept_unique"] += 1
                else:
                    counts["duplicate_same_answer"] += 1
                    old_quality = (
                        float(existing.get("coverage") or 0.0),
                        float(existing.get("extent_near_margin_deg") or 0.0),
                        int(existing.get("visible_px") or 0),
                    )
                    new_quality = (
                        float(candidate.get("coverage") or 0.0),
                        float(candidate.get("extent_near_margin_deg") or 0.0),
                        int(candidate.get("visible_px") or 0),
                    )
                    if new_quality > old_quality:
                        by_identity[identity] = candidate

    if answer_conflicts:
        raise ValueError(f"Task-B source contains {len(answer_conflicts)} conflicting gold identities")
    rows = list(by_identity.values())
    counts["eligible_unique"] = len(rows)
    return rows, {
        "source_hashes": source_hashes,
        "counts": dict(sorted(counts.items())),
        "answer_conflicts": answer_conflicts,
    }


def select_task_b(
    candidates: list[dict[str, Any]],
    *,
    items_per_rotation: int,
    category_cap: int,
    maximum_per_scene: int,
    maximum_per_frame: int,
    excluded_scenes: set[str],
    required_role: str | None,
    role_seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select an exactly support-balanced pool under diversity and role caps."""

    quotas = _task_b_quotas(items_per_rotation)
    pools: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["scene_id"] in excluded_scenes:
            continue
        if required_role is not None and scene_role(row["scene_id"], role_seed) != required_role:
            continue
        rotation = int(row["rotation_degrees"])
        turn = "any" if rotation == 180 else str(row["turn"]).lower()
        cell = (rotation, turn, str(row["answer"]))
        if cell in quotas:
            pools[cell].append(row)

    for cell in pools:
        pools[cell].sort(
            key=lambda row: (
                -float(row.get("coverage") or 0.0),
                -float(row.get("extent_near_margin_deg") or 0.0),
                -int(row.get("visible_px") or 0),
                row["_stable"],
            )
        )
    missing_capacity = {
        str(cell): {"available": len(pools[cell]), "required": quota}
        for cell, quota in quotas.items()
        if len(pools[cell]) < quota
    }
    if missing_capacity:
        raise ValueError(f"insufficient Task-B source capacity: {missing_capacity}")

    selected: list[dict[str, Any]] = []
    used: set[tuple[object, ...]] = set()
    category_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    frame_counts: Counter[str] = Counter()
    cell_counts: Counter[tuple[int, str, str]] = Counter()
    # Constrained cells are filled first.  The availability ratio is computed
    # before looking at model/human outcomes and is part of the frozen builder.
    cell_order = sorted(
        quotas,
        key=lambda cell: (len(pools[cell]) / quotas[cell], len(pools[cell]), cell),
    )
    for cell in cell_order:
        while cell_counts[cell] < quotas[cell]:
            best: dict[str, Any] | None = None
            best_score: tuple[object, ...] | None = None
            for row in pools[cell]:
                identity = row["_identity"]
                frame_key = f"{row['_asset_root']}::{row['_source_frame']}"
                if identity in used:
                    continue
                if category_counts[row["_category"]] >= category_cap:
                    continue
                if scene_counts[row["scene_id"]] >= maximum_per_scene:
                    continue
                if frame_counts[frame_key] >= maximum_per_frame:
                    continue
                score = (
                    -float(row.get("coverage") or 0.0),
                    -float(row.get("extent_near_margin_deg") or 0.0),
                    -int(row.get("visible_px") or 0),
                    category_counts[row["_category"]],
                    scene_counts[row["scene_id"]],
                    frame_counts[frame_key],
                    row["_stable"],
                )
                if best_score is None or score < best_score:
                    best, best_score = row, score
            if best is None:
                raise ValueError(
                    f"Task-B selection stalled at {cell}: {cell_counts[cell]}/{quotas[cell]}; "
                    f"category cap={category_cap}, scene cap={maximum_per_scene}, "
                    f"frame cap={maximum_per_frame}"
                )
            frame_key = f"{best['_asset_root']}::{best['_source_frame']}"
            used.add(best["_identity"])
            category_counts[best["_category"]] += 1
            scene_counts[best["scene_id"]] += 1
            frame_counts[frame_key] += 1
            cell_counts[cell] += 1
            selected.append(dict(best))

    # The geometric 180-degree action has identical left/right outcomes.  The
    # presented turn word is balanced exactly within every answer class.
    for answer in TASK_B_PLAN[(180, "any")]:
        group = sorted(
            [row for row in selected if int(row["rotation_degrees"]) == 180 and row["answer"] == answer],
            key=lambda row: stable_digest("180-turn", row["_stable"]),
        )
        if len(group) % 2:
            raise ValueError(f"180-degree answer group {answer!r} is not even")
        for index, row in enumerate(group):
            row["_presented_turn"] = "left" if index < len(group) // 2 else "right"
    for row in selected:
        if int(row["rotation_degrees"]) != 180:
            row["_presented_turn"] = str(row["turn"]).lower()

    return selected, {
        "items": len(selected),
        "items_per_rotation": dict(
            sorted(Counter(str(row["rotation_degrees"]) for row in selected).items())
        ),
        "raw_cell_counts": {str(cell): count for cell, count in sorted(cell_counts.items())},
        "category_counts": dict(category_counts.most_common()),
        "distinct_categories": len(category_counts),
        "maximum_category_count": max(category_counts.values()),
        "distinct_scenes": len(scene_counts),
        "maximum_per_scene_observed": max(scene_counts.values()),
        "distinct_source_frames": len(frame_counts),
        "maximum_per_source_frame_observed": max(frame_counts.values()),
        "source_counts": dict(Counter(row["_source_name"] for row in selected)),
        "required_role": required_role,
        "excluded_scene_count": len(excluded_scenes),
    }


def _task_b_question(category: str, direction: str, rotation: int) -> str:
    return (
        f"In View 1 you can see the {category}. Staying in place, you turn "
        f"{direction} by {rotation} degrees. Relative to the way you are now "
        f"facing, where is the {category}?"
    )


def materialize_task_b(
    selected: list[dict[str, Any]], candidate_id: str, split: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    for source in selected:
        rotation = int(source["rotation_degrees"])
        turn = source["_presented_turn"]
        item_hash = stable_digest(
            candidate_id,
            "B",
            source["scene_id"],
            source["target_oid"],
            source["_source_frame"],
            rotation,
            turn,
        )
        item_id = "eb_" + item_hash[:20]
        if item_id in ids:
            raise ValueError(f"Task-B public ID collision: {item_id}")
        ids.add(item_id)
        row = {
            "id": item_id,
            "task_family": "B",
            "format": "mcq",
            "frames": [source["_source_frame"]],
            "question": _task_b_question(source["_display_category"], turn, rotation),
            "options": list(TASK_B_LABELS),
            "answer": source["answer"],
            "metadata": {
                "rotation_degrees": rotation,
                "turn_direction": turn,
                "action_cell": f"{turn}_{rotation}",
            },
            "_split": split,
            "_source": source,
        }
        rows.append(row)
    _assign_balanced_options(
        rows,
        TASK_B_LABELS,
        group_key=lambda row: (
            row["metadata"]["rotation_degrees"],
            row["metadata"]["turn_direction"],
            row["answer"],
        ),
    )
    return rows


def _task_c_identity(row: Mapping[str, Any]) -> tuple[object, ...]:
    return (
        row["scene_id"],
        row.get("target_oid"),
        tuple(row["frames"]),
        row.get("source_step"),
        row.get("target_step"),
    )


def _validate_task_c_row(row: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping):
        return ["missing metadata"]
    answer_class = str(metadata.get("answer_class"))
    expected_answer = TASK_C_CLASS_TO_ANSWER.get(answer_class)
    if expected_answer is None:
        errors.append(f"unknown answer_class={answer_class!r}")
        return errors
    if row.get("answer") != expected_answer:
        errors.append("answer text disagrees with answer_class")
    state = metadata.get("target_visibility_state_target")
    pixels = int(metadata.get("target_pixel_count_target") or 0)
    inside = metadata.get("projected_target_region_inside_frame")
    outside = metadata.get("projected_target_region_outside_frame")
    if answer_class == "visible" and not (state == "VISIBLE" and pixels > 0):
        errors.append("visible state/pixel contract failed")
    if answer_class == "out_of_fov" and not (
        state == "OUT_OF_FOV" and pixels == 0 and inside is False and outside is True
    ):
        errors.append("out_of_fov state/projection contract failed")
    if answer_class == "occluded" and not (
        state == "OCCLUDED" and pixels == 0 and inside is True and outside is False
    ):
        errors.append("occluded state/projection contract failed")
    return errors


def load_task_c_rows(
    repo_root: Path, source_name: str, source: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_path = repo_root / str(source["path"])
    asset_root = repo_root / str(source["asset_root"])
    observed_hash = sha256_file(source_path)
    if observed_hash != source["sha256"]:
        raise ValueError(
            f"{source_name} hash mismatch: expected {source['sha256']}, observed {observed_hash}"
        )
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, object]] = []
    missing_assets: list[dict[str, object]] = []
    seen_identity: set[tuple[object, ...]] = set()
    for line_number, row in enumerate(read_jsonl(source_path), start=1):
        row_errors = _validate_task_c_row(row)
        if row_errors:
            errors.append({"line": line_number, "id": row.get("id"), "errors": row_errors})
            continue
        identity = _task_c_identity(row)
        if identity in seen_identity:
            errors.append({"line": line_number, "id": row.get("id"), "errors": ["duplicate identity"]})
            continue
        seen_identity.add(identity)
        resolved_assets = [asset_root / str(frame) for frame in row["frames"]]
        absent = [str(path) for path in resolved_assets if not path.is_file()]
        if absent:
            missing_assets.append({"line": line_number, "id": row.get("id"), "paths": absent})
            continue
        candidate = dict(row)
        candidate.update(
            {
                "_identity": identity,
                "_source_name": source_name,
                "_source_line": line_number,
                "_asset_root": str(source["asset_root"]),
                "_asset_paths": resolved_assets,
                "_source_frames": list(row["frames"]),
                "_stable": stable_digest(*identity),
            }
        )
        rows.append(candidate)
    if errors or missing_assets:
        raise ValueError(
            f"Task-C source {source_name} failed: {len(errors)} row errors, "
            f"{len(missing_assets)} rows with missing assets"
        )
    return rows, {
        "source_sha256": observed_hash,
        "items": len(rows),
        "scenes": len({row["scene_id"] for row in rows}),
        "class_counts": dict(Counter(row["metadata"]["answer_class"] for row in rows)),
        "row_errors": errors,
        "missing_assets": missing_assets,
    }


def select_task_c_development(
    candidates: list[dict[str, Any]],
    *,
    role: str,
    role_seed: str,
    excluded_scenes: set[str],
    items_per_class: int,
    maximum_per_scene: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pools: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        if row["scene_id"] in excluded_scenes:
            continue
        if scene_role(row["scene_id"], role_seed) != role:
            continue
        pools[row["metadata"]["answer_class"]].append(row)
    for answer_class, rows in pools.items():
        def evidence(row: Mapping[str, Any]) -> float:
            metadata = row["metadata"]
            if answer_class == "visible":
                return float(metadata.get("target_pixel_count_target") or 0)
            if answer_class == "occluded":
                return float(metadata.get("occluder_overlap_ratio") or 0.0)
            return float(metadata.get("target_pixel_count_source") or 0)

        rows.sort(key=lambda row: (-evidence(row), row["_stable"]))

    selected: list[dict[str, Any]] = []
    scene_counts: Counter[str] = Counter()
    for answer_class in TASK_C_CLASS_TO_ANSWER:
        for row in pools[answer_class]:
            if scene_counts[row["scene_id"]] >= maximum_per_scene:
                continue
            selected.append(dict(row))
            scene_counts[row["scene_id"]] += 1
            if sum(
                item["metadata"]["answer_class"] == answer_class for item in selected
            ) >= items_per_class:
                break
        observed = sum(item["metadata"]["answer_class"] == answer_class for item in selected)
        if observed != items_per_class:
            raise ValueError(
                f"Task-C {role} underfilled {answer_class}: {observed}/{items_per_class}"
            )
    return selected, {
        "items": len(selected),
        "class_counts": dict(Counter(row["metadata"]["answer_class"] for row in selected)),
        "distinct_scenes": len(scene_counts),
        "maximum_per_scene_observed": max(scene_counts.values()),
        "required_role": role,
    }


def materialize_task_c(
    selected: list[dict[str, Any]], candidate_id: str, split: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ids: set[str] = set()
    for source in selected:
        item_hash = stable_digest(candidate_id, "C", *_task_c_identity(source))
        item_id = "ec_" + item_hash[:20]
        if item_id in ids:
            raise ValueError(f"Task-C public ID collision: {item_id}")
        ids.add(item_id)
        answer_class = source["metadata"]["answer_class"]
        rows.append(
            {
                "id": item_id,
                "task_family": "C",
                "format": "mcq",
                "frames": list(source["_source_frames"]),
                "question": source["question"],
                "options": list(TASK_C_LABELS),
                "answer": TASK_C_CLASS_TO_ANSWER[answer_class],
                "metadata": {"answer_class": answer_class},
                "_split": split,
                "_source": source,
            }
        )
    _assign_balanced_options(
        rows,
        TASK_C_LABELS,
        group_key=lambda row: (row["answer"],),
    )
    return rows


def _assign_balanced_options(
    rows: list[dict[str, Any]],
    labels: tuple[str, ...],
    *,
    group_key,
) -> None:
    groups: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(group_key(row))].append(row)
    label_index = {label: index for index, label in enumerate(labels)}
    for key, group in sorted(groups.items(), key=lambda item: repr(item[0])):
        answer = group[0]["answer"]
        if any(row["answer"] != answer for row in group):
            raise ValueError(f"option-balance group mixes answers: {key!r}")
        offset = label_index[answer]
        ordered = sorted(group, key=lambda row: stable_digest("option-order", row["id"]))
        for index, row in enumerate(ordered):
            answer_position = (index + offset) % len(labels)
            distractors = [label for label in labels if label != answer]
            distractors.sort(key=lambda label: stable_digest("distractor", row["id"], label))
            options = list(distractors)
            options.insert(answer_position, answer)
            row["options"] = options


def _hash_and_map_assets(
    rows_by_split: Mapping[str, list[dict[str, Any]]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cache: dict[Path, tuple[str, int]] = {}
    public_to_digest: dict[str, str] = {}
    mappings: dict[tuple[str, str], dict[str, Any]] = {}
    split_assets: dict[str, set[str]] = defaultdict(set)
    for split, rows in rows_by_split.items():
        for row in rows:
            source = row["_source"]
            if row["task_family"] == "B":
                actual_paths = [source["_asset_path"]]
                source_frames = [source["_source_frame"]]
            else:
                actual_paths = source["_asset_paths"]
                source_frames = source["_source_frames"]
            public_frames: list[str] = []
            for actual_path, source_frame in zip(actual_paths, source_frames):
                actual_path = Path(actual_path)
                if actual_path not in cache:
                    cache[actual_path] = (sha256_file(actual_path), actual_path.stat().st_size)
                digest, size = cache[actual_path]
                public_path = f"assets/{row['task_family'].lower()}/{digest}.png"
                prior = public_to_digest.setdefault(public_path, digest)
                if prior != digest:
                    raise ValueError(f"asset public-path collision: {public_path}")
                public_frames.append(public_path)
                split_assets[split].add(digest)
                mapping_key = (digest, str(actual_path))
                mappings[mapping_key] = {
                    "sha256": digest,
                    "bytes": size,
                    "public_path": public_path,
                    "source_root": source["_asset_root"],
                    "source_relative_path": source_frame,
                    "task_family": row["task_family"],
                }
            row["frames"] = public_frames

    overlap: dict[str, int] = {}
    splits = sorted(split_assets)
    for first_index, first in enumerate(splits):
        for second in splits[first_index + 1 :]:
            overlap[f"{first}__{second}"] = len(split_assets[first] & split_assets[second])
    return sorted(mappings.values(), key=lambda item: (item["sha256"], item["source_relative_path"])), {
        "unique_source_files": len(cache),
        "unique_content_hashes": len({digest for digest, _ in cache.values()}),
        "referenced_bytes": sum(size for _, size in cache.values()),
        "cross_split_content_hash_overlap": overlap,
    }


def _public_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in PUBLIC_FIELDS}


def _gold_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "answer": row["answer"],
        "answer_index": row["options"].index(row["answer"]),
    }


def _provenance_row(row: Mapping[str, Any]) -> dict[str, Any]:
    source = row["_source"]
    common = {
        "id": row["id"],
        "split": row["_split"],
        "task_family": row["task_family"],
        "source_name": source["_source_name"],
        "source_line": source["_source_line"],
        "scene_id": source["scene_id"],
        "target_oid": source.get("target_oid"),
        "source_item_sha256": stable_digest(source["_source_name"], source["_source_line"], source["_stable"]),
    }
    if row["task_family"] == "B":
        common.update(
            {
                "target_category": source["_category"],
                "rotation_degrees": row["metadata"]["rotation_degrees"],
                "turn_direction": row["metadata"]["turn_direction"],
                "answer_class": row["answer"],
                "source_step": source.get("source_step"),
                "source_asset_root": source["_asset_root"],
                "source_frame": source["_source_frame"],
                "visible_pixels": source.get("visible_px"),
                "silhouette_coverage": source.get("coverage"),
                "extent_boundary_margin_degrees": source.get("extent_near_margin_deg"),
                "targeted": bool(source.get("v4_targeted")),
                "v4_pose": source.get("v4_pose"),
            }
        )
    else:
        metadata = source["metadata"]
        common.update(
            {
                "answer_class": metadata["answer_class"],
                "source_step": source.get("source_step"),
                "target_step": source.get("target_step"),
                "source_asset_root": source["_asset_root"],
                "source_frames": source["_source_frames"],
                "trajectory_template": metadata.get("trajectory_template"),
                "trajectory_subdir": metadata.get("trajectory_subdir"),
                "target_visibility_state": metadata.get("target_visibility_state_target"),
                "source_pixels": metadata.get("target_pixel_count_source"),
                "target_pixels": metadata.get("target_pixel_count_target"),
                "projected_inside_frame": metadata.get("projected_target_region_inside_frame"),
                "projected_outside_frame": metadata.get("projected_target_region_outside_frame"),
                "occluder_overlap_ratio": metadata.get("occluder_overlap_ratio"),
                "occluder_id": metadata.get("occluder_id"),
            }
        )
    return common


def _recursive_absolute_strings(value: Any, prefix: str = "") -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            findings.extend(_recursive_absolute_strings(child, f"{prefix}.{key}" if prefix else key))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(_recursive_absolute_strings(child, f"{prefix}[{index}]"))
    elif isinstance(value, str) and (value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value)):
        findings.append({"field": prefix, "value": value})
    return findings


def _summarize_materialized(
    rows_by_split: Mapping[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    public_field_violations: list[dict[str, object]] = []
    absolute_public_strings: list[dict[str, str]] = []
    all_ids: list[str] = []
    option_positions: dict[str, Any] = {}
    scene_roles: dict[str, set[str]] = defaultdict(set)
    frame_roles: dict[str, set[str]] = defaultdict(set)
    object_roles: dict[str, set[str]] = defaultdict(set)

    for split, rows in rows_by_split.items():
        all_ids.extend(row["id"] for row in rows)
        for row in rows:
            public = _public_row(row)
            if tuple(public) != PUBLIC_FIELDS:
                public_field_violations.append({"id": row["id"], "fields": list(public)})
            absolute_public_strings.extend(_recursive_absolute_strings(public))
            source = row["_source"]
            scene_roles[source["scene_id"]].add(split)
            for frame in row["frames"]:
                frame_roles[frame].add(split)
            object_roles[f"{source['scene_id']}::{source.get('target_oid')}"] .add(split)
        for task in ("B", "C"):
            task_rows = [row for row in rows if row["task_family"] == task]
            option_positions[f"{split}:{task}"] = dict(
                sorted(Counter(str(row["options"].index(row["answer"])) for row in task_rows).items())
            )
    duplicate_ids = [item_id for item_id, count in Counter(all_ids).items() if count > 1]
    scene_overlap = {key: sorted(value) for key, value in scene_roles.items() if len(value) > 1}
    frame_overlap = {key: sorted(value) for key, value in frame_roles.items() if len(value) > 1}
    object_overlap = {key: sorted(value) for key, value in object_roles.items() if len(value) > 1}
    return {
        "total_items": len(all_ids),
        "duplicate_public_ids": duplicate_ids,
        "public_field_violations": public_field_violations,
        "absolute_public_strings": absolute_public_strings,
        "cross_split_scene_overlap": scene_overlap,
        "cross_split_public_frame_overlap": frame_overlap,
        "cross_split_scene_object_overlap": object_overlap,
        "option_answer_positions": option_positions,
        "pass": not (
            duplicate_ids
            or public_field_violations
            or absolute_public_strings
            or scene_overlap
            or frame_overlap
            or object_overlap
        ),
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(write_jsonl_lines(rows), encoding="utf-8")


def build_candidate(repo_root: Path, config_path: Path, output_root: Path) -> dict[str, Any]:
    config = read_json(config_path)
    candidate_id = config["candidate_id"]
    split_contract = config["global_split_contract"]
    role_seed = split_contract["seed_string"]
    task_b_config = config["task_b"]
    task_c_config = config["task_c"]

    source_files = config["source_files"]
    task_b_candidates, task_b_source_report = load_task_b_candidates(
        repo_root,
        (
            ("task_b_reservoir", source_files["task_b_reservoir"]),
            ("task_b_targeted", source_files["task_b_targeted"]),
        ),
        minimum_pixels=task_b_config["candidate_gate"]["minimum_visible_pixels"],
        minimum_coverage=task_b_config["candidate_gate"]["minimum_silhouette_coverage"],
        minimum_margin=task_b_config["candidate_gate"]["minimum_extent_boundary_margin_degrees"],
    )
    task_c_dev, task_c_dev_source_report = load_task_c_rows(
        repo_root, "task_c_development_seed", source_files["task_c_development_seed"]
    )
    task_c_hidden, task_c_hidden_source_report = load_task_c_rows(
        repo_root, "task_c_hidden_seed", source_files["task_c_hidden_seed"]
    )

    c_dev_scenes = {row["scene_id"] for row in task_c_dev}
    c_hidden_scenes = {row["scene_id"] for row in task_c_hidden}
    hidden_b_cfg = task_b_config["hidden_test"]
    hidden_b, hidden_b_report = select_task_b(
        task_b_candidates,
        items_per_rotation=hidden_b_cfg["items_per_rotation"],
        category_cap=hidden_b_cfg["maximum_single_target_category_items"],
        maximum_per_scene=hidden_b_cfg["maximum_items_per_scene"],
        maximum_per_frame=hidden_b_cfg["maximum_items_per_source_frame"],
        excluded_scenes=c_dev_scenes,
        required_role=None,
        role_seed=role_seed,
    )
    hidden_scenes = c_hidden_scenes | {row["scene_id"] for row in hidden_b}

    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    construction_reports: dict[str, Any] = {"hidden_test:B": hidden_b_report}
    rows_by_split["hidden_test"].extend(materialize_task_b(hidden_b, candidate_id, "hidden_test"))
    rows_by_split["hidden_test"].extend(
        materialize_task_c(task_c_hidden, candidate_id, "hidden_test")
    )

    b_dev_cfg = task_b_config["each_development_role"]
    c_dev_cfg = task_c_config["each_development_role"]
    for role in ("tuning_dev", "public_dev"):
        selected_b, report_b = select_task_b(
            task_b_candidates,
            items_per_rotation=b_dev_cfg["items_per_rotation"],
            category_cap=b_dev_cfg["maximum_single_target_category_items"],
            maximum_per_scene=b_dev_cfg["maximum_items_per_scene"],
            maximum_per_frame=b_dev_cfg["maximum_items_per_source_frame"],
            excluded_scenes=hidden_scenes,
            required_role=role,
            role_seed=role_seed,
        )
        selected_c, report_c = select_task_c_development(
            task_c_dev,
            role=role,
            role_seed=role_seed,
            excluded_scenes=hidden_scenes,
            items_per_class=c_dev_cfg["items_per_class"],
        )
        construction_reports[f"{role}:B"] = report_b
        construction_reports[f"{role}:C"] = report_c
        rows_by_split[role].extend(materialize_task_b(selected_b, candidate_id, role))
        rows_by_split[role].extend(materialize_task_c(selected_c, candidate_id, role))

    # Content-address image references only after selection so no public field
    # exposes scene/trajectory/object information.
    asset_mappings, asset_report = _hash_and_map_assets(rows_by_split)
    audit = _summarize_materialized(rows_by_split)

    task_b_audits: dict[str, Any] = {}
    for split, rows in rows_by_split.items():
        task_b_rows = [row for row in rows if row["task_family"] == "B"]
        expected_items = 114 if split == "hidden_test" else 30
        task_b_audits[split] = audit_taskb_rows(
            task_b_rows,
            expected_actions={
                (float(rotation), direction)
                for rotation in (30, 60, 90, 120, 180)
                for direction in ("left", "right")
            },
            expected_items_per_cell=expected_items,
        )
        if not task_b_audits[split]["pass"]:
            raise ValueError(f"materialized Task-B split failed structural audit: {split}")

    if not audit["pass"]:
        raise ValueError("candidate public/private or cross-split audit failed")
    if any(asset_report["cross_split_content_hash_overlap"].values()):
        raise ValueError("content-identical assets occur across split roles")

    provenance_rows = [
        _provenance_row(row)
        for split in ("tuning_dev", "public_dev", "hidden_test")
        for row in sorted(rows_by_split[split], key=lambda item: (item["task_family"], item["id"]))
    ]
    for split in ("tuning_dev", "public_dev", "hidden_test"):
        for task in ("B", "C"):
            task_rows = sorted(
                [row for row in rows_by_split[split] if row["task_family"] == task],
                key=lambda item: item["id"],
            )
            base = output_root / "public" / split / f"task_{task.lower()}"
            _write_jsonl(base / "questions.jsonl", [_public_row(row) for row in task_rows])
            answers = [_gold_row(row) for row in task_rows]
            if split == "hidden_test":
                _write_jsonl(output_root / "private" / split / f"task_{task.lower()}_answers.jsonl", answers)
            else:
                _write_jsonl(base / "answers.jsonl", answers)

    _write_jsonl(output_root / "private" / "provenance.jsonl", provenance_rows)
    _write_jsonl(output_root / "private" / "asset_sources.jsonl", asset_mappings)

    report = {
        "schema_version": "0.1.0",
        "candidate_id": candidate_id,
        "status": "candidate_not_release",
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "source_reports": {
            "task_b": task_b_source_report,
            "task_c_development": task_c_dev_source_report,
            "task_c_hidden": task_c_hidden_source_report,
        },
        "construction_reports": construction_reports,
        "task_b_structural_audits": task_b_audits,
        "asset_report": asset_report,
        "public_private_split_audit": audit,
        "unresolved_release_blockers": [
            "Task-B simulator/segmentation replay, explicit source-center bearing envelope, gold, and same-type uniqueness",
            "Task-C independent simulator state replay",
            "S06 shortcut/presentation/action-necessity gates",
            "institutionally authorized independent item-quality audit",
            "U6 asset/license redistribution authorization",
        ],
        "pass_for_s05_static_construction": True,
        "pass_for_s07_release_freeze": False,
    }
    _write_json(output_root / "audits" / "build_report.json", report)

    files: list[dict[str, Any]] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": path.relative_to(output_root).as_posix(),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            )
    manifest = {
        "schema_version": "0.1.0",
        "manifest_id": candidate_id,
        "status": "candidate_not_release",
        "config_sha256": report["config"]["sha256"],
        "counts": {
            split: dict(Counter(row["task_family"] for row in rows))
            for split, rows in rows_by_split.items()
        },
        "files": files,
        "assets_included": False,
        "private_gold_in_bundle": True,
        "release_ready": False,
    }
    _write_json(output_root / "manifest.json", manifest)
    return {"manifest": manifest, "report": report}
