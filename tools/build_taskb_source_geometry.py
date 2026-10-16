#!/usr/bin/env python3
"""Create a hash-pinned, pose-derived Task-B candidate source index.

The output contains only rows that pass static image-quality, explicit
source-bearing, whole-3D-box post-zone, and stored-gold agreement gates.  It is
an internal construction artifact, never a public question file.

Reservoir poses are read from saved trajectory JSON.  Targeted rows must carry
their rendered ``v4_pose`` and live segmentation quality evidence.  No field of
view or rendered visibility is used to infer the source-bearing envelope.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from egoshift.taskb_geometry import (
    corner_post_diagnostic,
    item_pose_diagnostic,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _xyz(point: dict[str, Any] | list[float]) -> tuple[float, float, float]:
    if isinstance(point, dict):
        return float(point["x"]), float(point["y"]), float(point["z"])
    return float(point[0]), float(point[1]), float(point[2])


def load_aabb(patterns: Iterable[str]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    index: dict[str, dict[str, Any]] = {}
    files: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for pattern in patterns:
        for raw_path in sorted(glob.glob(pattern)):
            path = Path(raw_path).resolve()
            if path in seen_paths:
                continue
            seen_paths.add(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            for scene_id, objects in payload.items():
                index.setdefault(scene_id, {}).update(objects)
            files.append(
                {"path": str(path), "sha256": sha256_file(path), "scenes": len(payload)}
            )
    if not files:
        raise ValueError("no AABB files matched")
    return index, files


class TrajectoryReader:
    def __init__(self, root: Path):
        self.root = root
        self.key: tuple[str, str] | None = None
        self.steps: dict[int, dict[str, Any]] = {}

    def step(self, scene_id: str, trajectory_subdir: str, step_index: int) -> dict[str, Any]:
        key = (scene_id, trajectory_subdir)
        if key != self.key:
            path = self.root / scene_id / "trajectories" / trajectory_subdir / "trajectory.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.steps = {
                int(step.get("step_index", index)): step
                for index, step in enumerate(payload.get("steps") or [])
            }
            self.key = key
        if step_index not in self.steps:
            raise ValueError(f"missing trajectory step {key}:{step_index}")
        return self.steps[step_index]


def _camera_evidence(
    row: dict[str, Any], source_kind: str, trajectories: TrajectoryReader
) -> tuple[dict[str, float], float, float, dict[str, int] | None]:
    if source_kind == "targeted":
        pose = row.get("v4_pose") or {}
        position = pose.get("position")
        yaw = pose.get("rotation_y")
        if not isinstance(position, dict) or yaw is None:
            raise ValueError("targeted row lacks v4_pose position/rotation_y")
        return position, float(yaw), float(row.get("render_horizon_degrees", 30.0)), None
    trajectory = str(row.get("trajectory_subdir") or "")
    step_index = int(row.get("source_step"))
    step = trajectories.step(str(row["scene_id"]), trajectory, step_index)
    agent = step["agent"]
    return (
        agent["position"],
        float(agent["rotation"]["y"]),
        float(agent.get("cameraHorizon", 0.0)),
        step.get("instance_pixel_counts"),
    )


def _same_type_count(pixel_counts: dict[str, int] | None, target_oid: str, threshold: int) -> int | None:
    if pixel_counts is None:
        return None
    target_type = target_oid.split("|")[0]
    return sum(
        1
        for oid, pixels in pixel_counts.items()
        if oid.split("|")[0] == target_type and int(pixels) >= threshold
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    input_path = args.input.resolve()
    input_sha256 = sha256_file(input_path)
    asset_root = args.asset_root.resolve()
    aabb, aabb_files = load_aabb(args.aabb_glob)
    trajectories = TrajectoryReader(args.scenes_root.resolve())
    counts: Counter[str] = Counter()
    cell_capacity: dict[str, Counter[str]] = defaultdict(Counter)
    bearing_min = math.inf
    bearing_max = -math.inf
    output_rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            counts["input_rows"] += 1
            row = json.loads(line)
            try:
                if not row.get("gate1_pre"):
                    counts["reject_gate1"] += 1
                    continue
                if int(row.get("visible_px") or 0) < args.minimum_visible_pixels:
                    counts["reject_visible_pixels"] += 1
                    continue
                if args.source_kind == "reservoir" and float(row.get("coverage") or 0.0) < args.minimum_coverage:
                    counts["reject_coverage"] += 1
                    continue
                if float(row.get("extent_near_margin_deg") or 0.0) < args.minimum_margin:
                    counts["reject_stored_margin"] += 1
                    continue
                frames = row.get("frames")
                if not isinstance(frames, list) or len(frames) != 1:
                    counts["reject_frame_schema"] += 1
                    continue
                asset = asset_root / str(frames[0])
                if not asset.is_file():
                    counts["reject_missing_asset"] += 1
                    continue
                scene_id = str(row["scene_id"])
                target_oid = str(row["target_oid"])
                box_record = (aabb.get(scene_id) or {}).get(target_oid) or {}
                corners = row.get("aabb_corners") or box_record.get("oobb") or box_record.get("aabb")
                if not corners:
                    counts["reject_missing_3d_box"] += 1
                    continue
                camera_position, yaw, horizon, pixel_counts = _camera_evidence(
                    row, args.source_kind, trajectories
                )
                rotation = int(row["rotation_degrees"])
                raw_turn = str(row.get("turn") or "").lower()
                geometry_turn = "left" if rotation == 180 and raw_turn in {"any", "free180"} else raw_turn
                expected_answer = str(row["answer"])
                pose = item_pose_diagnostic(
                    camera_position=camera_position,
                    camera_yaw_degrees=yaw,
                    corners=corners,
                    rotation_degrees=rotation,
                    turn_direction=geometry_turn,
                    expected_answer=expected_answer,
                    source_half_width_degrees=args.source_half_width,
                )
                source_bearing = float(pose["source_center_bearing_degrees"])
                if not pose["source_envelope"]["strictly_inside"]:
                    counts["reject_source_envelope"] += 1
                    continue
                if not pose["answer_matches"]:
                    counts["reject_center_gold_mismatch"] += 1
                    continue
                extent = corner_post_diagnostic(
                    corners=corners,
                    camera_position=camera_position,
                    camera_yaw_degrees=yaw,
                    rotation_degrees=rotation,
                    turn_direction=geometry_turn,
                    expected_answer=expected_answer,
                )
                if not extent["all_corners_match_expected"]:
                    counts["reject_whole_box_zone"] += 1
                    continue
                if extent["minimum_corner_boundary_margin_degrees"] < args.minimum_margin:
                    counts["reject_replayed_margin"] += 1
                    continue
                same_type_count = _same_type_count(
                    pixel_counts, target_oid, args.same_type_pixel_threshold
                )
                if args.source_kind == "targeted":
                    same_type_count = row.get("same_type_visible_count", same_type_count)
                if same_type_count != 1:
                    counts["reject_same_type_uniqueness"] += 1
                    continue
                bbox = row.get("target_mask_bbox_xyxy")
                if args.source_kind == "targeted":
                    if not (isinstance(bbox, list) and len(bbox) == 4):
                        counts["reject_missing_target_bbox"] += 1
                        continue
                    x0, y0, x1, y1 = [int(value) for value in bbox]
                    if min(x0, y0, 640 - x1, 480 - y1) < args.minimum_edge_margin_pixels:
                        counts["reject_mask_edge_margin"] += 1
                        continue
                enriched = dict(row)
                bearing_min = min(bearing_min, source_bearing)
                bearing_max = max(bearing_max, source_bearing)
                enriched["geometry_evidence"] = {
                    "source_center_bearing_degrees": round(source_bearing, 8),
                    "source_envelope_half_width_degrees": args.source_half_width,
                    "source_envelope_margin_degrees": round(
                        float(pose["source_envelope"]["margin_degrees"]), 8
                    ),
                    "post_center_bearing_degrees": round(
                        float(pose["post_center_bearing_degrees"]), 8
                    ),
                    "derived_answer": pose["derived_answer"],
                    "whole_box": extent,
                    "camera_yaw_degrees": yaw,
                    "camera_horizon_degrees": horizon,
                    "render_vertical_fov_degrees": float(
                        row.get("render_vertical_fov_degrees", args.render_vertical_fov)
                    ),
                    "same_type_visible_count": int(same_type_count),
                    "same_type_pixel_threshold": args.same_type_pixel_threshold,
                    "target_bbox_xyxy": bbox,
                }
                enriched["source_kind"] = args.source_kind
                enriched["source_file_sha256"] = input_sha256
                enriched["source_line"] = line_number
                output_rows.append(enriched)
                direction_key = "any" if rotation == 180 else geometry_turn
                cell_capacity[f"{rotation}:{direction_key}"][expected_answer] += 1
                counts["accepted_rows"] += 1
            except Exception as exc:
                counts["row_errors"] += 1
                if len(errors) < 100:
                    errors.append(
                        {"line": line_number, "id": row.get("id"), "error": f"{type(exc).__name__}: {exc}"}
                    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "schema_version": "0.1.0",
        "audit": "taskb_static_pose_geometry_source_index",
        "source_kind": args.source_kind,
        "input": {"path": str(input_path), "sha256": input_sha256},
        "output": {"path": str(args.output), "sha256": sha256_file(args.output)},
        "aabb_files": aabb_files,
        "parameters": {
            "source_bearing_half_width_degrees": args.source_half_width,
            "minimum_visible_pixels": args.minimum_visible_pixels,
            "minimum_coverage": args.minimum_coverage,
            "minimum_whole_box_boundary_margin_degrees": args.minimum_margin,
            "same_type_pixel_threshold": args.same_type_pixel_threshold,
            "minimum_target_mask_edge_margin_pixels_for_targeted": args.minimum_edge_margin_pixels,
            "render_vertical_fov_degrees": args.render_vertical_fov,
        },
        "counts": dict(sorted(counts.items())),
        "accepted_source_bearing_range_degrees": (
            [bearing_min, bearing_max] if output_rows else None
        ),
        "accepted_capacity": {
            cell: dict(sorted(values.items())) for cell, values in sorted(cell_capacity.items())
        },
        "row_errors_preview": errors,
        "pass": counts["row_errors"] == 0 and bool(output_rows),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--source-kind", required=True, choices=("reservoir", "targeted"))
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--scenes-root", required=True, type=Path)
    parser.add_argument("--aabb-glob", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--source-half-width", type=float, default=66.58677555362947)
    parser.add_argument("--minimum-visible-pixels", type=int, default=2500)
    parser.add_argument("--minimum-coverage", type=float, default=0.8)
    parser.add_argument("--minimum-margin", type=float, default=15.0)
    parser.add_argument("--same-type-pixel-threshold", type=int, default=30)
    parser.add_argument("--minimum-edge-margin-pixels", type=int, default=2)
    parser.add_argument("--render-vertical-fov", type=float, default=120.0)
    args = parser.parse_args()
    report = build(args)
    print(json.dumps({"pass": report["pass"], "counts": report["counts"]}, indent=2))
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
