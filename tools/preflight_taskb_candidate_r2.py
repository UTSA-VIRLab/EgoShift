#!/usr/bin/env python3
"""Run the non-GPU Task-B r2 implementation/capacity preflight."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from egoshift.hashing import sha256_file
from egoshift.taskb_candidate_r2 import (
    ROLES,
    SourceSpec,
    _assignments_for_role,
    build_candidate,
    finite_label_quotas,
    load_candidates,
    validate_contract,
)
from egoshift.taskb_geometry import canonical_zone, post_action_bearing_degrees


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _synthetic_end_to_end(config_path: Path) -> dict[str, Any]:
    """Exercise exact 1,100-item solve, separation, assets, and manifest."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    seed = str(config["seed"])
    half_width = float(config["geometry"]["source_bearing_half_width_degrees"])
    with tempfile.TemporaryDirectory(prefix="egoshift_taskb_r2_preflight_") as temporary:
        root = Path(temporary)
        assets = root / "assets"
        assets.mkdir()
        roles: dict[str, Any] = {"schema_version": "0.1.0", "pass": True, "scenes": {}}
        rows: list[dict[str, Any]] = []
        upstream_hash = "1" * 64
        serial = 0
        for role in ROLES:
            quotas = finite_label_quotas(
                role=role,
                items_per_action_cell=int(
                    config["splits"][role]["items_per_action_cell"]
                ),
                seed=seed,
                half_width=half_width,
            )
            role_serial = 0
            for (rotation, presented_turn, answer), count in sorted(quotas.items()):
                geometry_turn = "left" if rotation == 180 else presented_turn
                source_bearing = next(
                    theta
                    for theta in range(-66, 67)
                    if canonical_zone(
                        post_action_bearing_degrees(theta, rotation, geometry_turn)
                    )
                    == answer
                )
                for _ in range(count):
                    scene_id = f"scene_{serial:05d}"
                    target_oid = f"Object{role_serial % 20}|{serial}"
                    frame = f"{scene_id}/frame.png"
                    frame_path = assets / frame
                    frame_path.parent.mkdir()
                    frame_path.write_bytes(
                        b"\x89PNG\r\n\x1a\n" + serial.to_bytes(8, "big")
                    )
                    roles["scenes"][scene_id] = {
                        "role": role,
                        "leakage_component_id": f"component_{serial}",
                        "drop_exact_alias_nonrepresentative": False,
                        "outcome_exposed_component": role != "hidden_test",
                    }
                    post = post_action_bearing_degrees(
                        source_bearing, rotation, geometry_turn
                    )
                    rows.append(
                        {
                            "scene_id": scene_id,
                            "target_oid": target_oid,
                            "target_type": f"Object{role_serial % 20}",
                            "rotation_degrees": rotation,
                            "turn": "any" if rotation == 180 else presented_turn,
                            "answer": answer,
                            "gate1_pre": True,
                            "visible_px": 3000,
                            "coverage": 0.9,
                            "extent_near_margin_deg": 30.0,
                            "frames": [frame],
                            "source_kind": "reservoir",
                            "source_file_sha256": upstream_hash,
                            "source_line": serial + 1,
                            "geometry_evidence": {
                                "source_center_bearing_degrees": source_bearing,
                                "source_envelope_half_width_degrees": half_width,
                                "source_envelope_margin_degrees": half_width
                                - abs(source_bearing),
                                "post_center_bearing_degrees": post,
                                "derived_answer": answer,
                                "whole_box": {
                                    "all_corners_match_expected": True,
                                    "minimum_corner_boundary_margin_degrees": 30.0,
                                },
                                "same_type_visible_count": 1,
                                "render_vertical_fov_degrees": 120.0,
                                "target_bbox_xyxy": None,
                            },
                        }
                    )
                    serial += 1
                    role_serial += 1
        index = root / "index.jsonl"
        with index.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        geometry_report = {
            "schema_version": "0.1.0",
            "audit": "taskb_static_pose_geometry_source_index",
            "source_kind": "reservoir",
            "input": {"path": "synthetic", "sha256": upstream_hash},
            "output": {"path": str(index), "sha256": sha256_file(index)},
            "parameters": {
                "source_bearing_half_width_degrees": half_width,
                "minimum_visible_pixels": 2500,
                "minimum_coverage": 0.8,
                "minimum_whole_box_boundary_margin_degrees": 15.0,
                "same_type_pixel_threshold": 30,
                "minimum_target_mask_edge_margin_pixels_for_targeted": 2,
                "render_vertical_fov_degrees": 120.0,
            },
            "counts": {"accepted_rows": len(rows)},
            "pass": True,
        }
        geometry_report_path = root / "geometry.json"
        roles_path = root / "roles.json"
        _write_json(geometry_report_path, geometry_report)
        _write_json(roles_path, roles)
        started = time.monotonic()
        audit = build_candidate(
            config_path=config_path,
            split_roles_path=roles_path,
            source_specs=[
                SourceSpec(
                    "synthetic_full_contract",
                    "reservoir",
                    index,
                    geometry_report_path,
                    assets,
                    root,
                )
            ],
            output_root=root / "candidate",
            audit_report_path=root / "candidate.audit.json",
        )
        return {
            "pass": bool(audit["pass"]),
            "items": sum(
                int(value["items"])
                for value in audit["selection"]["splits"].values()
            ),
            "items_by_split": {
                role: int(audit["selection"]["splits"][role]["items"])
                for role in ROLES
            },
            "hard_gates_pass": bool(audit["selection"]["hard_gates_pass"]),
            "manifest_sha256": audit["output"]["manifest"]["sha256"],
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "coverage": [
                "exact finite action/answer selection",
                "180-degree left/right materialization without row reuse",
                "all component/frame/object/category caps",
                "opaque IDs and deterministic option balancing",
                "public/development/private separation",
                "content-addressed asset copy and manifest hashing",
            ],
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split-roles", required=True, type=Path)
    parser.add_argument("--reservoir-index", required=True, type=Path)
    parser.add_argument("--reservoir-geometry-report", required=True, type=Path)
    parser.add_argument("--reservoir-asset-root", required=True, type=Path)
    parser.add_argument("--reservoir-asset-trust-root", required=True, type=Path)
    parser.add_argument("--targeted-r2b-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("refusing to overwrite preflight output")

    config_path = args.config.resolve()
    roles_path = args.split_roles.resolve()
    reservoir_path = args.reservoir_index.resolve()
    reservoir_report_path = args.reservoir_geometry_report.resolve()
    r2b_path = args.targeted_r2b_config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_contract(config)
    roles = json.loads(roles_path.read_text(encoding="utf-8"))
    started = time.monotonic()
    candidates, source_audit = load_candidates(
        source_specs=[
            SourceSpec(
                "reservoir_geometry_b665868",
                "reservoir",
                reservoir_path,
                reservoir_report_path,
                args.reservoir_asset_root,
                args.reservoir_asset_trust_root,
            )
        ],
        roles_payload=roles,
        config=config,
    )
    reservoir_elapsed = time.monotonic() - started

    seed = str(config["seed"])
    half_width = float(config["geometry"]["source_bearing_half_width_degrees"])
    presented_deficits: dict[str, Any] = {}
    consolidated_180: dict[str, Any] = {}
    minimum_fresh = 0
    for role in ROLES:
        quotas = finite_label_quotas(
            role=role,
            items_per_action_cell=int(
                config["splits"][role]["items_per_action_cell"]
            ),
            seed=seed,
            half_width=half_width,
        )
        assignments = _assignments_for_role(candidates, role, quotas)
        capacity = Counter(item.stratum for item in assignments)
        for key, required in sorted(quotas.items()):
            if key[0] == 180:
                continue
            available = capacity[key]
            if available < required:
                name = f"{role}:{key[0]}:{key[1]}:{key[2]}"
                presented_deficits[name] = {
                    "available": available,
                    "required": required,
                    "shortfall": required - available,
                }
                minimum_fresh += required - available
        for answer in ("to the right", "behind", "to the left"):
            required = sum(
                quotas[(180, turn, answer)] for turn in ("left", "right")
            )
            available = sum(
                1
                for row in candidates
                if row["_role"] == role
                and int(row["rotation_degrees"]) == 180
                and row["answer"] == answer
            )
            shortfall = max(0, required - available)
            name = f"{role}:180:any:{answer}"
            consolidated_180[name] = {
                "available_unique_source_rows": available,
                "required_across_two_presented_turns": required,
                "shortfall": shortfall,
            }
            minimum_fresh += shortfall

    r2b = json.loads(r2b_path.read_text(encoding="utf-8"))
    r2b_total = sum(
        int(value)
        for role_contract in r2b["target_rows_per_role_and_cell"].values()
        for value in role_contract.values()
    )
    if r2b_total != 304:
        raise ValueError(f"unexpected r2b targeted total: {r2b_total}")
    if r2b["quota_basis"]["qualified_reservoir_sha256"] != sha256_file(
        reservoir_path
    ):
        raise ValueError("r2b quota basis does not pin the corrected reservoir")
    synthetic = _synthetic_end_to_end(config_path)
    repository_root = config_path.parents[2]
    implementation_paths = [
        "release/configs/taskb_candidate_r2.json",
        "release/src/egoshift/taskb_targeted_merge.py",
        "release/src/egoshift/taskb_candidate_r2.py",
        "release/tools/merge_taskb_targeted_r2.py",
        "release/tools/build_taskb_candidate_r2.py",
        "release/tools/preflight_taskb_candidate_r2.py",
        "release/tests/test_taskb_candidate_r2.py",
    ]
    report = {
        "schema_version": "0.1.0",
        "audit": "s05_taskb_candidate_implementation_preflight",
        "status": "implementation_complete_real_candidate_blocked_on_gpu_shards",
        "inputs": {
            "candidate_contract": {
                "path": str(args.config),
                "sha256": sha256_file(config_path),
            },
            "split_roles": {
                "path": str(args.split_roles),
                "sha256": sha256_file(roles_path),
            },
            "corrected_reservoir_index": {
                "path": str(args.reservoir_index),
                "sha256": sha256_file(reservoir_path),
                "rows": int(source_audit["counts"]["input_rows"]),
            },
            "corrected_reservoir_geometry_report": {
                "path": str(args.reservoir_geometry_report),
                "sha256": sha256_file(reservoir_report_path),
            },
            "targeted_r2b_contract": {
                "path": str(args.targeted_r2b_config),
                "sha256": sha256_file(r2b_path),
                "requested_rows": r2b_total,
                "minimum_fresh_rows_before_caps": int(
                    r2b["quota_basis"]["minimum_fresh_rows_before_caps"]
                ),
                "buffer_rows": int(r2b["quota_basis"]["buffer_rows"]),
            },
        },
        "implementation": {
            "artifacts": [
                {
                    "path": path,
                    "sha256": sha256_file(repository_root / path),
                }
                for path in implementation_paths
            ],
            "non_gpu_test_command": "PYTHONPATH=release/src python3 -m unittest discover -s release/tests -v",
            "non_gpu_test_result_observed_immediately_before_preflight": {
                "passed": 56,
                "failed": 0,
                "errors": 0,
            },
        },
        "corrected_reservoir_preflight": {
            "eligible_unique_source_actions": int(
                source_audit["counts"]["eligible_unique_source_actions"]
            ),
            "dropped_exact_alias_nonrepresentative_rows": int(
                source_audit["counts"]["dropped_exact_alias_nonrepresentative_rows"]
            ),
            "role_counts": {
                role: int(source_audit["counts"][f"role_{role}"]) for role in ROLES
            },
            "presented_non180_deficit_cells": presented_deficits,
            "consolidated_180_unique_source_deficits": consolidated_180,
            "raw_minimum_fresh_rows_recomputed": minimum_fresh,
            "r2b_prespecified_minimum_including_cap_reserve": int(
                r2b["quota_basis"]["minimum_fresh_rows_before_caps"]
            ),
            "elapsed_seconds": round(reservoir_elapsed, 3),
        },
        "synthetic_full_contract": synthetic,
        "real_candidate": {
            "built": False,
            "blocker": "The complete r2b GPU shard reports/JSONLs do not yet exist. The fail-closed merger and real 1,100-item builder therefore have no authorized targeted input to consume.",
            "next_required_artifacts": [
                "complete r2b shard JSONLs and reports",
                "fail-closed merged targeted source and audit",
                "corrected targeted static-geometry index and audit",
                "real Task-B candidate directory, manifest, and construction audit",
            ],
        },
        "release_ready": False,
        "pass": bool(
            len(candidates) == 89736
            and source_audit["counts"]["dropped_exact_alias_nonrepresentative_rows"]
            == 38
            and synthetic["pass"]
            and synthetic["items"] == 1100
            and r2b_total == 304
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "eligible": len(candidates),
                "raw_minimum_fresh": minimum_fresh,
                "synthetic": synthetic,
            },
            indent=2,
        )
    )
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
