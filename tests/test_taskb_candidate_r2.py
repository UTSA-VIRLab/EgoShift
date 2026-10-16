from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from egoshift.hashing import sha256_file
from egoshift.split_roles import canonical_exposure_manifest_sha256
from egoshift.taskb_candidate_r2 import (
    CandidateCapacityError,
    _assign_options,
    finite_label_quotas,
    solve_role,
    validate_contract,
)
from egoshift.hashing import sha256_file as _sha256_file
from egoshift.taskb_targeted_merge import merge_targeted_shards


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs" / "taskb_candidate_r2.json"


def _candidate(index: int, answer: str, *, rotation: int = 30, turn: str = "left"):
    return {
        "_role": "public_dev",
        "_component": f"component_{index}",
        "_frame_path_key": f"frame_{index}",
        "_object_key": f"object_{index}",
        "_category": f"category_{index}",
        "_stable": f"{index:08d}",
        "rotation_degrees": rotation,
        "turn": turn,
        "answer": answer,
        "coverage": 0.9,
        "visible_px": 3000 + index,
        "geometry_evidence": {
            "whole_box": {"minimum_corner_boundary_margin_degrees": 30.0}
        },
    }


class CandidateContractTests(unittest.TestCase):
    def test_frozen_contract_and_finite_balancing(self) -> None:
        config = json.loads(CONTRACT.read_text(encoding="utf-8"))
        validate_contract(config)
        half_width = config["geometry"]["source_bearing_half_width_degrees"]
        for role, per_cell in (("hidden_test", 78), ("public_dev", 12), ("tuning_dev", 20)):
            quotas = finite_label_quotas(
                role=role,
                items_per_action_cell=per_cell,
                seed=config["seed"],
                half_width=half_width,
            )
            for rotation in (30, 60, 90, 120, 180):
                for turn in ("left", "right"):
                    values = [
                        value
                        for (candidate_rotation, candidate_turn, _), value in quotas.items()
                        if candidate_rotation == rotation and candidate_turn == turn
                    ]
                    self.assertEqual(sum(values), per_cell)
                    self.assertLessEqual(max(values) - min(values), 1)

    def test_milp_never_reuses_a_180_source_for_left_and_right(self) -> None:
        candidates = [
            _candidate(index, "behind", rotation=180, turn="any") for index in range(2)
        ]
        quotas = {(180, "left", "behind"): 1, (180, "right", "behind"): 1}
        caps = {
            "maximum_per_physical_component": 2,
            "maximum_per_source_frame": 1,
            "maximum_per_target_object": 1,
            "maximum_single_target_category_items": 2,
        }
        selected, report = solve_role(
            candidates=candidates,
            role="public_dev",
            quotas=quotas,
            caps=caps,
        )
        self.assertEqual(len(selected), 2)
        self.assertEqual(len({row.candidate_index for row in selected}), 2)
        self.assertEqual({row.turn_direction for row in selected}, {"left", "right"})
        self.assertTrue(report["solver_status"] == 0)

    def test_infeasible_category_cap_is_reported_not_weakened(self) -> None:
        candidates = [_candidate(index, "in front") for index in range(2)]
        for row in candidates:
            row["_category"] = "only_category"
        quotas = {(30, "left", "in front"): 2}
        caps = {
            "maximum_per_physical_component": 2,
            "maximum_per_source_frame": 1,
            "maximum_per_target_object": 1,
            "maximum_single_target_category_items": 1,
        }
        with self.assertRaises(CandidateCapacityError) as context:
            solve_role(
                candidates=candidates,
                role="public_dev",
                quotas=quotas,
                caps=caps,
            )
        self.assertEqual(
            context.exception.diagnostics["kind"], "joint_hard_cap_infeasibility"
        )

    def test_option_positions_are_deterministic_and_balanced(self) -> None:
        base = [
            {
                "id": f"eb2_{index}",
                "_split": "public_dev",
                "_rotation_degrees": 90,
                "_turn_direction": "left",
                "_answer": "to the right",
                "options": [],
            }
            for index in range(9)
        ]
        copy = [dict(row) for row in base]
        labels = ("in front", "to the right", "behind", "to the left")
        _assign_options(base, labels, "seed")
        _assign_options(copy, labels, "seed")
        self.assertEqual(base, copy)
        counts = Counter(row["options"].index(row["_answer"]) for row in base)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)


class TargetedMergeTests(unittest.TestCase):
    def _fixture(self, root: Path, *, shard_count: int = 1) -> dict[str, Path]:
        asset_root = root / "assets"
        frame = asset_root / "scene_1" / "targeted_r2" / "frame.png"
        mask = asset_root / "scene_1" / "targeted_r2_private_masks" / "mask.png"
        frame.parent.mkdir(parents=True)
        mask.parent.mkdir(parents=True)
        frame.write_bytes(b"\x89PNG\r\n\x1a\nframe")
        mask.write_bytes(b"\x89PNG\r\n\x1a\nmask")
        config = {
            "schema_version": "0.1.0",
            "mining_id": "test-miner",
            "render": {
                "width": 640,
                "height": 480,
                "vertical_fov_degrees": 120.0,
                "camera_horizon_degrees": 30.0,
            },
            "gates": {
                "source_bearing_half_width_degrees": 66.58677555362947,
                "minimum_visible_pixels": 2500,
                "minimum_mask_edge_margin_pixels": 2,
                "minimum_bbox_width_pixels": 20,
                "minimum_bbox_height_pixels": 20,
                "same_type_visible_count": 1,
                "minimum_whole_box_post_zone_margin_degrees": 15.0,
            },
            "cells": [
                {
                    "rotation_degrees": 90,
                    "turn": "left",
                    "answer": "to the right",
                    "desired_source_bearing_degrees": 0.0,
                }
            ],
            "target_rows_per_role_and_cell": {
                "hidden_test": {"90:left:to the right": 1}
            },
            "maximum_rows_per_physical_component": 2,
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
        exposure_files = [{"path": "fixture-history.jsonl", "sha256": "0" * 64}]
        roles = {
            "schema_version": "0.1.0",
            "pass": True,
            "exposure_files": exposure_files,
            "exposure_ledger": {
                "status": "complete_reviewed",
                "complete": True,
                "audit_sha256": "1" * 64,
                "manifest_sha256": canonical_exposure_manifest_sha256(exposure_files),
                "reviewed_manifest_count": 1,
            },
            "scenes": {
                "scene_1": {
                    "role": "hidden_test",
                    "leakage_component_id": "component_1",
                    "drop_exact_alias_nonrepresentative": False,
                    "outcome_exposed_component": False,
                }
            },
        }
        roles_path = root / "roles.json"
        roles_path.write_text(json.dumps(roles), encoding="utf-8")
        corners = [
            [x, y, z]
            for x in (-0.1, 0.1)
            for y in (0.0, 1.0)
            for z in (1.9, 2.1)
        ]
        row = {
            "scene_id": "scene_1",
            "physical_scene_component_id": "component_1",
            "preassigned_role": "hidden_test",
            "target_oid": "Chair|1",
            "target_type": "Chair",
            "rotation_degrees": 90,
            "turn": "left",
            "answer": "to the right",
            "visible_px": 3000,
            "extent_near_margin_deg": 42.273689,
            "gate1_pre": True,
            "frames": ["scene_1/targeted_r2/frame.png"],
            "target_mask_private_path": "scene_1/targeted_r2_private_masks/mask.png",
            "target_mask_sha256": sha256_file(mask),
            "rgb_sha256": sha256_file(frame),
            "target_mask_bbox_xyxy": [100, 100, 200, 200],
            "target_mask_edge_margin_pixels": 100,
            "same_type_visible_count": 1,
            "same_type_visible_instances": [{"object_id": "Chair|1", "pixels": 3000}],
            "aabb_corners": corners,
            "render_vertical_fov_degrees": 120.0,
            "render_horizon_degrees": 30.0,
            "v4_pose": {
                "position": {"x": 0.0, "y": 0.9, "z": 0.0},
                "rotation_y": 0.0,
                "theta_pre_deg": 0.0,
                "theta_post_deg": 90.0,
            },
            "geometry_evidence_live": {},
            "v4_targeted": True,
            "targeted_miner_id": "test-miner",
            "targeted_miner_config_sha256": sha256_file(config_path),
        }
        # Use the implementation itself for the exact floating-point margin
        # fixture, avoiding a brittle rounded constant.
        from egoshift.taskb_geometry import corner_post_diagnostic

        whole = corner_post_diagnostic(
            camera_position=row["v4_pose"]["position"],
            camera_yaw_degrees=0,
            corners=corners,
            rotation_degrees=90,
            turn_direction="left",
            expected_answer="to the right",
        )
        row["extent_near_margin_deg"] = round(
            whole["minimum_corner_boundary_margin_degrees"], 8
        )
        shard_path = root / "shard0.jsonl"
        shard_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        report = {
            "schema_version": "0.1.0",
            "audit": "taskb_targeted_r2_mining_shard",
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "split_roles": {"path": str(roles_path), "sha256": sha256_file(roles_path)},
            "shard": {"index": 0, "count": shard_count, "gpu": 0},
            "counts": {"accepted": 1},
            "quota_counts": {"hidden_test:90:left:to the right": 1},
            "unmet_local_quotas": {},
            "errors_preview": [],
            "output": {"path": str(shard_path), "sha256": sha256_file(shard_path)},
            "pass": True,
        }
        report_path = root / "shard0.report.json"
        report_path.write_text(json.dumps(report), encoding="utf-8")
        return {
            "config": config_path,
            "roles": roles_path,
            "asset_root": asset_root,
            "report": report_path,
        }

    def test_verified_single_shard_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._fixture(root)
            audit = merge_targeted_shards(
                config_path=paths["config"],
                split_roles_path=paths["roles"],
                asset_root=paths["asset_root"],
                shard_report_paths=[paths["report"]],
                output_path=root / "merged.jsonl",
                report_path=root / "merge.report.json",
            )
            self.assertTrue(audit["pass"])
            self.assertEqual(audit["counts"]["rows"], 1)

    def _short_quota_fixture(self, root: Path) -> dict[str, Path]:
        paths = self._fixture(root)
        config = json.loads(paths["config"].read_text(encoding="utf-8"))
        # Ask for two rows in the cell the fixture supplies one row for.
        config["target_rows_per_role_and_cell"]["hidden_test"][
            "90:left:to the right"
        ] = 2
        paths["config"].write_text(json.dumps(config), encoding="utf-8")
        config_hash = _sha256_file(paths["config"])
        report = json.loads(paths["report"].read_text(encoding="utf-8"))
        shard_path = Path(report["output"]["path"])
        rows = [
            json.loads(line)
            for line in shard_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for row in rows:
            row["targeted_miner_config_sha256"] = config_hash
        shard_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        report["config"]["sha256"] = config_hash
        report["output"]["sha256"] = _sha256_file(shard_path)
        paths["report"].write_text(json.dumps(report), encoding="utf-8")
        return paths

    def test_global_quota_shortfall_is_rejected_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._short_quota_fixture(root)
            with self.assertRaisesRegex(ValueError, "global targeted quotas are unmet"):
                merge_targeted_shards(
                    config_path=paths["config"],
                    split_roles_path=paths["roles"],
                    asset_root=paths["asset_root"],
                    shard_report_paths=[paths["report"]],
                    output_path=root / "merged.jsonl",
                    report_path=root / "merge.report.json",
                )
            self.assertFalse((root / "merged.jsonl").exists())

    def test_allowed_shortfall_merges_but_records_quota_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._short_quota_fixture(root)
            audit = merge_targeted_shards(
                config_path=paths["config"],
                split_roles_path=paths["roles"],
                asset_root=paths["asset_root"],
                shard_report_paths=[paths["report"]],
                output_path=root / "merged.jsonl",
                report_path=root / "merge.report.json",
                allow_global_shortfall=True,
            )
            # Row-level trust holds; contract fulfilment is explicitly false.
            self.assertTrue(audit["verification_pass"])
            self.assertFalse(audit["quota_pass"])
            self.assertFalse(audit["pass"])
            self.assertTrue(audit["partial_merge"])
            self.assertEqual(
                audit["unmet_global_quotas"]["hidden_test:90:left:to the right"][
                    "shortfall"
                ],
                1,
            )
            self.assertEqual(audit["counts"]["rows"], 1)

    def test_scene_errors_are_rejected_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._fixture(root)
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            report["errors_preview"] = [
                {"scene_id": "scene_9", "error": "TimeoutError: CreateHouse"}
            ]
            paths["report"].write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "scene errors"):
                merge_targeted_shards(
                    config_path=paths["config"],
                    split_roles_path=paths["roles"],
                    asset_root=paths["asset_root"],
                    shard_report_paths=[paths["report"]],
                    output_path=root / "merged.jsonl",
                    report_path=root / "merge.report.json",
                )
            self.assertFalse((root / "merged.jsonl").exists())

    def test_allowed_scene_errors_merge_but_are_recorded_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._fixture(root)
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            report["errors_preview"] = [
                {"scene_id": "scene_9", "error": "TimeoutError: CreateHouse"}
            ]
            paths["report"].write_text(json.dumps(report), encoding="utf-8")
            audit = merge_targeted_shards(
                config_path=paths["config"],
                split_roles_path=paths["roles"],
                asset_root=paths["asset_root"],
                shard_report_paths=[paths["report"]],
                output_path=root / "merged.jsonl",
                report_path=root / "merge.report.json",
                allow_scene_errors=True,
            )
            # Emitted rows keep full row-level trust; the run is not "clean".
            self.assertTrue(audit["verification_pass"])
            self.assertTrue(audit["quota_pass"])
            self.assertFalse(audit["operational_clean"])
            self.assertFalse(audit["pass"])
            self.assertEqual(audit["scene_error_count"], 1)
            self.assertEqual(
                audit["shards"][0]["scene_errors"][0]["error"],
                "TimeoutError: CreateHouse",
            )
            self.assertEqual(audit["counts"]["rows"], 1)

    def test_a_zero_yield_shard_merges_cleanly(self) -> None:
        # Counter omits zero keys, so a shard that accepted nothing has no
        # "accepted" entry at all. That is legitimate and must not read as a
        # count mismatch.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._fixture(root)
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            shard_path = Path(report["output"]["path"])
            shard_path.write_text("", encoding="utf-8")
            report["counts"] = {"render_attempts": 1200}   # no "accepted" key
            report["quota_counts"] = {}
            report["output"]["sha256"] = _sha256_file(shard_path)
            paths["report"].write_text(json.dumps(report), encoding="utf-8")
            audit = merge_targeted_shards(
                config_path=paths["config"],
                split_roles_path=paths["roles"],
                asset_root=paths["asset_root"],
                shard_report_paths=[paths["report"]],
                output_path=root / "merged.jsonl",
                report_path=root / "merge.report.json",
                allow_global_shortfall=True,
            )
            self.assertTrue(audit["verification_pass"])
            self.assertEqual(audit["counts"]["rows"], 0)

    def test_missing_accepted_with_claimed_quota_still_fails(self) -> None:
        # The permissive reading applies only when the report agrees it
        # produced nothing; a report claiming rows but omitting the count is
        # malformed and must still be rejected.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._fixture(root)
            report = json.loads(paths["report"].read_text(encoding="utf-8"))
            report["counts"] = {"render_attempts": 5}      # no "accepted" key
            paths["report"].write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "accepted-row count mismatch"):
                merge_targeted_shards(
                    config_path=paths["config"],
                    split_roles_path=paths["roles"],
                    asset_root=paths["asset_root"],
                    shard_report_paths=[paths["report"]],
                    output_path=root / "merged.jsonl",
                    report_path=root / "merge.report.json",
                )

    def test_incomplete_shard_set_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = self._fixture(root, shard_count=2)
            with self.assertRaisesRegex(ValueError, "incomplete shard set"):
                merge_targeted_shards(
                    config_path=paths["config"],
                    split_roles_path=paths["roles"],
                    asset_root=paths["asset_root"],
                    shard_report_paths=[paths["report"]],
                    output_path=root / "merged.jsonl",
                    report_path=root / "merge.report.json",
                )


if __name__ == "__main__":
    unittest.main()


class NonUniformQuotaTests(unittest.TestCase):
    """Per-rotation sizing, added so a rotation whose support cannot be built
    at scale can be shrunk instead of filled with an unbalanced label mix."""

    HALF_WIDTH = 66.58677555362947

    def test_omitting_the_override_reproduces_the_uniform_allocation(self) -> None:
        from egoshift.taskb_candidate_r2 import finite_label_quotas

        base = finite_label_quotas(
            role="hidden_test", items_per_action_cell=78,
            seed="s", half_width=self.HALF_WIDTH,
        )
        same = finite_label_quotas(
            role="hidden_test", items_per_action_cell=78,
            seed="s", half_width=self.HALF_WIDTH, items_by_rotation=None,
        )
        empty = finite_label_quotas(
            role="hidden_test", items_per_action_cell=78,
            seed="s", half_width=self.HALF_WIDTH, items_by_rotation={},
        )
        self.assertEqual(base, same)
        self.assertEqual(base, empty)

    def test_override_changes_only_the_named_rotation(self) -> None:
        from egoshift.taskb_candidate_r2 import finite_label_quotas

        base = finite_label_quotas(
            role="hidden_test", items_per_action_cell=120,
            seed="s", half_width=self.HALF_WIDTH,
        )
        mixed = finite_label_quotas(
            role="hidden_test", items_per_action_cell=120,
            seed="s", half_width=self.HALF_WIDTH,
            items_by_rotation={180: 19, 90: 78},
        )
        for cell, count in base.items():
            if cell[0] in (90, 180):
                continue
            self.assertEqual(mixed[cell], count, cell)
        for rotation, expected in ((180, 19), (90, 78)):
            total = sum(v for k, v in mixed.items() if k[0] == rotation and k[1] == "left")
            self.assertEqual(total, expected)

    def test_every_cell_still_sums_to_its_declared_size(self) -> None:
        from egoshift.taskb_candidate_r2 import action_supports, finite_label_quotas

        sizes = {30: 120, 60: 120, 90: 78, 120: 120, 180: 19}
        quotas = finite_label_quotas(
            role="hidden_test", items_per_action_cell=120,
            seed="s", half_width=self.HALF_WIDTH, items_by_rotation=sizes,
        )
        per_cell = {}
        for (rotation, turn, _label), count in quotas.items():
            per_cell[(rotation, turn)] = per_cell.get((rotation, turn), 0) + count
        for (rotation, turn), total in per_cell.items():
            self.assertEqual(total, sizes[rotation], (rotation, turn))
        # Allocation stays attainable: no label exceeds ceil(n/k).
        supports = action_supports(self.HALF_WIDTH)
        for (rotation, turn, _label), count in quotas.items():
            k = len(supports[(rotation, turn)])
            self.assertLessEqual(count, -(-sizes[rotation] // k))
