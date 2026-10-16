from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from egoshift.taskc_source import (
    CandidateRejected,
    TaskCSourceConfig,
    _apply_robust_guard,
    prepare_taskc_source,
)
from egoshift.split_roles import canonical_exposure_manifest_sha256


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TaskCFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.scenes_root = root / "scenes"
        self.source = root / "mixed.jsonl"
        self.split_roles = root / "split_roles.json"
        self.output = root / "selected.jsonl"
        self.report = root / "audit.json"
        self.rows: list[dict] = [{"id": "not_c", "task_family": "A"}]
        self.split_scenes: dict[str, dict] = {}

    def add_scene(
        self,
        scene_number: int,
        role: str,
        component: str,
        *,
        visible_target_pixels: int = 900,
        add_structural_occlusion: bool = False,
    ) -> None:
        scene_id = f"scene_{scene_number:05d}"
        trajectory_name = "traj_fixture"
        frame_root = self.scenes_root / scene_id / "trajectories" / trajectory_name / "frames"
        frame_root.mkdir(parents=True, exist_ok=True)
        (frame_root / "frame_00.png").write_bytes(b"source")
        (frame_root / "frame_01.png").write_bytes(b"target")

        visible_oid = "VisibleObject|1"
        oof_oid = "OutObject|1"
        occluded_oid = "HiddenObject|1"
        objects = [
            {"objectId": visible_oid, "position": {"x": 0.0, "y": 0.0, "z": 2.0}},
            {"objectId": oof_oid, "position": {"x": 3.0, "y": 0.0, "z": 1.0}},
            {"objectId": occluded_oid, "position": {"x": 0.0, "y": 0.0, "z": 2.0}},
        ]
        source_states = {
            oid: {"state": "VISIBLE", "cause": None}
            for oid in (visible_oid, oof_oid, occluded_oid)
        }
        source_counts = {visible_oid: 1000, oof_oid: 1100, occluded_oid: 1200}
        target_states = {
            visible_oid: {"state": "VISIBLE", "cause": None},
            oof_oid: {"state": "OUT_OF_FOV", "cause": "OUT_OF_FOV"},
            occluded_oid: {"state": "OCCLUDED", "cause": "OCCLUDED"},
        }
        target_counts = {visible_oid: visible_target_pixels}
        direct_evidence = {
            occluded_oid: {
                "projected_region_inside_frame": True,
                "occluders": {"Chair|1": 0.8},
            }
        }
        if add_structural_occlusion:
            structural_oid = "StructuralHidden|1"
            objects.append(
                {"objectId": structural_oid, "position": {"x": 0.0, "y": 0.0, "z": 2.0}}
            )
            source_states[structural_oid] = {"state": "VISIBLE", "cause": None}
            source_counts[structural_oid] = 1300
            target_states[structural_oid] = {"state": "OCCLUDED", "cause": "OCCLUDED"}
            direct_evidence[structural_oid] = {
                "projected_region_inside_frame": True,
                "occluders": {"wall|1": 0.9},
            }

        agent = {
            "position": {"x": 0.0, "y": 0.0, "z": 0.0},
            "rotation": {"x": 0.0, "y": 0.0, "z": 0.0},
            "cameraHorizon": 0.0,
        }
        trajectory = {
            "scene_id": scene_id,
            "steps": [
                {
                    "step_index": 0,
                    "agent": agent,
                    "visibility_states": source_states,
                    "instance_pixel_counts": source_counts,
                },
                {
                    "step_index": 1,
                    "agent": agent,
                    "visibility_states": target_states,
                    "instance_pixel_counts": target_counts,
                    "occlusion_evidence": direct_evidence,
                },
            ],
        }
        _write_json(
            self.scenes_root / scene_id / "metadata.json",
            {
                "scene_id": scene_id,
                "camera_field_of_view": 90.0,
                "objects": objects,
            },
        )
        _write_json(
            self.scenes_root
            / scene_id
            / "trajectories"
            / trajectory_name
            / "trajectory.json",
            trajectory,
        )
        self.split_scenes[scene_id] = {
            "role": role,
            "leakage_component_id": component,
            "drop_exact_alias_nonrepresentative": False,
            "component_representative": scene_id,
        }
        definitions = (
            (visible_oid, "visible", visible_target_pixels, "", 0.0),
            (oof_oid, "out_of_fov", 0, "", 0.0),
            (occluded_oid, "fully_occluded", 0, "Chair|1", 0.8),
        )
        if add_structural_occlusion:
            definitions += (("StructuralHidden|1", "fully_occluded", 0, "wall|1", 0.9),)
        for index, (target_oid, answer_class, target_pixels, occluder, overlap) in enumerate(
            definitions
        ):
            source_pixels = source_counts[target_oid]
            state = {
                "visible": "VISIBLE",
                "out_of_fov": "OUT_OF_FOV",
                "fully_occluded": "OCCLUDED",
            }[answer_class]
            item_id = f"qa_{scene_number:05d}_{index}"
            self.rows.append(
                {
                    "id": item_id,
                    "scene_id": scene_id,
                    "task_family": "C",
                    "question": "legacy question",
                    "format": "mcq",
                    "options": ["legacy"],
                    "answer": "legacy",
                    "frames": [
                        f"{scene_id}/trajectories/{trajectory_name}/frames/frame_00.png",
                        f"{scene_id}/trajectories/{trajectory_name}/frames/frame_01.png",
                    ],
                    "metadata": {
                        "object_ids": [target_oid],
                        "answer_class": answer_class,
                        "trajectory_subdir": trajectory_name,
                        "views_used": [0, 1],
                        "target_pixel_count_source": source_pixels,
                        "target_pixel_count_target": target_pixels,
                        "same_type_distractor_pixel_count_target": 0,
                        "target_visibility_state_target": state,
                        "occluder_id": occluder,
                        "occluder_overlap_ratio": overlap,
                    },
                    "target_oid": target_oid,
                    "source_step": 0,
                    "target_step": 1,
                    "candidate_answer_class": answer_class,
                    "rules_config": "/private/home/configs/rules.yaml",
                }
            )

    def add_exact_alias_row(self, source_row_index: int, scene_number: int) -> None:
        alias_id = f"scene_{scene_number:05d}"
        source = json.loads(json.dumps(self.rows[source_row_index]))
        source["id"] = f"qa_alias_{scene_number}"
        source["scene_id"] = alias_id
        self.rows.append(source)
        self.split_scenes[alias_id] = {
            "role": "tuning_dev",
            "leakage_component_id": "component_alias",
            "drop_exact_alias_nonrepresentative": True,
            "component_representative": "scene_00001",
        }

    def finish(self) -> None:
        with self.source.open("w", encoding="utf-8") as handle:
            for row in self.rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        exposure_files = [{"path": "fixture-history.jsonl", "sha256": "0" * 64}]
        _write_json(
            self.split_roles,
            {
                "schema_version": "0.1.0",
                "split_contract_id": "fixture",
                "seed": "fixture",
                "pass": True,
                "exposure_files": exposure_files,
                "exposure_ledger": {
                    "status": "complete_reviewed",
                    "complete": True,
                    "audit_sha256": "1" * 64,
                    "manifest_sha256": canonical_exposure_manifest_sha256(exposure_files),
                    "reviewed_manifest_count": 1,
                },
                "exposed_component_assigned_hidden_violations": [],
                "scenes": self.split_scenes,
            },
        )

    def config(self, per_class: int = 1, expected_hash: str | None = None) -> TaskCSourceConfig:
        return TaskCSourceConfig(
            source_path=self.source,
            expected_source_sha256=expected_hash or _sha256(self.source),
            split_roles_path=self.split_roles,
            scenes_root=self.scenes_root,
            robust_sidecar_paths=(),
            output_path=self.output,
            report_path=self.report,
            quotas={role: per_class for role in ("tuning_dev", "public_dev", "hidden_test")},
        )


class TaskCSourceTests(unittest.TestCase):
    def _fixture(self, root: Path) -> TaskCFixture:
        fixture = TaskCFixture(root)
        fixture.add_scene(1, "tuning_dev", "component_tuning", add_structural_occlusion=True)
        fixture.add_scene(2, "public_dev", "component_public")
        fixture.add_scene(3, "hidden_test", "component_hidden")
        fixture.add_scene(
            4,
            "tuning_dev",
            "component_tuning",
            visible_target_pixels=900,
        )
        fixture.add_exact_alias_row(1, 9)
        fixture.finish()
        return fixture

    def test_exact_quota_selection_is_deterministic_and_physically_unique(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            first = prepare_taskc_source(fixture.config())
            self.assertTrue(first["pass"])
            self.assertEqual(first["output"]["rows"], 9)
            self.assertEqual(first["rejections"]["structural_occluder"], 1)
            self.assertEqual(first["rejections"]["exact_alias_nonrepresentative"], 1)
            self.assertGreaterEqual(
                first["counts"]["duplicate_physical_target_class_rows"], 3
            )
            selected_bytes = fixture.output.read_bytes()
            selected = [json.loads(line) for line in selected_bytes.splitlines()]
            keys = {
                (
                    row["_candidate_source"]["physical_component_id"],
                    row["target_oid"],
                    row["metadata"]["answer_class"],
                )
                for row in selected
            }
            self.assertEqual(len(keys), len(selected))
            self.assertTrue(all("rules_config" not in row for row in selected))
            second = prepare_taskc_source(fixture.config())
            self.assertTrue(second["pass"])
            self.assertEqual(fixture.output.read_bytes(), selected_bytes)
            self.assertEqual(first["output"]["sha256"], second["output"]["sha256"])

    def test_quota_underfill_writes_audit_but_no_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            report = prepare_taskc_source(fixture.config(per_class=3))
            self.assertFalse(report["pass"])
            self.assertTrue(report["missing_quota_cells"])
            self.assertFalse(fixture.output.exists())
            self.assertTrue(fixture.report.is_file())

    def test_source_hash_mismatch_fails_before_streaming(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            report = prepare_taskc_source(fixture.config(expected_hash="0" * 64))
            self.assertFalse(report["pass"])
            self.assertIn("source hash mismatch", report["fatal_errors"][0])
            self.assertNotIn("source_rows", report["counts"])
            self.assertFalse(fixture.output.exists())

    def test_robust_clipped_view_guard_uses_exact_target_boxes(self) -> None:
        row = {"id": "qa", "frames": ["scene_1/a.png", "scene_1/b.png"]}
        entry = {
            "image_size": {"w": 640, "h": 480},
            "frames": row["frames"],
            "boxes": [
                {
                    "frame_index": 0,
                    "frame_path": row["frames"][0],
                    "object_id": "Target|1",
                    "bbox_xyxy": [0, 10, 20, 40],
                    "pixel_count": 1000,
                },
                {
                    "frame_index": 1,
                    "frame_path": row["frames"][1],
                    "object_id": "Target|1",
                    "bbox_xyxy": [100, 100, 120, 120],
                    "pixel_count": 100,
                },
            ],
        }
        with self.assertRaises(CandidateRejected) as raised:
            _apply_robust_guard(row, entry, "Target|1", "visible", 0, 1, 1000, 100)
        self.assertEqual(raised.exception.reason, "robust_clipped_view_guard")
        entry["boxes"][0]["pixel_count"] = 1500
        result = _apply_robust_guard(
            row, entry, "Target|1", "visible", 0, 1, 1500, 100
        )
        self.assertEqual(result["source"]["edges_touched"], 1)


if __name__ == "__main__":
    unittest.main()
