from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from egoshift.exact_mask_bbox import (
    COORDINATE_CONVENTION,
    EXACT_BBOX_SOURCE,
    EXACT_EVIDENCE_TYPE,
    ExactMaskBBoxError,
    bake_exact_mask_bbox_overlays,
    load_exact_mask_replay_sidecar,
    validate_exact_mask_bbox_entry,
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _valid_entry(frame_sha256: str = "a" * 64) -> dict:
    return {
        "evidence_type": EXACT_EVIDENCE_TYPE,
        "bbox_source": EXACT_BBOX_SOURCE,
        "coordinate_convention": COORDINATE_CONVENTION,
        "object_id": "Mug|1",
        "frame_path": "scene_00001/trajectories/traj/frames/frame_00.png",
        "frame_index": 0,
        "image_size": {"w": 20, "h": 20},
        "bbox_xyxy": [3, 4, 12, 15],
        "pixel_count": 50,
        "frame_sha256": frame_sha256,
        "mask_sha256": "b" * 64,
    }


class ExactMaskBBoxTests(unittest.TestCase):
    def test_validator_accepts_only_exact_mask_provenance(self) -> None:
        normalized = validate_exact_mask_bbox_entry(
            _valid_entry(),
            expected_object_id="Mug|1",
            expected_frame_path="scene_00001/trajectories/traj/frames/frame_00.png",
            expected_frame_index=0,
        )
        self.assertEqual(normalized["bbox_xyxy"], [3, 4, 12, 15])

        approximate = _valid_entry()
        approximate["bbox_source"] = "projected_center_estimate"
        with self.assertRaises(ExactMaskBBoxError):
            validate_exact_mask_bbox_entry(
                approximate,
                expected_object_id="Mug|1",
                expected_frame_path=approximate["frame_path"],
                expected_frame_index=0,
            )

        missing_mask_hash = _valid_entry()
        missing_mask_hash.pop("mask_sha256")
        with self.assertRaises(ExactMaskBBoxError):
            validate_exact_mask_bbox_entry(
                missing_mask_hash,
                expected_object_id="Mug|1",
                expected_frame_path=missing_mask_hash["frame_path"],
                expected_frame_index=0,
            )

        outside = _valid_entry()
        outside["bbox_xyxy"] = [3, 4, 21, 15]
        with self.assertRaises(ExactMaskBBoxError):
            validate_exact_mask_bbox_entry(
                outside,
                expected_object_id="Mug|1",
                expected_frame_path=outside["frame_path"],
                expected_frame_index=0,
            )

    def test_sidecar_envelope_rejects_non_exact_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sidecar.json"
            path.write_text(
                json.dumps(
                    {
                        "schema_version": "0.1.0",
                        "evidence_type": "projected_center_bbox",
                        "coordinate_convention": COORDINATE_CONVENTION,
                        "items": {},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ExactMaskBBoxError):
                load_exact_mask_replay_sidecar(path)

    def test_baker_preserves_raw_frame_and_draws_replayed_bbox(self) -> None:
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow not installed in the development environment")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_root = root / "assets"
            relative = "scene_00001/trajectories/traj/frames/frame_00.png"
            target_relative = "scene_00001/trajectories/traj/frames/frame_01.png"
            raw = asset_root / relative
            raw.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 20), (10, 20, 30)).save(raw, format="PNG")
            Image.new("RGB", (20, 20), (40, 50, 60)).save(
                asset_root / target_relative, format="PNG"
            )
            before = raw.read_bytes()

            input_path = root / "private.jsonl"
            row = {
                "id": "qa_fixture",
                "task_family": "C",
                "format": "mcq",
                "frames": [relative, target_relative],
                "question": "legacy",
                "options": [],
                "answer": "legacy",
                "metadata": {"answer_class": "visible"},
                "target_oid": "Mug|1",
                "source_step": 0,
                "target_step": 1,
                "_candidate_source": {"private": True},
            }
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            entry = _valid_entry(_digest(before))
            sidecar = root / "replay.json"
            sidecar.write_text(
                json.dumps(
                    {
                        "schema_version": "0.1.0",
                        "evidence_type": EXACT_EVIDENCE_TYPE,
                        "coordinate_convention": COORDINATE_CONVENTION,
                        "items": {"qa_fixture": entry},
                    }
                ),
                encoding="utf-8",
            )
            output_assets = root / "overlays"
            output_jsonl = root / "boxed.jsonl"
            report_path = root / "report.json"
            report = bake_exact_mask_bbox_overlays(
                input_jsonl=input_path,
                replay_sidecar=sidecar,
                asset_root=asset_root,
                output_asset_root=output_assets,
                output_jsonl=output_jsonl,
                report_path=report_path,
            )
            self.assertTrue(report["pass"], report["errors"])
            self.assertEqual(raw.read_bytes(), before)
            output_row = json.loads(output_jsonl.read_text(encoding="utf-8"))
            overlay = output_assets / output_row["frames"][0]
            self.assertTrue(overlay.is_file())
            self.assertNotEqual(overlay.read_bytes(), before)
            with Image.open(overlay) as image:
                self.assertEqual(image.getpixel((3, 4)), (220, 30, 30))
                self.assertEqual(image.getpixel((7, 9)), (10, 20, 30))
            self.assertIn("highlighted with a box", output_row["question"])
            self.assertEqual(
                output_row["_candidate_source"]["exact_mask_bbox_overlay"]["mask_sha256"],
                "b" * 64,
            )


if __name__ == "__main__":
    unittest.main()
