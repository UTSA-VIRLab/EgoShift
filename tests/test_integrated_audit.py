from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
for extra in (RELEASE / "src", Path(__file__).resolve().parent):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from egoshift.integrated_audit import audit_integrated_candidate  # noqa: E402

from test_taskc_candidate import Fixture  # noqa: E402


class IntegratedAuditTests(unittest.TestCase):
    """The gate is exercised on a real materialized Task-C candidate.

    Task B is supplied as None here; its finite-attainment branch is covered by
    the Task-B contract tests. What matters at this boundary is that the audit
    re-derives properties from artifacts on disk and fails closed on tampering.
    """

    def _built(self, root: Path) -> Fixture:
        fixture = Fixture(root, per_class=2)
        fixture.populate()
        # Materialize the v10 contract used by the real candidate: each target
        # appears once in every visibility class, with the same View 1 and a
        # distinct View 2.  The generic Task-C builder tests use singletons,
        # whereas this integrated-gate fixture must represent v10 itself.
        for role in ("hidden_test", "public_dev", "tuning_dev"):
            by_class = {}
            for row in fixture.rows:
                if row["candidate_role"] == role:
                    by_class.setdefault(row["candidate_answer_class"], []).append(row)
            ordered_classes = sorted(by_class)
            for slot in range(2):
                group = [by_class[answer_class][slot] for answer_class in ordered_classes]
                anchor = group[0]
                anchor_frame = anchor["frames"][0]
                anchor_overlay = anchor["_candidate_source"]["exact_mask_bbox_overlay"][
                    "overlay_sha256"
                ]
                for row in group[1:]:
                    row["scene_id"] = anchor["scene_id"]
                    row["target_oid"] = anchor["target_oid"]
                    row["frames"][0] = anchor_frame
                    row["_candidate_source"]["physical_component_id"] = anchor[
                        "_candidate_source"
                    ]["physical_component_id"]
                    row["_candidate_source"]["exact_mask_bbox_overlay"][
                        "overlay_sha256"
                    ] = anchor_overlay
        fixture.finish()
        fixture.build()
        return fixture

    def _audit(self, fixture: Fixture, root: Path, suffix: str = ""):
        # Task B is absent at this point in P1, which the gate must support:
        # the Task-C candidate is audited on its own.
        return audit_integrated_candidate(
            taskb_root=None,
            taskc_root=root / "candidate",
            split_roles_path=fixture.roles_path,
            taskb_config_path=None,
            report_path=root / f"integrated{suffix}.json",
        )

    def test_tampered_public_asset_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._built(root)
            assets = sorted((root / "candidate" / "assets").rglob("*.png"))
            self.assertTrue(assets)
            assets[0].write_bytes(b"\x89PNG\r\n\x1a\ntampered")
            report = self._audit(fixture, root)
            self.assertFalse(report["pass"])
            codes = {failure["check"] for failure in report["failures"]}
            self.assertTrue(
                {"artifact_hash_chain_broken", "asset_not_content_addressed"} & codes
            )

    def test_clean_candidate_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._built(root)
            report = self._audit(fixture, root)
            self.assertTrue(report["pass"], report["failures"])

    def test_same_target_may_carry_different_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._built(root)
            provenance = (
                root / "candidate" / "private" / "provenance" / "task_c_hidden_test.jsonl"
            )
            rows = [json.loads(line) for line in provenance.read_text().splitlines()]
            different_class = next(
                i for i in range(1, len(rows))
                if rows[i]["answer_class"] != rows[0]["answer_class"]
            )
            rows[different_class]["physical_scene_component_id"] = rows[0]["physical_scene_component_id"]
            rows[different_class]["target_oid"] = rows[0]["target_oid"]
            rows[different_class]["scene_id"] = rows[0]["scene_id"]
            # A matched identity must reuse the same View 1, but its View 2 is
            # outcome-specific.  Modify the public row and refresh the manifest
            # hash so this test isolates the semantic contract.
            questions = (
                root / "candidate" / "public" / "questions" / "task_c_hidden_test.jsonl"
            )
            qrows = [json.loads(line) for line in questions.read_text().splitlines()]
            qrows[different_class]["frames"][0] = qrows[0]["frames"][0]
            provenance.write_text("".join(json.dumps(row) + "\n" for row in rows))
            questions.write_text("".join(json.dumps(row) + "\n" for row in qrows))
            self._refresh_manifest_hash(root / "candidate", provenance)
            self._refresh_manifest_hash(root / "candidate", questions)
            report = self._audit(fixture, root)
            codes = {failure["check"] for failure in report["failures"]}
            self.assertNotIn("taskc_identity_class_repeated", codes)
            self.assertNotIn("taskc_matched_identity_view1_disagreement", codes)

    def test_same_target_cannot_repeat_a_class(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._built(root)
            provenance = (
                root / "candidate" / "private" / "provenance" / "task_c_hidden_test.jsonl"
            )
            rows = [json.loads(line) for line in provenance.read_text().splitlines()]
            same_class = next(i for i in range(1, len(rows)) if rows[i]["answer_class"] == rows[0]["answer_class"])
            rows[same_class]["physical_scene_component_id"] = rows[0]["physical_scene_component_id"]
            rows[same_class]["target_oid"] = rows[0]["target_oid"]
            provenance.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self._refresh_manifest_hash(root / "candidate", provenance)
            report = self._audit(fixture, root)
            self.assertIn(
                "taskc_identity_class_repeated",
                {failure["check"] for failure in report["failures"]},
            )

    def test_gold_leaked_into_public_tree_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._built(root)
            leak = root / "candidate" / "public" / "questions" / "task_c_hidden_test_gold.jsonl"
            leak.write_text('{"id":"x","answer":"y"}\n', encoding="utf-8")
            report = self._audit(fixture, root)
            self.assertFalse(report["pass"])
            self.assertIn(
                "gold_leaked_into_public_tree",
                {failure["check"] for failure in report["failures"]},
            )

    def test_release_ready_true_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._built(root)
            manifest_path = root / "candidate" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["release_ready"] = True
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            report = self._audit(fixture, root)
            self.assertFalse(report["pass"])
            self.assertIn(
                "release_ready_must_be_false",
                {failure["check"] for failure in report["failures"]},
            )

    def test_exposed_component_in_hidden_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._built(root)
            roles = json.loads(fixture.roles_path.read_text(encoding="utf-8"))
            for record in roles["scenes"].values():
                if record["role"] == "hidden_test":
                    record["outcome_exposed_component"] = True
            fixture.roles_path.write_text(json.dumps(roles), encoding="utf-8")
            report = self._audit(fixture, root)
            self.assertFalse(report["pass"])
            codes = {failure["check"] for failure in report["failures"]}
            # Rewriting the ledger after a build breaks the recorded hash chain;
            # either signal is a correct fail-closed detection of tampering.
            self.assertTrue(
                {"outcome_exposed_component_in_hidden", "split_roles_hash_mismatch"} & codes
            )

    def test_report_is_always_written_even_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._built(root)
            (root / "candidate" / "manifest.json").write_text(
                json.dumps({"release_ready": True, "artifacts": []}), encoding="utf-8"
            )
            self._audit(fixture, root)
            self.assertTrue((root / "integrated.json").is_file())

    @staticmethod
    def _refresh_manifest_hash(candidate: Path, changed: Path) -> None:
        import hashlib

        manifest_path = candidate / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        relative = str(changed.relative_to(candidate))
        for record in manifest["artifacts"]:
            if record["path"] == relative:
                record["sha256"] = hashlib.sha256(changed.read_bytes()).hexdigest()
                break
        else:
            raise AssertionError(f"artifact absent from manifest: {relative}")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
