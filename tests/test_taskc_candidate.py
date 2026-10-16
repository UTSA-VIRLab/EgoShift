from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

RELEASE = Path(__file__).resolve().parents[1]
if str(RELEASE / "src") not in sys.path:
    sys.path.insert(0, str(RELEASE / "src"))

from egoshift.split_roles import canonical_exposure_manifest_sha256  # noqa: E402
from egoshift.taskc_candidate import (  # noqa: E402
    PUBLIC_FIELDS,
    TaskCCandidateError,
    TaskCSourceSpec,
    build_taskc_candidate,
)

from egoshift.taskc_source import CLASS_ORDER as CLASSES  # noqa: E402
from egoshift.taskc_source import CLASS_TO_ANSWER as ANSWER  # noqa: E402


class Fixture:
    def __init__(self, root: Path, per_class: int = 2) -> None:
        self.root = root
        self.assets = root / "assets"
        self.jsonl = root / "private_rows.jsonl"
        self.roles_path = root / "split_roles.json"
        self.config_path = root / "taskc.json"
        self.per_class = per_class
        self.scenes: dict[str, dict] = {}
        self.rows: list[dict] = []

    def _frame(self, name: str) -> str:
        relative = f"frames/{name}.png"
        path = self.assets / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + name.encode())
        return relative

    def add(self, index: int, role: str, answer_class: str, *, target: str | None = None,
            exposed: bool = False) -> None:
        scene = f"scene_{index:05d}"
        component = f"component_{index:05d}"
        self.scenes[scene] = {
            "leakage_component_id": component,
            "role": role,
            "drop_exact_alias_nonrepresentative": False,
            "outcome_exposed_component": exposed,
        }
        target_oid = target or f"Object|{index}"
        view1 = self._frame(f"{scene}_v1_{answer_class}")
        view2 = self._frame(f"{scene}_v2_{answer_class}")
        overlay_hash = hashlib.sha256((self.assets / view1).read_bytes()).hexdigest()
        self.rows.append(
            {
                "id": f"src_{index}_{answer_class}",
                "task_family": "C",
                "format": "mcq",
                "scene_id": scene,
                "frames": [view1, view2],
                "question": "In View 2, what best describes its visibility?",
                "options": list(ANSWER.values()),
                "answer": ANSWER[answer_class],
                "metadata": {"answer_class": answer_class, "trajectory_template": "t"},
                "target_oid": target_oid,
                "candidate_answer_class": answer_class,
                "candidate_role": role,
                "_candidate_source": {
                    "physical_component_id": component,
                    "role": role,
                    "stable_identity_sha256": hashlib.sha256(
                        f"{index}{answer_class}".encode()
                    ).hexdigest(),
                    "gold_evidence": {"evidence_type": "live_exact_mask_and_geometry_replay"},
                    "exact_mask_bbox_overlay": {
                        "evidence_type": "exact_instance_mask_replay",
                        "overlay_sha256": overlay_hash,
                    },
                },
            }
        )

    def populate(self) -> None:
        index = 1
        for role in ("hidden_test", "public_dev", "tuning_dev"):
            for answer_class in CLASSES:
                for _ in range(self.per_class):
                    self.add(index, role, answer_class)
                    index += 1
        self.index = index

    def finish(self) -> None:
        self.jsonl.write_text(
            "".join(json.dumps(row) + "\n" for row in self.rows), encoding="utf-8"
        )
        exposure = [{"path": "history.jsonl", "sha256": "0" * 64}]
        self.roles_path.write_text(
            json.dumps(
                {
                    "schema_version": "0.1.0",
                    "pass": True,
                    "exposure_files": exposure,
                    "exposure_ledger": {
                        "status": "complete_reviewed",
                        "complete": True,
                        "audit_sha256": "1" * 64,
                        "manifest_sha256": canonical_exposure_manifest_sha256(exposure),
                        "reviewed_manifest_count": 1,
                    },
                    "exposed_component_assigned_hidden_violations": [],
                    "scenes": self.scenes,
                }
            ),
            encoding="utf-8",
        )
        base = json.loads(
            (RELEASE / "configs" / "taskc_candidate_r2.json").read_text(encoding="utf-8")
        )
        for role in ("hidden_test", "public_dev", "tuning_dev"):
            base["splits"][role]["items_per_class"] = self.per_class
            base["splits"][role]["total_items"] = self.per_class * 3
        self.config_path.write_text(json.dumps(base), encoding="utf-8")

    def spec(self) -> TaskCSourceSpec:
        return TaskCSourceSpec(name="mined", jsonl_path=self.jsonl, asset_roots=(self.assets,))

    def build(self, suffix: str = ""):
        return build_taskc_candidate(
            config_path=self.config_path,
            split_roles_path=self.roles_path,
            source_specs=[self.spec()],
            output_root=self.root / f"candidate{suffix}",
            audit_report_path=self.root / f"audit{suffix}.json",
        )


class TaskCCandidateTests(unittest.TestCase):
    def _fixture(self, root: Path, per_class: int = 2) -> Fixture:
        fixture = Fixture(root, per_class=per_class)
        fixture.populate()
        fixture.finish()
        return fixture

    def test_public_rows_are_allowlisted_opaque_and_content_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            audit = fixture.build()
            self.assertTrue(audit["pass"])
            self.assertFalse(audit["release_ready"])
            questions = [
                json.loads(line)
                for line in (
                    root / "candidate" / "public" / "questions" / "task_c_hidden_test.jsonl"
                ).read_text().splitlines()
            ]
            self.assertEqual(len(questions), 6)
            for row in questions:
                self.assertEqual(set(row), set(PUBLIC_FIELDS))
                self.assertTrue(row["id"].startswith("ec2_"))
                for frame in row["frames"]:
                    self.assertTrue(frame.startswith("assets/task_c/"))
                    self.assertTrue(frame.endswith(".png"))
                    self.assertNotIn("scene_", frame)

    def test_hidden_gold_is_private_and_dev_answers_are_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            fixture.build()
            candidate = root / "candidate"
            self.assertTrue((candidate / "private" / "gold" / "task_c_hidden_test.jsonl").is_file())
            self.assertTrue(
                (candidate / "development" / "answers" / "task_c_public_dev.jsonl").is_file()
            )
            self.assertFalse(
                (candidate / "public" / "questions" / "task_c_hidden_test_gold.jsonl").exists()
            )
            gold = [
                json.loads(line)
                for line in (
                    candidate / "private" / "gold" / "task_c_hidden_test.jsonl"
                ).read_text().splitlines()
            ]
            questions = {
                json.loads(line)["id"]: json.loads(line)
                for line in (
                    candidate / "public" / "questions" / "task_c_hidden_test.jsonl"
                ).read_text().splitlines()
            }
            for row in gold:
                self.assertEqual(
                    questions[row["id"]]["options"][row["answer_index"]], row["answer"]
                )

    def test_build_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            first = fixture.build("_a")
            second = fixture.build("_b")
            self.assertEqual(
                first["output"]["manifest"]["sha256"],
                second["output"]["manifest"]["sha256"],
            )

    def test_quota_shortfall_blocks_without_writing_a_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root, per_class=2)
            fixture.populate()
            fixture.rows = [r for r in fixture.rows if r["candidate_answer_class"] != "occluded"]
            fixture.finish()
            with self.assertRaises(TaskCCandidateError) as ctx:
                fixture.build()
            self.assertIn("quotas are unmet", str(ctx.exception))
            self.assertIn("shortfalls", ctx.exception.diagnostics)
            self.assertFalse((root / "candidate").exists())

    def test_identity_cap_prunes_a_repeat_of_the_same_class_only(self) -> None:
        """One object may answer every class once, but never one class twice.

        The rule used to be the reverse -- one released row per
        (component, target) across the whole task -- and that is what made
        object identity predict the answer, because a target's single class
        became its signature. A repeat of the SAME class is still pseudo-
        replication and is still pruned.
        """

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root, per_class=1)
            fixture.populate()
            duplicate = dict(fixture.rows[0])
            component = duplicate["_candidate_source"]["physical_component_id"]
            scene = duplicate["scene_id"]
            answer_class = duplicate["candidate_answer_class"]
            fixture.scenes[scene]["role"] = "hidden_test"
            fixture.add(9001, "hidden_test", answer_class, target=duplicate["target_oid"])
            fixture.rows[-1]["scene_id"] = scene
            fixture.rows[-1]["_candidate_source"]["physical_component_id"] = component
            fixture.finish()
            report = fixture.build()
            # The build used to raise here. A repeat of the same class is
            # dropped, the quota is still met exactly, and -- the point of the
            # change -- a target carrying more than one class no longer
            # collapses the other classes to nothing.
            self.assertTrue(report["pass"])
            self.assertEqual(report["counts"]["items"], 9)
            for split, per_class in report["counts"]["items_by_split_and_class"].items():
                self.assertEqual(
                    sorted(per_class.values()), [1, 1, 1], f"{split} lost a class"
                )

    def test_outcome_exposed_component_cannot_enter_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = Fixture(root, per_class=1)
            fixture.populate()
            fixture.finish()
            roles = json.loads(fixture.roles_path.read_text())
            for scene, record in roles["scenes"].items():
                if record["role"] == "hidden_test":
                    record["outcome_exposed_component"] = True
                    break
            fixture.roles_path.write_text(json.dumps(roles), encoding="utf-8")
            with self.assertRaisesRegex(TaskCCandidateError, "outcome-exposed"):
                fixture.build()


if __name__ == "__main__":
    unittest.main()
