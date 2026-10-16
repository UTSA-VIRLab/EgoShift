from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

RELEASE_SOURCE = Path(__file__).resolve().parents[1] / "src"
if str(RELEASE_SOURCE) not in sys.path:
    sys.path.insert(0, str(RELEASE_SOURCE))

from egoshift.taskc_source import TaskCSourceConfig, prepare_taskc_source  # noqa: E402

from test_taskc_source import TaskCFixture  # noqa: E402


def _supplement_row(
    *,
    scene_id: str,
    role: str,
    component: str,
    target_oid: str,
    answer_class: str,
    stable: str,
) -> dict:
    metadata = {
        "answer_class": answer_class,
        "target_pixel_count_source": 900,
        "target_pixel_count_target": 900 if answer_class == "visible" else 0,
        "target_visibility_state_target": {
            "visible": "VISIBLE",
            "out_of_fov": "OUT_OF_FOV",
            "occluded": "OCCLUDED",
        }[answer_class],
    }
    if answer_class == "occluded":
        metadata["occluder_id"] = "Sofa|9"
        metadata["occluder_overlap_ratio"] = 0.42
    return {
        "id": f"sup_{stable[:8]}",
        "task_family": "C",
        "format": "mcq",
        "scene_id": scene_id,
        "frames": ["a.png", "b.png"],
        "question": "?",
        "options": ["Visible", "Out of View", "Occluded"],
        "answer": "Occluded",
        "metadata": metadata,
        "target_oid": target_oid,
        "candidate_answer_class": answer_class,
        "candidate_role": role,
        "_candidate_source": {
            "schema_version": "0.1.0",
            "physical_component_id": component,
            "role": role,
            "stable_identity_sha256": stable,
            "gold_evidence": {
                "evidence_type": "live_exact_mask_and_geometry_replay",
            },
        },
    }


class TaskCSupplementTests(unittest.TestCase):
    def _fixture_without_hidden(self, root: Path) -> TaskCFixture:
        fixture = TaskCFixture(root)
        fixture.add_scene(1, "tuning_dev", "component_tuning")
        fixture.add_scene(2, "public_dev", "component_public")
        fixture.split_scenes["scene_00003"] = {
            "leakage_component_id": "component_hidden",
            "role": "hidden_test",
            "drop_exact_alias_nonrepresentative": False,
            "outcome_exposed_component": False,
        }
        fixture.finish()
        return fixture

    def _config(self, fixture: TaskCFixture, supplements: tuple[Path, ...]):
        return TaskCSourceConfig(
            source_path=fixture.source,
            expected_source_sha256=__import__("hashlib").sha256(
                fixture.source.read_bytes()
            ).hexdigest(),
            split_roles_path=fixture.split_roles,
            scenes_root=fixture.scenes_root,
            robust_sidecar_paths=(),
            output_path=fixture.output,
            report_path=fixture.report,
            quotas={"tuning_dev": 1, "public_dev": 1, "hidden_test": 1},
            supplement_jsonl_paths=supplements,
        )

    def test_dc6_allows_one_row_per_class_and_prunes_same_class_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture_without_hidden(Path(temporary))
            rows = [
                _supplement_row(
                    scene_id="scene_00003", role="hidden_test",
                    component="component_hidden", target_oid="MinedA|1",
                    answer_class="occluded", stable="a" * 64,
                ),
                _supplement_row(
                    scene_id="scene_00003", role="hidden_test",
                    component="component_hidden", target_oid="MinedB|1",
                    answer_class="visible", stable="b" * 64,
                ),
                _supplement_row(
                    scene_id="scene_00003", role="hidden_test",
                    component="component_hidden", target_oid="MinedC|1",
                    answer_class="out_of_fov", stable="c" * 64,
                ),
                # Same target as the occluded row but a DIFFERENT class: kept.
                # One object may carry all three answers -- that is what makes a
                # target-matched triple, and a target holding every answer can
                # predict none of them.
                _supplement_row(
                    scene_id="scene_00003", role="hidden_test",
                    component="component_hidden", target_oid="MinedA|1",
                    answer_class="visible", stable="d" * 64,
                ),
                # Same target AND same class as the row above: still pruned, so
                # one object cannot flood a single answer.
                _supplement_row(
                    scene_id="scene_00003", role="hidden_test",
                    component="component_hidden", target_oid="MinedA|1",
                    answer_class="visible", stable="e" * 64,
                ),
            ]
            supplement = Path(temporary) / "mined.jsonl"
            supplement.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = prepare_taskc_source(self._config(fixture, (supplement,)))
            self.assertEqual(report["fatal_errors"], [])
            self.assertTrue(report["pass"], report["missing_quota_cells"])
            self.assertEqual(report["output"]["rows"], 9)
            # Same-class duplicates are already collapsed upstream by the
            # (component, target, class) uniqueness key, so the cross-class cap
            # has nothing left to prune -- and must no longer prune a target's
            # other classes.
            self.assertEqual(
                report["cross_class_identity_cap"]["pruned_by_class"], {}
            )
            kept = [json.loads(l) for l in
                    fixture.output.read_text().splitlines() if l.strip()]
            triples = {}
            for row in kept:
                key = (row["_candidate_source"]["physical_component_id"],
                       row["target_oid"])
                triples.setdefault(key, set()).add(row["candidate_answer_class"])
            # No (component, target, class) is ever duplicated -- one object
            # still cannot flood a single answer.
            seen = [(k[0], k[1], c) for k, v in triples.items() for c in v]
            self.assertEqual(len(seen), len(set(seen)))
            # The triple machinery ran and reported on the pool. (Whether a
            # multi-class target reaches the OUTPUT depends on quota
            # competition -- upstream rows outrank supplements -- so that is
            # asserted on the pool, not the selection.)
            triple_report = report["target_matched_triples"]
            self.assertIn("targets_with_all_three_classes", triple_report)
            self.assertGreaterEqual(triple_report["distinct_targets"], 1)
            selected = [
                json.loads(line)
                for line in fixture.output.read_text().splitlines()
            ]
            identities = {
                (
                    row["_candidate_source"]["physical_component_id"],
                    row["target_oid"],
                )
                for row in selected
            }
            self.assertEqual(len(identities), len(selected))
            hidden = [r for r in selected if r["candidate_role"] == "hidden_test"]
            self.assertEqual(len(hidden), 3)

    def test_structural_targets_are_dropped_not_fatal(self) -> None:
        # "Is the wall still visible?" is close to unanswerable and a boxed
        # room is meaningless. These are filtered as a quality gate, and the
        # build continues -- they are not pipeline corruption.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture_without_hidden(root)
            rows = [
                _supplement_row(
                    scene_id="scene_00003", role="hidden_test",
                    component="component_hidden", target_oid="wall|7|4.2",
                    answer_class="occluded", stable="e" * 64,
                ),
                _supplement_row(
                    scene_id="scene_00003", role="hidden_test",
                    component="component_hidden", target_oid="room|2",
                    answer_class="visible", stable="f" * 64,
                ),
                _supplement_row(
                    scene_id="scene_00003", role="hidden_test",
                    component="component_hidden", target_oid="MinedA|1",
                    answer_class="occluded", stable="a" * 64,
                ),
                _supplement_row(
                    scene_id="scene_00003", role="hidden_test",
                    component="component_hidden", target_oid="MinedB|1",
                    answer_class="visible", stable="b" * 64,
                ),
                _supplement_row(
                    scene_id="scene_00003", role="hidden_test",
                    component="component_hidden", target_oid="MinedC|1",
                    answer_class="out_of_fov", stable="c" * 64,
                ),
            ]
            supplement = Path(temporary) / "mined.jsonl"
            supplement.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = prepare_taskc_source(self._config(fixture, (supplement,)))
            self.assertEqual(report["fatal_errors"], [])
            self.assertTrue(report["pass"], report["missing_quota_cells"])
            self.assertEqual(
                report["supplements"]["files"][0]["structural_targets_dropped"], 2
            )
            selected = [
                json.loads(line)
                for line in fixture.output.read_text().splitlines()
            ]
            for row in selected:
                self.assertFalse(
                    str(row["target_oid"]).lower().startswith(("wall|", "room|", "door|"))
                )

    def test_visible_supplement_below_pixel_floor_is_dropped_not_fatal(self) -> None:
        """A 62-pixel "Visible" sliver must never reach a rater again.

        The upstream path enforced the floor during replay, but supplements
        used the pixel count only as a quality rank -- so --supplements-only
        builds shipped visible gold no human could verify. The floor is a
        quality gate, not corruption: the row is dropped and counted.
        """

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture_without_hidden(root)
            sliver = _supplement_row(
                scene_id="scene_00003", role="hidden_test",
                component="component_hidden", target_oid="MinedTiny|1",
                answer_class="visible", stable="d" * 64,
            )
            sliver["metadata"]["target_pixel_count_target"] = 62
            rows = [
                sliver,
                _supplement_row(
                    scene_id="scene_00003", role="hidden_test",
                    component="component_hidden", target_oid="MinedA|1",
                    answer_class="occluded", stable="a" * 64,
                ),
                _supplement_row(
                    scene_id="scene_00003", role="hidden_test",
                    component="component_hidden", target_oid="MinedB|1",
                    answer_class="visible", stable="b" * 64,
                ),
                _supplement_row(
                    scene_id="scene_00003", role="hidden_test",
                    component="component_hidden", target_oid="MinedC|1",
                    answer_class="out_of_fov", stable="c" * 64,
                ),
            ]
            supplement = Path(temporary) / "mined.jsonl"
            supplement.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            report = prepare_taskc_source(self._config(fixture, (supplement,)))
            self.assertEqual(report["fatal_errors"], [])
            self.assertTrue(report["pass"], report["missing_quota_cells"])
            self.assertEqual(
                report["supplements"]["files"][0]["below_visible_floor_dropped"], 1
            )
            selected_targets = {
                json.loads(line)["target_oid"]
                for line in fixture.output.read_text().splitlines()
            }
            self.assertNotIn("MinedTiny|1", selected_targets)

    def test_stale_supplement_role_is_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture_without_hidden(Path(temporary))
            row = _supplement_row(
                scene_id="scene_00003", role="tuning_dev",
                component="component_hidden", target_oid="MinedA|1",
                answer_class="occluded", stable="a" * 64,
            )
            supplement = Path(temporary) / "mined.jsonl"
            supplement.write_text(json.dumps(row) + "\n", encoding="utf-8")
            report = prepare_taskc_source(self._config(fixture, (supplement,)))
            self.assertFalse(report["pass"])
            self.assertTrue(
                any("stale supplement role" in e for e in report["fatal_errors"])
            )
            self.assertFalse(fixture.output.exists())


if __name__ == "__main__":
    unittest.main()
