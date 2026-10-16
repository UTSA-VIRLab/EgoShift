from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from egoshift.cli import run_smoke
from egoshift.hashing import resolve_within, safe_relative_path, sha256_json
from egoshift.registry import build_result_record
from egoshift.schemas import ValidationError, validate_manifest, validate_prediction
from egoshift.scoring import parse_choice, score_rows


ZERO_SHA = "0" * 64


class HashingTests(unittest.TestCase):
    def test_safe_relative_path(self) -> None:
        self.assertEqual(safe_relative_path("sample/questions.jsonl"), "sample/questions.jsonl")

    def test_reject_absolute_and_traversal(self) -> None:
        for value in ("/tmp/file", "../secret", "C:\\secret", "a/../secret"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                safe_relative_path(value)

    def test_resolve_within(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            expected = Path(directory).resolve() / "a" / "b.json"
            self.assertEqual(resolve_within(directory, "a/b.json"), expected)

    def test_canonical_json_hash_ignores_key_order(self) -> None:
        self.assertEqual(sha256_json({"a": 1, "b": 2}), sha256_json({"b": 2, "a": 1}))


class SchemaTests(unittest.TestCase):
    def test_reject_bad_prediction_status(self) -> None:
        with self.assertRaises(ValidationError):
            validate_prediction({"id": "x", "prediction": "A", "status": "cached"})

    def test_manifest_rejects_undeclared_annotation(self) -> None:
        manifest = {
            "schema_version": "0.1.0",
            "manifest_id": "x",
            "dataset_version": "x",
            "status": "candidate",
            "files": [{"path": "a.jsonl", "sha256": ZERO_SHA, "media_type": "application/x-ndjson", "records": 1}],
            "tasks": {"B": {"role": "primary", "items": 1, "annotation_path": "b.jsonl"}},
            "provenance": {},
        }
        with self.assertRaises(ValidationError):
            validate_manifest(manifest)

    def test_manifest_rejects_unknown_schema_version(self) -> None:
        manifest = {
            "schema_version": "9.9.9",
            "manifest_id": "x",
            "dataset_version": "x",
            "status": "candidate",
            "files": [{"path": "a.jsonl", "sha256": ZERO_SHA, "media_type": "application/x-ndjson", "records": 1}],
            "tasks": {"B": {"role": "primary", "items": 1, "annotation_path": "a.jsonl"}},
            "provenance": {},
        }
        with self.assertRaises(ValidationError):
            validate_manifest(manifest)


class ScoringTests(unittest.TestCase):
    def setUp(self) -> None:
        self.questions = [
            {"id": "q1", "task_family": "B", "options": ["left", "right"], "answer": "right"},
            {"id": "q2", "task_family": "C", "options": ["visible", "occluded"], "answer": "visible"},
        ]

    def test_letter_and_text_parsing(self) -> None:
        self.assertEqual(parse_choice("B", ["left", "right"]), "right")
        self.assertEqual(parse_choice("  RIGHT ", ["left", "right"]), "right")
        self.assertIsNone(parse_choice("unknown", ["left", "right"]))

    def test_score_accounts_for_missing_and_extra(self) -> None:
        report = score_rows(self.questions, [{"id": "q1", "prediction": "B"}, {"id": "extra", "prediction": "A"}])
        self.assertEqual(report["overall"]["items"], 2)
        self.assertEqual(report["overall"]["correct"], 1)
        self.assertEqual(report["overall"]["missing"], 1)
        self.assertEqual(report["overall"]["extra_predictions"], 1)
        self.assertEqual(report["overall"]["accuracy"], 0.5)

    def test_duplicate_prediction_fails(self) -> None:
        with self.assertRaises(ValidationError):
            score_rows(self.questions, [{"id": "q1", "prediction": "A"}, {"id": "q1", "prediction": "B"}])


class RegistryTests(unittest.TestCase):
    def test_result_record_is_content_addressed(self) -> None:
        record = build_result_record(
            metrics={"overall": {"accuracy": 0.5}},
            manifest_sha256=ZERO_SHA,
            predictions_sha256="1" * 64,
            model={"id": "dummy", "revision": "none"},
            protocol={"id": "smoke", "sha256": "2" * 64},
            scorer={"id": "egoshift", "sha256": "3" * 64},
            statistics={"id": "descriptive", "sha256": "4" * 64},
            consumers=["test"],
        )
        self.assertEqual(len(record["result_id"]), 64)


class SmokeTests(unittest.TestCase):
    def test_bundled_fixture(self) -> None:
        root = Path(__file__).resolve().parents[1] / "sample"
        result = run_smoke(root)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["metrics"]["overall"]["correct"], 4)


if __name__ == "__main__":
    unittest.main()
