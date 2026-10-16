"""Deterministic scorer with explicit parse and coverage accounting."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

from .hashing import sha256_file
from .records import read_jsonl
from .schemas import ValidationError, validate_prediction


_SPACE = re.compile(r"\s+")
_LETTER = re.compile(r"^(?:option\s*)?([A-Z])(?:[\).:]?)$", re.IGNORECASE)


def normalize_answer(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip().casefold()
    return _SPACE.sub(" ", value)


def parse_choice(prediction: str | None, options: list[str]) -> str | None:
    if prediction is None:
        return None
    text = prediction.strip()
    if not text:
        return None
    letter = _LETTER.fullmatch(text)
    if letter:
        index = ord(letter.group(1).upper()) - ord("A")
        return options[index] if 0 <= index < len(options) else None
    normalized = normalize_answer(text)
    for option in options:
        if normalized == normalize_answer(option):
            return option
    return None


def _index_unique(rows: list[dict[str, Any]], context: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        item_id = row.get("id")
        if not isinstance(item_id, str) or not item_id:
            raise ValidationError(f"{context} row {row_number}: missing non-empty id")
        if item_id in result:
            raise ValidationError(f"{context}: duplicate id {item_id!r}")
        result[item_id] = row
    return result


def score_rows(
    questions: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    for prediction in predictions:
        validate_prediction(prediction)
    question_by_id = _index_unique(questions, "questions")
    prediction_by_id = _index_unique(predictions, "predictions")
    extra_ids = sorted(prediction_by_id.keys() - question_by_id.keys())
    missing_ids = sorted(question_by_id.keys() - prediction_by_id.keys())

    overall = defaultdict(int)
    by_task: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    item_results: list[dict[str, Any]] = []

    for item_id, question in question_by_id.items():
        task = str(question.get("task_family", "unknown"))
        options = question.get("options")
        answer = question.get("answer")
        if not isinstance(options, list) or not options or not all(isinstance(x, str) for x in options):
            raise ValidationError(f"question {item_id!r}: options must be a non-empty string array")
        if not isinstance(answer, str) or answer not in options:
            raise ValidationError(f"question {item_id!r}: answer must appear exactly in options")
        overall["items"] += 1
        by_task[task]["items"] += 1
        prediction = prediction_by_id.get(item_id)
        parsed: str | None = None
        status = "missing"
        if prediction is None:
            overall["missing"] += 1
            by_task[task]["missing"] += 1
        elif prediction.get("status", "ok") != "ok":
            status = str(prediction.get("status"))
            overall[status] += 1
            by_task[task][status] += 1
        else:
            parsed = parse_choice(prediction.get("prediction"), options)
            if parsed is None:
                status = "parse_failure"
                overall["parse_failures"] += 1
                by_task[task]["parse_failures"] += 1
            else:
                status = "correct" if normalize_answer(parsed) == normalize_answer(answer) else "incorrect"
                overall["valid_predictions"] += 1
                by_task[task]["valid_predictions"] += 1
                overall[status] += 1
                by_task[task][status] += 1
        item_results.append(
            {
                "id": item_id,
                "task_family": task,
                "gold": answer,
                "parsed_prediction": parsed,
                "status": status,
            }
        )

    overall["incorrect_total"] = overall["items"] - overall["correct"]
    overall["extra_predictions"] = len(extra_ids)
    overall_metrics: dict[str, Any] = dict(overall)
    overall_metrics["accuracy"] = overall["correct"] / overall["items"] if overall["items"] else None

    task_metrics: dict[str, Any] = {}
    for task in sorted(by_task):
        values = dict(by_task[task])
        values["incorrect_total"] = values["items"] - values.get("correct", 0)
        values["accuracy"] = values.get("correct", 0) / values["items"] if values["items"] else None
        task_metrics[task] = values

    return {
        "schema_version": "0.1.0",
        "overall": overall_metrics,
        "tasks": task_metrics,
        "missing_ids": missing_ids,
        "extra_ids": extra_ids,
        "items": item_results,
    }


def score_files(question_path: str | Path, prediction_path: str | Path) -> dict[str, Any]:
    report = score_rows(read_jsonl(question_path), read_jsonl(prediction_path))
    report["inputs"] = {
        "questions_sha256": sha256_file(question_path),
        "predictions_sha256": sha256_file(prediction_path),
    }
    return report

