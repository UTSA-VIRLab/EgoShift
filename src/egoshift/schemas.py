"""Dependency-free validation for the public JSON schemas."""

from __future__ import annotations

import re
from typing import Any

from .hashing import safe_relative_path


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SUPPORTED_SCHEMA_VERSION = "0.1.0"


class ValidationError(ValueError):
    """Raised when an artifact record violates its schema contract."""


def _require(mapping: dict[str, Any], fields: set[str], context: str) -> None:
    missing = sorted(fields - mapping.keys())
    if missing:
        raise ValidationError(f"{context}: missing fields: {', '.join(missing)}")


def _sha256(value: Any, context: str) -> None:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise ValidationError(f"{context}: expected lowercase SHA-256 hex")


def validate_manifest(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError("manifest: expected object")
    _require(
        value,
        {"schema_version", "manifest_id", "dataset_version", "status", "files", "tasks", "provenance"},
        "manifest",
    )
    if value["schema_version"] != SUPPORTED_SCHEMA_VERSION:
        raise ValidationError(
            f"manifest.schema_version: expected {SUPPORTED_SCHEMA_VERSION!r}, got {value['schema_version']!r}"
        )
    if not isinstance(value["files"], list) or not value["files"]:
        raise ValidationError("manifest.files: expected non-empty array")
    seen_paths: set[str] = set()
    for index, entry in enumerate(value["files"]):
        context = f"manifest.files[{index}]"
        if not isinstance(entry, dict):
            raise ValidationError(f"{context}: expected object")
        _require(entry, {"path", "sha256", "media_type", "records"}, context)
        try:
            path = safe_relative_path(entry["path"])
        except ValueError as exc:
            raise ValidationError(f"{context}.path: {exc}") from exc
        if path in seen_paths:
            raise ValidationError(f"{context}.path: duplicate {path!r}")
        seen_paths.add(path)
        _sha256(entry["sha256"], f"{context}.sha256")
        if not isinstance(entry["records"], int) or entry["records"] < 0:
            raise ValidationError(f"{context}.records: expected non-negative integer")
    if not isinstance(value["tasks"], dict) or not value["tasks"]:
        raise ValidationError("manifest.tasks: expected non-empty object")
    for task, entry in value["tasks"].items():
        context = f"manifest.tasks.{task}"
        if not isinstance(entry, dict):
            raise ValidationError(f"{context}: expected object")
        _require(entry, {"role", "items", "annotation_path"}, context)
        try:
            annotation_path = safe_relative_path(entry["annotation_path"])
        except ValueError as exc:
            raise ValidationError(f"{context}.annotation_path: {exc}") from exc
        if annotation_path not in seen_paths:
            raise ValidationError(f"{context}.annotation_path: not declared in manifest.files")
        if not isinstance(entry["items"], int) or entry["items"] < 0:
            raise ValidationError(f"{context}.items: expected non-negative integer")
    if not isinstance(value["provenance"], dict):
        raise ValidationError("manifest.provenance: expected object")
    return value


def validate_prediction(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError("prediction: expected object")
    _require(value, {"id", "prediction"}, "prediction")
    if not isinstance(value["id"], str) or not value["id"]:
        raise ValidationError("prediction.id: expected non-empty string")
    if value["prediction"] is not None and not isinstance(value["prediction"], str):
        raise ValidationError("prediction.prediction: expected string or null")
    status = value.get("status", "ok")
    if status not in {"ok", "refusal", "error", "missing"}:
        raise ValidationError(f"prediction.status: unsupported value {status!r}")
    return value


def validate_result_record(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError("result: expected object")
    _require(
        value,
        {
            "schema_version",
            "result_id",
            "dataset",
            "predictions",
            "model",
            "protocol",
            "scorer",
            "statistics",
            "metrics",
            "consumers",
        },
        "result",
    )
    if value["schema_version"] != SUPPORTED_SCHEMA_VERSION:
        raise ValidationError(
            f"result.schema_version: expected {SUPPORTED_SCHEMA_VERSION!r}, got {value['schema_version']!r}"
        )
    for section, key in (("dataset", "manifest_sha256"), ("predictions", "sha256"), ("protocol", "sha256"), ("scorer", "sha256"), ("statistics", "sha256")):
        if not isinstance(value[section], dict):
            raise ValidationError(f"result.{section}: expected object")
        _sha256(value[section].get(key), f"result.{section}.{key}")
    _sha256(value["result_id"], "result.result_id")
    if not isinstance(value["metrics"], dict):
        raise ValidationError("result.metrics: expected object")
    if not isinstance(value["consumers"], list):
        raise ValidationError("result.consumers: expected array")
    return value
