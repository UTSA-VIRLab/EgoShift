"""Result-record construction with complete lineage fields."""

from __future__ import annotations

from typing import Any

from .hashing import sha256_json
from .schemas import validate_result_record


def build_result_record(
    *,
    metrics: dict[str, Any],
    manifest_sha256: str,
    predictions_sha256: str,
    model: dict[str, Any],
    protocol: dict[str, Any],
    scorer: dict[str, Any],
    statistics: dict[str, Any],
    consumers: list[str] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": "0.1.0",
        "dataset": {"manifest_sha256": manifest_sha256},
        "predictions": {"sha256": predictions_sha256},
        "model": model,
        "protocol": protocol,
        "scorer": scorer,
        "statistics": statistics,
        "metrics": metrics,
        "consumers": consumers or [],
    }
    payload["result_id"] = sha256_json(payload)
    return validate_result_record(payload)

