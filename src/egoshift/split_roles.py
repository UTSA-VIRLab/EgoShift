"""Deterministic role assignment and fail-closed split-ledger validation."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence


ROLES = ("tuning_dev", "public_dev", "hidden_test")
EXPOSURE_AUDIT_ID = "egoshift_historical_exposure_ledger"
EXPOSURE_COMPLETE_STATUS = "complete_reviewed"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_exposure_manifest_sha256(
    records: Sequence[Mapping[str, Any]],
) -> str:
    """Hash the ordered path/content-hash ledger bound into split roles."""

    normalized: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"exposure record {index} is not an object")
        path = str(record.get("path") or "")
        digest = str(record.get("sha256") or "")
        if not path or path in seen_paths:
            raise ValueError(f"exposure record {index} has a missing/duplicate path")
        if not HEX_SHA256.fullmatch(digest):
            raise ValueError(f"exposure record {index} lacks a valid SHA-256")
        seen_paths.add(path)
        normalized.append({"path": path, "sha256": digest})
    if not normalized:
        raise ValueError("exposure manifest is empty")
    payload = json.dumps(
        sorted(normalized, key=lambda item: item["path"]),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_complete_exposure_audit(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an independently reviewed historical-exposure audit."""

    if payload.get("audit_id") != EXPOSURE_AUDIT_ID:
        raise ValueError("unexpected historical-exposure audit identifier")
    if payload.get("status") != EXPOSURE_COMPLETE_STATUS:
        raise ValueError("historical-exposure audit is not complete and reviewed")
    if payload.get("complete") is not True or payload.get("pass") is not True:
        raise ValueError("historical-exposure audit is not passing")
    records = payload.get("exposure_files")
    if not isinstance(records, list):
        raise ValueError("historical-exposure audit lacks an exposure-file manifest")
    observed_manifest_hash = canonical_exposure_manifest_sha256(records)
    if payload.get("manifest_sha256") != observed_manifest_hash:
        raise ValueError("historical-exposure manifest hash mismatch")
    if payload.get("reviewed_manifest_count") != len(records):
        raise ValueError("historical-exposure reviewed-manifest count mismatch")
    return {
        "audit_id": EXPOSURE_AUDIT_ID,
        "status": EXPOSURE_COMPLETE_STATUS,
        "complete": True,
        "manifest_sha256": observed_manifest_hash,
        "reviewed_manifest_count": len(records),
    }


def require_complete_split_roles(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Reject a role ledger unless it binds a complete exposure audit."""

    if payload.get("pass") is not True:
        raise ValueError("split-role audit is not passing")
    if payload.get("exposed_component_assigned_hidden_violations"):
        raise ValueError("split-role audit contains exposed hidden-test violations")
    binding = payload.get("exposure_ledger")
    if not isinstance(binding, Mapping):
        raise ValueError(
            "split-role audit is not bound to a complete reviewed exposure ledger"
        )
    if binding.get("status") != EXPOSURE_COMPLETE_STATUS or binding.get("complete") is not True:
        raise ValueError("split-role exposure binding is not complete and reviewed")
    for field in ("audit_sha256", "manifest_sha256"):
        if not HEX_SHA256.fullmatch(str(binding.get(field) or "")):
            raise ValueError(f"split-role exposure binding has invalid {field}")
    records = payload.get("exposure_files")
    if not isinstance(records, list):
        raise ValueError("split-role audit lacks exposure files")
    observed_manifest_hash = canonical_exposure_manifest_sha256(records)
    if binding.get("manifest_sha256") != observed_manifest_hash:
        raise ValueError("split-role exposure binding does not match exposure files")
    if binding.get("reviewed_manifest_count") != len(records):
        raise ValueError("split-role exposure binding count mismatch")
    return dict(binding)


def stable_bucket(seed: str, component_id: str, modulus: int = 10_000) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    digest = hashlib.sha256(f"{seed}|{component_id}".encode("utf-8")).hexdigest()
    return int(digest, 16) % modulus


def assign_component_role(
    component_id: str,
    *,
    seed: str,
    outcome_exposed: bool,
    hidden_fraction: float = 0.70,
    tuning_fraction: float = 0.15,
) -> str:
    """Assign a component before item selection, never hiding exposed outcomes."""

    if not (0.0 < hidden_fraction < 1.0):
        raise ValueError("hidden_fraction must be in (0,1)")
    if not (0.0 < tuning_fraction < 1.0 - hidden_fraction):
        raise ValueError("tuning_fraction must leave positive public fraction")
    bucket = stable_bucket(seed, component_id)
    if outcome_exposed:
        return "tuning_dev" if bucket % 2 == 0 else "public_dev"
    hidden_cut = round(hidden_fraction * 10_000)
    tuning_cut = hidden_cut + round(tuning_fraction * 10_000)
    if bucket < hidden_cut:
        return "hidden_test"
    if bucket < tuning_cut:
        return "tuning_dev"
    return "public_dev"
