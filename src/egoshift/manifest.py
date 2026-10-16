"""Manifest loading and content verification."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .hashing import resolve_within, sha256_file
from .records import read_json
from .schemas import ValidationError, validate_manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    return validate_manifest(read_json(path))


def verify_manifest_files(manifest: dict[str, Any], root: str | Path) -> dict[str, Any]:
    validate_manifest(manifest)
    checked: list[dict[str, Any]] = []
    for entry in manifest["files"]:
        path = resolve_within(root, entry["path"])
        if not path.is_file():
            raise ValidationError(f"manifest file missing: {entry['path']}")
        observed = sha256_file(path)
        if observed != entry["sha256"]:
            raise ValidationError(
                f"manifest hash mismatch for {entry['path']}: expected {entry['sha256']}, observed {observed}"
            )
        checked.append({"path": entry["path"], "sha256": observed, "status": "ok"})
    return {"manifest_id": manifest["manifest_id"], "checked": checked, "status": "ok"}

