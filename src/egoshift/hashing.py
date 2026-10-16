"""Canonical hashing and safe path utilities."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any


_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:")


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize JSON deterministically for identifiers and provenance."""
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: str) -> str:
    """Return a normalized POSIX path or reject absolute/traversal paths."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path must be a non-empty string")
    normalized = value.replace("\\", "/")
    if normalized.startswith("/") or _WINDOWS_DRIVE.match(normalized):
        raise ValueError(f"absolute path is forbidden: {value!r}")
    path = PurePosixPath(normalized)
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"non-canonical or traversing path is forbidden: {value!r}")
    return str(path)


def resolve_within(root: str | Path, relative: str) -> Path:
    """Resolve a safe relative path and prove it remains within root."""
    base = Path(root).resolve()
    candidate = (base / safe_relative_path(relative)).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"path escapes root: {relative!r}") from exc
    return candidate

