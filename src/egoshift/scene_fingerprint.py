"""Physical-scene fingerprints and conservative leakage components."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping


IGNORED_METADATA_KEYS = frozenset({"scene_id", "source_house_index"})


def normalized_scene_metadata(
    metadata: Mapping[str, Any],
    *,
    ignored_keys: frozenset[str] = IGNORED_METADATA_KEYS,
) -> dict[str, Any]:
    """Remove only nonphysical scene aliases and return JSON-canonical data."""

    return {key: value for key, value in metadata.items() if key not in ignored_keys}


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def metadata_fingerprint(metadata: Mapping[str, Any]) -> str:
    return canonical_sha256(normalized_scene_metadata(metadata))


class UnionFind:
    def __init__(self, values: Iterable[str]):
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        if value not in self.parent:
            self.parent[value] = value
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, first: str, second: str) -> None:
        root_first = self.find(first)
        root_second = self.find(second)
        if root_first == root_second:
            return
        lower, upper = sorted((root_first, root_second))
        self.parent[upper] = lower


def leakage_components(
    fingerprints: Mapping[str, str],
    *,
    conservative_pairs: Iterable[tuple[str, str]] = (),
) -> dict[str, str]:
    """Return stable component IDs joining exact and reviewed perceptual aliases."""

    union = UnionFind(fingerprints)
    by_fingerprint: dict[str, list[str]] = {}
    for scene_id, fingerprint in fingerprints.items():
        by_fingerprint.setdefault(fingerprint, []).append(scene_id)
    for scene_ids in by_fingerprint.values():
        for scene_id in scene_ids[1:]:
            union.union(scene_ids[0], scene_id)
    for first, second in conservative_pairs:
        if first not in fingerprints or second not in fingerprints:
            raise ValueError(f"override references unknown scene pair: {first}, {second}")
        union.union(first, second)
    members: dict[str, list[str]] = {}
    for scene_id in fingerprints:
        members.setdefault(union.find(scene_id), []).append(scene_id)
    root_to_component = {
        root: "psc_" + canonical_sha256(sorted(scene_ids))[:20]
        for root, scene_ids in members.items()
    }
    return {scene_id: root_to_component[union.find(scene_id)] for scene_id in fingerprints}


def load_metadata(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
