#!/usr/bin/env python3
"""Build a cross-task physical-scene leakage-component map."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from egoshift.scene_fingerprint import leakage_components, metadata_fingerprint


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-root", action="append", required=True, type=Path)
    parser.add_argument("--overrides", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    fingerprints: dict[str, str] = {}
    sources: dict[str, list[dict[str, str]]] = defaultdict(list)
    conflicts: list[dict[str, str]] = []
    for root in args.scenes_root:
        for metadata_path in sorted(root.glob("scene_*/metadata.json")):
            if not metadata_path.parent.is_dir():
                continue
            scene_id = metadata_path.parent.name
            if re.fullmatch(r"scene_[0-9]{5}", scene_id) is None:
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            fingerprint = metadata_fingerprint(metadata)
            if scene_id in fingerprints and fingerprints[scene_id] != fingerprint:
                conflicts.append(
                    {
                        "scene_id": scene_id,
                        "first_fingerprint": fingerprints[scene_id],
                        "second_fingerprint": fingerprint,
                        "path": str(metadata_path),
                    }
                )
                continue
            fingerprints[scene_id] = fingerprint
            sources[scene_id].append(
                {"path": str(metadata_path), "sha256": sha256_file(metadata_path)}
            )
    if conflicts:
        raise ValueError(f"{len(conflicts)} scene IDs have conflicting physical metadata")

    override_payload = json.loads(args.overrides.read_text(encoding="utf-8"))
    pairs = [tuple(item["scenes"]) for item in override_payload["conservative_pairs"]]
    components = leakage_components(fingerprints, conservative_pairs=pairs)
    by_fingerprint: dict[str, list[str]] = defaultdict(list)
    by_component: dict[str, list[str]] = defaultdict(list)
    for scene_id, fingerprint in fingerprints.items():
        by_fingerprint[fingerprint].append(scene_id)
        by_component[components[scene_id]].append(scene_id)

    output = {
        "schema_version": "0.1.0",
        "map_id": "egoshift-physical-scene-components-r1",
        "normalization": {"ignored_top_level_metadata_keys": ["scene_id", "source_house_index"]},
        "roots": [str(root) for root in args.scenes_root],
        "overrides": {"path": str(args.overrides), "sha256": sha256_file(args.overrides)},
        "counts": {
            "scene_ids": len(fingerprints),
            "metadata_fingerprints": len(by_fingerprint),
            "leakage_components": len(by_component),
            "exact_alias_groups": sum(len(rows) > 1 for rows in by_fingerprint.values()),
            "multi_scene_components": sum(len(rows) > 1 for rows in by_component.values()),
        },
        "exact_alias_groups": [
            sorted(rows) for rows in by_fingerprint.values() if len(rows) > 1
        ],
        "components_with_multiple_scene_ids": [
            sorted(rows) for rows in by_component.values() if len(rows) > 1
        ],
        "scenes": {
            scene_id: {
                "metadata_fingerprint_sha256": fingerprints[scene_id],
                "leakage_component_id": components[scene_id],
                "component_representative": min(by_component[components[scene_id]]),
                "metadata_sources": sources[scene_id],
            }
            for scene_id in sorted(fingerprints)
        },
        "conflicts": conflicts,
        "pass": not conflicts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["counts"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
