#!/usr/bin/env python3
"""Count geometry-qualified Task-B source capacity by frozen scene role."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from egoshift.split_roles import require_complete_split_roles


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--split-roles", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    roles_payload = json.loads(args.split_roles.read_text(encoding="utf-8"))
    require_complete_split_roles(roles_payload)
    roles = roles_payload["scenes"]
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    unique_observations: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    sources: list[dict[str, str]] = []
    for path in args.input:
        sources.append({"path": str(path), "sha256": sha256_file(path)})
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                scene_id = str(row["scene_id"])
                record = roles.get(scene_id)
                if record is None or record["drop_exact_alias_nonrepresentative"]:
                    continue
                role = record["role"]
                rotation = int(row["rotation_degrees"])
                turn = "any" if rotation == 180 else str(row["turn"]).lower()
                answer = str(row["answer"])
                key = f"{rotation}:{turn}:{answer}"
                counts[role][key] += 1
                unique_observations[role].add(
                    (scene_id, str(row["target_oid"]), str(row["frames"][0]))
                )
    output = {
        "schema_version": "0.1.0",
        "audit": "taskb_geometry_qualified_role_capacity",
        "sources": sources,
        "split_roles": {"path": str(args.split_roles), "sha256": sha256_file(args.split_roles)},
        "capacity": {role: dict(sorted(values.items())) for role, values in sorted(counts.items())},
        "unique_source_observations": {
            role: len(values) for role, values in sorted(unique_observations.items())
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output["capacity"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
