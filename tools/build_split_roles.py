#!/usr/bin/env python3
"""Freeze cross-task scene roles while preventing outcome-exposed hidden data."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from egoshift.split_roles import (
    assign_component_role,
    canonical_exposure_manifest_sha256,
    require_complete_exposure_audit,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_scene_ids(path: Path) -> set[str]:
    scenes: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                scene_id = json.loads(line).get("scene_id")
                if scene_id:
                    scenes.add(str(scene_id))
    return scenes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--components", required=True, type=Path)
    parser.add_argument(
        "--exposure-audit",
        required=True,
        type=Path,
        help="Passing, independently reviewed exhaustive exposure-ledger audit.",
    )
    parser.add_argument(
        "--exposed-jsonl",
        action="append",
        default=[],
        type=Path,
        help="Optional cross-check; if supplied, must exactly match the audit manifest.",
    )
    parser.add_argument("--seed", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--hidden-fraction", type=float, default=0.70)
    parser.add_argument("--tuning-fraction", type=float, default=0.15)
    args = parser.parse_args()

    component_payload = json.loads(args.components.read_text(encoding="utf-8"))
    scenes = component_payload["scenes"]
    exposure_audit = json.loads(args.exposure_audit.read_text(encoding="utf-8"))
    exposure_binding = require_complete_exposure_audit(exposure_audit)
    audited_records = exposure_audit["exposure_files"]
    audited_paths = [Path(str(record["path"])) for record in audited_records]
    if args.exposed_jsonl:
        requested = sorted(str(path) for path in args.exposed_jsonl)
        frozen = sorted(str(path) for path in audited_paths)
        if requested != frozen:
            raise ValueError("--exposed-jsonl inputs do not exactly match exposure audit")

    exposed_scenes: set[str] = set()
    exposure_files: list[dict[str, object]] = []
    for expected, path in zip(audited_records, audited_paths):
        observed_hash = sha256_file(path)
        if observed_hash != expected["sha256"]:
            raise ValueError(f"exposure file hash mismatch: {path}")
        observed = jsonl_scene_ids(path)
        exposed_scenes.update(observed)
        exposure_files.append(
            {"path": str(path), "sha256": observed_hash, "scene_ids": len(observed)}
        )
    if canonical_exposure_manifest_sha256(exposure_files) != exposure_binding["manifest_sha256"]:
        raise ValueError("materialized exposure files disagree with reviewed manifest")
    unknown = sorted(exposed_scenes - set(scenes))
    if unknown:
        raise ValueError(f"exposure files reference {len(unknown)} unknown scene IDs")
    exposed_components = {
        scenes[scene_id]["leakage_component_id"] for scene_id in exposed_scenes
    }
    component_roles: dict[str, str] = {}
    for record in scenes.values():
        component_id = record["leakage_component_id"]
        component_roles.setdefault(
            component_id,
            assign_component_role(
                component_id,
                seed=args.seed,
                outcome_exposed=component_id in exposed_components,
                hidden_fraction=args.hidden_fraction,
                tuning_fraction=args.tuning_fraction,
            ),
        )
    scene_records = {
        scene_id: {
            "leakage_component_id": record["leakage_component_id"],
            "metadata_fingerprint_sha256": record["metadata_fingerprint_sha256"],
            "component_representative": record["component_representative"],
            "drop_exact_alias_nonrepresentative": (
                record["metadata_fingerprint_sha256"]
                != scenes[record["component_representative"]]["metadata_fingerprint_sha256"]
                and False
            ),
            "outcome_exposed_component": record["leakage_component_id"] in exposed_components,
            "role": component_roles[record["leakage_component_id"]],
        }
        for scene_id, record in sorted(scenes.items())
    }
    # Exact metadata aliases share a fingerprint. Keep only the lexicographically
    # first scene ID for item construction; perceptual-only overrides are co-grouped
    # but are not automatically dropped.
    by_fingerprint: dict[str, list[str]] = defaultdict(list)
    for scene_id, record in scene_records.items():
        by_fingerprint[record["metadata_fingerprint_sha256"]].append(scene_id)
    dropped_aliases: list[str] = []
    for members in by_fingerprint.values():
        for scene_id in sorted(members)[1:]:
            scene_records[scene_id]["drop_exact_alias_nonrepresentative"] = True
            dropped_aliases.append(scene_id)

    role_counts = Counter(component_roles.values())
    exposed_hidden = [
        component_id
        for component_id in exposed_components
        if component_roles[component_id] == "hidden_test"
    ]
    output = {
        "schema_version": "0.1.0",
        "split_contract_id": "egoshift-candidate-r2-physical-scene-roles",
        "seed": args.seed,
        "fractions_for_unexposed_components": {
            "hidden_test": args.hidden_fraction,
            "tuning_dev": args.tuning_fraction,
            "public_dev": 1.0 - args.hidden_fraction - args.tuning_fraction,
        },
        "components": {"path": str(args.components), "sha256": sha256_file(args.components)},
        "exposure_ledger": {
            **exposure_binding,
            "audit_path": str(args.exposure_audit),
            "audit_sha256": sha256_file(args.exposure_audit),
        },
        "exposure_files": exposure_files,
        "counts": {
            "scenes": len(scene_records),
            "components": len(component_roles),
            "outcome_exposed_scenes": len(exposed_scenes),
            "outcome_exposed_components": len(exposed_components),
            "dropped_exact_alias_scene_ids": len(dropped_aliases),
            "component_roles": dict(sorted(role_counts.items())),
        },
        "dropped_exact_alias_scene_ids": sorted(dropped_aliases),
        "scenes": scene_records,
        "exposed_component_assigned_hidden_violations": exposed_hidden,
        "pass": not exposed_hidden,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output["counts"], indent=2, sort_keys=True))
    return 0 if output["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
