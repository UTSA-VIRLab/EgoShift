#!/usr/bin/env python3
"""Build the corrected, static S05 Task-B candidate (never a release)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egoshift.hashing import sha256_file
from egoshift.taskb_candidate_r2 import (
    CandidateCapacityError,
    SourceSpec,
    build_candidate,
)


def _failure_report(args: argparse.Namespace, exc: Exception) -> dict[str, object]:
    inputs: dict[str, object] = {}
    def _record(value: object) -> dict[str, object]:
        path = Path(str(value)).resolve()
        return {
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
        }

    for label, value in (
        ("config", args.config),
        ("split_roles", args.split_roles),
        ("reservoir_index", args.reservoir_index),
        ("reservoir_geometry_report", args.reservoir_geometry_report),
        ("reservoir_asset_trust_root", args.reservoir_asset_trust_root),
        ("targeted_index", args.targeted_index),
        ("targeted_geometry_report", args.targeted_geometry_report),
        ("targeted_merge_report", args.targeted_merge_report),
        ("targeted_asset_trust_root", args.targeted_asset_trust_root),
    ):
        if isinstance(value, list):
            inputs[label] = [_record(item) for item in value]
        else:
            inputs[label] = _record(value)
    payload: dict[str, object] = {
        "schema_version": "0.1.0",
        "audit": "taskb_candidate_r2_static_construction",
        "status": "blocked_fail_closed",
        "release_ready": False,
        "pass": False,
        "error": {"type": type(exc).__name__, "message": str(exc)},
        "inputs": inputs,
        "output_root_was_created": args.output_root.exists(),
    }
    if isinstance(exc, CandidateCapacityError):
        payload["capacity_blocker"] = exc.diagnostics
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split-roles", required=True, type=Path)
    parser.add_argument("--reservoir-index", required=True, type=Path)
    parser.add_argument("--reservoir-geometry-report", required=True, type=Path)
    parser.add_argument("--reservoir-asset-root", required=True, type=Path)
    parser.add_argument("--reservoir-asset-trust-root", required=True, type=Path)
    # Repeatable: one value per targeted mining run.  A single occurrence of
    # each reproduces the original two-source invocation exactly.
    parser.add_argument("--targeted-index", required=True, action="append", type=Path)
    parser.add_argument(
        "--targeted-geometry-report", required=True, action="append", type=Path
    )
    parser.add_argument(
        "--targeted-merge-report", required=True, action="append", type=Path
    )
    parser.add_argument("--targeted-asset-root", required=True, action="append", type=Path)
    parser.add_argument(
        "--targeted-asset-trust-root", required=True, action="append", type=Path
    )
    parser.add_argument("--targeted-name", action="append", default=None)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    args = parser.parse_args()
    if args.audit_report.exists():
        raise ValueError("refusing to overwrite audit report")
    targeted_lists = (
        args.targeted_index,
        args.targeted_geometry_report,
        args.targeted_merge_report,
        args.targeted_asset_root,
        args.targeted_asset_trust_root,
    )
    if len({len(item) for item in targeted_lists}) != 1:
        raise ValueError("targeted source flags must be supplied the same number of times")
    default_names = ["targeted_r2_geometry_b665868"] + [
        f"targeted_source_{index}" for index in range(2, len(args.targeted_index) + 1)
    ]
    names = args.targeted_name or default_names
    if len(names) != len(args.targeted_index):
        raise ValueError("--targeted-name count does not match targeted source count")
    if len(set(names)) != len(names):
        raise ValueError("targeted source names must be unique")
    specs = (
        SourceSpec(
            name="reservoir_geometry_b665868",
            kind="reservoir",
            index_path=args.reservoir_index,
            geometry_report_path=args.reservoir_geometry_report,
            asset_root=args.reservoir_asset_root,
            asset_trust_root=args.reservoir_asset_trust_root,
        ),
        *(
            SourceSpec(
                name=name,
                kind="targeted",
                index_path=index,
                geometry_report_path=report,
                asset_root=asset_root,
                asset_trust_root=trust_root,
                upstream_merge_report_path=merge_report,
            )
            for name, index, report, merge_report, asset_root, trust_root in zip(
                names,
                args.targeted_index,
                args.targeted_geometry_report,
                args.targeted_merge_report,
                args.targeted_asset_root,
                args.targeted_asset_trust_root,
            )
        ),
    )
    try:
        audit = build_candidate(
            config_path=args.config,
            split_roles_path=args.split_roles,
            source_specs=specs,
            output_root=args.output_root,
            audit_report_path=args.audit_report,
        )
    except Exception as exc:
        failure = _failure_report(args, exc)
        args.audit_report.parent.mkdir(parents=True, exist_ok=True)
        args.audit_report.write_text(
            json.dumps(failure, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "pass": audit["pass"],
                "release_ready": audit["release_ready"],
                "manifest": audit["output"]["manifest"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
