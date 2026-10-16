#!/usr/bin/env python3
"""Build the static S05 Task-C candidate (never a release)."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egoshift.hashing import sha256_file  # noqa: E402
from egoshift.taskc_candidate import (  # noqa: E402
    TaskCCandidateError,
    TaskCSourceSpec,
    build_taskc_candidate,
)


def _pairs(values: list[str], flag: str) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for value in values:
        name, separator, payload = value.partition("=")
        if not separator or not name or not payload:
            raise SystemExit(f"{flag} expects NAME=VALUE, got {value!r}")
        parsed.append((name, payload))
    return parsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split-roles", required=True, type=Path)
    parser.add_argument(
        "--source", required=True, action="append", default=[],
        help="NAME=path/to/private_rows.jsonl (repeatable).",
    )
    parser.add_argument(
        "--source-asset-root", required=True, action="append", default=[],
        help="NAME=path/to/asset/root (repeatable; several roots per source allowed).",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    args = parser.parse_args()
    if args.audit_report.exists():
        raise ValueError("refusing to overwrite audit report")

    sources = _pairs(args.source, "--source")
    roots: dict[str, list[Path]] = defaultdict(list)
    for name, value in _pairs(args.source_asset_root, "--source-asset-root"):
        roots[name].append(Path(value))
    specs: list[TaskCSourceSpec] = []
    for name, path in sources:
        if name not in roots:
            raise SystemExit(f"source {name!r} has no --source-asset-root")
        specs.append(
            TaskCSourceSpec(
                name=name, jsonl_path=Path(path), asset_roots=tuple(roots[name])
            )
        )

    try:
        audit = build_taskc_candidate(
            config_path=args.config,
            split_roles_path=args.split_roles,
            source_specs=specs,
            output_root=args.output_root,
            audit_report_path=args.audit_report,
        )
    except Exception as exc:
        failure: dict[str, object] = {
            "schema_version": "0.1.0",
            "audit": "taskc_candidate_r2_static_construction",
            "status": "blocked_fail_closed",
            "release_ready": False,
            "pass": False,
            "error": {"type": type(exc).__name__, "message": str(exc)},
            "inputs": {
                "config": {
                    "path": str(args.config),
                    "sha256": sha256_file(args.config) if args.config.is_file() else None,
                },
                "split_roles": {
                    "path": str(args.split_roles),
                    "sha256": (
                        sha256_file(args.split_roles) if args.split_roles.is_file() else None
                    ),
                },
                "sources": [
                    {
                        "name": spec.name,
                        "path": str(spec.jsonl_path),
                        "sha256": (
                            sha256_file(spec.jsonl_path)
                            if spec.jsonl_path.is_file()
                            else None
                        ),
                    }
                    for spec in specs
                ],
            },
            "output_root_was_created": args.output_root.exists(),
        }
        if isinstance(exc, TaskCCandidateError):
            failure["capacity_blocker"] = exc.diagnostics
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
                "counts": audit["counts"],
                "manifest": audit["output"]["manifest"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
