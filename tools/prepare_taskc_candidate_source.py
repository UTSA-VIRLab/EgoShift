#!/usr/bin/env python3
"""Build the hash-pinned private Task-C candidate source and audit report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = RELEASE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from egoshift.taskc_source import (  # noqa: E402
    DEFAULT_QUOTAS,
    EXPECTED_UPSTREAM_SHA256,
    TaskCSourceConfig,
    prepare_taskc_source,
)


def _repository_root() -> Path:
    return RELEASE_ROOT.parent


def build_parser() -> argparse.ArgumentParser:
    root = _repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=root / "data/sim/qa_mask_verified_taskc_2000target_afix.jsonl",
    )
    parser.add_argument("--expected-source-sha256", default=EXPECTED_UPSTREAM_SHA256)
    parser.add_argument(
        "--split-roles",
        type=Path,
        default=root / "work/candidate_r2/audits/split_roles.json",
    )
    parser.add_argument(
        "--scenes-root", type=Path, default=root / "data/sim/scenes"
    )
    parser.add_argument(
        "--robust-sidecar",
        action="append",
        type=Path,
        default=None,
        help="Robust bbox sidecar; repeat for multiple files.",
    )
    parser.add_argument(
        "--supplement",
        action="append",
        type=Path,
        default=[],
        help="Live-mined supplement JSONL; repeat for multiple shards.",
    )
    parser.add_argument(
        "--supplements-only",
        action="store_true",
        help=(
            "Build the pool from supplements alone, holding out the upstream "
            "template rows. The upstream file is still read and SHA-256 bound "
            "for provenance and split roles. Use this to keep Task C "
            "single-provenance: the template pool renders at a 90-degree "
            "vertical FOV along named orbits, the all-class miner at 120 "
            "degrees from jittered poses, and the difference is visible in "
            "View 1."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "work/candidate_r2/sources/taskc_candidate_source_private.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "work/candidate_r2/audits/taskc_candidate_source.json",
    )
    parser.add_argument("--tuning-per-class", type=int, default=DEFAULT_QUOTAS["tuning_dev"])
    parser.add_argument("--public-per-class", type=int, default=DEFAULT_QUOTAS["public_dev"])
    parser.add_argument("--hidden-per-class", type=int, default=DEFAULT_QUOTAS["hidden_test"])
    parser.add_argument("--minimum-source-pixels", type=int, default=500)
    parser.add_argument("--minimum-visible-target-pixels", type=int, default=30)
    parser.add_argument("--minimum-occluder-overlap", type=float, default=0.05)
    parser.add_argument("--minimum-partial-overlap", type=float, default=0.10)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = _repository_root()
    sidecars = args.robust_sidecar
    if sidecars is None:
        sidecars = [
            root / "data/sim/taskc_test_bboxes_ROBUST.json",
            root / "data/sim/taskc_dev_bboxes_ROBUST.json",
        ]
    config = TaskCSourceConfig(
        source_path=args.source,
        expected_source_sha256=args.expected_source_sha256,
        split_roles_path=args.split_roles,
        scenes_root=args.scenes_root,
        robust_sidecar_paths=tuple(sidecars),
        supplement_jsonl_paths=tuple(args.supplement),
        supplements_only=bool(args.supplements_only),
        output_path=args.output,
        report_path=args.report,
        quotas={
            "tuning_dev": args.tuning_per_class,
            "public_dev": args.public_per_class,
            "hidden_test": args.hidden_per_class,
        },
        minimum_source_pixels=args.minimum_source_pixels,
        minimum_visible_target_pixels=args.minimum_visible_target_pixels,
        minimum_occluder_overlap=args.minimum_occluder_overlap,
        minimum_partial_overlap=args.minimum_partial_overlap,
    )
    report = prepare_taskc_source(config)
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "source_sha256": report["source"].get("observed_sha256"),
                "capacity": report.get("capacity"),
                "missing_quota_cells": report.get("missing_quota_cells", []),
                "output": report["output"],
                "report": str(args.report),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

