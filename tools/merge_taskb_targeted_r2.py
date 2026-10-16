#!/usr/bin/env python3
"""Verify and merge a complete set of targeted-r2 Task-B miner shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egoshift.taskb_targeted_merge import merge_targeted_shards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split-roles", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--shard-report", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--allow-scene-errors",
        action="store_true",
        help=(
            "Merge shards that recorded per-scene simulator errors (e.g. render "
            "timeouts). Such a scene contributes no rows, so emitted rows are "
            "unaffected; every error is copied verbatim into the merge audit and "
            "operational_clean is set false."
        ),
    )
    parser.add_argument(
        "--allow-global-shortfall",
        action="store_true",
        help=(
            "Merge a mining round that did not reach its requested per-cell "
            "counts. Every row-level gate is still enforced; the audit records "
            "verification_pass=true with quota_pass=false and the exact "
            "shortfall, and the candidate builder still fails closed if all "
            "sources together cannot fill the finite quotas."
        ),
    )
    args = parser.parse_args()
    audit = merge_targeted_shards(
        config_path=args.config,
        split_roles_path=args.split_roles,
        asset_root=args.asset_root,
        shard_report_paths=args.shard_report,
        output_path=args.output,
        report_path=args.report,
        allow_global_shortfall=args.allow_global_shortfall,
        allow_scene_errors=args.allow_scene_errors,
    )
    print(
        json.dumps(
            {
                "pass": audit["pass"],
                "verification_pass": audit.get("verification_pass"),
                "quota_pass": audit.get("quota_pass"),
                "operational_clean": audit.get("operational_clean"),
                "scene_error_count": audit.get("scene_error_count"),
                "unmet_global_quotas": audit.get("unmet_global_quotas"),
                "counts": audit["counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

