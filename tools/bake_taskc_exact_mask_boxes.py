#!/usr/bin/env python3
"""Bake Task-C View-1 overlays from exact replayed instance-mask boxes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = RELEASE_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from egoshift.exact_mask_bbox import bake_exact_mask_bbox_overlays  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    root = RELEASE_ROOT.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "work/candidate_r2/sources/taskc_candidate_source_private.jsonl",
    )
    parser.add_argument(
        "--replay-sidecar",
        required=True,
        type=Path,
        help=(
            "JSON sidecar with exact_instance_mask_replay evidence, exact-mask "
            "nonzero-extent boxes, and frame/mask SHA-256 bindings"
        ),
    )
    parser.add_argument(
        "--asset-root", type=Path, default=root / "data/sim/scenes"
    )
    parser.add_argument(
        "--output-asset-root",
        type=Path,
        default=root / "work/candidate_r2/assets/taskc_exact_bbox",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "work/candidate_r2/sources/taskc_candidate_source_exact_bbox_private.jsonl",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=root / "work/candidate_r2/audits/taskc_exact_mask_bbox_baker.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = bake_exact_mask_bbox_overlays(
        input_jsonl=args.input,
        replay_sidecar=args.replay_sidecar,
        asset_root=args.asset_root,
        output_asset_root=args.output_asset_root,
        output_jsonl=args.output,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "counts": report.get("counts", {}),
                "errors": report.get("errors", []),
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

