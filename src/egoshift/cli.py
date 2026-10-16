"""Command-line interface for validation, scoring, and smoke testing."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .manifest import load_manifest, verify_manifest_files
from .records import read_json
from .schemas import ValidationError
from .scoring import score_files
from .support import GeometryConfig
from .taskb_audit import audit_taskb_file


def _dump(value: Any) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False))


def _validate_manifest(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    _dump(verify_manifest_files(manifest, args.root))
    return 0


def _score(args: argparse.Namespace) -> int:
    report = score_files(args.questions, args.predictions)
    if args.strict and (
        report["overall"].get("missing", 0)
        or report["overall"].get("extra_predictions", 0)
        or report["overall"].get("parse_failures", 0)
        or report["overall"].get("error", 0)
    ):
        _dump(report)
        return 2
    _dump(report)
    return 0


def run_smoke(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "manifest.json"
    manifest = load_manifest(manifest_path)
    verification = verify_manifest_files(manifest, root)
    report = score_files(root / "questions.jsonl", root / "predictions.jsonl")
    expected = read_json(root / "expected_report.json")
    observed = {
        "overall": report["overall"],
        "tasks": report["tasks"],
        "missing_ids": report["missing_ids"],
        "extra_ids": report["extra_ids"],
    }
    if observed != expected:
        raise ValidationError(
            "smoke metrics differ from expected fixture:\n"
            + json.dumps({"expected": expected, "observed": observed}, indent=2, sort_keys=True)
        )
    return {"status": "ok", "manifest": verification, "metrics": observed}


def _smoke(args: argparse.Namespace) -> int:
    _dump(run_smoke(args.root))
    return 0


def _audit_taskb(args: argparse.Namespace) -> int:
    geometry = GeometryConfig(
        source_bearing_half_width_degrees=args.source_bearing_half_width,
        zone_width_degrees=args.zone_width,
    )
    expected_rotations = [float(value) for value in args.expected_rotations.split(",") if value]
    expected_actions = {
        (rotation, direction)
        for rotation in expected_rotations
        for direction in ("left", "right")
    }
    report = audit_taskb_file(
        args.questions,
        geometry,
        expected_sha256=args.expected_sha256,
        expected_actions=expected_actions,
        expected_items_per_cell=args.expected_items_per_cell,
    )
    _dump(report)
    return 0 if report["pass"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="egoshift")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-manifest", help="validate schema, safe paths, and file hashes")
    validate.add_argument("manifest")
    validate.add_argument("--root", required=True)
    validate.set_defaults(handler=_validate_manifest)

    score = subparsers.add_parser("score", help="score a JSONL prediction file")
    score.add_argument("--questions", required=True)
    score.add_argument("--predictions", required=True)
    score.add_argument("--strict", action="store_true")
    score.set_defaults(handler=_score)

    smoke = subparsers.add_parser("smoke", help="validate and score the bundled synthetic fixture")
    smoke.add_argument("--root", required=True)
    smoke.set_defaults(handler=_smoke)

    taskb = subparsers.add_parser(
        "audit-taskb", help="audit Task-B action cells against analytic geometry support"
    )
    taskb.add_argument("--questions", required=True)
    taskb.add_argument("--expected-sha256")
    taskb.add_argument(
        "--source-bearing-half-width",
        type=float,
        default=66.58677555362947,
        help="explicit View-1 target-center horizontal-bearing gate; not camera HFOV",
    )
    taskb.add_argument("--zone-width", type=float, default=90.0)
    taskb.add_argument("--expected-rotations", default="30,60,90,120,180")
    taskb.add_argument("--expected-items-per-cell", type=int)
    taskb.set_defaults(handler=_audit_taskb)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, ValidationError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
