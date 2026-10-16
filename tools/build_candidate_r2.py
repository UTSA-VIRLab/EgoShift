#!/usr/bin/env python3
"""Build the mutable EgoShift candidate-r2 package from frozen seed hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from egoshift.candidate_build import build_candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("status") != "active_mutable_s05_candidate_build":
        raise SystemExit(
            "refusing candidate construction: config status must be "
            "'active_mutable_s05_candidate_build'; the supplied contract is "
            f"{config.get('status')!r}"
        )
    prohibitions = config.get("prohibitions") or {}
    if prohibitions.get("execute_this_builder_before_corrected_s04") is True:
        raise SystemExit(
            "refusing candidate construction while the corrected-S04 prohibition is active"
        )
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(
            f"refusing to overwrite non-empty candidate directory: {args.output}; "
            "use a new candidate version"
        )
    result = build_candidate(args.repo_root.resolve(), args.config.resolve(), args.output.resolve())
    summary = {
        "candidate_id": result["manifest"]["manifest_id"],
        "status": result["manifest"]["status"],
        "counts": result["manifest"]["counts"],
        "static_construction_pass": result["report"]["pass_for_s05_static_construction"],
        "release_ready": result["report"]["pass_for_s07_release_freeze"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
