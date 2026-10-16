#!/usr/bin/env python3
"""Run the S05 gate: the integrated static audit over Tasks A, B, and C."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egoshift.integrated_audit import audit_integrated_candidate  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--taska-root", type=Path, default=None)
    parser.add_argument("--taska-build-report", type=Path, default=None)
    parser.add_argument("--taskb-root", required=True, type=Path)
    parser.add_argument("--taskc-root", type=Path, default=None)
    parser.add_argument("--split-roles", required=True, type=Path)
    parser.add_argument("--taskb-config", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    report = audit_integrated_candidate(
        taskb_root=args.taskb_root,
        taskc_root=args.taskc_root,
        taska_root=args.taska_root,
        taska_build_report_path=args.taska_build_report,
        split_roles_path=args.split_roles,
        taskb_config_path=args.taskb_config,
        report_path=args.report,
    )
    print(
        json.dumps(
            {
                "pass": report["pass"],
                "release_ready": report["release_ready"],
                "failures": report["failures"][:10],
                "checks": report["checks"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
