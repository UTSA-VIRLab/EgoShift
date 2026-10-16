#!/usr/bin/env python3
"""Verify Task C's two structural guarantees on the released bytes.

The guarantees are established upstream, in the selector, on the SOURCE frames.
Baking then burns a referent box into View 1 and the builder copies assets
under new names, so neither guarantee is self-evidently still true in the
artifact people actually download. This checks the shipped files:

  1. Every physical target contributes exactly one item per class, and the
     three items reference a View-1 asset that is byte-identical. A View-1-only
     predictor therefore cannot beat chance -- not as a measured tendency, but
     because the three answers are attached to the same pixels.
  2. No two released items share a View-2 asset, so the pool is not padded with
     copies of the frame the question is actually about.

Both are checked by hashing the released PNGs, so a reviewer can reproduce the
result without rerunning any mining.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from egoshift.hashing import sha256_file  # noqa: E402

CLASSES = ("visible", "out_of_fov", "occluded")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    root = args.candidate_root

    digests: dict[Path, str] = {}

    def digest(relative: str) -> str:
        path = root / relative
        if path not in digests:
            digests[path] = sha256_file(path)
        return digests[path]

    failures: list[dict] = []
    per_split: dict[str, dict] = {}
    for split in ("hidden_test", "public_dev", "tuning_dev"):
        questions = {
            row["id"]: row
            for row in (
                json.loads(line)
                for line in (root / f"public/questions/task_c_{split}.jsonl")
                .read_text(encoding="utf-8").splitlines() if line.strip()
            )
        }
        provenance = [
            json.loads(line)
            for line in (root / f"private/provenance/task_c_{split}.jsonl")
            .read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        by_target: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
        view_two: Counter[str] = Counter()
        for record in provenance:
            item = questions[record["id"]]
            answer_class = record.get("answer_class") or record.get(
                "candidate_answer_class"
            )
            identity = (
                record.get("physical_scene_component_id"),
                record.get("target_oid"),
            )
            by_target[identity][answer_class] = digest(item["frames"][0])
            view_two[digest(item["frames"][1])] += 1

        incomplete = [k for k, v in by_target.items() if len(v) != len(CLASSES)]
        split_view_one = [
            k for k, v in by_target.items()
            if len(v) == len(CLASSES) and len(set(v.values())) != 1
        ]
        reused_view_two = {d: n for d, n in view_two.items() if n > 1}
        # A target-identity oracle: knowing the object, guess its commonest
        # answer. With one item per class per target this is exactly 1/3.
        oracle = sum(
            Counter(v.keys()).most_common(1)[0][1] for v in by_target.values()
        ) / max(1, len(provenance))

        per_split[split] = {
            "items": len(provenance),
            "distinct_targets": len(by_target),
            "targets_missing_a_class": len(incomplete),
            "targets_with_split_view_one": len(split_view_one),
            "reused_view_two_assets": len(reused_view_two),
            "target_identity_oracle": round(oracle, 6),
        }
        if incomplete:
            failures.append({"split": split, "targets_missing_a_class": len(incomplete)})
        if split_view_one:
            failures.append({"split": split, "targets_with_split_view_one": len(split_view_one)})
        if reused_view_two:
            failures.append({"split": split, "reused_view_two_assets": len(reused_view_two)})
        if abs(oracle - 1 / 3) > 1e-9:
            failures.append({"split": split, "target_identity_oracle": oracle})

    report = {
        "audit": "taskc_release_structure",
        "candidate_root": str(root),
        "assets_hashed": len(digests),
        "per_split": per_split,
        "guarantees": {
            "view_one_identical_within_every_target_triple": not any(
                "targets_with_split_view_one" in f for f in failures
            ),
            "view_two_unique_across_all_items": not any(
                "reused_view_two_assets" in f for f in failures
            ),
            "target_identity_oracle_exactly_chance": not any(
                "target_identity_oracle" in f for f in failures
            ),
        },
        "failures": failures,
        "pass": not failures,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("per_split", "guarantees", "pass")}, indent=1))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
