#!/usr/bin/env python3
"""DA-1 selection for Task A by within-type caliper pairing.

Pairing features are VIEW-ORDERED: matching min/max left the signed view-order
asymmetry alive (a *different* pair's second view shows the second instance),
and a probe read 54% from the signed coverage difference alone.

Count-per-bin matching (the v1..v3b selectors) equalises the classes to bin
resolution, but the same/different construction statistics differ in
distribution *shape*, so a gradient-boosting probe kept reading 62-65% from
within-bin structure. This selector matches the joint continuous distribution
directly: within every (role, object type) group, each kept *same* row is
greedily paired with the nearest unused *different* row in z-scored cue space
(min coverage, max coverage, coverage gap, box-centre distance), and the pair
is kept only if it lies within a caliper. Classes come out exactly 50/50 per
type with near-identical joint cue distributions; everything unpairable is
discarded. Deterministic given the pool: candidates are processed in stable-
hash order and ties break on the hash.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = ("same", "different")
ROLES = ("hidden_test", "tuning_dev", "public_dev")
FEATURE_NAMES = (
    "view_1_coverage",
    "view_2_coverage",
    "signed_log_pixel_ratio",
    "box_center_distance",
)


def features_of(row: dict) -> tuple[float, ...]:
    evidence = row["_candidate_source"]["gold_evidence"]
    centers = []
    for key in ("view_1", "view_2"):
        x0, y0, x1, y1 = evidence[key]["bbox_xyxy"]
        size = evidence[key]["projection"]["image_size"]
        centers.append(((x0 + x1) / 2 / size["w"], (y0 + y1) / 2 / size["h"]))
    (ax, ay), (bx, by) = centers
    view_1, view_2 = evidence["view_1"], evidence["view_2"]
    import math
    return (
        float(view_1["view_coverage"]),
        float(view_2["view_coverage"]),
        math.log(max(1, view_1["visible_pixels"]) / max(1, view_2["visible_pixels"])),
        ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5,
    )


def stable_key(row: dict) -> str:
    return row["_candidate_source"]["stable_identity_sha256"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
    parser.add_argument("--caliper", type=float, default=0.25,
                        help="Maximum pair distance in z-scored cue space.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise SystemExit("refusing to overwrite an existing selection/report")

    rows: list[dict] = []
    inputs = []
    for path in sorted(glob.glob(args.input_glob)):
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        inputs.append({"path": path, "sha256": digest})
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    if not rows:
        raise SystemExit("no input rows")

    pooled = [features_of(r) for r in rows]
    means = [statistics.mean(f[i] for f in pooled) for i in range(len(FEATURE_NAMES))]
    stds = [
        max(1e-9, statistics.pstdev(f[i] for f in pooled))
        for i in range(len(FEATURE_NAMES))
    ]

    def z(row: dict) -> tuple[float, ...]:
        raw = features_of(row)
        return tuple((raw[i] - means[i]) / stds[i] for i in range(len(raw)))

    selected_all: list[dict] = []
    role_reports: dict[str, dict] = {}
    for role in ROLES:
        groups: dict[str, dict[str, list[dict]]] = defaultdict(
            lambda: {name: [] for name in CLASSES}
        )
        for row in rows:
            if row["candidate_role"] == role:
                groups[row["metadata"]["object_type"]][row["candidate_answer_class"]].append(row)

        kept: list[dict] = []
        pair_distances: list[float] = []
        per_type: dict[str, int] = {}
        for object_type in sorted(groups):
            same = sorted(groups[object_type]["same"], key=stable_key)
            different = sorted(groups[object_type]["different"], key=stable_key)
            if not same or not different:
                continue
            different_z = [(z(r), r) for r in different]
            used: set[str] = set()
            pairs = 0
            for row in same:
                anchor = z(row)
                best = None
                best_distance = args.caliper
                for candidate_z, candidate in different_z:
                    key = stable_key(candidate)
                    if key in used:
                        continue
                    distance = sum(
                        (anchor[i] - candidate_z[i]) ** 2 for i in range(len(anchor))
                    ) ** 0.5
                    if distance < best_distance or (
                        best is not None
                        and abs(distance - best_distance) < 1e-12
                        and key < stable_key(best)
                    ):
                        best, best_distance = candidate, distance
                if best is None:
                    continue
                used.add(stable_key(best))
                kept.extend((row, best))
                pair_distances.append(best_distance)
                pairs += 1
            if pairs:
                per_type[object_type] = pairs

        residual = {}
        for index, field in enumerate(FEATURE_NAMES):
            by_class = defaultdict(list)
            for row in kept:
                by_class[row["candidate_answer_class"]].append(features_of(row)[index])
            residual[field] = {
                name: round(statistics.mean(values), 6)
                for name, values in sorted(by_class.items())
            }
        components = Counter(
            row["_candidate_source"]["physical_component_id"] for row in kept
        )
        role_reports[role] = {
            "pairs": len(pair_distances),
            "pairs_by_type": per_type,
            "mean_pair_distance": round(statistics.mean(pair_distances), 6)
            if pair_distances else None,
            "post_match_mean_by_class": residual,
            "distinct_physical_components": len(components),
            "largest_component_share": round(
                max(components.values()) / len(kept), 6
            ) if kept else 0.0,
        }
        selected_all.extend(kept)

    selected_all.sort(key=stable_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected_all:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    by_role_class = Counter(
        (r["candidate_role"], r["candidate_answer_class"]) for r in selected_all
    )
    payload = {
        "schema_version": "0.1.0",
        "audit": "taska_pair_matched_selection",
        "inputs": inputs,
        "input_rows": len(rows),
        "caliper_z": args.caliper,
        "features": list(FEATURE_NAMES),
        "feature_standardization": {
            "means": [round(m, 8) for m in means],
            "stds": [round(s, 8) for s in stds],
        },
        "roles": role_reports,
        "selected_rows": len(selected_all),
        "selected_by_role_class": {
            f"{role}:{name}": count
            for (role, name), count in sorted(by_role_class.items())
        },
        "pass": all(
            by_role_class[(role, "same")] == by_role_class[(role, "different")]
            for role in ROLES
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in (
        "input_rows", "selected_rows", "selected_by_role_class", "pass"
    )}, indent=1))
    return 0 if payload["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
