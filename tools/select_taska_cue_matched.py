#!/usr/bin/env python3
"""DA-1 cue-matched selection for Task A.

The mined pool carries a real construction cue: both the minimum and the
maximum per-view coverage statistic separate *same* from *different* (measured
on the full 2026-08-22 pool: min 0.611 vs 0.682, max 0.692 vs 0.791). Selection
therefore matches the two classes on the joint cell

    object type x min-coverage quintile x max-coverage tercile

keeping an equal number of each class in every cell, via the same matcher Task
C uses for its type shortcut. After matching, no rule reading the object noun
or either coverage statistic (to bin resolution) can beat chance by
construction. Bin edges are pooled quantiles of the input and are recorded in
the report, so the selection is deterministic given the mined pool.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from egoshift.taskc_type_matching import match_object_type_distribution  # noqa: E402

CLASSES = ("same", "different")
ROLES = ("hidden_test", "tuning_dev", "public_dev")
MIN_COVERAGE_QUANTILES = 10
MAX_COVERAGE_QUANTILES = 5
# The strongest residual cue after coverage matching is an aiming artifact:
# both views of a *same* pair aim at one instance, so its box lands in nearly
# the same frame position in both views (mean normalized centre distance
# 0.039), while a *different* pair's second view aims at the other instance
# (0.084). A grouped gradient-boosting probe read 68% from this distance
# alone, so it joins the matching cell.
CENTER_DISTANCE_QUANTILES = 3


def coverage(row: dict, field: str) -> float:
    return float(row["_candidate_source"]["gold_evidence"][field])


def center_distance(row: dict) -> float:
    evidence = row["_candidate_source"]["gold_evidence"]
    centers = []
    for key in ("view_1", "view_2"):
        x0, y0, x1, y1 = evidence[key]["bbox_xyxy"]
        size = evidence[key]["projection"]["image_size"]
        centers.append(((x0 + x1) / 2 / size["w"], (y0 + y1) / 2 / size["h"]))
    (ax, ay), (bx, by) = centers
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def quantile_edges(values: list[float], bins: int) -> list[float]:
    ordered = sorted(values)
    return [ordered[int(len(ordered) * k / bins)] for k in range(1, bins)]


def bin_index(value: float, edges: list[float]) -> int:
    for index, edge in enumerate(edges):
        if value < edge:
            return index
    return len(edges)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-glob", required=True)
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

    min_edges = quantile_edges(
        [coverage(r, "minimum_view_coverage") for r in rows], MIN_COVERAGE_QUANTILES
    )
    max_edges = quantile_edges(
        [coverage(r, "maximum_view_coverage") for r in rows], MAX_COVERAGE_QUANTILES
    )
    distance_edges = quantile_edges(
        [center_distance(r) for r in rows], CENTER_DISTANCE_QUANTILES
    )

    def cell_of(row: dict) -> str:
        low = bin_index(coverage(row, "minimum_view_coverage"), min_edges)
        high = bin_index(coverage(row, "maximum_view_coverage"), max_edges)
        distance = bin_index(center_distance(row), distance_edges)
        return f"{row['metadata']['object_type']}|m{low}|M{high}|d{distance}"

    selected_all: list[dict] = []
    role_reports: dict[str, dict] = {}
    for role in ROLES:
        candidates = [r for r in rows if r["candidate_role"] == role]
        selected, report = match_object_type_distribution(
            candidates,
            classes=CLASSES,
            class_of=lambda r: r["candidate_answer_class"],
            type_of=cell_of,
            order_key=lambda r: r["_candidate_source"]["stable_identity_sha256"],
        )
        residual = {}
        for field in ("minimum_view_coverage", "maximum_view_coverage"):
            by_class = defaultdict(list)
            for row in selected:
                by_class[row["candidate_answer_class"]].append(coverage(row, field))
            residual[field] = {
                name: round(statistics.mean(values), 6)
                for name, values in sorted(by_class.items())
            }
        components = Counter(
            row["_candidate_source"]["physical_component_id"] for row in selected
        )
        role_reports[role] = {
            "matcher": report,
            "post_match_mean_coverage_by_class": residual,
            "distinct_physical_components": len(components),
            "largest_component_share": round(
                max(components.values()) / len(selected), 6
            ) if selected else 0.0,
        }
        selected_all.extend(selected)

    selected_all.sort(
        key=lambda r: r["_candidate_source"]["stable_identity_sha256"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in selected_all:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    payload = {
        "schema_version": "0.1.0",
        "audit": "taska_cue_matched_selection",
        "inputs": inputs,
        "input_rows": len(rows),
        "coverage_bin_edges": {
            "minimum_view_coverage": [round(e, 8) for e in min_edges],
            "maximum_view_coverage": [round(e, 8) for e in max_edges],
            "box_center_distance": [round(e, 8) for e in distance_edges],
        },
        "matching_cell": (
            "object_type x min-coverage bin x max-coverage bin x "
            "box-center-distance bin"
        ),
        "roles": role_reports,
        "selected_rows": len(selected_all),
        "selected_by_role_class": {
            f"{role}:{name}": count
            for (role, name), count in sorted(
                Counter(
                    (r["candidate_role"], r["candidate_answer_class"])
                    for r in selected_all
                ).items()
            )
        },
        "pass": all(
            r["matcher"]["type_signal_removed"] for r in role_reports.values()
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
