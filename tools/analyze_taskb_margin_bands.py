#!/usr/bin/env python3
"""Accuracy by boundary-margin band for Task B.

The 15-degree whole-box margin is a frozen construction gate: every released
item keeps all eight 3-D bounding-box corners inside one answer zone with at
least that clearance. Raising it after the fact would have cost the benchmark
its balance -- the 180-degree cells fall from 28 items to 5 at a 25-degree gate
and 2 at 30 -- and would read as post-hoc easing. Reporting accuracy by band
answers the same question with more information and no pool loss:

  * if accuracy is flat across bands, the 15-degree gate is vindicated
    empirically rather than by assertion;
  * if it falls near the boundary, that is a graded-precision finding, and a
    limitation to state plainly rather than one to engineer away.

Works on model results, human responses, or both, so the same bands appear on
both sides of the comparison.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

BANDS = ((15, 20), (20, 25), (25, 30), (30, 999))


def band_of(margin: float) -> str:
    for low, high in BANDS:
        if low <= margin < high:
            return f"{low}-{high}deg" if high != 999 else f"{low}deg+"
    return "unknown"


def wilson(correct: int, total: int) -> tuple[float, float]:
    """Wilson score interval: behaves at small n, where a band may land."""

    if not total:
        return (0.0, 0.0)
    z = 1.959963985
    p = correct / total
    d = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / d
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-root", type=Path,
                    default=Path("work/candidate_r2/taskb_candidate_r2_static_v3"))
    ap.add_argument("--split", default="hidden_test")
    ap.add_argument("--model-results", nargs="*", type=Path,
                    default=[Path("work/model_panel/results_local.jsonl")])
    ap.add_argument("--human-scored", type=Path,
                    default=Path("work/human_study_r2/scored/majority_vote.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("work/model_panel/taskb_margin_bands.json"))
    args = ap.parse_args()

    prov = {r["id"]: r for r in (json.loads(l) for l in
            (args.candidate_root / f"private/provenance/task_b_{args.split}.jsonl")
            .read_text().splitlines() if l.strip())}
    gold = {r["id"]: r["answer"] for r in (json.loads(l) for l in
            (args.candidate_root / f"private/gold/task_b_{args.split}.jsonl")
            .read_text().splitlines() if l.strip())}
    margin = {}
    for item_id, r in prov.items():
        wb = (r.get("geometry_evidence") or {}).get("whole_box") or {}
        value = wb.get("minimum_corner_boundary_margin_degrees")
        if value is not None:
            margin[item_id] = float(value)

    pool = defaultdict(int)
    for item_id, value in margin.items():
        pool[band_of(value)] += 1

    per_model: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for path in args.model_results:
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("task") != "B" or r.get("split") != args.split:
                continue
            if r.get("parsed") is None or r["item_id"] not in margin:
                continue
            cell = per_model[r["model"]][band_of(margin[r["item_id"]])]
            cell[1] += 1
            if r["parsed"] == gold.get(r["item_id"]):
                cell[0] += 1

    human = {}
    if args.human_scored.exists():
        payload = json.loads(args.human_scored.read_text())
        for item_id, entry in (payload.get("B", {}).get("per_item") or {}).items():
            if item_id in margin:
                cell = human.setdefault(band_of(margin[item_id]), [0, 0])
                cell[1] += 1
                cell[0] += 1 if entry.get("correct") else 0

    bands = [band_of(low + 0.1) for low, _ in BANDS]
    report = {"pool_by_band": dict(pool), "models": {}, "human_majority": {}}
    print(f"Task B {args.split}: {len(margin)} items with a recorded whole-box margin")
    print("pool by band:", {b: pool.get(b, 0) for b in bands})
    print()
    header = "model".ljust(24) + "".join(b.rjust(14) for b in bands)
    print(header)
    for model, per_band in sorted(per_model.items()):
        cells = []
        for b in bands:
            correct, total = per_band.get(b, [0, 0])
            cells.append(f"{correct/total:.3f} (n{total})".rjust(14) if total else "—".rjust(14))
            if total:
                lo, hi = wilson(correct, total)
                report["models"].setdefault(model, {})[b] = {
                    "accuracy": round(correct / total, 4), "n": total,
                    "ci95": [round(lo, 4), round(hi, 4)]}
        print(model.ljust(24) + "".join(cells))
    if human:
        cells = []
        for b in bands:
            correct, total = human.get(b, [0, 0])
            cells.append(f"{correct/total:.3f} (n{total})".rjust(14) if total else "—".rjust(14))
            if total:
                lo, hi = wilson(correct, total)
                report["human_majority"][b] = {"accuracy": round(correct / total, 4),
                                               "n": total, "ci95": [round(lo, 4), round(hi, 4)]}
        print()
        print("human majority".ljust(24) + "".join(cells))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
