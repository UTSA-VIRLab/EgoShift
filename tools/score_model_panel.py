#!/usr/bin/env python3
"""Score model panel results against private gold, with the controls in view.

Reports accuracy per model and arm with a scene-clustered bootstrap interval,
plus refusal/parse-failure and error rates, because a model that fails to
answer 20% of items is not comparable to one that answers all of them at the
same accuracy.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_gold(candidate_root: Path, task: str, split: str) -> tuple[dict, dict]:
    t = task.lower()
    if task == "A":
        rows = [json.loads(l) for l in (candidate_root / f"private/task_a_{split}_gold.jsonl").read_text().splitlines() if l.strip()]
        gold = {r["id"]: r["answer"] for r in rows}
        scene = {r["id"]: r.get("scene_id") for r in rows}
    else:
        gpath = candidate_root / f"private/gold/task_{t}_{split}.jsonl"
        ppath = candidate_root / f"private/provenance/task_{t}_{split}.jsonl"
        gold = {r["id"]: r["answer"] for r in (json.loads(l) for l in gpath.read_text().splitlines() if l.strip())}
        scene = {r["id"]: r.get("scene_id") for r in (json.loads(l) for l in ppath.read_text().splitlines() if l.strip())} if ppath.exists() else {}
    return gold, scene


def bootstrap(items: list[tuple[str, bool]], clusters: dict, draws: int = 2000) -> tuple[float, float]:
    by_cluster = defaultdict(list)
    for item_id, hit in items:
        by_cluster[clusters.get(item_id, item_id)].append(hit)
    keys = list(by_cluster)
    rng = random.Random(20260824)
    means = []
    for _ in range(draws):
        pool = [h for _ in keys for h in by_cluster[rng.choice(keys)]]
        if pool:
            means.append(sum(pool) / len(pool))
    means.sort()
    return (means[int(0.025 * len(means))], means[int(0.975 * len(means))]) if means else (0.0, 0.0)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=Path("work/model_panel/results.jsonl"))
    p.add_argument("--candidate-root", required=True, type=Path)
    p.add_argument("--task", required=True, choices=("A", "B", "C"))
    p.add_argument("--split", default="hidden_test")
    p.add_argument("--out", type=Path, default=Path("work/model_panel/scores.json"))
    args = p.parse_args()

    gold, scene = load_gold(args.candidate_root, args.task, args.split)
    records = [json.loads(l) for l in args.results.read_text().splitlines() if l.strip()]
    records = [r for r in records if r["task"] == args.task and r["split"] == args.split]

    grouped = defaultdict(list)
    for r in records:
        grouped[(r["model"], r["arm"])].append(r)

    report = {}
    for (model, arm), rows in sorted(grouped.items()):
        answered = [r for r in rows if r["parsed"] is not None]
        hits = [(r["item_id"], r["parsed"] == gold.get(r["item_id"])) for r in answered]
        acc = sum(h for _, h in hits) / len(hits) if hits else None
        lo, hi = bootstrap(hits, scene) if hits else (None, None)
        report[f"{model} | {arm}"] = {
            "model": model, "arm": arm, "tier": rows[0]["tier"],
            "items": len({r["item_id"] for r in rows}),
            "responses": len(rows),
            "answered": len(answered),
            "errors": sum(1 for r in rows if r["error"]),
            "unparsed": sum(1 for r in rows if r["error"] is None and r["parsed"] is None),
            "accuracy": round(acc, 4) if acc is not None else None,
            "ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(f"{'model | arm':52s} {'acc':>7s} {'95% CI':>16s} {'ans':>5s} {'err':>4s} {'unp':>4s}")
    for key, r in sorted(report.items(), key=lambda kv: -(kv[1]["accuracy"] or 0)):
        ci = f"[{r['ci95'][0]:.3f},{r['ci95'][1]:.3f}]" if r["ci95"] else "n/a"
        acc = f"{r['accuracy']:.4f}" if r["accuracy"] is not None else "n/a"
        print(f"{key:52s} {acc:>7s} {ci:>16s} {r['answered']:>5d} {r['errors']:>4d} {r['unparsed']:>4d}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
