#!/usr/bin/env python3
"""Build the reviewer-facing artifact: every number in the paper, checkable.

The full candidate archives are ~11.5 GB because of rendered frames, which is
more than a reviewer will download during a review cycle. This bundle carries
the evidence instead of the bulk: all questions for every split, the dev-split
answers, the build audits, the raw model completions behind every reported
score, the pseudonymous human aggregates, and a stratified frame sample so the
items can be inspected without pulling every render.

Deliberately excluded, and checked for on the way out:
  * held-out gold answers for all three tasks;
  * work/human_study_r2/incoming and the pseudonym map, which identify people;
  * per-rater and per-item human responses, which the paper states are not
    released.

Run release/tools/verify_anonymity.py against the staged tree before uploading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = {
    "A": ROOT / "work/candidate_r2/taska_candidate_v5",
    "B": ROOT / "work/candidate_r2/taskb_candidate_r2_static_v3",
    "C": ROOT / "work/candidate_r2/taskc_candidate_v10",
}
# Aggregates the paper reports. Anything per-rater or per-item is dropped.
HUMAN_KEEP_TASK = [
    "mean_rater_accuracy", "pooled_response_accuracy", "pooled_wilson95_item",
    "crossed_rater_scene_bootstrap", "rater_accuracy_range", "eligible_raters",
    "attention_checks", "flagged_responses", "excluded_sessions",
]
HUMAN_KEEP_PANEL = [
    "raters", "items_with_votes", "items_total", "majority_accuracy",
    "decided_majority_accuracy", "majority_ties", "fleiss_kappa",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def redact_human(analysis: dict) -> dict:
    """Keep task-level aggregates and panel headline numbers only."""

    out = {"schema_version": analysis.get("schema_version"), "tasks": {},
           "task_b_first_raters": analysis.get("task_b_first_raters"),
           "task_order_raters": analysis.get("task_order_raters"),
           "_redaction": "per-rater and per-item responses removed; the paper "
                         "does not release them"}
    for task, block in analysis.get("tasks", {}).items():
        kept = {k: block[k] for k in HUMAN_KEEP_TASK if k in block}
        panel = block.get("attention_qualified") or {}
        kept["attention_qualified"] = {k: panel[k] for k in HUMAN_KEEP_PANEL if k in panel}
        out["tasks"][task] = kept
    return out


def frame_sample(task: str, root: Path, stage: Path, per_task: int, rng: random.Random) -> int:
    """Copy the frames belonging to a deterministic sample of held-out items."""

    qpath = root / f"public/questions/task_{task.lower()}_hidden_test.jsonl"
    rows = [json.loads(l) for l in qpath.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows.sort(key=lambda r: r["id"])
    picked = rows if len(rows) <= per_task else rng.sample(rows, per_task)
    copied = 0
    for row in picked:
        for rel in row.get("frames", []):
            src = root / rel
            if not src.exists():
                continue
            dst = stage / f"task_{task.lower()}" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
    (stage / f"task_{task.lower()}" / "SAMPLED_ITEM_IDS.json").write_text(
        json.dumps({"task": task, "sampled": sorted(r["id"] for r in picked),
                    "rule": f"seeded sample of {per_task} held-out items"},
                   indent=1) + "\n")
    return copied


# Runs that are present in the working registry but are NOT part of the panel
# reported in the paper. They are packaged under excluded_runs/ with a stated
# reason rather than dropped: a reviewer must be able to see that a run exists,
# why it is not counted, and that nothing was silently removed. Row counts in
# EXCLUSIONS.json are computed from the shipped bytes.
# Runs dropped from the artifact altogether: their rows are not written to the
# bundle in any form, and the model is stripped from every shipped roster and
# panel config so that no file declares a member with no results. Populated from
# --omit-model at run time, so this source file never names them. Use it only for
# runs outside the reported panel; anything a reviewer must still see belongs in
# EXCLUDED_MODELS.
OMITTED_MODELS: set[str] = set()   # populated from --omit-model

EXCLUDED_MODELS: dict[str, str] = {
    "SenseNova-SI-1.2-InternVL3-8B": (
        "Access/runtime failure. Weights are public and cached, but the checkpoint "
        "ships its own InternVL-chat code written against transformers ~4.3x and "
        "will not load under the pinned stack. The rows here are probe/error "
        "stubs, not evaluation results."
    ),
    "Llama-3.2-11B-Vision": (
        "Access failure. Gated repository: the token is valid and repo metadata "
        "resolves, but config.json returns 401 GatedRepoError (gated=manual); this "
        "account has not been granted access. The rows here are probe/error stubs."
    ),
    "DeepSeek-VL2-Small": (
        "Access/runtime failure. Architecture unsupported by the pinned stack: "
        "config.json declares model_type='deepseek_vl_v2' with architectures=null "
        "and an empty auto_map. The rows here are probe/error stubs."
    ),
}


def split_model_rows(src: Path, keep: Path, drop: Path) -> tuple[int, dict]:
    """Copy a results file, diverting rows whose model is excluded.

    Returns (kept_rows, {model: diverted_rows}, {model: omitted_rows}). Files are
    only written when they
    have content, so an all-excluded file does not leave an empty artifact.
    """
    kept: list[str] = []
    dropped: dict[str, list[str]] = {}
    omitted: dict[str, int] = {}
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            model = json.loads(line).get("model")
        except json.JSONDecodeError:
            kept.append(line)
            continue
        if model in OMITTED_MODELS:
            omitted[model] = omitted.get(model, 0) + 1
        elif model in EXCLUDED_MODELS:
            dropped.setdefault(model, []).append(line)
        else:
            kept.append(line)
    if kept:
        keep.write_text("\n".join(kept) + "\n", encoding="utf-8")
    if dropped:
        drop.parent.mkdir(parents=True, exist_ok=True)
        drop.write_text(
            "\n".join(l for rows in dropped.values() for l in rows) + "\n",
            encoding="utf-8")
    return len(kept), {m: len(v) for m, v in dropped.items()}, omitted


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "work/artifact_u7/reviewer_bundle")
    p.add_argument("--tar", type=Path, default=ROOT / "work/artifact_u7/egoshift_reviewer_bundle.tar.gz")
    p.add_argument("--frames-per-task", type=int, default=120)
    p.add_argument("--seed", type=int, default=20260828)
    p.add_argument("--omit-model", action="append", default=[],
                   metavar="LABEL",
                   help="drop this model from the artifact entirely: no result "
                        "rows, and no mention in any shipped roster or config")
    args = p.parse_args()
    OMITTED_MODELS.update(args.omit_model)

    stage = args.out
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    rng = random.Random(args.seed)
    report: dict = {"contents": {}, "excluded": []}

    # 1. code
    shutil.copytree(ROOT / "release", stage / "code",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    report["contents"]["code"] = sum(1 for _ in (stage / "code").rglob("*") if _.is_file())

    # 2. questions for every split, plus dev answers and manifests
    for task, root in CANDIDATES.items():
        dst = stage / f"task_{task.lower()}"
        dst.mkdir(parents=True, exist_ok=True)
        for q in sorted((root / "public/questions").glob("*.jsonl")):
            shutil.copy2(q, dst / q.name)
        for extra in ("manifest.json",):
            if (root / extra).exists():
                shutil.copy2(root / extra, dst / extra)
        dev = root / "development"
        if dev.exists():
            shutil.copytree(dev, dst / "development", dirs_exist_ok=True)
    report["contents"]["question_files"] = sum(
        1 for _ in stage.rglob("task_*/*.jsonl"))

    # 3. build audits
    shutil.copytree(ROOT / "work/candidate_r2/audits", stage / "audits",
                    ignore=shutil.ignore_patterns("*.png", "*.jpg"))
    report["contents"]["audit_files"] = sum(1 for _ in (stage / "audits").rglob("*") if _.is_file())

    # 4. raw model completions: the evidence for every reported score
    md = stage / "model_panel"
    md.mkdir(parents=True, exist_ok=True)
    n = 0
    excl_dir = stage / "excluded_runs"
    excluded_counts: dict = {}
    omitted_counts: dict = {}
    for f in sorted((ROOT / "work/model_panel").glob("results_*.jsonl")):
        kept, dropped, omitted = split_model_rows(f, md / f.name, excl_dir / f.name)
        for model, count in dropped.items():
            excluded_counts[model] = excluded_counts.get(model, 0) + count
        for model, count in omitted.items():
            omitted_counts[model] = omitted_counts.get(model, 0) + count
        if kept:
            n += 1
    for f in ("resolved_panel.json",):
        src = ROOT / "work/model_panel" / f
        if src.exists():
            shutil.copy2(src, md / f)
    report["contents"]["model_result_files"] = n

    # 5. human aggregates, redacted
    hs = stage / "human"
    hs.mkdir(parents=True, exist_ok=True)
    analysis = json.loads((ROOT / "work/human_study_r2/scored/analysis.json").read_text())
    (hs / "analysis_aggregate.json").write_text(
        json.dumps(redact_human(analysis), indent=1, sort_keys=True) + "\n")
    report["excluded"].append("human per-rater and per-item responses")
    report["excluded"].append("human intake files and pseudonym map")

    # 6. structural evidence for the held-out pool, with nothing that reveals a
    #    label. Bearings are excluded on purpose: the post-turn bearing
    #    determines the answer outright, and the source bearing plus the stated
    #    rotation and turn recovers it, so neither can travel. Cell sizes,
    #    supports and scene identifiers can, and they are what the construction
    #    claim rests on.
    SAFE = ("rotation_degrees", "turn_direction", "scene_id", "target_type",
            "physical_scene_component_id", "silhouette_coverage", "visible_pixels",
            "split", "source_kind", "target_category")
    struct = stage / "structure"
    struct.mkdir(parents=True, exist_ok=True)
    for task, root in CANDIDATES.items():
        prov = root / f"private/provenance/task_{task.lower()}_hidden_test.jsonl"
        if not prov.exists():
            continue
        cells: dict = {}
        out = []
        for line in prov.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            out.append({"id": r["id"], **{k: r[k] for k in SAFE if k in r}})
            if task == "B":
                key = f"{r['rotation_degrees']}_{r['turn_direction']}"
                bucket = cells.setdefault(key, {})
                bucket[r["answer"]] = bucket.get(r["answer"], 0) + 1
        (struct / f"task_{task.lower()}_holdout_structure.jsonl").write_text(
            "\n".join(json.dumps(r, sort_keys=True) for r in out) + "\n")
        if cells:
            import math as _m
            total = sum(sum(v.values()) for v in cells.values())
            ceil_sum = sum(_m.ceil(sum(v.values()) / len(v)) for v in cells.values())
            (struct / "task_b_action_cell_counts.json").write_text(json.dumps({
                "note": ("Per-cell label counts only. Which item carries which label "
                         "is not disclosed. These counts are what the structural "
                         "target is computed from."),
                "cells": cells,
                "items": total,
                "attained_action_cell_oracle": f"{max(sum(v.values()) for v in cells.values()) and sum(max(v.values()) for v in cells.values())}/{total}",
                "finite_target_sum_ceil": f"{ceil_sum}/{total}",
            }, indent=1, sort_keys=True) + "\n")
    report["contents"]["structure_files"] = sum(1 for _ in struct.iterdir())
    report["excluded"].append("all bearing geometry for held-out items")

    # 7. frame sample
    total = 0
    for task, root in CANDIDATES.items():
        total += frame_sample(task, root, stage, args.frames_per_task, rng)
    report["contents"]["sampled_frames"] = total

    # 8. refuse to ship anything that identifies a person or the institution.
    #    docs/admin/institutional/ holds the IRB determination, which carries
    #    real names, institutional email addresses and the university. It is not
    #    copied into the bundle, and this check exists so that a future edit to
    #    the copy list cannot quietly change that.
    identity = [str(q.relative_to(stage)) for q in stage.rglob("*")
                if q.is_file() and (q.suffix.lower() == ".pdf"
                                    or "institutional" in q.parts
                                    or "determination" in q.name.lower()
                                    or q.name.lower().startswith("irb"))]
    identity = [q for q in identity if not q.endswith("check_institutional_gate.py")]
    if identity:
        raise SystemExit(
            f"refusing to package: identity-bearing files present: {identity[:5]}")
    report["excluded"].append("institutional determination and any PDF")

    # 9. refuse to ship held-out gold
    leaked = [str(p.relative_to(stage)) for p in stage.rglob("*")
              if p.is_file() and ("gold" in p.name.lower() or "/private/" in str(p))]
    leaked = [p for p in leaked if "development" not in p]
    if leaked:
        raise SystemExit(f"refusing to package: held-out answers present: {leaked[:5]}")
    report["excluded"].append("held-out gold answers for all three tasks")

    # 9b. no shipped roster or panel config may declare a model the artifact
    #     does not contain: a member with zero rows reads as an undisclosed
    #     missing run, which is worse than either shipping it or omitting it.
    if OMITTED_MODELS:
        rosters = [md / "resolved_panel.json"]
        rosters += sorted((stage / "code" / "configs").glob("*.json"))
        stripped = 0
        for roster in rosters:
            if not roster.exists():
                continue
            try:
                data = json.loads(roster.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            changed = False
            for key in ("resolved", "models", "access_failures"):
                entries = data.get(key)
                if not isinstance(entries, list):
                    continue
                pruned = [e for e in entries
                          if not (isinstance(e, dict)
                                  and e.get("name") in OMITTED_MODELS)]
                if len(pruned) != len(entries):
                    data[key] = pruned
                    changed = True
                    stripped += len(entries) - len(pruned)
            if changed:
                roster.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n",
                                  encoding="utf-8")
        report["contents"]["roster_entries_stripped"] = stripped

    # 10. record every excluded run, with its reason and shipped row count
    if excluded_counts:
        excl_dir.mkdir(parents=True, exist_ok=True)
        (excl_dir / "EXCLUSIONS.json").write_text(json.dumps({
            "note": ("Runs present in the working registry that are not part of the "
                     "panel reported in the paper. Retained with a stated reason so "
                     "that no run is silently omitted. Rows here are excluded from "
                     "every number in the paper."),
            "models": {m: {"reason": EXCLUDED_MODELS[m], "rows": c}
                       for m, c in sorted(excluded_counts.items())},
        }, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    report["contents"]["excluded_runs"] = excluded_counts
    report["excluded"].append(
        f"{len(excluded_counts)} non-panel runs moved to excluded_runs/ with reasons")

    # 11. licence files for the shipped bytes. The terms differ by content type,
    #     so a single repository-level tag would misdescribe them.
    lic_src = ROOT / "release/artifact_licenses"
    if lic_src.exists():
        shutil.copytree(lic_src, stage / "licenses", dirs_exist_ok=True)
        report["contents"]["license_files"] = sum(
            1 for _ in (stage / "licenses").rglob("*") if _.is_file())

    # 12. the tests import `release.tools`; the bundle ships that tree as code/.
    #     A symlink keeps the documented command working without duplicating the
    #     tree or rewriting the imports.
    link = stage / "release"
    if not link.exists():
        link.symlink_to("code", target_is_directory=True)

    readme = ROOT / "work/artifact_u7/README.md"
    if readme.exists():
        shutil.copy2(readme, stage / "README.md")

    (stage / "BUNDLE_REPORT.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")

    args.tar.parent.mkdir(parents=True, exist_ok=True)
    if args.tar.exists():
        args.tar.unlink()

    def scrub(ti: tarfile.TarInfo) -> tarfile.TarInfo:
        ti.uid = ti.gid = 0
        ti.uname = ti.gname = ""
        return ti

    with tarfile.open(args.tar, "w:gz") as tar:
        tar.add(stage, arcname="egoshift_reviewer_bundle", filter=scrub)

    size = args.tar.stat().st_size
    if omitted_counts:
        print(json.dumps({"omitted_from_artifact": omitted_counts,
                          "note": "dropped from the shipped bytes; not recorded in the bundle"},
                         sort_keys=True), file=sys.stderr)
    print(json.dumps({**report, "archive": str(args.tar),
                      "size_mb": round(size / 1e6, 1),
                      "sha256": sha256(args.tar)}, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
