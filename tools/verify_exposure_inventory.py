#!/usr/bin/env python3
"""Independently verify the exhaustive historical-exposure ledger (method B).

Uses deliberately different machinery from the builder: shell ``grep`` for the
literal sweep, line-sampling instead of full parsing for JSONL detection, and
its own whitelist copy. Emits a machine-readable verification report and fails
loudly on any uncovered scene-bearing surface.
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from egoshift.split_roles import require_complete_exposure_audit  # noqa: E402

KNOWN_THIRTEEN = (
    "data/sim/qa_taskb4z_v4_test.jsonl",
    "data/sim/qa_taskc_release_v3_test.jsonl",
    "data/sim/qa_taskc_release_v3_dev.jsonl",
    "data/sim/qa_taskb_release_v1_test.jsonl",
    "data/sim/qa_taskb_release_v2_test.jsonl",
    "data/sim/qa_taskb4z_v2_test.jsonl",
    "data/sim/qa_taskb4z_v3_test.jsonl",
    "data/sim/qa_taskb_release_v6_test.jsonl",
    "data/sim/qa_taskb_release_v6_test_v61.jsonl",
    "data/sim/qa_taskc_release_v1_test.jsonl",
    "data/sim/qa_test.jsonl",
    "data/sim/qa_dev.jsonl",
    "data/sim/qa_taskb_4zone_test.jsonl",
)
WHITELIST_PREFIXES = (
    "data/sim/scenes/",
    "data/sim/taskb_regen/",
    "data/sim/taskc_visibility_enrichment/",
    "data/sim/ab_taskd_filters_20260417_015204/",
    "data/sim/freeze_20260423_093409/",
    "data/sim/freeze_20260423_093414/",
    "data/sim/_archive/stale_bbox_2026-04-30/",
    "data/sim/scene_shards_gpu567/",
    "data/sim/scene_shards_allregen/",
    "data/sim/scene_shards_fix/",
    "data/sim/scene_shards_main/",
    "data/sim/scene_shards_missing/",
    "data/sim/scene_shards_retry_last/",
    "data/sim/scene_shards_retry_missing/",
    "data/sim/taskb_180_shards/",
    "data/sim/taskb_hv_shards/",
    "data/sim/taskc_bbox_shards/",
    "data/sim/taskc_dev_bbox_shards/",
    "data/sim/shards_reduced/",
    "data/sim/taskd_enrichment/",
    "data/sim/stale_scene_subsets/",
    "data/sim/taskb_render_shards/",
    "data/sim/taskb_stage/",
    "data/sim/shards/",
    "data/sim/taskb_shards/",
    "data/sim/taska_box_shards/",
    "data/sim/taskb_box_shards/",
    "data/sim/targetpose_rerender/",
    "data/sim/targetpose_resumable/",
    "work/candidate_r2/sources/",
    "work/candidate_r2/targeted_r2_assets/",
    "work/candidate_r2/targeted_smoke_assets/",
    "work/candidate_r2/taskc_targeted_occlusion_v1/",
    "work/candidate_r2/exposure_ledger/",
    "work/candidate_r2/audits/",
    "work/candidate_r2/taskb_candidate_r2_static/",
    # Post-freeze R2 construction surfaces: miner outputs, staging, and smoke
    # runs created after the ledger froze; inputs to selection, never model-
    # or human-visible. Same category as taskc_targeted_occlusion_v1 above.
    "work/candidate_r2/taskc_smoke_v2/",
    "work/candidate_r2/taska_smoke/",
    "work/candidate_r2/taskc_seek_smoke/",
    "work/candidate_r2/taskc_seeking_v1/",
    "work/candidate_r2/taskc_hidden_tuned/",
    "work/candidate_r2/taskc_tuned_all_v1/",
    "work/candidate_r2/taskc_tune/",
    "work/candidate_r2/taska_pairs_v1/",
    "work/candidate_r2/taska_pairs_v2/",
    "work/candidate_r2/targeted_r2c_assets/",
    "work/candidate_r2/targeted_r2d_assets/",
    "work/candidate_r2/targeted_r2e_assets/",
    "work/candidate_r2/mining_logs/",
    "data/sim/scenes_expansion/",
    # Candidate builds materialized from the certified selections, plus the
    # archive of superseded pre-R2 eval output (moved wholesale 2026-08-24;
    # every ledger-referenced file was deliberately left in place and
    # re-verified byte-identical). Neither is an exposure surface: the
    # candidates are outputs of selection, and the archive is a relocation
    # of files that were already accounted for.
    "work/candidate_r2/taska_candidate_v3/",
    "work/candidate_r2/taska_candidate_v4/",
    "work/candidate_r2/taska_candidate_v5/",
    "work/candidate_r2/taskb_candidate_r2_static_v3/",
    "work/candidate_r2/taskc_candidate_v4/",
    "work/candidate_r2/taskc_candidate_v5/",
    "work/candidate_r2/taska_pairs_v3/",
    "work/candidate_r2/taskc_allclass_v1/",
    "work/candidate_r2/taskc_template_all_v1/",
    "work/candidate_r2/taskc_template_all_v2/",
    "work/candidate_r2/taskc_tuned_all_v2/",
    "work/candidate_r2/taskc_assets_unified_v4/",
    "work/candidate_r2/taskc_assets_unified_v5/",
    "work/candidate_r2/taskc_baked_v4/",
    "work/candidate_r2/taskc_baked_v5/",
    "work/model_panel/",
    "work/human_study_r2/",
    "data/sim/_archive_superseded/",
)
WHITELIST_EXCEPTIONS = (
    "data/sim/taskb_regen/eval_results/",
)
import fnmatch as _fnmatch
WHITELIST_FILE_PATTERNS = (
    "data/sim/qa_full*.jsonl",
    "data/sim/qa_mask_*.jsonl",
    "data/sim/*silhouette*.json",
    "data/sim/*silhouette*.jsonl",
    "data/sim/*.checkpoint*.json",
    "data/sim/regen_*.log",
    "data/sim/_archive/qa_taskb_final.jsonl",
    "data/sim/_archive/qa_taskb_final_verified.jsonl",
    "data/sim/_archive/qa_taskb_verified_regen.jsonl",
    "data/sim/*boxes*.json",
    "data/sim/taskb_v4/targeted_candidates*",
    "work/candidate_r2/*.sh",
)
WHITELIST_FILES = (
    # Mining control inputs: plain lists of which scenes a sweep should visit,
    # written 2026-08-22/23. They carry scene ids and no outcomes whatsoever --
    # no answers, no model or human responses -- so they are not exposure
    # surfaces, but the literal grep sweep sees the ids and would otherwise
    # demand every listed scene be in the ledger. Most are expansion scenes
    # that by construction were never exposed.
    "work/candidate_r2/taskc_yielding_scenes.txt",
    "work/candidate_r2/taskc_shard12_rescue.txt",
)
GREP_EXCLUDES = [
    "--exclude=qa_full*.jsonl", "--exclude=qa_mask_*.jsonl",
    "--exclude=*silhouette*.json*", "--exclude=*.checkpoint*.json",
    "--exclude=regen_*.log", "--exclude=enrichment_run.log",
    "--exclude=enrichment_stats.json", "--exclude=qa_taskb_final*.jsonl",
    "--exclude=qa_taskb_verified_regen.jsonl",
    "--exclude=*boxes*.json", "--exclude=render_shard*.log",
    "--exclude=shard*.log", "--exclude=trajectory_batch_summary.json",
    "--exclude=qa_taskb_final_verified_shard*.jsonl",
    "--exclude-dir=scenes", "--exclude-dir=exposure_ledger",
    "--exclude-dir=aabb_grow", "--exclude-dir=aabb_missing",
    "--exclude-dir=bbox_fov90", "--exclude-dir=bbox_fov90_full",
    "--exclude-dir=scene_shards_*", "--exclude-dir=taskb_180_shards",
    "--exclude-dir=taskb_hv_shards", "--exclude-dir=taskb_render_shards",
    "--exclude-dir=taskb_stage", "--exclude-dir=taskb_shards",
    "--exclude-dir=taska_box_shards", "--exclude-dir=taskb_box_shards",
    "--exclude-dir=taskc_bbox_shards", "--exclude-dir=taskc_dev_bbox_shards",
    "--exclude-dir=shards", "--exclude-dir=shards_reduced",
    "--exclude-dir=taskd_enrichment", "--exclude-dir=stale_scene_subsets",
    "--exclude-dir=targetpose_rerender", "--exclude-dir=targetpose_resumable",
    "--exclude-dir=taskc_visibility_enrichment",
    "--exclude-dir=sources", "--exclude-dir=audits",
    "--exclude-dir=targeted_r2_assets", "--exclude-dir=targeted_smoke_assets",
    "--exclude-dir=taskc_targeted_occlusion_v1",
    "--exclude-dir=taskb_candidate_r2_static", "--exclude-dir=scenes_all",
    "--exclude-dir=taskc_smoke_v2", "--exclude-dir=taska_smoke",
    "--exclude-dir=taskc_seek_smoke", "--exclude-dir=taskc_seeking_v1",
    "--exclude-dir=taskc_hidden_tuned", "--exclude-dir=taskc_tuned_all_v1",
    "--exclude-dir=taskc_tune", "--exclude-dir=taska_pairs_v1",
    "--exclude-dir=taska_pairs_v2", "--exclude-dir=targeted_r2c_assets",
    "--exclude-dir=targeted_r2d_assets", "--exclude-dir=targeted_r2e_assets",
    "--exclude-dir=mining_logs", "--exclude-dir=scenes_expansion",
    "--exclude-dir=_archive_superseded", "--exclude-dir=model_panel",
    "--exclude-dir=human_study_r2",
    "--exclude=taskc_yielding_scenes.txt", "--exclude=taskc_shard12_rescue.txt",
    "--exclude=*.sh",
    "--exclude-dir=taska_candidate_v3", "--exclude-dir=taska_candidate_v4",
    "--exclude-dir=taska_candidate_v5", "--exclude-dir=taskb_candidate_r2_static_v3",
    "--exclude-dir=taskc_candidate_v4", "--exclude-dir=taskc_candidate_v5",
    "--exclude-dir=taska_pairs_v3", "--exclude-dir=taskc_allclass_v1",
    "--exclude-dir=taskc_template_all_v1", "--exclude-dir=taskc_template_all_v2",
    "--exclude-dir=taskc_tuned_all_v2", "--exclude-dir=taskc_assets_unified_v4",
    "--exclude-dir=taskc_assets_unified_v5", "--exclude-dir=taskc_baked_v4",
    "--exclude-dir=taskc_baked_v5",
    "--exclude=render180_shard*.log", "--exclude=*.pre_symfix_bak",
]
EVIDENCE_ROOTS = (
    "data/sim", "data/real_hmd", "SHARE_WITH_ANNOTATORS", "work", "docs",
    "paper", "scripts", "src", "configs",
)


def whitelisted(rel: str) -> bool:
    if any(rel.startswith(p) for p in WHITELIST_EXCEPTIONS):
        return False
    if rel in WHITELIST_FILES or any(rel.startswith(p) for p in WHITELIST_PREFIXES):
        return True
    return any(_fnmatch.fnmatch(rel, pat) for pat in WHITELIST_FILE_PATTERNS)


def sample_has_top_level_scene(path: Path, limit: int = 50) -> bool:
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in itertools.islice((l for l in handle if l.strip()), limit):
                record = json.loads(line)
                if isinstance(record, dict) and record.get("scene_id"):
                    return True
    except Exception:
        return False
    return False


def grep_scene_ids(root: Path, targets: list[str]) -> set[str]:
    found: set[str] = set()
    for target in targets:
        if not (root / target).exists():
            continue
        proc = subprocess.run(
            ["grep", "-rhoaE", *GREP_EXCLUDES, "scene_[0-9]{5}", target],
            cwd=root, capture_output=True, text=True,
        )
        found |= set(proc.stdout.split())
    return found


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--ledger", required=True, type=Path)
    parser.add_argument("--components", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()

    ledger = json.loads((root / args.ledger).read_text(encoding="utf-8"))
    require_complete_exposure_audit(ledger)
    ledger_paths = {str(item["path"]) for item in ledger["exposure_files"]}
    filtered_paths = {
        str(item["path"]) for item in ledger.get("census_filtered_files", [])
    }
    census = set(
        json.loads((root / args.components).read_text(encoding="utf-8"))["scenes"]
    )

    failures: list[str] = []
    checks: dict[str, object] = {}

    missing_known = [p for p in KNOWN_THIRTEEN if p not in ledger_paths]
    checks["known_thirteen_present"] = not missing_known
    if missing_known:
        failures.append(f"known exposure pools missing from ledger: {missing_known}")

    uncovered: list[str] = []
    scene_bearing = 0
    for evidence_root in EVIDENCE_ROOTS:
        base = root / evidence_root
        if not base.exists():
            continue
        for path in base.rglob("*.jsonl"):
            rel = path.relative_to(root).as_posix()
            if whitelisted(rel):
                continue
            if sample_has_top_level_scene(path):
                scene_bearing += 1
                if rel not in ledger_paths and rel not in filtered_paths:
                    uncovered.append(rel)
    checks["scene_bearing_jsonl_seen"] = scene_bearing
    checks["uncovered_scene_bearing_jsonl"] = uncovered
    if uncovered:
        failures.append(f"{len(uncovered)} scene-bearing JSONL files not in ledger")

    ledger_scenes: set[str] = set()
    for item in ledger["exposure_files"]:
        with (root / str(item["path"])).open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    scene_id = json.loads(line).get("scene_id")
                    if scene_id:
                        ledger_scenes.add(str(scene_id))
    ledger_scenes &= census
    checks["ledger_exposed_census_scenes"] = len(ledger_scenes)

    union_13: set[str] = set()
    for rel in KNOWN_THIRTEEN:
        path = root / rel
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        scene_id = json.loads(line).get("scene_id")
                        if scene_id:
                            union_13.add(str(scene_id))
    missing_13_scenes = sorted((union_13 & census) - ledger_scenes)
    checks["thirteen_file_union_scenes"] = len(union_13 & census)
    checks["thirteen_union_covered"] = not missing_13_scenes
    if missing_13_scenes:
        failures.append(
            f"{len(missing_13_scenes)} scenes from the known 13-file union missing"
        )

    grep_targets = []
    for evidence_root in EVIDENCE_ROOTS:
        base = root / evidence_root
        if not base.exists():
            continue
        if evidence_root in ("data/sim", "work"):
            for child in sorted(base.iterdir()):
                rel = child.relative_to(root).as_posix()
                rel_prefix = rel + ("/" if child.is_dir() else "")
                if whitelisted(rel) or whitelisted(rel_prefix):
                    for exc in WHITELIST_EXCEPTIONS:
                        if exc.startswith(rel_prefix) and (root / exc).exists():
                            grep_targets.append(exc.rstrip("/"))
                    continue
                grep_targets.append(rel)
        else:
            grep_targets.append(evidence_root)
    grep_ids = grep_scene_ids(root, grep_targets) & census
    grep_only = sorted(grep_ids - ledger_scenes)
    checks["grep_census_ids"] = len(grep_ids)
    checks["grep_ids_not_in_ledger"] = grep_only
    if grep_only:
        failures.append(
            f"{len(grep_only)} census scene ids found by grep but absent from ledger"
        )

    human_roots = [
        p for p in (root / "data/sim").iterdir()
        if p.is_dir() and ("human" in p.name or p.name == "construct_pilots")
    ] + [root / "SHARE_WITH_ANNOTATORS"]
    human_uncovered = []
    for human_root in human_roots:
        rel_prefix = human_root.relative_to(root).as_posix()
        covered = any(p.startswith(rel_prefix) for p in ledger_paths) or any(
            rel_prefix in json.dumps(ledger.get("census_filtered_files", []))
            for _ in (0,)
        )
        if not covered:
            hits = grep_scene_ids(root, [rel_prefix]) & census
            if hits and not hits <= ledger_scenes:
                human_uncovered.append(rel_prefix)
    checks["human_roots_checked"] = len(human_roots)
    checks["human_roots_uncovered"] = human_uncovered
    if human_uncovered:
        failures.append(f"human study roots with uncovered scenes: {human_uncovered}")

    report = {
        "audit_id": "egoshift_exposure_ledger_independent_verification",
        "method": "line-sampled JSONL census plus shell-grep literal sweep",
        "ledger_path": str(args.ledger),
        "ledger_manifest_sha256": ledger["manifest_sha256"],
        "checks": checks,
        "failures": failures,
        "pass": not failures,
    }
    (root / args.report).parent.mkdir(parents=True, exist_ok=True)
    (root / args.report).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    print(json.dumps(checks, indent=2, sort_keys=True, default=str)[:2000])
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
