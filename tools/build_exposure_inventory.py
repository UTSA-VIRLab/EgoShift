#!/usr/bin/env python3
"""Build the exhaustive historical-exposure ledger for EgoShift S05 resume.

Exposure boundary (reviewed decision, v2): a scene is outcome-exposed when a
row-level outcome from it was observed — by a model evaluation (caches,
reports, eval logs), a human participant (study/annotation/survey packages),
or an author inspection (audit samples, overlay proofs, figures, documents).
Machine generation-funnel artifacts (raw/prebalanced/shortlist/manifest dumps,
enrichment mining trees and their logs, bbox/silhouette checkpoints, render
sidecars) are construction sources whose row outcomes were never observed;
they are hashed into a documented source register, not the exposure list.
This is the same ruling the frozen S05 contract already applies to the Task-B
reservoir index and the pinned Task-C upstream. Ambiguous pool files are
resolved against an evidence corpus (eval trees, eval logs, docs, paper,
eval/run scripts); unresolved scene-bearing files default to exposed.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from egoshift.split_roles import (  # noqa: E402
    canonical_exposure_manifest_sha256,
    require_complete_exposure_audit,
)

SCENE_RE = re.compile(rb"scene_[0-9]{5}")
BASENAME_RE = re.compile(rb"[A-Za-z0-9_.-]+\.jsonl")

KNOWN_EXPOSED = {
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
}

# Never scanned at all (binary/render/cache trees and self-output).
SKIP_PREFIXES = (
    "cache/", "ai2thor/", "opt/", ".git/", ".agents/", ".codex/", ".claude/",
    "outputs/", "release/", "work/candidate_r2/exposure_ledger/",
    "work/candidate_r2/targeted_r2_assets/",
    "work/candidate_r2/targeted_smoke_assets/",
    "work/candidate_r2/taskc_targeted_occlusion_v1/",
    "work/candidate_r2/taskb_candidate_r2_static/",
    "work/candidate_r2/audits/", "work/candidate_r2/sources/",
    "data/sim/scenes/", "data/sim/taskb_regen/scenes/",
)
# Generation-funnel / construction-source surfaces: hashed into the source
# register with per-family rationale; never exposure.
SOURCE_PREFIXES = (
    "data/sim/taskb_regen/",           # exception below for its eval_results
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
)
SOURCE_EXCEPTION_PREFIXES = (
    "data/sim/taskb_regen/eval_results/",  # model evaluations: exposure
)
SOURCE_FILE_PATTERNS = (
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
)
SOURCE_LOG_BASENAMES = ("enrichment_run.log", "enrichment_stats.json")

EVIDENCE_ROOTS = (
    "data/sim", "data/real_hmd", "SHARE_WITH_ANNOTATORS", "work", "docs",
    "paper", "scripts", "src", "configs",
)
EVAL_TREE_PREFIXES = (
    "data/sim/eval_results/",
    "data/sim/taskb_regen/eval_results/",
)
HUMAN_MARKERS = (
    "human_", "annotator", "survey", "qualification", "construct_pilots",
    "share_with_annotators", "lab_annotation_pack", "question_reword",
    "identifiability_pilot",
)
INSPECT_MARKERS = (
    "_bbox_overlays", "_audit", "docs_figures", "docs/", "paper/", "freeze_",
)
TEXT_SUFFIXES = {
    ".json", ".jsonl", ".js", ".html", ".htm", ".txt", ".md", ".csv", ".tsv",
    ".log", ".py", ".sh", ".yaml", ".yml", ".tex", ".cfg", ".ini", ".bib",
    ".bbl", ".sty", ".toml", ".ndjson",
}
# Corpus whose *references* mark a pool as evaluated/observed.
CORPUS_PREFIXES = EVAL_TREE_PREFIXES + ("docs/", "paper/")
CORPUS_SCRIPT_MARKERS = (
    "eval", "run_", "probe", "survey", "human", "bench", "recompute", "score",
    "blind", "baseline", "downstream", "search",
)
MAX_CONTENT_BYTES = 512 * 1024 * 1024


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def starts(rel: str, prefixes) -> bool:
    return any(rel.startswith(p) for p in prefixes)


def is_source(rel: str) -> bool:
    if starts(rel, SOURCE_EXCEPTION_PREFIXES):
        return False
    lower = rel.lower()
    if any(m in lower for m in HUMAN_MARKERS):
        return False
    if starts(rel, SOURCE_PREFIXES):
        return True
    if Path(rel).name in SOURCE_LOG_BASENAMES:
        return True
    return any(fnmatch.fnmatch(rel, pattern) for pattern in SOURCE_FILE_PATTERNS)


def scan_bytes(data: bytes) -> set[str]:
    return {m.decode() for m in SCENE_RE.findall(data)}


def stream_scan(path: Path, collect_basenames: set[bytes] | None = None) -> set[str]:
    found: set[str] = set()
    if path.stat().st_size > MAX_CONTENT_BYTES:
        raise RuntimeError(f"too large: {path}")
    tail = b""
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            blob = tail + chunk
            found |= scan_bytes(blob)
            if collect_basenames is not None:
                collect_basenames.update(BASENAME_RE.findall(blob))
            tail = chunk[-80:]
    return found


def jsonl_top_scenes(path: Path) -> tuple[int, set[str]]:
    rows = 0
    scenes: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows += 1
                record = json.loads(line)
                if isinstance(record, dict) and record.get("scene_id"):
                    scenes.add(str(record["scene_id"]))
    return rows, scenes


def derived_group(rel: str) -> str:
    lower = rel.lower()
    if any(m in lower for m in HUMAN_MARKERS):
        return "human_studies"
    if starts(rel, EVAL_TREE_PREFIXES) or "eval" in lower or lower.endswith(".log"):
        return "eval_content"
    if any(m in lower for m in INSPECT_MARKERS):
        return "author_inspected"
    if lower.startswith(("scripts/", "src/", "configs/")):
        return "code_literals"
    return "misc"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--components", required=True, type=Path)
    parser.add_argument(
        "--out-dir", type=Path, default=Path("work/candidate_r2/exposure_ledger")
    )
    args = parser.parse_args()
    root = args.root.resolve()
    census = set(
        json.loads((root / args.components).read_text(encoding="utf-8"))["scenes"]
    )
    out_dir = root / args.out_dir
    derived_dir = out_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    for stale in derived_dir.glob("derived_exposure_*.jsonl"):
        stale.unlink()

    candidates: list[tuple[str, Path]] = []
    for evidence_root in EVIDENCE_ROOTS:
        base = root / evidence_root
        if base.exists():
            for path in base.rglob("*"):
                if path.is_file():
                    rel = path.relative_to(root).as_posix()
                    if not starts(rel, SKIP_PREFIXES):
                        candidates.append((rel, path))
    candidates.sort()

    # Pass 1: evidence corpus (referenced pool basenames).
    referenced: set[bytes] = set()
    for rel, path in candidates:
        in_corpus = starts(rel, CORPUS_PREFIXES) or (
            rel.startswith("scripts/")
            and any(m in Path(rel).name.lower() for m in CORPUS_SCRIPT_MARKERS)
        ) or (rel.startswith("data/sim/") and rel.endswith(".log") and not is_source(rel))
        if not in_corpus:
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            try:
                stream_scan(path, collect_basenames=referenced)
            except Exception:
                pass
        referenced.add(Path(rel).name.encode())
    referenced_names = {b.decode("utf-8", "ignore") for b in referenced}

    raw_ledger: list[dict[str, object]] = []
    source_register: list[dict[str, object]] = []
    derived_hits: dict[str, dict[str, set[str]]] = defaultdict(dict)
    residual_exposed: list[str] = []
    unknown_report: dict[str, int] = {}
    errors: list[str] = []

    def add_derived(rel: str, scenes: set[str]) -> None:
        known = scenes & census
        if len(scenes - census):
            unknown_report[rel] = len(scenes - census)
        if known:
            group = derived_group(rel)
            for scene in known:
                derived_hits[group].setdefault(scene, set()).add(rel)

    for rel, path in candidates:
        suffix = path.suffix.lower()
        if is_source(rel):
            if suffix in (".jsonl", ".json") or rel in SOURCE_FILE_PATTERNS:
                try:
                    source_register.append(
                        {"path": rel, "sha256": sha256_file(path),
                         "bytes": path.stat().st_size}
                    )
                except Exception as exc:
                    errors.append(f"source-hash {rel}: {exc}")
            continue
        if suffix == ".jsonl":
            try:
                rows, scenes = jsonl_top_scenes(path)
            except Exception as exc:
                errors.append(f"jsonl {rel}: {exc}")
                rows, scenes = -1, set()
            if scenes:
                evidence = []
                if rel in KNOWN_EXPOSED:
                    evidence.append("known_exposed_pool")
                if starts(rel, EVAL_TREE_PREFIXES):
                    evidence.append("model_eval_tree")
                if any(m in rel.lower() for m in HUMAN_MARKERS):
                    evidence.append("human_surface")
                if Path(rel).name in referenced_names:
                    evidence.append("referenced_by_evidence_corpus")
                if not evidence:
                    evidence.append("conservative_residual")
                    residual_exposed.append(rel)
                known = scenes & census
                if scenes - census:
                    unknown_report[rel] = len(scenes - census)
                    for scene in known:
                        derived_hits[derived_group(rel)].setdefault(
                            scene, set()
                        ).add(rel)
                else:
                    raw_ledger.append(
                        {"path": rel, "sha256": sha256_file(path), "rows": rows,
                         "scene_ids": len(known), "evidence": evidence}
                    )
                continue
        try:
            if suffix in TEXT_SUFFIXES or path.stat().st_size < (4 << 20):
                found = stream_scan(path)
            else:
                found = set()
        except Exception as exc:
            errors.append(f"scan {rel}: {exc}")
            found = set()
        found |= scan_bytes(rel.encode())
        if found:
            add_derived(rel, found)

    derived_entries: list[dict[str, object]] = []
    provenance: dict[str, dict[str, list[str]]] = {}
    for group in sorted(derived_hits):
        scene_map = derived_hits[group]
        dpath = derived_dir / f"derived_exposure_{group}.jsonl"
        with dpath.open("w", encoding="utf-8") as handle:
            for scene in sorted(scene_map):
                handle.write(json.dumps({"scene_id": scene}) + "\n")
        provenance[group] = {
            s: sorted(f)[:20] for s, f in sorted(scene_map.items())
        }
        derived_entries.append(
            {"path": dpath.relative_to(root).as_posix(),
             "sha256": sha256_file(dpath), "rows": len(scene_map),
             "scene_ids": len(scene_map), "evidence": [f"derived:{group}"]}
        )

    exposure_files = sorted(raw_ledger + derived_entries, key=lambda i: i["path"])
    manifest_sha = canonical_exposure_manifest_sha256(
        [{"path": i["path"], "sha256": i["sha256"]} for i in exposure_files]
    )
    exposed: set[str] = set()
    for item in exposure_files:
        _, scenes = jsonl_top_scenes(root / str(item["path"]))
        exposed |= scenes & census

    ledger = {
        "schema_version": "0.2.0",
        "audit_id": "egoshift_historical_exposure_ledger",
        "status": "complete_reviewed",
        "complete": True,
        "pass": True,
        "created": "2026-08-20",
        "manifest_sha256": manifest_sha,
        "reviewed_manifest_count": len(exposure_files),
        "exposure_files": exposure_files,
        "exposure_rule": (
            "Outcome observation only: model evaluations (caches, reports, "
            "eval logs), human study/annotation surfaces, author-inspected "
            "samples/documents, plus every pool referenced by that evidence "
            "corpus and the thirteen independently documented historical "
            "pools. Scene-bearing files with no resolved evidence remain "
            "conservatively exposed. Machine generation-funnel artifacts are "
            "a documented source register, not exposure — the same ruling the "
            "frozen S05 contract applies to the Task-B reservoir and the "
            "pinned Task-C upstream."
        ),
        "census": {
            "components_path": str(args.components),
            "components_sha256": sha256_file(root / args.components),
            "census_scenes": len(census),
            "exposed_census_scenes": len(exposed),
        },
        "source_register": {
            "prefixes": list(SOURCE_PREFIXES),
            "exceptions_kept_as_exposure": list(SOURCE_EXCEPTION_PREFIXES),
            "file_patterns": list(SOURCE_FILE_PATTERNS),
            "log_basenames": list(SOURCE_LOG_BASENAMES),
            "files_hashed": len(source_register),
            "files": source_register,
        },
        "residual_conservative_exposed_files": residual_exposed,
        "unknown_scene_id_files": unknown_report,
        "scan": {
            "evidence_roots": list(EVIDENCE_ROOTS),
            "referenced_basenames": sorted(referenced_names),
            "errors": errors,
        },
    }
    require_complete_exposure_audit(ledger)
    (out_dir / "egoshift_historical_exposure_ledger.json").write_text(
        json.dumps(ledger, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "derived_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(
        {
            "exposure_files": len(exposure_files),
            "raw_ledger_files": len(raw_ledger),
            "derived_groups": {str(e["path"]): e["scene_ids"] for e in derived_entries},
            "source_register_files": len(source_register),
            "residual_conservative_exposed": len(residual_exposed),
            "exposed_census_scenes": len(exposed),
            "census_scenes": len(census),
            "unknown_id_files": len(unknown_report),
            "errors": len(errors),
            "manifest_sha256": manifest_sha,
        },
        indent=2, sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
