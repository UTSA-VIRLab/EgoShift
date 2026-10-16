"""The S05 gate: one integrated static audit over the three task candidates.

This is the only check that may move S05 from in-progress to complete. It is
deliberately independent of the builders: it re-reads the materialized
artifacts from disk and re-derives every property from them, so a builder bug
cannot certify itself. It never mutates a candidate and never writes a
release-ready manifest.
"""

from __future__ import annotations

import json
import hashlib
from io import BytesIO
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from .hashing import sha256_file, sha256_json
from .split_roles import ROLES, require_complete_split_roles
from .taskb_candidate_r2 import PUBLIC_FIELDS as TASKB_PUBLIC_FIELDS
from .taskb_candidate_r2 import finite_label_quotas
from .taskc_candidate import PUBLIC_FIELDS as TASKC_PUBLIC_FIELDS
from .taskc_source import CLASS_ORDER

SCHEMA_VERSION = "0.2.0"
AUDIT_ID = "egoshift_s05_integrated_candidate_audit"
TASKA_PUBLIC_FIELDS = {"id", "task_family", "format", "frames", "question", "options"}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _load_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _taska_overlay_sha256(source: Path, bbox: list[int]) -> str:
    """Reproduce Task A's deterministic red-box bake without writing a file."""

    from PIL import Image, ImageDraw

    with Image.open(source) as handle:
        image = handle.convert("RGB")
    x0, y0, x1, y1 = bbox
    ImageDraw.Draw(image).rectangle(
        (x0, y0, x1 - 1, y1 - 1), outline=(220, 30, 30), width=4
    )
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def audit_integrated_candidate(
    *,
    taskb_root: Path | None,
    taskc_root: Path | None,
    split_roles_path: Path,
    taskb_config_path: Path | None,
    report_path: Path,
    taska_root: Path | None = None,
    taska_build_report_path: Path | None = None,
) -> dict[str, Any]:
    """Re-derive every releasable property from the materialized artifacts."""

    failures: list[dict[str, Any]] = []
    checks: dict[str, Any] = {}

    def fail(code: str, detail: Any) -> None:
        failures.append({"check": code, "detail": detail})

    if taska_root is None and taskb_root is None and taskc_root is None:
        raise ValueError("at least one task candidate root is required")
    roles_payload = json.loads(split_roles_path.resolve().read_text(encoding="utf-8"))
    require_complete_split_roles(roles_payload)
    scenes = roles_payload["scenes"]

    roots: dict[str, Path] = {}
    if taska_root is not None:
        roots["A"] = taska_root.resolve()
    if taskb_root is not None:
        roots["B"] = taskb_root.resolve()
    if taskc_root is not None:
        roots["C"] = taskc_root.resolve()

    manifests: dict[str, dict[str, Any]] = {}
    for task, root in roots.items():
        if task == "A":
            continue
        manifest = _load_manifest(root)
        manifests[task] = manifest
        if manifest.get("release_ready") is not False:
            fail("release_ready_must_be_false", {"task": task})
        if manifest.get("split_roles", {}).get("sha256") != sha256_file(split_roles_path):
            fail("split_roles_hash_mismatch", {"task": task})
        # Every declared artifact must exist with the declared content hash.
        bad = []
        for record in manifest.get("artifacts") or []:
            path = root / str(record["path"])
            if not path.is_file():
                bad.append({"path": record["path"], "reason": "missing"})
            elif sha256_file(path) != record["sha256"]:
                bad.append({"path": record["path"], "reason": "hash_mismatch"})
        if bad:
            fail("artifact_hash_chain_broken", {"task": task, "artifacts": bad[:20]})
        checks[f"task_{task.lower()}_artifacts"] = len(manifest.get("artifacts") or [])

    candidate_ids = {task: m.get("candidate_id") for task, m in manifests.items()}
    if len(set(candidate_ids.values())) > 1:
        fail("candidate_id_disagreement", candidate_ids)

    # ---- public surface -------------------------------------------------
    allowlists = {
        "A": set(TASKA_PUBLIC_FIELDS),
        "B": set(TASKB_PUBLIC_FIELDS),
        "C": set(TASKC_PUBLIC_FIELDS),
    }
    public_ids: dict[str, set[str]] = {}
    asset_hashes: dict[str, set[str]] = {}
    public_counts: dict[str, dict[str, int]] = {}
    for task, root in roots.items():
        ids: set[str] = set()
        assets: set[str] = set()
        counts: dict[str, int] = {}
        for role in ROLES:
            rows = _read_jsonl(root / "public" / "questions" / f"task_{task.lower()}_{role}.jsonl")
            counts[role] = len(rows)
            for row in rows:
                if set(row) != allowlists[task]:
                    fail(
                        "public_field_allowlist_violation",
                        {"task": task, "id": row.get("id"), "fields": sorted(set(row))},
                    )
                item_id = str(row.get("id") or "")
                if not item_id or item_id in ids:
                    fail("duplicate_or_missing_public_id", {"task": task, "id": item_id})
                ids.add(item_id)
                for frame in row.get("frames") or []:
                    asset_path = root / str(frame)
                    if not asset_path.is_file():
                        fail("public_asset_missing", {"task": task, "frame": frame})
                    else:
                        assets.add(sha256_file(asset_path))
                options = row.get("options") or []
                if len(set(options)) != len(options):
                    fail("duplicate_option_text", {"task": task, "id": item_id})
        public_ids[task] = ids
        asset_hashes[task] = assets
        public_counts[task] = counts
    checks["public_items_by_task_and_split"] = public_counts

    cross_task_assets = {}
    for left_index, left in enumerate(sorted(roots)):
        for right in sorted(roots)[left_index + 1:]:
            shared_ids = public_ids[left] & public_ids[right]
            if shared_ids:
                fail(
                    "public_id_collision_across_tasks",
                    {"tasks": [left, right], "ids": sorted(shared_ids)[:20]},
                )
            shared_assets = asset_hashes[left] & asset_hashes[right]
            if shared_assets:
                fail(
                    "identical_rgb_content_across_tasks",
                    {"tasks": [left, right], "sha256": sorted(shared_assets)[:20]},
                )
            cross_task_assets[f"{left}:{right}"] = len(shared_assets)
    checks["cross_task_shared_assets"] = cross_task_assets

    # Public assets must be content-addressed: filename stem == file hash.
    for task, root in roots.items():
        if task == "A":
            continue
        mismatched = []
        for path in sorted((root / "assets").rglob("*.png")):
            if sha256_file(path) != path.stem:
                mismatched.append(str(path.relative_to(root)))
        if mismatched:
            fail("asset_not_content_addressed", {"task": task, "assets": mismatched[:20]})

    # ---- gold separation ------------------------------------------------
    for task, root in roots.items():
        hidden_gold = (
            root / "private" / "task_a_hidden_test_gold.jsonl"
            if task == "A"
            else root / "private" / "gold" / f"task_{task.lower()}_hidden_test.jsonl"
        )
        if not hidden_gold.is_file():
            fail("hidden_gold_missing", {"task": task})
        stray = [
            str(path.relative_to(root))
            for path in (root / "public").rglob("*")
            if path.is_file() and ("gold" in path.name or "answer" in path.name)
        ]
        if stray:
            fail("gold_leaked_into_public_tree", {"task": task, "paths": stray})
        gold_rows = _read_jsonl(hidden_gold)
        question_ids = set()
        for row in _read_jsonl(
            root / "public" / "questions" / f"task_{task.lower()}_hidden_test.jsonl"
        ):
            question_ids.add(row["id"])
        if {row["id"] for row in gold_rows} != question_ids:
            fail("hidden_gold_does_not_match_hidden_questions", {"task": task})

    # Task A predates the B/C manifest layout.  Its build report and private
    # gold records nevertheless contain a complete hash chain over the six
    # JSONL files and both frames of every item, so audit those exact bytes in
    # place rather than rebuilding the candidate.
    taska_report: dict[str, Any] | None = None
    if "A" in roots:
        if taska_build_report_path is None:
            raise ValueError("auditing Task A requires its build report")
        taska_report = json.loads(
            taska_build_report_path.resolve().read_text(encoding="utf-8")
        )
        if taska_report.get("pass") is not True:
            fail("taska_build_report_not_passed", None)
        bad_outputs = []
        for relative, declared in (taska_report.get("outputs") or {}).items():
            path = roots["A"] / relative
            if not path.is_file():
                bad_outputs.append({"path": relative, "reason": "missing"})
            elif sha256_file(path) != declared.get("sha256"):
                bad_outputs.append({"path": relative, "reason": "hash_mismatch"})
        if bad_outputs:
            fail("taska_output_hash_chain_broken", bad_outputs[:20])

        frame_mismatches = []
        frame_rows = 0
        public_frame_hashes = []
        fresh_checked = 0
        donor_root = Path(str((taska_report.get("frames") or {}).get("donor") or ""))
        for role in ROLES:
            questions = {
                str(row["id"]): row
                for row in _read_jsonl(
                    roots["A"] / "public" / "questions" / f"task_a_{role}.jsonl"
                )
            }
            gold_rows = _read_jsonl(roots["A"] / "private" / f"task_a_{role}_gold.jsonl")
            fresh_ids: set[str] = set()
            by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in gold_rows:
                by_class[str(row.get("candidate_answer_class"))].append(row)
            for class_rows in by_class.values():
                class_rows.sort(
                    key=lambda row: hashlib.sha256(
                        f"taska-fresh-overlay-audit|{row['id']}".encode()
                    ).hexdigest()
                )
                fresh_ids.update(str(row["id"]) for row in class_rows[:20])
            for row in gold_rows:
                question = questions.get(str(row.get("id")))
                source = row.get("_candidate_source") or {}
                expected_hashes = source.get("frame_sha256") or []
                frames = (question or {}).get("frames") or []
                frame_rows += len(frames)
                if len(frames) != 2 or len(expected_hashes) != 2:
                    frame_mismatches.append({"id": row.get("id"), "reason": "not_two_frames"})
                    continue
                actual_paths = [roots["A"] / str(frame) for frame in frames]
                actual_hashes = [sha256_file(path) for path in actual_paths]
                public_frame_hashes.extend(actual_hashes)

                raw_root = Path(str(source.get("frame_asset_root") or ""))
                raw_paths = [
                    raw_root / str(row.get("scene_id")) / "taska_pairs" / str(row.get("id")) / f"view_{index}.png"
                    for index in (1, 2)
                ]
                if any(not path.is_file() for path in raw_paths):
                    frame_mismatches.append({"id": row.get("id"), "reason": "raw_source_missing"})
                elif [sha256_file(path) for path in raw_paths] != list(expected_hashes):
                    frame_mismatches.append({"id": row.get("id"), "reason": "raw_source_hash_mismatch"})

                if donor_root:
                    donor_paths = [donor_root / str(frame) for frame in frames]
                    if any(not path.is_file() for path in donor_paths):
                        frame_mismatches.append({"id": row.get("id"), "reason": "declared_donor_missing"})
                    elif [sha256_file(path) for path in donor_paths] != actual_hashes:
                        frame_mismatches.append({"id": row.get("id"), "reason": "declared_donor_hash_mismatch"})

                if str(row.get("id")) in fresh_ids and all(path.is_file() for path in raw_paths):
                    boxes = [source.get("view_1_bbox_xyxy"), source.get("view_2_bbox_xyxy")]
                    order = (1, 0) if row.get("view_order_swapped") else (0, 1)
                    fresh_hashes = [
                        _taska_overlay_sha256(raw_paths[actual], boxes[actual])
                        for actual in order
                    ]
                    fresh_checked += 2
                    if fresh_hashes != actual_hashes:
                        frame_mismatches.append({"id": row.get("id"), "reason": "fresh_overlay_mismatch"})
                if question and (
                    row.get("answer_index") is None
                    or question["options"][int(row["answer_index"])] != row.get("answer")
                ):
                    frame_mismatches.append({"id": row.get("id"), "reason": "gold_option_mismatch"})
        if frame_mismatches:
            fail("taska_frame_or_gold_chain_broken", frame_mismatches[:20])
        checks["task_a_exact_frame_hashes"] = {
            "items": sum(public_counts["A"].values()),
            "frames": frame_rows,
            "public_frame_set_sha256": sha256_json(public_frame_hashes),
            "raw_source_hashes_verified": frame_rows,
            "declared_donor_hashes_verified": frame_rows if donor_root else 0,
            "fresh_overlays_verified": fresh_checked,
            "fresh_sampling_rule": "20 items per role x answer class; deterministic SHA-256 order",
            "mismatches": len(frame_mismatches),
        }

    # ---- role and exposure consistency across tasks ---------------------
    component_roles: dict[str, set[str]] = defaultdict(set)
    exposed_in_hidden: list[dict[str, Any]] = []
    provenance_counts: dict[str, int] = {}
    taskb_cells: dict[str, Counter] = {role: Counter() for role in ROLES}
    for task, root in roots.items():
        for role in ROLES:
            rows = (
                _read_jsonl(root / "private" / f"task_a_{role}_gold.jsonl")
                if task == "A"
                else _read_jsonl(
                    root / "private" / "provenance" / f"task_{task.lower()}_{role}.jsonl"
                )
            )
            provenance_counts[f"{task}:{role}"] = len(rows)
            for row in rows:
                source = row.get("_candidate_source") or {}
                scene_id = str(row.get("scene_id") or "")
                component = str(
                    row.get("physical_scene_component_id")
                    or source.get("physical_component_id")
                    or ""
                )
                ledger = scenes.get(scene_id)
                if not isinstance(ledger, Mapping):
                    fail("provenance_scene_absent_from_ledger", {"task": task, "scene": scene_id})
                    continue
                if ledger.get("leakage_component_id") != component:
                    fail(
                        "provenance_component_disagrees_with_ledger",
                        {"task": task, "scene": scene_id},
                    )
                if ledger.get("role") != role:
                    fail(
                        "provenance_role_disagrees_with_ledger",
                        {"task": task, "scene": scene_id, "role": role},
                    )
                component_roles[component].add(role)
                if role == "hidden_test" and ledger.get("outcome_exposed_component"):
                    exposed_in_hidden.append({"task": task, "component": component})
                if task == "B":
                    taskb_cells[role][
                        (
                            int(row["rotation_degrees"]),
                            str(row["turn_direction"]),
                            str(row["answer"]),
                        )
                    ] += 1
    crossed = {c: sorted(r) for c, r in component_roles.items() if len(r) > 1}
    if crossed:
        fail("component_crosses_roles_across_tasks", {"components": list(crossed)[:20]})
    if exposed_in_hidden:
        fail("outcome_exposed_component_in_hidden", exposed_in_hidden[:20])
    checks["provenance_rows"] = provenance_counts
    checks["distinct_hidden_components"] = len(
        {c for c, r in component_roles.items() if r == {"hidden_test"}}
    )

    # ---- Task-B finite-cell attainment ----------------------------------
    attainment: dict[str, Any] = {}
    if "B" in roots:
        if taskb_config_path is None:
            raise ValueError("auditing Task B requires its candidate contract")
        attainment = _taskb_attainment(taskb_config_path, taskb_cells, fail)
    checks["taskb_attainment"] = attainment

    # ---- Task-C class quotas and matched-target contract ---------------
    if "C" in roots:
        taskc_counts: dict[str, dict[str, int]] = {}
        taskc_matching: dict[str, Any] = {}
        for role in ROLES:
            rows = _read_jsonl(roots["C"] / "private" / "provenance" / f"task_c_{role}.jsonl")
            questions = {
                str(row["id"]): row
                for row in _read_jsonl(
                    roots["C"] / "public" / "questions" / f"task_c_{role}.jsonl"
                )
            }
            taskc_counts[role] = dict(Counter(str(row.get("answer_class")) for row in rows))
            identities: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                identities[
                    (
                        str(row.get("physical_scene_component_id")),
                        str(row.get("target_oid")),
                    )
                ].append(row)

            repeated_class: list[str] = []
            inconsistent_view1: list[str] = []
            repeated_view2: list[str] = []
            group_sizes = Counter()
            complete_triples = 0
            for identity, group in identities.items():
                label = f"{identity[0]}|{identity[1]}"
                group_sizes[len(group)] += 1
                classes = [str(row.get("answer_class")) for row in group]
                if len(classes) != len(set(classes)):
                    repeated_class.append(label)

                frame_pairs = [
                    tuple(str(frame) for frame in (questions.get(str(row.get("id"))) or {}).get("frames", []))
                    for row in group
                ]
                view1 = [pair[0] for pair in frame_pairs if len(pair) == 2]
                view2 = [pair[1] for pair in frame_pairs if len(pair) == 2]
                if len(frame_pairs) != len(group) or any(len(pair) != 2 for pair in frame_pairs):
                    inconsistent_view1.append(label)
                elif len(group) > 1 and len(set(view1)) != 1:
                    inconsistent_view1.append(label)
                if len(view2) != len(set(view2)):
                    repeated_view2.append(label)
                if len(group) == len(CLASS_ORDER) and set(classes) == set(CLASS_ORDER):
                    complete_triples += 1

            if repeated_class:
                fail(
                    "taskc_identity_class_repeated",
                    {"role": role, "identities": repeated_class[:20]},
                )
            if inconsistent_view1:
                fail(
                    "taskc_matched_identity_view1_disagreement",
                    {"role": role, "identities": inconsistent_view1[:20]},
                )
            if repeated_view2:
                fail(
                    "taskc_matched_identity_view2_reused",
                    {"role": role, "identities": repeated_view2[:20]},
                )

            # Candidate v10 is intentionally made of target-matched triples:
            # one byte-identical View 1 and one distinct View 2 for each of the
            # three visibility outcomes.  Older test fixtures and candidates
            # may contain singletons, so the strict requirement is tied to the
            # construction identifier rather than retroactively imposed on
            # every historical artifact.
            construction_id = str(manifests["C"].get("construction_id") or "")
            if "v10" in construction_id:
                incomplete = [
                    f"{component}|{target}"
                    for (component, target), group in identities.items()
                    if len(group) != len(CLASS_ORDER)
                    or {str(row.get("answer_class")) for row in group} != set(CLASS_ORDER)
                ]
                if incomplete:
                    fail(
                        "taskc_matched_triple_incomplete",
                        {"role": role, "identities": incomplete[:20]},
                    )
            missing = [c for c in CLASS_ORDER if c not in taskc_counts[role]]
            if missing:
                fail("taskc_class_absent", {"role": role, "classes": missing})
            taskc_matching[role] = {
                "target_identities": len(identities),
                "group_sizes": {str(size): count for size, count in sorted(group_sizes.items())},
                "complete_three_class_triples": complete_triples,
                "repeated_identity_class_pairs": len(repeated_class),
                "view1_disagreements": len(inconsistent_view1),
                "view2_reuse_within_identity": len(repeated_view2),
            }
        checks["taskc_items_by_split_and_class"] = taskc_counts
        checks["taskc_matched_target_contract"] = taskc_matching

    report = {
        "schema_version": SCHEMA_VERSION,
        "audit": AUDIT_ID,
        "status": "static_candidate_only_not_release_ready",
        "inputs": {
            "task_a_root": str(taska_root.resolve()) if taska_root else None,
            "task_a_build_report": (
                {
                    "path": str(taska_build_report_path),
                    "sha256": sha256_file(taska_build_report_path),
                }
                if taska_build_report_path is not None
                else None
            ),
            "task_b_root": str(taskb_root.resolve()) if taskb_root else None,
            "task_c_root": str(taskc_root.resolve()) if taskc_root else None,
            "split_roles": {
                "path": str(split_roles_path),
                "sha256": sha256_file(split_roles_path),
            },
            "task_b_config": (
                {
                    "path": str(taskb_config_path),
                    "sha256": sha256_file(taskb_config_path),
                }
                if taskb_config_path is not None
                else None
            ),
        },
        "task_manifests": {
            task: {
                "candidate_id": manifest.get("candidate_id"),
                "counts": manifest.get("counts"),
                "manifest_sha256": sha256_file(roots[task] / "manifest.json"),
            }
            for task, manifest in manifests.items()
        },
        "checks": checks,
        "failures": failures,
        "release_ready": False,
        "pass": not failures,
    }
    if taska_report is not None:
        report["task_manifests"]["A"] = {
            "candidate_id": "taska-candidate-v5",
            "counts": {"items": taska_report.get("items")},
            "manifest_sha256": sha256_file(taska_build_report_path),
            "manifest_kind": "legacy_build_report_companion",
        }
    report["integrated_manifest_sha256"] = sha256_json(
        [
            report["task_manifests"],
            report["inputs"]["split_roles"]["sha256"],
            report["pass"],
        ]
    )
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def _taskb_attainment(
    taskb_config_path: Path,
    taskb_cells: Mapping[str, Counter],
    fail,
) -> dict[str, Any]:
    """Re-derive per-cell attainment from provenance, independent of the builder."""

    taskb_config = json.loads(taskb_config_path.resolve().read_text(encoding="utf-8"))
    seed = str(taskb_config["seed"])
    half_width = float(taskb_config["geometry"]["source_bearing_half_width_degrees"])
    attainment: dict[str, Any] = {}
    for role in ROLES:
        quotas = finite_label_quotas(
            role=role,
            items_per_action_cell=int(taskb_config["splits"][role]["items_per_action_cell"]),
            seed=seed,
            half_width=half_width,
            items_by_rotation=taskb_config["splits"][role].get(
                "items_per_action_cell_by_rotation"
            ),
        )
        observed = taskb_cells[role]
        mismatches = {
            f"{cell[0]}:{cell[1]}:{cell[2]}": {"expected": expected, "observed": observed[cell]}
            for cell, expected in sorted(quotas.items())
            if observed[cell] != expected
        }
        extra = {
            f"{cell[0]}:{cell[1]}:{cell[2]}": count
            for cell, count in sorted(observed.items())
            if cell not in quotas
        }
        attainment[role] = {
            "cells": len(quotas),
            "mismatched_cells": mismatches,
            "unexpected_cells": extra,
            "items": sum(observed.values()),
            "structural_targets": _finite_action_cell_targets(observed),
        }
        if mismatches or extra:
            fail("taskb_finite_cell_attainment_failed", {"role": role, "mismatches": mismatches, "extra": extra})
    return attainment


def _finite_action_cell_targets(observed: Counter) -> dict[str, Any]:
    """Return item-weighted and equal-action-cell targets for exact built bytes."""

    by_action: dict[tuple[int, str], Counter] = defaultdict(Counter)
    for (rotation, turn, answer), count in observed.items():
        by_action[(int(rotation), str(turn))][str(answer)] += int(count)
    correct = sum(max(labels.values()) for labels in by_action.values() if labels)
    total = sum(sum(labels.values()) for labels in by_action.values())
    cell_scores = [
        max(labels.values()) / sum(labels.values())
        for labels in by_action.values()
        if labels and sum(labels.values())
    ]
    return {
        "action_cells": len(by_action),
        "correct": correct,
        "total": total,
        "micro_accuracy": correct / total if total else None,
        "macro_accuracy": sum(cell_scores) / len(cell_scores) if cell_scores else None,
    }
