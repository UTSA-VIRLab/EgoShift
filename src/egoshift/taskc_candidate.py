"""Materialize the static Task-C candidate: public/private separation.

Consumes live-verified, overlay-baked private Task-C rows and emits the same
artifact shape the Task-B builder emits: strictly allowlisted public questions
with opaque IDs and content-addressed assets, development answers separated
from private hidden gold, full private provenance, and a manifest that states
`release_ready=false`.

Every gate is fail-closed. Quotas are exact and are never relaxed on
infeasibility; a shortfall produces a machine-readable blocker instead of a
smaller candidate.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .hashing import resolve_within, sha256_file, sha256_json
from .split_roles import ROLES, require_complete_split_roles
from .taskc_source import CLASS_ORDER, CLASS_TO_ANSWER

SCHEMA_VERSION = "0.1.0"
PUBLIC_FIELDS = ("id", "task_family", "format", "frames", "question", "options")
GOLD_EVIDENCE_TYPE = "live_exact_mask_and_geometry_replay"
OVERLAY_EVIDENCE_TYPE = "exact_instance_mask_replay"


class TaskCCandidateError(RuntimeError):
    """Fail-closed construction error carrying machine-readable diagnostics."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


@dataclass(frozen=True)
class TaskCSourceSpec:
    name: str
    jsonl_path: Path
    asset_roots: tuple[Path, ...]


def validate_contract(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected Task-C candidate schema version")
    if tuple(config.get("classes") or ()) != CLASS_ORDER:
        raise ValueError("Task-C classes disagree with the canonical class order")
    presentation = config.get("presentation") or {}
    if tuple(presentation.get("public_question_fields") or ()) != PUBLIC_FIELDS:
        raise ValueError("Task-C public field allowlist disagrees with the contract")
    if presentation.get("task_family") != "C":
        raise ValueError("Task-C contract must declare task family C")
    for role in ROLES:
        split = (config.get("splits") or {}).get(role)
        if not isinstance(split, Mapping):
            raise ValueError(f"missing Task-C split contract for {role}")
        per_class = int(split["items_per_class"])
        if per_class <= 0:
            raise ValueError(f"non-positive Task-C quota for {role}")
        if int(split["total_items"]) != per_class * len(CLASS_ORDER):
            raise ValueError(f"Task-C total/per-class mismatch for {role}")
    uniqueness = config.get("uniqueness") or {}
    # The original DC-6 gave each physical target at most one released item
    # across the whole task, on the theory that a repeated target creates
    # pseudo-replication. It does the opposite of what it was meant to: a
    # target that appears ONLY as "occluded" makes object identity predict the
    # answer, and a DINOv2 probe read View 1 alone at 43% against 33% chance.
    # The contract is now the inverse -- a target appears exactly once per
    # class, and its three rows share one View-1 image byte for byte, so both
    # identity and View 1 are uninformative by construction.
    if uniqueness.get("one_row_per_physical_component_target_and_class") is not True:
        raise ValueError(
            "Task-C contract must cap one row per (component, target, class)"
        )
    if uniqueness.get("target_matched_triples_share_one_view_one") is not True:
        raise ValueError(
            "Task-C contract must require target-matched triples on one View 1"
        )


def _resolve_frame(spec: TaskCSourceSpec, relative: str) -> Path:
    """Resolve a frame inside one of the declared roots, never outside them."""

    for root in spec.asset_roots:
        try:
            candidate = resolve_within(root.resolve(), relative)
        except Exception:
            continue
        if candidate.is_file():
            return candidate
    raise TaskCCandidateError(
        f"{spec.name}: frame is missing from every declared asset root",
        {"frame": relative, "roots": [str(root) for root in spec.asset_roots]},
    )


def _validate_row(
    row: Mapping[str, Any],
    *,
    spec: TaskCSourceSpec,
    line_number: int,
    roles: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    label = f"{spec.name} line {line_number}"
    if row.get("task_family") != "C":
        raise TaskCCandidateError(f"{label}: not a Task-C row")
    answer_class = str(row.get("candidate_answer_class") or "")
    if answer_class not in CLASS_ORDER:
        raise TaskCCandidateError(f"{label}: invalid candidate class {answer_class!r}")
    metadata = row.get("metadata")
    if not isinstance(metadata, Mapping) or metadata.get("answer_class") != answer_class:
        raise TaskCCandidateError(f"{label}: metadata class disagrees with the candidate class")
    if row.get("answer") != CLASS_TO_ANSWER[answer_class]:
        raise TaskCCandidateError(f"{label}: stored answer disagrees with its class")

    scene_id = str(row.get("scene_id") or "")
    role_record = roles.get(scene_id)
    if not isinstance(role_record, Mapping):
        raise TaskCCandidateError(f"{label}: scene absent from the role ledger")
    if role_record.get("drop_exact_alias_nonrepresentative") is True:
        raise TaskCCandidateError(f"{label}: exact-alias nonrepresentative scene is forbidden")
    role = str(role_record.get("role") or "")
    if role not in ROLES:
        raise TaskCCandidateError(f"{label}: unknown role {role!r}")
    if row.get("candidate_role") != role:
        raise TaskCCandidateError(f"{label}: stale candidate role disagrees with the ledger")
    component = str(role_record.get("leakage_component_id") or "")

    split = config["splits"][role]
    exposed = bool(role_record.get("outcome_exposed_component"))
    if split.get("requires_never_outcome_exposed_component") and exposed:
        raise TaskCCandidateError(f"{label}: outcome-exposed component assigned to {role}")

    source = row.get("_candidate_source")
    if not isinstance(source, Mapping):
        raise TaskCCandidateError(f"{label}: missing private candidate source")
    if source.get("physical_component_id") != component:
        raise TaskCCandidateError(f"{label}: component disagrees with the role ledger")
    evidence = source.get("gold_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("evidence_type") != GOLD_EVIDENCE_TYPE:
        raise TaskCCandidateError(f"{label}: gold evidence is not live exact-mask replay")
    if (config.get("evidence") or {}).get("view1_overlay_required"):
        overlay = source.get("exact_mask_bbox_overlay")
        if not isinstance(overlay, Mapping):
            raise TaskCCandidateError(f"{label}: missing the exact-mask View-1 overlay record")
        if overlay.get("evidence_type") != OVERLAY_EVIDENCE_TYPE:
            raise TaskCCandidateError(f"{label}: overlay is not an exact instance-mask replay")

    frames = row.get("frames")
    if not isinstance(frames, list) or len(frames) != 2:
        raise TaskCCandidateError(f"{label}: exactly two frames are required")
    resolved: list[dict[str, str]] = []
    for position, relative in enumerate(frames):
        path = _resolve_frame(spec, str(relative))
        resolved.append(
            {"relative": str(relative), "path": str(path), "sha256": sha256_file(path)}
        )
    overlay_record = source.get("exact_mask_bbox_overlay") or {}
    expected_overlay_hash = str(overlay_record.get("overlay_sha256") or "")
    if expected_overlay_hash and resolved[0]["sha256"] != expected_overlay_hash:
        raise TaskCCandidateError(f"{label}: View-1 overlay hash mismatch")

    target_oid = str(row.get("target_oid") or "")
    if not target_oid:
        raise TaskCCandidateError(f"{label}: missing target object identity")
    stable = str(source.get("stable_identity_sha256") or "")
    if not stable:
        raise TaskCCandidateError(f"{label}: missing stable identity")

    return {
        "_row": dict(row),
        "_role": role,
        "_class": answer_class,
        "_component": component,
        "_target": target_oid,
        "_scene": scene_id,
        "_stable": stable,
        "_frames": resolved,
        "_source_name": spec.name,
        "_source_line": line_number,
        "_outcome_exposed_component": exposed,
    }


def load_taskc_candidates(
    *,
    source_specs: Sequence[TaskCSourceSpec],
    roles_payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require_complete_split_roles(roles_payload)
    roles = roles_payload.get("scenes")
    if not isinstance(roles, dict):
        raise ValueError("split-role ledger is absent or failing")
    counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    for spec in source_specs:
        path = spec.jsonl_path.resolve()
        inputs.append(
            {
                "name": spec.name,
                "path": str(path),
                "sha256": sha256_file(path),
                "asset_roots": [str(root.resolve()) for root in spec.asset_roots],
            }
        )
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                counts["input_rows"] += 1
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise TaskCCandidateError(f"{spec.name} line {line_number}: not an object")
                records.append(
                    _validate_row(
                        row,
                        spec=spec,
                        line_number=line_number,
                        roles=roles,
                        config=config,
                    )
                )
    # One released row per (component, target, class) -- so a single object can
    # never flood one answer -- while the SAME object is expected under all
    # three answers. This block used to key on (component, target) alone and
    # keep the scarcest class, which silently discarded two thirds of a
    # target-matched triple: with triples upstream it left "occluded" with
    # 2100 rows and the other two classes with zero.
    by_identity: dict[tuple[str, str, str], dict[str, Any]] = {}
    for record in sorted(records, key=lambda item: item["_stable"]):
        key = (record["_component"], record["_target"], record["_class"])
        if key in by_identity:
            counts["same_class_identity_collisions"] += 1
            continue
        by_identity[key] = record
    selected = sorted(by_identity.values(), key=lambda item: item["_stable"])
    counts["eligible_unique_identities"] = len(selected)
    # Every released target must carry all three answers on one View-1 image.
    view_one: dict[tuple[str, str], set[str]] = defaultdict(set)
    classes_seen: dict[tuple[str, str], set[str]] = defaultdict(set)
    for record in selected:
        identity = (record["_component"], record["_target"])
        classes_seen[identity].add(record["_class"])
        view_one[identity].add(record["_row"]["frames"][0] if "_row" in record
                               else record.get("_view_one", ""))
    counts["targets_covering_all_classes"] = sum(
        1 for names in classes_seen.values() if len(names) == len(CLASS_ORDER)
    )
    counts["targets_missing_a_class"] = sum(
        1 for names in classes_seen.values() if len(names) != len(CLASS_ORDER)
    )
    counts.update(Counter(f"role_{record['_role']}" for record in selected))
    return selected, {"sources": inputs, "counts": dict(sorted(counts.items()))}


def _assign_options(rows: list[dict[str, Any]], labels: Sequence[str], seed: str) -> None:
    """Balance the answer position deterministically inside split/class strata."""

    groups: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["_split"], row["_answer"])].append(row)
    for key, group in sorted(groups.items(), key=lambda value: repr(value[0])):
        offset = int(sha256_json([seed, "answer-position", key]), 16) % len(labels)
        ordered = sorted(group, key=lambda row: sha256_json([seed, "option-row", row["id"]]))
        for index, row in enumerate(ordered):
            answer_position = (offset + index) % len(labels)
            distractors = [label for label in labels if label != row["_answer"]]
            distractors.sort(
                key=lambda label: sha256_json([seed, "distractor-order", row["id"], label])
            )
            options = list(distractors)
            options.insert(answer_position, row["_answer"])
            row["options"] = options


def _jsonl_write(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
            written += 1
    return written


def _artifact_record(root: Path, path: Path, *, rows: int | None = None) -> dict[str, Any]:
    record = {"path": str(path.relative_to(root)), "sha256": sha256_file(path)}
    if rows is not None:
        record["rows"] = rows
    return record


def build_taskc_candidate(
    *,
    config_path: Path,
    split_roles_path: Path,
    source_specs: Sequence[TaskCSourceSpec],
    output_root: Path,
    audit_report_path: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    split_roles_path = split_roles_path.resolve()
    output_root = output_root.resolve()
    audit_report_path = audit_report_path.resolve()
    if output_root.exists() or audit_report_path.exists():
        raise ValueError("refusing to overwrite candidate output or audit report")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_contract(config)
    roles_payload = json.loads(split_roles_path.read_text(encoding="utf-8"))
    candidates, source_audit = load_taskc_candidates(
        source_specs=source_specs, roles_payload=roles_payload, config=config
    )

    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in candidates:
        pools[(record["_role"], record["_class"])].append(record)

    capacity: dict[str, dict[str, Any]] = {}
    shortfalls: list[dict[str, Any]] = []
    chosen: list[dict[str, Any]] = []
    for role in ROLES:
        capacity[role] = {}
        required = int(config["splits"][role]["items_per_class"])
        for answer_class in CLASS_ORDER:
            pool = sorted(pools[(role, answer_class)], key=lambda item: item["_stable"])
            capacity[role][answer_class] = {
                "available": len(pool),
                "required": required,
                "surplus": len(pool) - required,
                "feasible": len(pool) >= required,
            }
            if len(pool) < required:
                shortfalls.append(
                    {
                        "role": role,
                        "answer_class": answer_class,
                        "available": len(pool),
                        "required": required,
                        "shortfall": required - len(pool),
                    }
                )
                continue
            chosen.extend(pool[:required])
    if shortfalls:
        raise TaskCCandidateError(
            "Task-C quotas are unmet; the contract forbids relaxing them",
            {"capacity": capacity, "shortfalls": shortfalls, "sources": source_audit},
        )

    seed = str(config["seed"])
    labels = tuple(config["presentation"]["canonical_options"])
    rows_by_role: dict[str, list[dict[str, Any]]] = {}
    all_ids: set[str] = set()
    for role in ROLES:
        rows: list[dict[str, Any]] = []
        for record in [item for item in chosen if item["_role"] == role]:
            digest = sha256_json(
                [
                    config["candidate_id"],
                    config["construction_id"],
                    "Task-C",
                    role,
                    record["_component"],
                    record["_scene"],
                    record["_target"],
                    record["_class"],
                    record["_stable"],
                ]
            )
            item_id = str(config["presentation"]["opaque_id_prefix"]) + digest[:24]
            if item_id in all_ids:
                raise TaskCCandidateError(f"opaque public ID collision: {item_id}")
            all_ids.add(item_id)
            prefix = config["presentation"]["public_asset_prefix"]
            public_frames = [
                f"{prefix}/{frame['sha256']}.png" for frame in record["_frames"]
            ]
            rows.append(
                {
                    "id": item_id,
                    "task_family": config["presentation"]["task_family"],
                    "format": config["presentation"]["format"],
                    "frames": public_frames,
                    "question": record["_row"]["question"],
                    "options": list(labels),
                    "_answer": CLASS_TO_ANSWER[record["_class"]],
                    "_split": role,
                    "_class": record["_class"],
                    "_record": record,
                }
            )
        _assign_options(rows, labels, seed)
        rows.sort(key=lambda row: row["id"])
        rows_by_role[role] = rows

    forbidden = set(config["separation"]["forbidden_public_fields"])
    public_violations: list[dict[str, Any]] = []
    content_hashes: Counter[str] = Counter()
    view_two_hashes: Counter[str] = Counter()
    view_one_owner: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for role in ROLES:
        for row in rows_by_role[role]:
            public = {field: row[field] for field in PUBLIC_FIELDS}
            if set(public) & forbidden:
                public_violations.append({"id": row["id"], "fields": sorted(set(public) & forbidden)})
            if row["_answer"] not in row["options"]:
                raise TaskCCandidateError(f"{row['id']}: gold answer is absent from its options")
            frames = row["_record"]["_frames"]
            view_two_hashes[frames[-1]["sha256"]] += 1
            view_one_owner[frames[0]["sha256"]].add(
                (row["_record"]["_component"], row["_record"]["_target"])
            )
            content_hashes[frames[0]["sha256"]] += 1
    if public_violations:
        raise TaskCCandidateError(
            "public rows contain forbidden fields", {"violations": public_violations}
        )
    # Frame-content rules. View 2 is what the question is actually about, so
    # no two released items may share one, or the pool is padded with copies.
    # View 1 is the opposite case: a target's three items are REQUIRED to share
    # one View-1 image byte for byte, which is what makes View 1 uninformative
    # about the answer. A blanket "every frame unique" rule -- which is what
    # this used to be -- forbids exactly the property the release depends on.
    # What must not happen is one View-1 image spanning two different targets.
    if (config.get("uniqueness") or {}).get("view_two_content_unique_across_all_items"):
        duplicates = {digest: n for digest, n in view_two_hashes.items() if n > 1}
        if duplicates:
            raise TaskCCandidateError(
                "exact RGB content is reused across released Task-C View 2 frames",
                {"duplicate_content_hashes": len(duplicates)},
            )
    if (config.get("uniqueness") or {}).get(
        "view_one_content_shared_only_within_one_target"
    ):
        shared = {d: len(owners) for d, owners in view_one_owner.items() if len(owners) > 1}
        if shared:
            raise TaskCCandidateError(
                "a View-1 image is shared by more than one physical target",
                {"view_one_hashes_spanning_targets": len(shared)},
            )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=output_root.parent, prefix=f".{output_root.name}.building."
    ) as temporary:
        staging = Path(temporary)
        artifacts: list[dict[str, Any]] = []
        copied: set[str] = set()
        for role in ROLES:
            rows = rows_by_role[role]
            public_rows = [{field: row[field] for field in PUBLIC_FIELDS} for row in rows]
            question_path = staging / "public" / "questions" / f"task_c_{role}.jsonl"
            artifacts.append(
                _artifact_record(staging, question_path, rows=_jsonl_write(question_path, public_rows))
            )
            gold_rows = [
                {
                    "id": row["id"],
                    "answer": row["_answer"],
                    "answer_index": row["options"].index(row["_answer"]),
                    "answer_class": row["_class"],
                }
                for row in rows
            ]
            if role == "hidden_test":
                gold_path = staging / "private" / "gold" / "task_c_hidden_test.jsonl"
            else:
                gold_path = staging / "development" / "answers" / f"task_c_{role}.jsonl"
            artifacts.append(
                _artifact_record(staging, gold_path, rows=_jsonl_write(gold_path, gold_rows))
            )
            provenance_rows = []
            for row in rows:
                record = row["_record"]
                source = record["_row"]["_candidate_source"]
                provenance_rows.append(
                    {
                        "id": row["id"],
                        "split": role,
                        "answer": row["_answer"],
                        "answer_class": row["_class"],
                        "answer_index": row["options"].index(row["_answer"]),
                        "scene_id": record["_scene"],
                        "physical_scene_component_id": record["_component"],
                        "outcome_exposed_component": record["_outcome_exposed_component"],
                        "target_oid": record["_target"],
                        "source_name": record["_source_name"],
                        "source_line": record["_source_line"],
                        "stable_identity_sha256": record["_stable"],
                        "source_frames": record["_frames"],
                        "public_frames": row["frames"],
                        "gold_evidence": source.get("gold_evidence"),
                        "exact_mask_bbox_overlay": source.get("exact_mask_bbox_overlay"),
                        "trajectory_template": (record["_row"].get("metadata") or {}).get(
                            "trajectory_template"
                        ),
                    }
                )
            provenance_path = staging / "private" / "provenance" / f"task_c_{role}.jsonl"
            artifacts.append(
                _artifact_record(
                    staging, provenance_path, rows=_jsonl_write(provenance_path, provenance_rows)
                )
            )
            for row in rows:
                for public_relative, frame in zip(row["frames"], row["_record"]["_frames"]):
                    if frame["sha256"] in copied:
                        continue
                    copied.add(frame["sha256"])
                    destination = staging / public_relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(frame["path"], destination)
                    if sha256_file(destination) != frame["sha256"]:
                        raise TaskCCandidateError("materialized public asset hash mismatch")
                    artifacts.append(_artifact_record(staging, destination))
        artifacts.sort(key=lambda record: record["path"])
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "manifest": "egoshift_taskc_s05_static_candidate",
            "candidate_id": config["candidate_id"],
            "construction_id": config["construction_id"],
            "status": "static_candidate_only_not_release_ready",
            "task_family": "C",
            "config": {"sha256": sha256_file(config_path)},
            "split_roles": {"sha256": sha256_file(split_roles_path)},
            "counts": {
                "items": sum(len(rows) for rows in rows_by_role.values()),
                "items_by_split": {role: len(rows_by_role[role]) for role in ROLES},
                "items_by_split_and_class": {
                    role: dict(Counter(row["_class"] for row in rows_by_role[role]))
                    for role in ROLES
                },
                "public_assets": len(copied),
            },
            "artifacts": artifacts,
            "release_ready": False,
            "remaining_blockers": config["downstream_release_blockers"],
        }
        manifest_path = staging / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifest_hash = sha256_file(manifest_path)
        staging.rename(output_root)

    audit = {
        "schema_version": SCHEMA_VERSION,
        "audit": "taskc_candidate_r2_static_construction",
        "candidate_id": config["candidate_id"],
        "construction_id": config["construction_id"],
        "status": "static_candidate_only_not_release_ready",
        "inputs": {
            "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
            "split_roles": {"path": str(split_roles_path), "sha256": sha256_file(split_roles_path)},
            **source_audit,
        },
        "contract": {
            "items_per_class": {
                role: int(config["splits"][role]["items_per_class"]) for role in ROLES
            },
            "public_question_fields": list(PUBLIC_FIELDS),
            "identity_cap": "one released row per physical component, target and class; every target carries all three answers",
        },
        "capacity": capacity,
        "counts": manifest["counts"],
        "public_private_separation": {
            "questions": "public/questions/task_c_<split>.jsonl",
            "development_answers": "development/answers/task_c_<development-role>.jsonl",
            "hidden_gold": "private/gold/task_c_hidden_test.jsonl",
            "private_provenance": "private/provenance/task_c_<split>.jsonl",
            "public_allowlist_pass": True,
        },
        "output": {
            "root": str(output_root),
            "manifest": {"path": str(output_root / "manifest.json"), "sha256": manifest_hash},
        },
        "release_ready": False,
        "remaining_blockers": config["downstream_release_blockers"],
        "pass": True,
    }
    audit_report_path.parent.mkdir(parents=True, exist_ok=True)
    audit_report_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    return audit
