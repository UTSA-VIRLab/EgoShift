"""Deterministic Task-C overlays from exact replayed instance-mask boxes.

This interface intentionally refuses projected-center, estimated-size, or
otherwise approximate boxes.  Each accepted box must be the nonzero extent of
an exact replay mask and must bind both the raw frame bytes and mask bytes by
SHA-256.  Pillow is imported only by the rendering entry point so validation
remains usable in the dependency-free benchmark runtime.
"""

from __future__ import annotations

import copy
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .hashing import canonical_json_bytes, resolve_within, safe_relative_path, sha256_bytes, sha256_file
from .taskc_source import CANONICAL_OPTIONS, CLASS_TO_ANSWER


EXACT_EVIDENCE_TYPE = "exact_instance_mask_replay"
EXACT_BBOX_SOURCE = "exact_instance_mask_nonzero_extent"
COORDINATE_CONVENTION = "xyxy_half_open"
HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
OVERLAY_RGB = (220, 30, 30)
OVERLAY_WIDTH = 4


class ExactMaskBBoxError(ValueError):
    """Raised when replay evidence cannot prove an exact mask box."""


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ExactMaskBBoxError(f"{name} must be an integer, not boolean")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ExactMaskBBoxError(f"{name} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise ExactMaskBBoxError(f"{name} must be integral")
    return result


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
        raise ExactMaskBBoxError(f"{name} must be a lowercase SHA-256 digest")
    return value


def load_exact_mask_replay_sidecar(
    path: str | Path,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    """Load the strict replay-sidecar envelope and return its item mapping."""

    sidecar_path = Path(path)
    with sidecar_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ExactMaskBBoxError("replay sidecar must be a JSON object")
    if payload.get("evidence_type") != EXACT_EVIDENCE_TYPE:
        raise ExactMaskBBoxError(
            f"sidecar evidence_type must be {EXACT_EVIDENCE_TYPE!r}"
        )
    if payload.get("coordinate_convention") != COORDINATE_CONVENTION:
        raise ExactMaskBBoxError(
            f"coordinate_convention must be {COORDINATE_CONVENTION!r}"
        )
    items = payload.get("items")
    if not isinstance(items, dict):
        raise ExactMaskBBoxError("replay sidecar must contain an items object")
    for item_id, entry in items.items():
        if not isinstance(item_id, str) or not item_id:
            raise ExactMaskBBoxError("replay item IDs must be non-empty strings")
        if not isinstance(entry, Mapping):
            raise ExactMaskBBoxError(f"replay entry {item_id!r} must be an object")
    provenance = {
        "path": str(sidecar_path),
        "sha256": sha256_file(sidecar_path),
        "schema_version": payload.get("schema_version"),
        "evidence_type": EXACT_EVIDENCE_TYPE,
        "coordinate_convention": COORDINATE_CONVENTION,
        "items": len(items),
    }
    return dict(items), provenance


def validate_exact_mask_bbox_entry(
    entry: Mapping[str, Any],
    *,
    expected_object_id: str,
    expected_frame_path: str,
    expected_frame_index: int,
) -> dict[str, Any]:
    """Validate and normalize one exact replay-mask bbox record.

    The required provenance fields are what distinguish this interface from
    the repository's earlier projected-center box experiment.
    """

    if not isinstance(entry, Mapping):
        raise ExactMaskBBoxError("bbox replay entry must be an object")
    if entry.get("evidence_type") != EXACT_EVIDENCE_TYPE:
        raise ExactMaskBBoxError(
            f"entry evidence_type must be {EXACT_EVIDENCE_TYPE!r}; approximate boxes are forbidden"
        )
    if entry.get("bbox_source") != EXACT_BBOX_SOURCE:
        raise ExactMaskBBoxError(
            f"bbox_source must be {EXACT_BBOX_SOURCE!r}; approximate boxes are forbidden"
        )
    if entry.get("coordinate_convention") != COORDINATE_CONVENTION:
        raise ExactMaskBBoxError(
            f"coordinate_convention must be {COORDINATE_CONVENTION!r}"
        )
    if entry.get("object_id") != expected_object_id:
        raise ExactMaskBBoxError("replay object_id does not match the selected target")
    try:
        frame_path = safe_relative_path(entry.get("frame_path"))
        expected_path = safe_relative_path(expected_frame_path)
    except (TypeError, ValueError) as exc:
        raise ExactMaskBBoxError(str(exc)) from exc
    if frame_path != expected_path:
        raise ExactMaskBBoxError("replay frame_path does not match View 1")
    frame_index = _integer(entry.get("frame_index"), "frame_index")
    if frame_index != expected_frame_index:
        raise ExactMaskBBoxError("replay frame_index does not match source_step")
    image_size = entry.get("image_size")
    if not isinstance(image_size, Mapping):
        raise ExactMaskBBoxError("image_size must be an object")
    width = _integer(image_size.get("w"), "image width")
    height = _integer(image_size.get("h"), "image height")
    if width <= 0 or height <= 0:
        raise ExactMaskBBoxError("image dimensions must be positive")
    bbox_value = entry.get("bbox_xyxy")
    if not isinstance(bbox_value, list) or len(bbox_value) != 4:
        raise ExactMaskBBoxError("bbox_xyxy must contain four integers")
    bbox = [_integer(value, "bbox coordinate") for value in bbox_value]
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ExactMaskBBoxError("bbox_xyxy is degenerate or outside the image")
    pixel_count = _integer(entry.get("pixel_count"), "pixel_count")
    if pixel_count <= 0:
        raise ExactMaskBBoxError("an exact visible mask must contain positive pixels")
    if pixel_count > (x1 - x0) * (y1 - y0):
        raise ExactMaskBBoxError("pixel_count exceeds bbox area")
    frame_sha256 = _sha256(entry.get("frame_sha256"), "frame_sha256")
    mask_sha256 = _sha256(entry.get("mask_sha256"), "mask_sha256")
    return {
        "evidence_type": EXACT_EVIDENCE_TYPE,
        "bbox_source": EXACT_BBOX_SOURCE,
        "coordinate_convention": COORDINATE_CONVENTION,
        "object_id": expected_object_id,
        "frame_path": expected_path,
        "frame_index": frame_index,
        "image_size": {"w": width, "h": height},
        "bbox_xyxy": bbox,
        "pixel_count": pixel_count,
        "frame_sha256": frame_sha256,
        "mask_sha256": mask_sha256,
    }


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _display_object(target_oid: str) -> str:
    raw = target_oid.split("|", 1)[0].split("___", 1)[0]
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw).replace("_", " ")
    return value.strip().casefold() or "object"


def _overlay_relative_path(frame_path: str, item_id: str, mask_sha256: str) -> str:
    path = Path(safe_relative_path(frame_path))
    identity = sha256_bytes(
        canonical_json_bytes((item_id, frame_path, mask_sha256, OVERLAY_RGB, OVERLAY_WIDTH))
    )
    name = f"{path.stem}__exactmaskbbox_{identity[:16]}.png"
    return safe_relative_path(str(path.with_name(name)))


def _render_overlay(
    input_path: Path,
    output_path: Path,
    evidence: Mapping[str, Any],
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise ExactMaskBBoxError("Pillow is required to bake bbox overlays") from exc
    with Image.open(input_path) as source:
        image = source.convert("RGB")
    size = evidence["image_size"]
    if image.size != (size["w"], size["h"]):
        raise ExactMaskBBoxError(
            f"decoded image size {image.size!r} disagrees with replay evidence"
        )
    x0, y0, x1, y1 = evidence["bbox_xyxy"]
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (x0, y0, x1 - 1, y1 - 1),
        outline=OVERLAY_RGB,
        width=OVERLAY_WIDTH,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".png", dir=output_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        image.save(temporary, format="PNG", optimize=False, compress_level=9)
        temporary.replace(output_path)
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def bake_exact_mask_bbox_overlays(
    *,
    input_jsonl: str | Path,
    replay_sidecar: str | Path,
    asset_root: str | Path,
    output_asset_root: str | Path,
    output_jsonl: str | Path,
    report_path: str | Path,
) -> dict[str, Any]:
    """Bake exact View-1 overlays and emit a private row copy.

    Validation is completed for every row before any image is written.  The
    private output records a per-frame asset-root binding because the overlay
    and untouched View-2 frame intentionally live under different roots.
    """

    source_path = Path(input_jsonl)
    sidecar_path = Path(replay_sidecar)
    raw_root = Path(asset_root).resolve()
    overlay_root = Path(output_asset_root).resolve()
    destination = Path(output_jsonl)
    audit_path = Path(report_path)
    report: dict[str, Any] = {
        "schema_version": "0.1.0",
        "audit": "taskc_exact_instance_mask_bbox_baker",
        "pass": False,
        "input": {"path": str(source_path)},
        "replay_sidecar": {"path": str(sidecar_path)},
        "output": {"path": str(destination), "written": False},
        "output_asset_root": str(overlay_root),
        "errors": [],
    }
    try:
        if not source_path.is_file():
            raise ExactMaskBBoxError(f"input JSONL does not exist: {source_path}")
        report["input"]["sha256"] = sha256_file(source_path)
        items, sidecar_provenance = load_exact_mask_replay_sidecar(sidecar_path)
        report["replay_sidecar"] = sidecar_provenance
        rows: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        with source_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ExactMaskBBoxError(f"line {line_number} is not a JSON object")
                item_id = str(row.get("id") or "")
                if not item_id or item_id in seen_ids:
                    raise ExactMaskBBoxError(f"line {line_number} has a missing/duplicate ID")
                seen_ids.add(item_id)
                frames = row.get("frames")
                if not isinstance(frames, list) or len(frames) != 2:
                    raise ExactMaskBBoxError(f"{item_id}: expected exactly two frames")
                target_oid = str(row.get("target_oid") or "")
                source_step = _integer(row.get("source_step"), "source_step")
                if not target_oid:
                    raise ExactMaskBBoxError(f"{item_id}: missing target_oid")
                entry = items.get(item_id)
                if entry is None:
                    raise ExactMaskBBoxError(f"{item_id}: missing exact replay evidence")
                evidence = validate_exact_mask_bbox_entry(
                    entry,
                    expected_object_id=target_oid,
                    expected_frame_path=frames[0],
                    expected_frame_index=source_step,
                )
                raw_path = resolve_within(raw_root, frames[0])
                if not raw_path.is_file():
                    raise ExactMaskBBoxError(f"{item_id}: missing raw View-1 frame")
                observed_frame_hash = sha256_file(raw_path)
                if observed_frame_hash != evidence["frame_sha256"]:
                    raise ExactMaskBBoxError(
                        f"{item_id}: raw frame hash mismatch: {observed_frame_hash}"
                    )
                overlay_relative = _overlay_relative_path(
                    frames[0], item_id, evidence["mask_sha256"]
                )
                overlay_path = resolve_within(overlay_root, overlay_relative)
                if overlay_path == raw_path:
                    raise ExactMaskBBoxError(f"{item_id}: overlay would overwrite raw evidence")
                rows.append(
                    {
                        "row": row,
                        "item_id": item_id,
                        "raw_path": raw_path,
                        "overlay_path": overlay_path,
                        "overlay_relative": overlay_relative,
                        "evidence": evidence,
                    }
                )
        if not rows:
            raise ExactMaskBBoxError("input JSONL contains no rows")

        output_rows: list[dict[str, Any]] = []
        overlay_manifest: list[dict[str, Any]] = []
        for record in rows:
            _render_overlay(record["raw_path"], record["overlay_path"], record["evidence"])
            row = copy.deepcopy(record["row"])
            original_frames = list(row["frames"])
            row["frames"][0] = record["overlay_relative"]
            row["question"] = (
                f"The {_display_object(str(row['target_oid']))} is highlighted with a box "
                "in View 1. In View 2, what best describes its visibility?"
            )
            metadata = dict(row.get("metadata") or {})
            answer_class = str(metadata.get("answer_class") or "")
            if answer_class not in CLASS_TO_ANSWER:
                raise ExactMaskBBoxError(
                    f"{record['item_id']}: non-canonical answer_class={answer_class!r}"
                )
            row["answer"] = CLASS_TO_ANSWER[answer_class]
            row["options"] = list(CANONICAL_OPTIONS)
            row["metadata"] = metadata
            private = dict(row.get("_candidate_source") or {})
            private["frame_asset_bindings"] = [
                {"root": str(overlay_root), "path": record["overlay_relative"]},
                {"root": str(raw_root), "path": original_frames[1]},
            ]
            private["exact_mask_bbox_overlay"] = {
                **record["evidence"],
                "overlay_path": record["overlay_relative"],
                "overlay_sha256": sha256_file(record["overlay_path"]),
                "overlay_rgb": list(OVERLAY_RGB),
                "overlay_width": OVERLAY_WIDTH,
            }
            row["_candidate_source"] = private
            output_rows.append(row)
            overlay_manifest.append(
                {
                    "id": record["item_id"],
                    "path": record["overlay_relative"],
                    "sha256": private["exact_mask_bbox_overlay"]["overlay_sha256"],
                }
            )
        payload = b"".join(canonical_json_bytes(row) + b"\n" for row in output_rows)
        _atomic_write_bytes(destination, payload)
        report["counts"] = {
            "input_rows": len(rows),
            "overlay_rows": len(output_rows),
            "unused_sidecar_items": len(set(items) - seen_ids),
        }
        report["overlays"] = overlay_manifest
        report["output"] = {
            "path": str(destination),
            "written": True,
            "rows": len(output_rows),
            "sha256": sha256_file(destination),
        }
        report["pass"] = True
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["output"]["stale_file_present_on_failure"] = destination.exists()
    report_payload = json.dumps(
        report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    _atomic_write_bytes(audit_path, report_payload)
    return report

