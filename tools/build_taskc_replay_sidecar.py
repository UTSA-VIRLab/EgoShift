#!/usr/bin/env python3
"""Emit the Task-C exact-mask replay sidecar from the miner's own evidence.

Earlier candidates replayed the selected rows back through AI2-THOR to recover
an instance mask per referent. The all-class miner already does that work at
render time: it reads the instance mask for the target in View 1, records the
non-zero extent as ``gold_evidence.source.bbox_xyxy``, and binds both the frame
and the mask by SHA-256. Replaying would re-derive numbers this file already
holds, so this tool reshapes them into the sidecar envelope the baker demands
instead -- and verifies each frame on disk still hashes to what the miner
recorded, so a drifted asset cannot reach the overlay.

Only rows carrying complete View-1 mask evidence are emitted. A row without it
is reported, never guessed at: an approximate box is exactly what the baker's
``bbox_source`` guard exists to reject.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from egoshift.exact_mask_bbox import (  # noqa: E402
    COORDINATE_CONVENTION,
    EXACT_BBOX_SOURCE,
    EXACT_EVIDENCE_TYPE,
)
from egoshift.hashing import sha256_file  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument(
        "--verify-frames",
        type=int,
        default=0,
        help=(
            "Re-hash this many referent frames on disk against the value the "
            "miner recorded (0 verifies every frame)."
        ),
    )
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.selection.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    items: dict[str, dict] = {}
    skipped: list[dict] = []
    verified = 0
    limit = args.verify_frames or len(rows)
    for row in rows:
        source = row["_candidate_source"]
        evidence = (source.get("gold_evidence") or {}).get("source") or {}
        bbox = evidence.get("bbox_xyxy")
        frame_sha256 = source.get("source_frame_sha256")
        mask_sha256 = evidence.get("mask_sha256")
        pixels = evidence.get("pixel_count")
        if not (bbox and frame_sha256 and mask_sha256 and pixels):
            skipped.append({"id": row["id"], "reason": "incomplete view-1 mask evidence"})
            continue
        frame_relative = row["frames"][0]
        if verified < limit:
            asset_root = Path(source["frame_asset_root"])
            observed = sha256_file(asset_root / frame_relative)
            if observed != frame_sha256:
                raise SystemExit(f"frame hash drift for {row['id']}: {frame_relative}")
            verified += 1
        items[row["id"]] = {
            "bbox_source": EXACT_BBOX_SOURCE,
            "bbox_xyxy": [int(value) for value in bbox],
            "coordinate_convention": COORDINATE_CONVENTION,
            "evidence_type": EXACT_EVIDENCE_TYPE,
            "frame_index": int(row["source_step"]),
            "frame_path": frame_relative,
            "frame_sha256": frame_sha256,
            "image_size": {"h": 480, "w": 640},
            "mask_sha256": mask_sha256,
            "object_id": row["target_oid"],
            "pixel_count": int(pixels),
        }

    payload = {
        "coordinate_convention": COORDINATE_CONVENTION,
        "evidence_type": EXACT_EVIDENCE_TYPE,
        "items": items,
        "provenance": "miner-recorded instance-mask extent; no re-replay",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")

    report = {
        "audit": "taskc_replay_sidecar_from_miner_evidence",
        "selection": {"path": str(args.selection), "sha256": sha256_file(args.selection)},
        "rows": len(rows),
        "items": len(items),
        "frames_hash_verified": verified,
        "skipped": skipped[:20],
        "skipped_total": len(skipped),
        "output": {"path": str(args.output), "sha256": sha256_file(args.output)},
        "pass": not skipped,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("rows", "items", "frames_hash_verified", "skipped_total", "pass")}, indent=1))
    return 0 if not skipped else 1


if __name__ == "__main__":
    raise SystemExit(main())
