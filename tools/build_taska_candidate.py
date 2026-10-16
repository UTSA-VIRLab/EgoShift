#!/usr/bin/env python3
"""Materialize the Task-A candidate: baked boxes, order symmetry, public split.

Consumes the cue-matched selection and emits, per item:
  - two public frames with the referent box burned in (same overlay style as
    Task C: exact mask-derived bbox, shared OVERLAY_RGB / OVERLAY_WIDTH);
  - a deterministic 50% view-order swap (stable-hash parity per item). The
    presented view order is thereby symmetric by construction, which kills
    every signed view-order statistic exactly -- a probe read 54% from the
    signed coverage difference, because a *different* item's second-rendered
    view always showed the second instance;
  - strictly allowlisted public question rows;
  - a private gold JSONL retaining evidence, the swap flag, and provenance.

Frame bytes are verified against the miner's recorded SHA-256 before baking:
this tool must not silently materialize from drifted assets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from egoshift.exact_mask_bbox import OVERLAY_RGB, OVERLAY_WIDTH  # noqa: E402
from egoshift.hashing import sha256_file  # noqa: E402

PUBLIC_FIELDS = ("id", "task_family", "format", "frames", "question", "options")


def _display_noun(object_type: str) -> str:
    """CamelCase object type -> plain noun, e.g. DiningTable -> dining table."""

    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(object_type))
    return " ".join(spaced.split()).casefold() or "object"


def _plural(word: str) -> str:
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    if word.endswith("y") and word[-2:-1] not in "aeiou":
        return word[:-1] + "ies"
    return word + "s"


def _options_for(object_type: str) -> tuple[str, str]:
    """Return (same_text, different_text) as parallel, noun-naming options.

    The miner emitted a generic positive ("Same object seen from different
    viewpoints") against a noun-naming negative ("Two different beds"). That
    asymmetry cannot change which answer is correct -- it is identical in every
    item -- but surface form was then perfectly correlated with the class, so
    any preference for the more specific or shorter option becomes a class bias
    that is indistinguishable from a real perceptual tendency in the confusion
    matrix. It is the same hazard as fixed option position, and it also makes
    the two options harder for a rater to compare. Both options now name the
    object and share a parallel shape.
    """

    noun = _display_noun(object_type)
    return f"The same {noun} seen from two viewpoints", f"Two different {_plural(noun)}"


def _swap(item: dict) -> bool:
    stable = item["_candidate_source"]["stable_identity_sha256"]
    return int(hashlib.sha256(f"vieworder|{stable}".encode()).hexdigest()[:8], 16) % 2 == 1


def _bake(source: Path, destination: Path, bbox: list[int]) -> str:
    from PIL import Image, ImageDraw

    with Image.open(source) as handle:
        image = handle.convert("RGB")
    x0, y0, x1, y1 = bbox
    ImageDraw.Draw(image).rectangle(
        (x0, y0, x1 - 1, y1 - 1), outline=OVERLAY_RGB, width=OVERLAY_WIDTH
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination, format="PNG", optimize=False, compress_level=9)
    return sha256_file(destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--audit-report", required=True, type=Path)
    parser.add_argument(
        "--reuse-baked-from",
        type=Path,
        default=None,
        help=(
            "An earlier candidate root whose public frames may be hard-linked "
            "instead of re-baked. Sound only because a public frame's path and "
            "content are both fully determined by (item id, presented view "
            "index, source frame, box, view-order swap) -- none of which this "
            "tool's later changes touch. Each reused file is hash-checked "
            "against the freshly computed value for the first item, and any "
            "path absent from the donor is baked normally."
        ),
    )
    args = parser.parse_args()
    if args.output_root.exists() or args.audit_report.exists():
        raise SystemExit("refusing to overwrite candidate output or audit report")

    rows = [
        json.loads(line)
        for line in args.selection.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selection_sha256 = sha256_file(args.selection)

    # Balanced option order. The miner emits [same, different] for every item,
    # so the correct answer's position was fully determined by the class: a
    # model with first-option bias would answer "same" everywhere and score a
    # flattering 50% with a confusion matrix that looks like a real preference.
    # Within each (role, class) group, alternate the two orders in stable-hash
    # order, so position is exactly 50/50 inside each class and carries no
    # information about the answer.
    order_swapped: dict[str, bool] = {}
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["candidate_role"], row["candidate_answer_class"])].append(row)
    for group in grouped.values():
        group.sort(key=lambda r: r["_candidate_source"]["stable_identity_sha256"])
        for index, row in enumerate(group):
            order_swapped[row["id"]] = bool(index % 2)

    public_rows: dict[str, list[dict]] = {}
    private_rows: dict[str, list[dict]] = {}
    counts: Counter = Counter()
    swap_count = 0
    option_swap_count = 0
    reused = [0]
    baked = [0]
    verified = [0]
    for row in rows:
        source = row["_candidate_source"]
        evidence = source["gold_evidence"]
        asset_root = Path(source["frame_asset_root"])
        role = row["candidate_role"]
        item_id = row["id"]
        originals = [asset_root / relative for relative in row["frames"]]
        for path, expected in zip(originals, source["frame_sha256"]):
            observed = sha256_file(path)
            if observed != expected:
                raise SystemExit(f"frame hash drift: {path}")
        boxes = [evidence["view_1"]["bbox_xyxy"], evidence["view_2"]["bbox_xyxy"]]
        swapped = _swap(row)
        order = (1, 0) if swapped else (0, 1)
        swap_count += 1 if swapped else 0
        public_relatives = []
        for presented, actual in enumerate(order, start=1):
            relative = f"public/frames/{role}/{item_id}_view{presented}.png"
            destination = args.output_root / relative
            donor = (
                (args.reuse_baked_from / relative)
                if args.reuse_baked_from is not None
                else None
            )
            if donor is not None and donor.is_file():
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(donor, destination)
                except OSError:
                    shutil.copyfile(donor, destination)
                reused[0] += 1
                if verified[0] < 3:
                    # Prove the donor really is what baking would produce.
                    check = destination.with_suffix(".verify.png")
                    fresh = _bake(originals[actual], check, boxes[actual])
                    if fresh != sha256_file(destination):
                        raise SystemExit(
                            f"reused frame differs from a fresh bake: {relative}"
                        )
                    check.unlink()
                    verified[0] += 1
            else:
                _bake(originals[actual], destination, boxes[actual])
                baked[0] += 1
            public_relatives.append(relative)
        same_text, different_text = _options_for(
            row["metadata"]["object_type"]
        )
        answer_text = (
            same_text if row["candidate_answer_class"] == "same" else different_text
        )
        options = [same_text, different_text]
        if order_swapped[item_id]:
            options.reverse()
            option_swap_count += 1
        public_rows.setdefault(role, []).append(
            {
                "id": item_id,
                "task_family": row["task_family"],
                "format": row["format"],
                "frames": public_relatives,
                "question": row["question"],
                "options": options,
            }
        )
        private_rows.setdefault(role, []).append(
            {
                "id": item_id,
                "candidate_role": role,
                "answer": answer_text,
                "candidate_answer_class": row["candidate_answer_class"],
                "view_order_swapped": swapped,
                "option_order_swapped": order_swapped[item_id],
                "answer_index": options.index(answer_text),
                "scene_id": row["scene_id"],
                "metadata": row["metadata"],
                "_candidate_source": source,
            }
        )
        counts[(role, row["candidate_answer_class"])] += 1

    outputs = {}
    for kind, per_role in (("questions", public_rows), ("gold_private", private_rows)):
        for role, items in sorted(per_role.items()):
            relative = (
                f"public/questions/task_a_{role}.jsonl"
                if kind == "questions"
                else f"private/task_a_{role}_gold.jsonl"
            )
            path = args.output_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for item in sorted(items, key=lambda entry: entry["id"]):
                    handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
            outputs[relative] = {"rows": len(items), "sha256": sha256_file(path)}

    for role, items in public_rows.items():
        for item in items:
            extra = set(item) - set(PUBLIC_FIELDS)
            if extra:
                raise SystemExit(f"public field violation: {sorted(extra)}")

    payload = {
        "schema_version": "0.1.0",
        "audit": "taska_candidate_build",
        "selection": {"path": str(args.selection), "sha256": selection_sha256},
        "items": sum(counts.values()),
        "by_role_class": {f"{r}:{c}": n for (r, c), n in sorted(counts.items())},
        "view_order_swapped_fraction": round(swap_count / max(1, sum(counts.values())), 6),
        "option_order_swapped_fraction": round(
            option_swap_count / max(1, sum(counts.values())), 6
        ),
        "frames": {
            "baked": baked[0],
            "reused_from_prior_candidate": reused[0],
            "reused_hash_verified_against_fresh_bake": verified[0],
            "donor": str(args.reuse_baked_from) if args.reuse_baked_from else None,
        },
        "overlay": {"rgb": list(OVERLAY_RGB), "width": OVERLAY_WIDTH},
        "public_fields": list(PUBLIC_FIELDS),
        "outputs": outputs,
        "pass": True,
    }
    args.audit_report.parent.mkdir(parents=True, exist_ok=True)
    args.audit_report.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: payload[k] for k in (
        "items", "by_role_class", "view_order_swapped_fraction"
    )}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
