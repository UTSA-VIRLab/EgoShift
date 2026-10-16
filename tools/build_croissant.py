#!/usr/bin/env python3
"""Emit Croissant metadata, including the Responsible AI fields, for EgoShift.

The Evaluations and Datasets track expects a machine-readable Croissant record
alongside the data. Everything here is read from the built candidate rather than
typed by hand, so the counts in the metadata cannot drift from the counts in the
paper. Held-out answers are described but not distributed, which the record
states explicitly rather than leaving a reviewer to infer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = {
    "A": (ROOT / "work/candidate_r2/taska_candidate_v5", "cross-view identity"),
    "B": (ROOT / "work/candidate_r2/taskb_candidate_r2_static_v3", "egocentric bearing after self-rotation"),
    "C": (ROOT / "work/candidate_r2/taskc_candidate_v10", "post-motion visibility"),
}
SPLITS = ("hidden_test", "public_dev", "tuning_dev")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=ROOT / "work/artifact_u7/croissant.json")
    p.add_argument("--url", default="TO_BE_FILLED_AT_UPLOAD",
                   help="public or anonymous record URL once the archive is hosted")
    args = p.parse_args()

    distribution = [{
        "@type": "cr:FileObject",
        "@id": "reviewer-bundle",
        "name": "reviewer-bundle",
        "description": ("Questions for every split, development-split answers, build "
                        "audits, raw model completions, redacted human aggregates, and "
                        "a seeded frame sample. Held-out answers are not included."),
        "contentUrl": args.url,
        "encodingFormat": "application/gzip",
    }]
    bundle = ROOT / "work/artifact_u7/egoshift_reviewer_bundle.tar.gz"
    if bundle.exists():
        distribution[0]["sha256"] = sha256(bundle)
        distribution[0]["contentSize"] = f"{bundle.stat().st_size} B"

    record_sets = []
    for task, (root, blurb) in CANDIDATES.items():
        for split in SPLITS:
            qpath = root / f"public/questions/task_{task.lower()}_{split}.jsonl"
            n = count(qpath)
            if not n:
                continue
            held = split == "hidden_test"
            record_sets.append({
                "@type": "cr:RecordSet",
                "@id": f"task-{task.lower()}-{split.replace('_', '-')}",
                "name": f"task_{task.lower()}_{split}",
                "description": (f"Task {task}, {blurb}. {n} multiple-choice items. "
                                + ("Answers are withheld; the items are distributed so "
                                   "that systems can be evaluated on them."
                                   if held else
                                   "Answers are distributed with the split.")),
                "field": [
                    {"@type": "cr:Field", "@id": f"task-{task.lower()}-{split}/id",
                     "name": "id", "description": "stable item identifier",
                     "dataType": "sc:Text"},
                    {"@type": "cr:Field", "@id": f"task-{task.lower()}-{split}/question",
                     "name": "question", "description": "question text shown to the system",
                     "dataType": "sc:Text"},
                    {"@type": "cr:Field", "@id": f"task-{task.lower()}-{split}/options",
                     "name": "options", "description": "answer options in presentation order",
                     "dataType": "sc:Text"},
                    {"@type": "cr:Field", "@id": f"task-{task.lower()}-{split}/frames",
                     "name": "frames", "description": "one or two rendered views, 640x480",
                     "dataType": "sc:ImageObject"},
                ],
            })

    doc = {
        "@context": {
            "@vocab": "https://schema.org/",
            "sc": "https://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "dct": "http://purl.org/dc/terms/",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "field": "cr:field",
            "recordSet": "cr:recordSet",
        },
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "name": "EgoShift",
        "description": (
            "A simulator-grounded diagnostic of egocentric spatial updating in "
            "vision-language models. Task B states an in-place camera rotation, "
            "withholds the post-action view, and asks for the target's egocentric "
            "bearing. Task C asks whether a previously seen target is visible, out "
            "of view, or occluded after camera motion. Task A is a cross-view "
            "identity positive control. Items are built so that a declared adversary "
            "seeing only rotation magnitude and turn direction cannot exceed a "
            "per-cell bound fixed before evaluation."),
        "license": "https://creativecommons.org/licenses/by/4.0/",
        "url": args.url,
        "version": "candidate-r2",
        "citeAs": "See the accompanying paper; anonymous during review.",
        "distribution": distribution,
        "recordSet": record_sets,

        "rai:dataCollection": (
            "Scenes are procedurally generated interiors from ProcTHOR-10k, rendered "
            "with AI2-THOR CloudRendering at 640x480. Camera poses are sampled under a "
            "declared source-bearing envelope. No data is scraped and no third-party "
            "annotation service was used."),
        "rai:dataCollectionType": "Simulated",
        "rai:dataAnnotationProtocol": (
            "Labels are derived from simulator state and verified by replay against "
            "exact instance masks and camera frusta, not by human annotation. A trained "
            "panel answered a 120-item subset per task to estimate human performance; "
            "those responses are a reference measurement and were never used to set or "
            "amend any label."),
        "rai:dataUseCases": (
            "Evaluating spatial updating, cross-view identity and visibility reasoning "
            "in vision-language models, and reusing the construction tools to build "
            "other action-conditioned benchmarks with a declared adversary."),
        "rai:dataLimitations": (
            "All imagery is synthetic and rendered from one camera configuration and "
            "field of view, with a small family of in-place yaw rotations. It contains "
            "no photorealism, sensor noise, hand-held motion, translation, pitch or "
            "multi-step navigation, and supports no claim about transfer to natural "
            "images or embodied deployment. Task A is close to saturated and is a "
            "control rather than a challenge. Items sharing a physical scene are not "
            "independent, so uncertainty estimates should cluster on scenes."),
        "rai:dataBiases": (
            "Answer classes are balanced globally and, for Task B, within each "
            "rotation-by-turn cell, but the feasible answer set differs by cell: no "
            "item can be labelled behind at 30 or 60 degrees. Pooling across cells "
            "therefore mixes populations with different label supports and should be "
            "avoided. Object categories follow the AI2-THOR asset library and are not "
            "representative of household objects in general."),
        "rai:personalSensitiveInformation": (
            "None. Rendered scenes contain no people, faces, or personal information. "
            "Human-study response files, participant names and the pseudonym map are "
            "excluded from every distributed archive."),
        "rai:dataSocialImpact": (
            "The intended impact is better-calibrated reporting of spatial reasoning "
            "ability. A score here does not license a safety claim about a system "
            "operating in a real environment."),
        "rai:dataReleaseMaintenancePlan": (
            "Any change to items, gold labels or prompts produces a new candidate "
            "identifier and invalidates results tied to the previous one, so scores "
            "remain comparable only within a candidate. Superseded candidates are "
            "recorded rather than silently replaced."),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=False) + "\n")
    print(f"wrote {args.out}")
    print(f"  record sets: {len(record_sets)}")
    print(f"  items described: {sum(count(CANDIDATES[t][0] / f'public/questions/task_{t.lower()}_{s}.jsonl') for t in CANDIDATES for s in SPLITS)}")
    print(f"  RAI fields: {sum(1 for k in doc if k.startswith('rai:'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
