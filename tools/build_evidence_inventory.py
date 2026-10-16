#!/usr/bin/env python3
"""Freeze hashes for the evidence that supports the 2026-08-27 review.

The private ledger retains paths (including identifying human-intake names).
The public summary exposes only root-level counts, byte totals, and aggregate
digests.  This tool never changes an input and refuses to overwrite a freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path


ROOTS = (
    ("task_a_candidate_v5", Path("work/candidate_r2/taska_candidate_v5")),
    ("task_b_candidate_static_v3", Path("work/candidate_r2/taskb_candidate_r2_static_v3")),
    ("task_c_candidate_v10", Path("work/candidate_r2/taskc_candidate_v10")),
    ("candidate_audits", Path("work/candidate_r2/audits")),
    ("model_panel_evidence", Path("work/model_panel")),
    ("human_evidence_private", Path("work/human_study_r2")),
    ("anonymous_archives", Path("work/artifact_u7")),
    ("release_source", Path("release")),
    ("paper_source", Path("paper")),
    ("review_documents", Path("docs")),
)

SKIP_PARTS = {"__pycache__", ".pytest_cache", ".venv_molmo2"}
SKIP_SUFFIXES = {".pyc", ".lock"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def files_under(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and not (set(path.parts) & SKIP_PARTS)
        and path.suffix not in SKIP_SUFFIXES
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir", type=Path, default=Path("work/evidence_freeze_20260827")
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite evidence freeze: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    private_lines = []
    summaries = {}
    total_files = 0
    total_bytes = 0
    for label, root in ROOTS:
        if not root.is_dir():
            summaries[label] = {"status": "absent", "path": str(root)}
            continue
        aggregate = hashlib.sha256()
        file_count = 0
        byte_count = 0
        for path in files_under(root):
            relative = path.relative_to(root).as_posix()
            size = path.stat().st_size
            digest = sha256_file(path)
            private_lines.append(f"{digest}  {label}/{relative}\n")
            aggregate.update(relative.encode("utf-8") + b"\0")
            aggregate.update(str(size).encode("ascii") + b"\0")
            aggregate.update(digest.encode("ascii") + b"\n")
            file_count += 1
            byte_count += size
        summaries[label] = {
            "status": "hashed",
            "files": file_count,
            "bytes": byte_count,
            "root_digest_sha256": aggregate.hexdigest(),
        }
        total_files += file_count
        total_bytes += byte_count
        print(f"{label}: {file_count:,} files, {byte_count:,} bytes", flush=True)

    private_path = args.out_dir / "files_PRIVATE.sha256"
    private_path.write_text("".join(private_lines), encoding="utf-8")
    os.chmod(private_path, 0o600)
    private_digest = sha256_file(private_path)
    summary = {
        "schema_version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "evidence_snapshot_not_final_release",
        "scope": (
            "Canonical A/B/C candidates, candidate audits, model-panel evidence, "
            "human evidence, anonymous archives, release/paper source, and review documents. "
            "Exploratory mining/render caches outside these roots are not included."
        ),
        "privacy": (
            "files_PRIVATE.sha256 contains identifying intake filenames and must not be released."
        ),
        "private_ledger_sha256": private_digest,
        "total_files": total_files,
        "total_bytes": total_bytes,
        "roots": summaries,
    }
    (args.out_dir / "SUMMARY.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "total_files": total_files,
        "total_bytes": total_bytes,
        "private_ledger_sha256": private_digest,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
