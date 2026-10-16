#!/usr/bin/env python3
"""Inventory current model rows, reparse raw text, and expose key conflicts.

This is a non-mutating audit snapshot, not the final canonical result table.
It deliberately refuses to choose among conflicting valid responses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from egoshift.model_prompting import parse_answer  # noqa: E402


def row_sha256(row: dict) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("work/model_panel"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("work/model_panel/registry_audit_20260827_v2")
    )
    args = parser.parse_args()
    if args.out_dir.exists():
        raise SystemExit(f"refusing to overwrite registry audit: {args.out_dir}")
    args.out_dir.mkdir(parents=True)

    paths = sorted(args.results_dir.glob("results*.jsonl"))
    entries = []
    by_key = defaultdict(list)
    files = []
    parse_drift = []
    parsed_answer_drift = 0
    parse_mode_drift = 0
    field_presence = Counter()
    for path in paths:
        files.append({"path": str(path), "sha256": file_sha256(path)})
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("item_id") is None:
                    continue
                for field in ("request_id", "run_id", "execution", "usage", "model_id"):
                    field_presence[field] += int(row.get(field) is not None)
                key = (
                    str(row.get("model")), str(row.get("task")), str(row.get("arm")),
                    str(row.get("item_id")), int(row.get("permutation", 0)),
                )
                reparsed = None
                reparse_mode = "not_reparsed"
                if row.get("error") is None and isinstance(row.get("raw"), str):
                    reparsed, reparse_mode = parse_answer(row["raw"], row.get("option_order") or [])
                    if reparsed != row.get("parsed") or reparse_mode != row.get("parse_mode"):
                        parsed_answer_drift += int(reparsed != row.get("parsed"))
                        parse_mode_drift += int(reparse_mode != row.get("parse_mode"))
                        parse_drift.append({
                            "source": str(path), "line": line_number,
                            "row_sha256": row_sha256(row),
                            "stored_parsed": row.get("parsed"), "reparsed": reparsed,
                            "stored_mode": row.get("parse_mode"), "reparse_mode": reparse_mode,
                        })
                entry = {
                    "source": str(path),
                    "line": line_number,
                    "row_sha256": row_sha256(row),
                    "key": {
                        "model": key[0], "task": key[1], "arm": key[2],
                        "item_id": key[3], "permutation": key[4],
                    },
                    "error": row.get("error"),
                    "stored_parsed": row.get("parsed"),
                    "reparsed": reparsed,
                    "reparse_mode": reparse_mode,
                }
                entries.append(entry)
                by_key[key].append(entry)

    duplicate_groups = []
    conflicting_groups = []
    stored_conflicting_groups = []
    duplicate_nonerror_rows = 0
    for key, group in sorted(by_key.items()):
        valid = [entry for entry in group if entry["error"] is None]
        if len(valid) <= 1:
            continue
        duplicate_nonerror_rows += len(valid) - 1
        record = {
            "key": valid[0]["key"],
            "valid_rows": len(valid),
            "row_sha256": [entry["row_sha256"] for entry in valid],
            "reparsed_values": sorted({str(entry["reparsed"]) for entry in valid}),
        }
        duplicate_groups.append(record)
        if len({entry["reparsed"] for entry in valid}) > 1:
            conflicting_groups.append(record)
        if len({entry["stored_parsed"] for entry in valid}) > 1:
            stored_conflicting_groups.append(record)

    registry_path = args.out_dir / "raw_registry_index.jsonl"
    registry_path.write_text(
        "".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries),
        encoding="utf-8",
    )
    report = {
        "schema_version": "1.0.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "interim_audit_not_canonical_results",
        "policy": "No conflicting valid row is selected by this audit.",
        "parser_source": {
            "path": "release/src/egoshift/model_prompting.py",
            "sha256": file_sha256(Path("release/src/egoshift/model_prompting.py")),
        },
        "source_files": files,
        "rows": len(entries),
        "unique_keys": len(by_key),
        "duplicate_nonerror_rows_beyond_first": duplicate_nonerror_rows,
        "duplicate_nonerror_key_groups": len(duplicate_groups),
        "conflicting_reparsed_key_groups": len(conflicting_groups),
        "conflicting_stored_parsed_key_groups": len(stored_conflicting_groups),
        "stored_vs_current_parser_drift_rows": len(parse_drift),
        "stored_vs_current_parsed_answer_drift_rows": parsed_answer_drift,
        "stored_vs_current_parse_mode_drift_rows": parse_mode_drift,
        "metadata_field_presence": dict(sorted(field_presence.items())),
        "registry_index_sha256": file_sha256(registry_path),
        "duplicate_groups": duplicate_groups,
        "conflicting_groups": conflicting_groups,
        "parse_drift": parse_drift,
    }
    (args.out_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        key: report[key] for key in (
            "rows", "unique_keys", "duplicate_nonerror_rows_beyond_first",
            "duplicate_nonerror_key_groups", "conflicting_reparsed_key_groups",
            "conflicting_stored_parsed_key_groups",
            "stored_vs_current_parser_drift_rows",
            "stored_vs_current_parsed_answer_drift_rows",
            "stored_vs_current_parse_mode_drift_rows", "metadata_field_presence",
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
