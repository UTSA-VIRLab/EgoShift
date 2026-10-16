#!/usr/bin/env python3
"""Remove result rows for one (model, task) so a retry doesn't skip them.

run_local_models.py's resume logic marks an item "done" the moment any row
for (model, arm, item_id, task, split) exists in the output file -- even a
row that recorded a generation error. A model that OOM'd on every item in a
run therefore looks "already complete" to a naive retry and gets silently
skipped. This drops those rows first so the retry actually attempts them.

Race-safe against a still-running job appending to the same file: only the
byte range present at start is filtered; anything appended since (or a
trailing partial line) is copied through unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                     default=Path("work/model_panel/results_local.jsonl"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=("A", "B", "C"))
    ap.add_argument("--only-errors", action="store_true",
                     help="Drop only rows with a non-null error field, keeping any real answers.")
    args = ap.parse_args()

    with args.results.open("rb") as fh:
        snapshot = fh.read()
    cut = snapshot.rfind(b"\n") + 1
    processed, tail = snapshot[:cut], snapshot[cut:]

    backup = args.results.with_suffix(args.results.suffix + ".pre_purge.bak")
    backup.write_bytes(snapshot)

    removed = 0
    out = []
    for line in processed.decode("utf-8").splitlines():
        if not line.strip():
            out.append(line)
            continue
        r = json.loads(line)
        if r.get("model") == args.model and r.get("task") == args.task:
            if not args.only_errors or r.get("error") is not None:
                removed += 1
                continue
        out.append(line)

    new_content = ("\n".join(out) + "\n").encode("utf-8") if out else b""
    fd, tmp_path = tempfile.mkstemp(dir=args.results.parent, prefix=".purge_")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(new_content)
            with args.results.open("rb") as fh:
                fh.seek(len(processed))
                tmp.write(fh.read())
        os.replace(tmp_path, args.results)
    except BaseException:
        os.unlink(tmp_path)
        raise

    print(f"backed up pre-purge snapshot to {backup}")
    print(f"removed {removed} row(s) for model={args.model} task={args.task}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
