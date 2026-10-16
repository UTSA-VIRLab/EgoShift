#!/usr/bin/env python3
"""Re-parse already-collected raw completions with the current parse_answer.

For when parse_answer changes after rows were written (e.g. it now accepts a
short label like "Out of View." for the option "Out of View -- ..."). No model
is called; only "parsed"/"parse_mode" are recomputed from the stored "raw"
and "option_order". Rows with no raw completion (load/generation errors) are
left untouched.

The results file is written by long-running background jobs, so this only
rewrites the byte range present when it started; anything appended after that
(by a still-running job) is copied through verbatim rather than clobbered.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from egoshift.model_prompting import parse_answer  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path,
                     default=Path("work/model_panel/results_local.jsonl"))
    ap.add_argument("--task", default=None,
                     help="Only rescore rows for this task, e.g. C (default: all)")
    args = ap.parse_args()

    with args.results.open("rb") as fh:
        snapshot = fh.read()
    cut = snapshot.rfind(b"\n") + 1  # only fully-written lines; 0 if none found
    processed, tail = snapshot[:cut], snapshot[cut:]

    backup = args.results.with_suffix(args.results.suffix + ".bak")
    backup.write_bytes(snapshot)

    out = []
    changed = 0
    before, after = {}, {}
    for line in processed.decode("utf-8").splitlines():
        if not line.strip():
            out.append(line)
            continue
        r = json.loads(line)
        if (args.task and r.get("task") != args.task) or r.get("raw") is None:
            out.append(line)
            continue
        before[r.get("parse_mode")] = before.get(r.get("parse_mode"), 0) + 1
        parsed, how = parse_answer(r["raw"], r["option_order"])
        after[how] = after.get(how, 0) + 1
        if parsed != r.get("parsed") or how != r.get("parse_mode"):
            changed += 1
            r["parsed"] = parsed
            r["parse_mode"] = how
            line = json.dumps(r)
        out.append(line)

    new_content = ("\n".join(out) + "\n").encode("utf-8") if out else b""

    fd, tmp_path = tempfile.mkstemp(dir=args.results.parent, prefix=".rescore_")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(new_content)
            # Re-read anything appended by a live job since the snapshot, and
            # any trailing partial line the snapshot cut off, so nothing new
            # is lost to the rewrite.
            with args.results.open("rb") as fh:
                fh.seek(len(processed))
                tmp.write(fh.read())
        os.replace(tmp_path, args.results)
    except BaseException:
        os.unlink(tmp_path)
        raise

    print(f"backed up pre-rescore snapshot to {backup}")
    print(f"rescored {sum(before.values())} eligible rows, {changed} changed "
          f"(of {processed.count(chr(10).encode())} lines snapshotted; "
          f"{len(tail)} trailing bytes not yet newline-terminated were passed through)")
    print("before:", dict(sorted(before.items(), key=lambda kv: -kv[1])))
    print("after: ", dict(sorted(after.items(), key=lambda kv: -kv[1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
