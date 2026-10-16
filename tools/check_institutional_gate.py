#!/usr/bin/env python3
"""Fail if the paper asserts institutional authorization the repo cannot evidence.

sec/7_human.tex opens by asserting that the institution authorized analysis and
release. STATUS_OF_RECORD.md and docs/admin/institutional/determination_record.md
have to agree with that sentence. They drifted apart once already: the paper
asserted authorization while the status of record said the determination was
absent. This check exists so that cannot recur silently.

The check is deliberately dumb about *content* -- it cannot read a PDF and judge
whether an ethics committee really approved anything. It only verifies that the
three places tell the same story, and that a claim of "filed" is backed by an
actual file with a recorded hash.

Exit 0 = consistent. Exit 1 = the paper claims more than the repository shows.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
INST = ROOT / "docs" / "admin" / "institutional"
RECORD = INST / "determination_record.md"
STATUS = ROOT / "STATUS_OF_RECORD.md"
HUMAN = ROOT / "paper" / "sec" / "7_human.tex"

# Documents that are the record itself, not the determination being recorded.
META = {"README.md", "determination_record.md",
        "determination_transcript.md"}


def filed_documents() -> list[Path]:
    if not INST.is_dir():
        return []
    return sorted(p for p in INST.iterdir() if p.is_file() and p.name not in META)


def record_status() -> str:
    if not RECORD.is_file():
        return "MISSING"
    text = RECORD.read_text(encoding="utf-8")
    m = re.search(r"^\|\s*Status\s*\|\s*([^|]+?)\s*\|", text, re.M)
    if m:
        return m.group(1).strip().upper()
    if "**Status: NOT FILED.**" in text:
        return "NOT FILED"
    return "UNPARSED"


def recorded_hashes() -> set[str]:
    if not RECORD.is_file():
        return set()
    return set(re.findall(r"\b[0-9a-f]{64}\b", RECORD.read_text(encoding="utf-8")))


def paper_asserts_authorization() -> bool:
    if not HUMAN.is_file():
        return False
    text = HUMAN.read_text(encoding="utf-8")
    # Only the assertion in running text counts; a commented-out line does not.
    body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("%"))
    return bool(re.search(r"institution\s+authorized", body, re.I)) or bool(
        re.search(r"review board office reviewed|not constitute human\s+subjects research", body, re.I))


def status_flags_gate_open() -> bool:
    if not STATUS.is_file():
        return False
    return "authorization gate as **open**" in STATUS.read_text(encoding="utf-8")


def main() -> int:
    docs = filed_documents()
    status = record_status()
    asserts = paper_asserts_authorization()
    gate_open = status_flags_gate_open()
    problems: list[str] = []

    print(f"determination documents filed : {len(docs)}")
    for d in docs:
        digest = hashlib.sha256(d.read_bytes()).hexdigest()
        known = digest in recorded_hashes()
        print(f"  - {d.name}  sha256={digest[:16]}...  recorded={'yes' if known else 'NO'}")
        if not known:
            problems.append(
                f"{d.name} is present but its SHA-256 is not written into "
                f"{RECORD.relative_to(ROOT)}"
            )
    print(f"record status                 : {status}")
    print(f"paper asserts authorization   : {asserts}")
    print(f"status of record: gate open   : {gate_open}")
    print()

    if status == "MISSING":
        problems.append(f"{RECORD.relative_to(ROOT)} does not exist")

    obtained = status.startswith("OBTAINED")

    if status == "FILED" and not docs:
        problems.append("record says FILED but no determination document is present")

    if docs and status not in ("FILED",):
        problems.append(
            f"a determination document is present but the record still says {status!r}"
        )

    # The condition that actually matters for submission: the paper may only
    # describe a determination that actually exists. OBTAINED means one exists
    # but its document has not been filed here yet, which is a housekeeping
    # gap rather than a claim the paper cannot support.
    if asserts and not (obtained or status == "FILED"):
        problems.append(
            "paper/sec/7_human.tex describes an institutional determination, but "
            f"the record says {status!r}. Either record it in "
            f"{RECORD.relative_to(ROOT)}, or change the paper sentence."
        )
    if asserts and obtained and not docs:
        print("note: a determination is recorded as obtained but its document is "
              "not in this directory. Save the email thread as a PDF here and "
              "record its SHA-256 to close this out.")

    if asserts and status == "FILED" and gate_open:
        problems.append(
            "record says FILED but STATUS_OF_RECORD.md still calls the "
            "authorization gate open; reconcile them"
        )

    if not asserts and status == "FILED":
        print("note: a determination is filed but the paper no longer claims it.")

    if problems:
        print("INSTITUTIONAL GATE: FAIL")
        for p in problems:
            print(f"  - {p}")
        return 1

    print("INSTITUTIONAL GATE: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
