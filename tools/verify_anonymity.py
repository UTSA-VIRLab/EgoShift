#!/usr/bin/env python3
"""Fail the release if anything artifact-bound would deanonymise the authors.

WACV review is double-blind, and an identity leak inside a reviewer-visible
artifact is a desk-reject risk no amount of good data recovers from. The
dangerous leaks are rarely the obvious ones -- rater names are pseudonymised
and no email appears anywhere -- but an absolute path like /home/<user>/...
names the author just as surely, and those hide in tooling nobody thinks of as
"the paper".

The terms to search for are DERIVED AT RUNTIME, never written down here: the
account name comes from the environment, and the rater names come from the
private pseudonym map, which is deliberately outside the artifact. A scanner
that hard-codes the names it hunts becomes the leak it exists to prevent --
the first version of this file flagged itself.

Exits non-zero on any hit, so it can gate a publish script.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Ordinary English and technical words. A rater name part that is also one of
# these cannot be told apart from normal prose by string matching, so a bare
# occurrence is reported but does not gate publishing. The full name and any
# distinctive part stay fatal.
COMMON_WORDS = {
    "bin", "van", "lee", "may", "min", "max", "sum", "log", "run", "map",
    "set", "key", "row", "col", "bar", "arm", "bed", "box", "cap", "pan",
    "art", "ray", "sky", "sun", "ice", "tan", "kim", "man", "son", "dan",
    "ann", "eve", "amber", "grace", "hope", "jade", "rose", "faith",
}
PRIVATE_MAP = ROOT / "work/human_study_r2/rater_pseudonyms_PRIVATE.json"

# Everything a reviewer can download.
TARGETS = (
    "release",
    "docs/Human_Annotation_Results.md",
    "work/human_study_r2/scored",
    "work/candidate_r2/taska_candidate_v5/public",
    "work/candidate_r2/taska_candidate_v5/development",
    "work/candidate_r2/taskb_candidate_r2_static_v3/public",
    "work/candidate_r2/taskb_candidate_r2_static_v3/development",
    "work/candidate_r2/taskc_candidate_v10/public",
    "work/candidate_r2/taskc_candidate_v10/development",
    "SHARE_WITH_ANNOTATORS_R2",
)
TEXT_EXT = {".py", ".md", ".json", ".jsonl", ".txt", ".html", ".svg", ".tex",
            ".yaml", ".yml", ".cfg", ".toml"}
# Reported but never fatal: the panel really does evaluate these vendors'
# models, so the strings belong in the config. Surfaced so a human confirms
# each is a model id rather than a stray tooling trace.
EXPECTED = re.compile(r"claude|anthropic|openai|gemini", re.I)


def _secret_terms() -> dict[str, re.Pattern]:
    """Build the search patterns from live sources, not from literals."""

    terms: dict[str, re.Pattern] = {}
    user = os.environ.get("USER") or getpass.getuser()
    if user:
        terms["account name"] = re.compile(rf"\b{re.escape(user)}\b")
    home = str(Path.home())
    if home and home != "/":
        terms["home path"] = re.compile(re.escape(home))
    terms["absolute build path"] = re.compile(re.escape(str(ROOT)))
    terms["email address"] = re.compile(r"[\w.\-]+@[\w\-]+\.[a-z]{2,}", re.I)
    names: list[str] = []
    if PRIVATE_MAP.is_file():
        try:
            payload = json.loads(PRIVATE_MAP.read_text(encoding="utf-8"))
            for value in payload.values():
                if isinstance(value, dict):
                    names.extend(str(k) for k in value)
        except Exception:
            pass
    parts = sorted({p for n in names for p in re.split(r"\s+", n) if len(p) > 2},
                   key=len, reverse=True)
    # A short given name can be an ordinary word. Matching those
    # case-insensitively flags every "#!/usr/bin/env" line and every histogram
    # "bin", which is not a leak and, worse, teaches the operator to skip the
    # report. Short parts that collide with ordinary vocabulary are therefore
    # reported in a separate bucket that does not gate the publish, while the
    # full name and any distinctive part stay fatal.
    distinctive = [p for p in parts if len(p) >= 5 or p.lower() not in COMMON_WORDS]
    colliding = [p for p in parts if p not in distinctive]
    if distinctive:
        terms["rater name"] = re.compile(
            r"\b(?:" + "|".join(re.escape(p) for p in distinctive) + r")\b", re.I
        )
    if colliding:
        terms["~rater name (common word)"] = re.compile(
            r"\b(?:" + "|".join(re.escape(p) for p in colliding) + r")\b", re.I
        )
    # Two parts of one name close together is a person, whatever the words are.
    for full in {n for n in names if len(re.split(r"\s+", n.strip())) > 1}:
        bits = [re.escape(x) for x in re.split(r"\s+", full.strip()) if x]
        terms["rater full name"] = re.compile(
            r"\b" + r"\W{0,3}".join(bits) + r"\b", re.I)
        break
    return terms


def main() -> int:
    patterns = _secret_terms()
    self_path = Path(__file__).resolve()
    hits: dict[str, list[tuple[str, str]]] = defaultdict(list)
    expected_files: set[str] = set()
    scanned = 0
    for target in TARGETS:
        base = ROOT / target
        if not base.exists():
            continue
        paths = [base] if base.is_file() else base.rglob("*")
        for path in paths:
            if not path.is_file() or path.suffix.lower() not in TEXT_EXT:
                continue
            if "__pycache__" in path.parts or path.resolve() == self_path:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            scanned += 1
            relative = str(path.relative_to(ROOT))
            for label, rx in patterns.items():
                for match in sorted(set(rx.findall(text)))[:3]:
                    hits[label].append((relative, str(match)[:60]))
            if EXPECTED.search(text):
                expected_files.add(relative)

    print(f"scanned {scanned} text files across {len(TARGETS)} artifact areas")
    print(f"searching {len(patterns)} derived term(s): {', '.join(sorted(patterns))}\n")
    fatal = 0
    for label in sorted(patterns):
        rows = hits.get(label, [])
        if not label.startswith("~"):
            fatal += len(rows)
        files = sorted({r[0] for r in rows})
        status = "clean" if not rows else f"{len(rows)} hit(s) in {len(files)} file(s)"
        print(f"  {label:<22} {status}")
        for name in files[:6]:
            example = next(r[1] for r in rows if r[0] == name)
            print(f"        {name}   e.g. {example!r}")
    print(f"\n  vendor/model names     expected, not fatal - {len(expected_files)} file(s)")
    for name in sorted(expected_files)[:4]:
        print(f"        {name}")
    verdict = "PASS - safe to publish" if not fatal else f"FAIL - {fatal} leak(s)"
    print(f"\nANONYMITY: {verdict}")
    return 0 if not fatal else 1


if __name__ == "__main__":
    raise SystemExit(main())
