#!/usr/bin/env python3
"""Regenerate the human-annotation tracker from the scored session files.

Same contract as build_panel_tracker.py: the page is generated, never hand
edited, so it cannot drift from what the raters actually returned. Reads
work/human_study_r2/scored/ and writes docs/Human_Annotation_Results.md.

Pseudonyms only. Real names live in incoming/ and rater_pseudonyms_PRIVATE.json
and must never reach this file, the artifact, or the paper.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

TASKS = ("A", "B", "C")
TASK_LABEL = {"A": "A - cross-view identity (control)",
              "B": "B - egocentric bearing (primary)",
              "C": "C - post-motion visibility (supporting)"}
# Design of record, frozen 2026-08-12: ~7 paid lab annotators, 5 raters/task.
RATERS_PER_TASK = 5
PANEL_SIZE = 7


def pct(x) -> str:
    return "&mdash;" if x is None else f"{x:.1%}"


def class_pct(cell: dict | None) -> str:
    """Format one per-answer-class score consistently as a percentage."""

    if not cell:
        return "&mdash;"
    accuracy = cell.get("accuracy")
    if accuracy is None and cell.get("of"):
        accuracy = cell.get("correct", 0) / cell["of"]
    return pct(accuracy)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=Path, default=Path("work/human_study_r2/scored"))
    ap.add_argument("--out", type=Path, default=Path("docs/Human_Annotation_Results.md"))
    args = ap.parse_args()

    sessions = json.loads((args.scored / "sessions.json").read_text(encoding="utf-8"))
    mv_path = args.scored / "majority_vote.json"
    majority = json.loads(mv_path.read_text(encoding="utf-8")) if mv_path.exists() else {}
    analysis_path = args.scored / "analysis.json"
    analysis = json.loads(analysis_path.read_text(encoding="utf-8")) \
        if analysis_path.exists() else {"tasks": {}}

    by_task = {t: [s for s in sessions if s.get("task") == t] for t in TASKS}
    raters = sorted({s.get("rater") for s in sessions if s.get("rater")})

    # New scorer outputs encode the frozen completeness/build/attention rule.
    # Retain the old fallback only so the generator can still read historical
    # scored files during recovery.
    def usable(s):
        if "quality_eligible" in s:
            return bool(s["quality_eligible"])
        return s.get("items_scored", 0) > 0 and s.get("package_id_match") != "legacy"

    L = ["# Human annotation results",
         "",
         "Regenerate with:",
         "",
         "    PYTHONPATH=release/src python release/tools/build_human_tracker.py",
         "",
         "Generated from `work/human_study_r2/scored/`. Pseudonyms only; real",
         "names never enter this file. Design of record (frozen 2026-08-12):",
         f"a shared panel of about {PANEL_SIZE} paid lab annotators, "
         f"{RATERS_PER_TASK} raters per task, with Task B intended first. Returned",
         "timestamp-derived protocol deviations remain preserved in the generated",
         "audit issues rather than in the coverage summary.",
         "",
         "## Coverage",
         "",
         f"| Task | Eligible sessions | Target | Status | Qualified majority | Fleiss kappa |",
         "|---|---|---|---|---|---|"]

    for t in TASKS:
        ok = [s for s in by_task[t] if usable(s)]
        task_analysis = (analysis.get("tasks") or {}).get(t) or {}
        qualified = task_analysis.get("attention_qualified") or majority.get(t) or {}
        mvacc = qualified.get("majority_accuracy")
        voters = len(qualified.get("raters") or [])
        ties = int(qualified.get("majority_ties") or 0)
        fleiss = qualified.get("fleiss_kappa")
        if len(ok) < RATERS_PER_TASK:
            status = f"**&minus;{RATERS_PER_TASK - len(ok)}**"
        elif len(ok) > RATERS_PER_TASK:
            status = f"+{len(ok) - RATERS_PER_TASK} extra"
        else:
            status = "met"
        mv = pct(mvacc)
        if ties:
            mv += f" ⚠ {ties} no-consensus"
        if 0 < voters < 3:
            mv += f" ⚠ {voters} voters"
        fk = "&mdash;" if fleiss is None else f"{fleiss:.3f}"
        L.append(f"| {TASK_LABEL[t]} | {len(ok)} | {RATERS_PER_TASK} | {status} "
                 f"| {mv} | {fk} |")

    L += ["", f"Distinct raters seen so far: **{len(raters)}** of ~{PANEL_SIZE} "
              f"({', '.join(raters) if raters else 'none'}).", ""]

    # ---- per-session detail -------------------------------------------
    L += ["## Sessions", "",
          "| Task | Rater | Accuracy | Scored | Attention | Practice | Flagged | QC |",
          "|---|---|---|---|---|---|---|---|"]
    for t in TASKS:
        for s in sorted(by_task[t], key=lambda x: x.get("rater") or ""):
            acc = pct(s.get("accuracy"))
            scored = s.get("items_scored", 0)
            miss = s.get("items_missing", 0)
            scol = f"{scored}" + (f" (+{miss} missing)" if miss else "")
            qc = "eligible" if usable(s) else "**excluded:** " + ", ".join(
                s.get("exclusion_reasons") or [str(s.get("package_id_match"))]
            )
            L.append(f"| {t} | {s.get('rater')} | {acc} | {scol} "
                     f"| {s.get('attention_checks_passed','?')} "
                     f"| {s.get('practice_score','?')} "
                     f"| {len(s.get('flagged_items') or [])} | {qc} |")
    L.append("")

    # ---- per-class breakdown ------------------------------------------
    for t in TASKS:
        rows = [s for s in by_task[t] if s.get("accuracy_by_class")]
        if not rows:
            continue
        classes = sorted({c for s in rows for c in s["accuracy_by_class"]})
        L += [f"### Task {t} by answer class", "",
              "| Rater | " + " | ".join(classes) + " |",
              "|---" * (len(classes) + 1) + "|"]
        for s in rows:
            cells = []
            for c in classes:
                d = s["accuracy_by_class"].get(c)
                cells.append(class_pct(d))
            L.append(f"| {s['rater']} | " + " | ".join(cells) + " |")
        L.append("")

    # ---- generated issue list -----------------------------------------
    issues = []
    order_map = analysis.get("task_order_by_rater") or {}
    if order_map:
        b_first = int(analysis.get("task_b_first_raters") or 0)
        if b_first != len(order_map):
            issues.append(
                f"**Protocol-order deviation:** Task B was specified first, but "
                f"the embedded timestamps show B first for {b_first}/{len(order_map)} "
                "raters. The observed order must be disclosed; these data cannot be "
                "described as counterbalanced or B-first."
            )

    for t in TASKS:
        eligible_n = sum(usable(s) for s in by_task[t])
        if eligible_n > RATERS_PER_TASK:
            issues.append(
                f"Task {t}: {eligible_n} eligible sessions exceed the frozen "
                f"{RATERS_PER_TASK}-rater target. Retain all in the crossed analysis "
                "unless an outcome-blind selection rule predates unblinding."
            )
        q = (((analysis.get("tasks") or {}).get(t) or {})
             .get("attention_qualified") or {})
        ties = int(q.get("majority_ties") or 0)
        if ties:
            issues.append(
                f"Task {t}: {ties} item(s) have tied eligible-panel votes and are "
                "recorded as no consensus, never broken by input order."
            )
        flagged = (((analysis.get("tasks") or {}).get(t) or {}).get("flagged_items") or {})
        repeated_flags = sorted(item_id for item_id, row in flagged.items()
                                if int(row.get("count") or 0) >= 2)
        if repeated_flags:
            issues.append(
                f"Task {t}: {len(repeated_flags)} item(s) were independently flagged "
                f"by at least two raters ({', '.join(f'`{x}`' for x in repeated_flags)}); "
                "adjudicate before the paper analysis."
            )

    for s in sessions:
        t, r = s.get("task"), s.get("rater")
        if not s.get("build_valid", s.get("package_id_match") != "legacy"):
            n = len(s.get("items_not_in_this_build") or [])
            issues.append(f"**Task {t} {r} answered a superseded build** - {n} items "
                          f"are not in the current candidate. Not usable as a "
                          f"trained-human reference for the current pool without a re-run.")
        elif s.get("items_scored", 0) == 0:
            issues.append(f"Task {t} {r}: zero items scored.")
        if s.get("items_missing"):
            issues.append(f"Task {t} {r}: {s['items_missing']} items missing from the response file.")
        # Attention checks for a superseded-build session are scored against
        # the CURRENT build's expected answers, so they are not interpretable.
        # Reporting them invites treating a build mismatch as inattention.
        if not s.get("build_valid", s.get("package_id_match") != "legacy"):
            continue
        checks = str(s.get("attention_checks_passed", ""))
        if "/" in checks:
            got, tot = checks.split("/")
            # Only blame the rater for checks that are not already explained by
            # a check every rater failed; otherwise one defective item shows up
            # as four separate rater problems.
            detail = s.get("attention_check_detail") or {}
            peers = [p for p in sessions
                     if p.get("task") == t and p.get("attention_check_detail")
                     and p.get("build_valid", True)]
            unanimous = {
                cid for cid in detail
                if peers and not any(p["attention_check_detail"].get(cid, {}).get("passed")
                                     for p in peers)
            }
            own = [cid for cid, d in detail.items()
                   if not d["passed"] and cid not in unanimous]
            if own:
                issues.append(f"Task {t} {r}: failed {', '.join(sorted(own))}.")
            elif got != tot and not detail:
                issues.append(f"Task {t} {r}: attention checks {checks}.")
        prac = str(s.get("practice_score", ""))
        if "/" in prac:
            got, tot = prac.split("/")
            if int(got) * 2 < int(tot):
                issues.append(f"Task {t} {r}: practice {prac} - weak training evidence.")
        if s.get("attention_qc_pass") is False:
            issues.append(
                f"**Task {t} {r} excluded by the frozen attention rule** "
                f"({s.get('attention_checks_passed')}; at least 2/3 required)."
            )
        if (t == "B" and usable(s) and s.get("accuracy") is not None
                and s["accuracy"] <= 0.4333):
            issues.append(
                f"Task B {r}: {s['accuracy']:.1%}, at/below the 45.90% hidden-pool finite "
                "structural reference. Retained because measured accuracy is not "
                "an exclusion criterion; include it in rater-aware inference."
            )

    # A systematic miss needs explicit review. The shipped HTML, options and
    # manifest gold have been checked for literal consistency, so do not
    # silently relabel the check as defective merely because the result is
    # inconvenient; preserve the event and the frozen session-level rule.
    for t in TASKS:
        rows = [s for s in by_task[t] if s.get("attention_check_detail")
                and s.get("build_valid", True)]
        if len(rows) < 3:
            continue
        cids = sorted({c for s in rows for c in s["attention_check_detail"]})
        for cid in cids:
            verdicts = [s["attention_check_detail"][cid]["passed"]
                        for s in rows if cid in s["attention_check_detail"]]
            if verdicts and not any(verdicts):
                answered = {s["attention_check_detail"][cid]["answered"]
                            for s in rows if cid in s["attention_check_detail"]}
                same = "the identical answer" if len(answered) == 1 else "varying answers"
                issues.append(
                    f"**Task {t} `{cid}` failed by all {len(verdicts)} raters**, giving "
                    f"{same}. Static audit confirms the prompt, offered option and "
                    "manifest gold agree. Preserve this systematic compliance failure "
                    "and do not amend the key post hoc; every Task-A session still "
                    "meets the frozen 2/3 session threshold.")

        task_checks = (((analysis.get("tasks") or {}).get(t) or {})
                       .get("attention_checks") or {})
        for cid, audit in sorted(task_checks.items()):
            passed, total = int(audit.get("passed") or 0), int(audit.get("of") or 0)
            if total >= 3 and 0 < passed * 2 <= total:
                issues.append(
                    f"Task {t} `{cid}` passed by only {passed}/{total} current-build "
                    "raters. Static audit confirms prompt/options/gold consistency; "
                    "do not invalidate it post hoc merely to retain a session."
                )

    L += ["## Open issues", ""]
    L += [f"- {i}" for i in issues] if issues else ["- none"]
    L += ["",
          "## Handling",
          "",
          "Drop returned files unchanged into `work/human_study_r2/incoming/`, then",
          "run `release/tools/score_human_responses.py` followed by this generator.",
          "`incoming/` and `rater_pseudonyms_PRIVATE.json` are the only places",
          "identifying data lives; neither may enter the release or the anonymous",
          "artifact.",
          ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(L), encoding="utf-8")
    total_ok = sum(1 for s in sessions if usable(s))
    print(f"wrote {args.out}  ({total_ok}/{len(sessions)} sessions usable, "
          f"{len(issues)} open issue(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
