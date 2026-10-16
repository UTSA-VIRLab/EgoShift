#!/usr/bin/env python3
"""Score annotator response files against the private package manifests.

Reads every response JSON in an incoming directory, matches it to its task
manifest, and reports per-rater accuracy, majority-vote accuracy, attention
checks, practice scores, and flagged items.

Rater names are replaced with stable pseudonyms (R01, R02, ...) derived from a
sorted list of the names seen, so the same person keeps the same label across
tasks and reruns. The mapping is written to a separate file that is not part of
the analysis output, and no output file here contains a real name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import math
import statistics
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

TASKS = ("A", "B", "C")
ATTENTION_MIN_PASS = 2  # frozen rule: one lapse forgiven out of three checks


def wilson_interval(correct: int, total: int, z: float = 1.96) -> list[float] | None:
    """Item-level Wilson interval, returned as proportions for descriptive use."""

    if total <= 0:
        return None
    p = correct / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


def cohen_kappa(left: dict[str, str], right: dict[str, str]) -> tuple[float | None, float | None, int]:
    ids = sorted(set(left) & set(right))
    if not ids:
        return None, None, 0
    observed = sum(left[i] == right[i] for i in ids) / len(ids)
    lc = Counter(left[i] for i in ids)
    rc = Counter(right[i] for i in ids)
    expected = sum((lc[c] / len(ids)) * (rc[c] / len(ids)) for c in set(lc) | set(rc))
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    return round(observed, 4), round(kappa, 4), len(ids)


def scene_index(task: str, candidates: dict[str, Path]) -> dict[str, str]:
    """Map item id -> physical scene id, so the bootstrap can cluster on scenes.

    Responses are not independent twice over: one rater's habits correlate
    across all of their items, and items rendered from one scene share a room,
    a layout and a lighting setup. Treating 960 responses as 960 independent
    draws understates the interval, which is why the reported interval resamples
    both raters and scenes rather than responses.
    """

    root = candidates.get(task)
    if root is None:
        return {}
    if task == "A":
        path = root / "private" / "task_a_hidden_test_gold.jsonl"
    else:
        path = root / "private" / "provenance" / f"task_{task.lower()}_hidden_test.jsonl"
    if not path.exists():
        return {}
    index = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        scene = row.get("scene_id") or (row.get("_candidate_source") or {}).get("scene_id")
        if scene is not None:
            index[row["id"]] = scene
    return index


def crossed_bootstrap(gold: dict[str, str], answers: dict[str, dict[str, str]],
                      raters: list[str], scenes: dict[str, str],
                      draws: int = 2000, seed: int = 20260827) -> dict | None:
    """Two-way cluster bootstrap over raters and scenes.

    Each draw resamples the rater panel with replacement and the scene pool with
    replacement, then scores every response a resampled rater gave on an item
    from a resampled scene. Returns the percentile interval of mean accuracy,
    plus the inputs needed to check it.
    """

    grid: dict[str, dict[str, list[bool]]] = {}
    for rater in raters:
        for item_id, choice in answers.get(rater, {}).items():
            if item_id not in gold:
                continue
            scene = scenes.get(item_id)
            if scene is None:
                continue
            grid.setdefault(rater, {}).setdefault(scene, []).append(choice == gold[item_id])
    panel = sorted(grid)
    scene_pool = sorted({s for by_scene in grid.values() for s in by_scene})
    if len(panel) < 2 or len(scene_pool) < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(draws):
        drawn_raters = [rng.choice(panel) for _ in panel]
        drawn_scenes = [rng.choice(scene_pool) for _ in scene_pool]
        hits = total = 0
        for rater in drawn_raters:
            by_scene = grid[rater]
            for scene in drawn_scenes:
                cell = by_scene.get(scene)
                if cell:
                    hits += sum(cell)
                    total += len(cell)
        if total:
            means.append(hits / total)
    if not means:
        return None
    means.sort()
    return {
        "method": "two-way cluster bootstrap, raters x scenes, percentile interval",
        "draws": draws,
        "seed": seed,
        "raters": len(panel),
        "scenes": len(scene_pool),
        "point": round(sum(means) / len(means), 4),
        "ci95": [round(means[int(0.025 * len(means))], 4),
                 round(means[int(0.975 * len(means))], 4)],
    }


def panel_summary(gold: dict[str, str], answers: dict[str, dict[str, str]],
                  raters: list[str]) -> dict:
    """Tie-safe majority and agreement summary for a named rater set."""

    correct = covered = ties = 0
    per_item = {}
    fleiss_rows = []
    for item_id, answer in gold.items():
        cast = [answers[r][item_id] for r in raters if item_id in answers.get(r, {})]
        if not cast:
            continue
        covered += 1
        counts = Counter(cast)
        top_count = max(counts.values())
        winners = sorted(choice for choice, n in counts.items() if n == top_count)
        tied = len(winners) != 1
        majority = None if tied else winners[0]
        hit = majority == answer
        correct += int(hit)
        ties += int(tied)
        per_item[item_id] = {
            "votes": len(cast),
            "vote_counts": dict(sorted(counts.items())),
            "majority": majority,
            "gold": answer,
            "correct": hit,
            "tie": tied,
            "unanimous": len(counts) == 1,
        }
        if len(cast) == len(raters) and len(raters) >= 2:
            fleiss_rows.append(counts)

    fleiss = None
    if fleiss_rows and len(raters) >= 2:
        m = len(raters)
        categories = sorted({c for row in fleiss_rows for c in row})
        p_bar = sum(
            (sum(row[c] ** 2 for c in categories) - m) / (m * (m - 1))
            for row in fleiss_rows
        ) / len(fleiss_rows)
        p_j = [
            sum(row[c] for row in fleiss_rows) / (len(fleiss_rows) * m)
            for c in categories
        ]
        p_e = sum(p * p for p in p_j)
        fleiss = round((p_bar - p_e) / (1 - p_e), 4) if p_e < 1 else 1.0

    pairs = []
    for left, right in combinations(raters, 2):
        agreement, kappa, n = cohen_kappa(answers.get(left, {}), answers.get(right, {}))
        pairs.append({"left": left, "right": right, "items": n,
                      "agreement": agreement, "cohen_kappa": kappa})

    decided = covered - ties
    return {
        "raters": raters,
        "items_with_votes": covered,
        "items_total": len(gold),
        # A tie is not a correct majority. The decided-only figure is supplied
        # separately rather than resolving ties by input/file order.
        "majority_accuracy": round(correct / covered, 4) if covered else None,
        "decided_majority_accuracy": round(correct / decided, 4) if decided else None,
        "majority_ties": ties,
        "fleiss_kappa": fleiss,
        "pairwise": pairs,
        "per_item": per_item,
    }


def load_manifests(audits: Path) -> dict[str, dict]:
    manifests = {}
    for task in TASKS:
        path = audits / f"task{task.lower()}_annotator_package.json"
        if path.is_file():
            manifest = json.loads(path.read_text(encoding="utf-8"))
            candidate_root = Path(str(manifest["candidate_root"]))
            options_by_item = {}
            for role in ("hidden_test", "public_dev", "tuning_dev"):
                questions = (
                    candidate_root / "public" / "questions" / f"task_{task.lower()}_{role}.jsonl"
                )
                if not questions.is_file():
                    continue
                for line in questions.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    options_by_item[str(row["id"])] = list(row.get("options") or [])
            missing_options = sorted(set(manifest["sampled_ids"]) - set(options_by_item))
            if missing_options:
                raise SystemExit(
                    f"{path}: sampled ids absent from candidate questions: {missing_options[:10]}"
                )
            manifest["_options_by_item"] = options_by_item
            manifests[task] = manifest
    return manifests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--incoming", type=Path,
                        default=Path("work/human_study_r2/incoming"))
    parser.add_argument("--audits", type=Path,
                        default=Path("work/candidate_r2/audits"))
    parser.add_argument("--out-dir", type=Path,
                        default=Path("work/human_study_r2/scored"))
    parser.add_argument("--candidate-a", type=Path,
                        default=Path("work/candidate_r2/taska_candidate_v5"))
    parser.add_argument("--candidate-b", type=Path,
                        default=Path("work/candidate_r2/taskb_candidate_r2_static_v3"))
    parser.add_argument("--candidate-c", type=Path,
                        default=Path("work/candidate_r2/taskc_candidate_v10"))
    args = parser.parse_args()
    candidate_roots = {"A": args.candidate_a, "B": args.candidate_b, "C": args.candidate_c}

    manifests = load_manifests(args.audits)
    if not manifests:
        raise SystemExit(f"no package manifests found in {args.audits}")

    files = sorted(args.incoming.glob("*.json"))
    if not files:
        print(f"no response files yet in {args.incoming}")
        return 0

    # Persist assignments once minted. Re-sorting all real names on every run
    # can silently relabel prior participants when a newly added surname sorts
    # earlier than theirs.
    payloads = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        payloads.append((path, data))
    names = sorted({str(d.get("annotator_name") or d.get("annotator") or "?")
                    for _, d in payloads})
    private_map_path = args.out_dir.parent / "rater_pseudonyms_PRIVATE.json"
    existing = {}
    if private_map_path.exists():
        try:
            existing = json.loads(private_map_path.read_text(encoding="utf-8")).get(
                "pseudonyms", {}
            )
        except (OSError, json.JSONDecodeError):
            existing = {}
    pseudonym = {name: existing[name] for name in names if name in existing}
    used_numbers = {
        int(label[1:]) for label in pseudonym.values()
        if isinstance(label, str) and label.startswith("R") and label[1:].isdigit()
    }
    next_number = 1
    for name in names:
        if name in pseudonym:
            continue
        while next_number in used_numbers:
            next_number += 1
        pseudonym[name] = f"R{next_number:02d}"
        used_numbers.add(next_number)

    sessions = []
    answers_by_task: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    finished_by_rater: dict[str, dict[str, str]] = defaultdict(dict)
    seen_session_keys: set[tuple[str, str]] = set()
    for path, data in payloads:
        task = str(data.get("task") or "?").upper()
        raw_name = str(data.get("annotator_name") or data.get("annotator") or "?")
        rater = pseudonym[raw_name]
        manifest = manifests.get(task)
        if manifest is None:
            sessions.append({"task": task, "rater": rater,
                             "error": "no manifest for this task"})
            continue
        session_key = (rater, task)
        if session_key in seen_session_keys:
            sessions.append({"task": task, "rater": rater,
                             "error": "duplicate participant/task session"})
            continue
        seen_session_keys.add(session_key)

        gold = manifest["gold"]
        gold_class = manifest.get("gold_class") or {}
        checks = {c["id"]: c["expected"] for c in manifest["attention_checks"]}
        responses = data.get("responses") or {}
        schema_errors = []
        if not isinstance(responses, dict):
            schema_errors.append("responses is not an object")
            responses = {}
        for item_id, entry in responses.items():
            if not isinstance(entry, dict):
                schema_errors.append(f"response {item_id} is not an object")
                continue
            if not isinstance(entry.get("choice"), str) or not entry.get("choice"):
                schema_errors.append(f"response {item_id} has no choice")
            if not isinstance(entry.get("t"), (int, float)):
                schema_errors.append(f"response {item_id} has no numeric timestamp")

        # package_sha_note only fingerprints (seed, task, item count), so it
        # cannot detect a rater who answered a different build with the same
        # size. Compare the item ids themselves, and check that every choice is
        # one of the options this build actually renders.
        answered = set(responses) - set(checks)
        unknown_items = sorted(answered - set(gold))
        missing_check_ids = sorted(set(checks) - set(responses))
        unexpected_response_ids = sorted(set(responses) - set(gold) - set(checks))
        option_errors = sorted(
            (
                {
                    "item_id": item,
                    "choice": entry.get("choice"),
                    "offered_options": manifest["_options_by_item"].get(item, []),
                }
                for item, entry in responses.items()
                if isinstance(entry, dict)
                if item in gold and entry.get("choice")
                and entry["choice"] not in manifest["_options_by_item"].get(item, [])
            ),
            key=lambda record: record["item_id"],
        )
        stale_choices = sorted({record["choice"] for record in option_errors})

        package_match = (
            "content"
            if data.get("package_sha_note") == manifest.get("package_id")
            else "legacy"
            if data.get("package_sha_note") == manifest.get("legacy_package_id")
            else "MISMATCH"
        )

        scored = wrong = missing = 0
        flagged = []
        session_answers = {}
        by_class: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for item_id, answer in gold.items():
            entry = responses.get(item_id)
            if not isinstance(entry, dict) or not entry.get("choice"):
                missing += 1
                continue
            scored += 1
            session_answers[item_id] = entry["choice"]
            cls = gold_class.get(item_id, "?")
            by_class[cls][1] += 1
            if entry["choice"] != answer:
                wrong += 1
            else:
                by_class[cls][0] += 1
            if entry.get("flagged"):
                flagged.append(item_id)

        # Record each check individually, not just the tally. A bare "2/3"
        # cannot distinguish one careless rater from a check item that every
        # rater fails, and the latter is a defect in the instrument rather
        # than in the panel -- which is exactly what the r2 Task-A build hit.
        check_detail = {
            cid: {"expected": expected,
                  "answered": (responses.get(cid) or {}).get("choice"),
                  "passed": (responses.get(cid) or {}).get("choice") == expected}
            for cid, expected in checks.items()
        }
        check_pass = sum(1 for d in check_detail.values() if d["passed"])
        attention_qc_pass = check_pass >= min(ATTENTION_MIN_PASS, len(checks))

        # Recompute practice against the private manifest. The browser's own
        # score is retained only as a consistency check, never trusted as gold.
        practice = data.get("practice") or {}
        practice_responses = practice.get("responses") or {}
        expected_practice = {
            f"practice_{entry['id']}": entry["answer"]
            for entry in manifest.get("practice_detail") or []
        }
        practice_correct = sum(
            (practice_responses.get(item_id) or {}).get("choice") == answer
            for item_id, answer in expected_practice.items()
        )
        practice_missing = sorted(set(expected_practice) - set(practice_responses))
        practice_reported = [practice.get("score"), practice.get("of")]
        practice_recomputed = [practice_correct, len(expected_practice)]

        timing_values = sorted(
            float(entry["t"])
            for item_id, entry in responses.items()
            if item_id in set(gold) | set(checks)
            and isinstance(entry, dict) and isinstance(entry.get("t"), (int, float))
        )
        timing = {
            "response_span_minutes": (
                round((timing_values[-1] - timing_values[0]) / 60000, 2)
                if len(timing_values) >= 2 else None
            ),
            "median_inter_response_seconds": (
                round(statistics.median(
                    (right - left) / 1000
                    for left, right in zip(timing_values, timing_values[1:])
                ), 2) if len(timing_values) >= 2 else None
            ),
        }

        complete = scored == len(gold) and not missing_check_ids
        # A legacy package id alone is not enough to reject a response: older
        # browsers used a weaker id. Its actual item ids/choices must still be
        # an exact match to this build. The known superseded Task-C session
        # fails that content check and remains excluded.
        build_valid = (
            package_match in {"content", "legacy"}
            and not unknown_items and not unexpected_response_ids
            and not stale_choices and scored == len(gold)
        )
        exclusion_reasons = []
        if schema_errors:
            exclusion_reasons.append("schema_error")
        if not build_valid:
            exclusion_reasons.append("build_mismatch")
        if not complete:
            exclusion_reasons.append("incomplete")
        if not attention_qc_pass:
            exclusion_reasons.append("attention_below_2_of_3")
        quality_eligible = not exclusion_reasons

        session = {
            "task": task,
            "rater": rater,
            "package_id_reported": data.get("package_sha_note"),
            "package_id_expected": manifest.get("package_id"),
            # Responses collected before the id started hashing the item and
            # practice ids carry the old (seed, task, count) fingerprint. Those
            # are still valid -- accept them, but say which form matched so a
            # genuine build mismatch is not hidden among them.
            "package_id_match": package_match,
            "build_valid": build_valid,
            "complete": complete,
            "quality_eligible": quality_eligible,
            "exclusion_reasons": exclusion_reasons,
            "schema_errors": schema_errors,
            "items_not_in_this_build": unknown_items,
            "unexpected_response_ids": unexpected_response_ids,
            "missing_attention_check_ids": missing_check_ids,
            "choices_not_offered_by_this_build": stale_choices,
            "per_item_option_validation_errors": option_errors,
            "items_scored": scored,
            "items_missing": missing,
            "items_correct": scored - wrong,
            "accuracy": round((scored - wrong) / scored, 4) if scored else None,
            "accuracy_wilson95_item": wilson_interval(scored - wrong, scored),
            "attention_checks_passed": f"{check_pass}/{len(checks)}",
            "attention_qc_pass": attention_qc_pass,
            "attention_check_detail": check_detail,
            "practice_score": f"{practice_correct}/{len(expected_practice)}",
            "practice_score_reported": practice_reported,
            "practice_score_matches_reported": practice_recomputed == practice_reported,
            "practice_missing_ids": practice_missing,
            "flagged_items": flagged,
            "timing": timing,
            "accuracy_by_class": {
                name: {"correct": hit, "of": total,
                       "accuracy": round(hit / total, 4) if total else None}
                for name, (hit, total) in sorted(by_class.items())
            },
        }
        sessions.append(session)

        if build_valid:
            answers_by_task[task][rater] = session_answers
        if isinstance(data.get("finished"), str):
            finished_by_rater[rater][task] = data["finished"]

    majority = {}
    analysis = {"schema_version": 1, "tasks": {}, "task_order_by_rater": {}}
    for task in TASKS:
        gold = manifests[task]["gold"]
        all_raters = sorted(answers_by_task.get(task, {}))
        eligible_raters = sorted(
            s["rater"] for s in sessions
            if s.get("task") == task and s.get("quality_eligible")
        )
        all_summary = panel_summary(gold, answers_by_task.get(task, {}), all_raters)
        eligible_summary = panel_summary(
            gold, answers_by_task.get(task, {}), eligible_raters
        )
        chronological_target_raters = sorted(
            eligible_raters,
            key=lambda r: (finished_by_rater.get(r, {}).get(task, ""), r),
        )[:5]
        chronological_target_summary = panel_summary(
            gold, answers_by_task.get(task, {}), chronological_target_raters
        ) if len(eligible_raters) > 5 else None
        majority[task] = all_summary

        eligible_sessions = [
            s for s in sessions
            if s.get("task") == task and s.get("quality_eligible")
        ]
        correct_total = sum(s["items_correct"] for s in eligible_sessions)
        scored_total = sum(s["items_scored"] for s in eligible_sessions)
        accuracies = [s["accuracy"] for s in eligible_sessions if s.get("accuracy") is not None]
        check_ids = sorted({
            cid for s in sessions if s.get("task") == task
            for cid in (s.get("attention_check_detail") or {})
            if s.get("build_valid")
        })
        check_audit = {}
        current_sessions = [s for s in sessions if s.get("task") == task and s.get("build_valid")]
        flagged_by_item: dict[str, list[str]] = defaultdict(list)
        for session in current_sessions:
            for item_id in session.get("flagged_items") or []:
                flagged_by_item[item_id].append(session["rater"])
        for cid in check_ids:
            verdicts = [
                s["attention_check_detail"][cid]["passed"]
                for s in current_sessions if cid in s.get("attention_check_detail", {})
            ]
            check_audit[cid] = {
                "passed": sum(verdicts), "of": len(verdicts),
                "pass_rate": round(sum(verdicts) / len(verdicts), 4) if verdicts else None,
            }

        analysis["tasks"][task] = {
            "all_current": all_summary,
            "attention_qualified": eligible_summary,
            # Descriptive sensitivity only. The confirmatory analysis retains
            # every eligible session unless an outcome-blind five-rater rule
            # can be shown to predate unblinding.
            "chronological_first_five_sensitivity": chronological_target_summary,
            "eligible_raters": eligible_raters,
            "excluded_sessions": [
                {"rater": s.get("rater"), "reasons": s.get("exclusion_reasons")}
                for s in sessions
                if s.get("task") == task and not s.get("quality_eligible")
            ],
            "mean_rater_accuracy": round(sum(accuracies) / len(accuracies), 4)
                                   if accuracies else None,
            "rater_accuracy_range": [min(accuracies), max(accuracies)] if accuracies else None,
            "pooled_response_accuracy": round(correct_total / scored_total, 4)
                                        if scored_total else None,
            "pooled_wilson95_item": wilson_interval(correct_total, scored_total),
            # Wilson treats every response as an independent draw. It is kept
            # for continuity, but the interval the paper reports is the crossed
            # one below, which resamples raters and scenes.
            "crossed_rater_scene_bootstrap": crossed_bootstrap(
                gold, answers_by_task.get(task, {}), eligible_raters,
                scene_index(task, candidate_roots)),
            "attention_checks": check_audit,
            "flagged_responses": sum(len(s.get("flagged_items") or [])
                                     for s in current_sessions),
            "flagged_items": {
                item_id: {"count": len(flag_raters), "raters": sorted(flag_raters)}
                for item_id, flag_raters in sorted(flagged_by_item.items())
            },
        }

    for rater, task_times in sorted(finished_by_rater.items()):
        analysis["task_order_by_rater"][rater] = [
            task for task, _ in sorted(task_times.items(), key=lambda item: item[1])
        ]
    analysis["task_b_first_raters"] = sum(
        bool(order) and order[0] == "B"
        for order in analysis["task_order_by_rater"].values()
    )
    analysis["task_order_raters"] = len(analysis["task_order_by_rater"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "sessions.json").write_text(
        json.dumps(sessions, indent=1, sort_keys=True) + "\n")
    (args.out_dir / "majority_vote.json").write_text(
        json.dumps(majority, indent=1, sort_keys=True) + "\n")
    (args.out_dir / "analysis.json").write_text(
        json.dumps(analysis, indent=1, sort_keys=True) + "\n")
    # The only file that links a real person to a label. Response filenames
    # embed the rater's name, so they live here too and never in scored/.
    private_map_path.write_text(
        json.dumps(
            {
                "pseudonyms": pseudonym,
                "source_files": {
                    path.name: pseudonym[
                        str(data.get("annotator_name") or data.get("annotator") or "?")
                    ]
                    for path, data in payloads
                },
                "note": (
                    "Identifying data. Never copy into scored/, the release, "
                    "the artifact, or the paper."
                ),
            },
            indent=1, sort_keys=True) + "\n")

    # Freeze the exact raw bytes used for every scoring rebuild. Keeping this
    # in the private audit directory makes later replacements or accidental
    # edits detectable without leaking identifying filenames into scored/ or
    # the public tracker.
    intake_ledger = args.out_dir.parent / "private_audit/raw_intake_hashes_PRIVATE.sha256"
    intake_ledger.parent.mkdir(parents=True, exist_ok=True)
    intake_ledger.write_text(
        "".join(
            f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n"
            for path in files
        ),
        encoding="utf-8",
    )

    print(f"scored {len(files)} response file(s)\n")
    for s in sorted(sessions, key=lambda x: (x.get("task", ""), x.get("rater", ""))):
        if s.get("error"):
            print(f"  [task {s.get('task')}]: {s['error']}")
            continue
        print(f"  Task {s['task']} {s['rater']}: acc={s['accuracy']} "
              f"({s['items_scored']} scored, {s['items_missing']} missing) "
              f"checks={s['attention_checks_passed']} practice={s['practice_score']} "
              f"flagged={len(s['flagged_items'])} eligible={s['quality_eligible']}")
    print()
    for task in TASKS:
        m = analysis["tasks"][task]["attention_qualified"]
        print(f"  Task {task}: qualified majority accuracy {m['majority_accuracy']} "
              f"over {m['items_with_votes']}/{m['items_total']} items, "
              f"{len(m['raters'])} rater(s), {m['majority_ties']} tie(s), "
              f"Fleiss kappa={m['fleiss_kappa']}")
    print(f"\n  Task B first: {analysis['task_b_first_raters']}/"
          f"{analysis['task_order_raters']} rater(s)")
    print(f"\nwrote {args.out_dir}/sessions.json, majority_vote.json, and analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
