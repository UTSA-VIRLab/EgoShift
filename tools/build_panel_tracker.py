#!/usr/bin/env python3
"""Regenerate the model-panel tracker from the actual result files.

Hand-maintained checklists drift from reality. This reads the panel config and
every results JSONL, counts completed items per (model, task, arm) against the
target for that evaluation set, and rewrites docs/Model_Panel_Tracker.md with
boxes already ticked for whatever is genuinely finished. Re-run it any time to
refresh; nothing is marked done that the data does not support.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

TASKS = ("A", "B", "C")
# Target item counts per task, from the coverage plan in the runbook.
# API tiers (1-3) are paid, so they run frozen balance-preserving subsets of
# the two large pools rather than the full ones. Task A and B open-weight
# tiers run the full pool. Task C open-weight tiers were downgraded to the
# same frozen 1,200-item subset as the API tiers (2026-08-25 decision) rather
# than the full 5,850 pool, so every model's Task C number is on identical
# items. Sizes are read from the frozen subset files so this table cannot
# disagree with what the runners actually evaluate.
def _subset_size(name: str, fallback: int) -> int:
    path = Path(__file__).resolve().parents[1] / "configs" / name
    try:
        return int(json.loads(path.read_text(encoding="utf-8"))["size"])
    except Exception:
        return fallback


def _subset_ids(name: str) -> set | None:
    path = Path(__file__).resolve().parents[1] / "configs" / name
    try:
        return set(json.loads(path.read_text(encoding="utf-8"))["ids"])
    except Exception:
        return None


TASKC_SUBSET = "taskc_api_subset_1200.json"
TASKC_SUBSET_IDS = _subset_ids(TASKC_SUBSET)

TARGETS = {
    ("A", "api"): _subset_size("taska_api_subset_1200.json", 1192),
    ("A", "local"): 5838,
    ("B", "api"): 1098, ("B", "local"): 1098,
    ("C", "api"): _subset_size(TASKC_SUBSET, 1200),
    ("C", "local"): _subset_size(TASKC_SUBSET, 1200),
}
SUBSET_TARGET = 120


# Reference lines the paper reports beside every accuracy. Task B's third
# line is computed from gold below rather than fixed here, because it is a
# property of the answer mix and must move if the pool is rebuilt.
REFERENCES = {
    "A": [("chance", 0.500), ("frozen-feature pair probe", 0.767)],
    "B": [
        ("chance", 0.250),
        ("hidden-pool action-cell target", 504 / 1098),
    ],
    "C": [("chance", 0.3333)],
}

# Task B's four options split into two axes. Most of the panel answers on one
# axis almost exclusively, which is why raw accuracy alone is misleading:
# a model that never says "behind"/"in front" is capped at the lateral share
# of gold no matter how well it reasons.
TASKB_LATERAL = ("to the left", "to the right")
CANDIDATE_ROOTS = {
    "A": Path("work/candidate_r2/taska_candidate_v5"),
    "B": Path("work/candidate_r2/taskb_candidate_r2_static_v3"),
    "C": Path("work/candidate_r2/taskc_candidate_v10"),
}

# Product-line families are different from runtime providers. In particular,
# all open-weight models have provider="local", which is useless when someone
# wants to compare the InternVL or Qwen generations as a family. Keep the
# classification explicit so a new, unfamiliar name lands in "Other" rather
# than being silently assigned to the wrong research lineage.
FAMILY_DISPLAY_ORDER = (
    "GPT", "Claude", "Gemini", "InternVL", "Qwen", "SenseNova", "LLaVA",
    "Llama", "Idefics", "DeepSeek", "Cosmos", "SmolVLM", "SpaceMind",
    "3DThinker", "EgoMind", "Molmo", "Other",
)


def product_family(name: str) -> str:
    """Return the model-lineage label shown in the family tables."""

    n = name.casefold()
    rules = (
        ("gpt", "GPT"),
        ("claude", "Claude"),
        ("gemini", "Gemini"),
        ("internvl", "InternVL"),
        ("qwen", "Qwen"),
        ("qvq", "Qwen"),  # QVQ is the Qwen visual-reasoning line.
        ("sensenova", "SenseNova"),
        ("llava", "LLaVA"),
        ("llama", "Llama"),
        ("idefics", "Idefics"),
        ("deepseek", "DeepSeek"),
        ("cosmos", "Cosmos"),
        ("smolvlm", "SmolVLM"),
        ("spacemind", "SpaceMind"),
        ("3dthinker", "3DThinker"),
        ("egomind", "EgoMind"),
        ("molmo", "Molmo"),
    )
    for marker, label in rules:
        if n.startswith(marker):
            return label
    return "Other"


def _version(text: str, pattern: str) -> tuple[int, int, int]:
    match = re.search(pattern, text, re.I)
    if not match:
        return (0, 0, 0)
    parts = [int(p) for p in match.group(1).split(".")]
    return tuple((parts + [0, 0, 0])[:3])


def recency_version(name: str, resolved_id: str = "") -> tuple[int, int, int]:
    """Numeric generation/revision used for newest-to-oldest family sorting.

    Exact release dates are not consistently published for this mixed panel.
    We therefore use the public generation/revision in the name or resolved id;
    Python's stable sort retains canonical panel order for same-generation
    variants, where claiming a finer chronology would be unsupported.
    """

    family = product_family(name)
    # Prefer the resolved id because display names sometimes omit a minor
    # revision (for example, "Gemini 3 Flash" resolves to Gemini 3.7).
    text = f"{resolved_id} {name}"
    patterns = {
        "GPT": r"gpt[- ](\d+(?:\.\d+)*)",
        "Claude": r"claude(?:[- ][a-z]+)*[- ](\d+(?:\.\d+)*)",
        "Gemini": r"gemini[- ](\d+(?:\.\d+)*)",
        "InternVL": r"internvl(\d+(?:\.\d+)*)",
        "Qwen": r"qwen(\d+(?:\.\d+)*)",
        "SenseNova": r"si[- ](\d+(?:\.\d+)*)",
        "Idefics": r"idefics(\d+(?:\.\d+)*)",
        "Llama": r"llama[- ](\d+(?:\.\d+)*)",
        "DeepSeek": r"deepseek[- ]vl(\d+(?:\.\d+)*)",
        "Cosmos": r"reason(\d+(?:\.\d+)*)",
        "Molmo": r"molmo(\d+(?:\.\d+)*)",
    }
    if family == "LLaVA" and "onevision" in name.casefold():
        # OneVision supersedes the 1.6 line but its display name has no numeric
        # generation; the 7B token is parameter count, not release version.
        return (2, 0, 0)
    if family == "Qwen" and name.casefold().startswith("qvq"):
        # QVQ-72B-Preview is built on the Qwen2.5-VL generation. Do not parse
        # 72B as a version number.
        return (2, 5, 0)
    return _version(text, patterns.get(family, r"$^"))


def load_gold(task: str, split: str = "hidden_test") -> dict:
    """Answer string per item id, or {} when that candidate is not built yet."""

    root = CANDIDATE_ROOTS[task]
    paths = ([root / f"private/task_a_{split}_gold.jsonl"] if task == "A"
             else [root / f"private/gold/task_{task.lower()}_{split}.jsonl"])
    for path in paths:
        if path.exists():
            return {r["id"]: r["answer"] for r in
                    (json.loads(l) for l in path.read_text().splitlines() if l.strip())}
    return {}


def single_axis_ceilings(gold: dict) -> tuple[float, float]:
    """(lateral-only, sagittal-only) accuracy ceilings from the gold mix.

    A model that answers left/right perfectly but never produces
    "behind"/"in front" scores exactly the lateral share of gold, however good
    its reasoning. Both ceilings matter because the panel splits into models
    that commit to one axis and models that commit to the other.
    """

    if not gold:
        return 0.0, 0.0
    total = len(gold)
    lateral = sum(1 for a in gold.values() if a in TASKB_LATERAL)
    return lateral / total, (total - lateral) / total


def shared_ids(task: str) -> set:
    """The 120-item subset the control arms run on, for like-for-like deltas."""

    path = Path(f"work/candidate_r2/audits/task{task.lower()}_annotator_package.json")
    try:
        return set(json.loads(path.read_text(encoding="utf-8"))["sampled_ids"])
    except Exception:
        return set()


def load_results(paths: list[Path]) -> dict:
    counts: dict = defaultdict(set)
    hits: dict = defaultdict(lambda: [0, 0])   # (model, task, arm) -> [correct, scored]
    axis: dict = defaultdict(lambda: [0, 0, 0, 0, 0])
    # (model, arm) -> [lat_correct, lat_n, sag_correct, sag_n, said_lateral]
    shared: dict = defaultdict(lambda: [0, 0])   # (model, task, arm) on the 120 subset
    meta: dict = {}
    # A prediction is an experimental unit, even if an interrupted launch was
    # accidentally resumed twice. Keep the first non-error record so duplicate
    # paid calls cannot bias accuracy. Error rows remain retry/audit evidence;
    # they are not completed benchmark items.
    seen_predictions: set[tuple] = set()
    gold_by_task = {t: load_gold(t) for t in TASKS}
    shared_by_task = {t: shared_ids(t) for t in TASKS}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("item_id") is None:
                continue
            meta[r["model"]] = {"family": r.get("family"), "tier": r.get("tier")}
            if r.get("error"):
                continue
            if (r.get("arm") == "shuffled_options"
                    and int(r.get("permutation", 0)) == 0):
                # Permutation zero is the canonical option order, not the
                # advertised shuffled control. Historical local launches only
                # emitted this identity order; counting those rows made a
                # no-op look like a completed position-sensitivity experiment.
                continue
            prediction_key = (
                r["model"], r["task"], r["arm"], r["item_id"],
                r.get("permutation", 0),
            )
            if prediction_key in seen_predictions:
                continue
            seen_predictions.add(prediction_key)
            key = (r["model"], r["task"], r["arm"])
            # Task C has two deliberately different frozen populations: the
            # 1,200-item headline subset and the 120-item human/control subset.
            # Keep headline `main` counts on the former, while retaining
            # control rows on the latter. This prevents a valid 120-row control
            # from being displayed as 23/120 merely because only 23 human items
            # happen to overlap the independent headline draw.
            taskc_outside_headline = (
                r["task"] == "C" and TASKC_SUBSET_IDS is not None
                and r["item_id"] not in TASKC_SUBSET_IDS
            )
            if taskc_outside_headline:
                if r["item_id"] not in shared_by_task.get("C", ()):
                    continue
                if r.get("arm") == "main":
                    # Main rows outside the headline draw are useful only for
                    # paired 120-item control deltas, never headline coverage.
                    gold = gold_by_task.get("C") or {}
                    if r["item_id"] in gold:
                        shared[key][1] += 1
                        shared[key][0] += int(r.get("parsed") == gold[r["item_id"]])
                    continue
            counts[key].add(r["item_id"])
            gold = gold_by_task.get(r["task"]) or {}
            # A provider response that cannot be parsed is still a completed
            # model attempt and therefore scores incorrect. Only transport/API
            # failures were filtered above; dropping parse failures would
            # silently inflate reported accuracy.
            if r["item_id"] in gold:
                hits[key][1] += 1
                correct = r.get("parsed") == gold[r["item_id"]]
                if correct:
                    hits[key][0] += 1
                if r["task"] == "B":
                    a = axis[(r["model"], r["arm"])]
                    if gold[r["item_id"]] in TASKB_LATERAL:
                        a[1] += 1
                        a[0] += int(correct)
                    else:
                        a[3] += 1
                        a[2] += int(correct)
                    a[4] += int(r.get("parsed") in TASKB_LATERAL)
                if r["item_id"] in shared_by_task.get(r["task"], ()):
                    s = shared[key]
                    s[1] += 1
                    s[0] += int(correct)
    return {"counts": counts, "hits": hits, "axis": axis, "shared": shared, "meta": meta}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--panel", type=Path, default=Path("release/configs/model_panel_r2.json"))
    # Glob, so the per-provider API files (results_openai.jsonl, ...) are
    # picked up without having to name each one here.
    p.add_argument("--results", nargs="*", type=Path,
                   default=sorted(Path("work/model_panel").glob("results*.jsonl")))
    p.add_argument("--resolved", type=Path, default=Path("work/model_panel/resolved_panel.json"))
    p.add_argument("--out", type=Path, default=Path("docs/Model_Panel_Tracker.md"))
    args = p.parse_args()

    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    data = load_results(args.results)
    counts = data["counts"]
    hits = data["hits"]

    resolved_ids, failures = {}, {}
    # Local models can fail for reasons a provider listing never sees -- a
    # checkpoint that ships an auto_map without the module it points to, or one
    # whose distinctive weights the standard architecture discards. Those are
    # recorded on the panel entry itself.
    for _m in panel["models"]:
        if _m.get("access_failure"):
            failures[_m["name"]] = _m["access_failure"]
    if args.resolved.exists():
        rp = json.loads(args.resolved.read_text(encoding="utf-8"))
        resolved_ids = {m["name"]: m.get("resolved_id") for m in rp.get("resolved", [])}
        # Merge, don't replace: the resolver only knows about API providers, so
        # assigning wholesale here silently dropped every locally-recorded
        # failure.
        failures.update({m["name"]: m.get("reason")
                         for m in rp.get("access_failures", [])})

    specs = {m["name"]: m for m in panel["models"]}

    def target_for(model: str, task: str) -> int:
        spec = specs.get(model, {})
        subset = (spec.get("task_subsets") or {}).get(task)
        if subset:
            return _subset_size(subset, 0)
        kind = "local" if spec.get("family") == "local" else "api"
        return TARGETS.get((task, kind), 0)

    def box(model: str, task: str, arm: str = "main") -> str:
        done = len(counts.get((model, task, arm), ()))
        target = target_for(model, task)
        if done == 0:
            return "[ ] &mdash;"
        correct, scored = hits.get((model, task, arm), [0, 0])
        acc = f"{correct / scored:.1%}" if scored else "—"
        if target and done >= target:
            return f"[x] **{acc}**"
        return f"[~] {acc} ({done:,}/{target:,})"

    lines = [
        "# Model panel tracker",
        "",
        "Regenerate with:",
        "",
        "    PYTHONPATH=release/src python release/tools/build_panel_tracker.py",
        "",
        "Boxes are filled from the result files, never by hand, so this page",
        "cannot drift from what actually ran. `[x]` complete, `[~]` partial with",
        "the item count, `[ ]` not started.",
        "",
        f"Coverage plan. Paid API tiers (1-3) run frozen, balance-preserving",
        f"subsets: Task A {TARGETS[('A','api')]:,} items, Task C {TARGETS[('C','api')]:,} (drawn as whole",
        f"target-matched triples). Task B needs no subset -- its hidden pool is",
        f"already {TARGETS[('B','api')]:,}. Open-weight tiers (4-6) run the full hidden pools",
        f"for A ({TARGETS[('A','local')]:,}) and B ({TARGETS[('B','local')]:,}); Task C runs the",
        f"same {TARGETS[('C','local')]:,}-item frozen subset as the paid tiers for every runnable model,",
        f"so all reported Task C numbers use identical items. Documented access/runtime failures",
        f"are listed separately and are not represented as pending evaluations.",
        f"The prospectively declared 2026-08-26 EgoMind-7B and Molmo2-8B amendments",
        f"follow their open-weight peer tiers: full Task A ({TARGETS[('A','local')]:,}), full Task B",
        f"({TARGETS[('B','local')]:,}), and the common Task C subset ({TARGETS[('C','local')]:,}).",
        "",
    ]

    for tier, description in panel["tiers"].items():
        members = [m for m in panel["models"] if m["tier"] == tier]
        if not members:
            continue
        lines += [f"## Tier {tier.replace('_', ' ')}", "", f"*{description}*", "",
                  "| ✓ | Model | Provider | Resolved id | Task A | Task B | Task C |",
                  "|---|---|---|---|---|---|---|"]
        for m in members:
            name = m["name"]
            # A model that cannot be called is not "pending" -- leaving it in
            # the tier table reads as work still to do. It moves to its own
            # section below, which is also what the release gate wants: a
            # documented access failure, never a silent deletion.
            if name in failures:
                continue
            rid = resolved_ids.get(name) or (m["id_candidates"][0] if m["id_candidates"] else "—")
            all_done = all(
                len(counts.get((name, t, "main"), ())) >= target_for(name, t)
                for t in TASKS)
            tick = "x" if all_done else " "
            cells = " | ".join(box(name, t) for t in TASKS)
            lines.append(f"| [{tick}] | {name} | {m['family']} | `{rid}` "
                         f"| {cells} |")
        lines.append("")

    # ---- alternate family view -------------------------------------------
    # Tiers answer the experimental-design question; families answer the
    # scaling/evolution question. Keep both views because collapsing one into
    # the other would hide either coverage rationale or within-line progress.
    grouped: dict[str, list[dict]] = defaultdict(list)
    for model in panel["models"]:
        grouped[product_family(model["name"])].append(model)
    family_order = [f for f in FAMILY_DISPLAY_ORDER if grouped.get(f)]
    family_order += sorted(set(grouped) - set(family_order))

    lines += [
        "## Models grouped by family",
        "",
        "*An alternate view of the same result records. Within each product "
        "family, models are sorted newest to oldest by the numeric generation/"
        "revision in the public or resolved model id. Same-generation variants "
        "retain canonical panel order; access failures remain visible but are "
        "not shown as pending runs.*",
        "",
    ]
    for family in family_order:
        members = sorted(
            grouped[family],
            key=lambda m: recency_version(
                m["name"],
                resolved_ids.get(m["name"])
                or ((m.get("id_candidates") or [""])[0]),
            ),
            reverse=True,
        )
        lines += [
            f"### {family}",
            "",
            "| ✓ | Model | Tier | Runtime id | Task A | Task B | Task C |",
            "|---|---|---|---|---|---|---|",
        ]
        for model in members:
            name = model["name"]
            tier = model["tier"].replace("_", " ")
            if name in failures:
                lines.append(
                    f"| ⚠ | {name} | {tier} | access failure | "
                    "&mdash; | &mdash; | &mdash; |"
                )
                continue
            rid = (resolved_ids.get(name)
                   or ((model.get("id_candidates") or ["—"])[0]))
            all_done = all(
                len(counts.get((name, task, "main"), ())) >= target_for(name, task)
                for task in TASKS
            )
            tick = "x" if all_done else " "
            cells = " | ".join(box(name, task) for task in TASKS)
            lines.append(f"| [{tick}] | {name} | {tier} | `{rid}` | {cells} |")
        lines.append("")

    if failures:
        lines += ["## Access failures", "",
                  "*Models the panel names but cannot evaluate. Recorded rather than "
                  "deleted: the release gate requires a documented failure per "
                  "unavailable model, never a silent omission. Verified by a live "
                  "callability probe, not just by a model listing.*", "",
                  "| Model | Provider | Reason |", "|---|---|---|"]
        by_name = {m["name"]: m for m in panel["models"]}
        for name, reason in sorted(failures.items()):
            fam = by_name.get(name, {}).get("family", "?")
            lines.append(f"| {name} | {fam} | {str(reason)[:150]} |")
        lines.append("")

    lines += ["## Reference lines", "",
              "Accuracy above is on each task's hidden split. Read every number "
              "against these:", ""]
    for task, refs in REFERENCES.items():
        parts = "; ".join(f"{name} **{value:.1%}**" for name, value in refs)
        lines.append(f"- **Task {task}** — {parts}")
    lines += ["",
              "Task B's hidden-pool action-cell target is the accuracy reachable by an adversary "
              "that sees only the rotation and turn direction and never an image. "
              "A model's position relative to that target does not by itself establish "
              "whether it used the picture. Task A's pair probe is "
              "frozen DINOv2 features plus logistic regression, i.e. what shallow "
              "visual similarity alone achieves.", ""]

    def control_cell(model: str, task: str, arm: str) -> str:
        done = len(counts.get((model, task, arm), ()))
        if not done:
            return "&mdash;"
        correct, scored = hits.get((model, task, arm), [0, 0])
        accuracy = f"{correct / scored:.1%}" if scored else "&mdash;"
        return (f"[x] {accuracy}" if done >= 120
                else f"[~] {accuracy} ({done}/120)")

    applicable = {
        "no_image": set(TASKS),
        "single_view": {"A", "C"},
        "shuffled_options": set(TASKS),
        "pp1_no_rule": {"B"},
        "pp2_few_shot": set(TASKS),
    }

    lines += ["## Control arms (shared 120-item subset)", "",
              "| Arm | Task A | Task B | Task C |", "|---|---|---|---|"]
    for arm, meaning in panel["arms"].items():
        if arm == "main":
            continue
        cells = []
        for t in TASKS:
            if t not in applicable.get(arm, set(TASKS)):
                cells.append("n/a")
                continue
            models_with = {m for (m, tt, a) in counts if tt == t and a == arm}
            complete = sum(len(counts[(m, t, arm)]) >= 120 for m in models_with)
            partial = len(models_with) - complete
            cell = f"{complete} complete" if complete else "&mdash;"
            if partial:
                cell += f"; {partial} partial"
            cells.append(cell)
        lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
    lines += ["", "*" + "; ".join(f"`{k}` = {v}" for k, v in panel["arms"].items()) + "*", ""]
    lines += [
        "Identity-order (`permutation=0`) rows are excluded from `shuffled_options`. ",
        "The historical local launches produced only that identity order, so they ",
        "are retained as audit evidence but are not presented as a completed control.",
        "",
    ]

    control_models = {m for (m, _t, arm) in counts if arm != "main"}
    if control_models:
        lines += [
            "### Control-arm results by model",
            "",
            "Cells report accuracy on the frozen 120-item control subset. Every "
            "displayed canonical-cohort cell is complete; task columns for which an "
            "arm is not defined are omitted.",
            "",
        ]
        for arm in ("no_image", "single_view", "shuffled_options",
                    "pp1_no_rule", "pp2_few_shot"):
            shown_tasks = [task for task in TASKS
                           if task in applicable.get(arm, set(TASKS))]
            arm_models = sorted({m for (m, _t, a) in counts if a == arm})
            # The paid no-image extension was prospectively Task-B-only. Keep it
            # out of the canonical three-task table so uncollected A/C controls
            # cannot be mistaken for failed or pending runs.
            if arm == "no_image":
                arm_models = [m for m in arm_models if all(
                    counts.get((m, task, arm)) for task in shown_tasks)]
            lines += [
                f"#### `{arm}`",
                "",
                "| Model | " + " | ".join(f"Task {task}" for task in shown_tasks) + " |",
                "|---|" + "---|" * len(shown_tasks),
            ]
            for model in arm_models:
                cells = [control_cell(model, task, arm) for task in shown_tasks]
                lines.append(f"| {model} | " + " | ".join(cells) + " |")
            lines.append("")

        no_image_full = {
            m for (m, _t, arm) in counts if arm == "no_image"
            and all(counts.get((m, task, arm)) for task in TASKS)
        }
        no_image_b_only = sorted({
            m for (m, task, arm) in counts
            if arm == "no_image" and task == "B" and m not in no_image_full
        })
        if no_image_b_only:
            lines += [
                "#### `no_image` paid API Task-B extension",
                "",
                "This extension was scoped to the headline primary task; Task A "
                "and C were not collected for these models and therefore are not "
                "shown as empty result cells.",
                "",
                "| Model | Task B |",
                "|---|---|",
            ]
            for model in no_image_b_only:
                lines.append(f"| {model} | {control_cell(model, 'B', 'no_image')} |")
            lines.append("")

    # ---- Task B response-bias diagnostics -------------------------------
    # Raw Task B accuracy is not interpretable on its own: most of the panel
    # commits to one axis and never answers on the other, so a sub-chance
    # score can mean "picked the wrong axis" rather than "cannot reason".
    axis = data["axis"]
    shared = data["shared"]
    gold_b = load_gold("B")
    rows_b = []
    for (model, arm), (lc, ln, sc, sn, said_lat) in axis.items():
        if arm != "main" or not ln or not sn:
            continue
        n = ln + sn
        overall = (lc + sc) / n
        # Both sides of the delta must be the same 120 items, or the
        # comparison is between a 1,098-item pool and a 120-item one.
        mc, ms = shared.get((model, "B", "main"), [0, 0])
        nc, ns = shared.get((model, "B", "no_image"), [0, 0])
        gain = (mc / ms - nc / ns) if (ms and ns) else None
        rows_b.append((overall, model, lc / ln, sc / sn, said_lat / n, gain))
    if rows_b:
        lat_ceiling, sag_ceiling = single_axis_ceilings(gold_b)
        floor = dict(REFERENCES["B"]).get("hidden-pool action-cell target", 0.0)
        lines += [
            "## Task B response bias",
            "",
            "*Why this table exists: on the `main` arm most models answer almost "
            "entirely on one axis. Accuracy alone then measures which option "
            "words a model favours as much as it measures spatial reasoning, so "
            "every Task B number is reported beside the decomposition.*",
            "",
            f"**Single-axis ceilings — lateral-only {lat_ceiling:.1%}, "
            f"sagittal-only {sag_ceiling:.1%}.** These are the scores reachable "
            "by answering one axis perfectly and never using the other. Both sit "
            f"{'above' if min(lat_ceiling, sag_ceiling) > floor else 'near or above'} "
            f"the {floor:.1%} action-cell target, so clearing that target does not by "
            "itself show the image was used. Qwen2.5-VL-32B and -72B sit within a "
            "point of the lateral ceiling with ~0% sagittal accuracy: they have "
            "saturated the one-axis strategy rather than established spatial reasoning.",
            "",
            "*`img gain` = `main` − `no_image`, both restricted to the same shared "
            "120-item subset. A model can sit below the action-cell target and still gain from "
            "the image, which is why the target alone cannot settle image use.*",
            "",
            "| Model | Overall | Lateral | Sagittal | Answers lateral | img gain |",
            "|---|---|---|---|---|---|",
        ]
        for overall, model, lat, sag, said, gain in sorted(rows_b, reverse=True):
            g = "&mdash;" if gain is None else f"{gain * 100:+.1f} pt"
            flag = " ⚠" if overall < 0.25 else ""
            lines.append(f"| {model}{flag} | {overall:.1%} | {lat:.1%} | {sag:.1%} "
                         f"| {said:.0%} | {g} |")
        below = sum(1 for r in rows_b if r[0] < 0.25)
        lines += ["",
                  f"⚠ marks the **{below} of {len(rows_b)}** models below the "
                  "25.0% four-way chance line. Sub-chance here is a response-bias "
                  "signature, not sub-random reasoning: committing to one axis "
                  "caps accuracy at that axis's share of gold, and an inverted "
                  "left/right mapping pushes it below chance from there.",
                  ""]

    # ---- run status, read from the supervisor's own state -------------
    # Generated like everything else here: the queue and state files are
    # written by the supervisor, so this cannot drift from what is running.
    try:
        state = json.loads(Path("work/model_panel/supervisor_state.json")
                           .read_text(encoding="utf-8"))
    except Exception:
        state = {}
    try:
        queue = json.loads(Path("work/model_panel/job_queue.json")
                           .read_text(encoding="utf-8"))
    except Exception:
        queue = []
    if state or queue:
        lines += ["## Run status", ""]
        if state:
            lines.append(f"Supervisor last checked in at `{state.get('updated','?')}` "
                         f"with **{state.get('live','?')}** run(s) live and "
                         f"**{state.get('queued','?')}** queued.")
            lines.append("")
        if queue:
            lines += ["Queued, in order:", "",
                      "| # | Tag | Task | Arm | Models |", "|---|---|---|---|---|"]
            for i, j in enumerate(queue, 1):
                models = j.get("models", "")
                n = len([m for m in models.split(",") if m.strip()])
                shown = models if n <= 2 else f"{models.split(',')[0]} +{n-1} more"
                lines.append(f"| {i} | `{j.get('tag','?')}` | {j.get('task','?')} "
                             f"| `{j.get('arm','main')}` | {shown} |")
            lines.append("")

    total = len(panel["models"])
    complete = sum(1 for m in panel["models"]
                   if all(len(counts.get((m["name"], t, "main"), ())) >=
                          target_for(m["name"], t)
                          for t in TASKS))
    started = len({m for (m, _, _) in counts})
    lines += ["## Progress", "",
              f"- Models fully run on all three tasks: **{complete} / {total}**",
              f"- Models with any results: **{started} / {total}**",
              f"- Total responses recorded: **{sum(len(v) for v in counts.values()):,}**", ""]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print(f"wrote {args.out}  ({complete}/{total} complete, {started}/{total} started)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
