"""Build model prompts that match what the annotators were shown.

One module builds the prompt for every provider and for the local runner, so
"the models got the same instructions as the humans" is enforced by
construction rather than by three implementations agreeing by luck. The
decision rule is read from the annotator package manifest -- the same file the
survey was built from -- so it cannot drift from the surveys.

Arms are defined in release/configs/model_panel_r2.json. Each returns the same
question and options; they differ only in which images are attached and which
instruction sentences are present, which is what makes them interpretable as
controls.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

ANSWER_INSTRUCTION = (
    "Answer with exactly one of the options, copied verbatim, and nothing else."
)


def load_manifest(audits_dir: Path, task: str) -> dict:
    path = Path(audits_dir) / f"task{task.lower()}_annotator_package.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _rule_for_arm(manifest: Mapping[str, Any], arm: str) -> str:
    rule = str(manifest["decision_rule_text"])
    if arm == "pp1_no_rule":
        # PP-1 removes only the zone-convention sentence, leaving the rest of
        # the rule intact, so the ablation isolates that one fact.
        marker = "Zones are measured from the direction you are now facing"
        cut = rule.find(marker)
        if cut != -1:
            tail = rule.find(". ", rule.find("either way", cut)) if "either way" in rule[cut:] else -1
            rule = (rule[:cut] + (rule[tail + 2:] if tail != -1 else "")).strip()
    return rule


def few_shot_block(manifest: Mapping[str, Any]) -> str:
    """The annotators' practice items and written reasons, as text (PP-2)."""

    lines = ["Worked examples (the same ones the human annotators were shown):"]
    for index, item in enumerate(manifest.get("practice_detail") or [], start=1):
        lines.append(f"  Example {index}: correct answer is \"{item['answer']}\".")
        if item.get("reason"):
            lines.append(f"    Why: {item['reason']}")
    return "\n".join(lines)


def build_prompt(
    item: Mapping[str, Any],
    manifest: Mapping[str, Any],
    *,
    arm: str = "main",
    option_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return {text, images, options, prompt_sha256} for one item and arm."""

    options = list(option_order or item["options"])
    parts = [_rule_for_arm(manifest, arm)]
    if arm == "pp2_few_shot":
        parts.append(few_shot_block(manifest))
    parts.append(item["question"])
    parts.append("Options:\n" + "\n".join(f"- {o}" for o in options))
    parts.append(ANSWER_INSTRUCTION)
    text = "\n\n".join(p for p in parts if p)

    frames = list(item["frames"])
    if arm == "no_image":
        images: list[str] = []
    elif arm == "single_view":
        images = frames[:1]
    else:
        images = frames

    digest = hashlib.sha256(
        json.dumps({"text": text, "images": images, "options": options},
                   sort_keys=True).encode()
    ).hexdigest()
    return {"text": text, "images": images, "options": options, "prompt_sha256": digest}


def parse_answer(raw: str, options: Sequence[str]) -> tuple[str | None, str]:
    """Map a raw completion onto one option. Returns (option or None, how)."""

    if raw is None:
        return None, "empty"
    text = raw.strip()
    for option in options:
        if text == option:
            return option, "exact"
    lowered = text.casefold()
    hits = [o for o in options if o.casefold() == lowered]
    if len(hits) == 1:
        return hits[0], "case_insensitive"

    # Reasoning checkpoints may ignore the request for answer-only output but
    # still put an unambiguous answer on their final line.  Read only that
    # terminal declaration: selecting an option mentioned anywhere in a chain
    # of thought can turn a truncated sentence into a spurious prediction.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        terminal = lines[-1].strip(" `*_#>")
        # QVQ's official reasoning format ends with LaTeX such as
        # ``\[ \boxed{\text{in front}} \]``.  It has reached EOS and made an
        # unambiguous terminal declaration, but the ordinary final-line parser
        # sees only markup.  Accept only a boxed expression on the final
        # non-empty line; never scan intermediate reasoning for an option.
        boxed = re.search(
            r"\\boxed\{\s*(?:\\(?:text|mathrm|operatorname)\{([^{}]+)\}|([^{}]+))\s*\}",
            terminal,
            re.I,
        )
        if boxed:
            boxed_answer = (boxed.group(1) or boxed.group(2)).strip(" `*_'\".!:;")
            boxed_hits = [o for o in options
                          if o.casefold() == boxed_answer.casefold()]
            if len(boxed_hits) == 1:
                return boxed_hits[0], "final_boxed"
        declared = re.search(
            r"(?i)(?:final\s+answer|answer)\s*(?::|is)\s*(.+?)\s*$", terminal)
        candidate = (declared.group(1) if declared else terminal).strip(" `*_'\".!:;")
        terminal_hits = [o for o in options if o.casefold() == candidate.casefold()]
        if len(terminal_hits) == 1:
            return terminal_hits[0], "final_marker" if declared else "final_line"
        # Task C permits the short label (e.g. "Occluded") in place of its
        # longer explanatory option. Keep this terminal-only for CoT outputs.
        label_hits = [o for o in options
                      if o.split("—", 1)[0].strip().casefold() == candidate.casefold()]
        if len(label_hits) == 1:
            return label_hits[0], "final_label"

    contained = [o for o in options if o.casefold() in lowered]
    if len(contained) == 1:
        return contained[0], "contained"
    # A model that answered with just the label, e.g. "Out of View." for the
    # option "Out of View -- It is not in frame ...". Task C's options are
    # "label -- explanation" sentences; compare against the label alone.
    label = lowered.rstrip(" .!:;")
    by_label = [o for o in options
                if o.split("—", 1)[0].strip().casefold() == label]
    if len(by_label) == 1:
        return by_label[0], "label"
    # A model that answered with a letter or index, e.g. "B" or "2".
    stripped = lowered.strip(" .:)")
    if len(stripped) == 1 and stripped.isalpha():
        index = ord(stripped) - ord("a")
        if 0 <= index < len(options):
            return options[index], "letter"
    if stripped.isdigit():
        index = int(stripped) - 1
        if 0 <= index < len(options):
            return options[index], "index"
    return None, "unparsed"
