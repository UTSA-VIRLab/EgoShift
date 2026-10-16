#!/usr/bin/env python3
"""Build a self-contained annotator survey package from a certified candidate.

Draws the shared human subset AS A SUBSET of the frozen benchmark pool
(seeded, stratified so the pool's balance properties survive), then emits one
self-contained HTML survey per task: images embedded as data URIs, one item
per screen, responses collected locally and downloaded as a JSON the
annotator emails back. Practice items come from the public_dev split (their
answers are shown, and they are outside the hidden pool), plus three synthetic
attention checks. A private manifest records the sample, gold, and hashes.

Stratification:
  Task A: exact 50/50 same/different; within class, proportional over object
          types in stable-hash order.
  Task B: per (rotation, turn) cell, proportional quotas floored to multiples
          of the cell's answer-support size, remainder assigned by largest
          remainder in whole label-blocks -- every sampled cell keeps exact
          label balance, so the survey's structural floor equals the pool's.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import html as html_module
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from egoshift.hashing import sha256_file  # noqa: E402


# Decision rules shown on EVERY item screen, and injected verbatim into the
# model prompt so humans and models are told exactly the same thing. The text
# lives in release/annotator_rules/task_<x>.html so it can be edited without
# touching this file; the plain-text version handed to the models is derived
# from the same source by stripping tags, so the two cannot drift apart.
# The rules state conventions only -- nothing in them hints at which option is
# correct for any item.
RULES_DIR = ROOT / "annotator_rules"


def load_decision_rule(task: str) -> tuple[str, str]:
    """Return (html, plain_text) for a task's decision-rule panel."""

    html = (RULES_DIR / f"task_{task.lower()}.html").read_text(encoding="utf-8")
    html = re.sub(r"<!--.*?-->", "", html, flags=re.S).strip()
    if "__MAP__" in html:
        svg = (RULES_DIR / "_zone_map.svg").read_text(encoding="utf-8").strip()
        html = html.replace("__MAP__", svg)
    # A diagram cannot be sent to a model, so its <desc> stands in for it.
    # Dropping the whole <svg> instead left the model prompt saying "use the
    # map" with no map -- an information gap that would show up as a false
    # capability gap.
    def _svg_to_text(match: re.Match[str]) -> str:
        desc = re.search(r"<desc>(.*?)</desc>", match.group(0), flags=re.S)
        return f" {desc.group(1)} " if desc else " "

    text = re.sub(r"<svg.*?</svg>", _svg_to_text, html, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = (
        text.replace("&deg;", " degrees")
        .replace("&mdash;", "-")
        .replace("&nbsp;", " ")
        .replace("&amp;", "&")
    )
    return html, " ".join(text.split())


def stable(*parts: object) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sample_task_a(public: list[dict], gold: dict[str, dict], count: int, seed: str) -> list[dict]:
    per_class = count // 2
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in public:
        by_class[gold[row["id"]]["candidate_answer_class"]].append(row)
    chosen: list[dict] = []
    for name, rows in sorted(by_class.items()):
        by_type: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_type[gold[row["id"]]["metadata"]["object_type"]].append(row)
        for type_rows in by_type.values():
            type_rows.sort(key=lambda r: stable(seed, "order", r["id"]))
        # Proportional allocation over types by largest remainder.
        total = len(rows)
        exact = {t: per_class * len(v) / total for t, v in by_type.items()}
        take = {t: int(exact[t]) for t in exact}
        remainder = per_class - sum(take.values())
        for t in sorted(exact, key=lambda t: (-(exact[t] - take[t]), t))[:remainder]:
            take[t] += 1
        for t, n in sorted(take.items()):
            chosen.extend(by_type[t][:n])
    return chosen


def sample_task_c(
    public: list[dict],
    gold: dict[str, dict],
    count: int,
    seed: str,
    object_types: dict[str, str],
) -> list[dict]:
    """Equal items per visibility class, stratified by object type within each.

    Task C has three classes, so it cannot go through the Task-B sampler --
    which is what the dispatch used to do, and it died on the first item
    because a Task-C question has no "turn left by N degrees" in it. Object
    type is balanced within a class for the same reason Task A balances it: a
    rater (or a reader of the confusion matrix) should not be able to mistake
    "chairs are usually occluded in this survey" for a perceptual finding.
    """

    per_class = count // 3
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row in public:
        by_class[str(gold[row["id"]]["answer_class"])].append(row)

    def draw(rows: list[dict], want: int) -> list[dict]:
        """Proportional-by-object-type draw, largest remainder."""

        if want <= 0 or not rows:
            return []
        by_type: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_type[object_types.get(row["id"], "unknown")].append(row)
        for type_rows in by_type.values():
            type_rows.sort(key=lambda r: stable(seed, "order", r["id"]))
        total = len(rows)
        exact = {t: want * len(v) / total for t, v in by_type.items()}
        take = {t: min(int(exact[t]), len(by_type[t])) for t in exact}
        remainder = want - sum(take.values())
        for t in sorted(exact, key=lambda t: (-(exact[t] - take[t]), t)):
            if remainder <= 0:
                break
            if take[t] < len(by_type[t]):
                take[t] += 1
                remainder -= 1
        out: list[dict] = []
        for t, n in sorted(take.items()):
            out.extend(by_type[t][:n])
        return out

    # Balance the correct answer's POSITION inside each class as well as the
    # object type. The released pool is already exactly 650/650/650 per class,
    # but a 120-item draw from it is not: one build put 22 of 40 occluded
    # answers in the third slot (chi2 = 9.1, p ~ 0.01). Occluded is the class
    # raters find hardest, so a rater who reaches for the last option when
    # unsure would score well on it for the wrong reason, and the human
    # ceiling -- the number the whole study exists to produce -- would be
    # measuring option position instead of perception.
    chosen: list[dict] = []
    classes = sorted(by_class)
    for class_index, answer_class in enumerate(classes):
        rows = by_class[answer_class]
        by_position: dict[int, list[dict]] = defaultdict(list)
        for row in rows:
            by_position[row["options"].index(gold[row["id"]]["answer"])].append(row)
        slots = sorted(by_position)
        base, extra = divmod(per_class, len(slots) or 1)
        # Rotate which positions get the spare item, so the surplus does not
        # always land on the same slot across classes.
        wants = {
            slot: base + (1 if (index - class_index) % len(slots) < extra else 0)
            for index, slot in enumerate(slots)
        }
        for slot in slots:
            chosen.extend(draw(by_position[slot], wants[slot]))
    return chosen


QUESTION_B = re.compile(r"turn (left|right) by (\d+) degrees")


def sample_task_b(public: list[dict], gold: dict[str, dict], count: int, seed: str) -> list[dict]:
    cells: dict[tuple[int, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in public:
        match = QUESTION_B.search(row["question"])
        if not match:
            raise SystemExit(f"unparseable Task-B question: {row['id']}")
        turn, rotation = match.group(1), int(match.group(2))
        cells[(rotation, turn)][gold[row["id"]]["answer"]].append(row)
    for cell in cells.values():
        for rows in cell.values():
            rows.sort(key=lambda r: stable(seed, "order", r["id"]))
    # Uniform cells: every (rotation, turn) cell contributes the same number
    # of items, and that number must divide evenly over the cell's answer
    # support (2 for 30/60/120, 3 for 90/180), so the per-cell size is a
    # multiple of 6 and the survey a multiple of 60. Uniform beats
    # pool-proportional here on three counts: the survey floor is exactly
    # 43.33% (the historical reference line); the scarce 180-degree cells get
    # full power instead of ~6 items; and oversampling the hard cells makes
    # the measured human ceiling a lower bound on pool-wide performance.
    if count % (6 * len(cells)) != 0:
        raise SystemExit(
            f"survey size {count} is not a multiple of 6 x {len(cells)} cells"
        )
    per_cell = count // len(cells)
    quotas: dict[tuple[int, str], int] = {}
    for key, cell in cells.items():
        support = len(cell)
        if per_cell % support != 0:
            raise SystemExit(f"per-cell size {per_cell} not divisible by support {support}")
        per_label = per_cell // support
        for label, rows in cell.items():
            if len(rows) < per_label:
                raise SystemExit(
                    f"cell {key} label {label!r} has {len(rows)} < {per_label} items"
                )
        quotas[key] = per_cell
    chosen: list[dict] = []
    for key, quota in sorted(quotas.items()):
        support = len(cells[key])
        per_label = quota // support
        for label, rows in sorted(cells[key].items()):
            chosen.extend(rows[:per_label])
    return chosen


# Practice is training, not decoration, but it is kept at the minimum that
# still does its job: ONE worked item per answer option.
#
# That floor is what it is for three reasons. (1) Coverage: a rater must not
# finish training having never seen one of the options. (2) Misconception
# detection: each task has a single dominant failure mode, and one item per
# option exposes it -- a Task-B rater who reads the turn allocentrically
# inverts left/right and front/behind, so the four labels catch it; a Task-C
# rater who conflates out-of-view with occluded meets both; a Task-A rater who
# judges by appearance meets the negative case, which by construction is
# always two instances of the SAME object type, i.e. the look-alike case is
# not a lucky draw but a guarantee of the mining contract. (3) Depth comes
# from the redo loop rather than from length: a rater who misses is shown the
# rule again and repeats the block, so the struggling rater gets more practice
# and the competent one is not taxed.
PRACTICE_COUNT = {"A": 2, "B": 4, "C": 3}
# Redo is advisory, never blocking. One slip is tolerated when there is room
# for it; with only two items any error means chance-level, so it prompts.
PRACTICE_MAX_WRONG = {2: 0}  # by practice-block size; default 1


def display_noun(object_type: str) -> str:
    """CamelCase object type -> plain noun, e.g. GarbageCan -> garbage can."""

    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(object_type))
    return " ".join(spaced.split()).casefold() or "object"


def _zone_of(bearing: float) -> str:
    magnitude = abs(bearing)
    if magnitude < 45.0:
        return "in front"
    if magnitude > 135.0:
        return "behind"
    return "to the right" if bearing > 0 else "to the left"


def _side_phrase(bearing: float) -> str:
    magnitude = abs(bearing)
    if magnitude > 170.0:
        return "almost directly behind you"
    side = "right" if bearing > 0 else "left"
    return f"about {magnitude:.0f}\u00b0 to your {side}"


def build_reasons(task: str, candidate_root: Path, dev_gold: dict) -> dict[str, str]:
    """Short, concrete feedback for each practice item, from its own gold.

    Feedback that only says "correct answer: X" teaches nothing; the rater
    needs to know *why*, in the terms of the decision rule.
    """

    reasons: dict[str, str] = {}
    if task == "B":
        path = candidate_root / "private/provenance/task_b_public_dev.jsonl"
        for row in load_jsonl(path):
            evidence = row.get("geometry_evidence") or {}
            before = evidence.get("source_center_bearing_degrees")
            after = evidence.get("post_center_bearing_degrees")
            if before is None or after is None:
                continue
            noun = str(row.get("target_category") or row.get("target_type") or "object")
            turn = row.get("turn_direction")
            rotation = row.get("rotation_degrees")
            zone = _zone_of(float(after))
            band = {
                "in front": "within 45\u00b0 of straight ahead",
                "behind": "more than 135\u00b0 away",
            }.get(zone, "between 45\u00b0 and 135\u00b0")
            reasons[row["id"]] = (
                f"In View 1 the {noun} was {_side_phrase(float(before))}. "
                f"You turn {turn} by {rotation}\u00b0, so it swings a further "
                f"{rotation}\u00b0 the other way. Relative to the direction you "
                f"now face it ends up {_side_phrase(float(after))} — {band}, "
                f"which is \u201c{zone}\u201d."
            )
        return reasons
    if task == "A":
        for row in dev_gold.values():
            metadata = row.get("metadata") or {}
            noun = display_noun(str(metadata.get("object_type") or "object"))
            same = row.get("candidate_answer_class") == "same"
            reasons[row["id"]] = (
                f"Same object: both boxes mark one physical {noun}, seen from two "
                f"positions. Check where it sits relative to the furniture around "
                f"it — that is what settles it, not how it looks."
                if same else
                f"Two different objects: this room contains more than one {noun}, "
                f"and each box marks a different one. They can look identical, so "
                f"compare their positions in the room."
            )
        return reasons
    # Task C's dev ANSWERS file carries only {id, answer, answer_class}: no
    # target and no occluder. Reading them from it produced "the object" for
    # every noun and -- because it looked for Task A's field name
    # candidate_answer_class, which Task C does not use -- fell through to the
    # Occluded branch for all three practice items, so the "Visible" and
    # "Out of View" items each told the rater the opposite of their own
    # answer. The provenance file is where those fields actually live.
    rows = load_jsonl(candidate_root / "private/provenance/task_c_public_dev.jsonl")
    for row in rows:
        evidence = row.get("gold_evidence") or {}
        noun = display_noun(str(row.get("target_oid") or "object").split("|", 1)[0])
        answer_class = row.get("answer_class") or row.get("candidate_answer_class")
        occluder = display_noun(
            str((evidence.get("occluder") or {}).get("object_id") or "").split("|", 1)[0]
        )
        if answer_class == "visible":
            text = f"Visible: some part of the {noun} can still be seen in View 2."
        elif answer_class == "out_of_fov":
            text = (
                f"Out of View: the camera has turned away, so the {noun} is not in "
                f"the picture at all. Nothing is blocking it — it is simply "
                f"outside the frame."
            )
        else:
            blocker = f"the {occluder}" if occluder else "another object"
            text = (
                f"Occluded: the camera still points at the {noun}, but {blocker} is "
                f"completely in front of it, so no part of it shows."
            )
        reasons[row["id"]] = text
    return reasons


def copy_frame(source: Path, images_dir: Path) -> str:
    """Copy a frame into the package and return its relative href.

    Frames are emitted as ordinary files rather than embedded as data URIs.
    Inlining 254 base64 frames produced a single 83 MB <script> line, which
    browsers will not render from file:// -- the annotator saw a blank page.
    Separate files also load lazily, one screen at a time, instead of forcing
    the whole survey into memory before the first paint. Names are content
    hashes, so a frame reused by several screens is stored once.
    """

    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:20]
    destination = images_dir / f"{digest}.png"
    if not destination.exists():
        images_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    return f"images/{destination.name}"


HTML_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>EgoShift — Task __TASK__</title>
<style>
 *{box-sizing:border-box;}
 html,body{height:100%;}
 body{margin:0;background:#0b0e19;color:#edf1ff;font-family:system-ui,sans-serif;font-size:15px;line-height:1.45;}
 button{background:#6ea8ff;color:#0b0e19;border:0;border-radius:8px;padding:9px 18px;font-size:15px;cursor:pointer;}
 button.secondary{background:#2b344f;color:#edf1ff;}
 button:disabled{opacity:.4;cursor:default;}
 .muted{color:#a9b3d9;font-size:13px;}
 /* Full-page card: intro, practice summary, review, done. These MAY scroll. */
 .card{max-width:860px;margin:24px auto;background:#141827;border:1px solid #2b344f;border-radius:12px;padding:24px;}
 .card input{font-size:16px;padding:6px;min-width:16em;}
 /* Item screens: one viewport, no page scroll. Images left, controls right. */
 .screen{height:100vh;display:flex;flex-direction:column;padding:10px 16px 12px;gap:10px;}
 .bar{display:flex;align-items:center;gap:12px;width:100%;}
 .bar progress{flex:1;height:6px;}
 .main{flex:1;min-height:0;display:grid;grid-template-columns:minmax(0,640px) 1fr;gap:22px;width:100%;}
 .media{display:flex;flex-direction:column;gap:10px;min-height:0;align-items:center;justify-content:center;}
 .media img{flex:0 1 auto;min-height:0;width:auto;max-width:100%;max-height:100%;object-fit:contain;background:#05070f;border:1px solid #2b344f;border-radius:8px;cursor:zoom-in;}
 .side{min-height:0;display:flex;flex-direction:column;overflow-y:auto;background:#141827;border:1px solid #2b344f;border-radius:12px;padding:16px;}
 .q{margin:2px 0 8px;font-size:16px;}
 label{display:block;background:#101524;border:1px solid #2b344f;border-radius:8px;padding:9px 12px;margin:5px 0;cursor:pointer;}
 label:has(input:checked){border-color:#6ea8ff;background:#18233f;}
 .nav{display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap;}
 details.rule{background:#101524;border:1px solid #2b344f;border-radius:8px;padding:8px 12px;margin-top:10px;}
 details.rule summary{cursor:pointer;color:#6ea8ff;font-weight:600;}
 details.rule ol,details.rule ul{margin:8px 0 8px 18px;padding:0;}
 details.rule li{margin:4px 0;}
 details.rule svg{display:block;margin:12px auto;width:100%;max-width:300px;height:auto;}
 .practice{color:#58d68d;font-weight:600;}
 .fb{border-radius:8px;padding:10px 12px;margin:8px 0;border:1px solid;}
 .fb.ok{background:#12281c;border-color:#2f6b45;}
 .fb.no{background:#2a1618;border-color:#7a3340;}
 .fb b{display:block;margin-bottom:4px;}
 .imgerr{color:#ff9d9d;border:1px solid #5b2b2b;border-radius:8px;padding:10px;}
 .zoomhint{color:#a9b3d9;font-size:12px;text-align:center;}
 #zoom{position:fixed;inset:0;background:rgba(4,6,14,.95);display:none;align-items:center;justify-content:center;z-index:99;cursor:zoom-out;padding:16px;}
 #zoom img{max-width:min(96vw,1280px);max-height:min(92vh,960px);border:1px solid #2b344f;cursor:zoom-out;}
 /* Narrow window: stack, let the page scroll normally. */
 @media (max-width:920px){
   .screen{height:auto;min-height:100vh;}
   .main{grid-template-columns:1fr;}
   .media img{max-height:44vh;}
 }
</style></head><body><div id="root"></div>
<div id="zoom" onclick="this.style.display='none'"><img id="zoomimg" alt="enlarged view"/></div>
<script>
const ITEMS = __ITEMS__;
const RULE = __RULE__;
function zoom(src){ document.getElementById('zoomimg').src = src;
  document.getElementById('zoom').style.display = 'flex'; }
let index = -1; const responses = {}; let annotator = "";
const PRACTICE_N = ITEMS.filter(i => i.practice).length;
let practiceAck = false;
function reveal(id){ responses[id].revealed = true; render(); }
function redoPractice(){ ITEMS.filter(i=>i.practice).forEach(i=>{ delete responses[i.id]; });
  practiceAck = false; index = 0; render(); }
function ackPractice(){ practiceAck = true; render(); }
function practiceScore(){ let n=0, c=0;
  for(const it of ITEMS){ if(!it.practice) continue; const r=responses[it.id];
    if(r && r.choice){ n++; if(r.choice === it.practice) c++; } }
  return {answered:n, correct:c, total:PRACTICE_N}; }
const root = document.getElementById("root");
function start(){
  annotator = (document.getElementById("aid").value || "").trim();
  if(!annotator){ alert("Please enter your name."); return; }
  index = 0; render();
}
function pick(id, value){ responses[id] = responses[id] || {}; responses[id].choice = value; responses[id].t = Date.now(); render(); }
function flag(id){ responses[id] = responses[id] || {}; responses[id].flagged = !responses[id].flagged; render(); }
function nav(step){ index += step; render(); }
function finish(){
  const ps = practiceScore();
  const practice_responses = {};
  ITEMS.filter(i=>i.practice).forEach(i=>{ if(responses[i.id]) practice_responses[i.id] = {choice: responses[i.id].choice, correct: responses[i.id].choice === i.practice}; });
  const main_responses = {};
  Object.keys(responses).forEach(k => { if(!ITEMS.some(i=>i.practice && i.id===k)) main_responses[k] = responses[k]; });
  const payload = {task: "__TASK__", package_sha_note: "__PKGID__", annotator_name: annotator,
    finished: new Date().toISOString(), practice: {score: ps.correct, of: ps.total, responses: practice_responses},
    responses: main_responses};
  const blob = new Blob([JSON.stringify(payload, null, 1)], {type: "application/json"});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  const slug = annotator.replace(/[^A-Za-z0-9]+/g, "_").replace(/^_+|_+$/g, "") || "rater";
  a.download = `egoshift_task__TASK___${slug}.json`;
  a.click();
  root.innerHTML = `<div class="card"><h2>Done — thank you.</h2><p>A file named <code>egoshift_task__TASK___${slug}.json</code> was downloaded. Please email it back to the researcher.</p></div>`;
}
function render(){
  if(index < 0){
    root.innerHTML = `<div class="card"><h2>EgoShift — Task __TASK__</h2>
      <p>__INTRO__</p>
      <details class="rule" open><summary>How to decide — read this first</summary>${RULE}</details>
      <p class="muted">${ITEMS.length} screens including practice. Please work in one sitting. Flagging an unclear item never reduces your payment.</p>
      <p>Your name: <input id="aid" placeholder="first and last name"/></p>
      <button onclick="start()">Begin</button></div>`;
    return;
  }
  if(index >= ITEMS.length){
    const missing = ITEMS.filter(it => !it.practice && !(responses[it.id]||{}).choice);
    root.innerHTML = `<div class="card"><h2>Review</h2><p>${missing.length ? "Unanswered items: " + missing.length : "All items answered."}</p>
      ${missing.length ? '<button class="secondary" onclick="nav(-1)">Back</button>' : ''}
      <button onclick="finish()" ${missing.length ? "disabled" : ""}>Download my responses</button></div>`;
    return;
  }
  if(index === PRACTICE_N && PRACTICE_N > 0 && !practiceAck){
    const ps = practiceScore();
    const allowedWrong = ps.total <= 2 ? 0 : 1;
    const ok = (ps.total - ps.correct) <= allowedWrong;
    root.innerHTML = `<div class="card"><h2>Practice complete</h2>
      <p>You answered ${ps.correct} of ${ps.total} practice questions correctly.</p>
      ${ok ? '<p class="practice">That is a good sign you have the rule. The real questions start now — no answers are shown from here on.</p>'
           : '<div class="fb no"><b>Worth another look</b>Several practice answers did not match. Please re-read “How to decide” below and run the practice again — it only takes a couple of minutes, and it is the best predictor of how the real questions go. Your pay is not affected either way.</div>'}
      <details class="rule" open><summary>How to decide</summary>${RULE}</details>
      <p><button onclick="redoPractice()" class="${ok ? 'secondary' : ''}">Redo practice</button>
         <button onclick="ackPractice()" class="${ok ? '' : 'secondary'}">Start the real questions</button></p></div>`;
    return;
  }
  const it = ITEMS[index];
  const chosen = (responses[it.id]||{}).choice;
  const shown = (responses[it.id]||{}).revealed;
  const frames = it.frames.map((u,i) => `<img src="${u}" alt="View ${i+1}" loading="lazy" onclick="zoom(this.src)" onerror="this.replaceWith(Object.assign(document.createElement('p'),{className:'imgerr',textContent:'Image not found: ${u} — please make sure you unzipped the whole folder and kept the images folder next to this page.'}))"/>`).join("");
  const options = it.options.map(o => `<label><input type="radio" name="opt" ${chosen===o?"checked":""} onclick="pick('${it.id}', this.value)" value="${o.replace(/"/g,'&quot;')}"/> ${o}</label>`).join("");
  root.innerHTML = `
   <div class="screen">
    <div class="bar">
      <progress max="${ITEMS.length}" value="${index+1}"></progress>
      <span class="muted">Screen ${index+1} of ${ITEMS.length}${it.practice ? ` <span class="practice">— PRACTICE ${index+1} of ${PRACTICE_N} (answer your best guess, then check)</span>` : ''}</span>
    </div>
    <div class="main">
      <div class="media">${frames}<div class="zoomhint">Click a picture to enlarge it.</div></div>
      <div class="side">
        <div class="q">${it.question}</div>
        ${options}
        ${it.practice && shown ? `<div class="fb ${chosen===it.practice?'ok':'no'}">
            <b>${chosen===it.practice?'Correct':'Not quite'}</b>
            The answer is <b style="display:inline">${it.practice}</b>. ${it.reason||""}</div>` : ""}
        <div class="nav">
          <button class="secondary" onclick="nav(-1)" ${index===0?"disabled":""}>Back</button>
          ${it.practice && !shown ? `<button onclick="reveal('${it.id}')" ${!chosen?"disabled":""}>Check answer</button>`
             : `<button onclick="nav(1)" ${!chosen?"disabled":""}>Next</button>`}
          ${it.practice ? "" : `<button class="secondary" onclick="flag('${it.id}')">${(responses[it.id]||{}).flagged ? "Unflag" : "Flag as unclear"}</button>`}
        </div>
        <details class="rule" open><summary>How to decide</summary>${RULE}</details>
      </div>
    </div>
   </div>`;
}
render();
</script></body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=("A", "B", "C"))
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--items", type=int, default=120)
    parser.add_argument("--practice", type=int, default=0,
                        help="0 = use the per-task default in PRACTICE_COUNT.")
    parser.add_argument("--seed", default="egoshift-shared-subset-r2-v1")
    parser.add_argument(
        "--practice-exclude", action="append", default=[], metavar="ID",
        help=(
            "Drop this dev item from the practice draw and take the next "
            "candidate for its answer. Use this to replace one practice item "
            "without touching the seed -- changing the seed would reshuffle "
            "the survey items too, and the whole point of a shared seed is "
            "that every task draws from the same frozen ordering."
        ),
    )
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.out_dir.exists() or args.manifest.exists():
        raise SystemExit("refusing to overwrite an existing package/manifest")
    args.out_dir.mkdir(parents=True)

    task = args.task
    name = f"task_{task.lower()}"
    public = load_jsonl(args.candidate_root / f"public/questions/{name}_hidden_test.jsonl")
    if task == "A":
        gold_rows = load_jsonl(args.candidate_root / f"private/{name}_hidden_test_gold.jsonl")
        dev_public = load_jsonl(args.candidate_root / f"public/questions/{name}_public_dev.jsonl")
        dev_gold_rows = load_jsonl(args.candidate_root / f"private/{name}_public_dev_gold.jsonl")
    else:
        gold_rows = load_jsonl(args.candidate_root / f"private/gold/{name}_hidden_test.jsonl")
        dev_public = load_jsonl(args.candidate_root / f"public/questions/{name}_public_dev.jsonl")
        dev_gold_rows = load_jsonl(args.candidate_root / f"development/answers/{name}_public_dev.jsonl")
    gold = {r["id"]: r for r in gold_rows}
    dev_gold = {r["id"]: r for r in dev_gold_rows}

    def dev_or_hidden_answer(item_id: str) -> str:
        return str(gold[item_id]["answer"])

    if task == "A":
        chosen = sample_task_a(public, gold, args.items, args.seed)
    elif task == "C":
        # Object type lives in the private provenance for Task C, not in gold.
        types = {
            row["id"]: str(row.get("target_oid") or "").split("|")[0] or "unknown"
            for row in load_jsonl(
                args.candidate_root / f"private/provenance/{name}_hidden_test.jsonl"
            )
        }
        chosen = sample_task_c(public, gold, args.items, args.seed, types)
    else:
        chosen = sample_task_b(public, gold, args.items, args.seed)

    # Practice covers every answer option as evenly as the dev pool allows, so
    # no rater finishes training having never seen one of the labels.
    practice_total = args.practice or PRACTICE_COUNT[task]
    excluded = set(args.practice_exclude)
    unknown = excluded - {r["id"] for r in dev_public}
    if unknown:
        raise SystemExit(f"--practice-exclude names unknown dev item(s): {sorted(unknown)}")
    dev_sorted = [
        r for r in sorted(dev_public, key=lambda r: stable(args.seed, "practice", r["id"]))
        if r["id"] not in excluded
    ]

    def dev_answer(row: dict) -> str:
        """Balance practice on the semantic class, not the option string.

        Task A renders its negative option per noun ("Two different chairs",
        "Two different beds"), so balancing on the option text treats every
        noun as its own label and starves the positive class -- the first
        build came out 1 same / 7 different.
        """

        gold_row = dev_gold[row["id"]]
        return str(
            gold_row.get("candidate_answer_class")
            or gold_row.get("answer_class")
            or gold_row["answer"]
        )

    by_answer: dict[str, list[dict]] = defaultdict(list)
    for row in dev_sorted:
        by_answer[dev_answer(row)].append(row)
    practice: list[dict] = []
    labels = sorted(by_answer)
    cursor = {label: 0 for label in labels}
    while len(practice) < practice_total:
        progressed = False
        for label in labels:
            if len(practice) >= practice_total:
                break
            pool = by_answer[label]
            if cursor[label] < len(pool):
                practice.append(pool[cursor[label]])
                cursor[label] += 1
                progressed = True
        if not progressed:
            break
    practice.sort(key=lambda r: stable(args.seed, "practice-order", r["id"]))
    if task == "C" and practice_total >= 3:
        # Contrastive practice: the whole practice block is ONE target's
        # triple. Every Task C target exists as visible, out_of_fov and
        # occluded on a byte-identical View 1, so the three practice screens
        # show the same object and the same starting view with three
        # different second views -- which is the distinction raters actually
        # confuse (a real rater's errors were 8x occluded->out_of_fov and 0x
        # the reverse). Round-robin practice showed three unrelated scenes
        # and left that contrast implicit.
        by_target: dict[tuple, dict[str, dict]] = defaultdict(dict)
        prov_rows = load_jsonl(
            args.candidate_root / "private/provenance/task_c_public_dev.jsonl"
        )
        available = {row["id"] for row in dev_sorted}
        for row in prov_rows:
            if row["id"] in available:
                by_target[
                    (row["physical_scene_component_id"], row["target_oid"])
                ][row["answer_class"]] = row["id"]
        triples = {
            key: ids for key, ids in by_target.items() if len(ids) == 3
        }
        contrastive_practice: dict[str, str] = {}
        if triples:
            key = min(
                triples, key=lambda k: stable(args.seed, "practice-triple", *k)
            )
            dev_by_id = {row["id"]: row for row in dev_sorted}
            practice = [
                dev_by_id[triples[key][name]]
                for name in ("visible", "out_of_fov", "occluded")
            ]
            contrastive_practice = triples[key]
    used = {row["id"] for row in practice}
    attention_sources = [row for row in dev_sorted if row["id"] not in used][:3]
    reasons = build_reasons(task, args.candidate_root, dev_gold)
    if task == "C" and practice_total >= 3 and contrastive_practice:
        same = (
            " This is the SAME object and the SAME starting view as the "
            "previous screen — only the second camera position differs."
        )
        reasons[contrastive_practice["out_of_fov"]] = (
            reasons.get(contrastive_practice["out_of_fov"], "") + same
        )
        reasons[contrastive_practice["occluded"]] = (
            reasons.get(contrastive_practice["occluded"], "")
            + same
            + " Compare it with the previous screen: there the camera turned "
            "away from the object's spot; here the spot is still on screen "
            "but something stands in front."
        )

    images_dir = args.out_dir / "images"

    def frames_of(row: dict) -> list[str]:
        return [copy_frame(args.candidate_root / f, images_dir) for f in row["frames"]]

    items = []
    for row in practice:
        items.append({
            "id": f"practice_{row['id']}",
            "frames": frames_of(row),
            "question": row["question"],
            "options": row["options"],
            "practice": dev_gold[row["id"]]["answer"],
            "reason": reasons.get(row["id"], ""),
        })
    survey_rows = []
    for row in chosen:
        survey_rows.append({
            "id": row["id"],
            "frames": frames_of(row),
            "question": row["question"],
            "options": row["options"],
            "practice": None,
        })
    attention = []
    for index, row in enumerate(attention_sources):
        # Instruct the option that CONTRADICTS what the pictures show.
        #
        # This previously read `row["options"][index % len(row["options"])]`,
        # which picked the target by index parity and ignored the source item
        # entirely. Whenever that landed on the row's true answer, the check
        # became a free pass: a rater who never read the instruction and just
        # answered the picture still scored correct, so the check measured
        # nothing. In the r2 Task-A build two of three checks landed that way,
        # and all four raters "passed" both while failing the only one that
        # required reading -- a 2/3 that looked like inattention but was an
        # artifact of how the checks were built.
        truth = str(dev_gold[row["id"]]["answer"])
        distractors = [o for o in row["options"] if o != truth]
        if not distractors:                      # never expected; fail loudly
            raise SystemExit(
                f"attention check {index}: source row {row['id']} has no option "
                f"other than its own answer; cannot build a discriminating check"
            )
        target = distractors[index % len(distractors)]
        attention.append({
            "id": f"attention_{index}",
            "frames": frames_of(row),
            "question": (
                "Attention check: ignore the images and simply select the "
                f"option that reads “{html_module.escape(target)}”."
            ),
            "options": row["options"],
            "practice": None,
            "_expected": target,
        })
    ordered = sorted(survey_rows, key=lambda r: stable(args.seed, "shuffle", r["id"]))
    positions = sorted(
        range(len(ordered)),
        key=lambda i: stable(args.seed, "attention-pos", i),
    )[: len(attention)]
    for position, check in zip(sorted(positions), attention):
        ordered.insert(position + len(ordered) // 6, dict(check))
    items.extend(ordered)
    for item in items:
        item.pop("_expected", None)

    rule_html, rule_text = load_decision_rule(task)
    # The id must fingerprint the CONTENT, not just its shape. Deriving it from
    # (seed, task, count) alone meant two builds with the same size shared an
    # id: swapping a practice item left the id at 3540c42f0553506b, and had a
    # SURVEY item been swapped instead, responses would have been scored
    # against the wrong gold with nothing to reveal it. Hash the item ids and
    # the practice ids in.
    legacy_package_id = stable(args.seed, task, len(items))[:16]
    package_id = stable(
        args.seed,
        task,
        len(items),
        "|".join(sorted(r["id"] for r in chosen)),
        "|".join(sorted(r["id"] for r in practice)),
    )[:16]
    page = (
        HTML_TEMPLATE
        .replace("__TASK__", task)
        .replace("__PKGID__", package_id)
        .replace("__INTRO__", {
            "A": "Two pictures of the same room. Each shows an object of the same kind, "
                 "highlighted with a box. Decide whether the two boxes show the same "
                 "object or two different objects.",
            "B": "One picture, then an imagined turn. Answer where the named object "
                 "would be relative to your new facing direction.",
            "C": "Two pictures of the same room, taken from different places. View 1 "
                 "boxes an object. Decide what is true of that object in View 2: "
                 "you can see it, the camera turned away from it, or something "
                 "hides it completely.",
        }[task])
        .replace("__RULE__", json.dumps(rule_html))
        .replace("__ITEMS__", json.dumps(items, ensure_ascii=False))
    )
    html_path = args.out_dir / f"Task{task}.html"
    html_path.write_text(page, encoding="utf-8")

    manifest = {
        "schema_version": "0.1.0",
        "audit": f"task{task.lower()}_annotator_package",
        "seed": args.seed,
        "package_id": package_id,
        "decision_rule_text": rule_text,
        "decision_rule_source": str(RULES_DIR / f"task_{task.lower()}.html"),
        "decision_rule_parity_note": (
            "This exact text is shown to annotators on every screen and must be "
            "injected verbatim into the model prompt; otherwise the human-model "
            "gap includes a prompt-information gap. Two differences remain by "
            "design and are measured, not asserted: annotators also see the zone "
            "diagram (models get its <desc> instead) and annotators complete a "
            "practice block with feedback (models are zero-shot by default). See "
            "docs/Prompt_Parity.md for the required ablations PP-1/PP-2/PP-3; do "
            "not describe the instructions as identical."
        ),
        "candidate_root": str(args.candidate_root),
        "survey_items": len(chosen),
        "legacy_package_id": legacy_package_id,
        "practice_excluded": sorted(excluded),
        "practice_items": len(practice),
        "practice_max_wrong_before_redo_prompt": 0 if len(practice) <= 2 else 1,
        "practice_detail": [
            {
                "id": row["id"],
                "answer": dev_gold[row["id"]]["answer"],
                "reason": reasons.get(row["id"], ""),
            }
            for row in practice
        ],
        "practice_answer_coverage": dict(Counter(dev_answer(row) for row in practice)),
        "attention_checks": [
            {"id": f"attention_{i}", "expected": a["question"].split("“")[1].split("”")[0]}
            for i, a in enumerate(attention)
        ],
        "sampled_ids": sorted(r["id"] for r in chosen),
        "strata": dict(Counter(
            gold[r["id"]]["candidate_answer_class"] if task == "A" else gold[r["id"]]["answer"]
            for r in chosen
        )),
        # Gold is the exact option STRING the survey renders, because that is
        # what a response file contains. Storing Task A's class here instead
        # made every comparison fail and scored a real rater at 0%.
        "gold": {r["id"]: dev_or_hidden_answer(r["id"]) for r in chosen},
        "gold_class": {
            r["id"]: str(
                gold[r["id"]].get("candidate_answer_class")
                or gold[r["id"]].get("answer_class")
                or gold[r["id"]]["answer"]
            )
            for r in chosen
        },
        "html": {"path": str(html_path), "sha256": sha256_file(html_path)},
        "images": {
            "dir": str(images_dir),
            "files": len(list(images_dir.glob("*.png"))),
            "bytes": sum(f.stat().st_size for f in images_dir.glob("*.png")),
        },
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    print(json.dumps({
        "task": task, "survey_items": len(chosen), "screens": len(items),
        "practice": len(practice),
        "practice_coverage": manifest["practice_answer_coverage"],
        "strata": manifest["strata"],
        "html_kb": round(html_path.stat().st_size / 1e3, 1),
        "image_files": manifest["images"]["files"],
        "images_mb": round(manifest["images"]["bytes"] / 1e6, 1),
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
