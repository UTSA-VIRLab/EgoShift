#!/usr/bin/env python3
"""Run the model panel over a task split, one JSONL record per item.

Design points that matter for the paper:
  * Prompts come from egoshift.model_prompting, the same builder the parity
    tests cover, so every provider and the local runner send the same text.
  * Every record carries the model id, arm, prompt hash, raw completion, the
    parsed option, and how it parsed -- so a number in the paper can be traced
    to a specific response, and refusals/parse failures are counted rather
    than silently dropped.
  * Results append to a JSONL keyed by (model, arm, item, permutation), and a
    rerun skips finished work. Rate limits, credit exhaustion and Ctrl-C cost
    only the item in flight.
  * Failures retry with backoff and are recorded as errors, never as wrong
    answers -- scoring an API timeout as an incorrect response would bias the
    result downward.
"""

from __future__ import annotations

import argparse
import base64
import collections
import hashlib
import json
import mimetypes
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from egoshift.model_prompting import build_prompt, load_manifest, parse_answer  # noqa: E402

# Published list prices, USD per million tokens (input, output), checked
# 2026-08-25 against each provider's own pricing page. Used only for the
# in-run spend guard -- the authoritative number is always the provider's
# own billing console, and these are list rates that ignore any discount.
PRICING = {
    "GPT-5.1":                  (1.25, 10.00),
    "GPT-5.6 Sol":              (4.00, 20.00),
    "GPT-5.6 Sol (thinking)":   (4.00, 20.00),
    "Claude Sonnet 5":          (2.00, 10.00),
    "Claude Fable 5":           (10.00, 50.00),
    "Claude Opus 5":            (5.00, 25.00),
    "Claude Opus 5 (extended)": (5.00, 25.00),
    "Gemini-2.5-Pro":           (1.25, 10.00),
    "Gemini 3 Pro":             (2.00, 12.00),
    "Gemini 3 Pro (thinking)":  (2.00, 12.00),
    "Gemini 3 Flash":           (0.75,  3.75),
    "Claude Haiku 4.5":         (1.00,  5.00),
    # Flash Lite list pricing not separately published at time of writing;
    # priced at the Flash rate, which over-states it. Over-stating is the safe
    # direction for a guard.
    "Gemini 3 Flash Lite":      (0.75,  3.75),
}

# A model missing from PRICING must not be treated as free -- that silently
# disables the spend guard for exactly the model nobody costed. Charge it at
# the most expensive rate on the panel so the guard trips early instead of
# never, and say so once.
FALLBACK_RATE = (5.00, 25.00)
_warned_unpriced: set[str] = set()


def credit_balance_exhausted(error: str | None) -> bool:
    """True only for provider responses that require a billing-state change.

    Retrying a daily rate limit later is useful; retrying an empty prepaid
    balance immediately can append hundreds of identical failures.  Keep this
    deliberately narrow so transient 429/529 responses still follow the
    configured retry policy.
    """

    text = (error or "").casefold()
    return any(marker in text for marker in (
        "credit_balance_exhausted",
        "no credits remaining",
        "billing_hard_limit_reached",
    ))


def encode_image(path: Path) -> tuple[str, str]:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return mime, base64.b64encode(path.read_bytes()).decode()


def call_openai(model_id: str, prompt: Mapping[str, Any], root: Path, params: Mapping[str, Any]) -> str:
    from openai import OpenAI
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt["text"]}]
    for rel in prompt["images"]:
        mime, b64 = encode_image(root / rel)
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    kwargs: dict[str, Any] = {"model": model_id, "messages": [{"role": "user", "content": content}]}
    if params.get("reasoning_effort"):
        kwargs["reasoning_effort"] = params["reasoning_effort"]
    completion = OpenAI().chat.completions.create(**kwargs)
    usage = getattr(completion, "usage", None)
    return (completion.choices[0].message.content or ""), {
        "input_tokens": getattr(usage, "prompt_tokens", None),
        "output_tokens": getattr(usage, "completion_tokens", None),
        "stop_reason": completion.choices[0].finish_reason,
    }


def call_anthropic(model_id: str, prompt: Mapping[str, Any], root: Path, params: Mapping[str, Any]) -> str:
    import anthropic
    content: list[dict[str, Any]] = []
    for rel in prompt["images"]:
        mime, b64 = encode_image(root / rel)
        content.append({"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}})
    content.append({"type": "text", "text": prompt["text"]})
    kwargs: dict[str, Any] = {"model": model_id, "max_tokens": 2048,
                              "messages": [{"role": "user", "content": content}]}
    if params.get("thinking"):
        # budget_tokens is REMOVED on Opus 5 / Sonnet 5 / Fable 5 and returns a
        # 400; adaptive thinking plus output_config.effort replaces it. Thinking
        # is on by default on Opus 5, so the reasoning arm differs from the main
        # arm by effort, not by switching thinking on.
        kwargs["thinking"] = {"type": "adaptive"}
        kwargs["output_config"] = {"effort": params.get("effort", "high")}
        kwargs["max_tokens"] = 16000
    message = anthropic.Anthropic().messages.create(**kwargs)
    text = "".join(b.text for b in message.content if getattr(b, "type", "") == "text")
    usage = getattr(message, "usage", None)
    return text, {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "stop_reason": getattr(message, "stop_reason", None),
    }


def call_google(model_id: str, prompt: Mapping[str, Any], root: Path, params: Mapping[str, Any]) -> str:
    from google import genai
    from google.genai import types
    parts: list[Any] = [types.Part.from_text(text=prompt["text"])]
    for rel in prompt["images"]:
        mime, b64 = encode_image(root / rel)
        parts.append(types.Part.from_bytes(data=base64.b64decode(b64), mime_type=mime))
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    # params was previously ignored here, which silently made the Tier-3
    # "(thinking)" Gemini row a byte-identical rerun of the Tier-2 row: paid
    # for twice, and not a reasoning variant at all. Gemini 3.x takes a
    # thinking_level rather than a token budget.
    config = None
    if params.get("thinking"):
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=params.get("thinking_level", "high"),
            )
        )
    result = client.models.generate_content(
        model=model_id, contents=parts, config=config)
    usage = getattr(result, "usage_metadata", None)
    # Gemini bills thinking as output but reports it in its OWN field, not in
    # candidates_token_count. Reading only the latter under-reports a Gemini 3
    # Pro call by roughly 600x (2 visible tokens against ~1,900 thought ones)
    # and would make any spend guard useless.
    visible = getattr(usage, "candidates_token_count", None) or 0
    thoughts = getattr(usage, "thoughts_token_count", None) or 0
    return (result.text or ""), {
        "input_tokens": getattr(usage, "prompt_token_count", None),
        "output_tokens": visible + thoughts,
        "visible_output_tokens": visible,
        "thinking_tokens": thoughts,
        "stop_reason": None,
    }


CALLERS = {"openai": call_openai, "anthropic": call_anthropic, "google": call_google}


def balanced_sample(
    questions: list[dict], gold: dict, size: int, seed: str
) -> list[dict]:
    """Take `size` items while preserving the pool's answer balance exactly.

    Task A's hidden pool is 5,838 items -- five times MMSI-Bench's entire test
    set -- so running every API model over it costs more than it informs. A
    stratified draw keeps the label proportions the pool was constructed to
    have, so the subsample's structural floor equals the pool's. The open
    models run BOTH the full pool and this subsample for free, which shows
    empirically whether the subsample is representative rather than assuming
    it.
    """

    from collections import defaultdict
    strata: dict[str, list[dict]] = defaultdict(list)
    for q in questions:
        strata[str(gold.get(q["id"], "?"))].append(q)
    for rows in strata.values():
        rows.sort(key=lambda r: hashlib.sha256(f"{seed}|{r['id']}".encode()).hexdigest())
    total = len(questions)
    chosen: list[dict] = []
    for key in sorted(strata):
        take = round(size * len(strata[key]) / total)
        chosen.extend(strata[key][:take])
    return sorted(chosen, key=lambda r: r["id"])


def permutations_for(options: list[str], count: int, seed: str) -> list[list[str]]:
    orders = [list(options)]
    rng = random.Random(seed)
    while len(orders) < count:
        shuffled = list(options)
        rng.shuffle(shuffled)
        if shuffled not in orders:
            orders.append(shuffled)
    return orders


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=("A", "B", "C"))
    p.add_argument("--candidate-root", required=True, type=Path)
    p.add_argument("--split", default="hidden_test")
    p.add_argument("--resolved-panel", type=Path, default=Path("work/model_panel/resolved_panel.json"))
    p.add_argument("--audits", type=Path, default=Path("work/candidate_r2/audits"))
    p.add_argument("--out", type=Path, default=Path("work/model_panel/results.jsonl"))
    p.add_argument("--arm", default="main")
    p.add_argument("--models", default="", help="comma-separated names; default = all resolved API models")
    p.add_argument("--tier", default="", help="restrict to one tier, e.g. 2_frontier_api")
    p.add_argument("--shared-subset-only", action="store_true",
                   help="only the 120 items the annotators saw (use this first)")
    p.add_argument("--subset-ids", type=Path, default=None,
                   help=("JSON file with an 'ids' list -- evaluate only those "
                         "items. Use release/configs/taska_api_subset_1200.json "
                         "for Tier 1-3 Task A runs so every paid model is "
                         "scored on the identical frozen draw."))
    p.add_argument("--permutations", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--balanced-sample", type=int, default=0,
                   help="Stratified draw of N items preserving the pool's answer balance.")
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument(
        "--max-consecutive-errors", type=int, default=0,
        help=("Stop after this many consecutive failed items. Zero disables the "
              "guard. Use 1 for quota-limited paid runs so a 429 cannot fill the "
              "result file with failures; recorded failures are retried on resume."),
    )
    p.add_argument("--max-spend-usd", type=float, default=0.0,
                   help=("Hard stop once estimated list-price spend for THIS run "
                         "reaches this figure. 0 disables. Answers already written "
                         "are kept and skipped on resume, so stopping is safe."))
    args = p.parse_args()

    manifest = load_manifest(args.audits, args.task)
    questions = [json.loads(l) for l in
                 (args.candidate_root / f"public/questions/task_{args.task.lower()}_{args.split}.jsonl")
                 .read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.subset_ids:
        keep = set(json.loads(args.subset_ids.read_text(encoding="utf-8"))["ids"])
        missing = keep - {q["id"] for q in questions}
        if missing:
            raise SystemExit(
                f"--subset-ids lists {len(missing)} id(s) absent from this pool; "
                "the frozen subset and the candidate root disagree"
            )
        questions = [q for q in questions if q["id"] in keep]
    if args.shared_subset_only:
        keep = set(manifest["sampled_ids"])
        questions = [q for q in questions if q["id"] in keep]
    if args.balanced_sample:
        gold_path = (args.candidate_root / f"private/task_a_{args.split}_gold.jsonl"
                     if args.task == "A" else
                     args.candidate_root / f"private/gold/task_{args.task.lower()}_{args.split}.jsonl")
        gold_map = {r["id"]: r["answer"] for r in
                    (json.loads(l) for l in gold_path.read_text().splitlines() if l.strip())}
        questions = balanced_sample(questions, gold_map, args.balanced_sample,
                                    seed=f"egoshift-panel-{args.task}")
    if args.limit:
        questions = questions[: args.limit]

    panel = json.loads(args.resolved_panel.read_text(encoding="utf-8"))
    models = [m for m in panel["resolved"] if m["family"] in CALLERS]
    if args.tier:
        models = [m for m in models if m["tier"] == args.tier]
    if args.models:
        wanted = {n.strip() for n in args.models.split(",")}
        models = [m for m in models if m["name"] in wanted]
    if not models:
        raise SystemExit("no API models selected -- check --models/--tier and the resolved panel")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple] = set()
    prior_failures = 0
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("error"):
                    # A recorded failure is not an answer. Counting it as done
                    # turns a transient rate-limit into a permanent hole that a
                    # re-run silently skips -- the same trap that cost us two
                    # local-model retries. Re-attempt it instead.
                    prior_failures += 1
                    continue
                done.add((r["model"], r["arm"], r["item_id"], r["permutation"]))
    print(f"task {args.task} {args.split}: {len(questions)} items x {len(models)} models "
          f"x {args.permutations} permutation(s), arm={args.arm}; {len(done)} already done, "
          f"{prior_failures} prior failure(s) will be retried")

    def price(model_name: str, usage: dict) -> float:
        if not usage:
            return 0.0
        rate = PRICING.get(model_name)
        if rate is None:
            rate = FALLBACK_RATE
            if model_name not in _warned_unpriced:
                _warned_unpriced.add(model_name)
                print(f"  WARNING: no list price for {model_name}; charging the "
                      f"guard at ${rate[0]}/${rate[1]} per Mtok so the cap still "
                      f"binds. Add it to PRICING for an accurate figure.", flush=True)
        i = usage.get("input_tokens") or usage.get("prompt_tokens") or 0
        o = usage.get("output_tokens") or usage.get("completion_tokens") or 0
        return (i * rate[0] + o * rate[1]) / 1_000_000

    # The guard has to be CUMULATIVE per model, not per invocation. Each model
    # is driven once per task, so a per-invocation cap silently allows three
    # times the intended budget. Recovering prior spend from the rows already
    # written makes the cap hold across tasks, across restarts, and after a
    # resume -- with no extra state file to drift.
    prior = collections.defaultdict(float)
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            prior[r["model"]] += price(r["model"], r.get("usage") or {})
    if prior:
        for k, v in sorted(prior.items(), key=lambda x: -x[1])[:5]:
            print(f"  prior spend on {k}: ${v:.2f} (counts against --max-spend-usd)")

    spent = {"usd": 0.0, "calls": 0}

    for model in models:
        caller = CALLERS[model["family"]]
        consecutive_errors = 0
        for question in questions:
            orders = permutations_for(list(question["options"]), args.permutations,
                                      seed=f"{model['name']}|{question['id']}")
            for index, order in enumerate(orders):
                key = (model["name"], args.arm, question["id"], index)
                if key in done:
                    continue
                prompt = build_prompt(question, manifest, arm=args.arm, option_order=order)
                raw, usage, error = None, {}, None
                for attempt in range(args.max_retries):
                    try:
                        raw, usage = caller(model["resolved_id"], prompt,
                                            args.candidate_root, model.get("params") or {})
                        error = None
                        break
                    except Exception as exc:
                        error = f"{type(exc).__name__}: {exc}"
                        if attempt < args.max_retries - 1:
                            time.sleep(min(60, 2 ** attempt * 5))
                choice, how = parse_answer(raw, order) if raw is not None else (None, "error")
                record = {
                    "model": model["name"], "model_id": model["resolved_id"],
                    "family": model["family"], "tier": model["tier"],
                    "task": args.task, "split": args.split, "arm": args.arm,
                    "item_id": question["id"], "permutation": index,
                    "option_order": order, "prompt_sha256": prompt["prompt_sha256"],
                    "raw": raw, "parsed": choice, "parse_mode": how, "error": error,
                    "usage": usage,
                }
                with args.out.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                done.add(key)

                if credit_balance_exhausted(error):
                    print(
                        "\nSTOPPING: provider credit balance is exhausted. The "
                        "failed item is recorded and will be retried after billing "
                        "is replenished; immediate retries are disabled.",
                        flush=True,
                    )
                    return 4

                if error:
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0

                spent["usd"] += price(model["name"], usage or {})
                spent["calls"] += 1
                cumulative = prior[model["name"]] + spent["usd"]
                if spent["calls"] % 100 == 0:
                    print(f"    {model['name']}: {spent['calls']} calls, "
                          f"${spent['usd']:.2f} this run, ${cumulative:.2f} cumulative",
                          flush=True)
                if args.max_spend_usd and cumulative >= args.max_spend_usd:
                    print(f"\nSTOPPING: cumulative spend guard hit for "
                          f"{model['name']}: ${cumulative:.2f} >= "
                          f"${args.max_spend_usd:.2f} (${spent['usd']:.2f} this run, "
                          f"${prior[model['name']]:.2f} before it). Everything "
                          f"answered is written and will be skipped on resume; "
                          f"re-run with a higher --max-spend-usd to continue.",
                          flush=True)
                    return 2
                if (args.max_consecutive_errors
                        and consecutive_errors >= args.max_consecutive_errors):
                    print(
                        f"\nSTOPPING: {model['name']} produced "
                        f"{consecutive_errors} consecutive error(s). The failed "
                        "item is recorded and will be retried on resume. Last "
                        f"error: {error}",
                        flush=True,
                    )
                    return 3
        print(f"  {model['name']}: complete (${spent['usd']:.2f} cumulative)")
    print(f"\nwrote {args.out}  --  {spent['calls']} paid call(s), "
          f"${spent['usd']:.2f} estimated at list price")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
