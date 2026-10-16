#!/usr/bin/env python3
"""Run open-weight models locally, writing the same records as the API runner.

Prompts come from egoshift.model_prompting, identical to the API path, so an
open model and a frontier model answer the same text. Records share the schema
in run_model_panel.py so one scorer handles both.

One model is loaded at a time and freed before the next, because the panel
spans 2B to 78B and holding several would thrash. Progress appends per item,
so an interrupted run resumes.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from egoshift import gpu_memory  # noqa: E402
from egoshift.model_prompting import build_prompt, load_manifest, parse_answer  # noqa: E402

# Must happen before anything imports torch, which every path below does
# lazily. Fragmentation was a named cause in every OOM traceback we collected.
gpu_memory.set_alloc_conf()


DEFAULT_MAX_NEW_TOKENS = 48


def option_order_for(options: list[str], permutation_index: int, seed: str) -> list[str]:
    """Return a deterministic option order; index zero is the identity order."""

    if permutation_index < 0:
        raise ValueError("permutation_index must be non-negative")
    orders = [list(options)]
    rng = random.Random(seed)
    while len(orders) <= permutation_index:
        shuffled = list(options)
        rng.shuffle(shuffled)
        if shuffled not in orders:
            orders.append(shuffled)
    return orders[permutation_index]


def generation_limit(model_spec: dict, override: int = 0) -> int:
    """Return the audited generation allowance for one model.

    Most instruction-tuned panel members answer with one short option, so the
    historical 48-token default is ample.  Visual reasoning checkpoints such
    as QVQ deliberately emit a chain of thought before the requested answer;
    their official allowance belongs in the frozen panel specification rather
    than being silently clipped to the generic default.  A CLI override is
    retained for controlled probes.
    """

    value = override or int(model_spec.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS))
    if value <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {value}")
    return value


def ended_by_length(trimmed, max_new_tokens: int, eos_token_id) -> bool:
    """Whether generation exhausted its allowance without emitting EOS."""

    if len(trimmed) < max_new_tokens:
        return False
    eos_ids = eos_token_id if isinstance(eos_token_id, (list, tuple, set)) else [eos_token_id]
    eos_ids = {int(token) for token in eos_ids if token is not None}
    return not eos_ids or int(trimmed[-1]) not in eos_ids


def load_model(repo: str, device: str,
               reserve_gib: float = gpu_memory.DEFAULT_RESERVE_GIB,
               revision: str | None = None):
    """Load one model. `--device auto` shards across all visible GPUs.

    A 32B-78B model in bf16 needs 65-160 GB and cannot sit on one 46 GB card,
    so the Tier-4 models require device_map="auto"; the small cached models
    stay pinned to a single GPU so several can run in parallel.

    Under "auto" the per-card cap matters as much as the sharding: left
    uncapped, accelerate packs weights into every card up to its full
    capacity, the load reports success, and then every generate() OOMs
    because activations have nowhere to go.
    """

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    kwargs = {}
    if device == "auto":
        kwargs["max_memory"] = gpu_memory.max_memory_map(reserve_gib)

    # Some checkpoints -- the InternVL-chat family, which includes the
    # SpaceMind spatial specialist -- expose AutoModel/AutoModelForCausalLM in
    # their auto_map and no AutoModelForImageTextToText entry. Asking for the
    # image-text-to-text class raises "Unrecognized configuration class" and
    # looks like a dead checkpoint when the weights are perfectly good.
    from transformers import AutoConfig, AutoModel
    model_type = ""
    try:
        cfg = AutoConfig.from_pretrained(
            repo, trust_remote_code=True, revision=revision)
        model_type = getattr(cfg, "model_type", "")
        mapped = set((getattr(cfg, "auto_map", None) or {}))
        internvl_chat = (model_type == "internvl_chat"
                         or ("AutoModelForImageTextToText" not in mapped and mapped))
    except Exception:
        internvl_chat = False

    processor_kwargs = {"trust_remote_code": True, "revision": revision}
    if model_type == "molmo2":
        # Freeze the official slow image processor. Its model card warns that
        # a future default switch to the fast implementation changes outputs.
        processor_kwargs["use_fast"] = False
    processor = AutoProcessor.from_pretrained(repo, **processor_kwargs)
    loader = AutoModel if internvl_chat else AutoModelForImageTextToText
    model = loader.from_pretrained(
        repo, dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
        revision=revision, low_cpu_mem_usage=True, **kwargs,
    ).eval()
    return processor, model


def attempt_generate(processor, model, prompt, root: Path, retries: int,
                     max_new_tokens: int):
    """(raw, error, was_oom, truncated). Retry OOM once after dropping cache.

    A transient OOM -- one long prompt landing while a neighbour job spikes --
    usually clears after empty_cache(). A persistent one will not, which is
    what the consecutive-OOM abort in the caller is for.
    """

    import torch

    error, exc = None, None
    for _ in range(max(1, retries + 1)):
        try:
            raw, truncated = generate(
                processor, model, prompt, root, max_new_tokens=max_new_tokens)
            return raw, None, False, truncated
        except Exception as caught:            # noqa: BLE001 - recorded, not raised
            exc = caught
            error = f"{type(caught).__name__}: {caught}"
            if not gpu_memory.is_oom(caught):
                return None, error, False, False
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    return None, error, gpu_memory.is_oom(exc) if exc else False, False


def generate(processor, model, prompt, root: Path,
             max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> tuple[str, bool]:
    import torch
    from PIL import Image

    images = [Image.open(root / rel).convert("RGB") for rel in prompt["images"]]
    content = [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt["text"]}]
    chat = [{"role": "user", "content": content}]
    if getattr(model.config, "model_type", "") == "molmo2":
        # Molmo2's official multi-image interface consumes the PIL objects
        # inside the structured chat and performs tokenisation in
        # apply_chat_template.  Sending placeholders plus a separate `images`
        # argument happens to resemble the Qwen path but is not the interface
        # Ai2 documents, and can lose the image-to-token correspondence.
        molmo_content = ([{"type": "image", "image": image} for image in images]
                         + [{"type": "text", "text": prompt["text"]}])
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": molmo_content}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
    else:
        text = processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], images=images or None, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[0][inputs["input_ids"].shape[1]:]
    truncated = ended_by_length(
        trimmed, max_new_tokens, getattr(model.generation_config, "eos_token_id", None))
    return processor.decode(trimmed, skip_special_tokens=True).strip(), truncated


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=("A", "B", "C"))
    p.add_argument("--candidate-root", required=True, type=Path)
    p.add_argument("--split", default="hidden_test")
    p.add_argument("--panel", type=Path, default=Path("release/configs/model_panel_r2.json"))
    p.add_argument("--audits", type=Path, default=Path("work/candidate_r2/audits"))
    p.add_argument("--out", type=Path, default=Path("work/model_panel/results_local.jsonl"))
    p.add_argument("--arm", default="main")
    p.add_argument("--models", default="", help="comma-separated names; default = cached tier-5")
    p.add_argument("--device", default="cuda:0",
                   help="cuda:N to pin one GPU, or 'auto' to shard a large model across all of them.")
    p.add_argument("--shared-subset-only", action="store_true")
    p.add_argument("--subset-ids", type=Path, default=None,
                   help=("JSON file with an 'ids' list -- evaluate only those "
                         "items. Use release/configs/taska_api_subset_1200.json "
                         "for Tier 1-3 Task A runs so every paid model is "
                         "scored on the identical frozen draw."))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-new-tokens", type=int, default=0,
                   help=("Override the frozen per-model generation allowance. "
                         "0 uses model_panel_r2.json, falling back to 48."))
    p.add_argument("--permutation-index", type=int, default=0,
                   help=("Deterministic option-order permutation to run. Zero is the "
                         "identity order; use 1 for the shuffled_options control."))
    p.add_argument("--gpu-reserve-gib", type=float, default=gpu_memory.DEFAULT_RESERVE_GIB,
                   help=("VRAM held back per card for activations and KV cache. "
                         "Raise it if generation OOMs on a model that loaded."))
    p.add_argument("--max-consecutive-oom", type=int, default=10,
                   help=("Give up on a model after this many OOMs in a row. "
                         "Without it a model whose weights fit but whose "
                         "activations do not burns the entire pool one "
                         "failed item at a time."))
    p.add_argument("--oom-retries", type=int, default=1,
                   help="Retries per item after an OOM, each preceded by empty_cache().")
    p.add_argument("--force-load", action="store_true",
                   help="Load even when the VRAM preflight says it will not fit.")
    args = p.parse_args()

    manifest = load_manifest(args.audits, args.task)
    qpath = args.candidate_root / f"public/questions/task_{args.task.lower()}_{args.split}.jsonl"
    questions = [json.loads(l) for l in qpath.read_text(encoding="utf-8").splitlines() if l.strip()]
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
    if args.limit:
        questions = questions[: args.limit]

    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    models = [m for m in panel["models"] if m["family"] == "local" and m.get("cached")]
    if args.models:
        wanted = {n.strip() for n in args.models.split(",")}
        models = [m for m in panel["models"] if m["name"] in wanted]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    failed = 0
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("error"):
                    # A recorded failure is not a result. Counting one as done
                    # is why retrying an OOM'd model used to report "already
                    # complete" and skip every item it had failed.
                    failed += 1
                    continue
                done.add((r["model"], r["arm"], r["item_id"], r["task"], r["split"],
                          int(r.get("permutation", 0))))

    print(f"task {args.task} {args.split}: {len(questions)} items x {len(models)} local models "
          f"(arm={args.arm}) on {args.device}; {len(done)} already done, "
          f"{failed} prior failure(s) ignored", flush=True)
    print(f"  VRAM {gpu_memory.summary(args.gpu_reserve_gib)}", flush=True)

    for model_spec in models:
        repo = model_spec["id_candidates"][0]
        revision = model_spec.get("revision")
        max_new_tokens = generation_limit(model_spec, args.max_new_tokens)
        pending = [q for q in questions
                   if (model_spec["name"], args.arm, q["id"], args.task, args.split,
                       args.permutation_index) not in done]
        if not pending:
            print(f"  {model_spec['name']}: already complete", flush=True)
            continue
        ok, why = gpu_memory.fits(repo, args.gpu_reserve_gib)
        if not ok and not args.force_load:
            print(f"  {model_spec['name']}: SKIPPED -- {why}", flush=True)
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "model": model_spec["name"], "model_id": repo,
                    "model_revision": revision, "family": "local",
                    "tier": model_spec["tier"], "task": args.task, "split": args.split,
                    "arm": args.arm, "item_id": None, "permutation": 0,
                    "raw": None, "parsed": None, "parse_mode": "error",
                    "error": f"preflight: {why}"}) + "\n")
            continue

        print(f"  {model_spec['name']}: loading {repo} ({len(pending)} items left; {why}; "
              f"max_new_tokens={max_new_tokens})",
              flush=True)
        started = time.time()
        try:
            processor, model = load_model(
                repo, args.device, args.gpu_reserve_gib, revision=revision)
        except Exception as exc:
            print(f"    LOAD FAILED: {type(exc).__name__}: {exc}", flush=True)
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "model": model_spec["name"], "model_id": repo,
                    "model_revision": revision, "family": "local",
                    "tier": model_spec["tier"], "task": args.task, "split": args.split,
                    "arm": args.arm, "item_id": None, "permutation": 0,
                    "raw": None, "parsed": None, "parse_mode": "error",
                    "error": f"load failed: {type(exc).__name__}: {exc}"}) + "\n")
            continue
        consecutive_oom, oom_total, aborted = 0, 0, False
        for index, question in enumerate(pending, start=1):
            option_order = option_order_for(
                list(question["options"]), args.permutation_index,
                seed=f"{model_spec['name']}|{question['id']}")
            prompt = build_prompt(
                question, manifest, arm=args.arm, option_order=option_order)
            raw, error, was_oom, truncated = attempt_generate(
                processor, model, prompt, args.candidate_root, args.oom_retries,
                max_new_tokens)
            if was_oom:
                consecutive_oom += 1
                oom_total += 1
            else:
                consecutive_oom = 0
            if truncated:
                # Never score a phrase mentioned inside an unfinished chain of
                # thought.  This was the root cause of QVQ's spurious 0.6%.
                choice, how = None, "truncated"
            else:
                choice, how = (parse_answer(raw, prompt["options"])
                               if raw is not None else (None, "error"))
            with args.out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "model": model_spec["name"], "model_id": repo,
                    "model_revision": revision, "family": "local",
                    "tier": model_spec["tier"], "task": args.task, "split": args.split,
                    "arm": args.arm, "item_id": question["id"],
                    "permutation": args.permutation_index,
                    "option_order": prompt["options"], "prompt_sha256": prompt["prompt_sha256"],
                    "raw": raw, "parsed": choice, "parse_mode": how, "error": error,
                    "finish_reason": "length" if truncated else ("error" if error else "stop"),
                    "generation_max_new_tokens": max_new_tokens},
                    ensure_ascii=False) + "\n")
            if index % 100 == 0:
                rate = index / (time.time() - started)
                note = f", {oom_total} OOM" if oom_total else ""
                print(f"    {index}/{len(pending)} ({rate:.1f} items/s{note})", flush=True)
            if consecutive_oom >= args.max_consecutive_oom:
                print(f"    ABORTING {model_spec['name']}: {consecutive_oom} consecutive "
                      f"CUDA OOMs at item {index}/{len(pending)}. "
                      f"{gpu_memory.summary(args.gpu_reserve_gib)}. "
                      f"Re-run when the GPUs are free, or raise --gpu-reserve-gib.",
                      flush=True)
                aborted = True
                break
        del model, processor
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        status = "ABORTED after" if aborted else "done in"
        tail = f" ({oom_total} OOM)" if oom_total else ""
        print(f"  {model_spec['name']}: {status} {(time.time()-started)/60:.1f} min{tail}",
              flush=True)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
