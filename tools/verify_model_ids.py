#!/usr/bin/env python3
"""Resolve every panel entry against the live provider APIs before the run.

Model ids move faster than any config file. This asks each provider what it
actually serves today, matches the panel's candidate ids against that list,
and writes a resolved panel plus an access-failure record for anything it
cannot find. The panel gate requires a documented failure per unavailable
model, never a silent omission.

Costs nothing: it lists models, it does not run inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def list_openai() -> list[str]:
    from openai import OpenAI
    return sorted(m.id for m in OpenAI().models.list().data)


def list_anthropic() -> list[str]:
    import anthropic
    return sorted(m.id for m in anthropic.Anthropic().models.list(limit=1000).data)


def list_google() -> list[str]:
    from google import genai
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
    return sorted(m.name.split("/")[-1] for m in client.models.list())


LISTERS = {"openai": list_openai, "anthropic": list_anthropic, "google": list_google}



def probe_call(family: str, model_id: str) -> None:
    """One minimal text-only call. Raises if the model cannot be invoked."""

    if family == "openai":
        from openai import OpenAI
        # The GPT-5 family rejects max_tokens and wants max_completion_tokens.
        # Send neither: the probe only needs to learn whether the id answers.
        OpenAI().chat.completions.create(
            model=model_id, messages=[{"role": "user", "content": "hi"}])
    elif family == "anthropic":
        import anthropic
        anthropic.Anthropic().messages.create(
            model=model_id, max_tokens=1, messages=[{"role": "user", "content": "hi"}])
    elif family == "google":
        import os
        from google import genai
        # Bind the client: an inline genai.Client(...) can be collected while
        # the request is still in flight, which surfaces as the misleading
        # "client has been closed" rather than a real access error.
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        client.models.generate_content(model=model_id, contents="hi")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", type=Path, default=Path("release/configs/model_panel_r2.json"))
    parser.add_argument("--out", type=Path, default=Path("work/model_panel/resolved_panel.json"))
    parser.add_argument("--probe", action="store_true",
                        help="Send one minimal call per API model to prove it actually answers.")
    args = parser.parse_args()

    panel = json.loads(args.panel.read_text(encoding="utf-8"))
    available: dict[str, list[str]] = {}
    provider_errors: dict[str, str] = {}
    for family, lister in LISTERS.items():
        try:
            available[family] = lister()
            print(f"{family}: {len(available[family])} models visible")
        except Exception as exc:
            provider_errors[family] = f"{type(exc).__name__}: {exc}"
            print(f"{family}: UNAVAILABLE -- {provider_errors[family]}")

    resolved, failures = [], []
    for model in panel["models"]:
        family = model["family"]
        if family == "local":
            resolved.append({**model, "resolved_id": (model["id_candidates"] or [None])[0]})
            continue
        if family not in available:
            failures.append({**model, "reason": f"provider unavailable: {provider_errors.get(family)}"})
            continue
        hit = next((c for c in model["id_candidates"] if c in available[family]), None)
        if hit is None:
            near = [a for a in available[family]
                    if any(c.split("-")[0] in a for c in model["id_candidates"])][:8]
            failures.append({**model, "reason": "no candidate id served today",
                             "closest_available": near})
            continue
        resolved.append({**model, "resolved_id": hit})

    # Listing is not calling. gemini-2.5-pro appears in models.list() and then
    # returns 404 "no longer available to new users" on the first real request
    # -- a failure that would otherwise surface part-way through a paid run.
    # One tiny text-only call per API model settles it for a fraction of a cent.
    if args.probe:
        still, newly = [], []
        for m in resolved:
            if m["family"] == "local":
                still.append(m); continue
            try:
                probe_call(m["family"], m["resolved_id"])
                still.append(m)
            except Exception as exc:
                reason = f"listed but not callable: {type(exc).__name__}: {exc}"
                print(f"  NOT CALLABLE {m['name']} ({m['resolved_id']}): {str(exc)[:90]}")
                newly.append({**m, "reason": reason})
        if newly:
            resolved, failures = still, failures + newly
            print(f"  probe removed {len(newly)} model(s) that listed but do not answer")
        else:
            print(f"  probe: every resolved API model answered")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"panel_id": panel["panel_id"], "resolved": resolved,
         "access_failures": failures, "provider_errors": provider_errors,
         "available_counts": {k: len(v) for k, v in available.items()}},
        indent=1, sort_keys=True) + "\n")

    print(f"\nresolved {len(resolved)} / {len(panel['models'])}")
    for f in failures:
        print(f"  UNRESOLVED {f['name']} ({f['family']}): {f['reason']}")
        if f.get("closest_available"):
            print(f"     closest served ids: {f['closest_available']}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
