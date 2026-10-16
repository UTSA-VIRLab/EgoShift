#!/usr/bin/env python3
"""Deadline-oriented Gemini 3 Pro batch runner for the frozen model panel.

The tool deliberately reuses ``egoshift.model_prompting.build_prompt`` and
``parse_answer``.  It never invents a second benchmark prompt or parser.

Workflow:
  prepare  Freeze the exact unfinished IDs and split them into a disjoint
           interactive Task-A reserve and three Batch API request files.
  submit   Upload each non-empty request file and create a Gemini batch job.
  sync     Poll jobs, download completed outputs, and append canonical panel
           records without replacing an existing successful record.
  watch    Repeatedly sync until every submitted job is terminal.
  status   Print the frozen counts and current provider job states.

Credentials are read only from GOOGLE_API_KEY.  Job resource names and local
request metadata are safe to persist; API keys are never written or printed.
"""

from __future__ import annotations

import argparse
import base64
import collections
import datetime as dt
import fcntl
import functools
import glob
import hashlib
import json
import mimetypes
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from egoshift.model_prompting import build_prompt, load_manifest, parse_answer  # noqa: E402

MODEL_NAME = "Gemini 3 Pro"
MODEL_ID = "gemini-3.1-pro-preview"
FAMILY = "google"
TIER = "1_frontier_api"
ARM = "main"
THINKING_LEVEL: str | None = None
SHARED_SUBSET_ONLY = False

TASKS: dict[str, dict[str, Any]] = {
    "A": {
        "root": Path("work/candidate_r2/taska_candidate_v5"),
        "subset": Path("release/configs/taska_api_subset_1200.json"),
        "input_tokens": 2257,
    },
    "B": {
        "root": Path("work/candidate_r2/taskb_candidate_r2_static_v3"),
        "subset": None,
        "input_tokens": 1299,
    },
    "C": {
        "root": Path("work/candidate_r2/taskc_candidate_v10"),
        "subset": Path("release/configs/taskc_api_subset_1200.json"),
        "input_tokens": 2410,
    },
}

DEFAULT_STATE = Path("work/model_panel/gemini_batch")
DEFAULT_OUT = Path("work/model_panel/results_google_pro_batch.jsonl")
TERMINAL = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
    "JOB_STATE_EXPIRED",
}


def configure_variant(
    model_name: str,
    tier: str,
    thinking_level: str | None,
    arm: str = "main",
    shared_subset_only: bool = False,
) -> None:
    """Configure one isolated panel variant before prepare/submit/watch."""

    global MODEL_NAME, TIER, THINKING_LEVEL, ARM, SHARED_SUBSET_ONLY
    MODEL_NAME = model_name
    TIER = tier
    THINKING_LEVEL = thinking_level
    ARM = arm
    SHARED_SUBSET_ONLY = shared_subset_only


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def state_name(value: Any) -> str:
    """Normalize SDK enums across google-genai releases."""
    return str(getattr(value, "value", value))


def result_paths() -> list[Path]:
    return sorted(Path(p) for p in glob.glob("work/model_panel/results*.jsonl"))


def successful_keys(paths: Iterable[Path]) -> set[tuple[str, str, str, str, int]]:
    done: set[tuple[str, str, str, str, int]] = set()
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("error") or row.get("model") != MODEL_NAME:
                continue
            done.add((
                row.get("task", ""), row.get("model", ""),
                row.get("arm", ""), row.get("item_id", ""),
                int(row.get("permutation", 0)),
            ))
    return done


def load_questions(task: str) -> list[dict[str, Any]]:
    cfg = TASKS[task]
    question_path = cfg["root"] / f"public/questions/task_{task.lower()}_hidden_test.jsonl"
    questions = [json.loads(line) for line in question_path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    # Main paid runs use the frozen API subset. Control runs instead use the
    # independently frozen human/control subset, which is not required to be a
    # subset of that main-arm draw.
    if cfg["subset"] and not SHARED_SUBSET_ONLY:
        keep = set(read_json(cfg["subset"])["ids"])
        available = {q["id"] for q in questions}
        missing = keep - available
        if missing:
            raise SystemExit(f"Task {task}: frozen subset contains {len(missing)} absent ID(s)")
        questions = [q for q in questions if q["id"] in keep]
    if SHARED_SUBSET_ONLY:
        shared = set(load_manifest(Path("work/candidate_r2/audits"), task)["sampled_ids"])
        available = {q["id"] for q in questions}
        missing = shared - available
        if missing:
            raise SystemExit(
                f"Task {task}: shared control subset contains {len(missing)} absent ID(s)"
            )
        questions = [q for q in questions if q["id"] in shared]
    return questions


@functools.lru_cache(maxsize=3)
def question_map(task: str) -> dict[str, dict[str, Any]]:
    return {q["id"]: q for q in load_questions(task)}


def freeze_manifest(state_dir: Path, interactive_a: int, force: bool) -> dict[str, Any]:
    manifest_path = state_dir / "frozen_manifest.json"
    if manifest_path.exists() and not force:
        manifest = read_json(manifest_path)
        print(f"using existing frozen manifest: {manifest_path}")
        return manifest

    if (state_dir / "jobs.json").exists():
        raise SystemExit("refusing to replace a manifest after jobs were submitted")

    paths = result_paths()
    done = successful_keys(paths)
    remaining: dict[str, list[str]] = {}
    targets: dict[str, int] = {}
    for task in TASKS:
        questions = load_questions(task)
        targets[task] = len(questions)
        remaining[task] = [q["id"] for q in questions
                           if (task, MODEL_NAME, ARM, q["id"], 0) not in done]

    reserve = remaining["A"][:interactive_a]
    batch = {
        "A": remaining["A"][interactive_a:],
        "B": remaining["B"],
        "C": remaining["C"],
    }
    estimated = sum(len(batch[t]) * int(TASKS[t]["input_tokens"]) for t in TASKS)
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "model": MODEL_NAME,
        "model_id": MODEL_ID,
        "arm": ARM,
        "thinking_level": THINKING_LEVEL,
        "targets": targets,
        "successful_at_freeze": {t: targets[t] - len(remaining[t]) for t in TASKS},
        "remaining_at_freeze": {t: len(remaining[t]) for t in TASKS},
        "interactive": {"A": reserve, "B": [], "C": []},
        "batch": batch,
        "estimated_batch_input_tokens": estimated,
        "source_results": [
            {"path": str(p), "sha256": file_sha256(p)}
            for p in paths
        ],
    }

    # The three sets must be a lossless, disjoint partition of unfinished IDs.
    for task in TASKS:
        lhs = set(manifest["interactive"][task])
        rhs = set(manifest["batch"][task])
        if lhs & rhs or lhs | rhs != set(remaining[task]):
            raise AssertionError(f"Task {task}: invalid interactive/batch partition")

    atomic_json(manifest_path, manifest)
    state_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(state_dir / "interactive_task_a_ids.json", {"ids": reserve})
    print(f"froze {manifest_path}")
    return manifest


def request_and_meta(task: str, item_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = TASKS[task]
    question = question_map(task)[item_id]
    manifest = load_manifest(Path("work/candidate_r2/audits"), task)
    order = list(question["options"])
    prompt = build_prompt(question, manifest, arm=ARM, option_order=order)
    parts: list[dict[str, Any]] = [{"text": prompt["text"]}]
    for rel in prompt["images"]:
        path = cfg["root"] / rel
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        parts.append({
            "inline_data": {
                "mime_type": mime,
                "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        })
    key = f"{task}:{item_id}:0"
    provider_request: dict[str, Any] = {
        "contents": [{"role": "user", "parts": parts}],
    }
    if THINKING_LEVEL:
        # Batch JSONL uses the generateContent REST request schema. Keeping the
        # thinking setting inside every request makes the reasoning variant
        # auditable from the uploaded payload instead of relying on client
        # defaults that can change between Gemini previews.
        provider_request["generationConfig"] = {
            "thinkingConfig": {"thinkingLevel": THINKING_LEVEL.upper()}
        }
    request = {
        "key": key,
        "request": provider_request,
    }
    meta = {
        "key": key,
        "task": task,
        "item_id": item_id,
        "permutation": 0,
        "option_order": order,
        "prompt_sha256": prompt["prompt_sha256"],
    }
    return request, meta


def build_request_files(
    state_dir: Path, manifest: Mapping[str, Any], *, force: bool = False
) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    for task in TASKS:
        ids = list(manifest["batch"][task])
        if not ids:
            continue
        request_path = state_dir / f"task_{task}_requests.jsonl"
        meta_path = state_dir / f"task_{task}_metadata.jsonl"
        if force or not request_path.exists() or not meta_path.exists():
            with request_path.open("w", encoding="utf-8") as requests_handle, \
                    meta_path.open("w", encoding="utf-8") as meta_handle:
                for index, item_id in enumerate(ids, start=1):
                    request, meta = request_and_meta(task, item_id)
                    requests_handle.write(json.dumps(request, ensure_ascii=False) + "\n")
                    meta_handle.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    if index % 100 == 0:
                        print(f"  Task {task}: encoded {index}/{len(ids)}", flush=True)
        request_count = sum(1 for line in request_path.open(encoding="utf-8") if line.strip())
        meta_count = sum(1 for line in meta_path.open(encoding="utf-8") if line.strip())
        if request_count != len(ids) or meta_count != len(ids):
            raise SystemExit(f"Task {task}: request/metadata count does not match frozen manifest")
        shards.append({
            "task": task,
            "count": len(ids),
            "request_path": str(request_path),
            "metadata_path": str(meta_path),
            "bytes": request_path.stat().st_size,
            "sha256": file_sha256(request_path),
        })
    atomic_json(state_dir / "request_files.json", shards)
    return shards


def google_client():
    if not os.environ.get("GOOGLE_API_KEY"):
        raise SystemExit("GOOGLE_API_KEY is not set")
    from google import genai
    return genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


def submit(state_dir: Path) -> None:
    manifest = read_json(state_dir / "frozen_manifest.json")
    shards = build_request_files(state_dir, manifest)
    jobs_path = state_dir / "jobs.json"
    jobs = read_json(jobs_path) if jobs_path.exists() else []
    submitted_tasks = {j["task"] for j in jobs}
    client = google_client()
    from google.genai import types

    for shard in shards:
        if shard["task"] in submitted_tasks:
            print(f"Task {shard['task']}: already submitted; skipping")
            continue
        path = Path(shard["request_path"])
        print(f"Task {shard['task']}: uploading {path.stat().st_size / 2**20:.1f} MiB", flush=True)
        uploaded = client.files.upload(
            file=path,
            config=types.UploadFileConfig(
                display_name=f"egoshift-gemini-pro-task-{shard['task']}",
                mime_type="application/jsonl",
            ),
        )
        print(f"Task {shard['task']}: creating batch job", flush=True)
        try:
            job = client.batches.create(
                model=MODEL_ID,
                src=uploaded.name,
                config=types.CreateBatchJobConfig(
                    display_name=f"egoshift-wacv27-gemini-pro-task-{shard['task']}"
                ),
            )
        except Exception:
            # A failed create otherwise leaves an uploaded unpublished payload
            # behind even though no batch job can consume it.
            try:
                client.files.delete(name=uploaded.name)
                print(f"Task {shard['task']}: removed unreferenced upload", flush=True)
            except Exception as cleanup_exc:
                print(
                    f"Task {shard['task']}: WARNING could not remove unreferenced upload: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}",
                    flush=True,
                )
            raise
        row = {
            **shard,
            "input_file": uploaded.name,
            "job_name": job.name,
            "state": state_name(job.state),
            "submitted_at": utc_now(),
            "imported": False,
        }
        jobs.append(row)
        atomic_json(jobs_path, jobs)
        print(f"Task {shard['task']}: submitted {job.name} ({job.state})", flush=True)


def load_metadata(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def output_text(response: Mapping[str, Any]) -> str:
    texts: list[str] = []
    for candidate in response.get("candidates") or []:
        for part in ((candidate.get("content") or {}).get("parts") or []):
            if part.get("text") is not None and not part.get("thought", False):
                texts.append(str(part["text"]))
    return "".join(texts)


def usage_dict(response: Mapping[str, Any]) -> dict[str, Any]:
    usage = response.get("usageMetadata") or response.get("usage_metadata") or {}
    prompt = usage.get("promptTokenCount", usage.get("prompt_token_count"))
    visible = usage.get("candidatesTokenCount",
                        usage.get("responseTokenCount", usage.get("response_token_count", 0))) or 0
    thoughts = usage.get("thoughtsTokenCount", usage.get("thoughts_token_count", 0)) or 0
    return {
        "input_tokens": prompt,
        "output_tokens": visible + thoughts,
        "visible_output_tokens": visible,
        "thinking_tokens": thoughts,
        "stop_reason": None,
    }


def completed_rows(payload: bytes, metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lines = [json.loads(line) for line in payload.decode("utf-8").splitlines() if line.strip()]
    by_key = {m["key"]: m for m in metadata}
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(lines):
        key = item.get("key")
        meta = by_key.get(key) if key else (metadata[index] if index < len(metadata) else None)
        if meta is None:
            raise ValueError(f"batch response {index} has no matching request metadata")
        response = item.get("response") or {}
        error_obj = item.get("error")
        error = json.dumps(error_obj, ensure_ascii=False) if error_obj else None
        raw = output_text(response) if response else None
        parsed, mode = parse_answer(raw, meta["option_order"]) if raw is not None else (None, "error")
        rows.append({
            "model": MODEL_NAME,
            "model_id": MODEL_ID,
            "family": FAMILY,
            "tier": TIER,
            "task": meta["task"],
            "split": "hidden_test",
            "arm": ARM,
            "item_id": meta["item_id"],
            "permutation": meta["permutation"],
            "option_order": meta["option_order"],
            "prompt_sha256": meta["prompt_sha256"],
            "raw": raw,
            "parsed": parsed,
            "parse_mode": mode,
            "error": error,
            "usage": usage_dict(response),
            "execution": "gemini_batch_api",
        })
    if len(rows) != len(metadata):
        raise ValueError(f"batch returned {len(rows)} rows for {len(metadata)} requests")
    return rows


def append_new_rows(out: Path, rows: list[dict[str, Any]]) -> tuple[int, int]:
    out.parent.mkdir(parents=True, exist_ok=True)
    added = skipped = 0
    with out.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.seek(0)
        done = set()
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get("error"):
                done.add((row.get("task"), row.get("model"), row.get("arm"),
                          row.get("item_id"), int(row.get("permutation", 0))))
        handle.seek(0, os.SEEK_END)
        for row in rows:
            key = (row["task"], row["model"], row["arm"], row["item_id"], row["permutation"])
            if key in done:
                skipped += 1
                continue
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if not row.get("error"):
                done.add(key)
            added += 1
        handle.flush()
        os.fsync(handle.fileno())
    return added, skipped


def cleanup_remote_files(client: Any, row: dict[str, Any]) -> None:
    """Delete provider-side inputs/outputs only after durable local import."""
    if row.get("remote_cleanup_complete"):
        return
    names = [row.get("input_file"), row.get("output_file")]
    deleted: list[str] = []
    errors: list[str] = []
    for name in dict.fromkeys(n for n in names if n):
        try:
            client.files.delete(name=name)
            deleted.append(name)
        except Exception as exc:  # Preserve imported results even if cleanup is transient.
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    row["remote_files_deleted"] = deleted
    row["remote_cleanup_checked_at"] = utc_now()
    row["remote_cleanup_errors"] = errors
    row["remote_cleanup_complete"] = not errors


def sync(state_dir: Path, out: Path, online: bool = True) -> tuple[list[dict[str, Any]], bool]:
    jobs_path = state_dir / "jobs.json"
    if not jobs_path.exists():
        raise SystemExit("no jobs.json; run submit first")
    jobs = read_json(jobs_path)
    client = google_client() if online else None
    all_terminal = True
    for row in jobs:
        if row.get("imported"):
            cleanup_remote_files(client, row)
            continue
        job = client.batches.get(name=row["job_name"])
        state = state_name(job.state)
        row["state"] = state
        row["checked_at"] = utc_now()
        print(f"Task {row['task']}: {state}", flush=True)
        if state not in TERMINAL:
            all_terminal = False
            continue
        if state != "JOB_STATE_SUCCEEDED":
            row["provider_error"] = str(job.error)
            continue
        destination = job.dest
        if destination is None or not destination.file_name:
            raise RuntimeError(f"{row['job_name']} succeeded without an output file")
        payload = client.files.download(file=destination.file_name)
        local_response = state_dir / f"task_{row['task']}_responses.jsonl"
        local_response.write_bytes(payload)
        metadata = load_metadata(Path(row["metadata_path"]))
        records = completed_rows(payload, metadata)
        added, skipped = append_new_rows(out, records)
        row.update({
            "output_file": destination.file_name,
            "local_response": str(local_response),
            "imported": True,
            "imported_at": utc_now(),
            "records_added": added,
            "records_skipped": skipped,
        })
        cleanup_remote_files(client, row)
        print(f"Task {row['task']}: imported {added}, skipped {skipped}", flush=True)
    atomic_json(jobs_path, jobs)
    return jobs, all_terminal


def print_status(state_dir: Path, online: bool) -> None:
    manifest_path = state_dir / "frozen_manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        print("frozen remaining:", manifest["remaining_at_freeze"])
        print("interactive reserve:", {t: len(v) for t, v in manifest["interactive"].items()})
        print("batch requests:", {t: len(v) for t, v in manifest["batch"].items()})
        print("estimated batch input tokens:", f"{manifest['estimated_batch_input_tokens']:,}")
    jobs_path = state_dir / "jobs.json"
    if not jobs_path.exists():
        print("no jobs submitted")
        return
    if online:
        sync(state_dir, DEFAULT_OUT)
    else:
        for job in read_json(jobs_path):
            print(job["task"], job.get("state"), job["job_name"],
                  "imported" if job.get("imported") else "not-imported")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "submit", "sync", "watch", "status"))
    parser.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--interactive-a", type=int, default=400)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--online", action="store_true", help="query provider during status")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--model-name", default="Gemini 3 Pro")
    parser.add_argument("--tier", default="1_frontier_api")
    parser.add_argument("--thinking-level", default="",
                        help="Gemini thinking level embedded in every batch request, e.g. high")
    parser.add_argument("--arm", default="main",
                        help="Canonical prompt arm embedded in every batch request")
    parser.add_argument("--shared-subset-only", action="store_true",
                        help="evaluate only the frozen 120-item human/control subset per task")
    args = parser.parse_args()

    configure_variant(
        args.model_name,
        args.tier,
        args.thinking_level or None,
        arm=args.arm,
        shared_subset_only=args.shared_subset_only,
    )

    os.chdir(ROOT.parent)
    if args.command == "prepare":
        manifest = freeze_manifest(args.state_dir, args.interactive_a, args.force)
        build_request_files(args.state_dir, manifest, force=args.force)
        print_status(args.state_dir, online=False)
    elif args.command == "submit":
        submit(args.state_dir)
        print_status(args.state_dir, online=False)
    elif args.command == "sync":
        sync(args.state_dir, args.out)
    elif args.command == "watch":
        while True:
            try:
                _, terminal = sync(args.state_dir, args.out)
                if terminal:
                    return 0
            except Exception as exc:
                # A deadline monitor must survive transient DNS, TLS and 5xx
                # failures. Deterministic import/schema failures will remain
                # visible in the log on every retry instead of silently
                # terminating the only watcher.
                print(
                    f"WATCH ERROR {type(exc).__name__}: {exc}; retrying",
                    flush=True,
                )
            time.sleep(max(10, args.poll_seconds))
    else:
        print_status(args.state_dir, online=args.online)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
