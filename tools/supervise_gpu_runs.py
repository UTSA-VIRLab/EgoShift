#!/usr/bin/env python3
"""Keep the GPUs busy and unstick them when they jam.

Three failure modes have actually occurred on this project, and this watches
for all of them:

  * a hung AI2-THOR controller -- the process stays alive, holds a GPU, and
    never advances (twice: Task-C shards 12 and 5, 5h and 7h lost);
  * a silent model load failure -- the runner records the error as a row and
    exits 0, so a driver reports success while nothing ran (Tier-4 InternVL3);
  * an idle fleet -- a job finishes and the GPUs sit empty until someone looks.

Every cycle it verifies what finished, requeues or skips what is stuck, and
starts the next queued job the moment the machine is free. It never launches
past `max_concurrent_jobs`, so it cannot pile work onto a busy card.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Derived, never hard-coded: an absolute path under a real home directory
# names the author, and this file ships in the anonymous review artifact.
ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
RESULTS = ROOT / "work/model_panel/results_local.jsonl"
QUEUE = ROOT / "work/model_panel/job_queue.json"
LOGDIR = ROOT / "work/candidate_r2/mining_logs"
STATE = ROOT / "work/model_panel/supervisor_state.json"


def log(message: str) -> None:
    line = f"{datetime.now():%m-%d %H:%M:%S} {message}"
    print(line, flush=True)


def running(pattern: str) -> list[int]:
    out = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True).stdout
    return [int(p) for p in out.split()]


def completed_counts() -> dict:
    """(model, task, arm) -> distinct items answered, from the results file."""

    counts: dict = {}
    if not RESULTS.exists():
        return counts
    for line in RESULTS.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("item_id") is None:
            continue
        counts.setdefault((r["model"], r["task"], r["arm"]), set()).add(r["item_id"])
    return {k: len(v) for k, v in counts.items()}


def load_failures() -> list[tuple[str, str]]:
    out = []
    if not RESULTS.exists():
        return out
    for line in RESULTS.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("item_id") is None and r.get("error"):
            out.append((r.get("model", "?"), r.get("error", "")[:120]))
    return out


def error_storm(tail_rows: int = 200, threshold: float = 0.9,
                minimum: int = 50) -> list[tuple[str, str, int, int]]:
    """Models whose recent rows are almost all failures.

    The fourth failure mode, and the most expensive one observed: the weights
    load, the process stays healthy, exit code is 0, and every single item
    OOMs. InternVL3-78B produced 1,098 such rows over 69 minutes and reported
    success. load_failures() cannot see it because those rows carry a real
    item_id. The runner now aborts itself after N consecutive OOMs; this stays
    as the backstop for any failure that is frequent but not consecutive.
    """

    if not RESULTS.exists():
        return []
    recent: dict[tuple[str, str], list[int]] = {}
    lines = RESULTS.read_text(errors="replace").splitlines()[-tail_rows * 8:]
    for line in lines:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("item_id") is None:
            continue
        bucket = recent.setdefault((r.get("model", "?"), r.get("task", "?")), [])
        bucket.append(1 if r.get("error") else 0)
    out = []
    for (model, task), flags in recent.items():
        window = flags[-tail_rows:]
        if len(window) >= minimum and sum(window) / len(window) >= threshold:
            out.append((model, task, sum(window), len(window)))
    return out


def gpu_headroom() -> str:
    try:
        from egoshift import gpu_memory
        return gpu_memory.summary()
    except Exception:
        return "unknown"


def job_fits(job: dict) -> tuple[bool, str]:
    """Whether a queued job's largest model can load right now.

    max_concurrent_jobs counts processes, not gigabytes, so on its own it will
    happily start a 156 GB model beside a 137 GB one. Every OOM on 2026-08-25
    came from exactly that.
    """

    try:
        from egoshift import gpu_memory
        panel = json.loads((ROOT / "release/configs/model_panel_r2.json")
                           .read_text(encoding="utf-8"))
        by_name = {m["name"]: m for m in panel["models"]}
        # Models in a job run one at a time and are freed between, so the
        # binding constraint is the largest one, not their sum.
        biggest, biggest_repo, biggest_bytes = None, None, -1
        for name in (n.strip() for n in job.get("models", "").split(",") if n.strip()):
            spec = by_name.get(name)
            if not spec or not spec.get("id_candidates"):
                continue
            repo = spec["id_candidates"][0]
            size = gpu_memory.estimate_weight_bytes(repo) or 0
            if size > biggest_bytes:
                biggest, biggest_repo, biggest_bytes = name, repo, size
        if biggest_repo is None:
            return True, "no cached size for any model in job"
        # Check against the device the job will actually use, not the machine
        # total -- a pinned job cannot borrow another card's free memory.
        ok, why = gpu_memory.fits(biggest_repo, device=job.get("device", "auto"))
        return ok, f"largest is {biggest}: {why}"
    except Exception as exc:                    # never block on a checker bug
        return True, f"vram check unavailable ({type(exc).__name__})"


def stale_logs(minutes: int) -> list[Path]:
    """Logs of live jobs whose output stopped moving -- the hang signature."""

    # Only logs belonging to model-panel jobs. The "taskc_*.log" glob used to
    # be here and it was actively dangerous: a FINISHED mining log stops being
    # written, so it looks stale forever, and the owner lookup below then
    # matched any process whose command line merely mentioned that log's stem.
    # A bake reading taskc_source_v7.jsonl was killed twice that way. Mining
    # and bake jobs are run by hand; the supervisor has no business policing
    # them.
    cutoff = time.time() - minutes * 60
    stuck = []
    for path in list(LOGDIR.glob("localpanel_*.log")) + list(LOGDIR.glob("tier4_*.log")) \
            + list(LOGDIR.glob("controlarm_*.log")) + list(LOGDIR.glob("queued_*.log")):
        try:
            if path.stat().st_mtime < cutoff:
                stuck.append(path)
        except OSError:
            pass
    return stuck


def launch(job: dict) -> None:
    tag = job.get("tag") or job["models"].split(",")[0]
    logfile = LOGDIR / f"queued_{tag}_task{job['task']}_{job.get('arm','main')}.log"
    cmd = [PY, "release/tools/run_local_models.py",
           "--task", job["task"], "--candidate-root", job["candidate_root"],
           "--models", job["models"], "--device", job.get("device", "auto"),
           "--arm", job.get("arm", "main"),
           "--out", "work/model_panel/results_local.jsonl"]
    if job.get("shared_subset_only"):
        cmd.append("--shared-subset-only")
    if job.get("subset_ids"):
        cmd += ["--subset-ids", job["subset_ids"]]
    env = dict(os.environ, PYTHONPATH="release/src")
    with logfile.open("a") as handle:
        subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=handle,
                         start_new_session=True, stdin=subprocess.DEVNULL)
    log(f"LAUNCHED {tag} task {job['task']} arm={job.get('arm','main')} -> {logfile.name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycle-seconds", type=int, default=300)
    ap.add_argument("--stale-minutes", type=int, default=30)
    ap.add_argument("--max-concurrent-jobs", type=int, default=2)
    args = ap.parse_args()

    log(f"supervisor up: cycle={args.cycle_seconds}s stale={args.stale_minutes}min "
        f"max_jobs={args.max_concurrent_jobs}")
    seen_failures: set = set()
    seen_storms: set = set()
    held: set = set()
    last_report = 0.0

    while True:
        try:
            live = running("run_local_models.py")
            tier4 = running("run_tier4.sh")
            queue = json.loads(QUEUE.read_text()) if QUEUE.exists() else []

            # 1) surface any NEW silent load failure
            for model, err in load_failures():
                if (model, err) not in seen_failures:
                    seen_failures.add((model, err))
                    log(f"LOAD FAILURE {model}: {err}")

            # 1b) surface a run that is "succeeding" while producing only errors
            for model, task, bad, total in error_storm():
                if (model, task) not in seen_storms:
                    seen_storms.add((model, task))
                    log(f"ERROR STORM {model} task {task}: {bad}/{total} of the most "
                        f"recent rows are failures -- results are not usable. "
                        f"{gpu_headroom()}")

            # A freed card can un-hold a job, so let holds be re-evaluated.
            if not live:
                held.clear()

            # 2) unstick hangs -- a live job whose log stopped moving
            for path in stale_logs(args.stale_minutes):
                age = (time.time() - path.stat().st_mtime) / 60
                # Intersect with actual panel processes: a substring match on
                # a log stem is not proof of ownership.
                panel = set(running("run_local_models.py")) | set(running("run_tier4.sh"))
                owners = [
                    pid
                    for pid in running(path.stem.replace("queued_", "").split("_task")[0])
                    if pid in panel
                ]
                if owners and age > args.stale_minutes:
                    log(f"STALE {path.name}: no output for {age:.0f} min, killing {owners}")
                    for pid in owners:
                        try:
                            os.kill(pid, 9)
                        except OSError:
                            pass

            # 3) keep the machine busy
            # A tier-4 shell driver and the run_local_models.py child it
            # spawned are ONE job, not two: counting both filled the slot
            # table permanently and the queue sat idle for hours behind a
            # single live run. Count live children when there are any, and
            # fall back to the driver only while it sits between models.
            busy = len(live) if live else len(tier4)
            if busy < args.max_concurrent_jobs and queue:
                job = queue[0]
                fits, why = job_fits(job)
                if fits:
                    queue.pop(0)
                    QUEUE.write_text(json.dumps(queue, indent=1))
                    launch(job)
                else:
                    # Hold it in place rather than launching into an OOM. The
                    # queue keeps its order; the job starts on a later cycle
                    # once whatever is resident finishes and frees its cards.
                    tag = job.get("tag", "?")
                    if tag not in held:
                        log(f"HOLDING {tag}: {why}")
                        held.add(tag)

            # 4) periodic status
            if time.time() - last_report > 1800:
                counts = completed_counts()
                done = sum(1 for v in counts.values() if v > 0)
                log(f"status: {len(live)} run(s) live, {len(tier4)} driver(s), "
                    f"{len(queue)} queued, {done} (model,task,arm) cells with data, "
                    f"{sum(counts.values()):,} records")
                last_report = time.time()
                STATE.write_text(json.dumps(
                    {"updated": datetime.now().isoformat(), "live": len(live),
                     "queued": len(queue), "records": sum(counts.values())}, indent=1))
        except Exception as exc:
            log(f"supervisor error (continuing): {type(exc).__name__}: {exc}")
        time.sleep(args.cycle_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
