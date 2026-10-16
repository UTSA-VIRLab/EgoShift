#!/usr/bin/env python3
"""Publish the reviewer artifact to Hugging Face and verify it as a reviewer would.

Everything that can be automated is automated. What cannot be automated is the
account: creating one needs an email and a human at a signup form. Once the
account and an empty public dataset repo exist, this script does the rest.

    python release/tools/publish_artifact.py --repo <anon-name>/egoshift-review

Order of operations, and why:
  1. refuse to upload if the anonymity scan finds a fatal leak;
  2. rewrite croissant.json so it carries the real record URL rather than the
     placeholder, then refresh CHECKSUMS.json so the digests match what ships;
  3. upload;
  4. download the bundle back over plain HTTPS with no credentials and compare
     the digest. A repo that reads fine while you are logged in can still be
     invisible to a reviewer, and that is the failure this step catches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ART = ROOT / "work/artifact_u7"
UPLOAD = ["egoshift_reviewer_bundle.tar.gz", "croissant.json",
          "CHECKSUMS.json", "README.md", "LICENSE", "NOTICE"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def refresh_checksums() -> None:
    out = {"note": "Verify downloads against these digests.", "files": {}}
    for f in sorted(ART.iterdir()):
        # digest exactly what ships, so "a digest for every file here" is true
        # of the published repository rather than of this staging directory
        if f.is_file() and f.name in UPLOAD and f.name != "CHECKSUMS.json":
            out["files"][f.name] = {"sha256": sha256_file(f), "bytes": f.stat().st_size,
                                    "mb": round(f.stat().st_size / 1e6, 1)}
    (ART / "CHECKSUMS.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True,
                   help="Hugging Face dataset repo id, e.g. anon-name/egoshift-review")
    p.add_argument("--skip-anonymity", action="store_true",
                   help="only for a re-upload where the scan already passed")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    base = f"https://huggingface.co/datasets/{args.repo}"
    raw = f"{base}/resolve/main"

    # 1. anonymity gate
    if not args.skip_anonymity:
        print("[1/4] anonymity scan")
        rc = subprocess.run([sys.executable, str(ROOT / "release/tools/verify_anonymity.py")],
                            capture_output=True, text=True)
        tail = rc.stdout.strip().splitlines()[-1] if rc.stdout.strip() else ""
        print(f"      {tail}")
        if rc.returncode != 0:
            print("      refusing to publish: fix the fatal hits first")
            return 1
    else:
        print("[1/4] anonymity scan skipped by request")

    # 2. bake the real URL into the metadata, then re-hash
    print("[2/4] writing record URL into croissant.json")
    rc = subprocess.run([sys.executable, str(ROOT / "release/tools/build_croissant.py"),
                         "--url", base], capture_output=True, text=True)
    if rc.returncode != 0:
        print(rc.stdout + rc.stderr)
        return 1
    refresh_checksums()
    digest = json.loads((ART / "CHECKSUMS.json").read_text())["files"]
    bundle_sha = digest["egoshift_reviewer_bundle.tar.gz"]["sha256"]
    print(f"      bundle sha256 {bundle_sha}")

    if args.dry_run:
        print("[3/4] dry run, nothing uploaded")
        print(f"[4/4] would verify {raw}/egoshift_reviewer_bundle.tar.gz")
        return 0

    # 3. upload
    print(f"[3/4] uploading {len(UPLOAD)} files to {args.repo}")
    try:
        from huggingface_hub import HfApi
    except ImportError:
        print("      pip install huggingface_hub")
        return 1
    api = HfApi()
    for name in UPLOAD:
        path = ART / name
        if not path.exists():
            print(f"      missing, skipped: {name}")
            continue
        api.upload_file(path_or_fileobj=str(path), path_in_repo=name,
                        repo_id=args.repo, repo_type="dataset")
        print(f"      uploaded {name} ({path.stat().st_size / 1e6:.1f} MB)")

    # 4. verify as an anonymous reviewer: no token, no session
    print("[4/4] verifying the download with no credentials")
    url = f"{raw}/egoshift_reviewer_bundle.tar.gz"
    h = hashlib.sha256()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "anon-check"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                h.update(chunk)
    except Exception as exc:
        print(f"      FAILED: {exc}")
        print("      the repo is probably private or gated. A reviewer cannot")
        print("      reach it, which is the condition the track rules forbid.")
        return 1
    got = h.hexdigest()
    ok = got == bundle_sha
    print(f"      downloaded sha256 {got}")
    print(f"      {'matches' if ok else 'DOES NOT MATCH'} the local digest")
    if not ok:
        return 1

    print("\npublished and reachable without credentials")
    print(f"  record : {base}")
    print(f"  bundle : {url}")
    print("\nSend the record URL back so it can go into Section 4 and the")
    print("submission form.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
