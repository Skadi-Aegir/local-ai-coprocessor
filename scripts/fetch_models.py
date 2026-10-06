#!/usr/bin/env python3
"""
fetch_models.py - download the Hugging Face models this project uses, at pinned revisions.

The model files are not distributed with this repository (they are Apache-2.0 works of
the Qwen team, see each model's LICENSE upstream). This script fetches every file of
each model at the exact revision recorded in docs/models/, checks the upstream license
has not changed, and verifies sizes and SHA-256 (LFS files) after download.

GGUF models for local-fast / local-smart and the OCR models are fetched separately;
see README.md.

Usage:
  python scripts/fetch_models.py                 # download + verify all
  python scripts/fetch_models.py --check         # upstream metadata only, no download
  python scripts/fetch_models.py --verify        # verify local files only
  python scripts/fetch_models.py --only embedding
Set HF_ENDPOINT (e.g. https://hf-mirror.com) to use a mirror. Stdlib only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(os.environ.get("LOCAL_AI_ROOT") or Path(__file__).resolve().parents[1])
ENDPOINT = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")

# name -> (hub repo, pinned revision, destination under the repo root)
MODELS = {
    "embedding": ("Qwen/Qwen3-Embedding-0.6B", "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
                  "models/embedding/Qwen3-Embedding-0.6B"),
    "reranker": ("Qwen/Qwen3-Reranker-0.6B", "e61197ed45024b0ed8a2d74b80b4d909f1255473",
                 "models/reranker/Qwen3-Reranker-0.6B"),
}
LICENSE = "apache-2.0"


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def file_meta(repo: str, rev: str) -> dict[str, dict]:
    items = get_json(f"{ENDPOINT}/api/models/{repo}/tree/{rev}?recursive=1")
    return {it["path"]: {"size": it.get("size"), "sha256": (it.get("lfs") or {}).get("oid")}
            for it in items if it.get("type") == "file"}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def human(n: int | None) -> str:
    if n is None:
        return "?"
    return f"{n/2**20:.2f} MB" if n >= 2**20 else f"{n/2**10:.1f} KB"


def ok_local(target: Path, m: dict) -> tuple[bool, str]:
    if not target.exists():
        return False, "MISSING"
    if m["size"] is not None and target.stat().st_size != m["size"]:
        return False, f"SIZE MISMATCH (expected {human(m['size'])})"
    if m["sha256"] and sha256_file(target) != m["sha256"]:
        return False, "sha256 MISMATCH"
    return True, "sha256 OK" if m["sha256"] else "size OK"


def fetch(name: str, mode: str) -> bool:
    repo, rev, dest_rel = MODELS[name]
    dest = REPO_ROOT / dest_rel
    print(f"=== {name}: {repo} @ {rev[:12]} -> {dest}")
    lic = (get_json(f"{ENDPOINT}/api/models/{repo}/revision/{rev}").get("cardData") or {}).get("license")
    if lic != LICENSE:
        print(f"!! license is {lic!r}, expected {LICENSE!r}; refusing")
        return False
    meta = file_meta(repo, rev)
    total = sum(m["size"] or 0 for m in meta.values())
    print(f"license {lic}, {len(meta)} files, {human(total)}")
    if mode == "check":
        for p, m in sorted(meta.items()):
            print(f"  {'LFS' if m['sha256'] else '   '} {p:36} {human(m['size']):>10}")
        return True

    if mode == "download":
        dest.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(dest).free
        if free < total + 200 * 2**20:
            print(f"!! not enough disk: {human(free)} free, need ~{human(total)}")
            return False
        for p, m in sorted(meta.items()):
            target = dest / p
            if ok_local(target, m)[0]:
                print(f"  skip (ok) {p}")
                continue
            print(f"  fetching {p} ...", flush=True)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(target.name + ".part")
            with urllib.request.urlopen(f"{ENDPOINT}/{repo}/resolve/{rev}/{p}", timeout=600) as r, \
                    tmp.open("wb") as fh:
                shutil.copyfileobj(r, fh, 1 << 20)
            tmp.replace(target)

    good = True
    for p, m in sorted(meta.items()):
        ok, why = ok_local(dest / p, m)
        good &= ok
        print(f"  {p:36} {why}")
    print(f"VERIFY {name}: {'PASS' if good else 'FAIL'}")
    return good


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--check", action="store_true", help="upstream metadata only")
    g.add_argument("--verify", action="store_true", help="verify local files only")
    ap.add_argument("--only", choices=sorted(MODELS), action="append")
    args = ap.parse_args()
    mode = "check" if args.check else "verify" if args.verify else "download"
    results = [fetch(n, mode) for n in (args.only or MODELS)]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
