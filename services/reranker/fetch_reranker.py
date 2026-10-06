"""
fetch_reranker.py - download Qwen3-Reranker-0.6B from the official repository.

Follows the project rule: verify the official source, revision, license and file sizes
BEFORE downloading, and verify SHA-256 AFTER.

Only this one model is fetched. The 4B and 8B rerankers are explicitly out of scope for
Phase 5 and are NOT downloaded by this script.

Usage:
  python fetch_reranker.py --dest <repo>\\models\\reranker\\Qwen3-Reranker-0.6B --check
  python fetch_reranker.py --dest ... --download
  python fetch_reranker.py --dest ... --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

REPO = "Qwen/Qwen3-Reranker-0.6B"
REVISION = "e61197ed45024b0ed8a2d74b80b4d909f1255473"
LICENSE = "apache-2.0"

# Every file in the repo. None of them is tracked in git; scripts/fetch_models.py
# fetches both this model and local-embedding at their pinned revisions.
FILES = [
    ".gitattributes",
    "1_LogitScore/config.json",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "config_sentence_transformers.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


def api_info() -> dict:
    url = f"https://huggingface.co/api/models/{REPO}"
    with urllib.request.urlopen(url, timeout=60) as r:
        return json.load(r)


def file_meta() -> dict[str, dict]:
    """Sizes and LFS hashes straight from the Hub, via the tree API."""
    url = f"https://huggingface.co/api/models/{REPO}/tree/{REVISION}?recursive=1"
    with urllib.request.urlopen(url, timeout=60) as r:
        items = json.load(r)
    out = {}
    for it in items:
        if it.get("type") != "file":
            continue
        lfs = it.get("lfs") or {}
        out[it["path"]] = {
            "size": it.get("size"),
            "sha256": lfs.get("oid"),
            "pointer_size": (lfs.get("size") if lfs else None),
        }
    return out


def url_for(path: str) -> str:
    return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{path}"


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def human(n: int | None) -> str:
    if n is None:
        return "?"
    return f"{n/2**20:.2f} MB" if n >= 2**20 else f"{n/2**10:.1f} KB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--check", action="store_true", help="verify upstream metadata only")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--verify", action="store_true", help="verify local files")
    args = ap.parse_args()
    dest = Path(args.dest)

    info = api_info()
    if info.get("sha") != REVISION:
        print(f"!! upstream revision changed: {info.get('sha')} != pinned {REVISION}")
        return 3
    lic = (info.get("cardData") or {}).get("license")
    if lic != LICENSE:
        print(f"!! license changed: {lic} != expected {LICENSE}")
        return 3

    meta = file_meta()
    total = sum((m["size"] or 0) for m in meta.values())
    print(f"repo     : {REPO}")
    print(f"revision : {REVISION} (matches upstream)")
    print(f"license  : {lic} (matches expected)")
    print(f"downloads: {info.get('downloads')}   likes: {info.get('likes')}")
    print(f"files    : {len(meta)}   total {human(total)}")
    for p in FILES:
        m = meta.get(p)
        if m is None:
            print(f"  !! listed file missing upstream: {p}")
            continue
        tag = "LFS" if m["sha256"] else "   "
        print(f"  {tag} {p:36} {human(m['size']):>10}")

    if args.check and not (args.download or args.verify):
        return 0

    if args.download:
        dest.mkdir(parents=True, exist_ok=True)
        import shutil
        free = shutil.disk_usage(dest.anchor or dest).free
        need = total + 200 * 2**20
        if free < need:
            print(f"!! not enough disk: {human(free)} free, need ~{human(need)}")
            return 4
        for p in FILES:
            target = dest / p
            target.parent.mkdir(parents=True, exist_ok=True)
            expect = meta[p]["sha256"] or ""
            if target.exists():
                if expect and sha256_file(target) == expect:
                    print(f"  skip (hash ok) {p}")
                    continue
            print(f"  fetching {p} ...", flush=True)
            tmp = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(url_for(p), timeout=600) as r, tmp.open("wb") as fh:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    fh.write(b)
            tmp.replace(target)
        print("download complete")

    if args.verify or args.download:
        ok = True
        for p in FILES:
            target = dest / p
            m = meta[p]
            if not target.exists():
                print(f"  MISSING {p}")
                ok = False
                continue
            size = target.stat().st_size
            expect_size = m["size"]
            line = f"  {p:36} {human(size):>10}"
            if expect_size is not None and size != expect_size:
                line += f"  SIZE MISMATCH (expected {human(expect_size)})"
                ok = False
            if m["sha256"]:
                got = sha256_file(target)
                if got == m["sha256"]:
                    line += "  sha256 OK"
                else:
                    line += f"  sha256 MISMATCH got={got[:16]} want={m['sha256'][:16]}"
                    ok = False
            print(line)
        print("\nVERIFY:", "PASS" if ok else "FAIL")
        return 0 if ok else 5

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
