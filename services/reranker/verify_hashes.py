"""
verify_hashes.py - recompute SHA-256 for the models this project deploys and compare
against what the config and docs claim.

Written because hashes were transcribed by hand twice and both times were wrong: the
reranker's YAML briefly carried the EMBEDDING model's weight hash and its tokenizer hash.
A hash that is copied rather than computed is worse than no hash, because it looks like
verification. This script makes the check mechanical.

Usage:
  python verify_hashes.py                 # check every recorded hash
  python verify_hashes.py --emit          # print the snippet to paste into a config
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

TARGETS = [
    ("reranker", REPO / "models/reranker/Qwen3-Reranker-0.6B/model.safetensors",
     REPO / "config/local-reranker.yaml"),
    ("reranker-tokenizer", REPO / "models/reranker/Qwen3-Reranker-0.6B/tokenizer.json",
     REPO / "config/local-reranker.yaml"),
    ("embedding", REPO / "models/embedding/Qwen3-Embedding-0.6B/model.safetensors",
     REPO / "config/local-embedding.yaml"),
]

# Upstream LFS oids, recorded at download time from the Hub tree API.
UPSTREAM = {
    "reranker": "27cd75a405b9c1b46b59abfd88aaa209e6fed2a1972cde9b70e7659537c5e65b",
    "reranker-tokenizer": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "embedding": "0437e45c94563b09e13cb7a64478fc406947a93cb34a7e05870fc8dcd48e23fd",
}


def sha256(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def claimed_in(path: Path) -> list[str]:
    if not path.exists():
        return []
    txt = path.read_text(encoding="utf-8", errors="replace")
    return re.findall(r"\b([0-9a-f]{64})\b", txt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit", action="store_true")
    args = ap.parse_args()

    if args.emit:
        for name, f, _ in TARGETS:
            if f.exists():
                print(f"{name}: {sha256(f)}")
        return 0

    fails = 0
    for name, f, config in TARGETS:
        print(f"--- {name} ---")
        if not f.exists():
            print(f"  MISSING {f}")
            fails += 1
            continue
        actual = sha256(f)
        print(f"  file     : {f}")
        print(f"  size     : {f.stat().st_size/2**20:.2f} MB")
        print(f"  computed : {actual}")
        up = UPSTREAM.get(name)
        ok_up = (up == actual)
        print(f"  upstream : {up}  {'MATCH' if ok_up else 'MISMATCH'}")
        if not ok_up:
            fails += 1
        claims = claimed_in(config)
        if claims:
            ok_cfg = actual in claims
            print(f"  in {config.name}: {'MATCH' if ok_cfg else 'NOT FOUND/MISMATCH'}")
            if not ok_cfg:
                print(f"    claimed values: {claims}")
                fails += 1
        else:
            print(f"  in {config.name}: no hash recorded")
        print()

    print("VERIFY:", "PASS" if fails == 0 else f"FAIL ({fails} problems)")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
