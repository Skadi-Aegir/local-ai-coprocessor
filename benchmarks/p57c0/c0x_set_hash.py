"""
P57-C0 - close the freeze coverage gap, additively.

THE GAP: the frozen manifest's self-hash covers its own payload plus questions.txt, but questions.md
was written after the manifest was hashed. questions.md is not decoration - it carries the
DEVELOPMENT VALIDATION classification, the per-question tables, the pre-registered scoring rules and
the discriminator audit. A reader cannot currently verify that it has not changed.

THE FIX, which does NOT modify any frozen artifact: record a SET-LEVEL hash that covers all three
files, in a SEPARATE file alongside the set. The frozen manifest and questions.txt keep their original
bytes and their original hashes, so nothing already published becomes invalid; the set simply gains a
single hash that covers everything in it.

Writes benchmarks/p57c0/devset_v1/SET_HASH.json. Refuses if it already exists.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

D = (Path(__file__).resolve().parents[2] / r"benchmarks/p57c0/devset_v1")
OUT = D / "SET_HASH.json"

COVERED = ["manifest.json", "questions.txt", "questions.md"]


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    if OUT.exists():
        print(f"REFUSING to overwrite {OUT}")
        return 3

    per_file = {}
    for name in COVERED:
        p = D / name
        if not p.is_file():
            print(f"MISSING: {p}")
            return 4
        raw = p.read_bytes()
        per_file[name] = {
            "sha256": sha_bytes(raw),
            "sha256_lf": sha_bytes(raw.replace(b"\r\n", b"\n")),
            "bytes": len(raw),
        }

    # set hash over the LF-normalised bytes, so a CRLF checkout cannot change it
    h = hashlib.sha256()
    for name in COVERED:
        h.update(name.encode())
        h.update(b"\0")
        h.update((D / name).read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    set_hash = h.hexdigest()

    man = json.loads((D / "manifest.json").read_text(encoding="utf-8"))
    out = {
        "set_id": man["set_id"],
        "classification": man["classification"],
        "purpose": ("Set-level hash closing the freeze coverage gap: the manifest's own hash covers "
                    "only its payload and questions.txt, while questions.md carries the "
                    "classification, the scoring rules and the discriminator audit. This file covers "
                    "ALL THREE. It is ADDITIVE - no frozen artifact was modified to produce it, so "
                    "the manifest hash and the questions.txt hash are unchanged."),
        "hash_scheme": "sha256-lf-v1 over the concatenation of <name>\\0<lf-bytes>\\0 in the order "
                       "manifest.json, questions.txt, questions.md",
        "covered_files": COVERED,
        "per_file": per_file,
        "set_sha256": set_hash,
        "manifest_sha256_unchanged": man["manifest_sha256"],
    }
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== set hash written ===")
    for n, v in per_file.items():
        print(f"  {n:<16} {v['sha256'][:32]}  ({v['bytes']} bytes)")
    print(f"\n  SET SHA256 = {set_hash}")
    print(f"  manifest_sha256 (unchanged) = {man['manifest_sha256']}")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
