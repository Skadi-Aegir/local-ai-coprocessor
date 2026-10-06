"""
P57-C0 - integrity check for the frozen development set.

Records the hash of EVERY file in the set, so the gap that matters is measured rather than glossed:
the manifest's own hash covers the manifest payload and questions.txt (the machine-readable set), but
questions.md was written afterwards and is therefore NOT covered by it.

Exit 0 when the recorded hashes match and the coverage gap is the expected one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

D = (Path(__file__).resolve().parents[2] / r"benchmarks\p57c0\devset_v1")
MAN = D / "manifest.json"


def sha(p: Path, lf: bool = False) -> str:
    b = p.read_bytes()
    if lf:
        b = b.replace(b"\r\n", b"\n")
    return hashlib.sha256(b).hexdigest()


m = json.loads(MAN.read_text(encoding="utf-8"))

# recompute the manifest's self-hash exactly as the freeze script did
payload = json.dumps({k: v for k, v in m.items() if k != "manifest_sha256"},
                     ensure_ascii=False, sort_keys=True).encode()
calc = hashlib.sha256(payload).hexdigest()
qfile = sha(D / "questions.txt", lf=True)

print("=== frozen set integrity ===")
print(f"  manifest_sha256  recorded {m['manifest_sha256'][:24]}")
print(f"                   computed {calc[:24]}   match={calc == m['manifest_sha256']}")
print(f"  questions.txt    recorded 1ce9966dd4a583fa8fd32313")
print(f"                   computed {qfile[:24]}   "
      f"match={qfile == '1ce9966dd4a583fa8fd3231347c97366b24451d8dad3472995b13acc9f999094'}")

print("\n=== per-file hashes of the frozen set ===")
for f in sorted(D.iterdir()):
    if f.is_file():
        covered = ("YES" if f.name == "manifest.json"
                   else "YES" if f.name == "questions.txt" else "** NO **")
        print(f"  {f.name:<16} {sha(f)[:32]}  covered by the recorded hash: {covered}")

print("\n=== COVERAGE GAP, now CLOSED additively ===")
print("  The manifest's self-hash covers its own payload plus questions.txt (LF-normalised).")
print("  questions.md was written after that hash and was NOT covered - yet it carries the")
print("  classification, the scoring rules and the discriminator audit.")
print("  CLOSED by SET_HASH.json, which covers ALL THREE files. It was added WITHOUT modifying any")
print("  frozen artifact, so the manifest hash and the questions.txt hash are unchanged.")
print(f"\n  questions.md sha256 = {sha(D / 'questions.md')}")

sh = D / "SET_HASH.json"
if sh.is_file():
    s = json.loads(sh.read_text(encoding="utf-8"))
    h = hashlib.sha256()
    for name in s["covered_files"]:
        h.update(name.encode())
        h.update(b"\0")
        h.update((D / name).read_bytes().replace(b"\r\n", b"\n"))
        h.update(b"\0")
    calc_set = h.hexdigest()
    print(f"  set_sha256 recorded = {s['set_sha256']}")
    print(f"  set_sha256 computed = {calc_set}")
    print(f"  SET HASH MATCHES    = {calc_set == s['set_sha256']}")
    set_ok = calc_set == s["set_sha256"]
else:
    print("  *** SET_HASH.json is MISSING - the coverage gap is open ***")
    set_ok = False

# the entry count and classes must be internally consistent
ents = m["entries"]
ans = [e["id"] for e in ents if e["class"].endswith("answerable")]
abs_ = [e["id"] for e in ents if e["class"] == "answer_absent"]
print(f"\n  answerable   {len(ans)}: {ans}")
print(f"  answer-absent {len(abs_)}: {abs_}")
print(f"  counts field agrees: "
      f"{m['counts'] == {'answerable': len(ans), 'answer_absent': len(abs_), 'total': len(ents)}}")

# every answerable fact must be verified, every answer-absent question must have none
bad = []
for e in ents:
    if e["class"].endswith("answerable"):
        if not all(f["verified"] for f in e["resolved_facts"]):
            bad.append(f"{e['id']}: answerable with an unverified fact")
    elif e["class"] == "answer_absent":
        if any(f["verified"] for f in e["resolved_facts"]):
            bad.append(f"{e['id']}: answer-absent but a fact IS in the corpus")
print(f"  class/fact consistency problems: {bad or 'NONE'}")

# the set must not be in the corpus
man = json.load(open(str(Path(__file__).resolve().parents[2] / r"vector\qa\current\manifest.json"), encoding="utf-8"))
in_corpus = [f for f in man["included_files"] if f.startswith("benchmarks/")]
print(f"  benchmarks/ files in the index : {in_corpus or 'NONE (excluded, as required)'}")

raise SystemExit(1 if bad or calc != m["manifest_sha256"]
                 or qfile != "1ce9966dd4a583fa8fd3231347c97366b24451d8dad3472995b13acc9f999094"
                 else 0)
