"""
P57-C0 - the frozen set's answerable questions, end to end at the CURRENT k=5.

Step 4 requires per-question raw results. The pack comparison measured which facts reach the model;
this records what the model then DOES with them, which is the information needed to judge whether the
current path over-refuses (a concern for a cross-document set, and one the pack numbers alone cannot
settle).

Measurement only: cli/ask.py is imported, never modified. The frozen set is read, never modified, and
is verified against its recorded hashes before use.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IDX = REPO / "vector/qa/current"
SETDIR = REPO / "benchmarks/p57c0/devset_v1"
OUT = REPO / "benchmarks/p57c0/results/answerable_run.json"
sys.path.insert(0, str(REPO / "cli"))
import ask as ASK  # noqa: E402

REFUSAL = ASK.REFUSAL


def vram_free():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           stdout=subprocess.PIPE, timeout=30)
        return int(o.stdout.decode().strip())
    except Exception:  # noqa: BLE001
        return None


def main() -> int:
    if OUT.exists():
        print(f"REFUSING to overwrite {OUT}")
        return 3
    man = json.loads((SETDIR / "manifest.json").read_text(encoding="utf-8"))
    payload = json.dumps({k: v for k, v in man.items() if k != "manifest_sha256"},
                         ensure_ascii=False, sort_keys=True).encode()
    if hashlib.sha256(payload).hexdigest() != man["manifest_sha256"]:
        print("STOP: frozen set changed")
        return 4
    print(f"frozen set verified: {man['set_id']} {man['manifest_sha256'][:16]}")

    answerable = [e for e in man["entries"] if e["class"].endswith("answerable")]
    pack = json.loads((REPO / "benchmarks/p57c0/results/pack_comparison.json")
                      .read_text(encoding="utf-8"))
    r = ASK.Retriever(IDX, embedder="service")
    rep: dict = {"set_id": man["set_id"], "manifest_sha256": man["manifest_sha256"],
                 "k": 5, "results": {}}
    n_refused = 0
    try:
        for e in answerable:
            v0 = vram_free()
            rec = ASK.answer(e["question"], retriever=r, top_k=5, measure_vram=False)
            v1 = vram_free()
            facts_in_pack = pack["probes"][e["id"]]["by_k"]["5"]["facts"]
            n_in = sum(1 for v in facts_in_pack.values() if v["in_pack"])
            refused = bool(rec.get("refused"))
            n_refused += refused
            rep["results"][e["id"]] = {
                "question": e["question"],
                "facts_total": len(facts_in_pack), "facts_in_pack": n_in,
                "all_facts_in_pack": n_in == len(facts_in_pack),
                "outcome": "refused" if refused else "answered",
                "refused": refused,
                "answer_head": (rec.get("answer") or "")[:400],
                "n_citations": len(rec.get("citations") or []),
                "citation_verification": rec.get("citation_verification"),
                "finish_reason": rec.get("finish_reason"),
                "answer_truncated": rec.get("answer_truncated"),
                "e2e_s": rec.get("e2e_s"),
                "vram_free_before": v0, "vram_free_after": v1,
            }
            r_ = rep["results"][e["id"]]
            print(f"\n  {e['id']}: facts_in_pack={n_in}/{len(facts_in_pack)}  "
                  f"outcome={r_['outcome']}  cites={r_['n_citations']}  "
                  f"verify={r_['citation_verification']}  {r_['e2e_s']}s")
            print(f"     Q: {e['question'][:80]}")
            print(f"     A: {r_['answer_head'][:150]!r}")
    finally:
        r.close()

    ans = [k for k, v in rep["results"].items() if v["outcome"] == "answered"]
    ref = [k for k, v in rep["results"].items() if v["outcome"] == "refused"]
    print(f"\n\n=== SUMMARY at k=5 over {len(answerable)} answerable questions ===")
    print(f"  answered       : {len(ans)}  {ans}")
    print(f"  refused        : {len(ref)}  {ref}   <- each is an INCORRECT REFUSAL per the frozen rules")
    print(f"  facts in pack  : " + ", ".join(
        f"{k}={v['facts_in_pack']}/{v['facts_total']}" for k, v in rep["results"].items()))
    print(f"  truncated      : {[k for k, v in rep['results'].items() if v['answer_truncated']] or 'none'}")
    bad = [k for k, v in rep["results"].items()
           for s in (v["citation_verification"] or []) if s != "verified"]
    print(f"  non-verified citations: {bad or 'none'}")

    frees = [v for d in rep["results"].values()
             for v in (d["vram_free_before"], d["vram_free_after"]) if v]
    rep["summary"] = {"answered": ans, "refused": ref,
                      "incorrect_refusals": ref, "n_answered": len(ans), "n_refused": len(ref)}
    rep["vram"] = {"min_free_mb": min(frees) if frees else None,
                   "margin_held": bool(frees) and min(frees) >= 2048}
    print(f"  VRAM min {rep['vram']['min_free_mb']} MiB, margin held {rep['vram']['margin_held']}")

    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    print("\nNOTE: per the frozen scoring rules a refusal on an answerable question is an INCORRECT")
    print("refusal, recorded as such. The cause is reported separately and never used to re-label it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
