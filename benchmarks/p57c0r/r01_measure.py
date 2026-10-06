"""
P57-C0R - corrected per-question measurement.

WHAT WAS WRONG BEFORE, and what this does instead:

1. CARRIER TRUNCATION. The facts file recorded carriers[:12] as "carriers" while reporting a separate
   n_carriers, so any consumer reading `carriers` saw a truncated set. This version writes the FULL
   carrier set for every fact.

2. MIXED REQUESTS. The old pack comparison called r.search(...) once and ASK.answer(...) again, then
   reported the first call's candidates beside the second call's pack. Those are two different top-k
   lists; the pack is the only thing the model saw. This version makes ONE ASK.answer() call per
   (question, k) and derives EVERYTHING from that single call's own pack. No candidate list is
   reported at all, because it is not what the model saw.

3. TRUNCATED RECORDS. The old run stored answer_head (400 chars) only, so its central claim could not
   be re-verified. This version stores the FULL answer, the full pack with every supplied source's
   file and line range, the citations, the index version, token counts, latency and VRAM.

4. BINARY FACT DETECTION ON BRITTLE LITERALS. The old detector asked whether a carrier CHUNK ID was in
   the supplied list, which says nothing about whether the fact's CONTENT reached the model. This
   version tests the fact's markers against the ACTUAL PACK TEXT and against the ACTUAL ANSWER TEXT
   separately, and can say "undetermined by markers" instead of forcing a verdict.

Verdicts per answerable question are recorded as complete / partial / incorrect per the frozen rules,
with the note that these are MARKER-BASED and therefore provisional wherever a fact is undetermined:
marker presence is evidence about content, never proof of support.

Read-only with respect to code: cli/ask.py is imported, never modified.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IDX = REPO / "vector/qa/current"
SETDIR = REPO / "benchmarks/p57c0/devset_v1"
sys.path.insert(0, str(REPO / "cli"))
import ask as ASK  # noqa: E402
sys.path.insert(0, str(REPO / "services" / "retrieval"))
from index_store import IndexStore  # noqa: E402

KS = [5, 8, 10]

# Robust content markers per fact, keyed by the FROZEN manifest's fact names (a key mismatch silently
# yields "no markers defined", which is how the first run of this script mis-reported v02).
#
# The frozen manifest records each fact's detection literals as any_of ["torch_reserved",
# "reserved_mb"] - the COLUMN names. Measured, those are too strict: the fact is "what the embedding
# service's memory figure MEANS", and the passage that explains it says "VRAM, torch attribution" and
# "cudaMemGetInfo total - free" while never printing torch_reserved. So the marker set here is the
# SEMANTIC one, and the difference from the frozen literals is recorded as a correction in the errata
# rather than by editing the frozen artifacts.
FACT_MARKERS = {
    "v01": {
        "fast_port": {"only_of": ["8111"]},
        "embed_port": {"only_of": ["8113"]},
    },
    "v02": {
        "torch_reserved_meaning": {
            "any_of": ["cudaMemGetInfo", "torch attribution", "VRAM delta", "vram_mb",
                       "reserved_mb", "attribution"],
            "min_any": 1,
            "note": ("what the figure IS: cudaMemGetInfo total-free, not per-process attribution, "
                     "plus the measured delta. Note this deliberately does NOT accept the bare word "
                     "'margin', which belongs to the OTHER fact of this question."),
        },
        "margin_policy": {"any_of": ["2048", "2 GiB", "margin"], "min_any": 2},
    },
    "v03": {
        "unload_figures": {"any_of": ["1368", "1392", "2591", "2595"], "min_any": 1},
        "cycle_latency": {"conjunctive": ["22", "24"]},
    },
    "v04": {
        "char_cap": {"only_of": ["8000"]},
        "why_raised": {"any_of": ["Raised from 2000 to 8000", "capped every chunk at 2000"],
                       "min_any": 1},
    },
    "v05": {
        "reranker_port": {"only_of": ["8114"]},
        "ocr_port": {"only_of": ["8116"]},
    },
    "v06": {
        "score_method": {"any_of": ["softmax", "logit"], "min_any": 1},
    },
}


def vram_free():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           stdout=subprocess.PIPE, timeout=30)
        return int(o.stdout.decode().strip())
    except Exception:  # noqa: BLE001
        return None


def markers_hit(text: str, spec: dict) -> tuple[bool | None, list[str], str]:
    """Returns (verdict, matched, how). verdict None means the markers cannot decide."""
    low = (text or "").lower()
    if "conjunctive" in spec:
        need = spec["conjunctive"]
        got = [s for s in need if s.lower() in low]
        return (len(got) == len(need), got, "conjunctive")
    if "only_of" in spec:
        got = [s for s in spec["only_of"] if s.lower() in low]
        return (bool(got), got, "only_of")
    if "any_of" in spec:
        got = [s for s in spec["any_of"] if s.lower() in low]
        n = spec.get("min_any", 1)
        if got:
            return (len(got) >= n, got, f"any_of>={n}")
        return (False, [], f"any_of>={n}")
    return (None, [], "no markers defined")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_p = Path(args.out)
    if out_p.exists():
        print(f"REFUSING to overwrite {out_p}")
        return 3
    out_p.parent.mkdir(parents=True, exist_ok=True)

    man = json.loads((SETDIR / "manifest.json").read_text(encoding="utf-8"))
    payload = json.dumps({k: v for k, v in man.items() if k != "manifest_sha256"},
                         ensure_ascii=False, sort_keys=True).encode()
    if hashlib.sha256(payload).hexdigest() != man["manifest_sha256"]:
        print("STOP: the frozen set changed")
        return 4
    seth = json.loads((SETDIR / "SET_HASH.json").read_text(encoding="utf-8"))
    h = hashlib.sha256()
    for n in seth["covered_files"]:
        h.update(n.encode()); h.update(b"\0")
        h.update((SETDIR / n).read_bytes().replace(b"\r\n", b"\n")); h.update(b"\0")
    if h.hexdigest() != seth["set_sha256"]:
        print("STOP: the set-level hash does not match")
        return 4
    print(f"frozen set verified: {man['set_id']} manifest={man['manifest_sha256'][:16]} "
          f"set={seth['set_sha256'][:16]}")

    idxver = json.loads((IDX / "manifest.json").read_text(encoding="utf-8"))["index_version"]
    answerable = [e for e in man["entries"] if e["class"].endswith("answerable")]
    st = IndexStore.load(IDX, 1024)
    r = ASK.Retriever(IDX, embedder="service")
    rep: dict = {"set_id": man["set_id"], "manifest_sha256": man["manifest_sha256"],
                 "set_sha256": seth["set_sha256"], "index_version": idxver, "ks": KS,
                 "corrections_applied": {
                     "full_carrier_sets": True,
                     "one_request_per_measurement": True,
                     "full_answer_and_pack_stored": True,
                     "markers_tested_against_pack_and_answer_text": True},
                 "probes": {}}
    try:
        for e in answerable:
            qid = e["id"]
            specs = FACT_MARKERS.get(qid, {})
            per_k = {}
            for k in KS:
                v0 = vram_free()
                t0 = time.time()
                rec = ASK.answer(e["question"], retriever=r, top_k=k, measure_vram=False)
                dt = round(time.time() - t0, 3)
                v1 = vram_free()

                pack = rec.get("pack") or {}
                supplied = pack.get("supplied") or []
                omitted = pack.get("omitted") or []
                ans = rec.get("answer") or ""

                # the ACTUAL text the model saw, read from the same chunks this request supplied
                parts = []
                for s in supplied:
                    row = st.conn.execute("SELECT content FROM chunks WHERE chunk_id=?",
                                          (s.get("chunk_id"),)).fetchone()
                    parts.append((row[0] if row else "") or "")
                pack_text = "\n".join(parts)

                facts = {}
                for f in e["resolved_facts"]:
                    name = f["name"]
                    spec = specs.get(name, {})
                    in_pack, pm, phow = markers_hit(pack_text, spec)
                    in_ans, am, ahow = markers_hit(ans, spec)
                    facts[name] = {
                        "in_pack": in_pack, "in_answer": in_ans,
                        "pack_markers_matched": pm, "answer_markers_matched": am,
                        "detect": phow, "undetermined": in_pack is None,
                        "carrier_count": f["n_carriers"],
                        "carrier_chunks_full": f["carriers"],       # FULL set, not truncated
                    }

                decided = [v for v in facts.values() if not v["undetermined"]]
                n_pack = sum(1 for v in decided if v["in_pack"])
                n_ans = sum(1 for v in decided if v["in_answer"])

                if rec.get("error"):
                    verdict = "error"
                elif rec.get("refused"):
                    verdict = "incorrect_refusal"
                elif decided and n_ans == len(decided) and n_ans == len(e["resolved_facts"]):
                    verdict = "complete"
                elif n_ans > 0:
                    verdict = "partial"
                else:
                    verdict = "incorrect"

                per_k[str(k)] = {
                    "prompt_tokens": pack.get("prompt_tokens"),
                    "budget_tokens": pack.get("budget_tokens"),
                    "count_method": pack.get("count_method"),
                    "n_supplied": len(supplied), "n_omitted": len(omitted),
                    "supplied": [{"source": s.get("source"), "source_path": s.get("source_path"),
                                  "start_line": s.get("start_line"),
                                  "end_line": s.get("end_line"), "chunk_id": s.get("chunk_id"),
                                  "truncated": s.get("truncated")} for s in supplied],
                    "omitted": omitted,
                    "facts": facts,
                    "facts_in_pack": n_pack, "facts_in_answer": n_ans,
                    "facts_undetermined": sum(1 for v in facts.values() if v["undetermined"]),
                    "outcome": "refused" if rec.get("refused") else "answered",
                    "verdict_marker_based": verdict,
                    "finish_reason": rec.get("finish_reason"),
                    "answer_truncated": rec.get("answer_truncated"),
                    "answer": ans,                                  # FULL answer
                    "citations": [{"source": c.get("source"), "source_path": c.get("source_path"),
                                   "start_line": c.get("start_line"), "end_line": c.get("end_line"),
                                   "verification": (c.get("verification") or {}).get("status")}
                                  for c in (rec.get("citations") or [])],
                    "index_version": rec.get("index_version"),
                    "elapsed_s": dt, "vram_free_before": v0, "vram_free_after": v1,
                }
                print(f"  {qid} k={k:>2}: pack={n_pack}/{len(facts)} answer={n_ans}/{len(facts)} "
                      f"{verdict:<17} supplied={len(supplied)} tokens={pack.get('prompt_tokens')}")
            rep["probes"][qid] = {"question": e["question"],
                                  "required_facts": [f["name"] for f in e["resolved_facts"]],
                                  "by_k": per_k}
    finally:
        r.close()
        st.close()

    print("\n=== overall, facts PRESENT IN THE PACK (from the same request) ===")
    rep["summary"] = {}
    for k in KS:
        comp = [q for q, v in rep["probes"].items()
                if v["by_k"][str(k)]["facts_in_pack"] == len(v["required_facts"])]
        incomp = [q for q in rep["probes"] if q not in comp]
        toks = [v["by_k"][str(k)]["prompt_tokens"] for v in rep["probes"].values()]
        rep["summary"][str(k)] = {"complete": comp, "incomplete": incomp,
                                  "max_prompt_tokens": max(t for t in toks if t is not None)}
        print(f"  k={k:>2}: {len(comp)}/{len(rep['probes'])} complete   incomplete={incomp}"
              f"   max prompt_tokens={rep['summary'][str(k)]['max_prompt_tokens']}")

    print("\n=== overall, facts COVERED IN THE ANSWER ===")
    for k in KS:
        full = [q for q, v in rep["probes"].items()
                if v["by_k"][str(k)]["facts_in_answer"] == len(v["required_facts"])]
        print(f"  k={k:>2}: {len(full)}/{len(rep['probes'])} answers cover every required fact   "
              f"covered={full}")

    print("\n=== verdicts (marker-based) ===")
    for k in KS:
        vs = {q: v["by_k"][str(k)]["verdict_marker_based"] for q, v in rep["probes"].items()}
        print(f"  k={k:>2}: {vs}")

    frees = [v for q in rep["probes"].values() for d in q["by_k"].values()
             for v in (d["vram_free_before"], d["vram_free_after"]) if v]
    rep["vram"] = {"min_free_mb": min(frees) if frees else None,
                   "margin_mb": 2048, "margin_held": bool(frees) and min(frees) >= 2048}
    print(f"\n  VRAM min {rep['vram']['min_free_mb']} MiB, margin held {rep['vram']['margin_held']}")

    out_p.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
