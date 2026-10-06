"""
P57-C0 step 4 (remainder) - the pack-level comparison the round-1 report did not measure.

Round 1 measured whether a required FACT's carrier chunk appeared in the top-k. It did NOT measure
what the pack actually contains at each k, nor the token cost, nor whether raising k causes the packer
to DROP evidence to fit its budget. Step 4 requires those, and the "no squeezing" condition is decided
by them, so this is measurement rather than implementation.

Read-only with respect to code: cli/ask.py is imported, never modified. Nothing is written to the
index. The frozen set benchmarks/p57c0/devset_v1 is read, never modified.

For each k in {5, 8, 10} and each answerable question it records:
    * prompt tokens and the budget, and how many sources were supplied vs omitted for budget
    * whether each required fact's carrier chunk was SUPPLIED to the model (not merely retrieved)
    * latency of search + pack, and free VRAM sampled around it
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IDX = REPO / "vector/qa/current"
SETDIR = REPO / "benchmarks/p57c0/devset_v1"
OUT = REPO / "benchmarks/p57c0/results/pack_comparison.json"
sys.path.insert(0, str(REPO / "cli"))
import ask as ASK  # noqa: E402

KS = [5, 8, 10]


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
    # integrity gate: refuse to measure against a modified set
    import hashlib
    payload = json.dumps({k: v for k, v in man.items() if k != "manifest_sha256"},
                         ensure_ascii=False, sort_keys=True).encode()
    got = hashlib.sha256(payload).hexdigest()
    if got != man["manifest_sha256"]:
        print(f"STOP: the frozen set changed ({got} != {man['manifest_sha256']})")
        return 4
    print(f"frozen set verified: {man['set_id']} {man['manifest_sha256'][:16]}  "
          f"class={man['classification']}")

    answerable = [e for e in man["entries"] if e["class"].endswith("answerable")]
    r = ASK.Retriever(IDX, embedder="service")
    rep: dict = {"set_id": man["set_id"], "manifest_sha256": man["manifest_sha256"],
                 "index_version": man["index_version_at_freeze"], "ks": KS, "probes": {}}
    try:
        for e in answerable:
            print(f"\n=== {e['id']} (k sweep) ===")
            print(f"  Q: {e['question'][:84]}")
            per_k = {}
            for k in KS:
                v0 = vram_free()
                t0 = time.time()
                hits = r.search(e["question"], top_k=k)
                search_s = round(time.time() - t0, 3)
                # what the record's candidate list holds, then what the packer would keep
                rec = ASK.answer(e["question"], retriever=r, top_k=k, measure_vram=False)
                total_s = round(time.time() - t0, 3)
                v1 = vram_free()

                pack = rec.get("pack") or {}
                supplied = pack.get("supplied") or []
                omitted = pack.get("omitted") or []
                supplied_ids = {s.get("chunk_id") for s in supplied}

                fact_status = {}
                for f in e["resolved_facts"]:
                    carriers = {c["chunk_id"] for c in f["carriers"]}
                    in_pack = bool(carriers & supplied_ids)
                    # was it at least retrieved?
                    retrieved_ids = {h.get("chunk_id") for h in hits}
                    fact_status[f["name"]] = {
                        "in_pack": in_pack,
                        "retrieved": bool(carriers & retrieved_ids),
                        "n_carriers": f["n_carriers"],
                    }
                complete = all(v["in_pack"] for v in fact_status.values())
                per_k[str(k)] = {
                    "prompt_tokens": pack.get("prompt_tokens"),
                    "budget_tokens": pack.get("budget_tokens"),
                    "count_method": pack.get("count_method"),
                    "n_supplied": len(supplied),
                    "n_omitted": len(omitted),
                    "pack_chars": pack.get("pack_chars"),
                    "facts": fact_status,
                    "all_facts_in_pack": complete,
                    "outcome": ("refused" if rec.get("refused") else "answered"),
                    "finish_reason": rec.get("finish_reason"),
                    "answer_truncated": rec.get("answer_truncated"),
                    "citation_verification": rec.get("citation_verification"),
                    "search_s": search_s, "total_s": total_s,
                    "vram_free_before": v0, "vram_free_after": v1,
                    "n_retrieved": len(hits),
                }
                print(f"   k={k:>2}: supplied={len(supplied)} omitted={len(omitted)} "
                      f"prompt_tokens={pack.get('prompt_tokens')}/{pack.get('budget_tokens')} "
                      f"facts_in_pack={sum(1 for v in fact_status.values() if v['in_pack'])}"
                      f"/{len(fact_status)}  complete={complete}  {total_s}s")
            rep["probes"][e["id"]] = {"question": e["question"], "by_k": per_k}

        print("\n\n=== SUMMARY over answerable questions (fact present IN THE PACK) ===")
        rep["summary"] = {}
        for k in KS:
            comp = [qid for qid, v in rep["probes"].items()
                    if v["by_k"][str(k)]["all_facts_in_pack"]]
            incom = [qid for qid in rep["probes"] if qid not in comp]
            toks = [v["by_k"][str(k)]["prompt_tokens"] for v in rep["probes"].values()]
            omit = [v["by_k"][str(k)]["n_omitted"] for v in rep["probes"].values()]
            rep["summary"][str(k)] = {
                "complete": comp, "incomplete": incom,
                "prompt_tokens_max": max(t for t in toks if t is not None) if any(toks) else None,
                "any_omitted": any(o > 0 for o in omit),
            }
            print(f"  k={k:>2}: {len(comp)}/{len(rep['probes'])} complete   incomplete={incom}")
            print(f"        max prompt_tokens={rep['summary'][str(k)]['prompt_tokens_max']} "
                  f"(budget 7616)   any source omitted for budget: "
                  f"{rep['summary'][str(k)]['any_omitted']}")

        c5 = set(rep["summary"]["5"]["complete"])
        c8 = set(rep["summary"]["8"]["complete"])
        regressed = sorted(c5 - c8)
        rep["squeeze_check"] = {
            "complete_at_k5": sorted(c5), "complete_at_k8": sorted(c8),
            "regressed_by_raising_k": regressed,
            "verdict": ("NO REGRESSION - raising k displaced no evidence"
                        if not regressed else f"REGRESSION: {regressed}"),
        }
        print(f"\n  squeeze check: {rep['squeeze_check']['verdict']}")

        frees = [v["by_k"][str(k)]["vram_free_before"] for v in rep["probes"].values()
                 for k in KS if v["by_k"][str(k)]["vram_free_before"]]
        frees += [v["by_k"][str(k)]["vram_free_after"] for v in rep["probes"].values()
                  for k in KS if v["by_k"][str(k)]["vram_free_after"]]
        rep["vram"] = {"min_free_mb": min(frees) if frees else None,
                       "margin_mb": 2048,
                       "margin_held": bool(frees) and min(frees) >= 2048}
        print(f"  VRAM min free {rep['vram']['min_free_mb']} MiB, "
              f"margin held: {rep['vram']['margin_held']}")
    finally:
        r.close()

    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
