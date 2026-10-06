"""
P57-C0 - verify the answer-absent questions refuse, as the frozen set requires.

reads the frozen set, asks the three answer-absent questions through the CLI core, and requires a
refusal in each case. A refusal here is the CORRECT outcome; an answer is a failure.

Also re-checks that the set's files still hash as recorded, so this cannot run against a modified set.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IDX = REPO / "vector/qa/current"
SETDIR = REPO / "benchmarks/p57c0/devset_v1"
OUT = REPO / "benchmarks/p57c0/results/answer_absent.json"
sys.path.insert(0, str(REPO / "cli"))
import ask as ASK  # noqa: E402


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
    got = hashlib.sha256(payload).hexdigest()
    if got != man["manifest_sha256"]:
        print(f"STOP: frozen set changed ({got})")
        return 4
    qsha = hashlib.sha256((SETDIR / "questions.txt").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    if qsha != "1ce9966dd4a583fa8fd3231347c97366b24451d8dad3472995b13acc9f999094":
        print("STOP: questions.txt changed")
        return 4
    print(f"frozen set + questions verified ({man['manifest_sha256'][:16]})")

    absent = [e for e in man["entries"] if e["class"] == "answer_absent"]
    r = ASK.Retriever(IDX, embedder="service")
    rep: dict = {"set_id": man["set_id"], "results": {}}
    n_ok = 0
    try:
        for e in absent:
            v0 = vram_free()
            rec = ASK.answer(e["question"], retriever=r, top_k=5, measure_vram=False)
            v1 = vram_free()
            refused = bool(rec.get("refused"))
            n_ok += refused
            rep["results"][e["id"]] = {
                "question": e["question"],
                "expected": "refuse",
                "refused": refused,
                "correct": refused,
                "answer": rec.get("answer"),
                "reason": rec.get("reason"),
                "n_citations": len(rec.get("citations") or []),
                "finish_reason": rec.get("finish_reason"),
                "answer_truncated": rec.get("answer_truncated"),
                "vram_free_before": v0, "vram_free_after": v1,
            }
            print(f"\n  {e['id']}: refused={refused}  correct={refused}")
            print(f"     Q: {e['question'][:80]}")
            print(f"     A: {(rec.get('answer') or '')[:100]}")
            if not refused:
                print(f"     *** INCORRECT: it answered a question the corpus cannot answer ***")
    finally:
        r.close()

    rep["summary"] = {"n_answer_absent": len(absent), "n_correct_refusals": n_ok,
                      "all_correct": n_ok == len(absent)}
    frees = [v for d in rep["results"].values()
             for v in (d["vram_free_before"], d["vram_free_after"]) if v]
    rep["vram"] = {"min_free_mb": min(frees) if frees else None,
                   "margin_held": bool(frees) and min(frees) >= 2048}
    print(f"\n=== {n_ok}/{len(absent)} correct refusals; "
          f"VRAM min {rep['vram']['min_free_mb']} MiB, margin held {rep['vram']['margin_held']}")

    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0 if rep["summary"]["all_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
