"""
P57-C0 step 3 - author and FREEZE the development-validation question set.

Written BEFORE any retrieval-code change, as step 3 requires. Nothing under cli/, services/ or config/
is modified by this script.

Discipline applied here, learned from the last two rounds:
  * EVERY claimed fact is validated against the current index before it goes into the set. A question
    only enters the "answerable" class if at least one indexed CHUNK actually contains each of its
    required facts, and the set records that chunk id and its line range. Anything whose fact is not
    in the corpus is classed answer-absent instead - which is how d03 was caught.
  * The set is labelled DEVELOPMENT VALIDATION: the construction side wrote the questions, so
    independence is limited and it is NOT an independent held-out set.
  * benchmarks/ is outside the builder's INCLUDE_DIRS, so the set cannot enter the index; that is
    verified rather than assumed.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IDX = REPO / "vector/qa/current"
OUT = REPO / "benchmarks/p57c0/devset_v1"
sys.path.insert(0, str(REPO / "services" / "retrieval"))
from index_store import IndexStore  # noqa: E402

SET_ID = "p57c0_devset_v1"
CLASS = "DEVELOPMENT VALIDATION (not independent held-out)"

# Each question: id, class, question text, and the facts that must be supported.
# A fact is {name, any_of:[literals], } - it is satisfied by any chunk containing ANY of the literals.
QUESTIONS = [
    {
        "id": "v01",
        "class": "cross_file_answerable",
        "question": "Which components does a question pass through before reaching local-fast, "
                    "and which ports do they use?",
        "facts": [
            {"name": "fast_port", "any_of": ["8111"]},
            {"name": "embed_port", "any_of": ["8113"]},
        ],
        "note": "DIAGNOSTIC CARRY-OVER from d01: the 8113 fact is carried by 15 indexed chunks yet "
                "NONE reached top-10 for this wording - a ranking failure, not a depth one.",
    },
    {
        "id": "v02",
        "class": "cross_file_answerable",
        "question": "How do the embedding service memory figures and the scheduler margin relate "
                    "to each other?",
        "facts": [
            {"name": "torch_reserved_meaning", "any_of": ["torch_reserved", "reserved_mb"]},
            {"name": "margin_policy", "any_of": ["2048"]},
        ],
        "note": "CARRY-OVER from d02: torch_reserved carriers sit at rank 15 for this wording.",
    },
    {
        "id": "v03",
        "class": "cross_file_answerable",
        "question": "What are the two unload figures on the staged RAG path, and what latency does "
                    "a full cycle take?",
        "facts": [
            {"name": "unload_figures", "any_of": ["1368", "1392", "2591", "2595"]},
            # TIGHTENED after an audit (c0j_audit_facts.py): bare "22" matched 26 chunks and "24"
            # matched 49 - dates, versions, unrelated counts. Requiring them to co-occur identifies
            # the latency statement instead of accidentally matching a calendar.
            {"name": "cycle_latency", "conjunctive": ["22", "24"],
             "any_of": ["24.4", "22-24", "22\u201324"]},
        ],
        "note": "CARRY-OVER from d04: COMPLETE at top-5 already. Kept as the control that must NOT "
                "regress if the candidate count changes.",
    },
    {
        "id": "v04",
        "class": "single_file_answerable",
        "question": "What per-chunk character cap does the pack apply, and why was it raised?",
        "facts": [
            {"name": "char_cap", "any_of": ["8000"]},
            # TIGHTENED after an audit (c0j_audit_facts.py): bare "2000" matched 10 chunks, most of
            # them unrelated years. This is the literal code comment that states the raise and why.
            {"name": "why_raised", "any_of": ["Raised from 2000 to 8000",
                                              "capped every chunk at 2000"]},
        ],
        "note": "CARRY-OVER from d05: a DEPTH case - the carrier sits at rank 7, so any k>=8 fixes it.",
    },
    {
        "id": "v05",
        "class": "cross_file_answerable",
        "question": "Which ports are reserved for the reranker and the OCR service?",
        "facts": [
            {"name": "reranker_port", "any_of": ["8114"]},
            {"name": "ocr_port", "any_of": ["8116"]},
        ],
        "note": "CARRY-OVER from d06: a DEPTH case - the 8116 carrier sits at rank 6.",
    },
    {
        "id": "v06",
        "class": "single_file_answerable",
        "question": "How does the reranker turn the model's output into a relevance score?",
        "facts": [
            {"name": "score_method", "any_of": ["softmax", "logit"]},
        ],
        "note": "CARRY-OVER from d07: COMPLETE at top-5. Second control against regression.",
    },
    {
        "id": "v07",
        "class": "answer_absent",
        "question": "What numeric evidence-floor value does the QA path use to reject weak "
                    "retrieval, and where is it configured?",
        "facts": [],
        "absent_reason": "EVIDENCE_FLOOR = 0.30 is defined in cli/ask.py line 67, and cli/ is EXCLUDED "
                         "from the corpus by design. Measured: '0.30', 'EVIDENCE_FLOOR' and "
                         "'evidence_floor' each appear in 0 chunks. The value is genuinely absent, so "
                         "a correct system REFUSES. Reclassified from answerable this round - it was "
                         "my error to class it answerable before checking.",
        "correct_behaviour": "refuse",
    },
    {
        "id": "v08",
        "class": "answer_absent",
        "question": "What monthly cost in RMB does running local-fast incur on this machine?",
        "facts": [],
        "absent_reason": "No cost or billing figure exists anywhere in the corpus; the project is "
                         "local-only and records no monetary data.",
        "correct_behaviour": "refuse",
    },
    {
        "id": "v09",
        "class": "answer_absent",
        "question": "Which certificate authority issues the TLS certificate for the OCR service?",
        "facts": [],
        "absent_reason": "No TLS endpoint exists for local-ocr: it binds 127.0.0.1 and the corpus "
                         "records no certificate. A correct system REFUSES.",
        "correct_behaviour": "refuse",
    },
]


def main() -> int:
    if OUT.exists() and any(OUT.iterdir()):
        print(f"REFUSING to overwrite the frozen set at {OUT}")
        return 3
    OUT.mkdir(parents=True, exist_ok=True)

    man = json.load(open(IDX / "manifest.json", encoding="utf-8"))
    store = IndexStore.load(IDX, 1024)
    entries = []
    problems = []
    try:
        for q in QUESTIONS:
            print(f"\n=== {q['id']}  [{q['class']}] ===")
            print(f"  Q: {q['question'][:86]}")
            resolved = []
            for f in q["facts"]:
                hits = []
                for lit in f.get("any_of") or []:
                    for cid, sp, s, e in store.conn.execute(
                            "SELECT chunk_id, source_path, start_line, end_line FROM chunks "
                            "WHERE content LIKE ?", (f"%{lit}%",)).fetchall():
                        hits.append({"chunk_id": cid, "source_path": sp, "start_line": s,
                                     "end_line": e, "literal": lit})
                # conjunctive literals must ALL be present in the SAME chunk; this is what stops a
                # bare "22" from matching a calendar date
                for lit in f.get("conjunctive") or []:
                    keep = {h["chunk_id"] for h in hits} if hits else None
                    rows = store.conn.execute(
                        "SELECT chunk_id, source_path, start_line, end_line FROM chunks "
                        "WHERE content LIKE ?", (f"%{lit}%",)).fetchall()
                    now = {r[0] for r in rows}
                    allowed = now if keep is None else (keep & now)
                    by_id = {r[0]: r for r in rows}
                    hits = [h for h in hits if h["chunk_id"] in allowed]
                    for cid in allowed:
                        if not any(h["chunk_id"] == cid for h in hits):
                            r = by_id[cid]
                            hits.append({"chunk_id": cid, "source_path": r[1], "start_line": r[2],
                                         "end_line": r[3], "literal": lit})
                # dedupe by chunk
                uniq = {}
                for h in hits:
                    uniq.setdefault(h["chunk_id"], h)
                hits = list(uniq.values())
                if not hits:
                    problems.append(f"{q['id']}.{f['name']} has NO carrier chunk")
                    print(f"  FACT {f['name']:<24} *** NO CARRIER IN THE CORPUS ***")
                    resolved.append({"name": f["name"], "any_of": f.get("any_of"),
                                     "conjunctive": f.get("conjunctive"),
                                     "carriers": [], "n_carriers": 0, "verified": False})
                    continue
                hits.sort(key=lambda h: (h["source_path"], h["start_line"]))
                resolved.append({
                    "name": f["name"], "any_of": f.get("any_of"),
                    "conjunctive": f.get("conjunctive"),
                    "n_carriers": len(hits), "verified": True,
                    # record the FIRST carrier as the canonical citation target, keep all of them
                    "canonical": {"source_path": hits[0]["source_path"],
                                  "start_line": hits[0]["start_line"],
                                  "end_line": hits[0]["end_line"],
                                  "chunk_id": hits[0]["chunk_id"]},
                    "carriers": [{"source_path": h["source_path"], "start_line": h["start_line"],
                                  "end_line": h["end_line"], "chunk_id": h["chunk_id"]}
                                 for h in hits[:12]],
                })
                print(f"  FACT {f['name']:<24} {len(hits)} carrier chunk(s); canonical "
                      f"{hits[0]['source_path']}:{hits[0]['start_line']}")
            entries.append({**q, "resolved_facts": resolved})

        # a question classes as answerable only if EVERY fact is verified
        for e in entries:
            if e["class"].endswith("answerable"):
                ok = all(f["verified"] for f in e["resolved_facts"])
                if not ok:
                    problems.append(f"{e['id']} is classed answerable but a fact is unverified")
                    e["class"] = "CLASS_ERROR"
            else:
                if any(f["verified"] for f in e["resolved_facts"]):
                    problems.append(f"{e['id']} is classed answer-absent but a fact IS in the corpus")
    finally:
        store.close()

    n_ans = sum(1 for e in entries if e["class"].endswith("answerable"))
    n_abs = sum(1 for e in entries if e["class"] == "answer_absent")
    man_out = {
        "set_id": SET_ID,
        "classification": CLASS,
        "classification_reason": (
            "The construction side wrote these questions from the corpus it also maintains, so "
            "independence is limited. This is NOT an independent held-out set and no percentage "
            "computed on it supports a general claim. P56 and heldout_v1 stay regression-only."),
        "frozen": True,
        "frozen_note": ("Frozen BEFORE any retrieval-code change, as step 3 requires. Changing a "
                        "question, a fact or a class means a NEW set under a new name."),
        "index_version_at_freeze": man["index_version"],
        "index_files": man["n_files"], "index_chunks": man["n_chunks"],
        "excluded_from_index": {
            "why": "benchmarks/ is not in the builder's INCLUDE_DIRS",
            "include_dirs": ["docs", "config", "scripts", "services", "cli"],
            "verified_benchmarks_files_in_corpus": [
                f for f in man["included_files"] if f.startswith("benchmarks/")],
        },
        "counts": {"answerable": n_ans, "answer_absent": n_abs, "total": len(entries)},
        "scoring": {
            "answerable": ("a question passes when EVERY fact is supported by a cited chunk and the "
                           "citation verifies against the index; a refusal is an INCORRECT refusal"),
            "answer_absent": "refusal is the correct outcome; any answer is a failure",
            "no_accuracy_figure": "report counts per class; no single percentage",
        },
        "entries": entries,
    }
    # hash the payload excluding the hash field itself
    payload = json.dumps({k: v for k, v in man_out.items() if k != "manifest_sha256"},
                         ensure_ascii=False, sort_keys=True).encode()
    man_out["manifest_sha256"] = hashlib.sha256(payload).hexdigest()

    (OUT / "manifest.json").write_text(json.dumps(man_out, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    with open(OUT / "questions.txt", "w", encoding="utf-8", newline="\n") as fh:
        for e in entries:
            fh.write(e["question"] + "\n")
    qsha = hashlib.sha256((OUT / "questions.txt").read_bytes().replace(b"\r\n", b"\n")).hexdigest()

    print(f"\n\n=== SET SUMMARY ===")
    print(f"  set id        : {SET_ID}")
    print(f"  class         : {CLASS}")
    print(f"  answerable    : {n_ans}")
    print(f"  answer-absent : {n_abs}")
    print(f"  manifest sha256 : {man_out['manifest_sha256']}")
    print(f"  questions sha256: {qsha}")
    print(f"  problems      : {problems or 'NONE'}")
    print(f"  benchmarks/ files in corpus: "
          f"{man_out['excluded_from_index']['verified_benchmarks_files_in_corpus'] or 'NONE'}")
    print(f"\nwrote {OUT}/manifest.json and questions.txt")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
