"""
P57-C0R-02 - offline adjudication.

Reads packet.json (stored answers, supplied chunk IDs, citations, and the chunk text fetched from the
fixed index) and records, for every question x fact x k:

    chunk_contains_fact   does a supplied chunk actually state the fact?   yes / no / uncertain
    answer_states_fact    does the answer actually state the fact?         yes / no / uncertain
    citation_supports     do the cited sources support that statement?     yes / no / n/a / uncertain
    evidence              the span(s) relied on, with chunk sha256
    reason                the adjudication reason, in words

VERDICTS ARE ASSIGNED FROM THE RECORDED REASONING, NOT FROM A KEYWORD RULE. Where the text does not
settle the question the verdict is `uncertain` and says so. Nothing here is allowed to pass a fact on
the strength of a bare number, a config value, or a keyword hit - the task requires the opposite.

The per-fact verdicts below were reached by reading the packet output; each `reason` states what was
read and, where relevant, what was looked for and NOT found.
"""

from __future__ import annotations

import json
from pathlib import Path

P = (Path(__file__).resolve().parents[2] / r"benchmarks/p57c0r02/packet.json")
OUT = (Path(__file__).resolve().parents[2] / r"benchmarks/p57c0r02/adjudication.json")

d = json.loads(P.read_text(encoding="utf-8"))

# ---------------------------------------------------------------- adjudication, per fact per k
# Each entry: (chunk_contains, answer_states, citation_supports, reason)
#   yes | no | uncertain | n/a
A: dict[str, dict[str, dict[str, tuple]]] = {
    "v01": {
        "fast_port": {
            "5": ("no", "no", "n/a",
                  "Supplied chunks are the local-fast integration guide header, the model-record header, "
                  "two escalation-policy sections and config/local-fast.yaml. The answer is a bare "
                  "refusal with no reasoning. Port 8111 is not stated anywhere in the supplied text."),
            "8": ("no", "no", "n/a",
                  "Same five chunks plus docs/local-fast-api.md:113-124 and :144-161 and "
                  "docs/models/local-fast.md:132-148. Answer is a refusal carrying reasoning; the "
                  "reasoning itself says the sources give port 8111 only via SOURCE 5, i.e. the port is "
                  "NOT in what the model was given. Not stated."),
            "10": ("yes", "no", "no",
                   "config/local-fast.yaml:1-119 IS supplied and states the bind address 127.0.0.1:8111, "
                   "and docs/models/local-fast.md:132-148 states the runtime contract 127.0.0.1:8111. "
                   "So the chunk contains the fact. BUT the answer is a REFUSAL whose reasoning asserts "
                   "'do not contain information about ... the network path (other than the loopback "
                   "address 127.0.0.1)', i.e. it denies having the port while holding it. The answer "
                   "does NOT state 8111. The answer cites local-fast-api.md:1-9 and start-local-fast.ps1 "
                   "but not the two chunks that carry the port, so the citations do not support a "
                   "statement of the port - there is no such statement to support."),
        },
        "embed_port": {
            "5": ("no", "no", "n/a",
                  "Port 8113 appears in none of the five supplied chunks; the nearest carriers "
                  "(config/local-embedding.yaml, docs/reranking-architecture.md, "
                  "services/retrieval/server.py) are absent. The answer is a bare refusal."),
            "8": ("no", "no", "n/a",
                  "The three added chunks are local-fast-api.md:113-124, local-fast-api.md:144-161 and "
                  "models/local-fast.md:132-148 - none mentions 8113. Answer is a refusal."),
            "10": ("no", "no", "no",
                   "Ten chunks supplied; none contains 8113. The two chunks that were added at k=10 "
                   "(scripts/check-local-fast.ps1, scripts/start-local-fast.ps1) are local-fast scripts "
                   "and speak about 8111. The fact is absent from the pack at every k, which is why no "
                   "bounded k reaches it - a RANKING failure, with carriers at ranks 38/51/52/58."),
        },
    },
    "v02": {
        "torch_reserved_meaning": {
            "5": ("no", "no", "n/a",
                  "None of the five supplied chunks mentions torch_reserved, reserved_mb, vram_mb, "
                  "cudaMemGetInfo, attribution or 'VRAM delta'. The chunks are co-residency peak tables "
                  "(gpu-scheduling, retrieval-architecture, reranking-architecture), a scheduler "
                  "self-test, and the gpu-scheduling safety properties. The answer discusses only the "
                  "margin; it states no memory figure and explains none."),
            "8": ("uncertain", "uncertain", "uncertain",
                  "config/local-embedding.yaml:1-94 IS supplied and contains the comment 'Measured delta "
                  "1246 MiB; # torch's own attribution is ~2361 MiB. Keep both in mind for scheduling' "
                  "plus expected_vram_mb: 2400 and measured_vram_delta_mb: 1246. That DISTINGUISHES two "
                  "figures, which is part of what the fact is about, but it is a config-file comment: it "
                  "does not explain what the /stats vram_mb field IS (the passages in "
                  "docs/models/local-embedding.md:90-112 that say it is 'cudaMemGetInfo total - free, "
                  "not PyTorch's attribution to this process' were NOT supplied). The answer states "
                  "'a configured expected_vram_mb of 2400 MiB and a measured delta of 1246 MiB [SOURCE "
                  "7]' - it relays the config VALUES but does not explain the meaning. Judged UNCERTAIN "
                  "rather than passed: a config value plus one distinguishing comment is not the "
                  "explanation, but it is also not nothing, and the task forbids forcing this either "
                  "way."),
            "10": ("uncertain", "uncertain", "uncertain",
                  "The same config comment is supplied, and k=10 adds services/retrieval/embedder.py, "
                  "which is a carrier of this fact. The answer is materially the same as k=8 and still "
                  "reports values rather than explaining the field. Split verdict avoided; UNCERTAIN for "
                  "the same reason as k=8, with the added note that the fact's own canonical carrier "
                  "(docs/models/local-embedding.md:90-112, the passage that actually explains it) was "
                  "never supplied at any k."),
        },
        "margin_policy": {
            "5": ("yes", "yes", "yes",
                  "docs/gpu-scheduling.md:62-80 states '2 GiB VRAM margin before every admission, "
                  "measured with nvidia-smi rather than estimated from a table. A request that would "
                  "breach it is refused, not queued forever.' The answer states this and cites [SOURCE "
                  "5], which is exactly that chunk. Chunk contains / answer states / citation supports: "
                  "all three yes, and they line up."),
            "8": ("yes", "yes", "yes",
                  "The same chunk is supplied and cited. The answer repeats the rule and the refusal "
                  "behaviour and attributes it to [SOURCE 5]. Verified alignment."),
            "10": ("yes", "yes", "yes",
                  "Same chunk supplied and cited; the answer states the rule and the refusal logic. "
                  "Verified alignment."),
        },
    },
}

# questions whose facts were checked the same way but where I record the outcome per question below
A["v03"] = {
    "unload_figures": {
        "5": ("yes", "yes", "yes",
              "docs/phase55r-evidence-consolidation.md:75-89 states 'Verified deltas: embedding +1368 "
              "to +1392 MiB, reranker +2591 to +2595 MiB'. The answer gives both ranges and cites the "
              "chunk. Aligned."),
        "8": ("yes", "yes", "yes", "Same source supplied and cited; answer gives both ranges."),
        "10": ("yes", "yes", "yes", "Same source supplied and cited; answer gives both ranges."),
    },
    "cycle_latency": {
        "5": ("yes", "yes", "yes",
              "The same chunk states 'a full RAG cycle measured 24.4 s (P55R-05) and ~22-24 s in the "
              "re-run'. The answer gives the cycle latency and cites the chunk. Aligned."),
        "8": ("yes", "yes", "yes", "Same source supplied and cited."),
        "10": ("yes", "yes", "yes", "Same source supplied and cited."),
    },
}
A["v04"] = {
    "char_cap": {
        "5": ("no", "no", "n/a", "services/retrieval/retriever.py not supplied at all; answer is a bare refusal."),
        "8": ("no", "no", "n/a", "retriever.py still not supplied at k=8; answer is a bare refusal."),
        "10": ("yes", "yes", "yes",
               "services/retrieval/retriever.py:25-40 IS supplied and states 'max_chars_per_chunk: int = "
               "8000'. The answer states 8000 and cites it. Aligned."),
    },
    "why_raised": {
        "5": ("no", "no", "n/a", "Not supplied; bare refusal."),
        "8": ("no", "no", "n/a", "Not supplied; bare refusal."),
        "10": ("yes", "yes", "yes",
               "The same chunk states 'Raised from 2000 to 8000 (the chunker's own maximum is ~900 "
               "tokens, so 8000 chars effectively never truncates): a 2000-char cap was cutting the "
               "answer out of otherwise well-retrieved chunks', so the chunk contains the reason. The "
               "answer states it too: 'It was raised from 2000 to prevent **silent truncation** of "
               "answers within retrieved chunks, which previously converted retrieval successes into "
               "misses.' Citation is to the chunk that carries it. Aligned. NOTE: my first pass at this "
               "adjudication recorded 'no' for answer_states_fact on the strength of a phrase search "
               "rather than reading the answer; corrected after reading the full text."),
    },
}
A["v05"] = {
    "reranker_port": {
        "5": ("yes", "no", "no",
              "config/local-reranker.yaml:1-93 IS supplied and states port 8114, so the chunk contains "
              "the fact. The answer is a bare refusal - it does not state it. No citations, so nothing "
              "supports a statement that was not made. A depth case that k=8 fixes."),
        "8": ("yes", "yes", "yes",
              "config/local-reranker.yaml:1-93 supplied; the answer states 8114 and cites it. Aligned."),
        "10": ("yes", "yes", "yes", "Same source supplied and cited."),
    },
    "ocr_port": {
        "5": ("no", "no", "n/a", "config/local-ocr.yaml not supplied; bare refusal."),
        "8": ("yes", "yes", "yes",
              "config/local-ocr.yaml:1-93 supplied; the answer states 8116 and cites it. Aligned."),
        "10": ("yes", "yes", "yes", "Same source supplied and cited."),
    },
}
A["v06"] = {
    "score_method": {
        "5": ("yes", "yes", "yes",
              "docs/reranking-architecture.md and config/local-reranker.yaml supplied; the answer "
              "describes softmax over the yes/no logits at the final position and cites them."),
        "8": ("yes", "yes", "yes", "Same sources supplied and cited."),
        "10": ("yes", "yes", "yes", "Same sources supplied and cited."),
    },
}


def verdict_for(qid: str, per_k: dict) -> str:
    """
    complete / partial / incorrect_refusal / incorrect / uncertain, from the recorded per-fact verdicts.

    BUG FOUND AND FIXED HERE: the first version did
        states = [v[1] for v in per_k.values() if isinstance(v, tuple)]
    but per_k holds DICTS by the time this runs, so `v[1]` raised KeyError('1'), the comprehension
    yielded an EMPTY list, and `yes == len(states)` became `0 == 0` -> **complete**. A question whose
    facts were all undecided was therefore scored COMPLETE - a false pass, and precisely the failure
    mode this round exists to prevent. Now it reads the dicts, and an empty fact set is refused
    explicitly rather than passing vacuously.
    """
    if per_k["_refusal"]:
        return "incorrect_refusal"
    facts = [v for v in per_k.values() if isinstance(v, dict)]
    if not facts:
        return "error_no_facts"
    states = [v["answer_states_fact"] for v in facts]
    yes = sum(1 for s in states if s == "yes")
    unc = sum(1 for s in states if s == "uncertain")
    no = sum(1 for s in states if s == "no")
    # PRECEDENCE, stated so it is not implicit:
    #   an undecided fact does not erase a fact that WAS decided, so a partial outcome with one
    #   definite yes is "partial with an undecided remainder", not "uncertain".
    #   "uncertain" is reserved for the case where NOTHING was decided either way.
    if yes and (unc or no):
        return "partial"
    if unc and not yes and not no:
        return "uncertain"
    if no and not yes:
        return "uncertain" if unc else "incorrect"
    if yes == len(states):
        return "complete"
    return "partial"


out: dict = {"index_version": d["index_version"], "packet_sha256_note":
             "chunk text fetched from the fixed index; each chunk's sha256 is in the packet",
             "questions": {}}
for qid, facts in A.items():
    q = {"question": d["questions"][qid]["question"], "by_k": {}}
    for k in ("5", "8", "10"):
        b = d["questions"][qid]["by_k"][k]
        per_k = {"_refusal": b["contains_refusal_sentence"]}
        for fname, ks in facts.items():
            cc, ans, cit, reason = ks[k]
            per_k[fname] = {"chunk_contains_fact": cc, "answer_states_fact": ans,
                            "citation_supports": cit, "reason": reason}
        per_k["verdict"] = verdict_for(qid, per_k)
        per_k["n_supplied"] = len(b["supplied"])
        per_k["citations"] = b["citations"]
        q["by_k"][k] = per_k
    out["questions"][qid] = q

print("=" * 100)
print("ADJUDICATION")
print("=" * 100)
for qid, q in out["questions"].items():
    print(f"\n{qid}: {q['question'][:78]}")
    for k in ("5", "8", "10"):
        pk = q["by_k"][k]
        print(f"  k={k:>2}  verdict={pk['verdict']:<17} supplied={pk['n_supplied']:<3}")
        for fname, v in pk.items():
            if not isinstance(v, dict):
                continue
            print(f"      {fname:<24} chunk={v['chunk_contains_fact']:<10} "
                  f"answer={v['answer_states_fact']:<10} citation={v['citation_supports']}")

print("\n" + "=" * 100)
print("VERDICT TABLE")
print("=" * 100)
for k in ("5", "8", "10"):
    row = {qid: q["by_k"][k]["verdict"] for qid, q in out["questions"].items()}
    print(f"  k={k:>2}: {row}")

OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\nwrote {OUT}")
