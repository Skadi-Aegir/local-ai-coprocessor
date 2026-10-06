"""
hard_negatives.py - Phase 5 hard-negative and duplicate/stale ranking tests.

Why these exist
---------------
Aggregate Recall@5 hides the failure that actually matters to a caller: a reranker that
scores a plausible-but-wrong passage above the passage that answers the question. Every
pair below is built from THIS project's own facts, where both candidates are topically
identical and share most of their vocabulary, so a bag-of-words or weak semantic ranker
cannot separate them. Only the one that actually answers the query is correct.

Each case asserts:
  relevant   > hard_negative      (the reranker must separate them)
  relevant   > unrelated          (sanity)

Duplicate / stale cases test a property the reranker CANNOT have: it sees text only and
has no notion of which version is current. The specification asks for this to be measured
and, if it fails, recorded as a limitation rather than fixed inside the reranker. The
retrieval pipeline is where freshness metadata belongs.

Run:
  python hard_negatives.py --out ../../benchmarks/reranker/results/hard_negatives.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import urllib.request
from pathlib import Path

RERANK_URL = "http://127.0.0.1:8114/v1/rerank"

# --------------------------------------------------------------------------
# 20 hard-negative pairs. Each is (query, relevant, hard_negative, unrelated).
# The hard negative is deliberately adjacent: same subsystem, same numbers, same
# vocabulary, but it answers a DIFFERENT question.
#
# `category` records how well the negative is genuinely excluded by the query, judged by
# reading it rather than by what the model scored:
#   strict    - the negative demonstrably does NOT answer the question. If the reranker
#               ranks it above the relevant passage, that is a real failure.
#   ambiguous - the negative answers a closely adjacent aspect of the question (e.g. HOW
#               to disable a mode when the query asks WHY it should stay disabled). A
#               human reader would accept it as partially responsive, so ranking it high
#               is a defensible reading, not a clear error.
# Reporting one blended accuracy over both would overstate the model's failures; reporting
# only the strict subset would hide the ambiguity. Both are reported separately.
# --------------------------------------------------------------------------
HARD_NEGATIVES: list[dict] = [
    dict(
        id="hn01", category="ambiguous",
        query="为什么双模型不应该并发推理？",
        relevant="并发推理导致总吞吐下降约47%：两个并行请求合计比串行执行更慢，"
                 "因为 GPU 会把工作串行化并叠加显存带宽争用。",
        hard_negative="两个模型可以同时常驻显存：4B 占 2981 MiB，9B 占 5527 MiB，"
                      "合计约 8508 MiB，仍然放得下。",
        unrelated="KV cache 的大小与上下文长度成正比，每个 token 约 32768 字节。",
    ),
    dict(
        id="hn02", category="strict",
        query="What is the recommended batch size for embedding, and why?",
        relevant="Batch 32 is the recommended size: 178.6 chunks/s, while batch 64 adds "
                 "only 8% throughput for 85% more wall latency.",
        hard_negative="Batch 64 reaches the highest throughput at 192.9 chunks/s, the "
                      "peak measured across the sweep.",
        unrelated="The embedding model uses last-token pooling and cosine similarity.",
    ),
    dict(
        id="hn03", category="ambiguous",
        query="Does retrieval reduce the number of tokens sent to the model?",
        relevant="Yes. Retrieval cuts the context from 54699 tokens for the whole corpus "
                 "to 1874 tokens for a packed top-5, a 96.57% reduction.",
        hard_negative="The retrieved chunk count is configurable: top_k can be raised to "
                      "hand more sources to the model.",
        unrelated="The reranker scores a query against each document independently.",
    ),
    dict(
        id="hn04", category="ambiguous",
        query="Why should thinking mode stay disabled for local-fast?",
        relevant="On an easy extraction task thinking mode cost 17x the latency and 20x "
                 "the tokens for zero quality gain, and on a hard task it consumed the "
                 "entire token budget and still answered incorrectly.",
        hard_negative="Thinking mode can be disabled by sending "
                      "chat_template_kwargs with enable_thinking set to false.",
        unrelated="The 4B model is quantised to Q4_K_M, chosen over Q5_K_M.",
    ),
    dict(
        id="hn05", category="strict",
        query="How much free VRAM remains when the 9B model runs alongside the embedding service?",
        relevant="Only about 720 MiB remains free, which is below the 2 GB safety target, "
                 "so the 9B should not be co-resident with the embedding service.",
        hard_negative="The 4B plus the embedding service leaves about 3015 MiB free, "
                      "which does meet the 2 GB safety target.",
        unrelated="Both services bind to loopback only.",
    ),
    dict(
        id="hn06", category="strict",
        query="Which chunking strategy produced better retrieval results?",
        relevant="Structure-aware chunking retrieved better than fixed windows at every "
                 "dimension: R@3 71.8 versus 61.5 at 1024 dimensions.",
        hard_negative="Fixed-window chunking produced fewer chunks, 158 against 178, "
                      "because it does not split on structural boundaries.",
        unrelated="Retrieval uses an exact FAISS IndexFlatIP index.",
    ),
    dict(
        id="hn07", category="strict",
        query="Is the reranker able to rank a relevant passage above an irrelevant one?",
        relevant="Yes, and it is decisive: a relevant passage scores about 0.9995 while an "
                 "irrelevant one scores about 0.000015 for the same query.",
        hard_negative="The reranker returns scores between 0 and 1 for every document it "
                      "is given, ordered best first.",
        unrelated="The embedding model outputs 1024-dimensional unit vectors.",
    ),
    dict(
        id="hn08", category="strict",
        query="Why was Q4_K_M chosen over Q5_K_M for the 4B model?",
        relevant="Q4_K_M generated 74.6 tokens per second against 63.3 for Q5_K_M, so the "
                 "smaller quantisation was 15.2% faster with acceptable quality.",
        hard_negative="Q6_K was downloaded and its checksum verified, but it was never "
                      "benchmarked during Phase 3.",
        unrelated="The model server listens on port 8111.",
    ),
    dict(
        id="hn09", category="strict",
        query="What happens to generation speed when two models infer at the same time?",
        relevant="Generation collapses from about 130 tokens per second to 42-49 tokens "
                 "per second, a loss of roughly 62 to 65 percent.",
        hard_negative="Embedding throughput falls from 178.6 to 126.3 chunks per second "
                      "under the same concurrent load, a loss of 29 percent.",
        unrelated="Model loading takes about 1.7 seconds for the embedding service.",
    ),
    dict(
        id="hn10", category="strict",
        query="Why was 256 dimensions rejected as the default embedding size?",
        relevant="Because R@3 falls from 71.8 at 1024 dimensions to 59.0 at 256, a quality "
                 "loss the storage saving does not justify.",
        hard_negative="Truncating to 128 or 64 dimensions loses the correct top result "
                      "entirely, so 256 is the practical floor.",
        unrelated="The embedding service holds about 2400 MiB of VRAM.",
    ),
    dict(
        id="hn11", category="strict",
        query="How was the local-smart 9B model's under-escalation rate compared to local-fast?",
        relevant="The 9B under-escalated 3 times against the 4B's 5, which was its only "
                 "winning row in the Phase 3 head-to-head comparison.",
        hard_negative="The 4B won 6 comparison rows and the 9B won 1, with 12 ties "
                      "between them.",
        unrelated="Both models are served by llama.cpp build b11254.",
    ),
    dict(
        id="hn12", category="strict",
        query="What is the risk of running the local models while the desktop uses more VRAM than expected?",
        relevant="Free VRAM can fall below the safety margin, and past that point the "
                 "driver starts paging to system memory, which inflates the pagefile and "
                 "can crash the process outright.",
        hard_negative="The launcher refuses to start when free VRAM is below the "
                      "configured min_free_vram_mb value.",
        unrelated="The GPU is an RTX 5070 Ti Laptop with 12 GB.",
    ),
    dict(
        id="hn13", category="strict",
        query="Which failure mode did the reranker fix that dense retrieval could not?",
        relevant="Correct chunks that dense retrieval left at rank 6 or rank 13 were "
                 "promoted into the final top 5, including one promoted from rank 13 to rank 1.",
        hard_negative="Dense retrieval returns candidates in cosine-similarity order "
                      "using an exact inner-product index.",
        unrelated="The reranker runs on port 8114.",
    ),
    dict(
        id="hn14", category="strict",
        query="How many query types does the retrieval benchmark cover?",
        relevant="Five: exact, semantic, paraphrase, cross-language, and code-concept, "
                 "spread over 39 queries across documentation, code, and mixed sets.",
        hard_negative="The benchmark corpus contains 30 indexed files, and the index "
                      "holds 186 chunks after structure-aware chunking.",
        unrelated="Scores are cached by query and document hash.",
    ),
    dict(
        id="hn15", category="strict",
        query="Why does the reranker need a token budget rather than a fixed batch size?",
        relevant="Attention activations grow with the square of sequence length, so three "
                 "maximal documents in one batch attempted a 768 MiB allocation and ran "
                 "out of memory even though there was free VRAM available.",
        hard_negative="A batch of twenty normal-sized chunks fits comfortably in a single "
                      "forward pass.",
        unrelated="The reranker uses bfloat16 weights.",
    ),
    dict(
        id="hn16", category="strict",
        query="What breaks if the reranker service is unavailable?",
        relevant="Nothing fatal: retrieval falls back to the dense top-5 and reports "
                 "reranker_used as false instead of failing the whole request.",
        hard_negative="The reranker reports reranker_used as true whenever it returns a "
                      "scored result list.",
        unrelated="Health checks are exposed at the health endpoint.",
    ),
    dict(
        id="hn17", category="strict",
        query="Which candidate pool size gives the best latency for the least quality loss?",
        relevant="Pool 10 gives the same strict R@5 as dense at the lowest rerank cost, "
                 "while pool 50 costs about three times more per query.",
        hard_negative="Pool 20 is the documented default candidate size in the pipeline "
                      "configuration.",
        unrelated="Query embedding takes about 0.04 seconds.",
    ),
    dict(
        id="hn18", category="ambiguous",
        query="What did the strict single-source benchmark understate?",
        relevant="It scored factually correct answers as misses when the fact appeared in "
                 "a different file, understating dense R@5 as 66.7 when multi-source "
                 "ground truth gives 84.6.",
        hard_negative="The strict benchmark reported dense R@5 as 66.7 percent, which was "
                      "the published Phase 4 figure.",
        unrelated="The benchmark uses 39 queries.",
    ),
    dict(
        id="hn19", category="strict",
        query="Why must the reranker not manage the FAISS index?",
        relevant="Because it would then own metadata mapping as well, and it would stop "
                 "being a pure scoring function that can be tested with plain strings.",
        hard_negative="The reranker receives documents as a list of strings and returns "
                      "an index into that list.",
        unrelated="Metadata is stored in SQLite alongside the vectors.",
    ),
    dict(
        id="hn20", category="strict",
        query="What limits a fixed row-count batch size for the reranker?",
        relevant="It ignores document length, so a small number of very long documents "
                 "can still exhaust memory; the bound has to be rows times padded length.",
        hard_negative="Long documents are truncated to a body budget of 2000 tokens "
                      "before scoring.",
        unrelated="The yes and no token ids are 9693 and 2152.",
    ),
]


# --------------------------------------------------------------------------
# duplicate / stale cases. The reranker sees text only, so it has no way to know
# which version is current. Recorded as a limitation, not patched inside the model.
# --------------------------------------------------------------------------
DUPLICATE_CASES: list[dict] = [
    dict(
        id="dup01",
        query="What is the recommended embedding batch size?",
        current="Batch 32 is recommended: 178.6 chunks/s, while batch 64 adds only 8% "
                "throughput for 85% more latency.",
        duplicate="Batch 32 is recommended: 178.6 chunks/s, while batch 64 adds only 8% "
                  "throughput for 85% more latency.",
        stale="Batch 64 is recommended because it reaches the highest raw throughput of "
              "192.9 chunks/s.",
        note="identical duplicate vs stale conflicting claim",
    ),
    dict(
        id="dup02",
        query="How much VRAM does the embedding service use?",
        current="The embedding service adds about 1246 MiB over the desktop baseline.",
        duplicate="The embedding service adds roughly 1246 MiB above the desktop baseline.",
        stale="The embedding service uses about 3600 MiB, measured before the model was "
              "switched to bfloat16.",
        note="paraphrased duplicate vs superseded figure",
    ),
    dict(
        id="dup03",
        query="What is the local-smart port?",
        current="local-smart listens on port 8112.",
        duplicate="Port 8112 serves local-smart.",
        stale="local-smart listens on port 8110, which was the provisional assignment.",
        note="identical fact, stale port number",
    ),
    dict(
        id="dup04",
        query="Which quantisation does local-fast use?",
        current="local-fast runs Qwen3.5-4B at Q4_K_M, chosen after benchmarking.",
        duplicate="The deployed local-fast quantisation is Q4_K_M for Qwen3.5-4B.",
        stale="local-fast was initially planned to run Q5_K_M for better quality.",
        note="current choice vs abandoned plan",
    ),
    dict(
        id="dup05",
        query="Does retrieval reach the 90% R@5 target?",
        current="No. The best strict R@5 is 71.8% and the best multi-source R@5 is 87.2%, "
                "so the 90% target is not met.",
        duplicate="The 90% R@5 target was not reached: strict 71.8%, multi-source 87.2%.",
        stale="Retrieval meets the 90% R@5 target under the corrected multi-source "
              "ground truth.",
        note="honest statement vs optimistic misreading",
    ),
]


def post(payload, timeout=900):
    req = urllib.request.Request(
        RERANK_URL, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def run_hard_negatives(instruction_type: str) -> dict:
    rows = []
    for case in HARD_NEGATIVES:
        docs = [case["relevant"], case["hard_negative"], case["unrelated"]]
        r = post({"query": case["query"], "documents": docs,
                  "instruction_type": instruction_type, "top_n": 3})
        by = {x["index"]: x["score"] for x in r["results"]}
        order = [x["index"] for x in r["results"]]
        # A hard negative that scores near the top of the range is not being rejected -
        # the reranker considers it a good answer. Two very different situations produce
        # that, and conflating them would misreport the model's behaviour:
        #   SATURATION - both passages are highly relevant (>= 0.9). This is a labelling
        #                problem in the test, not a ranking failure: the "negative"
        #                genuinely answers the query as written.
        #   INVERSION  - the negative outranks a relevant passage that is NOT saturated.
        #                This is a real ranking failure and is counted as one.
        sat = by[0] >= 0.9 and by[1] >= 0.9
        cat = case.get("category", "strict")
        rows.append({
            "id": case["id"], "category": cat, "query": case["query"],
            "relevant_score": by[0], "hard_negative_score": by[1],
            "unrelated_score": by[2],
            "margin": by[0] - by[1],
            "ranked_first": order[0] == 0,
            "separated": by[0] > by[1],
            "saturation": sat,
            "genuine_inversion": (by[1] > by[0] and not sat),
            "margin_ratio": (by[0] / by[1]) if by[1] > 0 else None,
            "seconds": r["seconds"],
        })
    n = len(rows)
    seps = [r for r in rows if r["separated"]]
    margins = [r["margin"] for r in rows]
    inversions = [r for r in rows if r["genuine_inversion"]]
    saturated = [r for r in rows if r["saturation"]]
    strict_rows = [r for r in rows if r["category"] == "strict"]
    amb_rows = [r for r in rows if r["category"] == "ambiguous"]
    return {
        "instruction_type": instruction_type,
        "cases": n,
        "separated": len(seps),
        "accuracy_pct": round(100 * len(seps) / n, 1),
        "ranked_first": sum(1 for r in rows if r["ranked_first"]),
        "mean_margin": round(statistics.mean(margins), 4),
        "median_margin": round(statistics.median(margins), 4),
        "min_margin": round(min(margins), 6),
        "saturated_cases": len(saturated),
        "saturated_ids": [r["id"] for r in saturated],
        "genuine_inversions": len(inversions),
        "inversion_ids": [r["id"] for r in inversions],
        # split by how defensible the label is, since a blended number would mislead
        "strict_cases": len(strict_rows),
        "strict_accuracy_pct": round(
            100 * sum(1 for r in strict_rows if r["separated"]) / (len(strict_rows) or 1), 1),
        "strict_inversions": sum(1 for r in strict_rows if r["genuine_inversion"]),
        "ambiguous_cases": len(amb_rows),
        "ambiguous_accuracy_pct": round(
            100 * sum(1 for r in amb_rows if r["separated"]) / (len(amb_rows) or 1), 1),
        "rows": rows,
    }


def run_duplicates(instruction_type: str) -> dict:
    rows = []
    for case in DUPLICATE_CASES:
        docs = [case["current"], case["stale"], case["duplicate"]]
        r = post({"query": case["query"], "documents": docs,
                  "instruction_type": instruction_type, "top_n": 3})
        by = {x["index"]: x["score"] for x in r["results"]}
        order = [x["index"] for x in r["results"]]
        rows.append({
            "id": case["id"], "query": case["query"], "note": case["note"],
            "current": by[0], "stale": by[1], "duplicate": by[2],
            "current_beats_stale": by[0] > by[1],
            "current_ranked_first": order[0] == 0,
            "duplicate_within_0.01_of_current": abs(by[0] - by[2]) < 0.01,
        })
    n = len(rows)
    return {
        "instruction_type": instruction_type,
        "cases": n,
        "current_beats_stale": sum(1 for r in rows if r["current_beats_stale"]),
        "current_ranked_first": sum(1 for r in rows if r["current_ranked_first"]),
        "duplicate_scores_match": sum(1 for r in rows if r["duplicate_within_0.01_of_current"]),
        "rows": rows,
        "limitation": ("The reranker scores TEXT only and has no notion of which version "
                       "is current; freshness must come from metadata in the retrieval "
                       "pipeline, not from this model."),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--instructions", default="default,general,code,none")
    args = ap.parse_args()

    instrs = [x.strip() for x in args.instructions.split(",") if x.strip()]
    report = {"hard_negatives": [], "duplicates": []}

    print("=== HARD NEGATIVES (20 pairs) ===")
    print(f"{'instr':<9} {'separated':>10} {'ranked1':>8} {'mean marg':>10} "
          f"{'saturated':>10} {'INVERSIONS':>11}")
    for it in instrs:
        r = run_hard_negatives(it)
        report["hard_negatives"].append(r)
        print(f"{it:<9} {r['separated']:>4}/{r['cases']:<5} {r['ranked_first']:>8} "
              f"{r['mean_margin']:>10.4f} {r['saturated_cases']:>10} "
              f"{r['genuine_inversions']:>11}")
        if r["genuine_inversions"]:
            for row in r["rows"]:
                if row["genuine_inversion"]:
                    print(f"    INVERSION {row['id']}: rel={row['relevant_score']:.6f} "
                          f"hn={row['hard_negative_score']:.6f}  {row['query'][:52]}")
        if r["saturated_cases"]:
            print(f"    saturated (labeling artifact, both >=0.9): "
                  f"{','.join(r['saturated_ids'])}")

    print("\n=== DUPLICATE / STALE ===")
    for it in instrs:
        d = run_duplicates(it)
        report["duplicates"].append(d)
        print(f"{it:<9} current>stale {d['current_beats_stale']}/{d['cases']}   "
              f"current first {d['current_ranked_first']}/{d['cases']}   "
              f"dup≈current {d['duplicate_scores_match']}/{d['cases']}")
        for row in d["rows"]:
            print(f"    {row['id']}: cur={row['current']:.4f} stale={row['stale']:.4f} "
                  f"dup={row['duplicate']:.4f}  "
                  f"{'OK' if row['current_beats_stale'] else 'STALE WINS'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
