"""
p2_decon_pair.py - P55R-02 item 3: paired contamination validation.

Answers exactly one question: what does RETRIEVABILITY OF THE CONTAMINATED MATERIAL change about
the CURRENT FIXED PIPELINE?

What is held FIXED (frozen):
  * the same 12 questions
  * the same Dense arm and the same top_k = 5
  * the same embedding model and query instruction
  * the same candidate pool, the same reranker settings, the same pack shape
  * the same generation settings (temperature 0, fixed max_tokens, thinking disabled)
  * the same index IDENTITY apart from the excluded chunks: decon_v1 is a byte-for-byte copy of
    the frozen index minus the 112 contaminated chunks, built by reconstructing every vector

What VARIES: only whether the contaminated material is retrievable.

WHAT THIS IS NOT, stated so it cannot be misread:
  * not a reproduction of historical causality - the historical Phase 4/5 difference also involved
    a different pack PATH (see OPEN_ISSUES P1-2)
  * not an independent generalisation test - the questions are the original published set and are
    therefore exposed
  * it does NOT assume accuracy must improve. The measurement is reported with whatever sign it has.

Fact checking is per-fact presence in the PACK, and hallucination is NOT declared on keyword
presence: the report records the raw fact strings found and the forbidden values found, and any
disputed record is listed for fact-level review rather than scored as a verdict.

Usage:
  python p2_decon_pair.py --out results/<run>/decon_pair.json [--limit N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "retrieval"))

from index_store import IndexStore                    # noqa: E402
from retriever import RetrieveConfig, Retriever        # noqa: E402

EMBED = "http://127.0.0.1:8113"
RERANK = "http://127.0.0.1:8114"
FAST = "http://127.0.0.1:8111"

FROZEN = REPO / "vector/embedding/local-ai"
DECON = REPO / "cache/retrieval_bench/indexes/decon_v1"
ARMS = {"full": FROZEN, "decon": DECON}
TOP_K = 5
POOL = 10


def post(url, payload, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def load_questions() -> list[dict]:
    """The ORIGINAL 12 questions with their expected facts and forbidden values."""
    sys.path.insert(0, str(REPO / "services/retrieval"))
    import importlib
    mod = importlib.import_module("rag_compare")
    for name in ("TASKS", "CASES", "QUESTIONS", "RAG_CASES", "CASES_ALL"):
        cases = getattr(mod, name, None)
        if cases:
            out = []
            for c in cases:
                if isinstance(c, dict) and "q" in c:
                    out.append({"q": c["q"],
                                "facts": c.get("facts") or [],
                                "bad": c.get("bad") or []})
            if out:
                return out[:12]
    raise SystemExit("could not locate the question list in rag_compare.py")


def fact_in_text(fact: str, text: str) -> bool:
    return fact.lower() in text.lower()


_EMB = None


def get_embedder():
    """
    One Embedder for the whole run. Constructing one per arm would load the model twice, and
    this project has already been bitten by an Embedder leak exhausting the pagefile.
    """
    global _EMB
    if _EMB is None:
        from embedder import Embedder
        _EMB = Embedder()
        print(f"  embedder ready: {_EMB.info()}", flush=True)
    return _EMB


def run_arm(name: str, index_dir: Path, questions: list[dict], out_dir: Path) -> list[dict]:
    print(f"\n=== arm {name}: {index_dir} ===", flush=True)
    store = IndexStore.load(index_dir, 1024)
    stats = store.stats()
    print(f"  index: {stats}", flush=True)
    frames = []

    # fact presence ANYWHERE in this index - separates "corpus never had it" from "retrieval
    # did not find it", which is the conflation flagged in P1-4
    corpus_text = "\n".join(
        r[0] or "" for r in store.conn.execute("SELECT content FROM chunks").fetchall())

    for i, q in enumerate(questions):
        row: dict = {"arm": name, "i": i, "q": q["q"],
                     "index": str(index_dir), "index_stats": stats}
        try:
            retr = Retriever(get_embedder(), store,
                             RetrieveConfig(top_k=TOP_K, instruction=None))
            hits = retr.retrieve(q["q"], top_k=POOL)
            cands = [h.to_dict() for h in hits][:POOL]

            # rerank the pool down to TOP_K, using the real reranker service
            docs = [c.get("content", "") for c in cands]
            reranked_ok = False
            try:
                rr = post(f"{RERANK}/v1/rerank",
                          {"query": q["q"], "documents": docs, "top_n": TOP_K,
                           "instruction_type": "general", "use_cache": False})
                order = [x.get("index") for x in (rr.get("results") or [])]
                final = [cands[j] for j in order if isinstance(j, int) and j < len(cands)]
                reranked_ok = bool(final)
                if not final:
                    final = cands[:TOP_K]
                row["rerank_scores"] = [round(x.get("relevance_score", 0), 4)
                                        for x in (rr.get("results") or [])]
            except Exception as e:  # noqa: BLE001
                row["rerank_error"] = f"{type(e).__name__}: {e}"
                final = cands[:TOP_K]

            pack = "\n\n".join(
                f"SOURCE {k+1} ({c.get('source_path')} lines {c.get('start_line')}-"
                f"{c.get('end_line')})\n{(c.get('content') or '')[:2000]}"
                for k, c in enumerate(final))

            prompt = (f"{pack}\n\nAnswer the question using only the sources above.\n"
                      f"QUESTION: {q['q']}")
            t_gen = time.time()
            g = post(f"{FAST}/v1/chat/completions", {
                "model": "local-fast",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 192, "temperature": 0.0,
                "chat_template_kwargs": {"enable_thinking": False}})
            gen_s = time.time() - t_gen
            answer = (g.get("choices") or [{}])[0].get("message", {}).get("content", "")

            row.update({
                "candidates": cands,
                "reranked": reranked_ok,
                "pack": pack,
                "pack_chars": len(pack),
                "n_sources": len(final),
                "source_paths": [c.get("source_path") for c in final],
                "answer": answer,
                "answer_chars": len(answer),
                "gen_s": round(gen_s, 2),
                "prompt_tokens": (g.get("usage") or {}).get("prompt_tokens"),
            })

            # per-fact evidence, recorded NOT scored as a verdict.
            #
            # SEMANTICS, and this matters: a fact group like ["8192", "8k"] means EITHER string
            # is an acceptable way to state the fact - it is an OR of alternatives, not a
            # requirement to contain both. An earlier version joined the group and required the
            # literal string "8192 8k", which can never appear, so every fact scored as missing
            # and the run looked like a catastrophic retrieval failure. This is the same
            # OR-versus-AND confusion recorded in OPEN_ISSUES P2-1/P2-5.
            fact_rows = []
            for grp in q["facts"]:
                alts = list(grp) if isinstance(grp, (list, tuple)) else [grp]
                fact_rows.append({
                    "fact": " | ".join(alts),
                    "alternatives": alts,
                    "in_pack": any(fact_in_text(a, pack) for a in alts),
                    "in_answer": any(fact_in_text(a, answer) for a in alts),
                    "anywhere_in_index": any(fact_in_text(a, corpus_text) for a in alts),
                    "in_any_candidate": any(fact_in_text(a, c.get("content") or "")
                                            for c in cands for a in alts),
                })
            bad_rows = []
            for grp in q["bad"]:
                alts = list(grp) if isinstance(grp, (list, tuple)) else [grp]
                bad_rows.append({"value": " | ".join(alts),
                                 "appears_in_answer": any(fact_in_text(a, answer) for a in alts),
                                 "appears_in_pack": any(fact_in_text(a, pack) for a in alts)})
            row["facts"] = fact_rows
            row["forbidden"] = bad_rows
            row["all_facts_in_answer"] = bool(fact_rows) and all(f["in_answer"]
                                                                for f in fact_rows)
            row["answer_in_corpus_at_all"] = any(f["anywhere_in_index"] for f in fact_rows)
            print(f"  [{i+1}/{len(questions)}] sources={len(final)} "
                  f"facts_in_answer={sum(1 for f in fact_rows if f['in_answer'])}/"
                  f"{len(fact_rows)} corpus_has_answer={row['answer_in_corpus_at_all']}",
                  flush=True)
        except Exception as e:  # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"  [{i+1}/{len(questions)}] ERROR {row['error'][:90]}", flush=True)
        frames.append(row)

    store.close()
    (out_dir / f"arm_{name}.json").write_text(
        json.dumps(frames, ensure_ascii=False, indent=2), encoding="utf-8")
    return frames


def fact_in_text_any(keys: list[str], text: str) -> bool:
    low = text.lower()
    return all(k.lower() in low for k in keys)


def _paired_summary(results: dict) -> dict:
    """
    Build the paired comparison from the two arms.

    Separate function so an error here cannot abort the run before the raw per-question records
    are written. A first version inlined this and an exception destroyed 24 real generation
    calls.
    """
    diffs, improved, worsened = [], [], []
    for a, b in zip(results["full"], results["decon"]):
        d = {"q": a["q"]}
        d["facts_full"] = sum(1 for f in a.get("facts", []) if f["in_answer"])
        d["facts_decon"] = sum(1 for f in b.get("facts", []) if f["in_answer"])
        d["n_facts"] = len(a.get("facts", []))
        d["candidate_overlap"] = len(
            set(c["chunk_id"] for c in a.get("candidates", []))
            & set(c["chunk_id"] for c in b.get("candidates", [])))
        d["n_candidates"] = len(a.get("candidates", []))
        d["answer_in_corpus_after_decon"] = b.get("answer_in_corpus_at_all")
        d["contaminated_source_in_full_topk"] = any(
            s in (a.get("source_paths") or []) for s in EXCLUDED_FILES)
        d["answer_changed"] = a.get("answer") != b.get("answer")
        if d["facts_decon"] > d["facts_full"]:
            improved.append(a["q"])
        elif d["facts_full"] > d["facts_decon"]:
            worsened.append(a["q"])
        if not d["answer_in_corpus_after_decon"]:
            d["flag"] = ("ANSWER ABSENT FROM CORPUS AFTER DECONTAMINATION - this question "
                         "cannot be answered in this arm and is NOT a retrieval failure")
        diffs.append(d)

    return {
        "per_question": diffs,
        "improved_after_decon": improved,
        "worsened_after_decon": worsened,
        "totals": {
            "facts_in_answer_full": sum(d["facts_full"] for d in diffs),
            "facts_in_answer_decon": sum(d["facts_decon"] for d in diffs),
            "total_facts": sum(d["n_facts"] for d in diffs),
            "questions_answer_absent_after_decon":
                sum(1 for d in diffs if not d["answer_in_corpus_after_decon"]),
            "questions_with_contaminated_source_in_topk":
                sum(1 for d in diffs if d["contaminated_source_in_full_topk"]),
            "answers_that_changed": sum(1 for d in diffs if d["answer_changed"]),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3
    out.parent.mkdir(parents=True, exist_ok=True)

    questions = load_questions()
    if args.limit:
        questions = questions[:args.limit]
    print(f"loaded {len(questions)} questions (the original published set)")

    results = {}
    calls = 0
    for arm, path in ARMS.items():
        if not path.exists():
            print(f"  SKIP {arm}: {path} does not exist")
            continue
        frames = run_arm(arm, path, questions, out.parent)
        results[arm] = frames
        calls += sum(1 for f in frames if "answer" in f)

    # ---------------- paired comparison
    summary = {"questions": len(questions), "generation_calls": calls,
               "top_k": TOP_K, "pool": POOL,
               "arms": {k: str(v) for k, v in ARMS.items()},
               "is_not": ["not a reproduction of historical causality",
                          "not an independent generalisation test",
                          "no assumption that accuracy must improve"]}
    if "full" in results and "decon" in results:
        try:
            summary.update(_paired_summary(results))
            print("\n=== PAIRED RESULT ===")
            for k, v in summary["totals"].items():
                print(f"  {k}: {v}")
            print(f"  improved after decon: {len(summary['improved_after_decon'])}  "
                  f"worsened: {len(summary['worsened_after_decon'])}")
        except Exception as e:  # noqa: BLE001
            # Never let the summary computation destroy the raw per-question records: an
            # exception here previously aborted before pair.json was written, which would have
            # thrown away 24 real generation calls.
            summary["summary_error"] = f"{type(e).__name__}: {e}"
            print(f"  SUMMARY FAILED: {summary['summary_error']} (raw arm files are still on disk)")

    if _EMB is not None:
        try:
            _EMB.close()
        except Exception:  # noqa: BLE001
            pass

    out.write_text(json.dumps({"summary": summary, "arms": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


EXCLUDED_FILES = ("services/retrieval/rag_compare.py",
                  "services/retrieval/retrieval_bench.py",
                  "services/retrieval/benchmark_data.py",
                  "benchmarks/local-fast-results.md",
                  "benchmarks/local-smart-results.md",
                  "benchmarks/retrieval-results.md")

if __name__ == "__main__":
    raise SystemExit(main())

