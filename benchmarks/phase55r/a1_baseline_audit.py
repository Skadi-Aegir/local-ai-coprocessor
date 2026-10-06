"""
a1_baseline_audit.py - P55R-01 work package A1.

The question: does branch D matching 66.7% prove the Phase 5 Dense baseline was reproduced?

It does not, on its own. A matching aggregate can coincide for unrelated reasons. This
script does the comparison the phase asks for by ENUMERATING EVERY VARIABLE that could
differ between the historical runs and the current controlled comparison, and reading each
one out of the code rather than out of the narrative:

  corpus + index identity and content hashes
  question set and ground truth
  dense vs reranked arms
  embedding instruction
  candidate pool and top-K
  context pack (function, truncation, header)
  generation prompt, sampling settings
  grader

Anything that differs is a confound, and the phase is explicit: if other variables moved
alongside the index, the index alone must not be blamed.

Runs entirely offline against saved artifacts. No model calls.

Usage:
  python a1_baseline_audit.py --out ../results/<run-id>/a1_baseline_audit.json
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# The three generations of RAG harness plus the two retrieval harnesses.
HARNESSES = {
    "phase4_rag": REPO / "services/retrieval/rag_compare.py",
    "phase4_retrieval_bench": REPO / "services/retrieval/retrieval_bench.py",
    "phase5_rag": REPO / "services/retrieval/rag_compare_rerank.py",
    "phase55_reconcile": REPO / "benchmarks/retrieval-v2/baseline_reconcile.py",
}

INDEXES = {
    "served_local_ai": REPO / "vector/embedding/local-ai",
    "bench_d1024_structured": REPO / "cache/retrieval_bench/indexes/d1024_structured",
}
CORPORA = {
    "served_corpus": REPO / "vector/embedding/corpus",
    "bench_corpus": REPO / "cache/retrieval_bench/corpus",
}


def sha256_file(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def corpus_fingerprint(root: Path) -> dict:
    """Content-addressed fingerprint of a corpus, so two corpora can be compared exactly."""
    if not root.is_dir():
        return {"exists": False}
    files = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rel = p.relative_to(root).as_posix()
            files[rel] = {"size": p.stat().st_size, "sha256": sha256_file(p)}
    combined = hashlib.sha256(
        "\n".join(f"{k}:{v['sha256']}" for k, v in sorted(files.items())).encode()
    ).hexdigest()
    return {"exists": True, "file_count": len(files), "combined_sha256": combined,
            "files": files}


def index_identity(d: Path) -> dict:
    if not d.is_dir():
        return {"exists": False}
    out = {"exists": True}
    for name in ("index.faiss", "vectors.npy", "metadata.sqlite"):
        p = d / name
        out[name] = {"exists": p.exists(),
                     "size": p.stat().st_size if p.exists() else None,
                     "sha256": sha256_file(p) if p.exists() else None}
    # chunk count and distinct paths from the sqlite metadata
    try:
        sys.path.insert(0, str(REPO / "services" / "retrieval"))
        from index_store import IndexStore
        s = IndexStore.load(d, 1024)
        st = s.stats()
        rows = s.conn.execute(
            "SELECT COUNT(DISTINCT source_path) FROM chunks").fetchone()
        out.update({"chunks": st.get("chunks_in_metadata"),
                    "vectors": st.get("vectors_in_index"),
                    "consistent": st.get("consistent"),
                    "distinct_paths": rows[0] if rows else None})
        s.close()
    except Exception as e:  # noqa: BLE001
        out["stats_error"] = f"{type(e).__name__}: {e}"
    return out


def extract_settings(path: Path) -> dict:
    """
    Read the harness's actual settings out of its source, so the comparison is against CODE
    rather than against what a report says the code did.
    """
    if not path.is_file():
        return {"exists": False}
    src = path.read_text(encoding="utf-8", errors="replace")
    s: dict = {"exists": True, "sha256": sha256_file(path),
               "lines": len(src.splitlines())}

    # index / URL targets
    s["index_literals"] = sorted(set(re.findall(r'["\']([^"\']*(?:indexes?|vector)[^"\']*)["\']', src)))[:6]
    s["embed_urls"] = sorted(set(re.findall(r'https?://127\.0\.0\.1:\d+[^"\']*', src)))[:8]
    # top_k / pool / instruction / temperature / max_tokens
    for key in ("top_k", "candidate_top_n", "final_top_k", "max_tokens", "temperature"):
        vals = sorted(set(re.findall(rf'{key}\s*=\s*([^,\n\)]+)', src)))
        if vals:
            s[key] = vals[:6]
    # how the retriever is configured: the instruction argument matters
    s["retrieve_config"] = re.findall(r'RetrieveConfig\(([^)]*)\)', src)[:4]
    s["retrieve_calls"] = re.findall(r'\.retrieve\(([^)]*)\)', src)[:6]
    # the generation prompt
    sysp = re.findall(r'system\s*=\s*\((.*?)\)\n', src, re.S)
    s["system_prompts"] = [re.sub(r'\s+', ' ', x)[:200] for x in sysp][:4]
    # the pack function used
    s["uses_http_pack"] = "/v1/retrieve" in src
    s["uses_local_pack"] = bool(re.search(r'\bretr\.pack\(|retriever\.pack\(', src))
    s["uses_pipeline_pack"] = bool(re.search(r'\bpr\.pack\(', src))
    # the grading keywords
    s["facts_blocks"] = len(re.findall(r'facts=\[', src))
    s["bad_blocks"] = len(re.findall(r'bad=\[', src))
    # refusal detection strings
    s["refusal_strings"] = sorted(set(re.findall(r'"(not in sources|i do not know|'
                                                  r'not provided|no information)"', src)))
    return s


def question_set_fingerprint() -> dict:
    """
    Which questions each harness asked. The 12 RAG questions are duplicated by hand across
    harnesses, so a difference in wording or in the expected-fact lists is a confound.
    """
    qs = {}
    for label, path in HARNESSES.items():
        if not path.is_file():
            continue
        src = path.read_text(encoding="utf-8", errors="replace")
        found = re.findall(r'dict\(q="([^"]+)"', src)
        qs[label] = {"count": len(found), "questions": found}
    # cross-compare the RAG harnesses
    rag = {k: v for k, v in qs.items() if k in ("phase4_rag", "phase5_rag",
                                                "phase55_reconcile")}
    sets = {k: set(v["questions"]) for k, v in rag.items()}
    keys = sorted(sets)
    identical = all(sets[keys[0]] == sets[k] for k in keys) if keys else None
    return {"per_harness": qs, "rag_question_sets_identical": identical,
            "rag_set_sizes": {k: len(v) for k, v in sets.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-hashes", action="store_true",
                    help="skip corpus hashing (slow on large trees)")
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3

    report: dict = {"purpose": "attribute the Phase 4 91.7% vs Phase 5 66.7% difference",
                    "harnesses": {}, "indexes": {}, "corpora": {}}

    print("=== harness settings, read from source ===")
    for label, p in HARNESSES.items():
        s = extract_settings(p)
        report["harnesses"][label] = s
        print(f"  {label:<26} exists={s.get('exists')} "
              f"http_pack={s.get('uses_http_pack')} local_pack={s.get('uses_local_pack')} "
              f"pipeline_pack={s.get('uses_pipeline_pack')}")
        if s.get("retrieve_config"):
            print(f"      RetrieveConfig: {s['retrieve_config']}")
        if s.get("top_k"):
            print(f"      top_k={s['top_k']}  max_tokens={s.get('max_tokens')}  "
                  f"temperature={s.get('temperature')}")

    print("\n=== question sets ===")
    q = question_set_fingerprint()
    report["question_sets"] = q
    for k, v in q["per_harness"].items():
        print(f"  {k:<26} {v['count']} questions")
    print(f"  RAG question sets identical across generations: "
          f"{q['rag_question_sets_identical']}")

    if not args.skip_hashes:
        print("\n=== corpus fingerprints ===")
        for label, root in CORPORA.items():
            fp = corpus_fingerprint(root)
            report["corpora"][label] = {k: v for k, v in fp.items() if k != "files"}
            print(f"  {label:<16} files={fp.get('file_count')} "
                  f"combined={str(fp.get('combined_sha256'))[:16]}")
        # which corpus files feed which index is visible from the file lists
        s_files = set(report["corpora"]["served_corpus"].get("files", {})) \
            if "files" in report["corpora"].get("served_corpus", {}) else set()
        report["corpora_identical"] = (
            report["corpora"].get("served_corpus", {}).get("combined_sha256")
            == report["corpora"].get("bench_corpus", {}).get("combined_sha256"))

    print("\n=== index identity ===")
    for label, d in INDEXES.items():
        ident = index_identity(d)
        report["indexes"][label] = ident
        print(f"  {label:<24} chunks={ident.get('chunks')} "
              f"vectors={ident.get('vectors')} paths={ident.get('distinct_paths')} "
              f"consistent={ident.get('consistent')}")
        print(f"      faiss_sha={str(ident.get('index.faiss', {}).get('sha256'))[:16]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
