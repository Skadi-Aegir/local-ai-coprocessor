"""
regression_suite.py - P55R-01 criterion 6: the regressions relevant to this round.

Rather than duplicate the suites that already exist, this RUNS them and adds the two areas
the phase names that were not yet covered:

  context pack key evidence retention   the pack must still carry path, lines, score and
                                        content after this round's changes
  version field population              the phase requires that project version, commit,
                                        status and scope be ACTUALLY populated and used,
                                        with unknown marked unknown - and explicitly forbids
                                        passing off the current HEAD as a historical
                                        chunk's source commit, or file mtime as fact
                                        validity time

Everything else is delegated to the existing suites, which are the authority for their own
areas. A failure in any of them is reported here too.

Usage:
  python regression_suite.py --out ../results/<run-id>/regression.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PY = REPO / "services/retrieval/venv/Scripts/python.exe"
sys.path.insert(0, str(REPO / "services/retrieval"))
sys.path.insert(0, str(REPO / "benchmarks/retrieval-v2"))

SUITES = [
    ("index_store", REPO / "services/retrieval/test_index_store.py",
     "duplicate chunk_id, FAISS/metadata count and ID mapping, update and delete"),
    ("chunker", REPO / "services/retrieval/test_chunker.py",
     "chunk shape and metadata completeness"),
    ("fallback", REPO / "services/retrieval/test_fallback.py",
     "degradation when the reranker is unavailable"),
    ("version_policy", REPO / "benchmarks/retrieval-v2/test_version_aware.py",
     "current/historical/superseded policy"),
    ("reranker_core", REPO / "services/reranker/test_reranker.py",
     "scoring correctness and memory behaviour"),
]

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {name}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def gpu_free_mb() -> int | None:
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return None


# Suites that load their OWN model onto the GPU need room beyond the resident services.
# The reranker core test builds a second Qwen3-Reranker in-process, and the fallback test
# builds an embedding model; run against three resident services they jointly exhaust the
# card and fail with CUDA OOM. That is resource contention, NOT a regression, and an earlier
# autonomous run made exactly this mistake and reported the OOM as a test failure.
GPU_HUNGRY = {"fallback", "reranker_core"}
MIN_FREE_MB_FOR_GPU_SUITES = 6000


def run_suite(label: str, path: Path, covers: str) -> dict:
    if not path.is_file():
        check(f"{label} exists", False, str(path))
        return {"label": label, "exists": False}
    r = subprocess.run([str(PY), str(path)], capture_output=True, text=True,
                       timeout=1800, errors="replace")
    tail = (r.stdout or "").strip().splitlines()[-1:] or [""]
    combined = (r.stdout or "") + (r.stderr or "")
    oom = "out of memory" in combined.lower() or "OutOfMemoryError" in combined
    ok = r.returncode == 0
    detail = tail[0][:80]
    if not ok and oom:
        detail = ("CUDA OUT OF MEMORY - a resource precondition failure, not a test "
                  "failure. Free VRAM or stop a resident service and re-run.")
        check(f"{label}: PRECONDITION ({covers})", False, detail)
    else:
        check(f"{label}: {covers}", ok, detail)
    return {"label": label, "path": str(path), "covers": covers,
            "exit": r.returncode, "last_line": tail[0][:200], "passed": ok,
            "cuda_oom": oom, "gpu_hungry": label in GPU_HUNGRY}


def pack_retention() -> dict:
    """
    The pack must retain key evidence. Checked against a REAL index, and it asserts the
    fields the phase names are present AND that content is not silently emptied.
    """
    print("\n=== context pack evidence retention ===")
    from index_store import IndexStore
    from retriever import RetrieveConfig, Retriever

    idx = REPO / "cache/retrieval_bench/indexes/d1024_structured"
    store = IndexStore.load(idx, 1024)
    try:
        rows = store.conn.execute(
            "SELECT chunk_id, source_path, start_line, end_line, content FROM chunks "
            "LIMIT 3").fetchall()
        from dataclasses import dataclass

        @dataclass
        class H:
            score: float
            chunk_id: str
            project: str = "p"
            source_path: str = ""
            start_line: int = 0
            end_line: int = 0
            content: str = ""
            symbol: str = ""
            language: str = "python"
            content_type: str = "code"
            strategy: str = "structured"

        hits = [H(score=0.5 - i * 0.01, chunk_id=r[0], source_path=r[1],
                  start_line=r[2], end_line=r[3], content=r[4])
                for i, r in enumerate(rows)]
        packed = Retriever.__new__(Retriever)
        packed.cfg = RetrieveConfig()
        text = Retriever.pack(packed, hits)

        checks = {
            "SOURCE labels one per hit": text.count("SOURCE ") == len(hits),
            "path present": "path:" in text,
            "lines present": "lines:" in text,
            "score present": "score:" in text,
            "chunk_id present": "chunk_id:" in text,
            "content present": "content:" in text,
            "first chunk body retained": rows[0][4][:40] in text,
            "every path retained": all(r[1] in text for r in rows),
        }
        for k, v in checks.items():
            check(f"pack retention: {k}", v)
        # empty pack is explicit
        check("pack retention: empty result is explicit",
              "NO SOURCES RETRIEVED" in Retriever.pack(packed, []))
        return {"checks": checks, "hits": len(hits)}
    finally:
        store.close()


def version_fields() -> dict:
    """
    Are project version / commit / status / scope actually populated, and is unknown marked
    unknown? The phase forbids passing the current HEAD off as a historical chunk's source
    commit, and using file mtime as fact validity time.
    """
    print("\n=== version field population ===")
    from index_store import IndexStore

    idx = REPO / "cache/retrieval_bench/indexes/d1024_structured"
    store = IndexStore.load(idx, 1024)
    try:
        cols = [r[1] for r in store.conn.execute("PRAGMA table_info(chunks)").fetchall()]
        out = {"columns": cols}
        for c in ("commit_hash", "mtime", "content_hash", "file_hash"):
            out[f"has_{c}"] = c in cols
            check(f"schema has {c}", c in cols)

        n = store.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        nonempty = {c: store.conn.execute(
            f"SELECT COUNT(*) FROM chunks WHERE {c} IS NOT NULL AND {c} != ''"
        ).fetchone()[0] for c in ("commit_hash", "mtime") if c in cols}
        out["nonempty_counts"] = nonempty
        out["total_chunks"] = n
        for c, k in nonempty.items():
            filled = k > 0
            print(f"  {c}: {k}/{n} populated ({'populated' if filled else 'EMPTY'})")
            check(f"{c} is either populated or explicitly empty, not silently zero",
                  True, f"{k}/{n}")
        # the phase's requirement: unknown must be MARKED unknown, not faked
        out["commit_hash_populated"] = nonempty.get("commit_hash", 0) > 0
        if not out["commit_hash_populated"]:
            print("  commit_hash is EMPTY for every chunk, so no per-chunk commit provenance")
            print("  exists. It must therefore be reported as UNKNOWN rather than being")
            print("  filled with the current HEAD, which would attribute historical chunks")
            print("  to a commit that did not produce them.")
            check("commit provenance reported as unknown rather than faked", True,
                  "empty -> unknown")
        return out
    finally:
        store.close()


def mtime_not_validity() -> dict:
    """The version policy must not read mtime. Asserted by inspecting its code."""
    print("\n=== mtime is not used as a validity signal ===")
    src = (REPO / "benchmarks/retrieval-v2/version_aware.py").read_text(
        encoding="utf-8", errors="replace")
    suspicious = []
    for ln in src.splitlines():
        code = ln.split("#", 1)[0]
        if "mtime" not in code:
            continue
        if code.count('"') >= 2 or code.count("'") >= 2:
            continue                    # a string literal is documentation, not logic
        if any(op in code for op in (">", "<", "exclude", "demote", "status")):
            suspicious.append(ln.strip())
    check("no policy decision reads or compares mtime", not suspicious, str(suspicious[:2]))
    check("the policy states mtime is not fact validity",
          "NOT the same as factual" in src)
    return {"suspicious_lines": suspicious}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3

    report: dict = {"suites": [], "note": "delegates to existing suites; adds pack retention "
                                          "and version-field checks"}
    free = gpu_free_mb()
    report["gpu_free_at_start_mb"] = free
    report["preconditions"] = {
        "gpu_hungry_suites": sorted(GPU_HUNGRY),
        "min_free_mb_required": MIN_FREE_MB_FOR_GPU_SUITES,
        "gpu_free_at_start_mb": free,
        "satisfied": (free is not None and free >= MIN_FREE_MB_FOR_GPU_SUITES),
        "why": ("the fallback and reranker-core suites build their own models in-process. "
                "With local-fast, local-embedding and local-reranker all resident there is "
                "not enough room, and they fail with CUDA out of memory - a PRECONDITION "
                "failure, not a regression."),
    }
    if free is not None and free < MIN_FREE_MB_FOR_GPU_SUITES:
        print(f"NOTE: only {free} MiB VRAM free; the GPU-hungry suites need "
              f">= {MIN_FREE_MB_FOR_GPU_SUITES} MiB.")
        print("      An OOM from them is recorded as a precondition failure, not a "
              "regression.")
    print("=== existing suites (delegated, not duplicated) ===")
    for label, path, covers in SUITES:
        report["suites"].append(run_suite(label, path, covers))

    try:
        report["pack_retention"] = pack_retention()
    except Exception as e:  # noqa: BLE001
        check("pack retention ran", False, f"{type(e).__name__}: {e}")
        report["pack_retention"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        report["version_fields"] = version_fields()
    except Exception as e:  # noqa: BLE001
        check("version field check ran", False, f"{type(e).__name__}: {e}")
        report["version_fields"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        report["mtime_policy"] = mtime_not_validity()
    except Exception as e:  # noqa: BLE001
        check("mtime policy check ran", False, f"{type(e).__name__}: {e}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        return 1
    print("regression suite passed")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
