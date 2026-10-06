"""
test_a_build_index.py - offline checks for the Phase A index builder (a_build_index.py).

No embedding service and no GPU: everything that decides WHAT enters the corpus, HOW provenance
is recorded and HOW a build is swapped in is exercised directly, on the real repository where the
point is the real corpus, and on temporary directories where the point is a failure mode.

Run with the retrieval venv (faiss, numpy):
  <repo>\\services\\retrieval\\venv\\Scripts\\python.exe benchmarks\\phaseA\\test_a_build_index.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import a_build_index as B  # noqa: E402
import numpy as np  # noqa: E402
from index_store import IndexStore  # noqa: E402
from provenance import content_hash, file_git_status, tree_blobs  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail="") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {name}{(' :: ' + str(detail)) if detail != '' else ''}")
    if not cond:
        FAILS.append(name)


def test_exclusions() -> None:
    print("\n--- exclusion rules")
    cases = {
        "scripts/test-gpu-scheduling.ps1": "test_or_conftest",       # admitted by the first build
        "services/retrieval/test_chunker.py": "test_or_conftest",
        "cli/ask.py": "qa_entry_point_code",
        "cli/ask_server.py": "qa_entry_point_code",                  # would have entered the next
        "cli/test_qa_path.py": "qa_entry_point_code",
        "services/retrieval/rag_compare_rerank.py": "explicit_exclusion_list",
        "services/retrieval/retriever.py": None,                     # the old "eval" trap
        "services/reranker/verify_hashes.py": None,                  # operational, stays included
        "services/retrieval/provenance.py": None,
        "docs/retrieval-results.md": "name_contains_results",
    }
    for rel, want in cases.items():
        got = B.exclusion_reason(rel)
        check(f"{rel} -> {want}", got == want, got)
    check("no dead entry in the real exclusion list", B.dead_exclusions() == [],
          B.dead_exclusions())
    with tempfile.TemporaryDirectory() as td:
        check("a missing exclusion target is reported as dead",
              len(B.dead_exclusions(Path(td))) == len(B.EXCLUDE_EXACT))
    qsets, missing = B.load_frozen_questions()
    check("every frozen question file the builder checks exists", missing == [], missing)
    check("the P56 and held-out sets are both loaded",
          {Path(k).parent.name for k in qsets} == {"p56", "heldout_v1"}
          and all(qsets.values()), {k: len(v) for k, v in qsets.items()})
    check("no question keeps a byte-order mark (the P56 file starts with one)",
          not any(q.startswith("﻿") for qs in qsets.values() for q in qs))


def test_corpus_change() -> None:
    print("\n--- the corpus the next build would index, against the current index")
    included, excluded = B.collect_files()
    m = json.loads((B.REPO / "vector/qa/current/manifest.json").read_text(encoding="utf-8"))
    removed = sorted(set(m["included_files"]) - set(included))
    # Holds before and after the next rebuild: whatever the current index has that the next build
    # drops must be dropped by a rule added in this round, or be gone from disk.
    unexplained = [p for p in removed if (B.REPO / p).exists()
                   and excluded.get(p) not in ("test_or_conftest", "qa_entry_point_code")]
    check("files the next build drops are dropped by the new rules, or no longer exist",
          not unexplained, {"removed": removed, "unexplained": unexplained})
    check("the test script is out and provenance.py is in",
          excluded.get("scripts/test-gpu-scheduling.ps1") == "test_or_conftest"
          and "services/retrieval/provenance.py" in included)
    check("nothing under cli/ is included", not [p for p in included if p.startswith("cli/")])
    check("every exclusion carries a reason", all(excluded.values()))


def test_leaks() -> None:
    print("\n--- answer-key markers and frozen questions")

    def chunk(text):
        return {"source_path": "x.py", "chunk_id": "c", "content": text}
    qs = {"set": ["What port does the local-fast service listen on?"]}
    check("facts=[ caught", B.leak_findings([chunk("facts=[['8192']]")], {}))
    check("facts = [ (spaced) caught", B.leak_findings([chunk("facts = [['8192']]")], {}))
    check("bad =[ caught", B.leak_findings([chunk("bad =[['4096']]")], {}))
    check("factsheet = [ is not a marker", not B.leak_findings([chunk("factsheet = [1]")], {}))
    check("a frozen question is caught across case and line wrapping", B.leak_findings(
        [chunk("Q: what PORT does the local-fast\n  service listen on?")], qs))
    check("a different question is not", not B.leak_findings(
        [chunk("What port does local-smart use?")], qs))

    from chunker import chunk_file
    included, _ = B.collect_files()
    qsets, _ = B.load_frozen_questions()
    chunks = []
    for rel in included:
        t = (B.REPO / rel).read_bytes().replace(b"\r\n", b"\n").decode("utf-8", errors="replace")
        chunks += [{"source_path": rel, "chunk_id": c.chunk_id, "content": c.content}
                   for c in chunk_file(t, rel, "local-ai", strategy=B.STRATEGY)]
    found = B.leak_findings(chunks, qsets)
    check("the real next-build corpus carries no marker and no frozen question",
          found == [], found[:3])


def test_git_status() -> None:
    print("\n--- per-file status against a commit (CRLF working tree, LF blobs)")
    blobs = tree_blobs(B.REPO, "HEAD")
    check("ls-tree read", blobs is not None and "cli/ask.py" in blobs)
    for rel, want in (("docs/models/local-fast.md", "clean"),
                      ("config/local-fast.yaml", "clean")):
        raw = (B.REPO / rel).read_bytes()
        crlf = b"\r\n" in raw
        check(f"{rel} reads as clean (this copy has CRLF: {crlf})",
              file_git_status(rel, raw, blobs) == want)
    check("an edited byte reads as modified", file_git_status(
        "docs/models/local-fast.md", (B.REPO / "docs/models/local-fast.md").read_bytes() + b"x",
        blobs) == "modified")
    check("a path not in the commit reads as untracked",
          file_git_status("no/such/file.md", b"", blobs) == "untracked")
    check("no commit data reads as unknown", file_git_status("a", b"", None) == "unknown")


def _store(d: Path, n: int, mutate=None, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.random((n, B.DIM)).astype("float32")
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    chunks = [{"chunk_id": f"c{i}", "project": "t", "source_path": f"f{i}.md",
               "content": f"chunk {i}", "start_line": 1, "end_line": 1} for i in range(n)]
    s = IndexStore(d, B.DIM)
    s.add(v, chunks)
    s.save()
    s.close()
    m = {"index_version": f"v{seed}", "n_chunks": n, "file_hashes": {},
         "chunks_sha256": B.chunks_digest(chunks), "vectors_sha256": B.vectors_digest(v),
         "validation": {"validation_passed": True, "no_answer_key_material": True}}
    if mutate:
        mutate(m)
    (d / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    return v


def test_digests_and_compare(tmp: Path) -> None:
    print("\n--- what was built, not what went in")
    a = [{"chunk_id": "x", "source_path": "p", "start_line": 1, "end_line": 2, "content": "abc"}]
    b = [dict(a[0], content="abd")]
    check("chunks_sha256 is stable", B.chunks_digest(a) == B.chunks_digest([dict(a[0])]))
    check("chunks_sha256 sees one changed character", B.chunks_digest(a) != B.chunks_digest(b))
    x, y = tmp / "build_x", tmp / "build_y"
    _store(x, 3)
    _store(y, 3)
    r = B.compare_builds(x, y)
    check("two identical builds compare as the same index",
          r["same_index"] and r["vectors_bit_identical"] and r["max_abs_diff"] == 0.0, r)
    z = tmp / "build_z"
    _store(z, 3, seed=1)
    r = B.compare_builds(x, z)
    check("different vectors under the same chunk table are NOT the same index",
          r["chunks_equal"] and not r["same_index"], r)


def test_safe_switch(tmp: Path) -> None:
    print("\n--- switching a build in")
    root = tmp / "qa"
    root.mkdir()
    cur, prev, stg = root / "current", root / "current.prev", root / "staging"

    _store(stg, 2, seed=1)
    ok, msg = B.safe_switch(stg, cur, prev)
    check("first switch with no previous index", ok and cur.exists() and not stg.exists(), msg)

    _store(stg, 3, seed=2)
    ok, msg = B.safe_switch(stg, cur, prev)
    v = lambda d: json.loads((d / "manifest.json").read_text())["index_version"]  # noqa: E731
    check("second switch: new current, old current kept as prev",
          ok and v(cur) == "v2" and v(prev) == "v1", msg)
    check("no stray directories left", sorted(p.name for p in root.iterdir())
          == ["current", "current.prev"], sorted(p.name for p in root.iterdir()))

    _store(stg, 4, seed=3)
    reader = IndexStore.load(cur, B.DIM)          # what a running ask_server holds open
    try:
        ok, msg = B.safe_switch(stg, cur, prev)
    finally:
        reader.close()
    check("index in use: switch refused", not ok and "in use" in msg, msg)
    check("index in use: NOTHING was deleted or moved", v(cur) == "v2" and v(prev) == "v1"
          and v(stg) == "v3", (v(cur), v(prev), v(stg)))

    ok, msg = B.safe_switch(stg, cur, prev)
    check("once released, the same staging build switches in",
          ok and v(cur) == "v3" and v(prev) == "v2", msg)

    bad = tmp / "bad_staging"
    _store(bad, 2, mutate=lambda m: m["validation"].pop("no_answer_key_material"))
    saved = (B.DEFAULT_DIR, B.PREV_DIR)
    B.DEFAULT_DIR, B.PREV_DIR = cur, prev
    try:
        rc = B.switch_only(bad)
    finally:
        B.DEFAULT_DIR, B.PREV_DIR = saved
    check("--switch-only refuses a build without a passing leak check", rc == 4 and v(cur) == "v3")

    B.DEFAULT_DIR, B.PREV_DIR = cur, prev
    try:
        rc = B.switch_only(prev)                  # the documented rollback
    finally:
        B.DEFAULT_DIR, B.PREV_DIR = saved
    check("rollback: --switch-only on current.prev swaps the two, keeping both",
          rc == 0 and v(cur) == "v2" and v(prev) == "v3"
          and sorted(p.name for p in root.iterdir()) == ["current", "current.prev"],
          (rc, v(cur), v(prev)))


def test_retention_verdict() -> None:
    print("\n--- P1-11 retention verdict")
    before = {"vram_mb": 2390.6, "process": {"torch_reserved_mb": 1160.0}}
    after_fixed = {"vram_mb": 2400.0, "process": {"torch_reserved_mb": 1175.5}}
    after_leak = {"vram_mb": 7444.0, "process": {"torch_reserved_mb": 6213.5}}
    r = B.retention_verdict(before, after_fixed)
    check("released service: no restart", r["measured"] and not r["restart_required"], r)
    r = B.retention_verdict(before, after_leak)
    check("retaining service: restart required", r["restart_required"]
          and r["retained_mb"] == 5053.5, r)
    r = B.retention_verdict({"vram_mb": 2360.6, "process": None},
                            {"vram_mb": 7750.6, "process": None})
    check("a pre-fix service is judged on vram_mb, and says so",
          r["restart_required"] and "predates" in r["basis"], r)
    check("no /stats -> not measured", not B.retention_verdict(None, None)["measured"])


class _StubEmbedding:
    """local-embedding stand-in: deterministic unit vectors, /v1/models and /stats."""

    def __init__(self):
        import hashlib
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        self.requests = 0
        stub = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, obj):
                body = json.dumps(obj).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self._json({"data": [{"meta": {"dimension": B.DIM, "stub": True}}]})
                return self._json({"vram_mb": 2390.6, "memory": {"torch_reserved_mb": 1160.0}})

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.requests += 1
                out = []
                for text in req["input"]:
                    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
                    v = np.random.default_rng(seed).standard_normal(B.DIM)
                    out.append({"embedding": (v / np.linalg.norm(v)).tolist()})
                return self._json({"data": out})

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


def test_main_end_to_end(tmp: Path) -> None:
    print("\n--- the whole build on the real corpus, embedding service stubbed")
    stub = _StubEmbedding()
    saved = (B.EMBED_URL, B.DEFAULT_DIR, B.PREV_DIR, B.nvidia_free_mb, sys.argv)
    root = tmp / "e2e"
    B.EMBED_URL, B.DEFAULT_DIR, B.PREV_DIR = stub.url, root / "current", root / "current.prev"
    B.nvidia_free_mb = lambda: 9000

    def run(*argv) -> int:
        sys.argv = ["a_build_index.py", *argv]
        return B.main()
    try:
        rc = run("--out", str(root / "staging"), "--report", str(root / "r1.json"))
        check("build, validate and switch exit 0", rc == 0, rc)
        m = json.loads((root / "current/manifest.json").read_text(encoding="utf-8"))
        check("manifest: new hash scheme, version scheme 2, both digests",
              m["hash_scheme"] == "sha256-lf-v1" and m["index_version_scheme"] == 2
              and len(m["chunks_sha256"]) == 64 and len(m["vectors_sha256"]) == 64)
        check("manifest: validation passed, leak check ran on both frozen sets",
              m["validation"]["validation_passed"] and m["validation"]["no_answer_key_material"]
              and len(m["validation"]["question_sets_checked"]) == 2)
        # Judged against git itself, so the check holds in any checkout, committed or not.
        import subprocess
        porcelain = subprocess.run(
            ["git", "-C", str(B.REPO), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True, text=True, timeout=60).stdout.splitlines()
        want = {}
        for ln in porcelain:
            path = ln[3:].strip().strip('"')
            if path in m["file_git_status"]:
                want[path] = "untracked" if ln[:2] == "??" else "modified"
        got = {k: s for k, s in m["file_git_status"].items() if s != "clean"}
        check("manifest: each corpus file's status agrees with git status, file by file",
              got == want and m["corpus_files_not_at_commit"] == sorted(want)
              and set(m["file_git_status"]) == set(m["file_hashes"]), {"got": got, "git": want})
        check("4 texts per embedding request", stub.requests == -(-m["n_chunks"] // B.BATCH),
              (stub.requests, m["n_chunks"]))
        rep = json.loads((root / "r1.json").read_text(encoding="utf-8"))
        check("report: margin sampled during the build, retention measured",
              rep["build_vram"]["batch"] == 4 and rep["embedding_service_memory"]["retention"]
              ["measured"], rep["build_vram"])
        import ask as ASK
        store, prov = ASK.open_index(root / "current")
        check("the switched-in build opens on the QA path",
              store.stats()["vectors_in_index"] == m["n_chunks"] == prov.n_chunks)
        store.close()

        rc = run("--out", str(root / "staging2"), "--report", str(root / "r2.json"),
                 "--no-switch", "--compare-to", str(root / "current"))
        cmp_ = json.loads((root / "r2.json").read_text(encoding="utf-8"))["comparison"]
        check("an identical rebuild compares as the same index, on content",
              rc == 0 and cmp_["same_index"] and cmp_["chunks_equal"]
              and cmp_["vectors_bit_identical"], cmp_)
        check("a second report never overwrites the first", (root / "r1.json").exists())

        rc = run("--out", str(root / "current"), "--report", str(root / "r3.json"))
        check("--out pointed at the live index is refused before anything is touched",
              rc == 2 and (root / "current/manifest.json").exists())
        B.nvidia_free_mb = lambda: 2048 + B.BUILD_TRANSIENT_ALLOWANCE_MB - 1
        rc = run("--out", str(root / "staging3"), "--report", str(root / "r4.json"))
        check("preflight refuses when the transient would breach the margin",
              rc == 7 and not (root / "staging3").exists())
    finally:
        B.EMBED_URL, B.DEFAULT_DIR, B.PREV_DIR, B.nvidia_free_mb, sys.argv = saved
        stub.close()


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="a_build_test_"))
    try:
        test_exclusions()
        test_corpus_change()
        test_leaks()
        test_git_status()
        test_digests_and_compare(tmp)
        test_safe_switch(tmp)
        test_retention_verdict()
        test_main_end_to_end(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        return 1
    print("index builder self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
