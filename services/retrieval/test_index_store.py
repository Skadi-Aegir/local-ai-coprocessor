"""
test_index_store.py - asserting tests for the FAISS + SQLite retrieval store.

The most important test here is that FAISS inner-product search over L2-normalised
vectors really does reproduce reference cosine similarity. The specification asks for
that to be verified rather than assumed, and getting it wrong would corrupt every
retrieval metric downstream while still "working".
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from index_store import IndexStore, cosine_similarity, l2_normalize  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {name}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def mk_chunks(n: int, project: str = "p", path: str = "a.py") -> list[dict]:
    return [{
        "chunk_id": f"{project}-{path}-{i}", "project": project, "source_path": path,
        "content_type": "code", "language": "python", "start_line": i * 10 + 1,
        "end_line": i * 10 + 9, "content": f"content {i}", "symbol": f"f{i}",
        "strategy": "structured", "content_hash": f"h{i}", "commit_hash": "", "mtime": "",
    } for i in range(n)]


def main() -> int:
    rng = np.random.default_rng(0)
    dim = 64

    # ---------------- normalisation helper
    print("--- normalisation ---")
    raw = rng.normal(size=(5, dim)).astype(np.float32)
    n = l2_normalize(raw)
    check("l2_normalize yields unit rows",
          bool(np.allclose(np.linalg.norm(n, axis=1), 1.0, atol=1e-6)))
    z = l2_normalize(np.zeros((2, dim), dtype=np.float32))
    check("zero vectors do not produce NaN", bool(np.isfinite(z).all()))

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        store = IndexStore(d, dim)

        docs = l2_normalize(rng.normal(size=(20, dim)).astype(np.float32))
        chunks = mk_chunks(20)
        added = store.add(docs, chunks, file_hashes={"a.py": "filehash1"})
        check("add returns the number of chunks", added == 20, str(added))
        check("index reports 20 vectors", store.stats()["vectors_in_index"] == 20,
              str(store.stats()))

        # ---------------- the critical contract: FAISS IP == cosine
        print("\n--- FAISS inner product equals reference cosine ---")
        q = l2_normalize(rng.normal(size=(1, dim)).astype(np.float32))
        hits = store.search(q, top_k=20)
        faiss_scores = np.array([h.score for h in hits])
        ref = cosine_similarity(q, docs)[0]
        ref_sorted = np.sort(ref)[::-1]
        check("FAISS scores match reference cosine to 1e-5",
              bool(np.allclose(faiss_scores, ref_sorted[:len(faiss_scores)], atol=1e-5)),
              f"max diff {np.abs(faiss_scores - ref_sorted[:len(faiss_scores)]).max():.2e}")

        # ordering must match an independent numpy argmax
        best_ref = int(np.argmax(ref))
        best_faiss = int(hits[0].chunk_id.split("-")[-1])
        check("top-1 chunk agrees with numpy argmax", best_ref == best_faiss,
              f"ref={best_ref} faiss={best_faiss}")

        # ---------------- rejecting non-normalised input
        print("\n--- input validation ---")
        try:
            store.add(rng.normal(size=(1, dim)).astype(np.float32) * 5, mk_chunks(1))
            check("unnormalised vectors are rejected", False, "no exception raised")
        except ValueError as e:
            check("unnormalised vectors are rejected", "normalis" in str(e).lower(), str(e)[:80])
        try:
            store.add(rng.normal(size=(1, dim)).astype(np.float64), mk_chunks(1))
            check("float64 vectors are rejected", False, "no exception raised")
        except ValueError as e:
            check("float64 vectors are rejected", "float32" in str(e), str(e)[:80])
        try:
            store.add(rng.normal(size=(2, dim)).astype(np.float32), mk_chunks(1))
            check("vector/chunk count mismatch is rejected", False, "no exception")
        except ValueError as e:
            check("vector/chunk count mismatch is rejected", "vectors for" in str(e), str(e)[:80])

        # ---------------- metadata mapping
        print("\n--- metadata mapping ---")
        h = hits[0]
        row = store.conn.execute(
            "SELECT content, source_path, start_line FROM chunks WHERE chunk_id=?",
            (h.chunk_id,)).fetchone()
        check("hit resolves back to its metadata row", row is not None)
        check("metadata content matches the hit content", row[0] == h.content)
        check("metadata path matches", row[1] == h.source_path)
        check("known_files lists the indexed file", store.known_files("p") == {"a.py"},
              str(store.known_files("p")))
        check("file_hash is retrievable", store.file_hash("p", "a.py") == "filehash1")

        # ---------------- project filtering
        print("\n--- project filter ---")
        docs2 = l2_normalize(rng.normal(size=(5, dim)).astype(np.float32))
        store.add(docs2, mk_chunks(5, project="q", path="b.py"),
                  file_hashes={"b.py": "filehash2"})
        mixed = store.search(q, top_k=5)
        only_p = store.search(q, top_k=5, project="p")
        check("unfiltered search can return both projects", len(mixed) > 0)
        check("filtered search returns only the requested project",
              all(x.project == "p" for x in only_p), str([x.project for x in only_p]))
        check("filter still returns up to top_k", len(only_p) == 5, str(len(only_p)))

        # ---------------- save / load round trip
        print("\n--- save / load ---")
        store.save()
        dim_saved = store.stats()["dimension"]
        store.close()

        # Regression: the plain constructor must ALSO load the saved index. An earlier
        # version only read index.faiss inside load(), so `IndexStore(dir, dim)` gave
        # an empty index while metadata reported every chunk - search silently
        # returned nothing, and the HTTP service hit exactly this.
        direct = IndexStore(d, dim)
        try:
            check("plain constructor loads the saved index",
                  direct.stats()["vectors_in_index"] == 25, str(direct.stats()))
            check("plain-constructor search actually returns hits",
                  len(direct.search(q, top_k=3)) == 3,
                  str(len(direct.search(q, top_k=3))))
        finally:
            direct.close()

        store2 = IndexStore.load(d, dim)
        try:
            check("reloaded index has the same vector count",
                  store2.stats()["vectors_in_index"] == 25, str(store2.stats()))
            check("reloaded dimension matches", store2.stats()["dimension"] == dim_saved)
            check("reloaded metadata is intact",
                  store2.stats()["chunks_in_metadata"] == 25, str(store2.stats()))
            h2 = store2.search(q, top_k=5)
            check("reloaded search returns the same top-1",
                  h2[0].chunk_id == hits[0].chunk_id, f"{h2[0].chunk_id} vs {hits[0].chunk_id}")

            # ---------------- incremental removal
            print("\n--- remove a file (incremental) ---")
            removed = store2.remove_file("p", "a.py")
            check("remove_file reports the chunks removed", removed == 20, str(removed))
            st = store2.stats()
            check("metadata no longer holds the file", store2.known_files("p") == set(),
                  str(store2.known_files("p")))
            check("index and metadata stay consistent after removal", st["consistent"],
                  str(st))
            check("index shrank to the surviving chunks", st["vectors_in_index"] == 5, str(st))
            h3 = store2.search(q, top_k=10)
            check("no hits reference the removed file",
                  all(x.source_path != "a.py" for x in h3),
                  str([x.source_path for x in h3]))
        finally:
            # Windows keeps metadata.sqlite locked while the handle is open, which
            # makes TemporaryDirectory cleanup fail with WinError 32.
            store2.close()

    # ---------------- non-ASCII path portability (real bug found on this machine)
    print("\n--- non-ASCII path save/load (faiss C++ fopen limitation) ---")
    import os
    from index_store import faiss_read_index, faiss_write_index

    unicode_root = Path(os.path.expanduser("~")) / "dsh_faiss_unicode_test"
    try:
        ustore = IndexStore(unicode_root, dim)
        try:
            udocs = l2_normalize(rng.normal(size=(4, dim)).astype(np.float32))
            ustore.add(udocs, mk_chunks(4, project="u", path="u.py"),
                       file_hashes={"u.py": "ufh"})
            ustore.save()
            check("index saved under a non-ASCII path", ustore.index_path.exists(),
                  str(ustore.index_path))
        finally:
            ustore.close()

        uload = IndexStore.load(unicode_root, dim)
        try:
            check("index reloaded from a non-ASCII path",
                  uload.stats()["vectors_in_index"] == 4, str(uload.stats()))
            uh = uload.search(l2_normalize(rng.normal(size=(1, dim)).astype(np.float32)), top_k=2)
            check("search works after a non-ASCII path reload", len(uh) == 2, str(len(uh)))
        finally:
            uload.close()

        # and confirm the raw faiss calls really do fail there, so the workaround
        # is not dead code that we would silently lose
        raw_failed = False
        try:
            import faiss as _f
            _f.write_index(_f.IndexFlatIP(dim), str(unicode_root / "raw.faiss"))
        except Exception:
            raw_failed = True
        check("raw faiss.write_index does fail on this path (workaround is needed)",
              raw_failed, "raw call unexpectedly succeeded - workaround may be unnecessary")
    finally:
        import shutil as _sh
        _sh.rmtree(unicode_root, ignore_errors=True)

    # ---------------- duplicate chunk_id must not desync the index
    # Regression: add() pushed vectors to FAISS first and inserted metadata second with
    # INSERT OR REPLACE. Two vectors carrying the same chunk_id therefore ADDED two
    # vectors but kept one row, leaving index.ntotal > chunk count and every faiss_id
    # above the collision pointing at the wrong chunk. Observed in practice as a
    # 1179-vs-1173 drift on the real corpus.
    print("\n--- duplicate chunk_id must not desync index and metadata ---")
    with tempfile.TemporaryDirectory() as td:
        d2 = Path(td)
        st = IndexStore(d2, dim)
        try:
            v = l2_normalize(rng.normal(size=(3, dim)).astype(np.float32))
            chunks = mk_chunks(3, project="dup", path="x.py")
            chunks[2]["chunk_id"] = chunks[0]["chunk_id"]   # deliberate collision
            n = st.add(v, chunks)
            s = st.stats()
            check("duplicate chunk_id is collapsed (returns fewer rows than vectors)",
                  n == 2, f"add returned {n}")
            check("index stays consistent after a chunk_id collision",
                  s["consistent"], str(s))
            check("vector count equals chunk count after collision",
                  s["vectors_in_index"] == s["chunks_in_metadata"] == 2, str(s))
            # the surviving row must be searchable
            hits = st.search(v[0], top_k=2)
            check("search still resolves after a collision", len(hits) == 2, str(len(hits)))
        finally:
            st.close()

    # ---------------- verify_and_repair heals an index/metadata mismatch
    print("\n--- verify_and_repair ---")
    with tempfile.TemporaryDirectory() as td:
        d3 = Path(td)
        st = IndexStore(d3, dim)
        try:
            v = l2_normalize(rng.normal(size=(6, dim)).astype(np.float32))
            st.add(v, mk_chunks(6, project="r", path="r.py"))
            check("fresh store is consistent", st.stats()["consistent"], str(st.stats()))
            # simulate drift: add two orphan vectors directly to FAISS
            st.index.add(l2_normalize(rng.normal(size=(2, dim)).astype(np.float32)))
            check("drift is detectable", not st.stats()["consistent"], str(st.stats()))
            rep = st.verify_and_repair()
            check("repair restores consistency", rep.get("consistent_after"), str(rep))
            check("repair leaves the right vector count",
                  st.stats()["vectors_in_index"] == 6, str(st.stats()))
            check("search works after repair", len(st.search(v[0], top_k=3)) == 3)
        finally:
            st.close()

    # ---------------- repair must REFUSE when vectors are genuinely missing
    print("\n--- repair refuses when vectors are unrecoverable ---")
    with tempfile.TemporaryDirectory() as td:
        d4 = Path(td)
        st = IndexStore(d4, dim)
        try:
            v = l2_normalize(rng.normal(size=(5, dim)).astype(np.float32))
            st.add(v, mk_chunks(5, project="m", path="m.py"))
            # metadata claims more chunks than the index holds
            st.conn.execute(
                "INSERT INTO chunks VALUES (99,'ghost','m','ghost.py','code','python',"
                "1,2,'ghost content','','fixed','','','','',0)")
            st.conn.commit()
            raised = False
            try:
                st.verify_and_repair()
            except RuntimeError as e:
                raised = "Rebuild from" in str(e) or "no longer exist" in str(e)
            check("repair refuses instead of guessing a pairing", raised)
        finally:
            st.close()

    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        return 1
    print("index_store self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
