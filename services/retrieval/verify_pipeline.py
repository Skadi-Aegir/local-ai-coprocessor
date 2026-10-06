"""
verify_pipeline.py - end-to-end check of scan -> chunk -> embed -> FAISS -> retrieve.

Runs against a real slice of this repository (docs + scripts + a few services) and
asserts the properties the Phase 4 specification requires:

  * a build indexes files and produces a searchable index
  * a SECOND build over an unchanged tree embeds ZERO chunks (incremental indexing)
  * touching one file re-embeds only that file
  * deleting a file removes exactly its chunks
  * retrieval returns the file a human would expect

Run: python verify_pipeline.py
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from embedder import Embedder, EmbedderConfig  # noqa: E402
from index_builder import IndexBuilder  # noqa: E402
from index_store import IndexStore  # noqa: E402
from retriever import Retriever, RetrieveConfig  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SOURCE_DIRS = ["docs", "scripts", "config"]
WORK = (Path(__file__).resolve().parents[2] / r"cache\pipeline_test")

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"{'OK  ' if cond else 'FAIL'} {name}{(' :: ' + detail) if detail else ''}")
    if not cond:
        FAILS.append(name)


def make_corpus(dest: Path) -> None:
    """Copy a real slice of the repo so we can mutate it safely."""
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    for d in SOURCE_DIRS:
        src = REPO / d
        if src.is_dir():
            shutil.copytree(src, dest / d)


def main() -> int:
    corpus = WORK / "corpus"
    index_dir = WORK / "index"
    make_corpus(corpus)
    if index_dir.exists():
        shutil.rmtree(index_dir, ignore_errors=True)

    n_files = sum(1 for _ in corpus.rglob("*") if _.is_file())
    print(f"corpus: {corpus}  ({n_files} files)")

    emb = Embedder(EmbedderConfig(dimension=512, batch_size=32, max_seq_length=2048))
    store = IndexStore(index_dir, 512)
    builder = IndexBuilder(emb, store, strategy="structured", max_tokens=900)

    # ---------------- first build
    print("\n--- build 1 (cold) ---")
    r1 = builder.build("local-ai", corpus, verbose=True)
    print(f"  {r1.to_dict()}")
    check("first build indexes files", r1.chunks_embedded > 0, str(r1.chunks_embedded))
    check("first build reports files as new", r1.files_new > 0, str(r1.files_new))
    st = store.stats()
    check("index and metadata are consistent", st["consistent"], str(st))
    check("vector count equals chunk count",
          st["vectors_in_index"] == st["chunks_in_metadata"] == r1.chunks_embedded,
          str(st))

    # ---------------- second build must be incremental
    print("\n--- build 2 (nothing changed - must embed ZERO) ---")
    t0 = time.time()
    r2 = builder.build("local-ai", corpus)
    dt = time.time() - t0
    print(f"  {r2.to_dict()}  ({dt:.2f}s)")
    check("second build embeds ZERO chunks (incremental)",
          r2.chunks_embedded == 0, f"embedded {r2.chunks_embedded}")
    check("second build sees all files as unchanged",
          r2.files_unchanged == r1.files_new and r2.files_new == 0 and r2.files_changed == 0,
          f"unchanged={r2.files_unchanged} new={r2.files_new} changed={r2.files_changed}")
    check("index unchanged after a no-op build",
          store.stats()["vectors_in_index"] == st["vectors_in_index"],
          str(store.stats()))

    # ---------------- one file changes
    print("\n--- build 3 (one file modified) ---")
    target = None
    for p in sorted(corpus.rglob("*.md")):
        target = p
        break
    check("found a markdown file to modify", target is not None)
    before = store.stats()["vectors_in_index"]
    target.write_text(target.read_text(encoding="utf-8") +
                      "\n\n## Incremental marker\n\nA distinctive phrase: zyzzyva-quokka-7788.\n",
                      encoding="utf-8")
    r3 = builder.build("local-ai", corpus)
    print(f"  {r3.to_dict()}")
    check("one file reported as changed", r3.files_changed == 1, str(r3.files_changed))
    check("only that file's chunks were re-embedded",
          r3.chunks_embedded > 0 and r3.chunks_embedded < before,
          f"re-embedded {r3.chunks_embedded} of {before}")
    check("index is consistent after the change", store.stats()["consistent"],
          str(store.stats()))

    # the new text must be findable
    retr = Retriever(emb, store, RetrieveConfig(top_k=5))
    hits = retr.retrieve("zyzzyva-quokka-7788", instruction=None)
    check("newly added content is retrievable",
          any("zyzzyva-quokka-7788" in h.content for h in hits),
          str([h.source_path for h in hits]))

    # ---------------- delete a file
    print("\n--- build 4 (one file deleted) ---")
    victim = None
    for p in sorted(corpus.rglob("*.md")):
        victim = p
        break
    vrel = victim.relative_to(corpus).as_posix()
    before_n = store.stats()["chunks_in_metadata"]
    victim.unlink()
    r4 = builder.build("local-ai", corpus)
    print(f"  {r4.to_dict()}")
    check("deletion reported", r4.files_deleted == 1, str(r4.files_deleted))
    check("chunks were removed", r4.chunks_removed > 0, str(r4.chunks_removed))
    check("index consistent after deletion", store.stats()["consistent"], str(store.stats()))
    check("removed file no longer appears in results",
          all(h.source_path != vrel for h in retr.retrieve("incremental marker", instruction=None)),
          vrel)

    # ---------------- real retrieval sanity
    print("\n--- retrieval sanity on real content ---")
    probe = [
        ("Which script samples GPU memory while a workload runs?", "vram_watch"),
        ("Where is the local-fast start script?", "start-local-fast"),
        ("What port does the fast worker listen on?", "local-fast"),
    ]
    for q, expect in probe:
        hs = retr.retrieve(q, instruction=None, top_k=5)
        paths = [h.source_path for h in hs]
        hit = any(expect in p for p in paths)
        check(f"retrieval finds '{expect}'", hit, str(paths[:3]))

    # ---------------- context packaging
    print("\n--- context packaging ---")
    hs = retr.retrieve("What port does the fast worker listen on?", instruction=None, top_k=3)
    packed = retr.pack(hs)
    check("pack labels each source", packed.count("SOURCE ") == len(hs),
          f"{packed.count('SOURCE ')} labels for {len(hs)} hits")
    check("pack includes path/lines/score",
          "path:" in packed and "lines:" in packed and "score:" in packed)
    check("empty result is explicit", "NO SOURCES RETRIEVED" in retr.pack([]))

    store.close()
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        return 1
    print("pipeline verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
