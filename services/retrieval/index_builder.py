"""
index_builder.py - scan -> chunk -> embed -> FAISS, with incremental updates.

Ties the pieces together. The contract:

  build(project, root)  indexes new and changed files, removes deleted ones, and
                        leaves unchanged files untouched (no re-embedding).

The incremental path is the reason the scanner returns hashes: a second build over an
unchanged tree must embed exactly zero chunks. That is asserted in the tests rather
than assumed, because a silently non-incremental indexer still "works" while being
unusably slow on a real repository.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from chunker import TokenCounter, chunk_file
from embedder import Embedder
from index_store import IndexStore
from scanner import scan_project, read_text


@dataclass
class BuildReport:
    project: str
    root: str
    files_new: int = 0
    files_changed: int = 0
    files_unchanged: int = 0
    files_deleted: int = 0
    chunks_embedded: int = 0
    chunks_removed: int = 0
    seconds: float = 0.0
    strategy: str = "structured"
    dimension: int = 0
    skipped_large: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class IndexBuilder:
    def __init__(self, embedder: Embedder, store: IndexStore, *,
                 strategy: str = "structured", size_tokens: int = 400,
                 overlap_tokens: int = 60, max_tokens: int = 1200,
                 tokenizer=None):
        self.emb = embedder
        self.store = store
        self.strategy = strategy
        self.size_tokens = size_tokens
        self.overlap_tokens = overlap_tokens
        self.max_tokens = max_tokens
        self.counter = TokenCounter(tokenizer)

    def build(self, project: str, root: str | Path, *,
              include: list[str] | None = None,
              verbose: bool = False) -> BuildReport:
        t0 = time.time()
        root = Path(root).resolve()
        rep = BuildReport(project=project, root=str(root),
                          strategy=self.strategy, dimension=self.store.dimension)

        # previous hashes for this project
        known = {r[0]: r[1] for r in self.store.conn.execute(
            "SELECT source_path, file_hash FROM files WHERE project=?", (project,))}

        plan = scan_project(root, project, known=known, include=include)
        rep.files_new = len(plan["new"])
        rep.files_changed = len(plan["changed"])
        rep.files_unchanged = len(plan["unchanged"])
        rep.files_deleted = len(plan["deleted"])

        if verbose:
            print(f"  new={rep.files_new} changed={rep.files_changed} "
                  f"unchanged={rep.files_unchanged} deleted={rep.files_deleted}", flush=True)

        # ---- deletions first, so ids do not churn more than necessary
        for f in plan["deleted"]:
            rep.chunks_removed += self.store.remove_file(project, f.rel_path)
            if verbose:
                print(f"  - removed {f.rel_path}", flush=True)

        # ---- new + changed (changed must be removed before re-adding)
        todo = plan["new"] + plan["changed"]
        for f in plan["changed"]:
            rep.chunks_removed += self.store.remove_file(project, f.rel_path)

        for i, f in enumerate(todo, 1):
            try:
                text = read_text(f.abs_path)
            except OSError as e:
                rep.errors.append(f"{f.rel_path}: {e}")
                continue
            chunks = chunk_file(
                text, f.rel_path, project, strategy=self.strategy,
                size_tokens=self.size_tokens, overlap_tokens=self.overlap_tokens,
                max_tokens=self.max_tokens, counter=self.counter,
                mtime=f.mtime,
            )
            if not chunks:
                continue
            vecs = self.emb.encode_documents([c.content for c in chunks])
            self.store.add(vecs, [c.to_dict() for c in chunks],
                           file_hashes={f.rel_path: f.file_hash})
            rep.chunks_embedded += len(chunks)
            if verbose and (i % 25 == 0 or i == len(todo)):
                print(f"  [{i}/{len(todo)}] {f.rel_path} (+{len(chunks)} chunks)", flush=True)

        self.store.set_meta(f"root:{project}", str(root))
        self.store.set_meta(f"strategy:{project}", self.strategy)
        # Heal any index/metadata drift before persisting, so a corpus manipulated
        # outside the scanner cannot leave search silently degraded.
        rep.consistency = self.store.verify_and_repair(verbose=verbose)
        self.store.save()
        rep.seconds = round(time.time() - t0, 2)
        return rep


def build_from_config(cfg: dict, verbose: bool = True) -> BuildReport:
    """
    Build an index from a config dict (see config/local-embedding.yaml).

    Expected keys: project, root, index_dir, dimension, strategy, batch_size.
    """
    from embedder import EmbedderConfig

    project = cfg["project"]
    root = cfg["root"]
    index_dir = cfg["index_dir"]
    dim = int(cfg.get("dimension", 1024))

    emb = Embedder(EmbedderConfig(dimension=dim, batch_size=int(cfg.get("batch_size", 32))))
    store = IndexStore(index_dir, dim)
    builder = IndexBuilder(
        emb, store,
        strategy=cfg.get("strategy", "structured"),
        size_tokens=int(cfg.get("size_tokens", 400)),
        overlap_tokens=int(cfg.get("overlap_tokens", 60)),
        max_tokens=int(cfg.get("max_tokens", 1200)),
    )
    try:
        rep = builder.build(project, root, verbose=verbose)
    finally:
        store.close()
    return rep

