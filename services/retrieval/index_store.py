"""
index_store.py - FAISS index + metadata store for the retrieval layer.

Design:
  * FAISS `IndexFlatIP` over L2-normalised vectors. Inner product on unit vectors is
    exactly cosine similarity, which is what the model card declares
    (`similarity_fn_name: "cosine"`). IndexFlatIP is exact, so Phase 4 has no
    approximation error and no ANN tuning to get wrong.
  * Metadata lives OUTSIDE FAISS, in SQLite (the specification's recommendation).
    FAISS holds only vectors and integer ids; SQLite holds everything else and is
    the authority for project filtering, file removal and incremental updates.
  * Ids are assigned by this class and mapped 1:1 to FAISS ids, so a FAISS hit can
    always be resolved back to a chunk.

Vectors are stored float32. FAISS's Python bindings reject float64, and casting
implicitly would hide a caller's mistake, so the dtype is checked explicitly.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import faiss
import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    faiss_id      INTEGER PRIMARY KEY,
    chunk_id      TEXT UNIQUE NOT NULL,
    project       TEXT NOT NULL,
    source_path   TEXT NOT NULL,
    content_type  TEXT,
    language      TEXT,
    start_line    INTEGER,
    end_line      INTEGER,
    content       TEXT NOT NULL,
    symbol        TEXT,
    strategy      TEXT,
    commit_hash   TEXT,
    mtime         TEXT,
    content_hash  TEXT,
    file_hash     TEXT,
    indexed_at    REAL
);
CREATE INDEX IF NOT EXISTS idx_chunks_project ON chunks(project);
CREATE INDEX IF NOT EXISTS idx_chunks_path    ON chunks(project, source_path);
CREATE INDEX IF NOT EXISTS idx_chunks_fileh   ON chunks(project, source_path, file_hash);

CREATE TABLE IF NOT EXISTS files (
    project      TEXT NOT NULL,
    source_path  TEXT NOT NULL,
    file_hash    TEXT NOT NULL,
    n_chunks     INTEGER,
    indexed_at   REAL,
    PRIMARY KEY (project, source_path)
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


@dataclass
class SearchHit:
    score: float
    chunk_id: str
    project: str
    source_path: str
    start_line: int
    end_line: int
    content: str
    symbol: str
    language: str
    content_type: str
    strategy: str

    def to_dict(self) -> dict:
        return {
            "score": round(float(self.score), 6),
            "chunk_id": self.chunk_id,
            "project": self.project,
            "source_path": self.source_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "symbol": self.symbol,
            "language": self.language,
            "content_type": self.content_type,
            "strategy": self.strategy,
            "content": self.content,
        }


def _ascii_safe(path: Path) -> bool:
    try:
        str(path).encode("ascii")
        return True
    except UnicodeEncodeError:
        return False


def faiss_write_index(index, path: Path) -> None:
    """
    Write a FAISS index, working around non-ASCII Windows paths.

    faiss-cpu 1.15.1 opens files through C++ std::fopen, which does not accept
    UTF-8 Windows paths: both write_index AND read_index fail with
    "could not open ...: No such file or directory" when any component contains
    non-ASCII characters (reproduced on this machine under C:\\Users\\<user>\...).
    The project's own directory is ASCII so it is unaffected, but a user with a
    non-ASCII TEMP or profile path would hit this, so we do not rely on it.

    Workaround: serialise to an in-memory buffer and let Python's own file I/O
    (which is Unicode-correct) do the writing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if _ascii_safe(path):
        faiss.write_index(index, str(path))
        return
    buf = faiss.serialize_index(index)
    path.write_bytes(bytes(buf))


def faiss_read_index(path: Path):
    """Read a FAISS index. Mirror of faiss_write_index for non-ASCII paths."""
    if _ascii_safe(path):
        return faiss.read_index(str(path))
    buf = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    return faiss.deserialize_index(buf)


class IndexStore:
    """
    FAISS index plus SQLite metadata.

    Thread safety: the HTTP service serves each request on its own thread, so a
    sqlite3 connection created on the main thread raises
    "SQLite objects created in a thread can only be used in that same thread".
    Two things fix that: check_same_thread=False, and a re-entrant lock around public
    operations so a concurrent request cannot interleave a read with a write. FAISS
    search itself is read-only and safe once built, but the metadata lookup that
    follows a search touches SQLite, so it is guarded too.
    """

    def __init__(self, directory: str | Path, dimension: int):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dimension = int(dimension)
        self.index_path = self.dir / "index.faiss"
        self.db_path = self.dir / "metadata.sqlite"
        self.index = faiss.IndexFlatIP(self.dimension)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        # Load an existing index automatically. This MUST happen in the constructor:
        # an earlier version only read the file inside the load() classmethod, so a
        # caller doing the obvious thing - IndexStore(dir, dim) - silently got an
        # EMPTY index while metadata held every chunk, and search returned nothing.
        # Loading here makes both entry points behave identically.
        if self.index_path.exists():
            self.index = faiss_read_index(self.index_path)
            if self.index.d != self.dimension:
                raise ValueError(
                    f"saved index dim {self.index.d} != requested {self.dimension} "
                    f"({self.index_path})")

    # ---------------------------------------------------------------- meta
    def set_meta(self, key: str, value) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)),
        )
        self.conn.commit()

    def get_meta(self, key: str, default=None):
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    # ---------------------------------------------------------------- add
    def add(self, vectors: np.ndarray, chunks: Sequence[dict],
            file_hashes: dict[str, str] | None = None) -> int:
        """
        Add vectors with their chunk metadata. Returns the number added.

        Vectors must be float32 and L2-normalised: the index is exact inner product,
        so un-normalised rows would silently rank by magnitude.
        """
        vectors = np.asarray(vectors)
        if vectors.dtype != np.float32:
            raise ValueError(f"vectors must be float32, got {vectors.dtype}")
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"expected (n,{self.dimension}) vectors, got {vectors.shape}")
        if len(vectors) != len(chunks):
            raise ValueError(f"{len(vectors)} vectors for {len(chunks)} chunks")
        if len(vectors) == 0:
            return 0

        norms = np.linalg.norm(vectors, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError(
                "vectors are not L2-normalised (norms "
                f"{norms.min():.4f}..{norms.max():.4f}); cosine ranking would be wrong"
            )

        # De-duplicate by chunk_id BEFORE touching FAISS.
        #
        # chunk_id is UNIQUE and the metadata insert uses INSERT OR REPLACE, so two
        # vectors carrying the same chunk_id would ADD two vectors but keep one row -
        # the index would end up with more vectors than chunks and every faiss_id above
        # the collision would point at the wrong chunk. That is exactly how a 1179-vs-
        # 1173 drift appeared in practice (the same file content reached the builder by
        # two routes). Keeping the LAST occurrence matches INSERT OR REPLACE semantics.
        dedup: dict[str, int] = {}
        for i, c in enumerate(chunks):
            dedup[c["chunk_id"]] = i
        keep = sorted(dedup.values())

        start_id = self.index.ntotal
        self.index.add(vectors[keep].astype(np.float32))

        now = time.time()
        rows = []
        for offset, i in enumerate(keep):
            c = chunks[i]
            fid = start_id + offset
            fh = (file_hashes or {}).get(c.get("source_path", ""), "")
            rows.append((
                fid, c["chunk_id"], c["project"], c["source_path"],
                c.get("content_type"), c.get("language"),
                int(c.get("start_line", 0)), int(c.get("end_line", 0)),
                c["content"], c.get("symbol", ""), c.get("strategy", ""),
                c.get("commit_hash", ""), c.get("mtime", ""),
                c.get("content_hash", ""), fh, now,
            ))
        self.conn.executemany(
            "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

        if file_hashes:
            counts: dict[str, int] = {}
            for c in chunks:
                counts[c["source_path"]] = counts.get(c["source_path"], 0) + 1
            for path, h in file_hashes.items():
                self.conn.execute(
                    "INSERT OR REPLACE INTO files VALUES (?,?,?,?,?)",
                    (chunks[0]["project"] if chunks else "", path, h,
                     counts.get(path, 0), now),
                )
        self.conn.commit()
        return len(rows)

    # ---------------------------------------------------------------- delete
    def remove_file(self, project: str, source_path: str) -> int:
        """
        Remove a file's chunks from both the metadata table and the FAISS index.

        FAISS IndexFlatIP has no delete, so the index is rebuilt from the surviving
        vectors. That is exact and, at Phase 4 scale, cheap.
        """
        ids = [r[0] for r in self.conn.execute(
            "SELECT faiss_id FROM chunks WHERE project=? AND source_path=?",
            (project, source_path)).fetchall()]
        if not ids:
            self.conn.execute("DELETE FROM files WHERE project=? AND source_path=?",
                              (project, source_path))
            self.conn.commit()
            return 0

        self.conn.execute("DELETE FROM chunks WHERE project=? AND source_path=?",
                          (project, source_path))
        self.conn.execute("DELETE FROM files WHERE project=? AND source_path=?",
                          (project, source_path))
        self.conn.commit()
        self._rebuild_index()
        return len(ids)

    def _rebuild_index(self) -> None:
        """Rebuild FAISS from whatever vectors are still referenced by metadata."""
        rows = self.conn.execute(
            "SELECT faiss_id FROM chunks ORDER BY faiss_id").fetchall()
        # We keep a parallel copy of vectors so a rebuild never needs the model.
        vec_path = self.dir / "vectors.npy"
        if vec_path.exists():
            all_vecs = np.load(vec_path)
        else:
            raise RuntimeError("vectors.npy missing; cannot rebuild index")
        keep = [r[0] for r in rows]
        new = faiss.IndexFlatIP(self.dimension)
        if keep:
            sub = all_vecs[keep].astype(np.float32)
            new.add(sub)
            # renumber so faiss_id == row position
            remap = {old: i for i, old in enumerate(sorted(keep))}
            self.conn.execute("BEGIN")
            for old, newid in remap.items():
                self.conn.execute("UPDATE chunks SET faiss_id=? WHERE faiss_id=?",
                                  (newid, old))
            self.conn.commit()
            np.save(vec_path, sub)
        else:
            np.save(vec_path, np.zeros((0, self.dimension), dtype=np.float32))
        self.index = new

    def persist_vectors(self) -> None:
        """
        Store one vector per metadata row, in row order.

        vectors.npy is the repair source for _rebuild_from_vectors, which assumes row i
        of the file corresponds to the i-th chunk by rowid. Writing THE WHOLE index
        would break that assumption as soon as index.ntotal exceeded the chunk count
        (which happens when a reset left a stale index file behind), and the repair
        would then either fail or - worse - pair chunks with the wrong vectors.

        So the write is driven by the metadata, and the row counts are asserted.
        """
        vec_path = self.dir / "vectors.npy"
        n_meta = int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        if n_meta == 0:
            np.save(vec_path, np.zeros((0, self.dimension), dtype=np.float32))
            return
        if int(self.index.ntotal) != n_meta:
            raise RuntimeError(
                f"refusing to persist vectors: index has {self.index.ntotal} vectors but "
                f"metadata has {n_meta} chunks; call verify_and_repair() first")
        all_vecs = self.index.reconstruct_n(0, self.index.ntotal).astype(np.float32)
        np.save(vec_path, all_vecs)

    def verify_and_repair(self, verbose: bool = False) -> dict:
        """
        Ensure the FAISS index and the metadata table agree, repairing if not.

        They can drift when the corpus directory is manipulated outside the scanner
        (e.g. wiped and re-copied): content-identical files are correctly reported as
        "unchanged" and so are never re-added, while their metadata rows persist from
        an earlier index that no longer holds matching vectors. The result is an index
        that returns fewer hits than the metadata claims - search silently degrades.

        Repair is possible because vectors.npy keeps a copy of every vector, so this
        needs neither the model nor re-embedding.
        """
        n_vec = int(self.index.ntotal)
        n_meta = int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        state = {"vectors": n_vec, "chunks": n_meta, "repaired": False}
        if n_vec == n_meta:
            if verbose:
                print(f"  index consistent: {n_vec} vectors == {n_meta} chunks")
            return state

        if verbose:
            print(f"  INCONSISTENT: {n_vec} vectors vs {n_meta} chunks - repairing")
        self._rebuild_from_vectors()
        n_vec2 = int(self.index.ntotal)
        state["vectors_after"] = n_vec2
        state["repaired"] = True
        state["consistent_after"] = (n_vec2 == n_meta)
        if verbose:
            print(f"  after repair: {n_vec2} vectors vs {n_meta} chunks "
                  f"(consistent={state['consistent_after']})")
        return state

    def _rebuild_from_vectors(self) -> None:
        """
        Rebuild FAISS so vector positions match chunk row order exactly.

        Vectors are reconstructed from the LIVE index when its count matches the
        metadata, so no side file is needed. vectors.npy is deliberately NOT the
        primary source: it is only written by save() at the end of a build, so during
        a build it is always stale or empty, and an earlier version of this method
        failed (or would have mis-paired chunks with vectors) for exactly that reason.

        If the live index holds FEWER vectors than the metadata, the missing vectors no
        longer exist anywhere and the only honest outcome is an error telling the caller
        to rebuild from scratch - a guessed pairing would silently corrupt retrieval.
        """
        rows = self.conn.execute(
            "SELECT rowid, faiss_id FROM chunks ORDER BY rowid").fetchall()
        n_rows = len(rows)
        n_vec = int(self.index.ntotal)
        new = faiss.IndexFlatIP(self.dimension)

        if n_rows == 0:
            self.index = new
            return
        if n_vec < n_rows:
            raise RuntimeError(
                f"cannot repair: index holds {n_vec} vectors but metadata has {n_rows} "
                f"chunks, so {n_rows - n_vec} vectors no longer exist. Rebuild from "
                f"scratch (build_index.py --reset).")
        if n_vec > n_rows:
            # More vectors than chunks: the first n_rows correspond to chunk rowids
            # because ids are assigned sequentially as chunks are added.
            all_vecs = self.index.reconstruct_n(0, n_rows).astype(np.float32)
        else:
            all_vecs = self.index.reconstruct_n(0, n_vec).astype(np.float32)

        norms = np.linalg.norm(all_vecs, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            all_vecs = all_vecs / np.where(norms == 0, 1.0, norms)[:, None]
        new.add(all_vecs)
        self.index = new
        self.conn.execute("BEGIN")
        for i, (rowid, _old) in enumerate(rows):
            self.conn.execute("UPDATE chunks SET faiss_id=? WHERE rowid=?", (i, rowid))
        self.conn.commit()

    def reset(self, verbose: bool = False) -> None:
        """
        Discard the index and ALL metadata, leaving an empty store.

        This is the honest escape hatch when repair is impossible - specifically when
        vectors.npy holds fewer rows than the metadata claims, meaning the vectors for
        some chunks no longer exist anywhere and those chunks can never be retrieved.
        A partial repair would silently misalign ids, so a full re-embed is safer than
        a clever recovery.

        index.faiss is written IMMEDIATELY as an empty index. Deleting it is not
        enough: if the process dies before the following build reaches save(), the next
        run would load a stale index file from a previous generation and the store
        would come up describing chunks it has no vectors for.
        """
        n_meta = int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        self.index = faiss.IndexFlatIP(self.dimension)
        self.conn.executescript("DELETE FROM chunks; DELETE FROM files; DELETE FROM meta;")
        self.conn.commit()
        for p in (self.dir / "vectors.npy",):
            if p.exists():
                p.unlink()
        faiss_write_index(self.index, self.index_path)
        np.save(self.dir / "vectors.npy", np.zeros((0, self.dimension), dtype=np.float32))
        if verbose:
            print(f"  reset: discarded {n_meta} chunks and {n_meta} vectors "
                  f"(empty index written)")

    # ---------------------------------------------------------------- search
    def search(self, query_vec: np.ndarray, top_k: int = 5,
               project: str | None = None) -> list[SearchHit]:
        with self._lock:
            return self._search_locked(query_vec, top_k, project)

    def _search_locked(self, query_vec: np.ndarray, top_k: int,
                       project: str | None) -> list[SearchHit]:
        q = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dimension:
            raise ValueError(f"query dim {q.shape[1]} != index dim {self.dimension}")
        n = self.index.ntotal
        if n == 0:
            return []
        # over-fetch when filtering by project so the filter does not starve results
        fetch = min(n, max(top_k * 8, top_k)) if project else min(n, top_k)
        scores, ids = self.index.search(q, fetch)
        scores, ids = scores[0], ids[0]

        hits: list[SearchHit] = []
        for score, fid in zip(scores, ids):
            if fid < 0:
                continue
            row = self.conn.execute(
                """SELECT chunk_id, project, source_path, start_line, end_line, content,
                          symbol, language, content_type, strategy
                   FROM chunks WHERE faiss_id=?""", (int(fid),)).fetchone()
            if row is None:
                continue
            if project and row[1] != project:
                continue
            hits.append(SearchHit(
                score=float(score), chunk_id=row[0], project=row[1], source_path=row[2],
                start_line=row[3], end_line=row[4], content=row[5], symbol=row[6] or "",
                language=row[7] or "", content_type=row[8] or "", strategy=row[9] or "",
            ))
            if len(hits) >= top_k:
                break
        return hits

    # ---------------------------------------------------------------- files
    def file_hash(self, project: str, source_path: str) -> str | None:
        row = self.conn.execute(
            "SELECT file_hash FROM files WHERE project=? AND source_path=?",
            (project, source_path)).fetchone()
        return row[0] if row else None

    def known_files(self, project: str) -> set[str]:
        return {r[0] for r in self.conn.execute(
            "SELECT source_path FROM files WHERE project=?", (project,)).fetchall()}

    def stats(self) -> dict:
        n = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        projects = [r[0] for r in self.conn.execute(
            "SELECT DISTINCT project FROM chunks").fetchall()]
        return {
            "vectors_in_index": int(self.index.ntotal),
            "chunks_in_metadata": int(n),
            "projects": projects,
            "dimension": self.dimension,
            "consistent": int(self.index.ntotal) == int(n),
        }

    # ---------------------------------------------------------------- persist
    def save(self) -> None:
        # Re-create the directory: it can legitimately have been removed or never
        # created (temp dirs, a fresh checkout, a caller that passed a nested path).
        # faiss.write_index fails with an opaque C++ error if the parent is missing,
        # so we make it explicit here.
        self.dir.mkdir(parents=True, exist_ok=True)
        self.persist_vectors()
        faiss_write_index(self.index, self.index_path)
        self.conn.commit()

    @classmethod
    def load(cls, directory: str | Path, dimension: int) -> "IndexStore":
        """
        Open an existing index. The constructor already loads index.faiss, so this is
        now a thin, explicit alias - kept because call sites read better as
        `IndexStore.load(...)` than as a bare constructor when reuse is intended.
        """
        return cls(directory, dimension)

    def close(self) -> None:
        """
        Close the metadata connection. Must be called before deleting the directory:
        an open SQLite handle keeps metadata.sqlite locked on Windows and blocks
        cleanup with WinError 32.
        """
        try:
            self.conn.commit()
        except Exception:
            pass
        try:
            self.conn.close()
        except Exception:
            pass

    def __enter__(self) -> "IndexStore":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False


# --------------------------------------------------------------------------
# normalisation contract
# --------------------------------------------------------------------------

def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation, safe for zero rows."""
    v = np.asarray(vectors, dtype=np.float32)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norms == 0, 1.0, norms)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reference cosine similarity, used to verify the FAISS path."""
    return l2_normalize(a) @ l2_normalize(b).T


