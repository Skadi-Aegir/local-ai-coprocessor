"""
embedder.py - local-embedding backend.

Wraps Qwen3-Embedding-0.6B via sentence-transformers with:
  * GPU inference on CUDA
  * configurable Matryoshka output dimension
  * query-side instruction prefixing (the official recommendation: instructions on
    queries only, NEVER on documents)
  * batching

Correctness notes established by probe_embedding.py on this model:
  * pooling is LAST-TOKEN (1_Pooling/config.json: pooling_mode_lasttoken = true)
  * the repo ships a 2_Normalize module, so encode() already returns unit-length
    vectors. We therefore do NOT blindly re-normalise; we verify and re-normalise
    only if a vector is meaningfully off unit length.
  * Matryoshka truncation is done by slicing then re-normalising, because a sliced
    unit vector is no longer unit length.
"""

from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_PATH = str(Path(__file__).resolve().parents[2] / r"models\embedding\Qwen3-Embedding-0.6B")

# Official retrieval instruction from the Qwen3-Embedding model card.
RETRIEVAL_INSTRUCT = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
CODE_INSTRUCT = (
    "Given a programming question, retrieve code or documentation relevant to solving it"
)

# sanity threshold for "is this vector already unit length"
UNIT_TOL = 0.01


def format_query(query: str, instruction: str | None) -> str:
    """
    Apply an instruction to a QUERY only.

    The official format is:
        Instruct: {task}\nQuery:{query}
    """
    if not instruction:
        return query
    return f"Instruct: {instruction}\nQuery:{query}"


@dataclass
class EmbedderConfig:
    model_path: str = DEFAULT_MODEL_PATH
    device: str = "cuda"
    dimension: int = 1024          # Matryoshka output dimension
    batch_size: int = 32
    max_seq_length: int = 8192     # keep well below the 32K limit for speed/memory
    dtype: str = "float32"
    # OPEN_ISSUES P1-11: after an encode, hand the allocator's cached blocks back to the driver
    # once they exceed this many MiB above what is actually allocated. 0 disables the release.
    release_cache_mb: int = 256
    extra: dict = field(default_factory=dict)


class Embedder:
    def __init__(self, cfg: EmbedderConfig | None = None):
        self.cfg = cfg or EmbedderConfig()
        self._lock = threading.Lock()
        self._releases = 0
        self._released_mb = 0.0
        t0 = time.time()
        self.model = SentenceTransformer(
            self.cfg.model_path, device=self.cfg.device, trust_remote_code=True
        )
        if self.cfg.max_seq_length:
            self.model.max_seq_length = self.cfg.max_seq_length
        self.load_s = round(time.time() - t0, 2)
        # sentence-transformers 6.x renamed this; fall back for older versions
        try:
            self.native_dim = int(self.model.get_embedding_dimension())
        except AttributeError:
            self.native_dim = int(self.model.get_sentence_embedding_dimension())
        if not (32 <= self.cfg.dimension <= self.native_dim):
            raise ValueError(
                f"dimension must be 32..{self.native_dim}, got {self.cfg.dimension}"
            )

    # ---------------------------------------------------------------- info
    @property
    def device(self) -> str:
        return str(self.model.device)

    def vram_mb(self) -> float | None:
        """
        total - free from cudaMemGetInfo. Kept under its old name because existing scripts print
        it, but it is neither of the things it is easily taken for. Measured on this Windows (WDDM)
        machine (benchmarks/p1_11): with only a CUDA context in the process it read 1232.6 MiB
        while nvidia-smi reported 3438-3469 MiB used, and it then rose one-for-one with this
        process's allocations. So it is NOT the device-wide figure (use nvidia-smi for the margin)
        and NOT this process's footprint (use memory_stats()["torch_reserved_mb"]).
        """
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        return round((total - free) / 2**20, 1)

    def memory_stats(self) -> dict | None:
        """This process's PyTorch allocator, next to the device-wide figures."""
        if not torch.cuda.is_available():
            return None
        free, total = torch.cuda.mem_get_info()
        mib = 2**20
        return {
            "device_used_mb": round((total - free) / mib, 1),
            "device_free_mb": round(free / mib, 1),
            "torch_allocated_mb": round(torch.cuda.memory_allocated() / mib, 1),
            "torch_reserved_mb": round(torch.cuda.memory_reserved() / mib, 1),
            "torch_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / mib, 1),
            "cache_releases": self._releases,
            "released_mb_total": round(self._released_mb, 1),
            "release_threshold_mb": self.cfg.release_cache_mb,
        }

    def _release_cache(self) -> None:
        """
        Return the allocator's cached blocks to the driver after an encode (OPEN_ISSUES P1-11).

        A document batch pads every text to the longest one, and the activations of a long batch
        are far larger than the weights. PyTorch's caching allocator keeps those blocks reserved
        after the batch is done, so the process grows to its high-water mark and stays there:
        Phase A and B saw /stats vram_mb go 2361 -> 7751 MiB after a 520-chunk build, free VRAM
        below the 2048 MiB margin, and only a restart gave it back.

        Measured in isolation on that exact corpus (benchmarks/p1_11, 16 texts per request): with
        the release off, 5053.5 MiB stayed reserved above the weights after the last request;
        with it on, 15.5 MiB. Same wall time (22.18 s / 22.49 s) and bit-identical vectors. The
        release also lowers the peak (5062 -> 3752 MiB above the weights), because blocks cached
        from one request no longer fragment the next.
        """
        if self.cfg.release_cache_mb <= 0 or not torch.cuda.is_available():
            return
        cached = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        if cached > self.cfg.release_cache_mb * 2**20:
            torch.cuda.empty_cache()
            self._releases += 1
            self._released_mb += cached / 2**20

    def info(self) -> dict:
        return {
            "model_path": self.cfg.model_path,
            "device": self.device,
            "load_s": self.load_s,
            "native_dim": self.native_dim,
            "output_dim": self.cfg.dimension,
            "max_seq_length": self.model.max_seq_length,
            "batch_size": self.cfg.batch_size,
            "vram_mb": self.vram_mb(),
        }

    def close(self) -> None:
        """
        Release the model and free its memory.

        Needed because a sweep that constructs one Embedder per configuration leaks a
        full model each time: the GPU allocations are held until the interpreter exits.
        On this 32 GB machine that exhausted the commit limit and the next load failed
        with "OSError: [WinError 1455] 椤甸潰鏂囦欢澶皬, 鏃犳硶瀹屾垚鎿嶄綔" - a confusing error
        whose real cause was the previous model never being released.
        """
        try:
            del self.model
        except AttributeError:
            pass
        self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def __enter__(self) -> "Embedder":
        return self

    def __exit__(self, *exc) -> bool:
        self.close()
        return False

    # ---------------------------------------------------------------- encode
    def _postprocess(self, arr: np.ndarray) -> np.ndarray:
        """
        Truncate to the configured dimension and return exact unit vectors.

        Always normalise. An earlier version used a tolerance and skipped
        normalisation when the vector was "close enough" to unit length; that meant
        the 1024-dim path returned norms of 0.9987..1.0016 while the 512/256 paths
        happened to return exactly 1.0, so cosine scores were silently inconsistent
        across dimensions. Unit length is cheap to enforce and is a stated invariant,
        so the tolerance was removed.
        """
        arr = np.asarray(arr, dtype=np.float32)
        if self.cfg.dimension < arr.shape[1]:
            arr = arr[:, : self.cfg.dimension]
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1.0, norms)

    def encode_documents(self, texts: list[str], batch_size: int | None = None) -> np.ndarray:
        """Encode documents. Documents get NO instruction."""
        return self._encode(texts, batch_size)

    def encode_queries(
        self, texts: list[str], instruction: str | None = None,
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode queries, optionally with a query-side instruction."""
        prepared = [format_query(t, instruction) for t in texts]
        return self._encode(prepared, batch_size)

    def _encode(self, texts: list[str], batch_size: int | None) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.cfg.dimension), dtype=np.float32)
        bs = batch_size or self.cfg.batch_size
        with self._lock:                      # torch module is not thread-safe here
            with torch.inference_mode():
                out = self.model.encode(
                    texts,
                    batch_size=bs,
                    convert_to_numpy=True,
                    normalize_embeddings=False,   # we normalise deliberately below
                    show_progress_bar=False,
                )
            self._release_cache()
        return self._postprocess(out)

    # ---------------------------------------------------------------- bench
    def bench(self, texts: list[str], batch_size: int) -> dict:
        t0 = time.time()
        arr = self.encode_documents(texts, batch_size=batch_size)
        dt = time.time() - t0
        return {
            "batch_size": batch_size,
            "chunks": len(texts),
            "seconds": round(dt, 3),
            "chunks_per_s": round(len(texts) / dt, 1) if dt else None,
            "dimension": int(arr.shape[1]),
        }


def self_test() -> int:
    """
    Unit checks for the properties this module claims.

    Asserts rather than prints, so it can gate a build.
    """
    cfg = EmbedderConfig(dimension=1024, batch_size=8, max_seq_length=512)
    emb = Embedder(cfg)
    fails = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"{'OK  ' if cond else 'FAIL'} {name}{(' :: ' + detail) if detail else ''}")
        if not cond:
            fails.append(name)

    docs = ["alpha document about pagination",
            "beta document about vram sampling",
            "gamma document about chunking"]
    q = ["how does pagination work"]

    d_full = emb.encode_documents(docs)
    check("document shape is (n, dimension)", d_full.shape == (3, 1024), str(d_full.shape))
    check("documents are unit length",
          bool(np.allclose(np.linalg.norm(d_full, axis=1), 1.0, atol=1e-3)),
          str(np.linalg.norm(d_full, axis=1).round(5).tolist()))
    check("dtype is float32", d_full.dtype == np.float32, str(d_full.dtype))

    # Matryoshka slicing must still yield unit vectors
    for dim in (512, 256):
        e = Embedder(EmbedderConfig(dimension=dim, batch_size=8, max_seq_length=512))
        a = e.encode_documents(docs)
        check(f"dim {dim} output shape", a.shape == (3, dim), str(a.shape))
        check(f"dim {dim} vectors are unit length",
              bool(np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-3)),
              str(np.linalg.norm(a, axis=1).round(5).tolist()))

    # instruction must actually change the query vector
    q_plain = emb.encode_queries(q, instruction=None)
    q_instr = emb.encode_queries(q, instruction=RETRIEVAL_INSTRUCT)
    cos = float(np.dot(q_plain[0], q_instr[0]))
    check("instruction changes the query vector", cos < 0.999, f"cosine={cos:.4f}")
    check("both remain unit length",
          abs(float(np.linalg.norm(q_instr[0])) - 1.0) < 1e-3)

    # documents must NOT receive the instruction
    before = emb.encode_documents(["plain text"])
    after = emb.encode_documents(["plain text"])
    check("document encoding is deterministic",
          bool(np.allclose(before, after, atol=1e-5)))

    # relevance ordering on a trivial case
    d = emb.encode_documents(["pagination and page bounds",
                              "gpu memory sampling",
                              "markdown chunking by heading"])
    qv = emb.encode_queries(["pagination"], instruction=RETRIEVAL_INSTRUCT)
    sims = (d @ qv[0])
    check("most relevant doc ranks first", int(np.argmax(sims)) == 0, str(sims.round(3).tolist()))

    print()
    if fails:
        print(f"FAILED ({len(fails)}): {fails}")
        return 1
    print("embedder self-test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(self_test())

