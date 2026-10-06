"""
rerank_pipeline.py - two-stage retrieval: embedding -> rerank -> context pack.

This is the Phase 5 orchestration layer. It lives in the retrieval service because that
service owns the FAISS index and the chunk metadata; the reranker only ever sees plain
strings and returns indices into the array it was given. Mapping those indices back to
paths and line ranges happens HERE, which is what keeps the reranker testable in
isolation.

    query
      -> embedding           (8113, or in-process)
      -> Top-N candidates
      -> reranker            (8114)
      -> final Top-K
      -> context pack

Failure policy: if the reranker is unreachable the pipeline degrades to the dense top-K
and reports `reranker_used=false` instead of failing the request. Retrieval that works
without the reranker should not become unavailable because a second service is down.

Fallback is implemented by catching connection errors specifically, not by catching
everything: a bug in our own code must surface rather than silently masquerade as
"reranker unavailable".
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from index_store import SearchHit
from retriever import Retriever


@dataclass
class PipelineConfig:
    candidate_top_n: int = 20
    final_top_k: int = 5
    instruction_type: str = "general"
    rerank_url: str = "http://127.0.0.1:8114/v1/rerank"
    timeout_s: float = 120.0


@dataclass
class PipelineResult:
    query: str
    hits: list[SearchHit]
    reranker_used: bool = False
    fallback_reason: str | None = None
    candidate_count: int = 0
    rerank_scores: dict[int, float] = field(default_factory=dict)
    dense_order: list[int] = field(default_factory=list)
    reranked_order: list[int] = field(default_factory=list)
    embed_s: float = 0.0
    rerank_s: float = 0.0

    def pack(self, retriever: Retriever, **kw) -> str:
        """
        Context block that keeps the Phase 4 contract (path / lines / score / content)
        and ADDS the dense and rerank scores plus both ranks, as Phase 5 requires for
        debugging why a source did or did not make the cut.
        """
        if not self.hits:
            return "NO SOURCES RETRIEVED."
        limit = kw.get("max_chars") or retriever.cfg.max_chars_per_chunk
        dense_rank = {i: r + 1 for r, i in enumerate(self.dense_order)}
        rerank_rank = {i: r + 1 for r, i in enumerate(self.reranked_order)}
        parts = []
        for n, h in enumerate(self.hits, 1):
            body = h.content
            if limit and len(body) > limit:
                body = body[:limit] + "\n... [truncated]"
            idx = self.dense_order[n - 1] if n - 1 < len(self.dense_order) else None
            lines = [
                f"SOURCE {n}",
                f"path: {h.source_path}",
                f"lines: {h.start_line}-{h.end_line}",
                f"score: {h.score:.4f}",
                f"chunk_id: {h.chunk_id}",
            ]
            if self.reranker_used and idx is not None:
                lines.append(f"dense_rank: {dense_rank.get(idx, '?')}")
                lines.append(f"dense_score: {h.score:.4f}")
                lines.append(f"rerank_rank: {rerank_rank.get(idx, '?')}")
                rs = self.rerank_scores.get(idx)
                if rs is not None:
                    lines.append(f"rerank_score: {rs:.4f}")
            if h.symbol:
                lines.append(f"symbol: {h.symbol}")
            lines.append(f"content:\n{body}")
            parts.append("\n".join(lines))
        header = ("The following sources were retrieved from the project. They are the "
                  "only project facts available to you. If the answer is not present, "
                  "say so instead of guessing.\n")
        return header + "\n\n" + "\n\n".join(parts)


def rerank_http(query: str, documents: list[str], url: str, instruction_type: str,
                timeout: float = 120.0) -> dict:
    payload = {"query": query, "documents": documents,
               "instruction_type": instruction_type}
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def two_stage_retrieve(retriever: Retriever, query: str,
                       cfg: PipelineConfig | None = None) -> PipelineResult:
    cfg = cfg or PipelineConfig()
    import time

    if cfg.candidate_top_n <= 0 or cfg.final_top_k <= 0:
        # A degenerate configuration should not reach the reranker at all: asking it to
        # score zero documents is a pointless round trip, and some servers reject an empty
        # document list outright. Return early and say why.
        res = PipelineResult(query=query, hits=[], candidate_count=0,
                             fallback_reason="candidate_top_n or final_top_k is zero")
        return res

    t0 = time.time()
    candidates = retriever.retrieve(query, top_k=cfg.candidate_top_n, instruction=None)
    embed_s = time.time() - t0
    res = PipelineResult(query=query, hits=[], candidate_count=len(candidates),
                         dense_order=list(range(len(candidates))),
                         embed_s=embed_s)
    if not candidates:
        return res

    if not cfg.rerank_url:
        res.hits = candidates[: cfg.final_top_k]
        res.fallback_reason = "reranking disabled by configuration"
        return res

    docs = [c.content for c in candidates]
    t0 = time.time()
    try:
        rr = rerank_http(query, docs, cfg.rerank_url, cfg.instruction_type,
                         timeout=cfg.timeout_s)
    except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError,
            TimeoutError) as e:
        # transport failure only - degrade rather than fail the request
        res.hits = candidates[: cfg.final_top_k]
        res.reranker_used = False
        res.fallback_reason = f"{type(e).__name__}: {e}"
        res.rerank_s = time.time() - t0
        return res

    res.rerank_s = time.time() - t0
    order = [x["index"] for x in rr["results"]]
    res.rerank_scores = {x["index"]: x["score"] for x in rr["results"]}
    res.reranked_order = order
    res.reranker_used = True
    res.hits = []
    for i in order[: cfg.final_top_k]:
        h = candidates[i]
        # keep the DENSE score in `score` so callers that only understand one score are
        # not silently given a rerank probability in the same field
        res.hits.append(SearchHit(
            score=h.score, chunk_id=h.chunk_id, project=h.project,
            source_path=h.source_path, start_line=h.start_line, end_line=h.end_line,
            content=h.content, symbol=h.symbol, language=h.language,
            content_type=h.content_type, strategy=h.strategy))
    return res
