"""
retriever.py - query -> top-k chunks, with context packaging.

Two responsibilities:

  retrieve()  embed a query (optionally instruction-prefixed) and search FAISS.
  pack()      format hits into a context block for an upstream model.

pack() exists so retrieved content is never confused with the model's own knowledge:
every source is labelled with its path, line range, score and chunk id, and the block
is explicitly delimited. The specification asks for exactly this, and it is also what
makes a downstream hallucination checkable.
"""

from __future__ import annotations

from dataclasses import dataclass

from embedder import CODE_INSTRUCT, RETRIEVAL_INSTRUCT, Embedder
from index_store import IndexStore, SearchHit

DOC_INSTRUCT = RETRIEVAL_INSTRUCT


@dataclass
class RetrieveConfig:
    top_k: int = 5
    instruction: str | None = RETRIEVAL_INSTRUCT
    project: str | None = None
    # Per-chunk character cap when packing. Raised from 2000 to 8000 (the chunker's
    # own maximum is ~900 tokens, so 8000 chars effectively never truncates): a
    # 2000-char cap was cutting the answer out of otherwise well-retrieved chunks.
    # The RAG experiment showed `config/local-fast.yaml` ranking #1 for
    # "what context size does local-fast use by default?" while the packed text had
    # the `context:` line truncated away, so the model correctly reported the answer
    # as absent. Truncation silently converts a retrieval success into a miss.
    max_chars_per_chunk: int = 8000
    # Optional ceiling on the whole packed block, so a large top_k cannot blow up an
    # upstream model's prompt budget. 0 disables it.
    max_total_chars: int = 0


class Retriever:
    def __init__(self, embedder: Embedder, store: IndexStore, cfg: RetrieveConfig | None = None):
        self.emb = embedder
        self.store = store
        self.cfg = cfg or RetrieveConfig()

    def retrieve(self, query: str, *, top_k: int | None = None,
                 instruction: str | None = ..., project: str | None = None,
                 ) -> list[SearchHit]:
        """
        Return the top-k chunks for a query.

        `instruction=None` means "no instruction"; omitting the argument uses the
        configured default. That distinction matters for the A/B benchmark, so it is
        expressed with an Ellipsis sentinel rather than by overloading None.
        """
        k = top_k if top_k is not None else self.cfg.top_k
        instr = self.cfg.instruction if instruction is ... else instruction
        proj = project if project is not None else self.cfg.project
        qv = self.emb.encode_queries([query], instruction=instr)
        return self.store.search(qv[0], top_k=k, project=proj)

    def pack(self, hits: list[SearchHit], *, max_chars: int | None = None,
             max_total_chars: int | None = None) -> str:
        """
        Format hits as a labelled context block.

        Deliberately not a bare concatenation: each source is numbered and carries its
        path, line range and score so a reader (human or model) can attribute every
        statement and verify it against the source.

        Truncation is the LAST option, not the first. An earlier version capped every
        chunk at 2000 characters, which cut the answering line out of correctly
        retrieved chunks and made the model report "not in sources" for facts that had
        in fact been retrieved. Per-chunk truncation now defaults to 8000 characters,
        which exceeds what the chunker produces, so it is effectively disabled.
        """
        if not hits:
            return "NO SOURCES RETRIEVED."
        limit = max_chars or self.cfg.max_chars_per_chunk
        total_cap = max_total_chars if max_total_chars is not None else self.cfg.max_total_chars

        parts = []
        used = 0
        for i, h in enumerate(hits, 1):
            body = h.content
            if limit and len(body) > limit:
                body = body[:limit] + "\n... [truncated]"
            sym = f"\nsymbol: {h.symbol}" if h.symbol else ""
            block = (
                f"SOURCE {i}\n"
                f"path: {h.source_path}\n"
                f"lines: {h.start_line}-{h.end_line}\n"
                f"score: {h.score:.4f}\n"
                f"chunk_id: {h.chunk_id}{sym}\n"
                f"content:\n{body}"
            )
            if total_cap and used + len(block) > total_cap and parts:
                parts.append(f"[remaining {len(hits) - i + 1} source(s) omitted: "
                             f"context budget {total_cap} chars reached]")
                break
            parts.append(block)
            used += len(block)

        header = ("The following sources were retrieved from the project. They are the "
                  "only project facts available to you. If the answer is not present, "
                  "say so instead of guessing.\n")
        return header + "\n\n" + "\n\n".join(parts)

    def retrieve_and_pack(self, query: str, **kw) -> tuple[list[SearchHit], str]:
        hits = self.retrieve(query, **kw)
        return hits, self.pack(hits)


def instruction_for(query_type: str) -> str | None:
    """Map a query type to the official instruction (or None for no instruction)."""
    q = (query_type or "").lower()
    if q in ("none", "off", ""):
        return None
    if q in ("code", "code-concept", "programming"):
        return CODE_INSTRUCT
    return DOC_INSTRUCT
