"""
ask.py - minimal local Q&A entry point (CLI).

Deliberately minimal: no Gateway, no HTTP layer, no reranker. Dense retrieval plus local-fast, with
citable sources and an explicit refusal when the evidence is not there.

  default corpus   the clean task-material index (config/, docs/, production sources)
  default model    local-fast (8111)
  reranker         OFF and its flag DISABLED - not yet integrated with the safe
                   service lifecycle, so enabling it here would load a model with no protection

Every answer is returned with the file and the line range the model was ACTUALLY SHOWN, not merely
the range of the chunk it came from. The CLI refuses rather than answering from model memory, and it
matters which mechanism does the refusing:

  * the evidence floor - no candidate above it means no generation at all. MEASURED: it has never
    fired. Every candidate in the P56 and Phase A runs scored 0.486 or higher, and the answer-absent
    questions scored as high as the answerable ones (top score 0.518-0.623 against 0.560-0.814).
    It guards against an empty or broken index; it is NOT what refuses answer-absent questions.
  * the model's own refusal - the prompt asks for the exact refusal sentence. Every answer-absent
    refusal recorded so far came from here, so it rests on the model following the instruction.
  * the citation gates - an answer with no valid in-range citation, or one citing a file that no
    longer matches the indexed version, is downgraded to a refusal.

The index is opened only if its build manifest passed validation, including the answer-key leak
check, and only if the index on disk is the one the manifest describes (see open_index).

Usage:
  python ask.py "your question"
  python ask.py --question-file qs.txt --out results.json --json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from http.client import HTTPException
from pathlib import Path

# This file lives at <repo>/cli/ask.py, so the repo root is ONE level up. The benchmark scripts
# under benchmarks/<dir>/ need two, and copying that pattern here silently resolved REPO to D:\ and
# broke the import with a misleading ModuleNotFoundError.
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "services" / "retrieval"))

from provenance import (HASH_SCHEME, LEGACY_HASH_SCHEME, content_hash,  # noqa: E402
                        line_count, match_recorded, raw_hash)

# The Phase A index, built directly from current file contents with a provenance manifest.
CLEAN_INDEX = REPO / "vector/qa/current"
DIM = 1024
EMBED_URL = "http://127.0.0.1:8113"
RERANK_URL = "http://127.0.0.1:8114"
FAST_URL = "http://127.0.0.1:8111"

# A chunk must clear this cosine similarity to be treated as evidence at all. This is an ADMISSION
# floor, not a quality score. See the module docstring: on every recorded run it admitted all five
# candidates, so it is a guard against a broken index rather than a calibrated relevance threshold.
EVIDENCE_FLOOR = 0.30
TOP_K = 5

# UPSTREAM TIMEOUTS, in seconds. Every upstream call used to carry a 600 s timeout, so one hung
# service held the caller - and in ask_server, the pipeline lock and with it every other request -
# for up to ten minutes per call. Measured on this path: retrieval 0.08-4.6 s, generation
# 0.3-5.1 s. These bounds sit an order of magnitude above that and turn a hang into an error.
EMBED_TIMEOUT_S = 30
TOKENIZE_TIMEOUT_S = 15
GENERATE_TIMEOUT_S = 90
RERANK_TIMEOUT_S = 60

# PACK BUDGET. local-fast serves ONE 8192-token slot (config/local-fast.yaml: context 8192,
# parallel 1), and the instructions, the sources, the question and the answer must all fit in it.
CONTEXT_TOKENS = 8192
# Raised 256 -> 512 (2026-10-05, AD-22). MEASURED BASIS: at 256 the answers to h05, p05 and p07 each
# consumed EXACTLY 256 tokens and ended mid-sentence, and nothing in the record showed it - the
# request never read back finish_reason. The cap was the binding constraint, not the model. 512
# doubles the headroom past that point while leaving PROMPT_TOKEN_BUDGET (7616) above every pack this
# corpus has produced. It was NOT tuned against the exposed question sets.
MAX_ANSWER_TOKENS = 512
TEMPLATE_RESERVE_TOKENS = 64          # chat-template markers the server wraps around the prompt
PROMPT_TOKEN_BUDGET = CONTEXT_TOKENS - MAX_ANSWER_TOKENS - TEMPLATE_RESERVE_TOKENS
# Per-chunk ceiling. The P56 pack cut every chunk at 2000 characters - the very cap
# services/retrieval/retriever.py had already removed after it cut `context: 8192` out of
# config/local-fast.yaml and turned a retrieval success into a refusal. Measured on the Phase A
# index: 48 of 520 chunks exceed 2000 characters and none exceeds 6456, so 8000 (the retriever's
# value) cuts nothing today. The token budget above is what bounds the prompt.
MAX_CHARS_PER_CHUNK = 8000
# Used only when local-fast cannot count tokens itself. The densest chunk measured on this corpus
# is 0.83 tokens per character, so the estimate errs towards dropping a source, never overflowing.
FALLBACK_TOKENS_PER_CHAR = 0.85

REFUSAL = "I could not find enough evidence in the local corpus to answer this."

# qa-v1 (P56 to Phase B) also said "a bare 'SOURCE n' is not counted as a citation". That was false:
# the parser has accepted the bare form since P56, because the model writes it reliably. The
# instruction no longer misstates the rule. Every record carries the version, because a prompt
# change makes runs on either side of it non-comparable.
PROMPT_VERSION = "qa-v2"
PROMPT = (
    "Answer the QUESTION using ONLY the SOURCES below.\n"
    "Every factual statement must be traceable to a source. Cite the source of each statement "
    "inline as [SOURCE n].\n"
    "If the sources do not contain the answer, reply EXACTLY with: {refusal}\n"
    "Do not use prior knowledge. Do not guess.\n\n"
    "{pack}\n\nQUESTION: {q}\n\nANSWER:"
)


class IndexUnusable(RuntimeError):
    """The index cannot be served: missing, unvalidated, or not the index its manifest describes."""


class PromptTooLong(ValueError):
    """The question alone leaves no room for evidence inside the model's context window."""


def _free_vram() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def classify_error(e: BaseException) -> str:
    """
    Name a pipeline failure so a caller can tell a dead upstream from a slow one from a defect.

      upstream_unavailable  the service refused, reset or dropped the connection
      upstream_timeout      the service accepted the request but did not answer in time
      upstream_http_error   the service answered with an HTTP error status
      prompt_too_long       the question leaves no room for evidence in the context window
      internal              anything else - a defect in this code
    """
    if isinstance(e, PromptTooLong):
        return "prompt_too_long"
    if isinstance(e, urllib.error.HTTPError):
        return "upstream_http_error"
    reason = e.reason if isinstance(e, urllib.error.URLError) else e
    if isinstance(reason, TimeoutError):
        return "upstream_timeout"
    if isinstance(e, urllib.error.URLError) or isinstance(reason, (ConnectionError, HTTPException)):
        return "upstream_unavailable"
    return "internal"


class IndexProvenance:
    """
    The build manifest of the index in use, plus citation verification against it.

    WHY THIS EXISTS. A citation is only trustworthy if the source it names is the version that was
    actually indexed. The manifest records a content hash per source file and its line count at
    build time, so each citation can be checked against the file on disk NOW:

      verified        the file's content matches the indexed version and the cited lines exist
      stale           the file has CHANGED since the index was built -> the citation may not
                      support the claim any more, and the answer is downgraded
      missing         the file no longer exists
      out_of_range    the cited lines do not exist in the file
      not_in_manifest the source is not recorded in the manifest
      unreadable      the file exists but could not be read
      unverifiable    there is no manifest to check against

    LINE ENDINGS DO NOT COUNT AS CHANGE. Manifests with `hash_scheme: sha256-lf-v1` record the hash
    of the LF-normalised bytes. Older manifests recorded raw bytes; for those a file that differs
    only in line endings from an LF original is still accepted, and the record says so. Without
    this, a fresh checkout under core.autocrlf=true reported every citation as stale.

    Note the separate, weaker rule this does NOT replace: having a legal source NUMBER is not
    evidence that the source supports the statement. This class verifies identity and range; the
    support question is answered per question by review.
    """

    def __init__(self, index_dir: Path):
        self.dir = Path(index_dir)
        self.path = self.dir / "manifest.json"
        self.manifest: dict | None = None
        self.error: str | None = None
        try:
            self.manifest = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"

    @property
    def version(self) -> str:
        if not self.manifest:
            return "unknown"
        return self.manifest.get("index_version", "unknown")

    @property
    def dirty(self) -> bool:
        return bool((self.manifest or {}).get("working_tree_dirty"))

    @property
    def scheme(self) -> str:
        return (self.manifest or {}).get("hash_scheme") or LEGACY_HASH_SCHEME

    @property
    def n_chunks(self) -> int | None:
        n = (self.manifest or {}).get("n_chunks")
        return n if isinstance(n, int) else None

    def problems(self) -> list[str]:
        """Reasons this manifest does not license serving its index; empty when it does."""
        if not self.manifest:
            return [f"no readable manifest ({self.error})"]
        out = []
        v = self.manifest.get("validation") or {}
        if v.get("validation_passed") is not True:
            out.append("the build manifest does not record a passing validation")
        # The rollback slot built before the leak check existed (current.prev, a3e3c050636f838e)
        # holds services/retrieval/rag_compare_rerank.py - the 12 tuning questions with their
        # facts=[ answer lists. An index whose manifest cannot show a passing leak check is not
        # served, whatever else it records.
        if v.get("no_answer_key_material") is not True:
            out.append("the build manifest records no passing answer-key leak check")
        if self.n_chunks is None:
            out.append("the build manifest does not record n_chunks")
        return out

    def verify(self, source_path: str, start_line: int, end_line: int) -> dict:
        out = {"source_path": source_path, "start_line": start_line, "end_line": end_line}
        if not self.manifest:
            out["status"] = "unverifiable"
            out["detail"] = f"no manifest: {self.error}"
            return out
        recorded = (self.manifest.get("file_hashes") or {}).get(source_path)
        if recorded is None:
            out["status"] = "not_in_manifest"
            out["detail"] = "the source is not recorded in the index manifest"
            return out
        f = REPO / source_path
        if not f.is_file():
            out["status"] = "missing"
            out["detail"] = "the source file no longer exists"
            return out
        try:
            raw = f.read_bytes()
        except Exception as e:  # noqa: BLE001
            out["status"] = "unreadable"
            out["detail"] = f"{type(e).__name__}: {e}"
            return out
        basis = match_recorded(recorded, raw, self.scheme)
        n_lines = line_count(raw)
        out["hash_scheme"] = self.scheme
        out["indexed_hash"] = recorded[:16]
        out["current_hash"] = (content_hash(raw) if self.scheme == HASH_SCHEME
                               else raw_hash(raw))[:16]
        out["current_lines"] = n_lines
        if basis is None:
            out["status"] = "stale"
            out["detail"] = ("the file has CHANGED since this index was built, so the cited lines "
                             "may no longer support the statement")
            return out
        if not (1 <= start_line <= end_line <= n_lines):
            out["status"] = "out_of_range"
            out["detail"] = f"cited lines {start_line}-{end_line} exceed the file's {n_lines} lines"
            return out
        out["status"] = "verified"
        out["match"] = basis
        out["detail"] = "file content matches the indexed version and the cited lines exist"
        if basis != "hash":
            out["detail"] += " (this copy differs from the indexed one only in line endings)"
        return out


def open_index(index_dir: Path):
    """
    Open an index for serving, or raise IndexUnusable. Returns (IndexStore, IndexProvenance).

    Checked BEFORE IndexStore is touched, because IndexStore.load() creates a missing directory and
    an empty metadata.sqlite rather than failing: measured, a directory holding only manifest.json
    loaded with 0 vectors and `consistent: True`, and every question was then refused as "below
    the evidence floor" - indistinguishable from a genuinely answer-absent question, while /health
    reported the manifest's version as if the index were there.
    """
    index_dir = Path(index_dir)
    prov = IndexProvenance(index_dir)
    problems = prov.problems()
    missing = [n for n in ("index.faiss", "metadata.sqlite") if not (index_dir / n).is_file()]
    if missing:
        problems.append(f"index files missing: {', '.join(missing)}")
    if problems:
        raise IndexUnusable(f"{index_dir}: " + "; ".join(problems))

    from index_store import IndexStore
    try:
        store = IndexStore.load(index_dir, DIM)
    except Exception as e:  # noqa: BLE001
        raise IndexUnusable(f"{index_dir}: could not open the index ({type(e).__name__}: {e})")
    try:
        st = store.stats()
        if not (st.get("consistent") and st.get("vectors_in_index") == st.get("chunks_in_metadata")
                == prov.n_chunks):
            raise IndexUnusable(
                f"{index_dir}: the index on disk is not the one its manifest describes "
                f"(vectors {st.get('vectors_in_index')}, chunks {st.get('chunks_in_metadata')}, "
                f"manifest n_chunks {prov.n_chunks}, consistent {st.get('consistent')})")
    except BaseException:
        store.close()
        raise
    return store, prov


def _post(url: str, payload: dict, timeout: float) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


class ServiceEmbedder:
    """
    Query embedding via the ALREADY-RUNNING local-embedding service on 8113.

    This exists because loading an in-process Embedder creates a SECOND copy of the embedding
    model: with all three services resident, the CLI's own copy dropped free VRAM to **1942 MiB**,
    below the 2048 MiB margin. The service is already resident and holds the same model, so using
    it costs no additional VRAM and keeps the margin intact.

    Duck-types the two methods the Retriever needs: `encode_queries` and `encode_documents`.
    """

    def __init__(self, url: str | None = None):
        self.url = url or EMBED_URL

    def encode_queries(self, texts: list[str], instruction: str | None = None,
                       batch_size: int | None = None):
        import numpy as np
        payload = {"model": "local-embedding", "input": list(texts), "query": True}
        if instruction:
            payload["instruction"] = instruction
        body = _post(self.url + "/v1/embeddings", payload, timeout=EMBED_TIMEOUT_S)
        vecs = [d["embedding"] for d in body["data"]]
        return np.asarray(vecs, dtype="float32")

    def encode_documents(self, texts: list[str], batch_size: int | None = None):
        import numpy as np
        body = _post(self.url + "/v1/embeddings",
                     {"model": "local-embedding", "input": list(texts)}, timeout=EMBED_TIMEOUT_S)
        return np.asarray([d["embedding"] for d in body["data"]], dtype="float32")

    def close(self) -> None:
        return None


class Retriever:
    """Dense retrieval over a provenance-checked index (see open_index)."""

    def __init__(self, index_dir: Path = CLEAN_INDEX, embedder: str = "service"):
        self.index_dir = Path(index_dir)
        # The manifest is captured with the index it describes and used for every verification,
        # so a citation is always checked against the build that produced the chunk.
        self.store, self.provenance = open_index(self.index_dir)
        # Read once, single-threaded: the served index is read-only, and IndexStore.stats() takes
        # no lock, so a status endpoint must not run it against a connection a search is using.
        self.stats_at_load = self.store.stats()
        self.embedder_kind = embedder
        self._emb = None

    def _get_embedder(self):
        if self._emb is None:
            if self.embedder_kind == "service":
                self._emb = ServiceEmbedder()
            else:
                from embedder import Embedder
                self._emb = Embedder()
        return self._emb

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        from retriever import RetrieveConfig, Retriever as R
        r = R(self._get_embedder(), self.store, RetrieveConfig(top_k=top_k, instruction=None))
        hits = r.retrieve(query, top_k=top_k)
        return [h.to_dict() for h in hits]

    def close(self) -> None:
        try:
            if self._emb is not None:
                self._emb.close()
        finally:
            self.store.close()


def _clip_lines(content: str, start_line: int, end_line: int,
                max_chars: int) -> tuple[str, int, bool]:
    """
    Cut a chunk at a LINE boundary to at most `max_chars`.

    Returns (text, last line actually included, partial_line). A chunk's text is exactly lines
    start..end of the file joined by newlines (verified byte-for-byte on all 520 Phase A chunks),
    so the included range is computable; a single line longer than the limit is cut inside the
    line and reported as such.
    """
    if len(content) <= max_chars:
        return content, end_line, False
    lines = content.split("\n")
    used, n = 0, 0
    for ln in lines:
        extra = len(ln) + (1 if n else 0)
        if used + extra > max_chars:
            break
        used += extra
        n += 1
    if n == 0:
        return lines[0][:max_chars], start_line, True
    return "\n".join(lines[:n]), start_line + n - 1, False


def _source_entry(n: int, h: dict, max_chars: int) -> dict:
    start, end = int(h.get("start_line") or 0), int(h.get("end_line") or 0)
    text, supplied_end, partial = _clip_lines(h.get("content") or "", start, end, max_chars)
    truncated = supplied_end < end or partial
    span = f"lines {start}-{supplied_end}"
    if truncated:
        span += f" of {start}-{end}, truncated"
    return {"source": n, "source_path": h.get("source_path"), "chunk_id": h.get("chunk_id"),
            "start_line": start, "end_line": supplied_end,
            "chunk_start_line": start, "chunk_end_line": end,
            "truncated": truncated, "partial_line": partial,
            "block": (f"SOURCE {n} ({h.get('source_path')} {span}, "
                      f"score {h.get('score', 0):.4f})\n{text}")}


def _count_tokens(text: str) -> int | None:
    """Exact token count from local-fast itself; None if the server has no /tokenize endpoint."""
    try:
        body = _post(f"{FAST_URL}/tokenize", {"content": text}, timeout=TOKENIZE_TIMEOUT_S)
    except urllib.error.HTTPError as e:
        if e.code in (404, 405, 501):
            return None
        raise
    toks = body.get("tokens")
    return len(toks) if isinstance(toks, list) else None


def build_prompt(question: str, kept: list[dict]) -> tuple[str, dict]:
    """
    Assemble the prompt within the model's context window.

    Sources keep their rank order and are included WHOLE (up to MAX_CHARS_PER_CHUNK). If the prompt
    exceeds PROMPT_TOKEN_BUDGET, the lowest-ranked sources are dropped - recorded, never silent -
    until it fits. Only if the top source alone does not fit is it cut, at a line boundary, and the
    citation then reports the lines the model actually saw.
    """
    entries = [_source_entry(i, h, MAX_CHARS_PER_CHUNK) for i, h in enumerate(kept, 1)]
    method = "tokenize"

    def measure(es: list[dict]) -> tuple[str, int]:
        nonlocal method
        text = PROMPT.format(refusal=REFUSAL, pack="\n\n".join(e["block"] for e in es), q=question)
        n = _count_tokens(text) if method == "tokenize" else None
        if n is None:
            method = "estimate"
            n = math.ceil(len(text) * FALLBACK_TOKENS_PER_CHAR)
        return text, n

    k = len(entries)
    prompt, tokens = measure(entries[:k])
    while tokens > PROMPT_TOKEN_BUDGET and k > 1:
        k -= 1
        prompt, tokens = measure(entries[:k])
    if tokens > PROMPT_TOKEN_BUDGET:
        # The top source alone does not fit: halve its character allowance until it does.
        top = kept[0]
        limit = len(top.get("content") or "")
        while tokens > PROMPT_TOKEN_BUDGET:
            limit //= 2
            if limit < 80:
                raise PromptTooLong(
                    f"the question leaves no room for evidence: {tokens} prompt tokens against a "
                    f"budget of {PROMPT_TOKEN_BUDGET}")
            entries[0] = _source_entry(1, top, limit)
            prompt, tokens = measure(entries[:1])
    supplied = [{k_: v for k_, v in e.items() if k_ != "block"} for e in entries[:k]]
    omitted = [{"source_path": h.get("source_path"), "start_line": h.get("start_line"),
                "end_line": h.get("end_line"), "chunk_id": h.get("chunk_id"),
                "reason": "context budget"} for h in kept[k:]]
    return prompt, {"supplied": supplied, "omitted": omitted, "prompt_tokens": tokens,
                    "budget_tokens": PROMPT_TOKEN_BUDGET, "count_method": method,
                    "pack_chars": sum(len(e["block"]) for e in entries[:k])}


def answer(question: str, *, rerank: bool = False, retriever: Retriever | None = None,
           top_k: int = TOP_K, embedder: str = "service", measure_vram: bool = True) -> dict:
    """
    One end-to-end question. Returns a record with the answer, citations and timings.

    `measure_vram` brackets the call with nvidia-smi. The CLI keeps it for its records; the HTTP
    interface turns it off, because two nvidia-smi calls per request added latency inside the
    pipeline lock and the result was never returned to the caller.
    """
    own = retriever is None
    retr = retriever or Retriever(embedder=embedder)
    prov = retr.provenance
    rec: dict = {"question": question, "rerank": rerank,
                 "models": {"generative": "local-fast", "embedder": embedder},
                 "top_k": top_k, "evidence_floor": EVIDENCE_FLOOR,
                 "prompt_version": PROMPT_VERSION,
                 "index_version": prov.version, "index_working_tree_dirty": prov.dirty}
    t_all = time.time()
    rec["free_vram_before_mb"] = _free_vram() if measure_vram else None

    def done(**fields) -> dict:
        rec.update(fields)
        # EVERY record carries these two, on every exit - the answer path, the citation-gate
        # downgrade, the evidence-floor refusal and the error path alike (AD-22). Defaults are None
        # and False, so a record that never reached generation says so instead of silently omitting
        # the keys; cli/ask_server.py relies on both being present to keep answers and refusals the
        # same shape.
        rec.setdefault("finish_reason", None)
        rec.setdefault("answer_truncated", False)
        rec["e2e_s"] = round(time.time() - t_all, 3)
        rec["free_vram_after_mb"] = _free_vram() if measure_vram else None
        return rec

    try:
        t0 = time.time()
        hits = retr.search(question, top_k=top_k)
        rec["retrieve_s"] = round(time.time() - t0, 3)
        rec["candidates"] = [{"source_path": h.get("source_path"),
                              "start_line": h.get("start_line"),
                              "end_line": h.get("end_line"),
                              "score": round(h.get("score", 0), 4),
                              "chunk_id": h.get("chunk_id")} for h in hits]

        kept = [h for h in hits if h.get("score", 0) >= EVIDENCE_FLOOR]
        rec["n_candidates"] = len(hits)
        rec["n_above_floor"] = len(kept)
        rec["max_score"] = round(max((h.get("score", 0) for h in hits), default=0.0), 4)

        # RERANKER IS OFF BY DEFAULT and only reachable by explicit request.
        if rerank and kept:
            try:
                t0 = time.time()
                rr = _post(f"{RERANK_URL}/v1/rerank",
                           {"query": question,
                            "documents": [(h.get("content") or "") for h in kept],
                            "top_n": min(5, len(kept)), "instruction_type": "general",
                            "use_cache": False}, timeout=RERANK_TIMEOUT_S)
                order = [x.get("index") for x in (rr.get("results") or [])]
                reordered = [kept[i] for i in order if isinstance(i, int) and i < len(kept)]
                if reordered:
                    kept = reordered
                rec["rerank_s"] = round(time.time() - t0, 3)
                rec["rerank_used"] = True
            except Exception as e:  # noqa: BLE001
                rec["rerank_error"] = f"{type(e).__name__}: {e}"
                rec["rerank_used"] = False
        else:
            rec["rerank_used"] = False

        # EVIDENCE GATE: refuse rather than answer from model memory.
        if not kept:
            return done(refused=True, answer=REFUSAL, citations=[],
                        reason=(f"no candidate above the {EVIDENCE_FLOOR} evidence floor "
                                f"(best score {rec['max_score']})"))

        prompt, pack = build_prompt(question, kept)
        rec["pack"] = pack
        rec["pack_chars"] = pack["pack_chars"]
        supplied = pack["supplied"]
        t0 = time.time()
        g = _post(f"{FAST_URL}/v1/chat/completions", {
            "model": "local-fast",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": MAX_ANSWER_TOKENS, "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False}}, timeout=GENERATE_TIMEOUT_S)
        rec["generate_s"] = round(time.time() - t0, 3)
        choice = (g.get("choices") or [{}])[0]
        text = (choice.get("message") or {}).get("content", "").strip()
        rec["usage"] = g.get("usage")
        # FINISH REASON, read back on every record (AD-22). Without this a reply cut off by the
        # max_tokens cap was indistinguishable from a finished one: h05/p05/p07 hit exactly 256
        # tokens and ended mid-sentence while the record looked clean. Recorded, not acted on - a
        # truncated answer is returned and flagged, never silently retried or suppressed.
        rec["finish_reason"] = choice.get("finish_reason")
        rec["answer_truncated"] = rec["finish_reason"] == "length"

        # EXACT refusal match, not substring. Measured reason: a substring test flags any answer
        # that QUOTES the refusal sentence while still answering - q04 does exactly that, and was
        # misreported as a refusal. A refusal is the whole reply and nothing else.
        refused = text.strip().lower() == REFUSAL.lower()
        # Accept BOTH citation forms. Measured: the model reliably writes a bare "SOURCE 3" even
        # when the prompt asks for "[SOURCE 3]", and a parser matching only the bracketed form
        # reported ZERO citations for an answer that was in fact correctly cited.
        # KNOWN LIMIT: any mention counts - "SOURCE 2 does not say" registers source 2. That is the
        # mention-versus-assertion weakness of the old grader (OPEN_ISSUES P2-1) in a new place;
        # the verifier below establishes identity and range, never support.
        cited_all = sorted({int(m) for m in re.findall(r"\[SOURCE\s*(\d+)\]", text)}
                           | {int(m) for m in re.findall(r"(?<!\[)\bSOURCE\s+(\d+)\b", text)})
        # IN-RANGE only: a number outside 1..len(supplied) names nothing the model was shown - a
        # source dropped for the context budget included - so it cannot support the answer.
        cited = [n for n in cited_all if 1 <= n <= len(supplied)]
        out_of_range = [n for n in cited_all if not (1 <= n <= len(supplied))]
        citations = []
        for n in cited:
            s = supplied[n - 1]
            citations.append({"source": n, "source_path": s["source_path"],
                              "start_line": s["start_line"], "end_line": s["end_line"],
                              "chunk_start_line": s["chunk_start_line"],
                              "chunk_end_line": s["chunk_end_line"],
                              "truncated": s["truncated"]})

        # NO VALID IN-RANGE CITATION -> REFUSE. An answer with no traceable source is not an
        # answer this entry point is allowed to return, however fluent it looks.
        if not refused and not citations:
            return done(refused=True, answer=REFUSAL, citations=[], unsupported_draft=text,
                        reported_source_numbers=cited_all,
                        out_of_range_source_numbers=out_of_range,
                        reason=("the model produced no valid in-range source citation; "
                                "refusing rather than returning an untraceable answer"))

        # ---- VERIFY each citation against the manifest captured with this index
        for c in citations:
            c["verification"] = prov.verify(c["source_path"], c["start_line"], c["end_line"])
        rec["citation_verification"] = [c["verification"]["status"] for c in citations]

        unverified = [c for c in citations if c["verification"]["status"] != "verified"]
        if unverified:
            # The citation points somewhere real but NOT at the version that was indexed, or the
            # file is gone: downgraded to a refusal-with-explanation rather than presented as fact.
            return done(refused=True, unsupported_draft=text,
                        answer=("I cannot give a confirmed answer: " + REFUSAL),
                        citations=citations,
                        stale_or_unverifiable_citations=[
                            {"source_path": c["source_path"],
                             "status": c["verification"]["status"],
                             "detail": c["verification"]["detail"]} for c in unverified],
                        reason=("one or more cited sources could not be verified against the "
                                "indexed version; the index may be stale"))

        # ATTACH THE ANSWER BEFORE RETURNING. An earlier version put these fields after a return
        # and they were never executed: every record came back verified but with NO answer text and
        # NO citations, and every question looked answered-but-empty.
        return done(refused=bool(refused), answer=text, citations=citations,
                    cited_source_numbers=cited, reported_source_numbers=cited_all,
                    out_of_range_source_numbers=out_of_range)
    except Exception as e:  # noqa: BLE001
        return done(refused=False, answer=None, citations=[],
                    error=f"{type(e).__name__}: {e}", error_kind=classify_error(e))
    finally:
        if own:
            retr.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="minimal local Q&A over the clean corpus")
    ap.add_argument("question", nargs="?", help="the question to ask")
    ap.add_argument("--question-file", help="one question per line")
    ap.add_argument("--out", help="write per-question JSON records here")
    ap.add_argument("--index", default=str(CLEAN_INDEX))
    ap.add_argument("--embedder", default="service", choices=["service", "in-process"],
                    help="service (default) reuses the running 8113 model and costs no extra VRAM. "
                         "in-process is DISABLED: it loads a second model copy and breaches the "
                         "2048 MiB margin (measured: free VRAM fell to 1942 MiB)")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--rerank", action="store_true",
                    help="DISABLED pending integration with the safe service lifecycle. The "
                         "reranker must be loaded and unloaded under the scheduler's reservation "
                         "and verified-unload rules; until then enabling it here would load a "
                         "model with no lifecycle protection")
    ap.add_argument("--json", action="store_true", help="print the raw record as JSON")
    args = ap.parse_args()

    # ---- SAFETY REFUSALS, before any work is done -------------------------------------------
    #
    # Both of these are refused rather than removed, so the reason is visible and the code stays
    # reviewable. Diagnostic use lives in the isolated scripts under benchmarks/, not in the
    # user-facing entry point.
    if args.embedder == "in-process":
        print("REFUSED: --embedder in-process is disabled. It loads a SECOND copy of the embedding "
              "model while the 8113 service already holds one; measured effect was free VRAM "
              "falling to 1942 MiB, below the 2048 MiB safety margin. Use the default "
              "(--embedder service), which reuses the running model and costs no extra VRAM.",
              file=sys.stderr)
        return 2
    if args.rerank:
        print("REFUSED: --rerank is disabled. The reranker is not yet integrated with the safe "
              "service lifecycle (reservation-protected load, verified unload, health-gated "
              "recovery), so enabling it here would load a model with no lifecycle protection. "
              "Reranking experiments belong in the isolated script under benchmarks/, which "
              "performs the required verified unloads.", file=sys.stderr)
        return 2

    qs: list[str] = []
    if args.question_file:
        # utf-8-sig: benchmarks/p56/frozen_questions.txt starts with a byte-order mark (written by
        # PowerShell), and reading it as plain utf-8 sent q01 to the embedder and the model as
        # "﻿What port ..." in both the P56 and the Phase A runs.
        qs += [ln.strip() for ln in Path(args.question_file).read_text(
            encoding="utf-8-sig").splitlines() if ln.strip() and not ln.startswith("#")]
    if args.question:
        qs.append(args.question)
    if not qs:
        ap.error("provide a question or --question-file")

    try:
        retr = Retriever(Path(args.index), embedder=args.embedder)
    except IndexUnusable as e:
        print(f"REFUSED: the index cannot be served: {e}", file=sys.stderr)
        return 3
    records = []
    try:
        for q in qs:
            rec = answer(q, rerank=args.rerank, retriever=retr, top_k=args.top_k,
                         embedder=args.embedder)
            records.append(rec)
            if args.json:
                print(json.dumps(rec, ensure_ascii=False))
            else:
                print(f"\nQ: {q}")
                if rec.get("error"):
                    print(f"  ERROR ({rec.get('error_kind')}): {rec['error']}")
                elif rec.get("refused"):
                    print(f"  REFUSED: {rec['answer']}")
                    print(f"  reason: {rec.get('reason') or 'the model declined'}")
                    for u in rec.get("stale_or_unverifiable_citations", []):
                        print(f"     {u['status']}: {u['source_path']} -- {u['detail']}")
                else:
                    print(f"  A: {rec.get('answer', '(no answer field)')}")
                    for c in rec.get("citations", []):
                        v = c.get("verification", {})
                        cut = (f" (shown from chunk {c['chunk_start_line']}-{c['chunk_end_line']})"
                               if c.get("truncated") else "")
                        print(f"     [{c['source']}] {c['source_path']}:"
                              f"{c['start_line']}-{c['end_line']}{cut}  "
                              f"({v.get('status', 'unverified')})")
                omitted = (rec.get("pack") or {}).get("omitted") or []
                if omitted:
                    print(f"  {len(omitted)} lower-ranked source(s) left out for the context "
                          f"budget")
                print(f"  latency {rec.get('e2e_s')}s  top_score {rec.get('max_score')}  "
                      f"free_vram {rec.get('free_vram_after_mb')} MiB  "
                      f"index {rec.get('index_version')}")
                # A truncated answer must never pass unnoticed (AD-22). Stated on its own line, in
                # the same place for every outcome, and NOT auto-retried - the caller decides.
                if rec.get("answer_truncated"):
                    n = (rec.get("usage") or {}).get("completion_tokens")
                    print(f"  *** ANSWER TRUNCATED at {n} tokens (finish_reason=length): the text "
                          f"above is incomplete ***")
    finally:
        retr.close()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
