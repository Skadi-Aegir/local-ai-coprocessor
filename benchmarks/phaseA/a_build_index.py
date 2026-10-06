"""
a_build_index.py - Phase A: build the clean QA index DIRECTLY FROM CURRENT FILE CONTENTS.

This replaces the P56 approach, which took vectors out of the historical 1173-chunk index. Here the
source files are read from disk, chunked, and embedded through the running embedding service. No
historical vector is reused.

WHY THE EMBEDDING SERVICE AND NOT AN IN-PROCESS MODEL
  A second in-process copy of the embedding model dropped free VRAM to 1942 MiB in P56 - below the
  2048 MiB margin. Query AND document embedding therefore both go through the already-resident
  service on 8113. The service embeds documents when the request omits `query`, which matches the
  convention the historical index was built with (`encode_documents`).

WHAT IS RECORDED (required by Phase A item 2)
  * the Git commit at build time, the repo-wide dirty flag, AND each corpus file's own status
    against that commit (clean / modified / untracked) - the repo-wide flag alone could not say
    whether the dirt was in the corpus (in the first Phase A build it was not)
  * per-source-file content hash, scheme `sha256-lf-v1`: CRLF is folded to LF before hashing, so a
    checkout's line endings do not change a file's identity (see services/retrieval/provenance.py)
  * per-chunk line numbers as they exist in the CURRENT file
  * a corpus manifest listing every included file and every exclusion reason
  * digests of what was BUILT (chunks_sha256, vectors_sha256), because index_version only names
    the inputs; whether two builds produced the same index is decided by these, or by --compare-to

ATOMIC-AS-POSSIBLE SWITCH + ROLLBACK
  The new index is built into staging, validated, and only then swapped in by `safe_switch`, which
  deletes nothing until the new index is in place and fails cleanly if the live index is in use.
  `--switch-only` swaps a previously built and validated staging index without rebuilding.

Usage:
  python a_build_index.py                       # build, validate, switch
  python a_build_index.py --no-switch           # build and validate only
  python a_build_index.py --switch-only         # switch an already validated staging build
  python a_build_index.py --switch-only --out vector/qa/current.prev
                                                # ROLL BACK: swaps current and current.prev
  python a_build_index.py --no-switch --out vector/qa/staging2 --compare-to vector/qa/current
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "retrieval"))
sys.path.insert(0, str(REPO / "cli"))

from provenance import (HASH_SCHEME, content_hash, file_git_status, git,  # noqa: E402
                        line_count, tree_blobs)

INDEX_ROOT = REPO / "vector/qa"
DEFAULT_DIR = INDEX_ROOT / "current"
PREV_DIR = INDEX_ROOT / "current.prev"

INCLUDE_DIRS = ["docs", "config", "scripts", "services", "cli"]
INCLUDE_SUFFIXES = {".md", ".yaml", ".yml", ".py", ".ps1", ".txt"}

# Exclusion is by explicit rule and every exclusion is recorded with its reason, so the corpus is
# reviewable rather than implied. Test scripts, fixtures, questions, scoring rules and answer keys
# must not enter the corpus.
# "test-" as well as "test_": the first Phase A build admitted scripts/test-gpu-scheduling.ps1, a
# self-described regression test with assertions, because only the underscore form was listed.
EXCLUDE_NAME_PREFIXES = ("test_", "test-", "conftest")
EXCLUDE_DIR_PARTS = ("fixtures", "__pycache__", "venv", ".git")
# The QA tool's own code is not project knowledge for the QA corpus. The first build excluded only
# cli/ask.py by name, so cli/ask_server.py (added in Phase B) would have entered the next build.
EXCLUDE_PATH_PREFIXES = {"cli/": "qa_entry_point_code"}
EXCLUDE_EXACT = {
    "services/retrieval/benchmark_data.py",
    "services/retrieval/retrieval_bench.py",
    "services/retrieval/rag_compare.py",
    # SAME ANSWER KEY AS rag_compare.py, different filename. Verified it embeds the 12 tuning
    # questions verbatim plus their `facts=[` / `bad=[` lists. A filename-based rule did not catch
    # it, so it is listed explicitly.
    "services/retrieval/rag_compare_rerank.py",
    "services/retrieval/bench_throughput.py",
    "services/retrieval/probe_embedding.py",
    "services/retrieval/token_savings.py",
    "services/retrieval/verify_pipeline.py",
    # Evaluation harnesses that carry the queries and their relevance judgements.
    "services/reranker/rerank_bench.py",
    "services/reranker/bench_reranker_params.py",
    "services/reranker/hard_negatives.py",
    "services/reranker/vram_coresidency.py",
    "scripts/test_lifecycle_gates.ps1",
    "scripts/llama_bench_suite.ps1",
    "scripts/phase2_sweep.ps1",
    "scripts/phase3_coresidency.ps1",
}
# An entry that names no file excludes nothing, silently. The list once carried
# services/retrieval/verify_hashes.py, which does not exist (the file is
# services/reranker/verify_hashes.py, an operational model-hash checker that is legitimately
# included). Dead entries now stop the build before anything is embedded.
EXCLUDE_SUBSTR = (
    # Filename-level tokens that mark benchmark/answer-key material. Deliberately NARROW: earlier
    # versions carried bare "eval", "bench", "score" and "label", and "eval" alone wiped out the
    # whole services/retrieval/ tree by matching "retrieval_...". A false exclusion is silent and
    # removes real material, so the list is kept to tokens that are unambiguous in this repo.
    "results", "benchmark", "bench_", "sweep", "questions", "answers",
    "gold", "grading", "scoring", "labels", "harness",
)

# Answer-key markers, tolerant of whitespace: the first check matched only the literal `facts=[`,
# so `facts = [` would have passed. Measured: these patterns match 0 lines in the 48 included
# files and 12 + 12 lines in the two excluded answer keys.
KEY_MARKERS = (
    ("facts_list", re.compile(r"\bfacts\s*=\s*\[")),
    ("bad_list", re.compile(r"\bbad\s*=\s*\[")),
    ("answer_key", re.compile(r"answer_key")),
    ("expected_fact", re.compile(r"expected_fact")),
)
# Every frozen evaluation set run against this corpus. None of their questions may appear in a
# chunk (compared case- and whitespace-insensitively). A listed file that is missing stops the build.
FROZEN_QUESTION_FILES = (
    "benchmarks/p56/frozen_questions.txt",
    "benchmarks/phaseA/heldout_v1/frozen_questions.txt",
)

EMBED_URL = "http://127.0.0.1:8113"
DIM = 1024
STRATEGY = "structured"
# Texts per embedding request. Was 16. Measured in isolation on the Phase A corpus
# (benchmarks/p1_11, results 2026-10-05-rs1 and -rs16b): the transient above the weights is
# 3752 MiB at 16 per request and 986 MiB at 4, and 4 is FASTER (12.2 s against 22.0 s: less
# padding). At 16, a build with local-fast resident cannot stay above the 2048 MiB margin.
# The request size does change the vectors slightly - but every batched size is equally close to
# the unpadded single-text vectors (mean cosine 0.99985 at 4, 0.99984 at 16), so 4 loses nothing.
BATCH = 4
MARGIN_MB = 2048
# Peak device memory the build itself adds inside the embedding service at BATCH=4: measured
# 986 MiB above the weights on the Phase A corpus, plus headroom for a longer chunk.
BUILD_TRANSIENT_ALLOWANCE_MB = 1280
# The embedding service retaining more than this after a build means it must be restarted before
# it serves the QA path (OPEN_ISSUES P1-11).
RETENTION_WARN_MB = 512


def exclusion_reason(rel: str) -> str | None:
    """
    Decide whether a repo-relative path may enter the corpus.

    Substring rules are applied to the BASENAME, not the whole path. Reason, measured: matching the
    whole path with the substring "eval" also matched "retr**ieval**", which silently excluded the
    entire `services/retrieval/` codebase - the single most important body of material for these
    questions. The build printed only an aggregate excluded count, so the loss stayed invisible
    until the walk was inspected per reason.
    """
    p = Path(rel)
    if rel in EXCLUDE_EXACT:
        return "explicit_exclusion_list"
    for prefix, why in EXCLUDE_PATH_PREFIXES.items():
        if rel.startswith(prefix):
            return why
    if p.name.startswith(EXCLUDE_NAME_PREFIXES):
        return "test_or_conftest"
    if any(part in EXCLUDE_DIR_PARTS for part in p.parts):
        return "excluded_directory"
    low = p.name.lower()
    for s in EXCLUDE_SUBSTR:
        if s in low:
            return f"name_contains_{s}"
    if p.suffix.lower() not in INCLUDE_SUFFIXES:
        return "suffix_not_included"
    return None


def dead_exclusions(repo: Path = REPO) -> list[str]:
    return sorted(p for p in EXCLUDE_EXACT if not (repo / p).exists())


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def load_frozen_questions(repo: Path = REPO) -> tuple[dict[str, list[str]], list[str]]:
    """Returns ({file: questions}, [missing files])."""
    sets, missing = {}, []
    for rel in FROZEN_QUESTION_FILES:
        f = repo / rel
        if not f.is_file():
            missing.append(rel)
            continue
        # utf-8-sig: the P56 file starts with a byte-order mark. Read as plain utf-8, its first
        # question carried "﻿" and the verbatim check could never match it.
        sets[rel] = [ln.strip() for ln in f.read_text(encoding="utf-8-sig").splitlines()
                     if ln.strip() and not ln.lstrip().startswith("#")]
    return sets, missing


def leak_findings(chunks: list[dict], question_sets: dict[str, list[str]]) -> list[dict]:
    """Answer-key markers and verbatim frozen questions found in chunk text."""
    found = []
    questions = [(name, q, _norm(q)) for name, qs in question_sets.items() for q in qs]
    for c in chunks:
        body = c["content"] or ""
        for name, pat in KEY_MARKERS:
            if pat.search(body):
                found.append({"source_path": c["source_path"], "chunk_id": c["chunk_id"],
                              "marker": name})
                break
        nb = _norm(body)
        for qset, q, nq in questions:
            if nq and nq in nb:
                found.append({"source_path": c["source_path"], "chunk_id": c["chunk_id"],
                              "marker": "frozen_question", "set": qset, "question": q})
    return found


def _http_json(url: str, payload: dict | None = None, timeout: int = 900) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Document embedding via the service. The `query` flag is deliberately OMITTED."""
    out: list[list[float]] = []
    for i in range(0, len(texts), BATCH):
        chunk = texts[i:i + BATCH]
        body = _http_json(EMBED_URL + "/v1/embeddings",
                          {"model": "local-embedding", "input": chunk})
        out.extend(d["embedding"] for d in body["data"])
    return out


def embedding_identity() -> dict | None:
    """What the service says it is: dimension, pooling, sequence length. None if unreachable."""
    try:
        body = _http_json(EMBED_URL + "/v1/models", timeout=15)
        return (body.get("data") or [{}])[0].get("meta")
    except Exception:  # noqa: BLE001
        return None


def embedding_memory() -> dict | None:
    """
    The embedding service's memory as it reports it. `memory` - present once the service carries
    the P1-11 fix - is the service's own PyTorch allocator, which retention is judged on.
    `vram_mb` is cudaMemGetInfo's total - free: on this WDDM machine it rises one-for-one with
    the service's allocations but is offset by a baseline of unclear origin, so it is used only
    when `memory` is absent.
    """
    try:
        st = _http_json(EMBED_URL + "/stats", timeout=15)
    except Exception:  # noqa: BLE001
        return None
    return {"vram_mb": st.get("vram_mb"), "process": st.get("memory")}


def nvidia_free_mb() -> int | None:
    """Device-wide free memory, the figure the 2048 MiB margin is defined on."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


class FreeVramWatch:
    """Samples device-wide free memory on a thread while the build embeds."""

    def __init__(self, interval: float = 0.2):
        import threading
        self.interval = interval
        self.vals: list[int] = []
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            v = nvidia_free_mb()
            if v is not None:
                self.vals.append(v)
            self.stop.wait(self.interval)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.t.join(timeout=20)
        return False

    def summary(self) -> dict:
        if not self.vals:
            return {"samples": 0}
        return {"samples": len(self.vals), "free_min_mb": min(self.vals),
                "margin_mb": MARGIN_MB, "margin_held": min(self.vals) >= MARGIN_MB}


def retention_verdict(before: dict | None, after: dict | None) -> dict:
    """Did the bulk embed leave memory behind in the embedding service? (OPEN_ISSUES P1-11)"""
    if not before or not after:
        return {"measured": False, "note": "embedding service /stats unavailable"}
    pb, pa = before.get("process") or {}, after.get("process") or {}
    if "torch_reserved_mb" in pb and "torch_reserved_mb" in pa:
        grown = round(pa["torch_reserved_mb"] - pb["torch_reserved_mb"], 1)
        basis = "process allocator (torch_reserved_mb)"
    else:
        grown = round((after.get("vram_mb") or 0) - (before.get("vram_mb") or 0), 1)
        basis = ("/stats vram_mb (cudaMemGetInfo): the service predates the P1-11 fix; the "
                 "difference tracks its own allocations but is not a footprint")
    restart = grown > RETENTION_WARN_MB
    return {"measured": True, "basis": basis, "retained_mb": grown,
            "restart_required": restart,
            "note": ("restart local-embedding (8113) before it serves the QA path - "
                     "OPEN_ISSUES P1-11" if restart else "no retention above the threshold")}


def collect_files() -> tuple[list[str], dict[str, str]]:
    """
    Walk the include dirs with PRUNING, applying the exclusion rule.

    os.walk + in-place `dirnames` pruning rather than rglob: an earlier rglob descended into
    `services/llm/venv` and similar trees and reported 51497 excluded entries, which is noise that
    hides the exclusions that actually matter. Pruning also makes the walk fast.
    """
    included: list[str] = []
    excluded: dict[str, str] = {}
    for d in INCLUDE_DIRS:
        base = REPO / d
        if not base.is_dir():
            continue
        for root, dirnames, filenames in os.walk(base):
            dirnames[:] = sorted(n for n in dirnames if n not in EXCLUDE_DIR_PARTS)
            for name in sorted(filenames):
                f = Path(root) / name
                rel = f.relative_to(REPO).as_posix()
                why = exclusion_reason(rel)
                if why:
                    excluded[rel] = why
                else:
                    included.append(rel)
    return included, excluded


def chunks_digest(chunks: list[dict]) -> str:
    """Digest of the chunk table in build order: ids, paths, line ranges and content."""
    h = hashlib.sha256()
    for c in chunks:
        h.update(json.dumps([c["chunk_id"], c["source_path"], c["start_line"], c["end_line"],
                             hashlib.sha256(c["content"].encode("utf-8")).hexdigest()]).encode())
        h.update(b"\n")
    return h.hexdigest()


def vectors_digest(vecs) -> str:
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(vecs, dtype=np.float32).tobytes()).hexdigest()


def compare_builds(a: Path, b: Path) -> dict:
    """
    Is build `a` the same index as build `b`? Decided on what was built, not on index_version.

    index_version hashes the build INPUTS, so two builds of the same files always share it -
    "identical index_version" was therefore no evidence that a rebuild reproduces the index. This
    compares the chunk tables exactly and the vectors numerically (GPU embedding need not be
    bit-identical across runs, so the largest per-component difference is reported, not assumed).
    """
    import numpy as np
    ma = json.loads((a / "manifest.json").read_text(encoding="utf-8"))
    mb = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
    out = {"a": str(a), "b": str(b),
           "index_version_equal": ma.get("index_version") == mb.get("index_version"),
           "chunks_equal": (ma.get("chunks_sha256") is not None
                            and ma.get("chunks_sha256") == mb.get("chunks_sha256")),
           "vectors_bit_identical": (ma.get("vectors_sha256") is not None
                                     and ma.get("vectors_sha256") == mb.get("vectors_sha256"))}
    va, vb = a / "vectors.npy", b / "vectors.npy"
    if va.is_file() and vb.is_file():
        xa, xb = np.load(va), np.load(vb)
        out["shapes"] = [list(xa.shape), list(xb.shape)]
        if xa.shape == xb.shape and len(xa):
            out["max_abs_diff"] = float(np.max(np.abs(xa - xb)))
            out["min_row_cosine"] = float(np.min(np.sum(xa * xb, axis=1)))
    out["same_index"] = bool(out["chunks_equal"] and (
        out["vectors_bit_identical"] or out.get("min_row_cosine", 0.0) >= 0.9999))
    return out


def safe_switch(staging: Path, current: Path, prev: Path) -> tuple[bool, str]:
    """
    Swap a validated staging build into `current`, keeping the old `current` as `prev`.

    The first version deleted `prev` FIRST and then renamed twice. Measured on Windows: while a
    reader (ask_server, a CLI run) holds current/metadata.sqlite open, the first rename fails with
    WinError 5 - after the rollback copy had already been deleted, leaving staging dangling. This
    order deletes nothing until the new index is in place:
      1. current -> current.old-<stamp>   if this fails the index is in use; nothing has changed
      2. staging -> current               on failure, step 1 is undone
      3. prev    -> retired-<stamp>       the old rollback is retired only once 2 has succeeded
      4. current.old-<stamp> -> prev
      5. retired-<stamp> deleted          best effort
    Between steps 1 and 2 `current` is absent for a moment; a reader starting then is refused by
    open_index rather than silently creating an empty index.
    """
    stamp = time.strftime("%Y%m%d-%H%M%S")
    root = current.parent
    old = root / f"{current.name}.old-{stamp}"
    retired = root / f"retired-{stamp}"
    had_current = current.exists()
    if had_current:
        try:
            os.replace(current, old)
        except OSError as e:
            return False, (f"NOT switched: {current} is in use ({type(e).__name__}: {e}). Stop "
                           f"the QA server or CLI holding it, then re-run with --switch-only. "
                           f"The validated build stays at {staging}; nothing was deleted.")
    try:
        os.replace(staging, current)
    except OSError as e:
        if had_current:
            os.replace(old, current)
        return False, (f"NOT switched: could not move {staging} into place "
                       f"({type(e).__name__}: {e}); the previous index was restored.")
    if not had_current:
        return True, f"switched: {current} is the default; there was no previous index"
    if prev.exists():
        try:
            os.replace(prev, retired)
        except OSError as e:
            return True, (f"switched: {current} is the default. The previous index is kept at "
                          f"{old}, because {prev} could not be retired ({type(e).__name__}).")
    os.replace(old, prev)
    if retired.exists():
        shutil.rmtree(retired, ignore_errors=True)
    return True, f"switched: {current} is the default; previous kept at {prev}"


def switch_only(staging: Path) -> int:
    """Swap an already built staging index in, after re-checking it as the QA path would."""
    from ask import IndexUnusable, open_index
    try:
        store, prov = open_index(staging)
        store.close()
    except IndexUnusable as e:
        print(f"REFUSING to switch: {e}")
        return 4
    ok, msg = safe_switch(staging, DEFAULT_DIR, PREV_DIR)
    print(msg)
    return 0 if ok else 6


def main() -> int:
    ap = argparse.ArgumentParser()
    # Build into STAGING by default, never into DEFAULT_DIR. An earlier version defaulted `--out` to
    # DEFAULT_DIR, which made the later "atomic switch" move DEFAULT_DIR onto itself - i.e. delete
    # its own output. Staging keeps the live index serving until validation has passed.
    ap.add_argument("--out", default=str(INDEX_ROOT / "staging"))
    ap.add_argument("--report", default=str(HERE / "results/a_index_build.json"))
    ap.add_argument("--no-switch", action="store_true")
    ap.add_argument("--switch-only", action="store_true",
                    help="switch the already validated build at --out; no rebuild")
    ap.add_argument("--compare-to", help="another build directory to compare this build with")
    args = ap.parse_args()

    if args.switch_only:
        return switch_only(Path(args.out))
    # The build directory is deleted and recreated below, so it must never be a live index.
    if Path(args.out).resolve() in (DEFAULT_DIR.resolve(), PREV_DIR.resolve()):
        print(f"REFUSING: --out {args.out} is the live or rollback index; build into staging")
        return 2

    # ---- configuration preconditions, before anything is embedded
    dead = dead_exclusions()
    qsets, missing_q = load_frozen_questions()
    if dead or missing_q:
        for p in dead:
            print(f"CONFIG ERROR: exclusion entry names no file: {p}")
        for p in missing_q:
            print(f"CONFIG ERROR: frozen question file missing: {p}")
        return 5

    from chunker import chunk_file
    from index_store import IndexStore
    import numpy as np

    t_all = time.time()
    # Git state is read ONCE, so the chunks and the manifest cannot describe different commits.
    commit = git(REPO, "rev-parse", "HEAD") or "unavailable"
    branch = git(REPO, "rev-parse", "--abbrev-ref", "HEAD") or "unavailable"
    porcelain = git(REPO, "status", "--porcelain")
    repo_dirty = porcelain is None or bool(porcelain)
    blobs = tree_blobs(REPO, commit) if commit != "unavailable" else None
    included, excluded = collect_files()

    chunks: list[dict] = []
    file_bytes: dict[str, bytes] = {}
    file_hashes: dict[str, str] = {}
    file_lines: dict[str, int] = {}
    file_status: dict[str, str] = {}
    skipped: dict[str, str] = {}

    # Each file is read ONCE: the bytes that are hashed are the bytes that are chunked, so a file
    # edited during the build cannot end up with a hash that does not describe its chunks.
    for rel in included:
        try:
            raw = (REPO / rel).read_bytes()
        except Exception as e:  # noqa: BLE001
            skipped[rel] = f"read_error: {type(e).__name__}"
            continue
        file_bytes[rel] = raw
        file_hashes[rel] = content_hash(raw)
        file_lines[rel] = line_count(raw)
        file_status[rel] = file_git_status(rel, raw, blobs)
    dirty_files = sorted(f for f, s in file_status.items() if s != "clean")
    commit_tag = commit if not dirty_files else f"{commit}-dirty"
    print(f"included {len(included)} files; excluded {len(excluded)}; commit={commit[:12]} "
          f"repo_dirty={repo_dirty} corpus_files_not_at_commit={len(dirty_files)}")

    for rel, raw in file_bytes.items():
        text = raw.replace(b"\r\n", b"\n").decode("utf-8", errors="replace")
        try:
            cs = chunk_file(text, rel, "local-ai", strategy=STRATEGY, commit_hash=commit_tag)
        except Exception as e:  # noqa: BLE001
            skipped[rel] = f"chunk_error: {type(e).__name__}: {e}"
            continue
        for c in cs:
            chunks.append({"chunk_id": c.chunk_id, "project": "local-ai", "source_path": rel,
                           "content": c.content, "start_line": c.start_line,
                           "end_line": c.end_line, "symbol": c.symbol,
                           "language": c.language, "content_type": c.content_type,
                           "strategy": c.strategy, "commit_hash": commit_tag})
    print(f"chunked {len(chunks)} chunks from {len(file_hashes)} files; {len(skipped)} skipped")
    if not chunks:
        print("REFUSING to build an empty index")
        return 3

    identity = embedding_identity()
    mem_before = embedding_memory()
    # PREFLIGHT: the build adds a transient inside the embedding service. Phase A's builds ran with
    # other models resident and took free VRAM to 640-743 MiB; refuse instead.
    free_before = nvidia_free_mb()
    if free_before is None or free_before - BUILD_TRANSIENT_ALLOWANCE_MB < MARGIN_MB:
        print(f"REFUSING to embed: free VRAM is {free_before} MiB, and the build's transient "
              f"(~{BUILD_TRANSIENT_ALLOWANCE_MB} MiB at {BATCH} texts per request) must leave the "
              f"{MARGIN_MB} MiB margin intact. Stop a resident service the build does not need - "
              f"the reranker (8114) or local-fast (8111) - and re-run.")
        return 7
    t0 = time.time()
    with FreeVramWatch() as watch:
        vecs = np.asarray(embed_documents([c["content"] for c in chunks]), dtype="float32")
    embed_s = round(time.time() - t0, 1)
    build_vram = {"free_before_mb": free_before, "batch": BATCH,
                  "transient_allowance_mb": BUILD_TRANSIENT_ALLOWANCE_MB, **watch.summary()}
    mem_after = embedding_memory()
    retention = retention_verdict(mem_before, mem_after)
    print(f"embedded {len(vecs)} chunks in {embed_s}s; free VRAM min during the build "
          f"{build_vram.get('free_min_mb')} MiB (margin held: {build_vram.get('margin_held')}); "
          f"embedding service retention: {retention.get('retained_mb')} MiB "
          f"({retention.get('basis', 'not measured')})")

    out_dir = Path(args.out)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    store = IndexStore(out_dir, DIM)
    store.add(vecs, chunks, file_hashes=file_hashes)
    store.save()
    stats = store.stats()
    store.close()
    print(f"built {out_dir}: {stats}")

    # ---- VALIDATION before any switch
    v = {"stats": stats}
    v["vectors_eq_metadata"] = stats["vectors_in_index"] == stats["chunks_in_metadata"]
    v["count_matches_chunks"] = stats["chunks_in_metadata"] == len(chunks)
    v["consistent"] = bool(stats.get("consistent"))
    # every chunk's cited line range must exist in the CURRENT file
    bad_lines = []
    for c in chunks:
        n = file_lines.get(c["source_path"], 0)
        if not (1 <= c["start_line"] <= c["end_line"] <= n):
            bad_lines.append({"source_path": c["source_path"],
                              "start_line": c["start_line"], "end_line": c["end_line"],
                              "file_lines": n})
    v["line_ranges_valid"] = not bad_lines
    v["bad_line_ranges"] = bad_lines[:10]
    v["n_bad_line_ranges"] = len(bad_lines)
    # the answer-key material must be absent. Checked two ways: no included path may be excludable,
    # and no chunk may carry a scoring-rule marker or a question from a frozen evaluation set.
    present = [c["source_path"] for c in chunks
               if exclusion_reason(c["source_path"]) is not None]
    v["excluded_material_present"] = sorted(set(present))
    v["no_excluded_material"] = not present
    leak = leak_findings(chunks, qsets)
    v["question_sets_checked"] = {k: len(qs) for k, qs in qsets.items()}
    v["answer_key_leaks"] = leak[:10]
    v["n_answer_key_leaks"] = len(leak)
    v["no_answer_key_material"] = not leak

    ok = (v["vectors_eq_metadata"] and v["count_matches_chunks"] and v["consistent"]
          and v["line_ranges_valid"] and v["no_excluded_material"]
          and v["no_answer_key_material"])
    v["validation_passed"] = ok

    code_ids = {name: content_hash(p.read_bytes()) for name, p in (
        ("chunker", REPO / "services/retrieval/chunker.py"),
        ("provenance", REPO / "services/retrieval/provenance.py"),
        ("builder", Path(__file__).resolve()))}
    rules = {
        "include_dirs": INCLUDE_DIRS, "include_suffixes": sorted(INCLUDE_SUFFIXES),
        "exclude_exact": sorted(EXCLUDE_EXACT),
        "exclude_path_prefixes": EXCLUDE_PATH_PREFIXES,
        "exclude_name_prefixes": list(EXCLUDE_NAME_PREFIXES),
        "exclude_dir_parts": list(EXCLUDE_DIR_PARTS),
        "exclude_substrings": list(EXCLUDE_SUBSTR),
    }
    # index_version names the build INPUTS: commit, corpus content, configuration and code. It is
    # NOT evidence that two builds produced the same index - chunks_sha256 / vectors_sha256 are.
    version_inputs = {"scheme": 2, "commit": commit, "dirty_files": dirty_files,
                      "hash_scheme": HASH_SCHEME, "strategy": STRATEGY, "dimension": DIM,
                      "rules": rules, "code": code_ids,
                      "embedding": {"service": EMBED_URL, "identity": identity}}
    index_version = hashlib.sha256(json.dumps({**version_inputs, "files": file_hashes},
                                              sort_keys=True).encode()).hexdigest()[:16]

    manifest = {
        "index_version": index_version,
        "index_version_scheme": 2,
        "built_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": commit,
        "git_branch": branch,
        "working_tree_dirty": repo_dirty,
        "corpus_files_not_at_commit": dirty_files,
        "dirty_note": (
            "corpus files differ from the commit: " + ", ".join(dirty_files) +
            "; for those files the hash, not the commit, identifies the indexed content"
            if dirty_files else
            "every corpus file is identical to its blob at git_commit (line endings aside)"
            + ("; the working tree had changes OUTSIDE the corpus" if repo_dirty else "")),
        "strategy": STRATEGY, "dimension": DIM,
        "embedding": {"service": EMBED_URL, "mode": "documents (no query flag)",
                      "identity": identity},
        "code": code_ids,
        "n_files": len(file_hashes), "n_chunks": len(chunks),
        "chunks_sha256": chunks_digest(chunks),
        "vectors_sha256": vectors_digest(vecs),
        "hash_scheme": HASH_SCHEME,
        "file_hashes": file_hashes, "file_lines": file_lines,
        "file_git_status": file_status,
        "included_files": sorted(file_hashes),
        "excluded_files": excluded, "skipped_files": skipped,
        "exclusion_rules": rules,
        "validation": v,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    report = {"build_seconds": round(time.time() - t_all, 1), "embed_seconds": embed_s,
              "out_dir": str(out_dir),
              "build_vram": build_vram,
              "embedding_service_memory": {"before": mem_before, "after": mem_after,
                                           "retention": retention},
              "manifest": manifest}
    if args.compare_to:
        try:
            report["comparison"] = compare_builds(out_dir, Path(args.compare_to))
        except Exception as e:  # noqa: BLE001  a bad comparison target must not lose the build
            report["comparison"] = {"error": f"{type(e).__name__}: {e}"}
        print(f"comparison with {args.compare_to}: {json.dumps(report['comparison'])}")
    rep = Path(args.report)
    rep.parent.mkdir(parents=True, exist_ok=True)
    if rep.exists():
        rep = rep.with_name(f"{rep.stem}-{time.strftime('%Y%m%d-%H%M%S')}{rep.suffix}")
    rep.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nvalidation: {json.dumps({k: v[k] for k in v if k != 'bad_line_ranges'})}")
    print(f"index_version: {index_version}  corpus_files_not_at_commit={len(dirty_files)}")
    print(f"wrote {rep}")
    if retention.get("restart_required"):
        print(f"WARNING: {retention['note']}")
    if build_vram.get("margin_held") is False:
        print(f"WARNING: free VRAM fell to {build_vram['free_min_mb']} MiB during the build, below "
              f"the {MARGIN_MB} MiB margin")

    if not ok:
        print("VALIDATION FAILED - not switching. The previous index is untouched.")
        return 4

    if args.no_switch:
        print("--no-switch: built and validated, switch skipped")
        return 0

    switched, msg = safe_switch(out_dir, DEFAULT_DIR, PREV_DIR)
    print(msg)
    return 0 if switched else 6


if __name__ == "__main__":
    raise SystemExit(main())
