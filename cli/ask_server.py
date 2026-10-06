"""
ask_server.py - minimal LOCAL interface for the QA path (Phase B).

Bound to loopback only. Deliberately not a general Gateway: two endpoints, one caller, no auth
layer, no service discovery, no model routing.

    GET  /health   liveness, the index ACTUALLY loaded (version, chunks), and the resource policy
    POST /ask      {"question": "..."} -> answer or refusal, with citations

It REUSES the same core as the CLI: `cli/ask.py` is imported, not reimplemented, so retrieval,
prompting, the evidence floor, the refusal rules and citation verification are identical on both
paths. The index path is server-side configuration; a client cannot pass one. The server refuses
to start on an index whose manifest does not license serving it (ask.open_index).

RESOURCE BOUNDARY. Three limits, stated as they actually behave:
  * MAX_QUESTION_CHARS / MAX_BODY_BYTES - input size; the body size is checked before the body is
    read.
  * MAX_CONCURRENCY = 2 admitted requests, of which ONE executes at a time. The whole pipeline runs
    under one lock, so the second admitted request waits for the first; the third gets a structured
    429. Measured in Phase B: two admitted requests took 1.83 s and 3.54 s, i.e. serially. Running
    them in parallel would gain little here anyway: local-fast serves a single slot
    (config/local-fast.yaml parallel: 1), so generation is serialised there regardless.
  * DURATION is bounded by the CLI core's upstream timeouts (embedding 30 s, tokenize 15 s,
    generation 90 s). A client that gives up does NOT free its slot: the server keeps working until
    the pipeline returns or one of those timeouts fires. (The Phase B report said the slot was held
    "until the client's own timeout fires"; that was wrong, and with the earlier 600 s upstream
    timeouts one hung call could block every request for ten minutes.)
  The scheduler's 2048 MiB margin still applies to the embedding and generative services, which are
  external; this server never loads a model of its own.

ERRORS are always a JSON object {"error": {"kind", "message"}} with a real HTTP status:
  400 bad request (malformed JSON, not an object, bad question, index override)
  404 unknown path        413 body or question too large      429 too many requests
  500 internal            502 upstream answered with an error
  503 upstream unavailable                                    504 upstream timed out
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import ask as ASK  # noqa: E402  the CLI core, reused rather than reimplemented

MAX_QUESTION_CHARS = 1000
MAX_CONCURRENCY = 2
MAX_EXECUTING = 1
MAX_BODY_BYTES = 64 * 1024
# Socket timeout for reading a request: a client that announces a body and never sends it must not
# hold a thread forever.
REQUEST_READ_TIMEOUT_S = 30

LOOPBACK = ("127.0.0.1", "localhost", "::1")

# error_kind from ask.answer() -> (HTTP status, error kind returned to the caller)
_ERROR_STATUS = {
    "upstream_unavailable": (503, "upstream_unavailable"),
    "upstream_timeout": (504, "upstream_timeout"),
    "upstream_http_error": (502, "upstream_error"),
    "prompt_too_long": (413, "question_too_long"),
}

_SEM = threading.BoundedSemaphore(MAX_CONCURRENCY)
_INFLIGHT = {"current": 0, "peak": 0}
_INFLIGHT_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "localQA/1.1"
    timeout = REQUEST_READ_TIMEOUT_S

    # ---------------------------------------------------------------- helpers
    def _send(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, kind: str, message: str) -> None:
        self._send({"error": {"kind": kind, "message": message}}, code)

    def log_message(self, fmt, *args):  # quieter than the default
        if self.server.verbose:            # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    # ---------------------------------------------------------------- routes
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            # Reports the index this process LOADED - not the default path - so a server started
            # with --index cannot advertise a different corpus from the one it serves.
            retr = self.server.retriever        # type: ignore[attr-defined]
            prov = retr.provenance
            st = retr.stats_at_load
            self._send({
                "status": "ok",
                "index_version": prov.version,
                "index_dir": str(retr.index_dir),
                "index_working_tree_dirty": prov.dirty,
                "index_hash_scheme": prov.scheme,
                "index_chunks": {"manifest": prov.n_chunks,
                                 "vectors": st.get("vectors_in_index"),
                                 "metadata": st.get("chunks_in_metadata")},
                "embedding_service": ASK.EMBED_URL,
                "generation_service": ASK.FAST_URL,
                "retrieval": "dense",
                "reranker": "disabled",
                "prompt_version": ASK.PROMPT_VERSION,
                "limits": {"max_question_chars": MAX_QUESTION_CHARS,
                           "max_concurrency": MAX_CONCURRENCY,
                           "max_executing": MAX_EXECUTING,
                           "max_body_bytes": MAX_BODY_BYTES,
                           # The answer cap is a limit a caller can observe, so it is published
                           # (AD-22): without it a client cannot tell whether a short answer was
                           # short because the model finished or because the cap cut it.
                           "max_answer_tokens": ASK.MAX_ANSWER_TOKENS,
                           "upstream_timeouts_s": {"embed": ASK.EMBED_TIMEOUT_S,
                                                   "tokenize": ASK.TOKENIZE_TIMEOUT_S,
                                                   "generate": ASK.GENERATE_TIMEOUT_S}},
                "inflight": dict(_INFLIGHT),
                "note": "liveness of this process only; the upstream services are not probed",
            })
            return
        self._error(404, "not_found", f"unknown path {self.path}")

    def do_POST(self) -> None:  # noqa: N802
        try:
            q = self._read_question()
        except Exception as e:  # noqa: BLE001  never drop a connection without a response
            traceback.print_exc()
            self._error(500, "internal_error", f"{type(e).__name__}: {e}")
            return
        if q is None:
            return          # an error response has already been sent

        # ---- concurrency bound
        if not _SEM.acquire(blocking=False):
            self._error(429, "too_many_requests",
                        f"more than {MAX_CONCURRENCY} requests in flight; retry shortly")
            return
        with _INFLIGHT_LOCK:
            _INFLIGHT["current"] += 1
            _INFLIGHT["peak"] = max(_INFLIGHT["peak"], _INFLIGHT["current"])
        t0 = time.time()
        try:
            # ONE pipeline at a time (see the module docstring).
            with self.server.pipeline_lock:        # type: ignore[attr-defined]
                rec = ASK.answer(q, retriever=self.server.retriever,  # type: ignore[attr-defined]
                                 embedder="service", measure_vram=False)
            if rec.get("error"):
                code, kind = _ERROR_STATUS.get(rec.get("error_kind"), (500, "pipeline_error"))
                self._error(code, kind, rec["error"])
                return
            # ALWAYS the same shape, whether the outcome is an answer or a refusal, so a caller
            # never has to guess which keys exist.
            pack = rec.get("pack") or {}
            self._send({
                "question": q,
                "outcome": "refused" if rec.get("refused") else "answered",
                "answer": rec.get("answer"),
                "reason": rec.get("reason"),
                "citations": [{"source": c["source"], "source_path": c["source_path"],
                               "start_line": c["start_line"], "end_line": c["end_line"],
                               "truncated": bool(c.get("truncated")),
                               "verification": (c.get("verification") or {}).get("status")}
                              for c in (rec.get("citations") or [])],
                "index_version": rec.get("index_version"),
                "index_working_tree_dirty": rec.get("index_working_tree_dirty"),
                # ALWAYS present, answer or refusal (AD-22). A caller must be able to tell that an
                # answer was cut off by the length cap without inspecting the text itself.
                "finish_reason": rec.get("finish_reason"),
                "answer_truncated": bool(rec.get("answer_truncated")),
                "elapsed_s": round(time.time() - t0, 3),
                "pipeline": {"retrieve_s": rec.get("retrieve_s"),
                             "generate_s": rec.get("generate_s"),
                             "n_candidates": rec.get("n_candidates"),
                             "n_above_floor": rec.get("n_above_floor"),
                             "max_score": rec.get("max_score"),
                             "sources_supplied": len(pack.get("supplied") or []),
                             "sources_omitted_for_budget": len(pack.get("omitted") or []),
                             "prompt_version": rec.get("prompt_version"),
                             "retrieval": "dense", "reranker": "disabled"},
                "server_elapsed_s": round(time.time() - t0, 3),
            })
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._error(500, "internal_error", f"{type(e).__name__}: {e}")
        finally:
            with _INFLIGHT_LOCK:
                _INFLIGHT["current"] -= 1
            _SEM.release()

    def _read_question(self) -> str | None:
        """Parse and validate the request. Returns the question, or None after sending an error."""
        if self.path != "/ask":
            self._error(404, "not_found", f"unknown path {self.path}")
            return None

        # ---- body size checked BEFORE reading it into memory
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._error(400, "bad_request", "Content-Length is not a number")
            return None
        if length <= 0:
            self._error(400, "bad_request", "empty body")
            return None
        if length > MAX_BODY_BYTES:
            # The unread body is left on the socket, so the connection must not be reused.
            self.close_connection = True
            self._error(413, "body_too_large", f"body exceeds {MAX_BODY_BYTES} bytes")
            return None

        try:
            raw = self.rfile.read(length)
        except (socket.timeout, TimeoutError):
            self.close_connection = True
            self._error(408, "request_timeout",
                        f"the request body did not arrive within {REQUEST_READ_TIMEOUT_S} s")
            return None
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as e:  # noqa: BLE001
            self._error(400, "bad_json", f"{type(e).__name__}: {e}")
            return None
        # Valid JSON that is not an object ([1,2], 123, null, "text") used to raise past this
        # handler, and the connection closed with NO response at all.
        if not isinstance(payload, dict):
            self._error(400, "bad_request",
                        f"the body must be a JSON object, got {type(payload).__name__}")
            return None

        # ---- the client may NOT choose an index. Rejected explicitly rather than ignored, so a
        # caller cannot believe it selected a corpus when it did not.
        for forbidden in ("index", "index_dir", "index_path", "corpus"):
            if forbidden in payload:
                self._error(400, "index_override_rejected",
                            f"'{forbidden}' is not accepted: the index is server-side "
                            f"configuration and cannot be chosen by a client")
                return None

        q = payload.get("question")
        if not isinstance(q, str) or not q.strip():
            self._error(400, "bad_question", "question must be a non-empty string")
            return None
        q = q.strip()
        if len(q) > MAX_QUESTION_CHARS:
            self._error(413, "question_too_long",
                        f"question is {len(q)} chars, limit is {MAX_QUESTION_CHARS}")
            return None
        return q


class LoopbackServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, host: str, port: int, handler):
        # ThreadingHTTPServer is AF_INET by default, so "::1" passed the loopback check and then
        # failed to bind. Choose the family from the address.
        self.address_family = socket.AF_INET6 if ":" in host else socket.AF_INET
        super().__init__((host, port), handler)


def make_server(retriever, host: str = "127.0.0.1", port: int = 8115,
                verbose: bool = False) -> LoopbackServer:
    srv = LoopbackServer(host, port, Handler)
    srv.retriever = retriever               # type: ignore[attr-defined]
    srv.pipeline_lock = threading.Lock()    # type: ignore[attr-defined]
    srv.verbose = verbose                   # type: ignore[attr-defined]
    return srv


def main() -> int:
    ap = argparse.ArgumentParser(description="minimal local QA interface (loopback only)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="defaults to loopback and there is deliberately no option to expose it")
    ap.add_argument("--port", type=int, default=8115)
    ap.add_argument("--index", default=str(ASK.CLEAN_INDEX))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.host not in LOOPBACK:
        print(f"REFUSING to bind {args.host}: this interface is loopback-only. Public exposure is "
              f"out of scope.", file=sys.stderr)
        return 2

    # The retriever is created ONCE and shared. Creating one per request would load the index and
    # re-open the embedding client repeatedly, and a per-request in-process model is forbidden.
    try:
        retr = ASK.Retriever(Path(args.index), embedder="service")
    except ASK.IndexUnusable as e:
        print(f"REFUSING to start: the index cannot be served: {e}", file=sys.stderr)
        return 3
    srv = make_server(retr, args.host, args.port, args.verbose)
    prov = retr.provenance
    print(f"local QA interface on http://{args.host}:{args.port}")
    print(f"  index        : {retr.index_dir}  version {prov.version} "
          f"(chunks={prov.n_chunks}, dirty={prov.dirty}, hashes={prov.scheme})")
    print(f"  retrieval    : dense     reranker: disabled     prompt {ASK.PROMPT_VERSION}")
    print(f"  limits       : question<={MAX_QUESTION_CHARS} chars, admitted<={MAX_CONCURRENCY} "
          f"(executing {MAX_EXECUTING}), body<={MAX_BODY_BYTES} bytes")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        srv.server_close()
        retr.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
