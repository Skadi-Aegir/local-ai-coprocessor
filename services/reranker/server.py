"""
server.py - local-reranker HTTP service on 127.0.0.1:8114.

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/rerank      score a query against a list of documents
  GET  /stats
  POST /v1/cache/clear

Design boundary, stated by the specification: the reranker does NOT manage FAISS and
knows nothing about chunk metadata. It receives plain strings and returns
{index, score} pairs referring to positions in the request's own `documents` array. The
retrieval service maps those indices back to paths and line ranges. This keeps the
reranker a pure scoring function, which is what makes it testable in isolation.

Loopback only. Requests are serialised behind the model's lock because a single GPU
forward pass is not meaningfully parallel here, and Phase 4 showed concurrent GPU work on
this card loses throughput rather than gaining it.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from reranker import (CODE_INSTRUCTION, DEFAULT_INSTRUCTION, GENERAL_INSTRUCTION,  # noqa: E402
                      Reranker, RerankerConfig)

MODEL_ALIAS = "local-reranker"
START_TS = time.time()

# Named instructions so a caller does not have to send the full prompt text.
INSTRUCTIONS = {
    "general": GENERAL_INSTRUCTION,
    "code": CODE_INSTRUCTION,
    "default": DEFAULT_INSTRUCTION,
    "none": None,
}


class Service:
    def __init__(self, cfg: RerankerConfig):
        self.rr = Reranker(cfg)
        self.lock = threading.Lock()
        self.requests = 0
        self.pairs = 0
        self.errors = 0
        self.total_s = 0.0

    def stats(self) -> dict:
        info = self.rr.info()
        info.update({
            "alias": MODEL_ALIAS,
            "requests": self.requests,
            "pairs_scored": self.pairs,
            "errors": self.errors,
            "total_seconds": round(self.total_s, 3),
            "uptime_s": round(time.time() - START_TS, 1),
        })
        return info


def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "local-reranker/0.1"

        def log_message(self, *a):
            pass

        # ------------------------------------------------------------ helpers
        def _json(self, obj, status=200):
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status, message, etype="invalid_request_error"):
            self._json({"error": {"message": message, "type": etype}}, status)

        def _read_json(self):
            n = int(self.headers.get("Content-Length") or 0)
            if n == 0:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8"))

        def _resolve_instruction(self, req) -> tuple[str | None, str | None]:
            """
            Returns (instruction_text, label). Priority: explicit `instruction` text, then
            a named `instruction_type`, then the configured default.
            """
            if req.get("instruction") is not None:
                return req["instruction"], "explicit"
            itype = req.get("instruction_type")
            if itype is not None:
                key = str(itype).lower()
                if key not in INSTRUCTIONS:
                    raise ValueError(
                        f"unknown instruction_type '{itype}'; "
                        f"valid: {sorted(INSTRUCTIONS)}")
                return INSTRUCTIONS[key], key
            return ..., "default"

        # ------------------------------------------------------------ GET
        def do_GET(self):
            if self.path == "/health":
                return self._json({
                    "status": "ok", "model": MODEL_ALIAS,
                    "device": svc.rr.device, "dtype": svc.rr.cfg.dtype,
                    "max_length": svc.rr.cfg.max_length,
                    "uptime_s": round(time.time() - START_TS, 1),
                })
            if self.path == "/v1/models":
                return self._json({
                    "object": "list",
                    "data": [{
                        "id": MODEL_ALIAS, "object": "model", "owned_by": "local",
                        "created": int(START_TS),
                        "meta": {
                            "max_length": svc.rr.cfg.max_length,
                            "body_budget": svc.rr.body_budget,
                            "yes_token_id": svc.rr.yes_id,
                            "no_token_id": svc.rr.no_id,
                            "scoring": "softmax([logit(no), logit(yes)])[1]",
                            "instructions": list(INSTRUCTIONS),
                        },
                    }],
                })
            if self.path == "/stats":
                return self._json(svc.stats())
            return self._error(404, f"unknown path {self.path}")

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Allow", "GET, POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

        # ------------------------------------------------------------ POST
        def do_POST(self):
            try:
                if self.path == "/v1/rerank":
                    return self._rerank()
                if self.path == "/v1/cache/clear":
                    svc.rr.cache.clear()
                    return self._json({"cleared": True})
                return self._error(404, f"unknown path {self.path}")
            except json.JSONDecodeError as e:
                return self._error(400, f"invalid JSON body: {e}")
            except ValueError as e:
                return self._error(400, str(e))
            except Exception as e:  # noqa: BLE001
                svc.errors += 1
                traceback.print_exc()
                return self._error(500, f"{type(e).__name__}: {e}", "server_error")

        def _rerank(self):
            req = self._read_json()
            model = req.get("model", MODEL_ALIAS)
            if model not in (MODEL_ALIAS, None, ""):
                return self._error(404, f"model '{model}' not found; use '{MODEL_ALIAS}'",
                                   "model_not_found")

            query = req.get("query")
            if not isinstance(query, str) or not query.strip():
                return self._error(400, "query must be a non-empty string")
            docs = req.get("documents")
            if not isinstance(docs, list) or not all(isinstance(d, str) for d in docs):
                return self._error(400, "documents must be a list of strings")
            if not docs:
                return self._json({"model": MODEL_ALIAS, "results": [], "count": 0,
                                   "seconds": 0.0, "reranker_used": True})

            try:
                instr, label = self._resolve_instruction(req)
            except ValueError as e:
                return self._error(400, str(e))

            top_n = req.get("top_n")
            if top_n is not None:
                try:
                    top_n = int(top_n)
                except (TypeError, ValueError):
                    return self._error(400, "top_n must be an integer")
                if not (1 <= top_n <= len(docs)):
                    return self._error(400, f"top_n must be 1..{len(docs)}")

            use_cache = bool(req.get("use_cache", True))

            t0 = time.time()
            with svc.lock:
                results = svc.rr.rerank(
                    query, docs, top_n=top_n,
                    instruction=instr,
                    batch_size=req.get("batch_size"),
                )
            dt = time.time() - t0

            svc.requests += 1
            svc.pairs += len(docs)
            svc.total_s += dt

            return self._json({
                "model": MODEL_ALIAS,
                "results": [r.to_dict() for r in results],
                "count": len(results),
                "seconds": round(dt, 4),
                "pairs_per_s": round(len(docs) / dt, 1) if dt else None,
                "reranker_used": True,
                "instruction_type": label,
                "truncated": getattr(svc.rr, "last_truncated", 0),
                "batches": getattr(svc.rr, "last_batches", None),
                "cache": svc.rr.cache.stats(),
            })

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8114)
    ap.add_argument("--model-path",
                    default=str(Path(__file__).resolve().parents[2] / r"models\reranker\Qwen3-Reranker-0.6B"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=20)
    ap.add_argument("--max-batch-tokens", type=int, default=8192)
    ap.add_argument("--instruction-type", default="default",
                    choices=sorted(INSTRUCTIONS))
    args = ap.parse_args()

    cfg = RerankerConfig(
        model_path=args.model_path, device=args.device, dtype=args.dtype,
        max_length=args.max_length, batch_size=args.batch_size,
        max_batch_tokens=args.max_batch_tokens,
        instruction=INSTRUCTIONS[args.instruction_type],
    )
    print(f"loading {MODEL_ALIAS} ...", flush=True)
    svc = Service(cfg)
    info = svc.rr.info()
    print(f"device={info['device']} dtype={info['dtype']} max_length={info['max_length']} "
          f"body_budget={info['body_budget']} vram={info['vram_mb']} MiB "
          f"load={info['load_s']}s", flush=True)

    srv = ThreadingHTTPServer((args.host, args.port), make_handler(svc))
    print(f"listening on http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        svc.rr.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
