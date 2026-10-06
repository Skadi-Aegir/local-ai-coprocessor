"""
server.py - local-embedding HTTP service on 127.0.0.1:8113.

Endpoints:
  GET  /health
  GET  /v1/models
  POST /v1/embeddings     OpenAI-compatible embeddings
  POST /v1/retrieve       project-scoped vector retrieval (Phase 4 prototype)
  GET  /stats             index statistics

Design notes:
  * stdlib http.server, no web framework. This service does two things and adding
    FastAPI/uvicorn would add a dependency tree for no functional gain, while the
    project rule is that each component carries only what it needs.
  * Loopback only. It binds 127.0.0.1 and never 0.0.0.0; no LAN exposure.
  * The model and the FAISS index are loaded once at startup. Requests are serialised
    behind the embedder's own lock, because a torch module is not thread-safe for
    concurrent encode calls, and Phase 3 already showed concurrent GPU work on this
    card loses ~half its throughput rather than gaining anything.
  * `/v1/embeddings` accepts the OpenAI `input` as either a string or a list, and
    honours `dimensions` when supplied (Matryoshka truncation).
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from embedder import CODE_INSTRUCT, RETRIEVAL_INSTRUCT, Embedder, EmbedderConfig  # noqa: E402
from index_store import IndexStore  # noqa: E402
from retriever import Retriever, RetrieveConfig, instruction_for  # noqa: E402

MODEL_ALIAS = "local-embedding"
START_TS = time.time()


class Service:
    """Holds the loaded model plus one index per project."""

    def __init__(self, model_path: str, index_root: str, dimension: int,
                 batch_size: int, max_seq_length: int, projects: dict[str, int],
                 release_cache_mb: int = 256):
        self.dimension = dimension
        self.index_root = Path(index_root)
        self.emb = Embedder(EmbedderConfig(
            model_path=model_path, dimension=dimension,
            batch_size=batch_size, max_seq_length=max_seq_length,
            release_cache_mb=release_cache_mb))
        self._stores: dict[str, IndexStore] = {}
        self._lock = threading.Lock()
        self.projects = projects or {}
        # preload any project whose index already exists
        for name, dim in self.projects.items():
            self.get_store(name, dim, create=False)

    # ---------------------------------------------------------------- indexes
    def get_store(self, project: str, dimension: int | None = None,
                  create: bool = True) -> IndexStore | None:
        with self._lock:
            st = self._stores.get(project)
            if st is not None:
                return st
            dim = int(dimension or self.dimension)
            d = self.index_root / project
            if not d.exists() and not create:
                return None
            st = IndexStore(d, dim)
            self._stores[project] = st
            return st

    def retriever(self, project: str) -> Retriever | None:
        st = self.get_store(project, create=False)
        if st is None:
            return None
        return Retriever(self.emb, st, RetrieveConfig(top_k=5, instruction=None))

    def stats(self) -> dict:
        out = {}
        for name, st in self._stores.items():
            try:
                out[name] = st.stats()
            except Exception as e:  # noqa: BLE001
                out[name] = {"error": str(e)}
        return {
            "model": MODEL_ALIAS,
            "dimension": self.dimension,
            "device": self.emb.device,
            # cudaMemGetInfo total - free, kept for the scripts that print it. On this WDDM machine
            # it is neither the device-wide figure (nvidia-smi) nor this service's footprint
            # (`memory.torch_reserved_mb`); see Embedder.vram_mb.
            "vram_mb": self.emb.vram_mb(),
            "memory": self.emb.memory_stats(),
            "max_seq_length": self.emb.model.max_seq_length,
            "native_dimension": self.emb.native_dim,
            "uptime_s": round(time.time() - START_TS, 1),
            "projects": out,
        }

    def close(self) -> None:
        for st in self._stores.values():
            try:
                st.close()
            except Exception:
                pass


def make_handler(svc: Service):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "local-embedding/0.1"

        def log_message(self, *a):     # keep the console readable
            pass

        # ---------------------------------------------------------- helpers
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
            raw = self.rfile.read(n)
            return json.loads(raw.decode("utf-8"))

        # ---------------------------------------------------------- GET
        def do_GET(self):
            if self.path == "/health":
                return self._json({
                    "status": "ok",
                    "model": MODEL_ALIAS,
                    "device": svc.emb.device,
                    "dimension": svc.dimension,
                    "uptime_s": round(time.time() - START_TS, 1),
                })
            if self.path == "/v1/models":
                return self._json({
                    "object": "list",
                    "data": [{
                        "id": MODEL_ALIAS, "object": "model",
                        "owned_by": "local", "created": int(START_TS),
                        "meta": {
                            "dimension": svc.dimension,
                            "native_dimension": svc.emb.native_dim,
                            "max_seq_length": svc.emb.model.max_seq_length,
                            "pooling": "last_token",
                            "similarity": "cosine",
                        },
                    }],
                })
            if self.path == "/stats":
                return self._json(svc.stats())
            return self._error(404, f"unknown path {self.path}")

        # ---------------------------------------------------------- POST
        def do_POST(self):
            try:
                if self.path == "/v1/embeddings":
                    return self._embeddings()
                if self.path == "/v1/retrieve":
                    return self._retrieve()
                return self._error(404, f"unknown path {self.path}")
            except json.JSONDecodeError as e:
                return self._error(400, f"invalid JSON body: {e}")
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                return self._error(500, f"{type(e).__name__}: {e}", "server_error")

        def _embeddings(self):
            req = self._read_json()
            model = req.get("model", MODEL_ALIAS)
            if model != MODEL_ALIAS:
                return self._error(404, f"model '{model}' not found; use '{MODEL_ALIAS}'",
                                   "model_not_found")

            inp = req.get("input")
            if isinstance(inp, str):
                texts = [inp]
            elif isinstance(inp, list) and all(isinstance(x, str) for x in inp):
                texts = inp
            else:
                return self._error(400, "input must be a string or a list of strings")

            # OpenAI uses `input_type`-free API: instructions are a local extension.
            query_mode = bool(req.get("query", False))
            instr = req.get("instruction")
            if instr is None and query_mode:
                instr = RETRIEVAL_INSTRUCT

            t0 = time.time()
            if query_mode or instr:
                arr = svc.emb.encode_queries(texts, instruction=instr)
            else:
                arr = svc.emb.encode_documents(texts)
            dt = time.time() - t0

            # optional Matryoshka override per request
            dims = req.get("dimensions")
            if dims and int(dims) != svc.dimension:
                k = int(dims)
                if not (32 <= k <= svc.emb.native_dim):
                    return self._error(400, f"dimensions must be 32..{svc.emb.native_dim}")
                arr = arr[:, :k].copy()
                norms = np.linalg.norm(arr, axis=1, keepdims=True)
                arr = arr / np.where(norms == 0, 1.0, norms)

            data = [{"object": "embedding", "index": i, "embedding": arr[i].tolist()}
                    for i in range(arr.shape[0])]
            return self._json({
                "object": "list",
                "data": data,
                "model": MODEL_ALIAS,
                "usage": {"prompt_tokens": 0, "total_tokens": 0},
                "local": {
                    "dimension": int(arr.shape[1]),
                    "count": int(arr.shape[0]),
                    "seconds": round(dt, 4),
                    "chunks_per_s": round(arr.shape[0] / dt, 1) if dt else None,
                    "query_mode": query_mode,
                    "instruction": instr,
                },
            })

        def _retrieve(self):
            req = self._read_json()
            query = req.get("query")
            if not isinstance(query, str) or not query.strip():
                return self._error(400, "query must be a non-empty string")
            project = req.get("project")
            top_k = int(req.get("top_k", 5))
            if not (1 <= top_k <= 50):
                return self._error(400, "top_k must be 1..50")

            st = svc.get_store(project or "", create=False) if project else None
            if st is None:
                # fall back to the only loaded project, if there is exactly one
                loaded = list(svc._stores)
                if len(loaded) == 1:
                    project, st = loaded[0], svc._stores[loaded[0]]
                else:
                    return self._error(
                        404, f"no index for project '{project}'; "
                             f"loaded projects: {loaded}")
            if st.stats()["vectors_in_index"] == 0:
                return self._json({"results": [], "note": "index is empty",
                                   "project": project, "query": query})

            # instruction: explicit override, else by query type, else default
            itype = req.get("instruction_type")
            instr = req.get("instruction")
            if instr is None:
                instr = instruction_for(itype) if itype else RETRIEVAL_INSTRUCT

            t0 = time.time()
            retr = Retriever(svc.emb, st, RetrieveConfig(top_k=top_k, instruction=None))
            hits = retr.retrieve(query, top_k=top_k, instruction=instr, project=None)
            dt = time.time() - t0

            body = {
                "project": project,
                "query": query,
                "top_k": top_k,
                "instruction": instr,
                "seconds": round(dt, 4),
                "results": [h.to_dict() for h in hits],
            }
            if req.get("pack", False):
                body["context"] = retr.pack(hits)
            return self._json(body)

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Allow", "GET, POST, OPTIONS")
            self.send_header("Content-Length", "0")
            self.end_headers()

    return Handler


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8113)
    ap.add_argument("--model-path", default=str(Path(__file__).resolve().parents[2] / r"models\embedding\Qwen3-Embedding-0.6B"))
    ap.add_argument("--index-root", default=str(Path(__file__).resolve().parents[2] / r"vector\embedding"))
    ap.add_argument("--dimension", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-seq-length", type=int, default=8192)
    ap.add_argument("--release-cache-mb", type=int, default=256,
                    help="after an encode, release cached GPU blocks above this many MiB "
                         "(OPEN_ISSUES P1-11); 0 restores the old keep-everything behaviour")
    ap.add_argument("--project", action="append", default=[],
                    help="project=dimension, repeatable; preloads an existing index")
    args = ap.parse_args()

    projects: dict[str, int] = {}
    for spec in args.project:
        if "=" in spec:
            name, _, dim = spec.partition("=")
            projects[name.strip()] = int(dim)
        else:
            projects[spec.strip()] = args.dimension

    print(f"loading {MODEL_ALIAS} ...", flush=True)
    svc = Service(args.model_path, args.index_root, args.dimension,
                  args.batch_size, args.max_seq_length, projects,
                  release_cache_mb=args.release_cache_mb)
    info = svc.stats()
    mem = info.get("memory") or {}
    print(f"device={info['device']} dimension={info['dimension']} "
          f"vram_mb={info['vram_mb']} (cudaMemGetInfo; not the device-wide figure) "
          f"this_process_reserved={mem.get('torch_reserved_mb')} MiB "
          f"cache_release_above={mem.get('release_threshold_mb')} MiB", flush=True)
    print(f"projects preloaded: {list(info['projects'])}", flush=True)

    srv = ThreadingHTTPServer((args.host, args.port), make_handler(svc))
    print(f"listening on http://{args.host}:{args.port}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        svc.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
