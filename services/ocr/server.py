"""
local-ocr service - Phase 6. Stdlib HTTP server, loopback only, CPU only.

    GET  /health   liveness, logical name, model name and version, device
    GET  /stats    process memory, request counts, average and max latency
    POST /v1/ocr    {"image": "<base64>", "request_id": optional}

Response:
    {text, blocks:[{text, confidence, box}], confidence, image:{width,height}, model, elapsed_s}

`blocks` are in READING ORDER (top-to-bottom, then left-to-right); `text` joins them line by line;
`confidence` is a character-count-weighted mean of block confidences, and null when no text was found.

Implementation choices follow services/retrieval/server.py and cli/ask_server.py:
  * body size is checked from Content-Length BEFORE the body is read (max 12 MiB)
  * 1 request executing, 1 more admitted and queued, the rest 429
  * request reads have a timeout
  * every error is structured as {"error": {"kind", "message"}}
  * the socket is NEVER bound to anything but loopback

Engine note: oneDNN must be disabled on this paddlepaddle build, otherwise predict raises
    NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support ...
See docs/models/local-ocr.md section 3. The environment variable is set before paddle is imported.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
MODELS = REPO / "models/ocr"
LOGS = REPO / "logs"

# MUST precede the paddle import - oneDNN is unusable on this build.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(MODELS / "_rt_cache"))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

MAX_BODY_BYTES = 12 * 1024 * 1024
MAX_SIDE_PX = 10000
MAX_EXECUTING = 1
MAX_ADMITTED = 2
READ_TIMEOUT_S = 30.0
LOGICAL_NAME = "local-ocr"
MODEL_VERSION = "PP-OCRv5"

_exec_sem = threading.BoundedSemaphore(MAX_EXECUTING)
_admit_sem = threading.BoundedSemaphore(MAX_ADMITTED)
_state_lock = threading.Lock()
_state = {"requests": 0, "ok": 0, "errors": 0, "inflight": 0, "peak_inflight": 0,
          "latency_sum": 0.0, "latency_max": 0.0}


def read_config(path: Path) -> dict:
    """Minimal YAML-subset reader, same spirit as scripts/lf_config.py: scalars and simple maps."""
    cfg: dict = {}
    cur = cfg
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if ":" not in line:
            continue
        k, v = line.strip().split(":", 1)
        v = v.strip().strip('"').strip("'").replace("${REPO}", str(REPO))
        if not v:
            if indent == 0:
                cfg[k] = {}
                cur = cfg[k]
            continue
        target = cfg if indent == 0 else cur
        try:
            target[k] = int(v)
        except ValueError:
            try:
                target[k] = float(v)
            except ValueError:
                target[k] = {"true": True, "false": False}.get(v.lower(), v)
    return cfg


class Log:
    """One line per request. The recognised text is NOT logged by default."""

    def __init__(self, path: Path, log_text: bool = False):
        self.path = path
        self.log_text = log_text
        self.lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **f) -> None:
        f["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(f, ensure_ascii=False) + "\n")


class Engine:
    """Wraps PaddleOCR. Imported lazily so the module can be unit-tested without paddle."""

    def __init__(self, tier: str = "mobile", threads: int = 8, model_dir: Path = MODELS):
        self.tier = tier
        self.load_s: float | None = None
        t0 = time.time()
        import paddle
        try:
            paddle.set_num_threads(threads)
        except Exception:  # noqa: BLE001
            pass
        from paddleocr import PaddleOCR
        self._ocr = PaddleOCR(
            text_detection_model_name=f"PP-OCRv5_{tier}_det",
            text_recognition_model_name=f"PP-OCRv5_{tier}_rec",
            text_detection_model_dir=str(model_dir / f"PP-OCRv5_{tier}_det"),
            text_recognition_model_dir=str(model_dir / f"PP-OCRv5_{tier}_rec"),
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=False,
        )
        self.load_s = round(time.time() - t0, 3)

    def run(self, image_bytes: bytes) -> dict:
        """Returns the raw block list for one image, in reading order."""
        import numpy as np
        from PIL import Image
        img = Image.open(io.BytesIO(image_bytes))
        w, h = img.size
        arr = np.asarray(img.convert("RGB"))
        res = self._ocr.predict(arr)
        texts, scores, polys = [], [], []
        for r in res:
            j = r.json if hasattr(r, "json") else r
            if callable(j):
                j = j()
            d = j.get("res", j) if isinstance(j, dict) else {}
            texts.extend(d.get("rec_texts") or [])
            scores.extend(d.get("rec_scores") or [])
            for b in (d.get("rec_polys") or d.get("dt_polys") or []):
                try:
                    polys.append([[float(p[0]), float(p[1])] for p in b])
                except Exception:  # noqa: BLE001
                    polys.append([])
        blocks = []
        for i, t in enumerate(texts):
            box = polys[i] if i < len(polys) else []
            blocks.append({"text": str(t),
                           "confidence": float(scores[i]) if i < len(scores) else None,
                           "box": box})
        return {"width": w, "height": h, "blocks": blocks}


# ---------------------------------------------------------------------------- pure helpers
# These are what the offline tests exercise: they need no paddle and no GPU.

def order_blocks(blocks: list[dict]) -> list[dict]:
    """
    Reading order: top-to-bottom, then left-to-right.

    Blocks are grouped into visual lines by vertical overlap, then sorted left to right within a
    line. A block with no box is kept in its original relative position at the end, so a boxless
    engine cannot silently reorder text.
    """
    def top(b):
        return min((p[1] for p in b.get("box") or []), default=None)

    def left(b):
        return min((p[0] for p in b.get("box") or []), default=None)

    def bottom(b):
        return max((p[1] for p in b.get("box") or []), default=None)

    withbox = [b for b in blocks if b.get("box")]
    nobox = [b for b in blocks if not b.get("box")]
    if not withbox:
        return list(blocks)

    withbox.sort(key=lambda b: (top(b) if top(b) is not None else 0.0))
    lines: list[list[dict]] = []
    for b in withbox:
        placed = False
        for ln in lines:
            ref_t = top(ln[0])
            ref_b = bottom(ln[0])
            bt, bb = top(b), bottom(b)
            if bt is None or ref_t is None or ref_b is None or bb is None:
                continue
            # same visual line when the vertical spans overlap by more than half the shorter one
            overlap = min(ref_b, bb) - max(ref_t, bt)
            shorter = min(ref_b - ref_t, bb - bt) or 1.0
            if overlap > 0.5 * shorter:
                ln.append(b)
                placed = True
                break
        if not placed:
            lines.append([b])
    lines.sort(key=lambda ln: top(ln[0]) if top(ln[0]) is not None else 0.0)
    out: list[dict] = []
    for ln in lines:
        ln.sort(key=lambda b: left(b) if left(b) is not None else 0.0)
        out.extend(ln)
    out.extend(nobox)
    return out


def join_text(blocks: list[dict]) -> str:
    """Line by line: a block, then a newline. Empty blocks contribute nothing."""
    return "\n".join(b["text"] for b in blocks if b.get("text"))

def weighted_confidence(blocks: list[dict]) -> float | None:
    """
    Character-count-weighted mean of block confidences. Blocks without a confidence are ignored.
    Returns None when there is no text at all (which is different from 0.0).
    """
    num = 0.0
    den = 0
    for b in blocks:
        t = b.get("text") or ""
        c = b.get("confidence")
        if c is None or not t:
            continue
        n = len(t)
        num += float(c) * n
        den += n
    if den == 0:
        return None
    return round(num / den, 6)


def sniff_image(data: bytes) -> str | None:
    """PNG or JPEG by magic bytes. Returns the format or None."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "PNG"
    if data[:3] == b"\xff\xd8\xff":
        return "JPEG"
    return None


class Handler(BaseHTTPRequestHandler):
    server_version = "localOCR/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet by default; we log structurally instead
        if getattr(self.server, "verbose", False):
            super().log_message(*a)

    def _send(self, obj: dict, code: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, code: int, kind: str, message: str) -> None:
        with _state_lock:
            _state["errors"] += 1
        self._send({"error": {"kind": kind, "message": message}}, code)

    # ---------------------------------------------------------------- GET
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            eng = getattr(self.server, "engine", None)
            self._send({
                "status": "ok",
                "logical_name": LOGICAL_NAME,
                "model": getattr(self.server, "model_name", MODEL_VERSION),
                "model_version": MODEL_VERSION,
                "tier": getattr(self.server, "tier", None),
                "device": "cpu",
                "cuda_compiled": False,
                "onednn": "disabled",
                "model_load_s": getattr(eng, "load_s", None),
                "limits": {"max_body_bytes": MAX_BODY_BYTES, "max_side_px": MAX_SIDE_PX,
                           "max_executing": MAX_EXECUTING, "max_admitted": MAX_ADMITTED,
                           "read_timeout_s": READ_TIMEOUT_S},
                "note": "liveness of this process only",
            })
            return
        if self.path == "/stats":
            import psutil
            p = psutil.Process()
            with _state_lock:
                st = dict(_state)
            n = st["ok"] or 0
            self._send({
                "requests": st["requests"], "ok": st["ok"], "errors": st["errors"],
                "inflight": st["inflight"], "peak_inflight": st["peak_inflight"],
                "latency_avg_s": round(st["latency_sum"] / n, 4) if n else None,
                "latency_max_s": round(st["latency_max"], 4) if st["latency_max"] else None,
                "rss_mb": round(p.memory_info().rss / 1024 / 1024, 1),
                "uptime_s": round(time.time() - getattr(self.server, "t0", time.time()), 1),
            })
            return
        self._error(404, "not_found", f"unknown path {self.path}")

    # ---------------------------------------------------------------- POST
    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/v1/ocr":
            self._error(404, "not_found", f"unknown path {self.path}")
            return

        # size first, from the header, BEFORE reading
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self._error(400, "bad_request", "Content-Length is not a number")
            return
        if length <= 0:
            self._error(400, "bad_request", "empty body")
            return
        if length > MAX_BODY_BYTES:
            self._error(413, "body_too_large", f"body exceeds {MAX_BODY_BYTES} bytes")
            return

        self.connection.settimeout(READ_TIMEOUT_S)
        try:
            raw = self.rfile.read(length)
        except Exception as e:  # noqa: BLE001
            self._error(408, "read_timeout", f"{type(e).__name__}: {e}")
            return
        try:
            payload = json.loads(raw.decode("utf-8"))
        except UnicodeDecodeError as e:
            self._error(400, "bad_request", f"body is not UTF-8: {e}")
            return
        except json.JSONDecodeError as e:
            self._error(400, "bad_request", f"invalid JSON: {e}")
            return
        if not isinstance(payload, dict):
            self._error(400, "bad_request",
                        f"body must be a JSON object, got {type(payload).__name__}")
            return
        b64 = payload.get("image")
        if not isinstance(b64, str) or not b64:
            self._error(400, "bad_request", "'image' must be a non-empty base64 string")
            return
        rid = payload.get("request_id")
        if rid is not None and not isinstance(rid, str):
            rid = str(rid)
        try:
            data = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError) as e:
            self._error(400, "bad_base64", f"image is not valid base64: {e}")
            return
        if not data:
            self._error(400, "bad_base64", "image decoded to zero bytes")
            return

        fmt = sniff_image(data)
        if fmt is None:
            self._error(415, "unsupported_media_type",
                        "image must be PNG or JPEG (detected by magic bytes)")
            return
        try:
            from PIL import Image
            with Image.open(io.BytesIO(data)) as im:
                w, h = im.size
        except Exception as e:  # noqa: BLE001
            self._error(415, "undecodable_image",
                        f"could not decode as {fmt}: {type(e).__name__}: {e}")
            return
        if max(w, h) > MAX_SIDE_PX:
            self._error(413, "image_too_large",
                        f"image side {max(w, h)} px exceeds {MAX_SIDE_PX} px")
            return

        if not _admit_sem.acquire(blocking=False):
            self._error(429, "too_many_requests",
                        f"more than {MAX_ADMITTED} requests admitted; retry shortly")
            return
        t0 = time.time()
        try:
            # second gate: only MAX_EXECUTING run at once; the rest wait here
            _exec_sem.acquire()
            with _state_lock:
                _state["inflight"] += 1
                _state["peak_inflight"] = max(_state["peak_inflight"], _state["inflight"])
            try:
                out = self.server.engine.run(data)          # type: ignore[attr-defined]
            finally:
                with _state_lock:
                    _state["inflight"] -= 1
                _exec_sem.release()

            blocks = order_blocks(out["blocks"])
            elapsed = time.time() - t0
            with _state_lock:
                _state["requests"] += 1
                _state["ok"] += 1
                _state["latency_sum"] += elapsed
                _state["latency_max"] = max(_state["latency_max"], elapsed)
            self._send({
                "text": join_text(blocks),
                "blocks": blocks,
                "confidence": weighted_confidence(blocks),
                "image": {"width": out["width"], "height": out["height"]},
                "model": f"{MODEL_VERSION}_{self.server.tier}",   # type: ignore[attr-defined]
                "elapsed_s": round(elapsed, 4),
                "request_id": rid,
            })
            self.server.log.write(request_id=rid, width=out["width"], height=out["height"],  # type: ignore[attr-defined]
                                  blocks=len(blocks), elapsed_s=round(elapsed, 4),
                                  ok=True, tier=self.server.tier)                # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            with _state_lock:
                _state["requests"] += 1
            self.server.log.write(request_id=rid, ok=False,  # type: ignore[attr-defined]
                                  error=f"{type(e).__name__}: {e}")
            self._error(500, "internal_error", f"{type(e).__name__}: {e}")
        finally:
            _admit_sem.release()


def make_server(engine, host: str = "127.0.0.1", port: int = 8116, tier: str = "mobile",
                log: Log | None = None, verbose: bool = False) -> ThreadingHTTPServer:
    if host not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError(f"refusing to bind {host!r}: this service is loopback-only")
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    srv.engine = engine                        # type: ignore[attr-defined]
    srv.tier = tier                            # type: ignore[attr-defined]
    srv.model_name = f"{MODEL_VERSION}_{tier}"  # type: ignore[attr-defined]
    srv.log = log or Log(LOGS / "local-ocr.log")  # type: ignore[attr-defined]
    srv.verbose = verbose                      # type: ignore[attr-defined]
    srv.t0 = time.time()                       # type: ignore[attr-defined]
    return srv


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8116)
    ap.add_argument("--config", default=str(REPO / "config/local-ocr.yaml"))
    ap.add_argument("--tier", default=None)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg = read_config(Path(args.config)) if Path(args.config).is_file() else {}
    tier = args.tier or cfg.get("tier") or "mobile"
    if tier not in ("mobile", "server"):
        print(f"REFUSED: unknown tier {tier!r}")
        return 2
    threads = args.threads or int(cfg.get("threads") or 8)

    print(f"{LOGICAL_NAME}: loading PP-OCRv5_{tier} on CPU (oneDNN disabled)")
    eng = Engine(tier=tier, threads=threads)
    print(f"{LOGICAL_NAME}: model loaded in {eng.load_s}s")

    srv = make_server(eng, args.host, args.port, tier=tier, verbose=args.verbose)
    print(f"{LOGICAL_NAME}: listening on http://{args.host}:{args.port}")
    print(f"  tier {tier}  device cpu  threads {threads}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
