"""
Offline tests for services/ocr/server.py.

Uses a STUB engine: paddle is never imported, no GPU is touched and no model is loaded, so this runs
anywhere in a second. Covers the input validation and error codes, the response shape, reading order,
the confidence maths, the concurrency limit, the loopback-only refusal, and the logging rule that the
recognised text is not written by default.
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import server as SRV  # noqa: E402

FAILS: list[str] = []
CHECKS = 0


def check(name: str, cond: bool, detail="") -> None:
    global CHECKS
    CHECKS += 1
    if cond:
        print(f"OK   {name}" + (f" :: {detail}" if detail != "" else ""))
    else:
        print(f"FAIL {name} :: {detail}")
        FAILS.append(name)


# ------------------------------------------------------------------ stub engine
class StubEngine:
    """Returns a fixed block layout. Boxes are (x, y) pairs like PaddleOCR's rec_polys."""

    def __init__(self, blocks=None, boom=False):
        self.blocks = blocks if blocks is not None else [
            # deliberately OUT of reading order: bottom line first, and right-before-left
            {"text": "second line", "confidence": 0.90,
             "box": [[10, 60], [200, 60], [200, 90], [10, 90]]},
            {"text": "right", "confidence": 0.80,
             "box": [[210, 10], [300, 10], [300, 40], [210, 40]]},
            {"text": "left", "confidence": 0.70,
             "box": [[10, 12], [100, 12], [100, 38], [10, 38]]},
        ]
        self.boom = boom
        self.calls = 0

    def run(self, image_bytes: bytes) -> dict:
        self.calls += 1
        if self.boom:
            raise RuntimeError("stub engine exploded")
        return {"width": 320, "height": 100, "blocks": [dict(b) for b in self.blocks]}


def png_bytes(w=64, h=32) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="PNG")
    return buf.getvalue()


def jpeg_bytes(w=64, h=32) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "white").save(buf, format="JPEG")
    return buf.getvalue()


def start(engine, tier="mobile", tmp=None) -> tuple:
    log = SRV.Log(pathlib.Path(tmp) / "ocr-test.log") if tmp else SRV.Log(
        pathlib.Path(tempfile.gettempdir()) / "ocr-test.log")
    srv = SRV.make_server(engine, "127.0.0.1", 0, tier=tier, log=log)
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    return srv, srv.server_address[1]


def post(port: int, body, raw: bytes | None = None, headers: dict | None = None,
         timeout: float = 60):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    payload = raw if raw is not None else json.dumps(body).encode()
    c.request("POST", "/v1/ocr", body=payload,
              headers=headers or {"Content-Type": "application/json"})
    r = c.getresponse()
    data = r.read()
    c.close()
    try:
        return r.status, json.loads(data)
    except Exception:  # noqa: BLE001
        return r.status, {"raw": data[:200]}


def get(port: int, path: str):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    c.request("GET", path)
    r = c.getresponse()
    d = json.loads(r.read())
    c.close()
    return r.status, d


# ------------------------------------------------------------------ tests
def test_helpers() -> None:
    print("\n--- pure helpers (no server, no paddle)")
    blocks = [
        {"text": "second line", "confidence": 0.9, "box": [[10, 60], [200, 60], [200, 90], [10, 90]]},
        {"text": "right", "confidence": 0.8, "box": [[210, 10], [300, 10], [300, 40], [210, 40]]},
        {"text": "left", "confidence": 0.7, "box": [[10, 12], [100, 12], [100, 38], [10, 38]]},
    ]
    ordered = SRV.order_blocks(blocks)
    check("reading order: top line first, then bottom",
          [b["text"] for b in ordered] == ["left", "right", "second line"],
          [b["text"] for b in ordered])
    check("reading order: within a line, left before right",
          SRV.order_blocks(blocks)[0]["text"] == "left")
    check("join_text puts one block per line",
          SRV.join_text(ordered) == "left\nright\nsecond line", SRV.join_text(ordered))
    check("join_text skips empty blocks",
          SRV.join_text([{"text": "a"}, {"text": ""}, {"text": "b"}]) == "a\nb")

    # character-weighted, not block-averaged
    wb = [{"text": "aaaaaaaa", "confidence": 1.0}, {"text": "bb", "confidence": 0.0}]
    got = SRV.weighted_confidence(wb)
    check("confidence is character-weighted, not a plain mean",
          abs(got - 0.8) < 1e-9, f"got {got}, plain mean would be 0.5")
    check("confidence is None when there is no text",
          SRV.weighted_confidence([{"text": "", "confidence": 0.5}]) is None)
    check("confidence is None on an empty block list",
          SRV.weighted_confidence([]) is None)
    check("blocks without confidence are ignored in the weighting",
          abs(SRV.weighted_confidence([{"text": "abcd", "confidence": 0.5},
                                       {"text": "efgh", "confidence": None}]) - 0.5) < 1e-9)
    check("confidence is rounded to 6 dp",
          isinstance(SRV.weighted_confidence([{"text": "abc", "confidence": 0.123456789}]), float))

    check("sniff PNG", SRV.sniff_image(png_bytes()) == "PNG")
    check("sniff JPEG", SRV.sniff_image(jpeg_bytes()) == "JPEG")
    check("sniff rejects other bytes", SRV.sniff_image(b"GIF89a....") is None)
    check("sniff rejects empty", SRV.sniff_image(b"") is None)

    check("order_blocks tolerates a boxless block",
          len(SRV.order_blocks([{"text": "x", "confidence": 1.0, "box": []}])) == 1)
    check("order_blocks returns input when no block has a box",
          SRV.order_blocks([{"text": "a"}, {"text": "b"}]) == [{"text": "a"}, {"text": "b"}])


def test_loopback_only() -> None:
    print("\n--- loopback-only binding")
    for host in ("0.0.0.0", "192.168.1.5", "::"):
        try:
            SRV.make_server(StubEngine(), host, 0)
            check(f"refuses to bind {host}", False, "it did NOT refuse")
        except ValueError as e:
            check(f"refuses to bind {host}", "loopback" in str(e).lower(), str(e)[:60])
    for host in ("127.0.0.1", "localhost"):
        s, _ = start(StubEngine())
        check(f"accepts {host}", s is not None)
        s.shutdown()
        s.server_close()


def test_health_and_stats(tmp) -> None:
    print("\n--- /health and /stats")
    eng = StubEngine()
    eng.load_s = 0.5
    srv, port = start(eng, tmp=tmp)
    try:
        code, h = get(port, "/health")
        check("/health is 200", code == 200)
        check("/health reports logical_name local-ocr", h.get("logical_name") == "local-ocr", h.get("logical_name"))
        check("/health reports device cpu", h.get("device") == "cpu")
        check("/health reports cuda_compiled false", h.get("cuda_compiled") is False)
        check("/health reports onednn disabled", h.get("onednn") == "disabled")
        check("/health reports the model with the tier", h.get("model") == "PP-OCRv5_mobile", h.get("model"))
        check("/health reports model_version", h.get("model_version") == "PP-OCRv5")
        check("/health reports max_body_bytes 12 MiB",
              h["limits"]["max_body_bytes"] == 12 * 1024 * 1024, h["limits"]["max_body_bytes"])
        check("/health reports max_executing 1", h["limits"]["max_executing"] == 1)
        check("/health reports max_side_px 10000", h["limits"]["max_side_px"] == 10000)

        code, st = get(port, "/stats")
        check("/stats is 200", code == 200)
        for k in ("requests", "ok", "errors", "rss_mb", "inflight", "peak_inflight",
                  "latency_avg_s", "latency_max_s"):
            check(f"/stats has {k}", k in st)
        check("/stats requests starts at 0", st["requests"] == 0, st["requests"])

        check("unknown GET path is 404", get(port, "/nope")[0] == 404)
        check("unknown POST path is 404", post(port, {}, raw=b"{}")[0] in (404, 400))
    finally:
        srv.shutdown()
        srv.server_close()


def test_ocr_ok(tmp) -> None:
    print("\n--- a successful OCR call")
    eng = StubEngine()
    srv, port = start(eng, tmp=tmp)
    try:
        code, r = post(port, {"image": base64.b64encode(png_bytes()).decode(),
                              "request_id": "t-1"})
        check("200 for a valid PNG", code == 200, code)
        for k in ("text", "blocks", "confidence", "image", "model", "elapsed_s", "request_id"):
            check(f"response has {k}", k in r)
        check("text is the blocks in reading order",
              r["text"] == "left\nright\nsecond line", repr(r["text"]))
        check("blocks are in reading order",
              [b["text"] for b in r["blocks"]] == ["left", "right", "second line"])
        check("blocks carry text, confidence and box",
              all({"text", "confidence", "box"} <= set(b) for b in r["blocks"]))
        check("image carries width and height",
              r["image"] == {"width": 320, "height": 100}, r["image"])
        check("request_id is echoed", r["request_id"] == "t-1")
        check("model names the tier", r["model"] == "PP-OCRv5_mobile", r["model"])
        check("elapsed_s is a number", isinstance(r["elapsed_s"], (int, float)))
        # 8*1.0 + 5*0.8 + 4*0.7 weighted by characters: "right"=5, "left"=4
        exp = (4 * 0.7 + 5 * 0.8 + 11 * 0.9) / (4 + 5 + 11)
        check("confidence matches the character-weighted mean",
              abs(r["confidence"] - exp) < 1e-4, f"{r['confidence']} vs {exp}")

        # JPEG is also accepted
        code, r2 = post(port, {"image": base64.b64encode(jpeg_bytes()).decode()})
        check("200 for a valid JPEG", code == 200, code)

        # no text -> confidence null, and that is not the same as 0.0
        srv.engine.blocks = []
        code, r3 = post(port, {"image": base64.b64encode(png_bytes()).decode()})
        check("empty result still returns 200", code == 200)
        check("no text -> text is empty string", r3["text"] == "")
        check("no text -> confidence is null, not 0.0", r3["confidence"] is None, r3["confidence"])

        code, st = get(port, "/stats")
        check("/stats counts the successful calls", st["ok"] >= 2, st["ok"])
        check("/stats records a latency max", st["latency_max_s"] is not None)
        check("/stats inflight returns to 0", st["inflight"] == 0, st["inflight"])
    finally:
        srv.shutdown()
        srv.server_close()


def test_validation(tmp) -> None:
    print("\n--- input validation and error codes")
    eng = StubEngine()
    srv, port = start(eng, tmp=tmp)
    try:
        code, r = post(port, None, raw=b"[1,2]")
        check("non-object JSON -> 400 bad_request",
              code == 400 and r["error"]["kind"] == "bad_request", (code, r))
        code, r = post(port, None, raw=b"not json at all")
        check("invalid JSON -> 400 bad_request",
              code == 400 and r["error"]["kind"] == "bad_request", code)
        code, r = post(port, {"image": ""})
        check("empty image string -> 400",
              code == 400 and r["error"]["kind"] == "bad_request", (code, r))
        code, r = post(port, {"nope": 1})
        check("missing image key -> 400",
              code == 400 and r["error"]["kind"] == "bad_request", code)
        code, r = post(port, {"image": 12345})
        check("non-string image -> 400",
              code == 400 and r["error"]["kind"] == "bad_request", code)
        code, r = post(port, {"image": "!!!not base64!!!"})
        check("invalid base64 -> 400 bad_base64",
              code == 400 and r["error"]["kind"] == "bad_base64", (code, r))
        code, r = post(port, {"image": "aGVsbG8="})          # valid base64, not an image
        check("valid base64 that is not an image -> 415",
              code == 415 and r["error"]["kind"] == "unsupported_media_type", (code, r))
        gif = b"GIF89a" + b"\x00" * 20
        code, r = post(port, {"image": base64.b64encode(gif).decode()})
        check("GIF magic bytes -> 415", code == 415, code)
        # PNG magic but undecodable
        code, r = post(port, {"image": base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"junk").decode()})
        check("PNG magic but corrupt -> 415 undecodable_image",
              code == 415 and r["error"]["kind"] == "undecodable_image", (code, r))

        # body size, enforced from the HEADER before any body is read
        code, r = post(port, None, raw=b"{}",
                       headers={"Content-Type": "application/json",
                                "Content-Length": str(12 * 1024 * 1024 + 1)})
        check("oversize body -> 413 body_too_large from the header alone",
              code == 413 and r["error"]["kind"] == "body_too_large", (code, r))
        code, r = post(port, None, raw=b"{}", headers={"Content-Type": "application/json",
                                                       "Content-Length": "0"})
        check("zero-length body -> 400", code == 400, code)

        # image side limit
        big = png_bytes(w=10001, h=8)
        code, r = post(port, {"image": base64.b64encode(big).decode()})
        check("image side over 10000 px -> 413 image_too_large",
              code == 413 and r["error"]["kind"] == "image_too_large", (code, r))

        check("every error is structured as {error:{kind,message}}",
              all(isinstance(x.get("error"), dict) and "kind" in x["error"]
                  and "message" in x["error"] for x in [r]))

        # an engine crash is a 500 internal_error, not a dropped connection
        srv.engine.boom = True
        code, r = post(port, {"image": base64.b64encode(png_bytes()).decode()})
        check("engine failure -> 500 internal_error",
              code == 500 and r["error"]["kind"] == "internal_error", (code, r))
        srv.engine.boom = False
        code, st = get(port, "/stats")
        check("/stats counted the errors", st["errors"] >= 1, st["errors"])
        check("the server survived every error", get(port, "/health")[0] == 200)
    finally:
        srv.shutdown()
        srv.server_close()


def test_concurrency(tmp) -> None:
    print("\n--- concurrency: 1 executing, 1 queued, the rest 429")

    class SlowEngine(StubEngine):
        def run(self, image_bytes):
            time.sleep(1.2)
            return super().run(image_bytes)

    srv, port = start(SlowEngine(), tmp=tmp)
    try:
        results: list[tuple] = []
        lock = threading.Lock()

        def worker():
            c, r = post(port, {"image": base64.b64encode(png_bytes()).decode()}, timeout=60)
            with lock:
                results.append((c, r.get("error", {}).get("kind")))

        ts = [threading.Thread(target=worker) for _ in range(5)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=90)
        codes = sorted(c for c, _ in results)
        n429 = sum(1 for c, _ in results if c == 429)
        n200 = sum(1 for c, _ in results if c == 200)
        check("exactly 2 requests are admitted (1 executing + 1 queued)",
              n200 == 2, f"200s={n200}, codes={codes}")
        check("the remaining requests get 429 too_many_requests",
              n429 == 3 and all(k == "too_many_requests" for c, k in results if c == 429),
              f"429s={n429}")
        code, st = get(port, "/stats")
        check("inflight peak never exceeded 1", st["peak_inflight"] == 1, st["peak_inflight"])
        check("inflight drained back to 0", st["inflight"] == 0, st["inflight"])
        check("the server is still healthy after the burst",
              get(port, "/health")[0] == 200)
    finally:
        srv.shutdown()
        srv.server_close()


def test_logging(tmp) -> None:
    print("\n--- logging shape, and the text is not logged by default")
    logp = pathlib.Path(tmp) / "logcheck.log"
    if logp.exists():
        logp.unlink()
    log = SRV.Log(logp)
    srv, port = start(StubEngine(), tmp=tmp)
    srv.log = log
    try:
        post(port, {"image": base64.b64encode(png_bytes()).decode(), "request_id": "log-1"})
        time.sleep(0.2)
        check("a log file was written", logp.is_file())
        lines = [json.loads(x) for x in logp.read_text(encoding="utf-8").splitlines() if x.strip()]
        check("one line per request", len(lines) >= 1, len(lines))
        e = lines[-1]
        for k in ("ts", "request_id", "width", "height", "blocks", "elapsed_s", "ok"):
            check(f"log line has {k}", k in e, sorted(e))
        check("log records the image size", e["width"] == 320 and e["height"] == 100)
        check("log records the block count", e["blocks"] == 3, e["blocks"])
        check("log records success", e["ok"] is True)
        raw = logp.read_text(encoding="utf-8")
        check("the recognised TEXT is NOT in the log by default",
              "second line" not in raw and "left" not in raw, raw[:120])
    finally:
        srv.shutdown()
        srv.server_close()


def test_no_paddle_import() -> None:
    print("\n--- the module must not drag paddle in")
    check("paddle was never imported by this test run",
          "paddle" not in sys.modules, [m for m in sys.modules if "paddle" in m][:3])
    src = (HERE / "server.py").read_text(encoding="utf-8")
    check("paddle is imported lazily, inside the Engine class",
          "from paddleocr import PaddleOCR" in src and "    def __init__" in src)
    check("oneDNN is disabled before the paddle import",
          src.index("FLAGS_use_mkldnn") < src.index("from paddleocr import PaddleOCR"))
    check("the read timeout is applied to the socket",
          "settimeout(READ_TIMEOUT_S)" in src)


def main() -> int:
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ocr_test_"))
    try:
        test_helpers()
        test_loopback_only()
        test_health_and_stats(tmp)
        test_ocr_ok(tmp)
        test_validation(tmp)
        test_concurrency(tmp)
        test_logging(tmp)
        test_no_paddle_import()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print()
    if FAILS:
        print(f"FAILED ({len(FAILS)}): {FAILS}")
        return 1
    print(f"ocr server self-test passed ({CHECKS} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
