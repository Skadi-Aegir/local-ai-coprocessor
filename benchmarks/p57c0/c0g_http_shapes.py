"""
P57-C0 - 8115 request-shape checks. Part of the acceptance condition that the local HTTP server's
normal, refusal and error responses stay consistent.

The earlier scratch version of this was never committed, so it is re-created here as a durable file.
Sends three malformed/size-violating requests and requires a STRUCTURED error with the right status.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8115"
OUT = (Path(__file__).resolve().parents[2] / r"benchmarks\p57c0\results\supplementary.json")


def call(payload: bytes | None, headers: dict | None = None, method: str = "POST",
         path: str = "/ask", timeout: float = 60) -> dict:
    req = urllib.request.Request(BASE + path, data=payload, headers=headers or {}, method=method)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"http": r.status, "body": json.loads(r.read().decode()),
                    "elapsed_s": round(time.time() - t0, 3)}
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            body = json.loads(raw)
        except Exception:  # noqa: BLE001
            body = {"raw": raw[:300]}
        return {"http": e.code, "body": body, "elapsed_s": round(time.time() - t0, 3)}
    except Exception as e:  # noqa: BLE001
        return {"http": None, "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.time() - t0, 3)}


def header_only_oversize() -> dict:
    """
    Headers only, Content-Length 70000, NO body. Sent over a raw socket because urllib will not let us
    declare a length we do not send. The server must reject from the HEADER, before reading the body -
    otherwise it would block waiting for bytes that never arrive.
    """
    t0 = time.time()
    try:
        s = socket.create_connection(("127.0.0.1", 8115), timeout=30)
        s.sendall(b"POST /ask HTTP/1.1\r\nHost: 127.0.0.1:8115\r\n"
                  b"Content-Type: application/json\r\nContent-Length: 70000\r\n\r\n")
        s.settimeout(30)
        buf = b""
        while True:
            try:
                b = s.recv(4096)
            except socket.timeout:
                break
            if not b:
                break
            buf += b
            if b"\r\n\r\n" in buf:
                head_b, body_b = buf.split(b"\r\n\r\n", 1)
                cl = None
                for line in head_b.decode("latin-1").splitlines():
                    if line.lower().startswith("content-length:"):
                        cl = int(line.split(":", 1)[1].strip())
                if cl is None or len(body_b) >= cl:
                    break
        s.close()
        raw = buf.decode("utf-8", errors="replace")
        status = int(raw.split(" ")[1]) if raw.startswith("HTTP/") else None
        body = {}
        if "\r\n\r\n" in raw:
            try:
                body = json.loads(raw.split("\r\n\r\n", 1)[1])
            except Exception:  # noqa: BLE001
                body = {"raw": raw.split("\r\n\r\n", 1)[1][:200]}
        return {"http": status, "body": body, "elapsed_s": round(time.time() - t0, 3),
                "sent_body": False, "declared_length": 70000}
    except Exception as e:  # noqa: BLE001
        return {"http": None, "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.time() - t0, 3)}


def main() -> int:
    rep: dict = {}
    print("=== 8115 request-shape checks ===\n")

    r1 = header_only_oversize()
    k1 = (r1.get("body") or {}).get("error", {}).get("kind")
    ok1 = r1["http"] == 413 and k1 == "body_too_large"
    print(f"[a] headers only, Content-Length 70000, NO body")
    print(f"    http={r1['http']} kind={k1}  expected 413/body_too_large  ok={ok1}  "
          f"elapsed={r1['elapsed_s']}s (rejected from the header)\n")

    r2 = call(b"[1,2]", {"Content-Type": "application/json"})
    k2 = (r2.get("body") or {}).get("error", {}).get("kind")
    ok2 = r2["http"] == 400 and k2 == "bad_request"
    print(f"[b] body [1,2] (non-object JSON)")
    print(f"    http={r2['http']} kind={k2}  expected 400/bad_request  ok={ok2}\n")

    r3 = call(json.dumps({"question": "x", "index": "D:/x"}).encode(),
              {"Content-Type": "application/json"})
    k3 = (r3.get("body") or {}).get("error", {}).get("kind")
    ok3 = r3["http"] == 400 and k3 == "index_override_rejected"
    print(f"[c] body carrying an index key")
    print(f"    http={r3['http']} kind={k3}  expected 400/index_override_rejected  ok={ok3}\n")

    h = call(None, method="GET", path="/health")
    print(f"=== /health after: inflight={h['body'].get('inflight')} "
          f"status={h['body'].get('status')}")

    rep.update({"body_too_large": r1, "non_object_json": r2, "index_override": r3,
                "health_after": h, "all_three_ok": bool(ok1 and ok2 and ok3)})
    print(f"\nall three as expected: {rep['all_three_ok']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0 if rep["all_three_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
