"""
p1_11_service_smoke.py - OPEN_ISSUES P1-11 at the SERVICE level: after a document embed through
/v1/embeddings, does local-embedding give its cache back?

  --start   launch services/retrieval/server.py from THIS tree on --port (never a project port),
            with an empty index root, its output going to a FILE (never a pipe: a full pipe buffer
            wedged child processes three times in this project, see OPEN_ISSUES P1-8), and stop
            it afterwards
  without   measure a service that is already running, e.g. the deployed 8113 once it carries the
            fix - the embed is 8 documents, light enough for a live service

Reads /stats before and after embedding the 8 longest Phase A chunks, and reports what the
service's own allocator kept.

Usage:
  python p1_11_service_smoke.py --start --port 8199 --run-id <id>
  python p1_11_service_smoke.py --port 8113 --run-id <id>
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PROJECT_PORTS = {8111, 8112, 8113, 8114, 8115}
MODEL_PATH = str(Path(__file__).resolve().parents[2] / r"models\embedding\Qwen3-Embedding-0.6B")
VENV_PY = str(Path(__file__).resolve().parents[2] / r"services\retrieval\venv\Scripts\python.exe")


def http(url: str, payload: dict | None = None, timeout: float = 120) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def port_in_use(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def longest_chunks(n: int) -> list[str]:
    sys.path.insert(0, str(REPO / "services" / "retrieval"))
    from chunker import chunk_file
    m = json.loads((REPO / "vector/qa/current/manifest.json").read_text(encoding="utf-8"))
    texts = []
    for rel in m["included_files"]:
        t = (REPO / rel).read_bytes().replace(b"\r\n", b"\n").decode("utf-8", errors="replace")
        texts += [c.content for c in chunk_file(t, rel, "local-ai", strategy="structured")]
    return sorted(texts, key=len, reverse=True)[:n]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--start", action="store_true")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--docs", type=int, default=8)
    args = ap.parse_args()
    out = HERE / "results" / f"{args.run_id}.json"
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3
    base = f"http://127.0.0.1:{args.port}"
    proc, log = None, None
    if args.start:
        if args.port in PROJECT_PORTS or port_in_use(args.port):
            print(f"REFUSING: port {args.port} is a project port or already in use")
            return 2
        log = HERE / "results" / f"{args.run_id}.server.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        idx = Path(tempfile.mkdtemp(prefix="p1_11_idx_"))
        with open(log, "wb") as fh:
            proc = subprocess.Popen([VENV_PY, str(REPO / "services/retrieval/server.py"),
                                     "--port", str(args.port), "--index-root", str(idx),
                                     "--model-path", MODEL_PATH],
                                    stdout=fh, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 180:
            try:
                if http(base + "/health", timeout=5).get("status") == "ok":
                    break
            except Exception:  # noqa: BLE001
                time.sleep(1)
        else:
            proc.terminate()
            print(f"service did not become healthy; see {log}")
            return 4
    report: dict = {"run_id": args.run_id, "port": args.port, "started_here": bool(proc)}
    try:
        docs = longest_chunks(args.docs)
        before = http(base + "/stats")
        t0 = time.time()
        body = http(base + "/v1/embeddings", {"model": "local-embedding", "input": docs})
        report["embed_seconds"] = round(time.time() - t0, 2)
        after = http(base + "/stats")
        mb, ma = before.get("memory") or {}, after.get("memory") or {}
        report.update({
            "docs": len(docs), "longest_chars": max(len(d) for d in docs),
            "vectors_returned": len(body.get("data") or []),
            "stats_before": before, "stats_after": after,
            "memory_field_present": bool(mb) and bool(ma),
        })
        if mb and ma:
            report["peak_above_weights_mb"] = round(ma["torch_peak_reserved_mb"]
                                                    - mb["torch_allocated_mb"], 1)
            report["retained_mb"] = round(ma["torch_reserved_mb"] - mb["torch_reserved_mb"], 1)
            report["cache_releases_during_embed"] = ma["cache_releases"] - mb["cache_releases"]
            report["verdict"] = ("released" if report["retained_mb"] < 256
                                 and report["cache_releases_during_embed"] >= 1 else "RETAINED")
        else:
            report["verdict"] = "service predates the P1-11 fix (no memory field)"
        print(json.dumps({k: v for k, v in report.items()
                          if k not in ("stats_before", "stats_after")}, indent=1))
    finally:
        if proc:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            time.sleep(1)
            report["port_free_after_stop"] = not port_in_use(args.port)
            print(f"stopped the service started here; port free: {report['port_free_after_stop']}")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0 if report.get("verdict") == "released" else 1


if __name__ == "__main__":
    raise SystemExit(main())
