"""
b_client.py - Phase B end-to-end validation with a REAL client.

Covers the five scenarios the phase requires:
  1 normal answer
  2 answer-absent (corpus has no answer -> refusal)
  3 stale index (a cited source changes after indexing -> the citation cannot be verified)
  4 service unavailable (the embedding service down -> structured error, not a hang)
  5 concurrency and timeout (limits enforced, no resource-boundary bypass)

Records latency, VRAM and recovery behaviour throughout.

Usage: python b_client.py --out results/b_e2e.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = "http://127.0.0.1:8115"


def free_vram() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


def run_start_stop(script: str, timeout: int = 600) -> int:
    """
    Run a start/stop script with its output going to a FILE, never a pipe.

    Measured reason: this client originally used subprocess.run(..., capture_output=True) on
    start-local-embedding.ps1, and the child blocked once it had written more than the pipe buffer
    while the parent blocked waiting for it to exit. Two powershell processes were found wedged for
    1867 s. It is the same pipe-buffer deadlock that already cost this project runs during P55R-02
    and P55R-04; the fix is the same and belongs here too.
    """
    log = REPO / "logs" / f"b-{Path(script).stem}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "wb") as fh:
        r = subprocess.run(["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
                            "-ExecutionPolicy", "Bypass", "-File", str(REPO / "scripts" / script)],
                           stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
    return r.returncode


def call(path: str, payload: dict | None = None, timeout: float = 300) -> dict:
    url = BASE + path
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST" if data else "GET")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = json.loads(r.read().decode())
            return {"http": r.status, "body": body, "elapsed_s": round(time.time() - t0, 3)}
    except urllib.error.HTTPError as e:
        raw = e.read().decode()
        try:
            body = json.loads(raw)
        except Exception:  # noqa: BLE001
            body = {"raw": raw[:200]}
        return {"http": e.code, "body": body, "elapsed_s": round(time.time() - t0, 3)}
    except Exception as e:  # noqa: BLE001
        return {"http": None, "error": f"{type(e).__name__}: {e}",
                "elapsed_s": round(time.time() - t0, 3)}


class VramWatch:
    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.vals: list[int] = []
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            v = free_vram()
            if v:
                self.vals.append(v)
            time.sleep(self.interval)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *e):
        self.stop.set()
        self.t.join(timeout=5)
        return False

    def summary(self) -> dict:
        if not self.vals:
            return {"n": 0}
        return {"n": len(self.vals), "free_min": min(self.vals), "free_max": max(self.vals),
                "margin_mb": 2048, "margin_held": min(self.vals) >= 2048}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3
    out.parent.mkdir(parents=True, exist_ok=True)

    rep: dict = {"base": BASE, "scenarios": {}, "vram": {}, "limits_verified": {}}
    print(f"free VRAM at start: {free_vram()} MiB")

    # ---------------- 0: health
    with VramWatch() as w0:
        h = call("/health")
    rep["scenarios"]["health"] = h
    rep["vram"]["health"] = w0.summary()
    print(f"\n[health] http={h['http']} status={h['body'].get('status')} "
          f"index={h['body'].get('index_version')}")

    # ---------------- 1: normal answer
    with VramWatch() as w:
        r1 = call("/ask", {"question": "What port does the local-fast service listen on, and what "
                                       "context size does it use by default?"})
    rep["scenarios"]["normal_answer"] = r1
    rep["vram"]["normal_answer"] = w.summary()
    b = r1["body"]
    print(f"\n[1 normal] http={r1['http']} outcome={b.get('outcome')} "
          f"cites={len(b.get('citations') or [])} elapsed={b.get('elapsed_s')}s")
    if b.get("citations"):
        c = b["citations"][0]
        print(f"    {c['source_path']}:{c['start_line']}-{c['end_line']} {c['verification']}")
    print(f"    answer: {str(b.get('answer'))[:120]!r}")

    # ---------------- 2: answer-absent
    with VramWatch() as w:
        r2 = call("/ask", {"question": "What is the monthly monetary cost of running local-fast on "
                                       "this machine?"})
    rep["scenarios"]["answer_absent"] = r2
    rep["vram"]["answer_absent"] = w.summary()
    b2 = r2["body"]
    print(f"\n[2 absent] http={r2['http']} outcome={b2.get('outcome')} "
          f"cites={len(b2.get('citations') or [])} elapsed={b2.get('elapsed_s')}s")
    print(f"    answer: {str(b2.get('answer'))[:120]!r}")

    # ---------------- 3: index override is rejected (client cannot choose a corpus)
    r3a = call("/ask", {"question": "anything", "index": "D:\\somewhere\\else"})
    rep["scenarios"]["index_override_rejected"] = r3a
    print(f"\n[3a index override] http={r3a['http']} "
          f"kind={(r3a['body'].get('error') or {}).get('kind')}")

    # ---------------- 3b: stale index - change a cited source, then ask
    target = REPO / "docs/models/local-fast.md"
    orig = target.read_bytes()
    print(f"\n[3b stale] temporarily modifying {target.name} "
          f"({len(orig)} bytes) to invalidate indexed hashes")
    stale = {}
    try:
        target.write_bytes(orig + b"\n\n<!-- phase B staleness probe -->\n")
        with VramWatch() as w:
            r3b = call("/ask", {"question": "What port does the local-fast service listen on, and "
                                            "what context size does it use by default?"})
        rep["scenarios"]["stale_index"] = r3b
        rep["vram"]["stale_index"] = w.summary()
        b3 = r3b["body"]
        stale = {"outcome": b3.get("outcome"), "reason": b3.get("reason"),
                 "citations": b3.get("citations")}
        print(f"    http={r3b['http']} outcome={b3.get('outcome')}")
        print(f"    reason: {str(b3.get('reason'))[:150]}")
        for c in (b3.get("citations") or []):
            print(f"      {c['source_path']}:{c['start_line']}-{c['end_line']} {c['verification']}")
    finally:
        target.write_bytes(orig)
        print(f"    source restored (sha matches the index again: "
              f"{__import__('hashlib').sha256(target.read_bytes()).hexdigest()[:16]})")
    rep["scenarios"]["stale_index_summary"] = stale

    # ---------------- 4: service unavailable - stop the embedding service
    print("\n[4 unavailable] stopping local-embedding to test structured failure")
    run_start_stop("stop-local-embedding.ps1", timeout=300)
    time.sleep(3)
    with VramWatch() as w:
        r4 = call("/ask", {"question": "What port does local-fast listen on?"}, timeout=120)
    rep["scenarios"]["service_unavailable"] = r4
    rep["vram"]["service_unavailable"] = w.summary()
    print(f"    http={r4['http']} error={(r4['body'].get('error') or {}).get('kind')} "
          f"elapsed={r4['elapsed_s']}s")
    print(f"    message: {str((r4['body'].get('error') or {}).get('message'))[:120]}")

    t0 = time.time()
    run_start_stop("start-local-embedding.ps1", timeout=600)
    time.sleep(3)
    rec_h = call("/health")
    rep["scenarios"]["recovery_after_unavailable"] = {
        "restored_in_s": round(time.time() - t0, 2),
        "health_status": rec_h["body"].get("status"),
        "probing_question": call("/ask", {"question": "What port does local-fast listen on?"})["body"].get("outcome"),
    }
    print(f"    recovered in {rep['scenarios']['recovery_after_unavailable']['restored_in_s']}s, "
          f"health={rec_h['body'].get('status')}, "
          f"probe={rep['scenarios']['recovery_after_unavailable']['probing_question']}")

    # ---------------- 5: limits - oversize question and concurrency
    # (No oversize BODY is sent from here, although an earlier version of this comment said so:
    # MAX_BODY_BYTES was never exercised by this client - ERRATA E24. cli/test_qa_path.py covers it.)
    r5a = call("/ask", {"question": "x" * 1200})
    rep["scenarios"]["question_too_long"] = {"http": r5a["http"],
                                             "kind": (r5a["body"].get("error") or {}).get("kind")}
    print(f"\n[5a too long] http={r5a['http']} "
          f"kind={(r5a['body'].get('error') or {}).get('kind')}")

    # concurrency: fire more requests than the limit and confirm some are rejected with 429
    results: list[dict] = []
    lock = threading.Lock()

    def worker(i: int):
        r = call("/ask", {"question": f"What port does local-fast listen on? (probe {i})"},
                 timeout=300)
        with lock:
            results.append({"i": i, "http": r["http"],
                            "kind": (r["body"].get("error") or {}).get("kind"),
                            "elapsed_s": r["elapsed_s"]})

    with VramWatch() as w:
        ts = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
        for t in ts:
            t.start()
        for t in ts:
            t.join(timeout=400)
    codes = [r["http"] for r in results]
    rep["scenarios"]["concurrency"] = {"requests": len(results), "results": sorted(
        results, key=lambda x: x["i"]), "http_codes": codes,
        "n_429": sum(1 for c in codes if c == 429),
        "n_200": sum(1 for c in codes if c == 200)}
    rep["vram"]["concurrency"] = w.summary()
    rep["limits_verified"] = {
        "max_concurrency_enforced": sum(1 for c in codes if c == 429) > 0,
        "observed_max_simultaneous_200": sum(1 for c in codes if c == 200),
        "question_length_enforced": (r5a["body"].get("error") or {}).get("kind") == "question_too_long",
        "index_override_rejected": (r3a["body"].get("error") or {}).get("kind") == "index_override_rejected",
    }
    print(f"[5b concurrency] codes={codes} n_429={rep['scenarios']['concurrency']['n_429']}")
    print(f"    limits verified: {json.dumps(rep['limits_verified'])}")

    # ---------------- final health + VRAM
    rep["vram"]["end"] = {"free": free_vram()}
    rep["vram_overall"] = {
        "min_across_scenarios": min((v.get("free_min") or 10**9)
                                    for v in rep["vram"].values() if isinstance(v, dict)),
    }
    ih = call("/health")
    rep["inflight_peak"] = (ih["body"] or {}).get("inflight")
    print(f"\nfinal: free={free_vram()} MiB  inflight={rep['inflight_peak']}")
    print(f"vram overall min across scenarios: {rep['vram_overall']['min_across_scenarios']} MiB")

    out.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

