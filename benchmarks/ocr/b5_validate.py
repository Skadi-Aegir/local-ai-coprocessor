"""
B5 - functional equivalence and coexistence.

1. FUNCTIONAL: take 3 images from the FROZEN set, call the service over HTTP, and require the text to
   be IDENTICAL to the in-process B3 result for the same image. Nothing is re-scored; the stored
   benchmark prediction is the reference.

2. COEXISTENCE: with 8111 (local-fast), 8113 (local-embedding) and 8116 (local-ocr) all running, do
   one QA CLI smoke on a question OUTSIDE the frozen sets, then 20 consecutive OCR requests, sampling
   VRAM throughout. The OCR service must not move the VRAM figure beyond noise.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OCR = "http://127.0.0.1:8116"
BENCH = REPO / "benchmarks/ocr/results/OCR-03-bench/bench.json"
EVID = REPO / "benchmarks/ocr/results/OCR-05-validate"
PY = REPO / "services/retrieval/venv/Scripts/python.exe"

IDS = ["syn_en_01", "syn_code_01", "real_nvidia_smi"]


def call(path, payload=None, timeout=300):
    url = OCR + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                headers={"Content-Type": "application/json"},
                                method="POST" if data else "GET")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()), round(time.time() - t0, 3)


def vram_free():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                            "--format=csv,noheader,nounits"], stdout=subprocess.PIPE, timeout=30)
        u, f = o.stdout.decode().strip().split(",")
        return int(u), int(f)
    except Exception:  # noqa: BLE001
        return None, None


def main() -> int:
    if EVID.exists() and any(EVID.iterdir()):
        print(f"REFUSING to overwrite {EVID}")
        return 3
    EVID.mkdir(parents=True, exist_ok=True)
    rep: dict = {}

    man = json.loads((REPO / "benchmarks/ocr/fixtures/manifest.json").read_text(encoding="utf-8"))
    bench = json.load(open(BENCH, encoding="utf-8"))
    by_id = {r["id"]: r for r in bench["tiers"]["mobile"]["images"]}
    entry = {e["id"]: e for e in man["entries"]}

    # ---------------- 1. functional equivalence
    print("=== 1. HTTP result vs the in-process B3 result ===")
    func = []
    for cid in IDS:
        img = REPO / "benchmarks/ocr/fixtures" / entry[cid]["file"]
        b64 = base64.b64encode(img.read_bytes()).decode()
        out, wall = call("/v1/ocr", {"image": b64, "request_id": f"b5-{cid}"})
        ref = by_id[cid]["pred"]
        same = out["text"] == ref
        func.append({"id": cid, "category": entry[cid]["category"],
                     "http_text": out["text"], "b3_inprocess_text": ref,
                     "identical": same, "n_blocks": len(out["blocks"]),
                     "confidence": out["confidence"], "wall_s": wall,
                     "elapsed_s": out["elapsed_s"]})
        print(f"  {cid:<20} blocks={len(out['blocks'])} identical={same} "
              f"elapsed={out['elapsed_s']}s")
        if not same:
            print(f"     http      : {out['text']!r}")
            print(f"     in-process: {ref!r}")
    rep["functional"] = {"images": func, "all_identical": all(f["identical"] for f in func),
                         "n": len(func)}

    # ---------------- 2. coexistence
    print("\n=== 2. coexistence: 8111 + 8113 + 8116 ===")
    u0, f0 = vram_free()
    rep["coexistence"] = {"vram_used_before_mb": u0, "vram_free_before_mb": f0}
    print(f"  VRAM before: used {u0} MiB / free {f0} MiB")

    # VRAM samples in a thread while the work runs
    import threading
    samples: list[tuple] = []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            u, f = vram_free()
            if f is not None:
                samples.append((u, f))
            time.sleep(0.2)

    th = threading.Thread(target=watch, daemon=True)
    th.start()

    # QA smoke on a question OUTSIDE both frozen sets
    q = "What does the escalation policy say about who owns the final review of a change?"
    log = REPO / "logs/ocr-b5-qasmoke.txt"
    with open(log, "wb") as fh:
        rc = subprocess.run([str(PY), "cli/ask.py", q, "--json"], cwd=str(REPO),
                            stdout=fh, stderr=subprocess.STDOUT, timeout=900).returncode
    qa_txt = log.read_text(encoding="utf-8", errors="replace")
    qa_rec = None
    for line in qa_txt.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                qa_rec = json.loads(line)
            except Exception:  # noqa: BLE001
                pass
    rep["coexistence"]["qa_smoke"] = {
        "question": q, "exit": rc,
        "refused": (qa_rec or {}).get("refused"),
        "answer_head": ((qa_rec or {}).get("answer") or "")[:200],
        "citations": len((qa_rec or {}).get("citations") or []),
        "finish_reason": (qa_rec or {}).get("finish_reason"),
    }
    print(f"  QA smoke exit={rc} refused={(qa_rec or {}).get('refused')} "
          f"cites={len((qa_rec or {}).get('citations') or [])}")

    # 20 consecutive OCR requests
    lat = []
    ok = 0
    for i in range(20):
        img = REPO / "benchmarks/ocr/fixtures" / entry[IDS[i % len(IDS)]]["file"]
        b64 = base64.b64encode(img.read_bytes()).decode()
        try:
            out, wall = call("/v1/ocr", {"image": b64, "request_id": f"coex-{i}"})
            ok += 1
            lat.append(out["elapsed_s"])
        except Exception as e:  # noqa: BLE001
            print(f"    request {i} failed: {type(e).__name__}: {e}")
    stop.set()
    th.join(timeout=5)

    frees = [f for _, f in samples]
    rep["coexistence"].update({
        "ocr_requests": 20, "ocr_ok": ok,
        "ocr_latency_min_s": min(lat) if lat else None,
        "ocr_latency_max_s": max(lat) if lat else None,
        "vram_samples": len(frees),
        "vram_free_min_mb": min(frees) if frees else None,
        "vram_free_max_mb": max(frees) if frees else None,
        "margin_mb": 2048,
        "margin_held": (min(frees) >= 2048) if frees else None,
    })
    u1, f1 = vram_free()
    rep["coexistence"]["vram_used_after_mb"] = u1
    rep["coexistence"]["vram_free_after_mb"] = f1
    rep["coexistence"]["vram_used_delta_mb"] = (u1 - u0) if (u1 is not None and u0 is not None) else None
    print(f"  20 OCR requests: {ok} ok, latency {min(lat) if lat else '-'}-"
          f"{max(lat) if lat else '-'}s")
    print(f"  VRAM free min during {min(frees) if frees else '-'} MiB "
          f"(margin held: {rep['coexistence']['margin_held']})")
    print(f"  VRAM used delta across the whole test: {rep['coexistence']['vram_used_delta_mb']} MiB")

    # commit memory
    import psutil
    rep["commit_free_mb"] = int(psutil.virtual_memory().available / 1024 / 1024)
    print(f"  commit free now: {rep['commit_free_mb']} MB")

    (EVID / "b5_validation.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(f"\nwrote {EVID / 'b5_validation.json'}")
    return 0 if rep["functional"]["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
