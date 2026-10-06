"""
ctx_sweep.py - measure the deployed server at several context sizes.

llama-bench has no context flag, so the only way to characterise the context-length
behaviour of the ACTUAL deployment is to start llama-server at each size and drive a
real request through it. For each context this records:
  - VRAM from llama.cpp's own fit_memory_breakdown (device row)
  - prompt-processing and generation tok/s from the server timings
  - time to first token, from streaming

Usage:
  python ctx_sweep.py --contexts 4096,8192,16384 --out results/ctx_sweep.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import Client  # noqa: E402

SERVER_LOG = (Path(__file__).resolve().parents[2] / r"logs\local-fast.server.jsonl")
START = str(Path(__file__).resolve().parents[2] / r"scripts\start-local-fast.ps1")
STOP = str(Path(__file__).resolve().parents[2] / r"scripts\stop-local-fast.ps1")

# A prompt long enough that prompt-processing dominates the measurement.
PROMPT_TODO = (
    "Summarise the following engineering notes into exactly 5 bullet points, "
    "using only information present in the text.\n\n"
)
BODY = (
    "The inference scheduler was rebuilt over six weeks. The old design pushed every "
    "request into a single Redis queue consumed by eight polling workers, and model calls "
    "were synchronous. Peak queueing delay degraded from 200 ms to 6.8 seconds, and a "
    "worker blocked on a slow request could not serve anything else. The new design has "
    "three tiers: an ingress layer exposing a unified endpoint with per-tenant rate "
    "limiting at 50 requests per second, a scheduling layer with interactive, batch and "
    "backfill priority classes, and an execution layer with an 8 second hard timeout that "
    "releases the slot immediately. We abandoned a self-built vector index for pgvector, "
    "accepting retrieval latency rising from 12 ms to 35 ms in exchange for reusing the "
    "existing Postgres backup and monitoring. Caching is exact-match only, keyed on tenant, "
    "model version and a normalised prompt hash, giving a 23 percent hit rate; semantic "
    "caching was rejected because it once returned a refund flow to a user asking about "
    "shipping. During load testing GPU memory filled gradually until a batch job was found "
    "not releasing its KV cache; capping context at 16K and clearing on the timeout path "
    "stabilised it at 9.2 GB over 72 hours. Structured logging produced 240 GB per day and "
    "filled the disk in two days, so sampling was cut to 10 percent with full retention for "
    "failures, reducing volume to 24 GB per day. Canary has run two weeks with error rate "
    "down from 0.8 percent to 0.15 percent. One problem remains: above 200 tenants the "
    "scheduling decision itself takes 40 ms. Rollback is a switch that returns traffic to "
    "the old queue in about 90 seconds without a restart. Monthly cloud spend rose from "
    "4200 to 5100 yuan, but cost per request fell 8 percent."
) * 2


def run_ps(script: str, extra: list[str] | None = None, timeout: int = 420) -> tuple[int, str]:
    cmd = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script]
    if extra:
        cmd += extra
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def read_breakdown() -> dict | None:
    """Parse the last fit_memory_breakdown device row from the server log."""
    if not SERVER_LOG.exists():
        return None
    last = None
    for line in SERVER_LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        if "fit_memory_breakdown" in line:
            last = line
    if not last:
        return None
    m = re.search(r'\{"type":"fit_memory_breakdown".*$', last)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    rows = obj.get("data", {}).get("rows", [])
    dev = next((r for r in rows if r.get("kind") == "device"), None)
    return dev


def measure(client: Client, ctx: int) -> dict:
    # warm up so the first request does not carry graph/alloc overhead
    client.chat([{"role": "user", "content": "hi"}], max_tokens=4,
                temperature=0.0, enable_thinking=False)

    r = client.chat(
        [{"role": "user", "content": PROMPT_TODO + BODY}],
        max_tokens=400, temperature=0.0, enable_thinking=False,
    )
    return {
        "ctx": ctx,
        "ok": r.ok,
        "error": r.error,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "prompt_tps": round(r.prompt_tps, 1),
        "gen_tps": round(r.gen_tps, 2),
        "ttft_s": round(r.ttft_s, 3) if r.ttft_s else None,
        "total_s": round(r.total_s, 3),
        "breakdown": None,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="4096,8192,16384")
    ap.add_argument("--out", default=str(HERE / "results" / "ctx_sweep.json"))
    ap.add_argument("--base-url", default="http://127.0.0.1:8111")
    args = ap.parse_args()

    results = []
    for ctx in [int(c) for c in args.contexts.split(",")]:
        print(f"\n===== context {ctx} =====", flush=True)
        code, out = run_ps(STOP)
        print(f"  stop   exit={code}")
        time.sleep(2)

        code, out = run_ps(START, ["-Context", str(ctx), "-Force"])
        if code != 0:
            print(f"  start FAILED exit={code}")
            print("\n".join("    " + l for l in out.splitlines()[-8:]))
            results.append({"ctx": ctx, "ok": False, "error": f"start exit {code}"})
            continue
        print(f"  start  exit={code}")

        dev = read_breakdown()
        client = Client(args.base_url)
        m = measure(client, ctx)
        m["breakdown"] = dev
        results.append(m)

        if dev:
            print(f"  VRAM   model={dev.get('model')} context={dev.get('context')} "
                  f"compute={dev.get('compute')} self={dev.get('self')} MiB")
        print(f"  prompt {m['prompt_tps']} t/s | gen {m['gen_tps']} t/s | "
              f"ttft {m['ttft_s']}s | prompt_tok {m['prompt_tokens']}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}")

    # leave the server running at the default context
    print("\nrestarting at default 8192 ...")
    run_ps(STOP)
    time.sleep(2)
    code, _ = run_ps(START)
    print(f"final start exit={code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
