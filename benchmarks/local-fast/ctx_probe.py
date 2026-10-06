"""One-shot context probe: send a fixed long prompt and report server-side timings."""
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Client  # noqa: E402

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

PROMPT = ("Summarise the following engineering notes into exactly 5 bullet points, "
          "using only information present in the text.\n\n" + BODY)


def main() -> int:
    ctx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    base = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8111"
    c = Client(base, timeout=240)

    # warm-up: avoids charging graph/alloc overhead to the measurement
    c.chat([{"role": "user", "content": "hi"}], max_tokens=4, temperature=0.0,
           enable_thinking=False)

    r = c.chat([{"role": "user", "content": PROMPT}], max_tokens=400,
               temperature=0.0, enable_thinking=False)

    out = {
        "ctx": ctx, "ok": r.ok, "error": r.error,
        "prompt_tokens": r.prompt_tokens,
        "completion_tokens": r.completion_tokens,
        "prompt_tps": round(r.prompt_tps, 1),
        "gen_tps": round(r.gen_tps, 2),
        "ttft_s": round(r.ttft_s, 3) if r.ttft_s else None,
        "total_s": round(r.total_s, 3),
    }
    print(json.dumps(out, ensure_ascii=False))
    Path(str(Path(__file__).resolve().parents[2] / r"benchmarks\local-fast\results\ctx_probe_%d.json") % ctx).write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if r.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
