"""
p1_11_embed_retention.py - OPEN_ISSUES P1-11: does a bulk document embed leave GPU memory behind
in the embedding process, and does Embedder._release_cache stop it?

ISOLATED. The embedding model is loaded IN THIS PROCESS; the services on 8111/8113/8114/8115 are
never touched, and the script refuses to run while any of them is listening or while free VRAM is
short. Its own allocator is capped so the card keeps the 2048 MiB margin even if a batch is larger
than expected: an over-large batch fails HERE with an out-of-memory error, it never squeezes
another process.

The cap is computed from nvidia-smi, NOT from cudaMemGetInfo. Measured in run 2026-10-05-rs16 on
this Windows (WDDM) machine: with this process holding only a CUDA context, cudaMemGetInfo put
used memory at 1232.6 MiB while nvidia-smi reported 3438 MiB. That run derived its cap from
cudaMemGetInfo, so the cap (8690 MiB) did not actually protect the margin; the margin held on its
own (lowest sampled free 2269 MiB). The same figure is what the service's /stats `vram_mb`
reports, which is therefore neither the device-wide figure nor the service's own footprint.

WORKLOAD. The exact Phase A corpus: the 48 manifest files read from the manifest's git_commit and
checked against the manifest's content hashes, chunked as the builder chunks them (520 chunks),
sent in requests of a chosen size through encode_documents - the path the service's
/v1/embeddings takes for documents. Service defaults otherwise (batch_size 32, max_seq_length
8192, float32). (Run 2026-10-05-rs16 read the files from the working tree, which already carried
this fix to embedder.py and server.py: 522 chunks.)

ARMS, one process, allocator emptied and peak reset between them:
  release OFF   release_cache_mb = 0     the behaviour the service had in Phase A and B
  release ON    release_cache_mb = 256   the fix
Per request: reserved / allocated after it, its peak, device free (nvidia-smi). Per arm: the
footprint left behind, the time taken, and the vectors - compared across arms, since the release
must not change what is computed. Device-wide used memory is read both from cudaMemGetInfo and
nvidia-smi, to settle what the service's `/stats vram_mb` actually measures.

Usage:
  python p1_11_embed_retention.py --run-id <id> [--request-size 16]
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "retrieval"))

MARGIN_MB = 2048
SAFETY_MB = 256
MIN_FREE_TO_START_MB = 7000
SERVICE_PORTS = (8111, 8113, 8114, 8115)
MODEL_PATH = str(Path(__file__).resolve().parents[2] / r"models\embedding\Qwen3-Embedding-0.6B")


def listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def smi() -> dict | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
                              "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=15).stdout
        total, used, free = (int(x) for x in out.strip().splitlines()[0].split(","))
        return {"total": total, "used": used, "free": free}
    except Exception:  # noqa: BLE001
        return None


class FreeSampler:
    """nvidia-smi in a loop on a thread: the device-wide free minimum DURING requests, not after."""

    def __init__(self):
        self.vals: list[int] = []
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            s = smi()
            if s:
                self.vals.append(s["free"])

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.t.join(timeout=20)
        return False


def corpus_texts() -> tuple[list[str], dict]:
    """The indexed bytes exactly: each file from the manifest's commit, checked against its hash."""
    from chunker import chunk_file
    from provenance import HASH_SCHEME, content_hash
    m = json.loads((REPO / "vector/qa/current/manifest.json").read_text(encoding="utf-8"))
    if m.get("hash_scheme") != HASH_SCHEME:
        raise SystemExit("the manifest predates sha256-lf-v1; run a_migrate_manifest.py first")
    commit = m["git_commit"]
    texts = []
    for rel in m["included_files"]:
        raw = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{rel}"],
                             capture_output=True, check=True).stdout
        if content_hash(raw) != m["file_hashes"][rel]:
            raise SystemExit(f"{rel} at {commit[:12]} is not the indexed content")
        t = raw.replace(b"\r\n", b"\n").decode("utf-8", errors="replace")
        texts += [c.content for c in chunk_file(t, rel, "local-ai", strategy="structured")]
    return texts, {"source": f"git {commit[:12]}, every file checked against the manifest hash",
                   "files": len(m["included_files"]), "manifest_chunks": m["n_chunks"]}


def run_arm(emb, torch, texts: list[str], request_size: int, release_mb: int) -> tuple[dict, list]:
    mib = 2**20
    emb.cfg.release_cache_mb = release_mb
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = emb.memory_stats()
    rows, vecs, oom = [], [], None
    t_arm = time.time()
    with FreeSampler() as sampler:
        for i in range(0, len(texts), request_size):
            batch = texts[i:i + request_size]
            torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            try:
                vecs.append(emb.encode_documents(batch))
            except torch.OutOfMemoryError as e:
                oom = {"request": i // request_size, "error": str(e)[:300]}
                torch.cuda.empty_cache()
                break
            rows.append({"request": i // request_size, "n": len(batch),
                         "max_chars": max(len(t) for t in batch),
                         "seconds": round(time.time() - t0, 3),
                         "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / mib, 1),
                         "reserved_after_mb": round(torch.cuda.memory_reserved() / mib, 1),
                         "allocated_after_mb": round(torch.cuda.memory_allocated() / mib, 1)})
    seconds = round(time.time() - t_arm, 2)
    end = emb.memory_stats()
    weights = start["torch_allocated_mb"]
    arm = {
        "release_cache_mb": release_mb, "request_size": request_size,
        "requests_completed": len(rows), "oom": oom, "seconds": seconds,
        "start": start, "end": end, "end_nvidia_smi": smi(),
        "peak_reserved_mb": max((r["peak_reserved_mb"] for r in rows), default=None),
        "peak_above_weights_mb": round(max((r["peak_reserved_mb"] for r in rows),
                                           default=weights) - weights, 1),
        "retained_above_weights_mb": round(end["torch_reserved_mb"] - end["torch_allocated_mb"], 1),
        "device_free_samples": len(sampler.vals),
        "device_free_min_mb": min(sampler.vals) if sampler.vals else None,
        "margin_held_in_samples": bool(sampler.vals) and min(sampler.vals) >= MARGIN_MB,
        "per_request": rows,
    }
    return arm, vecs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--request-size", type=int, default=16, help="texts per request (builder: 16)")
    ap.add_argument("--also-request-size", type=int, nargs="*", default=[],
                    help="further arms, release on, at these request sizes; their vectors are "
                         "compared with the release-on arm's - does the request size change the "
                         "index? Use --request-size 1 for the unpadded reference")
    args = ap.parse_args()
    out = HERE / "results" / f"{args.run_id}.json"
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3

    busy = [p for p in SERVICE_PORTS if listening(p)]
    pre = smi()
    if busy:
        print(f"REFUSING: project services are listening on {busy}; this experiment must not "
              f"compete with them")
        return 2
    if not pre or pre["free"] < MIN_FREE_TO_START_MB:
        print(f"REFUSING: free VRAM {pre and pre['free']} MiB < {MIN_FREE_TO_START_MB} MiB")
        return 2

    import numpy as np
    import torch
    from embedder import Embedder, EmbedderConfig

    mib = 2**20
    torch.cuda.init()
    torch.zeros(1, device="cuda")                       # create the context
    free_b, total_b = torch.cuda.mem_get_info()
    ctx_smi = smi()
    what_vram_mb_measures = {
        "torch_device_used_mb": round((total_b - free_b) / mib, 1),
        "nvidia_smi_used_mb": ctx_smi and ctx_smi["used"],
        "this_process_allocator_reserved_mb": round(torch.cuda.memory_reserved() / mib, 1),
        "note": ("cudaMemGetInfo-based used memory (what /stats vram_mb reports) compared with "
                 "nvidia-smi's device-wide figure, taken when this process holds only its context"),
    }
    # Device-wide free memory AFTER the context exists, from nvidia-smi: the allocator's reserved
    # memory shows up one-for-one there, so capping it at this figure minus the margin keeps the
    # card's free memory at or above the margin (other processes permitting).
    cap_mb = ctx_smi["free"] - MARGIN_MB - SAFETY_MB
    torch.cuda.set_per_process_memory_fraction(cap_mb * mib / total_b)
    print(f"free after context {ctx_smi['free']} MiB (nvidia-smi); allocator capped at {cap_mb} "
          f"MiB so the card keeps >= {MARGIN_MB + SAFETY_MB} MiB free")
    print(f"what /stats vram_mb measures: {what_vram_mb_measures}")

    texts, corpus_info = corpus_texts()
    print(f"corpus: {len(texts)} chunks ({corpus_info['source']}), "
          f"longest {max(len(t) for t in texts)} chars")
    emb = Embedder(EmbedderConfig(model_path=MODEL_PATH, batch_size=32, max_seq_length=8192,
                                  release_cache_mb=0))
    loaded = emb.memory_stats()
    print(f"model loaded: {loaded}")

    report = {"run_id": args.run_id, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                                                  time.gmtime()),
              "preconditions": {"services_listening": busy, "nvidia_smi_before": pre,
                                "allocator_cap_mb": cap_mb, "margin_mb": MARGIN_MB},
              "what_vram_mb_measures": what_vram_mb_measures,
              "corpus": {"chunks": len(texts), "longest_chars": max(len(t) for t in texts),
                         **corpus_info},
              "model_loaded": loaded, "arms": {}}
    vec_by_arm = {}
    try:
        for name, release in (("release_off", 0), ("release_on", 256)):
            arm, vecs = run_arm(emb, torch, texts, args.request_size, release)
            report["arms"][name] = arm
            vec_by_arm[name] = np.concatenate(vecs) if vecs and not arm["oom"] else None
            print(f"[{name}] completed {arm['requests_completed']} requests in {arm['seconds']}s; "
                  f"peak above the weights {arm['peak_above_weights_mb']} MiB; left behind above "
                  f"the weights {arm['retained_above_weights_mb']} MiB; device free min "
                  f"{arm['device_free_min_mb']} MiB over {arm['device_free_samples']} samples; "
                  f"oom={arm['oom'] is not None}")
        a, b = vec_by_arm.get("release_off"), vec_by_arm.get("release_on")
        if a is not None and b is not None and a.shape == b.shape:
            report["vectors_across_arms"] = {
                "max_abs_diff": float(np.max(np.abs(a - b))),
                "bit_identical": bool(np.array_equal(a, b))}
            print(f"vectors across arms: {report['vectors_across_arms']}")
        report["request_size_effect"] = []
        for size in args.also_request_size:
            arm, vecs = run_arm(emb, torch, texts, size, 256)
            report["arms"][f"release_on_rs{size}"] = arm
            c = np.concatenate(vecs) if vecs and not arm["oom"] else None
            print(f"[release_on_rs{size}] {arm['seconds']}s; peak above the weights "
                  f"{arm['peak_above_weights_mb']} MiB; left behind "
                  f"{arm['retained_above_weights_mb']} MiB; device free min "
                  f"{arm['device_free_min_mb']} MiB")
            if b is not None and c is not None and b.shape == c.shape:
                cos = np.sum(b * c, axis=1)
                # does the request size change which chunk a query would retrieve? Use every
                # chunk as a query against both vector sets and compare the nearest neighbours.
                sims_b, sims_c = b @ b.T, c @ c.T
                np.fill_diagonal(sims_b, -1)
                np.fill_diagonal(sims_c, -1)
                effect = {
                    "compared": [args.request_size, size],
                    "max_abs_diff": float(np.max(np.abs(b - c))),
                    "min_row_cosine": float(np.min(cos)),
                    "mean_row_cosine": float(np.mean(cos)),
                    "bit_identical": bool(np.array_equal(b, c)),
                    "nearest_neighbour_agreement": float(np.mean(
                        np.argmax(sims_b, axis=1) == np.argmax(sims_c, axis=1)))}
                report["request_size_effect"].append(effect)
                print(f"request size {args.request_size} vs {size}: {effect}")
    finally:
        emb.close()
        report["after_close"] = {"nvidia_smi": smi()}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
