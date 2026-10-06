"""
p2_residency.py - P55R-02 item 2: minimal-residency controlled experiment.

Compares residency strategies on a FIXED workload, with EVERY operation admitted through the
same scheduler instance using request ids and real declared dependencies.

  A  three models co-resident
  B  local-fast + embedding resident, reranker loaded ON DEMAND
  C  only tested if BOTH A and B fail (local-fast resident, retrieval models staged)

Measurement discipline required by the task:
  * the 2048 MiB margin is NEVER lowered
  * VRAM is sampled CONTINUOUSLY through the run, and the RAG stage is measured through
    rerank AND generation - not just idle-before-load, because B can return to three-model
    residency once the reranker loads
  * health-passed time and first-successful-inference time are reported SEPARATELY
  * a strategy that cannot complete the supported workload is a FAIL
  * "no margin breach" is only ever reported as "not observed within the sampling window"

Usage:
  python p2_residency.py --mode A --out results/<run>/residency_A.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "services" / "gpu_scheduler"))
sys.path.insert(0, str(REPO / "scripts"))

from gpu_scheduler import REGISTRY, GpuScheduler, service_healthy  # noqa: E402
from p3_atomic import atomic_write_json  # noqa: E402

FAST = "http://127.0.0.1:8111"
EMBED = "http://127.0.0.1:8113"
RERANK = "http://127.0.0.1:8114"
MARGIN = 2048
RAG_Q = 'What context size does the local-fast worker use by default?'
OUTER_TIMEOUT = 900.0


def post(url, payload, timeout=600):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def vram() -> tuple[int, int] | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20)
        used, free = out.stdout.strip().splitlines()[0].split(",")
        return int(used.strip()), int(free.strip())
    except Exception:
        return None


class Sampler:
    """Continuous VRAM sampling for the whole run."""

    def __init__(self, interval: float = 0.15):
        self.interval = interval
        self.samples: list[dict] = []
        self.stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.is_set():
            v = vram()
            if v:
                self.samples.append({"t": round(time.time(), 2), "used": v[0], "free": v[1]})
            time.sleep(self.interval)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *e):
        self.stop.set()
        self.t.join(timeout=10)
        return False

    def between(self, t0: float, t1: float | None = None) -> dict:
        t1 = t1 or time.time()
        vals = [s for s in self.samples if t0 <= s["t"] <= t1]
        if not vals:
            return {"n": 0}
        used = [s["used"] for s in vals]
        free = [s["free"] for s in vals]
        return {"n": len(vals), "used_min": min(used), "used_max": max(used),
                "free_min": min(free), "free_max": max(free),
                "margin_breached": min(free) < MARGIN}


STOP_SCRIPT = {"local-fast": "stop-local-fast.ps1",
               "local-embedding": "stop-local-embedding.ps1",
               "local-reranker": "stop-local-reranker.ps1"}
START_SCRIPT = {"local-fast": "start-local-fast.ps1",
                "local-embedding": "start-local-embedding.ps1",
                "local-reranker": "start-local-reranker.ps1"}


def _shell() -> str:
    """
    Resolve a PowerShell EXECUTABLE that this Python process can actually launch.

    `pwsh` is not on PATH for the venv python here, so spawning it raised FileNotFoundError
    (WinError 2) - which initially looked like a sandbox permission problem and was not one.
    `powershell.exe` (5.1) resolves and runs fine. The resolved path is recorded so the
    environment assumption is visible in the result file rather than implied.
    """
    for exe in ("pwsh", "pwsh.exe", "powershell.exe"):
        try:
            subprocess.run([exe, "-NoProfile", "-Command", "exit 0"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
            return exe
        except FileNotFoundError:
            continue
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("no usable PowerShell executable found (tried pwsh, powershell.exe)")


SHELL = None


def _run_script(script: Path, timeout: int) -> int:
    """
    Run a service start/stop script and return its exit code.

    Output goes to a TEMPORARY FILE, never to a pipe. Measured reason: with
    `capture_output=True` the child powershell blocks once it has written more than the pipe
    buffer, while `subprocess.run` blocks waiting for the child to exit - a classic pipe
    deadlock. It presented as "mode C hangs" after the embedding service came up and the
    reranker load never completed: a powershell running start-local-reranker.ps1 was found
    alive for 700 s with the parent blocked on it. The transcript is still preserved - it is
    read back from the file - so nothing is lost by not using a pipe.
    """
    global SHELL
    if SHELL is None:
        SHELL = _shell()
    log = REPO / "logs" / f"p2-svc-{script.stem}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "wb") as fh:
        r = subprocess.run([SHELL, "-NoLogo", "-NoProfile", "-NonInteractive",
                            "-ExecutionPolicy", "Bypass", "-File", str(script)],
                           stdout=fh, stderr=subprocess.STDOUT, timeout=timeout)
    return r.returncode


def stop_service(name: str) -> bool:
    """
    Stop a service via its own script.

    The script name is looked up EXPLICITLY rather than derived as
    f"stop-local-{name.split('-')[-1]}.ps1": that derivation happens to give the right
    filename for these three aliases, but it is a coincidence of naming and it silently
    produced the wrong file for a name it did not fit. A name with no mapping is refused
    rather than guessed.
    """
    script = REPO / "scripts" / STOP_SCRIPT.get(name, "")
    if not script.is_file():
        raise ValueError(f"no stop script mapped for {name!r}")
    _run_script(script, 300)
    return True


def start_service(name: str, wait_s: float = 180) -> bool:
    script = REPO / "scripts" / START_SCRIPT.get(name, "")
    if not script.is_file():
        raise ValueError(f"no start script mapped for {name!r}")
    _run_script(script, int(wait_s) + 120)
    return service_healthy(REGISTRY[name].port)


# ---------------------------------------------------------------- the fixed workload
def w_interactive() -> dict:
    t0 = time.time()
    r = post(f"{FAST}/v1/chat/completions", {
        "model": "local-fast",
        "messages": [{"role": "user", "content": "Name two indexing strategies."}],
        "max_tokens": 32, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False}})
    return {"ok": bool(r.get("choices")), "s": round(time.time() - t0, 3),
            "gen_tps": round(r.get("timings", {}).get("predicted_per_second", 0), 1)}


def w_embed_batch() -> dict:
    t0 = time.time()
    r = post(f"{EMBED}/v1/embeddings",
             {"model": "local-embedding", "input": [f"chunk {i}" for i in range(32)]})
    n = len(r.get("data") or [])
    return {"ok": n == 32, "s": round(time.time() - t0, 3), "vectors": n}


def w_full_rag() -> dict:
    """
    A complete RAG stage: query embedding -> dense retrieval -> RERANK -> pack -> GENERATE.

    The rerank step calls the real reranker SERVICE on 8114 rather than a local model, because
    the point of the experiment is whether the residency strategy can carry the rerank stage -
    in particular whether B can load the reranker on demand and still finish generation.
    `/v1/retrieve` does not apply the reranker itself (result has no reranker_used field), so
    the stage is assembled here from the services that actually exist.
    """
    t0 = time.time()
    r = post(f"{EMBED}/v1/retrieve",
             {"query": RAG_Q, "top_k": 10})
    retrieve_s = time.time() - t0
    hits = r.get("results") or []

    rerank_s = None
    used_reranker = False
    if hits and service_healthy(REGISTRY["local-reranker"].port):
        docs = [h.get("content", "") for h in hits]
        try:
            t1 = time.time()
            rr = post(f"{RERANK}/v1/rerank",
                      {"query": RAG_Q, "documents": docs, "top_n": 5,
                       "instruction_type": "general", "use_cache": False})
            rerank_s = time.time() - t1
            order = [x.get("index") for x in (rr.get("results") or [])]
            hits = [hits[i] for i in order if isinstance(i, int) and i < len(hits)] or hits
            hits = hits[:5]
            used_reranker = True
        except Exception as e:  # noqa: BLE001
            rerank_s = f"failed: {type(e).__name__}: {e}"

    # pack
    pack = "\n\n".join(
        f"SOURCE {i+1} ({h.get('source_path')} lines {h.get('start_line')}-{h.get('end_line')})\n"
        f"{h.get('content','')[:2000]}"
        for i, h in enumerate(hits))

    t2 = time.time()
    g = post(f"{FAST}/v1/chat/completions", {
        "model": "local-fast",
        "messages": [{"role": "user", "content": f"{pack}\n\nQUESTION: {RAG_Q}"}],
        "max_tokens": 64, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False}})
    gen_s = time.time() - t2
    text = (g.get("choices") or [{}])[0].get("message", {}).get("content", "")
    return {"ok": bool(text) and bool(hits), "s": round(time.time() - t0, 3),
            "retrieve_s": round(retrieve_s, 3),
            "rerank_s": rerank_s if isinstance(rerank_s, str) else (
                round(rerank_s, 3) if rerank_s else None),
            "generate_s": round(gen_s, 3),
            "n_sources": len(hits), "pack_chars": len(pack),
            "used_reranker": used_reranker,
            "answer_has_8192": "8192" in text,
            "answer_head": text.strip()[:160]}


def run_through_scheduler(s: GpuScheduler, job_class: str, rid: str, requires: list[str],
                          fn, timeout_s: float = 240.0) -> dict:
    """Reserve -> (load declared deps) -> execute -> release. Refusal means NO call is made."""
    t_queue0 = time.time()
    d = s.acquire(job_class, timeout_s=timeout_s, requires=requires, request_id=rid)
    queue_s = time.time() - t_queue0
    out: dict = {"request_id": rid, "job_class": job_class, "requires": requires,
                 "allowed": d.allowed, "queue_s": round(queue_s, 3),
                 "required_mb": d.required_mb, "reason": d.reason}
    if not d.allowed:
        out["status"] = "refused"
        return out

    load_s, health_t = 0.0, None
    try:
        # load any declared dependency that is not up, and time the HEALTH gate separately
        for name in requires:
            if name in REGISTRY and not service_healthy(REGISTRY[name].port):
                t0 = time.time()
                ok = start_service(name)
                health_t = round(time.time() - t0, 3)
                load_s += health_t
                out.setdefault("load", {})[name] = {"healthy": ok, "seconds": health_t}
                if not ok:
                    out["status"] = "failed"
                    out["error"] = f"dependency {name} did not become healthy"
                    return out
        t_exec = time.time()
        res = fn()
        out["exec_s"] = round(time.time() - t_exec, 3)
        out["load_s"] = round(load_s, 3)
        out["health_gate_s"] = health_t
        out["first_inference_s"] = round(queue_s + load_s, 3)
        out.update({k: v for k, v in res.items() if k != "s"})
        out["status"] = "completed" if res.get("ok") else "failed"
    except Exception as e:  # noqa: BLE001
        out["status"] = "failed"
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        s.release(job_class, d.reservation_id)
        out["released"] = s.reserved_mb() == 0
    return out


def measure_health_vs_inference(service: str, port: int, probe) -> dict:
    """
    Health-passed time and first-successful-inference time, measured SEPARATELY.
    A service answering /health is not necessarily ready to serve inference.
    """
    t0 = time.time()
    healthy_at = None
    while time.time() - t0 < 180:
        if service_healthy(port):
            healthy_at = time.time() - t0
            break
        time.sleep(0.1)
    first_inf = None
    if healthy_at is not None:
        t1 = time.time()
        for _ in range(30):
            try:
                probe()
                first_inf = time.time() - t1
                break
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
    return {"health_gate_s": round(healthy_at, 3) if healthy_at is not None else None,
            "first_inference_s": round(first_inf, 3) if first_inf is not None else None,
            "note": "health passing is NOT used as a proxy for inference readiness"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["A", "B", "C"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=0,
                    help="0 = screening round only; the driver adds rounds only if it passes")
    args = ap.parse_args()
    out = Path(args.out)
    report: dict = {"mode": args.mode, "margin_mb": MARGIN, "rounds": [], "vram": {}}

    # ---------------- provenance, so the result file is self-describing
    #
    # P55R-02's mode B result was LOST because the run crashed before the summary was written
    # and the log truncated every refusal reason. This block records what the run needs in order
    # to be interpretable on its own: the commit, the config paths, and the service state.
    def _git(*a: str) -> str:
        try:
            return subprocess.run(["git", *a], cwd=str(REPO), capture_output=True, text=True,
                                  timeout=30).stdout.strip()
        except Exception:  # noqa: BLE001
            return "unavailable"

    report["provenance"] = {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "margin_mb": MARGIN,
        "registry": {n: {"port": s.port, "vram_mb": s.vram_mb,
                         "peak_extra_mb": s.peak_extra_mb, "essential": s.essential,
                         "load_s": s.load_s}
                     for n, s in REGISTRY.items()},
        "configs": {n: str(REPO / "config" / f"{n}.yaml") for n in REGISTRY},
        "outer_timeout_s": OUTER_TIMEOUT,
    }
    # write the provenance immediately so it survives a later crash
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, report)

    # ---------------- establish the requested residency
    print(f"=== establishing residency for mode {args.mode} ===", flush=True)
    if args.mode == "A":
        for n in ("local-embedding", "local-reranker"):
            if not service_healthy(REGISTRY[n].port):
                print(f"  starting {n}", flush=True)
                start_service(n)
    elif args.mode == "B":
        # B = fast + embedding resident, reranker ON DEMAND
        if not service_healthy(REGISTRY["local-fast"].port):
            print("  starting local-fast", flush=True)
            start_service("local-fast")
        if not service_healthy(REGISTRY["local-embedding"].port):
            print("  starting local-embedding (resident in B)", flush=True)
            start_service("local-embedding")
        if service_healthy(REGISTRY["local-reranker"].port):
            print("  stopping local-reranker (on demand in B)", flush=True)
            stop_service("local-reranker")
    elif args.mode == "C":
        # C = fast resident only, both retrieval models staged
        if not service_healthy(REGISTRY["local-fast"].port):
            print("  starting local-fast", flush=True)
            start_service("local-fast")
        for n in ("local-embedding", "local-reranker"):
            if service_healthy(REGISTRY[n].port):
                print(f"  stopping {n} (staged in C)", flush=True)
                stop_service(n)
    time.sleep(3)
    resident = [n for n in REGISTRY if service_healthy(REGISTRY[n].port)]
    v_cold = vram()
    print(f"  resident={resident}  vram={v_cold}", flush=True)
    report["vram"]["cold_start"] = v_cold
    report["resident_at_start"] = resident

    # ---------------- warm the resident models
    print("\n=== warm-up ===", flush=True)
    if service_healthy(REGISTRY["local-fast"].port):
        try:
            w_interactive()
        except Exception as e:  # noqa: BLE001
            print(f"  warm-up generation failed: {e}", flush=True)
    if service_healthy(REGISTRY["local-embedding"].port):
        try:
            w_embed_batch()
        except Exception as e:  # noqa: BLE001
            print(f"  warm-up embedding failed: {e}", flush=True)
    report["vram"]["after_warmup"] = vram()
    print(f"  vram after warm-up = {report['vram']['after_warmup']}", flush=True)

    # ---------------- the run
    max_rounds = args.rounds if args.rounds > 0 else 1
    s = GpuScheduler(strategy="conditional", verbose=False, dry_run=False,
                     max_background_units=2)
    with Sampler() as smp:
        for rnd in range(max_rounds):
            print(f"\n=== round {rnd+1}/{max_rounds} ===", flush=True)
            t_round = time.time()
            results: list[dict] = []

            # job 1: interactive generation
            results.append(run_through_scheduler(
                s, "interactive", f"r{rnd}-gen", ["local-fast"], w_interactive))

            # job 2: a REAL embedding background task
            results.append(run_through_scheduler(
                s, "background", f"r{rnd}-embed", ["local-embedding"], w_embed_batch))

            # job 3: full RAG, ending in generation
            results.append(run_through_scheduler(
                s, "background", f"r{rnd}-rag",
                ["local-embedding", "local-reranker", "local-fast"], w_full_rag))

            # job 4: interactive arriving DURING a background task
            bg_out: list[dict] = []
            int_out: list[dict] = []
            tg = threading.Thread(target=lambda: bg_out.append(run_through_scheduler(
                s, "background", f"r{rnd}-bg2", ["local-embedding"], w_embed_batch)))
            tg.start()
            time.sleep(0.05)
            ti = threading.Thread(target=lambda: int_out.append(run_through_scheduler(
                s, "interactive", f"r{rnd}-int2", ["local-fast"], w_interactive)))
            ti.start()
            tg.join(timeout=OUTER_TIMEOUT)
            ti.join(timeout=OUTER_TIMEOUT)
            results.extend(bg_out or [{"request_id": f"r{rnd}-bg2", "status": "hung"}])
            results.extend(int_out or [{"request_id": f"r{rnd}-int2", "status": "hung"}])

            round_s = time.time() - t_round
            done = sum(1 for r in results if r["status"] == "completed")
            report["rounds"].append({
                "round": rnd + 1, "results": results, "round_s": round(round_s, 2),
                "completed": done, "refused": sum(1 for r in results
                                                  if r["status"] == "refused"),
                "failed": sum(1 for r in results if r["status"] == "failed"),
                "vram": smp.between(t_round, t_round + round_s),
            })
            print(f"  completed={done}/{len(results)} "
                  f"refused={sum(1 for r in results if r['status']=='refused')} "
                  f"failed={sum(1 for r in results if r['status']=='failed')}", flush=True)
            for r in results:
                extra = (f"queue={r.get('queue_s')} exec={r.get('exec_s')} "
                         f"load={r.get('load_s')}") if r["status"] != "refused" else ""
                print(f"    {r['request_id']:<12} {r['status']:<10} {extra}"
                      f"{(r.get('reason') or '')[:110]}", flush=True)

            # Persist after EVERY round. The mode B result was lost because the run crashed
            # later and the only write happened at the very end.
            atomic_write_json(out, report)

            # only add rounds for a config that PASSED screening
            if rnd == 0 and done < len(results):
                print("  SCREENING FAILED - not adding further rounds for this config",
                      flush=True)
                break

    report["vram"]["after_run"] = vram()
    time.sleep(3)
    report["vram"]["after_idle"] = vram()
    peaks = [rd["vram"].get("used_max") for rd in report["rounds"]
             if rd["vram"].get("used_max")]
    report["vram"]["task_peak"] = max(peaks) if peaks else None
    breaches = [rd["round"] for rd in report["rounds"] if rd["vram"].get("margin_breached")]
    report["margin_check"] = {
        "breached_rounds": breaches,
        "statement": ("no margin breach observed within the sampling window"
                      if not breaches else f"margin breached in rounds {breaches}"),
        "caveat": "sampling cannot establish a continuous-time guarantee",
    }
    report["sampler"] = {"samples": len(smp.samples),
                         "interval_s": smp.interval}

    # ---------------- health vs first inference, measured separately
    print("\n=== health gate vs first inference (measured separately) ===", flush=True)
    if args.mode in ("B", "C") and not service_healthy(REGISTRY["local-reranker"].port):
        t0 = time.time()
        ok = start_service("local-reranker")
        gate = time.time() - t0
        first = None
        if ok:
            t1 = time.time()
            try:
                post(f"{RERANK}/v1/rerank", {"query": "q", "documents": ["d"], "top_n": 1})
                first = time.time() - t1
            except Exception:  # noqa: BLE001
                pass
        report["health_vs_first_inference"] = {
            "service": "local-reranker", "health_gate_s": round(gate, 3),
            "first_inference_s": round(first, 3) if first is not None else None,
            "note": "health passing is NOT used as a proxy for inference readiness"}
        print(f"  health gate={gate:.2f}s  first inference={first}", flush=True)

    # write the report BEFORE the summary/verdict computation, so a failure in those cannot
    # discard the round data that was already collected
    atomic_write_json(out, report)

    total_completed = sum(rd["completed"] for rd in report["rounds"])
    total_reqs = sum(len(rd["results"]) for rd in report["rounds"])
    report["summary"] = {
        "requests": total_reqs, "completed": total_completed,
        "completion_rate_pct": round(100 * total_completed / total_reqs, 1)
        if total_reqs else 0.0,
    }
    verdict = ("PASS" if total_completed == total_reqs and total_reqs > 0
               else "FAIL - the supported workload did not complete")
    report["verdict"] = verdict
    print(f"\nVERDICT mode {args.mode}: {verdict} "
          f"({total_completed}/{total_reqs} completed)")

    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, report)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




