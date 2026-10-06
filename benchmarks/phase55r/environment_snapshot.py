"""
environment_snapshot.py - Phase 5.5R step 1: record the ACTUAL environment.

The phase instructions warn not to assume the reviewer's working tree is the deployment
directory. This script therefore establishes, from the machine rather than from
documentation:

  * which git repository is actually deployed, and its HEAD relative to the audit baseline
  * every listening service, its PID, its executable and its command line, so the path a
    process was started from is a FACT rather than an assumption
  * which process owns each port, and whether that process's executable lives under the
    project root
  * the deployed models with sizes and hashes
  * GPU, RAM and pagefile state
  * where historical raw results actually live

Written to a run directory, never over an existing one.

Usage:
  python environment_snapshot.py --out ../results/<run-id>/environment_snapshot.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIT_BASELINE = "bac96ab1b5714845bd2f4a46f0a96af3561a33e3"

SERVICES = {
    8111: "local-fast",
    8113: "local-embedding",
    8114: "local-reranker",
    8188: "possible Qwen-Image/ComfyUI",
    3080: "DSH web GUI",
}


def run(cmd: list[str], timeout: int = 40) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           errors="replace")
        return (r.stdout or "").strip()
    except Exception as e:  # noqa: BLE001
        return f"<error: {type(e).__name__}: {e}>"


def pwsh(cmd: str, timeout: int = 60) -> str:
    return run(["powershell", "-NoProfile", "-Command", cmd], timeout)


def sha256(p: Path, chunk: int = 1 << 20) -> str | None:
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    try:
        with p.open("rb") as fh:
            while True:
                b = fh.read(chunk)
                if not b:
                    break
                h.update(b)
    except OSError:
        return None
    return h.hexdigest()


def git(*args: str) -> str:
    return run(["git", *args], timeout=30)


def port_owner(port: int) -> dict:
    """
    Which PID owns the port, and CRUCIALLY what interpreter/executable it was launched with.

    `Get-Process .Path` is NOT reliable for this: a venv's python.exe is a launcher that
    re-executes the base interpreter, so .Path reports C:\\...\\Python\\python.exe even when
    the process was started from a venv. The first snapshot therefore reported the 8113 and
    8114 services as "not under the project root", which was an artefact of how venvs work
    rather than a fact about deployment. Win32_Process.CommandLine shows what was actually
    invoked, so ownership is judged from that.
    """
    out = pwsh(
        f"$c = Get-NetTCPConnection -State Listen -LocalPort {port} "
        f"-ErrorAction SilentlyContinue | Select-Object -First 1; "
        f"if ($c) {{ $p = Get-CimInstance Win32_Process -Filter \"ProcessId=$($c.OwningProcess)\" "
        f"-ErrorAction SilentlyContinue; "
        f"Write-Output $c.OwningProcess; Write-Output $c.LocalAddress; "
        f"Write-Output ($p.Name); Write-Output ($p.ExecutablePath); "
        f"Write-Output ($p.CommandLine) }}")
    lines = [ln for ln in out.splitlines()]
    if not lines or not lines[0].strip():
        return {"listening": False}
    d = {"listening": True, "pid": lines[0].strip(),
         "local_address": lines[1].strip() if len(lines) > 1 else None,
         "process_name": lines[2].strip() if len(lines) > 2 else None,
         "executable": lines[3].strip() if len(lines) > 3 else None,
         "command_line": lines[4].strip() if len(lines) > 4 else None}
    # ownership is judged from the COMMAND LINE, which names the venv python explicitly
    cl = (d.get("command_line") or "")
    d["under_project_root"] = str(PROJECT_ROOT).lower() in cl.lower()
    d["uses_venv"] = "\\venv\\" in cl.lower() or "/venv/" in cl.lower()
    return d


def health(port: int) -> dict:
    import urllib.request
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=4) as r:
            body = json.loads(r.read().decode())
        return {"reachable": True, "status": body.get("status"), "body": body}
    except Exception as e:  # noqa: BLE001
        return {"reachable": False, "error": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"REFUSING to overwrite an existing snapshot: {out}")
        return 3

    snap: dict = {
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "task": "P55R-01 environment snapshot",
    }

    # ---------------- repository identity
    repo = {
        "project_root": str(PROJECT_ROOT),
        "cwd_at_capture": os.getcwd(),
        "is_git_repo": (PROJECT_ROOT / ".git").exists(),
        "head": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "ahead_of_baseline": git("rev-list", "--count", f"{AUDIT_BASELINE}..HEAD"),
        "behind_baseline": git("rev-list", "--count", f"HEAD..{AUDIT_BASELINE}"),
        "commit_count": git("rev-list", "--count", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "uncommitted_files": [ln for ln in git("status", "--porcelain").splitlines() if ln.strip()],
        "audit_baseline": AUDIT_BASELINE,
        "head_matches_baseline": git("rev-parse", "HEAD") == AUDIT_BASELINE,
        "remotes": git("remote", "-v"),
    }
    # any other working copy of this project on the machine?
    others = run(["powershell", "-NoProfile", "-Command",
                  "Get-ChildItem 'D:\\','C:\\Users' -Directory -ErrorAction SilentlyContinue | "
                  "Where-Object { $_.Name -match 'MultiAgent' } | Select-Object -ExpandProperty FullName"])
    repo["other_working_copies"] = [x for x in others.splitlines() if x.strip()]
    snap["repository"] = repo

    # ---------------- services: port owner is a FACT, not an assumption
    svc = {}
    for port, name in SERVICES.items():
        owner = port_owner(port)
        entry = {"expected_name": name, **owner}
        if owner.get("listening"):
            entry["health"] = health(port)
        svc[str(port)] = entry
    snap["services"] = svc

    # ---------------- deployed models
    models = {}
    for label, rel in (
        ("local-fast 4B Q4_K_M", "models/llm/Qwen3.5-4B-Q4_K_M.gguf"),
        ("local-smart 9B Q4_K_M", "models/llm/Qwen3.5-9B-Q4_K_M.gguf"),
        ("local-embedding weights", "models/embedding/Qwen3-Embedding-0.6B/model.safetensors"),
        ("local-reranker weights", "models/reranker/Qwen3-Reranker-0.6B/model.safetensors"),
    ):
        p = PROJECT_ROOT / rel
        models[label] = {
            "path": str(p), "exists": p.exists(),
            "size_mb": round(p.stat().st_size / 2**20, 2) if p.exists() else None,
            "sha256": sha256(p) if p.exists() and p.suffix == ".safetensors" else None,
        }
    snap["models"] = models

    # ---------------- hardware
    gpu = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,"
                               "utilization.gpu,temperature.gpu",
               "--format=csv,noheader"])
    snap["gpu"] = {"raw": gpu}
    try:
        parts = [x.strip() for x in gpu.splitlines()[0].split(",")]
        snap["gpu"].update({"name": parts[0], "total_mb": int(parts[1].split()[0]),
                            "used_mb": int(parts[2].split()[0]),
                            "free_mb": int(parts[3].split()[0])})
    except Exception:
        pass

    mem = pwsh("$o=Get-CimInstance Win32_OperatingSystem;"
               "$p=Get-CimInstance Win32_PageFileUsage -ErrorAction SilentlyContinue;"
               "$g=Get-CimInstance Win32_ComputerSystem;"
               "Write-Output ('{0}|{1}|{2}|{3}|{4}|{5}' -f $o.TotalVisibleMemorySize,"
               "$o.FreePhysicalMemory, $p.AllocatedBaseSize, $p.PeakUsage, "
               "$g.TotalPhysicalMemory, $o.TotalVirtualMemorySize)")
    try:
        f = mem.splitlines()[0].split("|")
        snap["memory"] = {
            "ram_total_mb": int(f[0]) // 1024, "ram_free_mb": int(f[1]) // 1024,
            "pagefile_allocated_mb": int(f[2] or 0), "pagefile_peak_mb": int(f[3] or 0),
            "commit_limit_mb": int(f[5]) // 1024,
            "pagefile_headroom_mb": (int(f[2] or 0) - int(f[3] or 0)),
        }
    except Exception:
        snap["memory"] = {"raw": mem}

    # ---------------- where the historical raw results actually live
    hist = {}
    for label, rel in (
        ("phase4_retrieval", "benchmarks/retrieval/results"),
        ("phase4_reranker", "benchmarks/reranker/results"),
        ("phase55_v2", "benchmarks/retrieval-v2/results"),
        ("phase55_v2_baseline", "benchmarks/retrieval-v2"),
        ("served_index", "vector/embedding/local-ai"),
        ("bench_index", "cache/retrieval_bench/indexes"),
    ):
        p = PROJECT_ROOT / rel
        hist[label] = {
            "path": str(p), "exists": p.exists(),
            "files": sorted(x.name for x in p.iterdir())[:40] if p.is_dir() else None,
        }
    snap["historical_artifacts"] = hist

    # ---------------- protected environments still present
    snap["protected_paths"] = {
        p: (PROJECT_ROOT / p).exists() if not p.startswith("D:\\DSH") else Path(p).exists()
        for p in ("D:\\DSH\\computer_science\\qwen-image-deploy",
                  "tools/llama.cpp/b11254-cuda13.4/llama-server.exe",
                  "services/retrieval/venv/Scripts/python.exe",
                  "services/reranker/venv/Scripts/python.exe")
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------------- human-readable summary
    print(f"snapshot -> {out}\n")
    print(f"repo        {repo['project_root']}  branch={repo['branch']}")
    print(f"HEAD        {repo['head'][:12]}  matches audit baseline: "
          f"{repo['head_matches_baseline']}  dirty={repo['dirty']}")
    print(f"other copies of this project: {repo['other_working_copies'] or 'none'}")
    print("\nservices:")
    for port, e in svc.items():
        if e.get("listening"):
            h = e.get("health", {})
            print(f"  {port:<6} {e['expected_name']:<16} pid={e['pid']:<7} "
                  f"{e.get('process_name','?'):<14} "
                  f"project_owned={e.get('under_project_root')} "
                  f"health={h.get('status') or h.get('error','?')}")
            print(f"         exe {e.get('executable')}")
        else:
            print(f"  {port:<6} {e['expected_name']:<16} not listening")
    print(f"\ngpu         {snap['gpu'].get('name')}  used={snap['gpu'].get('used_mb')} "
          f"free={snap['gpu'].get('free_mb')} MiB")
    print(f"memory      {snap['memory']}")
    print(f"\nmodels:")
    for k, v in models.items():
        print(f"  {k:<26} exists={v['exists']} {v['size_mb']} MB")
    print(f"\nprotected: {snap['protected_paths']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
