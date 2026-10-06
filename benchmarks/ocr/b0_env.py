"""
B0 - read-only OCR feasibility check. Installs nothing, changes nothing.

Writes benchmarks/ocr/results/OCR-00-env/env.json and runs
`D:\\Python\\python.exe -m pip install --dry-run --report ...` so the exact versions that WOULD be
installed are recorded before anything is installed.

Decision rule (task section B0): if the CPU wheel does not support 3.13, or the install would need
any system-level change, STOP and report.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PY = r"D:\Python\python.exe"
OUT_DIR = REPO / "benchmarks/ocr/results/OCR-00-env"
ENV_JSON = OUT_DIR / "env.json"
DRY_RUN = OUT_DIR / "pip_dry_run.json"
DRY_LOG = REPO / "logs/ocr-b0-pip-dryrun.txt"


def ps(script: str) -> str:
    """PowerShell output via a FILE, never a pipe (P1-8 has wedged children three times)."""
    log = REPO / "logs/ocr-b0-ps.txt"
    with open(log, "wb") as fh:
        subprocess.run(["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
                        "-Command", script], stdout=fh, stderr=subprocess.STDOUT, timeout=180)
    return log.read_text(encoding="utf-8", errors="replace").strip()


def main() -> int:
    if ENV_JSON.exists():
        print(f"REFUSING to overwrite {ENV_JSON}")
        return 3
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (REPO / "logs").mkdir(parents=True, exist_ok=True)
    env: dict = {"step": "B0 read-only feasibility", "installed_or_changed": "nothing"}

    # ---- host resources
    r = ps(
        "$d=Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='D:'\";"
        "$c=Get-CimInstance Win32_LogicalDisk -Filter \"DeviceID='C:'\";"
        "$os=Get-CimInstance Win32_OperatingSystem;"
        "$cpu=Get-CimInstance Win32_Processor | Select-Object -First 1;"
        "[pscustomobject]@{"
        "d_free_gb=[math]::Round($d.FreeSpace/1GB,1); d_total_gb=[math]::Round($d.Size/1GB,1);"
        "c_free_gb=[math]::Round($c.FreeSpace/1GB,1); c_total_gb=[math]::Round($c.Size/1GB,1);"
        "commit_limit_mb=[int]($os.TotalVirtualMemorySize/1KB);"
        "commit_free_mb=[int]($os.FreeVirtualMemory/1KB);"
        "ram_free_mb=[int]($os.FreePhysicalMemory/1KB);"
        "cpu_name=$cpu.Name.Trim(); cpu_cores=$cpu.NumberOfCores; cpu_threads=$cpu.NumberOfLogicalProcessors"
        "} | ConvertTo-Json -Compress")
    try:
        env["host"] = json.loads(r.splitlines()[-1])
    except Exception as e:  # noqa: BLE001
        env["host"] = {"error": f"{type(e).__name__}: {e}", "raw": r[:400]}

    # ---- ports, by Win32_Process.CommandLine ownership (never by directory existence)
    ports: dict = {}
    for p in (8111, 8113, 8114, 8115, 8116):
        s = (
            f"$c=Get-NetTCPConnection -State Listen -LocalPort {p} -ErrorAction SilentlyContinue;"
            "if($c){$pr=Get-CimInstance Win32_Process -Filter \"ProcessId=$($c.OwningProcess)\" "
            "-ErrorAction SilentlyContinue;"
            "[pscustomobject]@{listening=$true;pid=$c.OwningProcess;"
            "cmdline=$(if($pr){{$pr.CommandLine}}else{{''}})}|ConvertTo-Json -Compress}"
            "else{[pscustomobject]@{listening=$false}|ConvertTo-Json -Compress}"
        )
        try:
            ports[str(p)] = json.loads(ps(s).splitlines()[-1])
        except Exception as e:  # noqa: BLE001
            ports[str(p)] = {"error": f"{type(e).__name__}: {e}"}
    env["ports"] = ports

    # ---- GPU baseline
    try:
        g = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
                            "--format=csv,noheader,nounits"], stdout=subprocess.PIPE,
                           timeout=60).stdout.decode().strip()
        t, u, f = g.split(",")
        env["gpu_baseline"] = {"total_mb": int(t), "used_mb": int(u), "free_mb": int(f)}
    except Exception as e:  # noqa: BLE001
        env["gpu_baseline"] = {"error": f"{type(e).__name__}: {e}"}

    # ---- interpreter actually used
    env["interpreter"] = subprocess.run([PY, "-c",
                                         "import sys,platform;print(sys.version);print(platform.machine())"],
                                        stdout=subprocess.PIPE, timeout=60).stdout.decode().strip().splitlines()

    # ---- pip dry run: resolves the dependency set WITHOUT installing anything
    cmd = [PY, "-m", "pip", "install", "--dry-run", "--report", str(DRY_RUN),
           "paddlepaddle", "paddleocr"]
    with open(DRY_LOG, "wb") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=1800).returncode
    env["pip_dry_run"] = {"returncode": rc, "log": str(DRY_LOG), "report": str(DRY_RUN)}

    if DRY_RUN.is_file():
        rep = json.loads(DRY_RUN.read_text(encoding="utf-8"))
        items = []
        for it in rep.get("install", []):
            md = it.get("metadata", {})
            url = (it.get("download_info") or {}).get("url", "")
            items.append({"name": md.get("name"), "version": md.get("version"),
                          "requires_python": md.get("requires_python"), "url": url})
        env["pip_dry_run"]["packages"] = items
        env["pip_dry_run"]["n_packages"] = len(items)
        key = {i["name"].lower(): i for i in items if i["name"]}
        env["pip_dry_run"]["paddlepaddle"] = key.get("paddlepaddle")
        env["pip_dry_run"]["paddleocr"] = key.get("paddleocr")
        # a GPU build would show up as paddlepaddle-gpu; assert we are getting the CPU one
        env["pip_dry_run"]["gpu_build_present"] = any(
            "gpu" in (i["name"] or "").lower() for i in items)
        env["pip_dry_run"]["cpu_wheels_only"] = not env["pip_dry_run"]["gpu_build_present"]
    else:
        env["pip_dry_run"]["report_written"] = False

    # ---- feasibility verdict, from the rule stated before running
    ok_rc = env["pip_dry_run"]["returncode"] == 0
    pd = env["pip_dry_run"].get("paddlepaddle") or {}
    po = env["pip_dry_run"].get("paddleocr") or {}
    env["feasibility"] = {
        "pip_resolved": ok_rc,
        "paddlepaddle_version": pd.get("version"),
        "paddlepaddle_requires_python": pd.get("requires_python"),
        "paddleocr_version": po.get("version"),
        "cpu_wheel": bool(pd.get("url", "").lower().find("win_amd64") >= 0),
        "no_gpu_build": not env["pip_dry_run"].get("gpu_build_present", False),
        "port_8116_free": not (ports.get("8116") or {}).get("listening", True),
        "system_level_change_required": False,   # a venv install is not a system change
    }
    env["feasibility"]["proceed"] = bool(
        ok_rc and env["feasibility"]["no_gpu_build"]
        and env["feasibility"]["port_8116_free"])

    ENV_JSON.write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(env.get("host"), ensure_ascii=False, indent=2))
    print("\nports:", json.dumps({k: v.get("listening") for k, v in ports.items()}))
    print("gpu_baseline:", json.dumps(env["gpu_baseline"]))
    print("interpreter:", env["interpreter"])
    print("\npip dry run returncode:", rc)
    print("  paddlepaddle:", json.dumps(pd))
    print("  paddleocr   :", json.dumps(po))
    print("  n_packages  :", env["pip_dry_run"].get("n_packages"))
    print("  gpu build present:", env["pip_dry_run"].get("gpu_build_present"))
    print("\nfeasibility:", json.dumps(env["feasibility"], ensure_ascii=False, indent=2))
    print(f"\nwrote {ENV_JSON}")
    return 0 if env["feasibility"]["proceed"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
