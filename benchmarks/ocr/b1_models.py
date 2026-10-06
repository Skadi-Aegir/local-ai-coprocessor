"""
B1 - download the OCR models into <repo>\\models\\ocr\\ and record their provenance.

CPU-only, and the models must NOT land in C:\\Users\\<user>\\.paddlex. paddlex reads its cache root
from the PADDLE_PDX_CACHE_HOME environment variable at import time (paddlex/utils/cache.py line 29),
so it is set here BEFORE paddlex is imported.

Downloads BOTH PP-OCRv5 tiers - mobile and server, detection and recognition - because B3 compares
them. Document orientation classification, text-line orientation classification and document
unwarping are turned OFF: they are for photographed/scanned pages, not for screenshots, and the task
scope excludes them.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MODELS = REPO / "models/ocr"
CACHE = MODELS / "_paddlex_cache"
EVID = REPO / "benchmarks/ocr/results/OCR-01-env"

# set before importing paddlex
os.environ["PADDLE_PDX_CACHE_HOME"] = str(CACHE)
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

WANTED = ["PP-OCRv5_mobile_det", "PP-OCRv5_mobile_rec",
          "PP-OCRv5_server_det", "PP-OCRv5_server_rec"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ps_home(dirpath: Path) -> list[dict]:
    files = []
    for f in sorted(dirpath.rglob("*")):
        if f.is_file():
            files.append({"rel": f.relative_to(dirpath).as_posix(),
                          "bytes": f.stat().st_size, "sha256": sha256(f)})
    return files


def main() -> int:
    EVID.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    out = EVID / "models.json"
    if out.exists():
        print(f"REFUSING to overwrite {out}")
        return 3

    import paddlex  # noqa: E402
    from paddlex.utils.cache import CACHE_DIR  # noqa: E402
    from paddlex.inference.utils import official_models as om  # noqa: E402

    print("paddlex", paddlex.__version__)
    print("CACHE_DIR in force:", CACHE_DIR)
    if str(CACHE_DIR).lower().startswith("c:\\"):
        print("REFUSING: the cache root is on C:. PADDLE_PDX_CACHE_HOME was not honoured.")
        return 5

    mgr = om.official_models
    reg = {}
    for name in WANTED:
        reg[name] = {
            "in_model_list": name in getattr(mgr, "model_list", []),
            "model_source": om.MODEL_SOURCE,
            "hosters": [type(h).__name__ for h in getattr(mgr, "hoster_candidates", [])],
        }

    report = {"paddlex_version": paddlex.__version__,
              "cache_dir": str(CACHE_DIR), "registry": reg, "models": {}}

    for name in WANTED:
        t0 = time.time()
        print(f"\n--- {name}: downloading/resolving")
        try:
            d = Path(mgr.get_model_path(name))
        except Exception as e:  # noqa: BLE001
            print(f"    FAILED: {type(e).__name__}: {e}")
            report["models"][name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        dt = round(time.time() - t0, 1)
        files = ps_home(d)
        total = sum(f["bytes"] for f in files)
        print(f"    dir  : {d}")
        print(f"    files: {len(files)}  total {total/1024/1024:.2f} MB  in {dt}s")
        report["models"][name] = {"dir": str(d), "n_files": len(files),
                                  "total_bytes": total, "seconds": dt, "files": files}

    # land each model in models/ocr/<name>/ so the tree is predictable and reviewable
    print("\n--- staging model dirs under models/ocr/<name>/")
    for name, info in report["models"].items():
        if "dir" not in info:
            continue
        src = Path(info["dir"])
        dst = MODELS / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        info["staged_at"] = str(dst)
        info["staged_n_files"] = sum(1 for f in dst.rglob("*") if f.is_file())
        print(f"    {name:<24} -> {dst}  ({info['staged_n_files']} files)")

    # cache must be removable now that the models are staged
    report["cache_dir_removed"] = False
    if CACHE.exists():
        shutil.rmtree(CACHE, ignore_errors=True)
        report["cache_dir_removed"] = not CACHE.exists()
    print(f"\ncache under models/ocr removed: {report['cache_dir_removed']}")

    # did anything land on C:?
    home_px = Path.home() / ".paddlex"
    new_c = []
    if home_px.exists():
        for f in home_px.rglob("*"):
            if f.is_file():
                new_c.append({"rel": f.relative_to(home_px).as_posix(),
                              "bytes": f.stat().st_size,
                              "mtime": time.strftime("%Y-%m-%dT%H:%M:%S",
                                                     time.localtime(f.stat().st_mtime))})
    report["c_user_paddlex"] = {"path": str(home_px), "exists": home_px.exists(),
                                "n_files": len(new_c), "files": new_c[:50]}
    print(f"C:\\...\\.paddlex exists: {home_px.exists()}  files: {len(new_c)}")

    report["staged"] = {}
    for name in WANTED:
        dst = MODELS / name
        if dst.is_dir():
            fs = ps_home(dst)
            report["staged"][name] = {"n_files": len(fs),
                                      "total_bytes": sum(f["bytes"] for f in fs),
                                      "files": fs}

    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nstaged totals:")
    for name, s in report["staged"].items():
        print(f"  {name:<24} {s['n_files']:>3} files  {s['total_bytes']/1024/1024:>7.2f} MB")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
