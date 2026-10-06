"""
inspect_gguf.py - decode GGUF headers to compare quantisations without a GPU.

Motivation: the Phase 2 quant decision (Q4_K_M vs Q5_K_M vs Q6_K) should rest on
more than tokens/s. This reads each file's metadata and tensor-type histogram
directly from the GGUF container, so we can show exactly where the bits went -
which tensors were quantised, what fraction of the weight bytes each type holds,
and how much of the file is non-weight overhead.

Stdlib only. Parses the public GGUF v3 format.

Usage:
  python inspect_gguf.py <file.gguf> [more.gguf ...]
  python inspect_gguf.py --json out.json <file.gguf> ...
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from collections import Counter
from pathlib import Path

# GGUF metadata value type ids
UINT8, INT8, UINT16, INT16, UINT32, INT32, FLOAT32, BOOL, STRING, ARRAY, UINT64, INT64, FLOAT64 = range(13)

# GGML tensor type ids -> (name, block_size, bytes_per_block)
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
}


class Reader:
    def __init__(self, fh):
        self.fh = fh

    def raw(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise EOFError(f"wanted {n} bytes, got {len(b)}")
        return b

    def unpack(self, fmt: str):
        size = struct.calcsize(fmt)
        return struct.unpack(fmt, self.raw(size))

    def u32(self) -> int:
        return self.unpack("<I")[0]

    def u64(self) -> int:
        return self.unpack("<Q")[0]

    def i32(self) -> int:
        return self.unpack("<i")[0]

    def string(self) -> str:
        n = self.u64()
        return self.raw(n).decode("utf-8", "replace")


def read_value(r: Reader, vtype: int):
    """Read one metadata value. Arrays are returned as (type, [values]) and may be huge."""
    if vtype == UINT8:
        return r.unpack("<B")[0]
    if vtype == INT8:
        return r.unpack("<b")[0]
    if vtype == UINT16:
        return r.unpack("<H")[0]
    if vtype == INT16:
        return r.unpack("<h")[0]
    if vtype == UINT32:
        return r.u32()
    if vtype == INT32:
        return r.i32()
    if vtype == FLOAT32:
        return r.unpack("<f")[0]
    if vtype == BOOL:
        return bool(r.unpack("<B")[0])
    if vtype == STRING:
        return r.string()
    if vtype == ARRAY:
        elem_type = r.u32()
        count = r.u64()
        # token arrays in the vocab can be enormous; summarise them instead of
        # materialising hundreds of thousands of strings
        if elem_type == STRING and count > 1000:
            for _ in range(count):
                r.string()
            return ("array<string>", count)
        vals = [read_value(r, elem_type) for _ in range(count)]
        return (f"array<{elem_type}>", vals)
    if vtype == UINT64:
        return r.u64()
    if vtype == INT64:
        return r.unpack("<q")[0]
    if vtype == FLOAT64:
        return r.unpack("<d")[0]
    raise ValueError(f"unknown metadata value type {vtype}")


def nbytes_for(ttype: int, n_elements: int) -> int:
    if ttype in GGML_TYPES:
        _, blk, bytes_per_blk = GGML_TYPES[ttype]
        return (n_elements // blk) * bytes_per_blk if blk > 1 else n_elements * bytes_per_blk
    return 0


def inspect(path: Path, keep_tensors: bool = False) -> dict:
    with path.open("rb") as fh:
        r = Reader(fh)
        magic = r.raw(4)
        if magic != b"GGUF":
            raise ValueError(f"{path.name}: not a GGUF file (magic={magic!r})")
        version = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()

        meta: dict = {}
        for _ in range(n_kv):
            key = r.string()
            vtype = r.u32()
            meta[key] = read_value(r, vtype)

        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            n_dims = r.u32()
            dims = [r.u64() for _ in range(n_dims)]
            ttype = r.u32()
            offset = r.u64()
            n_elem = 1
            for d in dims:
                n_elem *= d
            tensors.append(
                {"name": name, "dims": dims, "type_id": ttype,
                 "type": GGML_TYPES.get(ttype, (f"UNKNOWN_{ttype}", 1, 0))[0],
                 "n_elements": n_elem, "bytes": nbytes_for(ttype, n_elem)}
            )

    by_type: Counter = Counter()
    bytes_by_type: Counter = Counter()
    for t in tensors:
        by_type[t["type"]] += 1
        bytes_by_type[t["type"]] += t["bytes"]

    total_weight_bytes = sum(t["bytes"] for t in tensors)
    file_size = path.stat().st_size

    # how many tensors are still full precision (a quality signal)
    unquantised = sum(
        t["bytes"] for t in tensors if t["type"] in ("F32", "F16", "BF16")
    )

    out = {
        "file": path.name,
        "file_size_mb": round(file_size / 1e6, 1),
        "gguf_version": version,
        "n_tensors": n_tensors,
        "n_metadata_kv": n_kv,
        "architecture": None,
        "model_name": None,
        "quantisation_version": None,
        "parameter_count": None,
        "context_length": None,
        "n_layers": None,
        "n_heads": None,
        "n_kv_heads": None,
        "embedding_length": None,
        "feed_forward_length": None,
        "total_weight_bytes": total_weight_bytes,
        "weight_bytes_mb": round(total_weight_bytes / 1e6, 1),
        "non_weight_overhead_mb": round((file_size - total_weight_bytes) / 1e6, 1),
        "overhead_pct": round(100 * (file_size - total_weight_bytes) / file_size, 2),
        "tensor_type_counts": dict(by_type),
        "tensor_bytes_by_type": {k: v for k, v in sorted(bytes_by_type.items(), key=lambda kv: -kv[1])},
        "tensor_bytes_by_type_pct": {
            k: round(100 * v / total_weight_bytes, 2)
            for k, v in sorted(bytes_by_type.items(), key=lambda kv: -kv[1])
        },
        "full_precision_bytes_mb": round(unquantised / 1e6, 1),
        "full_precision_pct": round(100 * unquantised / total_weight_bytes, 2),
        "bits_per_weight": round(8 * total_weight_bytes / max(meta.get("qwen3_5.embedding_length", 0) or 1, 1)
                                 if False else 0, 3),
    }

    # General keys first: the architecture prefix for the model-specific keys below
    # comes from general.architecture. For this model it is "qwen35", NOT "qwen3_5"
    # as the HuggingFace card's `qwen3_5` tag would suggest.
    for key, val in meta.items():
        if key == "general.architecture":
            out["architecture"] = val
        elif key == "general.name":
            out["model_name"] = val
        elif key == "general.size_label":
            out["size_label"] = val
        elif key == "general.license":
            out["license"] = val
        elif key == "general.quantized_by":
            out["quantized_by"] = val
        elif key == "general.base_model.0.repo_url":
            out["base_model_repo"] = val
        elif key.startswith("quantize.imatrix"):
            out.setdefault("imatrix", {})[key.split(".")[-1]] = val

    arch = out.get("architecture")
    prefix = f"{arch}." if arch else ""
    wanted = {
        "block_count": "n_layers",
        "context_length": "context_length",
        "embedding_length": "embedding_length",
        "feed_forward_length": "feed_forward_length",
        "attention.head_count": "n_heads",
        "attention.head_count_kv": "n_kv_heads",
        "attention.key_length": "key_length",
        "attention.value_length": "value_length",
        "full_attention_interval": "full_attention_interval",
        "ssm.state_size": "ssm_state_size",
        "ssm.inner_size": "ssm_inner_size",
        "ssm.conv_kernel": "ssm_conv_kernel",
        "ssm.group_count": "ssm_group_count",
        "rope.freq_base": "rope_freq_base",
    }
    for suffix, field in wanted.items():
        key = prefix + suffix
        if key in meta:
            out[field] = meta[key]

    # KV-cache footprint for the FULL-ATTENTION layers only. A hybrid model such as
    # qwen35 keeps a fixed-size recurrent state on its linear-attention layers, so
    # only every `full_attention_interval`-th layer contributes a growing KV cache.
    #
    # For this model metadata reports key_length = value_length = 256, which is the
    # FULL projection width already, not a per-head dimension:
    #     n_kv_heads * head_dim = 4 * 64 = 256
    # so the cache is  2 tensors (K,V) x 256 elems x 2 bytes = 1024 B per layer-token,
    # NOT 2 x (key_length + value_length). Getting this wrong double-counts V.
    # (An earlier revision of this file did exactly that; the value happened to come
    # out right only because it was then multiplied by the same factor again.)
    kl, vl = out.get("key_length"), out.get("value_length")
    nkv, layers, interval = out.get("n_kv_heads"), out.get("n_layers"), out.get("full_attention_interval")
    if all(isinstance(x, int) for x in (kl, vl, nkv, layers, interval)) and interval > 0:
        n_full = (layers + interval - 1) // interval
        out["n_full_attention_layers"] = n_full
        width = max(kl, vl)  # full KV projection width per head-group
        bytes_per_token = n_full * nkv * width * 2 * 2  # x2 for K,V  x2 for f16
        out["kv_bytes_per_token_f16"] = bytes_per_token
        out["kv_formula"] = f"{n_full} layers x {nkv} kv_heads x {width} width x 2 (K,V) x 2 bytes"
        out["kv_cache_mb"] = {
            f"{ctx // 1024}K": round(bytes_per_token * ctx / 1e6, 1)
            for ctx in (4096, 8192, 16384, 32768)
        }
        # Fair hybrid saving: same KV geometry, but 32 full-attention layers instead
        # of 8. The whole gain comes from full_attention_interval, i.e. 4x here.
        out["kv_saving_vs_all_attention"] = round(layers / n_full, 2)

    n_params = sum(t["n_elements"] for t in tensors)
    out["parameter_count_tensors"] = n_params
    if total_weight_bytes:
        out["bits_per_weight"] = round(8 * total_weight_bytes / n_params, 3)

    if keep_tensors:
        out["tensors"] = tensors
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json", default=None, help="write full results to this JSON file")
    ap.add_argument("--tensors", action="store_true", help="include per-tensor detail")
    args = ap.parse_args()

    results = []
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"MISSING: {p}")
            continue
        try:
            info = inspect(p, keep_tensors=args.tensors)
        except Exception as e:  # noqa: BLE001
            print(f"FAILED {p.name}: {type(e).__name__}: {e}")
            continue
        results.append(info)
        print(f"\n================ {info['file']} ================")
        for k in ("model_name", "architecture", "license", "quantized_by", "base_model_repo",
                  "file_size_mb", "weight_bytes_mb",
                  "non_weight_overhead_mb", "overhead_pct", "n_tensors",
                  "parameter_count_tensors", "bits_per_weight", "n_layers",
                  "context_length", "embedding_length", "n_heads", "n_kv_heads",
                  "key_length", "value_length", "full_attention_interval",
                  "n_full_attention_layers",
                  "full_precision_bytes_mb", "full_precision_pct"):
            if k in info:
                print(f"  {k:24} {info[k]}")
        if "kv_cache_mb" in info:
            print(f"  {'kv_bytes_per_token':24} {info['kv_bytes_per_token_f16']}")
            print(f"  {'kv_formula':24} {info.get('kv_formula','')}")
            print(f"  {'kv_saving_vs_all_attn':24} {info.get('kv_saving_vs_all_attention')}x")
            print(f"  {'kv_cache_f16':24} {info['kv_cache_mb']}")
        if "imatrix" in info:
            print(f"  {'imatrix':24} {info['imatrix']}")
        print("  tensor bytes by type:")
        for k, v in info["tensor_bytes_by_type"].items():
            pct = info["tensor_bytes_by_type_pct"][k]
            cnt = info["tensor_type_counts"].get(k, 0)
            print(f"    {k:10} {v/1e6:9.1f} MB  {pct:6.2f}%   ({cnt} tensors)")

    if args.json and results:
        Path(args.json).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
